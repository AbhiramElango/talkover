"""Run a trained classifier every 100 ms over a clip and turn probabilities into stop decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from talkover.interruption.model.data import INTERRUPT, WindowConfig, cut_window, model_inputs


STEPS = tuple(round(x, 1) for x in np.arange(-0.9, 0.51, 0.1))


class WindowScorer(Protocol):
    config: WindowConfig
    steps: tuple[float, ...]

    def probabilities(self, clip: np.ndarray) -> np.ndarray: ...


class StreamingScorer:
    """P(interrupt) at each decision step relative to the onset (seconds)."""

    def __init__(self, model: torch.nn.Module, device: str, config: WindowConfig = WindowConfig(), steps: Sequence[float] = STEPS) -> None:
        self.model, self.device, self.config, self.steps = model.eval(), device, config, tuple(steps)

    @torch.no_grad()
    def probabilities(self, clip: np.ndarray) -> np.ndarray:
        windows = model_inputs(np.stack([cut_window(clip, step, self.config) for step in self.steps]), self.config)
        logits = self.model(torch.from_numpy(np.ascontiguousarray(windows)).to(self.device))
        return torch.softmax(logits, dim=-1)[:, INTERRUPT].float().cpu().numpy()


class OnnxStreamingScorer:
    """Same as StreamingScorer, backed by an exported ONNX model that outputs probabilities."""

    def __init__(self, model_path: str, config: WindowConfig = WindowConfig(), steps: Sequence[float] = STEPS, threads: int = 1) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self.config, self.steps = config, tuple(steps)

    def probabilities(self, clip: np.ndarray) -> np.ndarray:
        """One window per run, as deployed: dynamic INT8 quantisation depends on the batch."""

        windows = model_inputs(np.stack([cut_window(clip, step, self.config) for step in self.steps]), self.config)
        return np.array([self.session.run(None, {"windows": np.ascontiguousarray(w[None])})[0][0, INTERRUPT] for w in windows])


def stop_decisions(probabilities: np.ndarray, steps: Sequence[float], threshold: float, horizons: Sequence[float]) -> tuple[bool, ...]:
    """Horizon 0: any pre-onset step fired. Horizon h > 0: any step in (0, h] fired (re-armed at onset)."""

    steps_array = np.asarray(steps)
    fired = probabilities >= threshold
    decisions = []
    for h in horizons:
        mask = steps_array <= 1e-9 if h == 0 else (steps_array > 1e-9) & (steps_array <= h + 1e-9)
        decisions.append(bool(fired[mask].any()))
    return tuple(decisions)


@dataclass
class ModelStopPolicy:
    """StopPolicy adapter over any WindowScorer (PyTorch or ONNX)."""

    name: str
    scorer: WindowScorer
    threshold: float
    horizons: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    def decisions(self, audio, sample_rate_hz, onset_seconds, horizons):
        if audio.ndim != 2 or sample_rate_hz != self.scorer.config.sample_rate_hz:
            raise ValueError("model policy needs the 16 kHz stereo clip")
        probabilities = self.scorer.probabilities(audio)
        return stop_decisions(probabilities, self.scorer.steps, self.threshold, horizons)


def threshold_for_recall(interrupt_max: np.ndarray, target_recall: float) -> float:
    """Largest threshold whose interrupt recall still reaches ``target_recall``."""

    ordered = np.sort(interrupt_max)[::-1]
    index = min(len(ordered) - 1, max(0, int(np.ceil(target_recall * len(ordered))) - 1))
    return float(ordered[index])
