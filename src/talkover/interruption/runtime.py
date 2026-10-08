"""Streaming interruption detector: numpy + onnxruntime only.

Feed caller (echo-cancelled mic) and agent (played TTS) audio as it arrives;
every hop the detector scores the last window exactly as training and
offline evaluation do, and reports whether the caller is interrupting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from talkover.interruption.model.windows import INTERRUPT, WindowConfig, finalize_window, model_inputs

WindowModel = Callable[[NDArray[np.float32]], NDArray[np.float32]]
BUNDLE_FILE = "bundle.json"
DEFAULT_REPO = "Abhiram4abm/talkover"


@dataclass(frozen=True)
class Decision:
    time_seconds: float
    probability: float
    interrupt: bool


@dataclass(frozen=True)
class ModelBundle:
    """A shipped model: ONNX file, its window geometry and a calibrated threshold."""

    directory: Path
    model_file: str
    config: WindowConfig
    threshold: float

    @classmethod
    def load(cls, directory: str | Path) -> ModelBundle:
        directory = Path(directory)
        meta = json.loads((directory / BUNDLE_FILE).read_text())
        model = directory / meta["model"]
        if hashlib.sha256(model.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"{model.name} does not match the bundle checksum")
        return cls(directory, meta["model"], replace(WindowConfig(), **meta["window"]), float(meta["threshold"]))

    @classmethod
    def from_pretrained(cls, repo_id: str = DEFAULT_REPO, revision: str | None = None) -> ModelBundle:
        """Download the bundle from the Hugging Face Hub, or reuse the local cache."""

        from huggingface_hub import snapshot_download

        return cls.load(snapshot_download(repo_id, revision=revision, allow_patterns=[BUNDLE_FILE, "*.onnx"]))

    def save(self, extra: dict | None = None) -> None:
        path = self.directory / BUNDLE_FILE
        meta = {
            **(json.loads(path.read_text()) if path.exists() else {}),
            "model": self.model_file,
            "sha256": hashlib.sha256((self.directory / self.model_file).read_bytes()).hexdigest(),
            "threshold": self.threshold,
            "window": asdict(self.config),
            **(extra or {}),
        }
        path.write_text(json.dumps(meta, indent=1))

    def onnx_model(self, threads: int = 1) -> WindowModel:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        session = ort.InferenceSession(str(self.directory / self.model_file), options, providers=["CPUExecutionProvider"])
        return lambda windows: session.run(None, {"windows": windows})[0][:, INTERRUPT]


class _Stream:
    """Most recent audio at its native rate, plus how many samples were ever pushed."""

    def __init__(self, keep_seconds: float) -> None:
        self.keep_seconds = keep_seconds
        self.rate: int | None = None
        self.samples = np.zeros(0, dtype=np.float32)
        self.total = 0

    def push(self, samples: NDArray[np.float32], rate: int) -> None:
        if self.rate is not None and rate != self.rate:
            self.samples, self.total = np.zeros(0, dtype=np.float32), 0
        self.rate = rate
        keep = int(self.keep_seconds * rate)
        self.samples = np.concatenate([self.samples, samples.astype(np.float32)])[-keep:]
        self.total += samples.size

    def seconds(self) -> float:
        return self.total / self.rate if self.rate else 0.0

    def last(self, seconds: float, target_rate: int) -> NDArray[np.float32]:
        """Last ``seconds`` resampled to ``target_rate``, zero-padded on the left."""

        size = round(seconds * target_rate)
        if not self.rate or not self.samples.size:
            return np.zeros(size, dtype=np.float32)
        tail = self.samples[-int(round(seconds * self.rate)) :]
        if self.rate != target_rate:
            divisor = np.gcd(self.rate, target_rate)
            tail = resample_poly(tail, target_rate // divisor, self.rate // divisor).astype(np.float32)
        out = np.zeros(size, dtype=np.float32)
        out[size - min(size, tail.size) :] = tail[-size:]
        return out


class InterruptionDetector:
    """Scores the last window every ``hop_seconds`` of caller audio."""

    def __init__(
        self,
        model: WindowModel,
        config: WindowConfig,
        threshold: float,
        hop_seconds: float = 0.1,
        max_agent_lag_seconds: float = 0.2,
    ) -> None:
        self.model, self.config, self.threshold = model, config, threshold
        self.hop_seconds, self.max_agent_lag_seconds = hop_seconds, max_agent_lag_seconds
        keep = config.window_seconds + 0.5
        self.caller, self.agent = _Stream(keep), _Stream(keep)
        self._next_hop = hop_seconds

    @classmethod
    def from_bundle(cls, bundle: ModelBundle | str | Path, threshold: float | None = None, threads: int = 1) -> InterruptionDetector:
        bundle = bundle if isinstance(bundle, ModelBundle) else ModelBundle.load(bundle)
        return cls(bundle.onnx_model(threads), bundle.config, bundle.threshold if threshold is None else threshold)

    @classmethod
    def from_pretrained(
        cls, repo_id: str = DEFAULT_REPO, revision: str | None = None, threshold: float | None = None, threads: int = 1
    ) -> InterruptionDetector:
        return cls.from_bundle(ModelBundle.from_pretrained(repo_id, revision), threshold, threads)

    def push_agent(self, samples: NDArray[np.float32], rate: int) -> None:
        self.agent.push(samples, rate)

    def push_caller(self, samples: NDArray[np.float32], rate: int) -> list[Decision]:
        self.caller.push(samples, rate)
        lag = self.caller.seconds() - self.agent.seconds()
        if lag > self.max_agent_lag_seconds:
            agent_rate = self.agent.rate or self.config.sample_rate_hz
            self.agent.push(np.zeros(int(round(lag * agent_rate)), dtype=np.float32), agent_rate)
        decisions = []
        while self.caller.seconds() + 1e-9 >= self._next_hop:
            decisions.append(self._score(self._next_hop))
            self._next_hop += self.hop_seconds
        return decisions

    def window(self) -> NDArray[np.float32]:
        rate, seconds = self.config.sample_rate_hz, self.config.window_seconds
        window = np.stack([self.caller.last(seconds, rate), self.agent.last(seconds, rate)])
        return model_inputs(finalize_window(window, self.config)[None], self.config)

    def _score(self, time_seconds: float) -> Decision:
        probability = float(self.model(np.ascontiguousarray(self.window()))[0])
        return Decision(round(time_seconds, 3), probability, probability >= self.threshold)
