"""Causal stop policies over the caller stream.

Horizon 0 reports a stop before the onset (echo or noise). Horizons > 0 re-arm
at the onset: only activity from the onset on counts, so every policy is
judged on the same clips.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from talkover.interruption.augment import rms_dbfs


Audio = NDArray[np.float32]


class StopPolicy(Protocol):
    name: str
    horizons: tuple[float, ...] | None

    def decisions(
        self, audio: Audio, sample_rate_hz: int, onset_seconds: float, horizons: Sequence[float]
    ) -> tuple[bool, ...]:
        """``audio`` is the (frames, 2) clip (caller, agent) or the caller channel alone."""
        ...


def caller_channel(audio: Audio) -> Audio:
    return audio[:, 0] if audio.ndim == 2 else audio


class FrameScorer(Protocol):
    frame_seconds: float

    def scores(self, caller: Audio, sample_rate_hz: int) -> NDArray[np.float64]: ...


class EnergyScorer:
    """Frame level in dBFS."""

    frame_seconds = 0.032

    def scores(self, caller, sample_rate_hz):
        size = round(self.frame_seconds * sample_rate_hz)
        frames = caller[: caller.size // size * size].reshape(-1, size)
        return np.array([rms_dbfs(frame) for frame in frames])


class SileroScorer:
    """Streaming Silero VAD speech probability per 32 ms frame."""

    frame_seconds = 0.032

    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad(onnx=True)

    def scores(self, caller, sample_rate_hz):
        self._model.reset_states()
        size = round(self.frame_seconds * sample_rate_hz)
        audio = self._torch.from_numpy(np.ascontiguousarray(caller[: caller.size // size * size]))
        probs = self._model.audio_forward(audio, sample_rate_hz)
        return probs.numpy().reshape(-1).astype(np.float64)


class LevelGatedScorer:
    """Zero the inner score on frames quieter than ``min_dbfs`` (a VAD volume gate)."""

    def __init__(self, inner: FrameScorer, min_dbfs: float) -> None:
        self.inner, self.min_dbfs = inner, min_dbfs
        self.frame_seconds = inner.frame_seconds
        self._energy = EnergyScorer()

    def scores(self, caller, sample_rate_hz):
        scores = self.inner.scores(caller, sample_rate_hz)
        levels = self._energy.scores(caller, sample_rate_hz)[: scores.size]
        return np.where(levels >= self.min_dbfs, scores[: levels.size], 0.0)


def first_sustained(
    active: NDArray[np.bool_], frame_seconds: float, min_active_seconds: float, start_seconds: float = 0.0
) -> float | None:
    """End time of the first run of active frames, counted from ``start_seconds``, lasting ``min_active_seconds``."""

    needed = max(1, round(min_active_seconds / frame_seconds))
    first = int(start_seconds / frame_seconds + 1e-9)
    run = 0
    for index in range(first, active.size):
        run = run + 1 if active[index] else 0
        if run >= needed:
            return (index + 1) * frame_seconds
    return None


class SustainedActivityPolicy:
    """Stop once activity stays above ``threshold`` for ``min_active_seconds`` (stock VAD barge-in)."""

    horizons = None

    def __init__(self, name: str, scorer: FrameScorer, threshold: float, min_active_seconds: float) -> None:
        self.name, self.scorer = name, scorer
        self.threshold, self.min_active_seconds = threshold, min_active_seconds

    def trigger_times(self, caller: Audio, sample_rate_hz: int, onset_seconds: float) -> tuple[float | None, float | None]:
        """(first trigger from clip start, first trigger re-armed at the onset)."""

        active = self.scorer.scores(caller, sample_rate_hz) >= self.threshold
        frame = self.scorer.frame_seconds
        return (
            first_sustained(active, frame, self.min_active_seconds),
            first_sustained(active, frame, self.min_active_seconds, onset_seconds),
        )

    def decisions(self, audio, sample_rate_hz, onset_seconds, horizons):
        early, armed = self.trigger_times(caller_channel(audio), sample_rate_hz, onset_seconds)
        return tuple(_stopped(early, armed, onset_seconds, h) for h in horizons)


def _stopped(early: float | None, armed: float | None, onset_seconds: float, horizon: float) -> bool:
    if horizon == 0:
        return early is not None and early <= onset_seconds + 1e-9
    return armed is not None and armed <= onset_seconds + horizon + 1e-9


class Transcriber(Protocol):
    def transcribe(self, audio: Audio, sample_rate_hz: int) -> str: ...


class FasterWhisperTranscriber:
    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8") -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio, sample_rate_hz):
        if sample_rate_hz != 16_000:
            raise ValueError("faster-whisper expects 16 kHz")
        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False
        )
        return " ".join(segment.text for segment in segments)


class MlxWhisperTranscriber:
    """Whisper on the Apple GPU via MLX."""

    def __init__(self, repo: str = "mlx-community/whisper-small-mlx") -> None:
        import mlx_whisper

        self._transcribe, self.repo = mlx_whisper.transcribe, repo

    def transcribe(self, audio, sample_rate_hz):
        if sample_rate_hz != 16_000:
            raise ValueError("whisper expects 16 kHz")
        result = self._transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            path_or_hf_repo=self.repo,
            language="en",
            condition_on_previous_text=False,
            verbose=None,
        )
        return result["text"]


BACKCHANNEL_PHRASES = frozenset(
    {"mm", "mhm", "mmhmm", "mm-hmm", "uh-huh", "uhhuh", "hmm", "yeah", "yes", "yep", "ok", "okay",
     "right", "sure", "alright", "i", "see", "uh", "um", "ah", "oh", "so", "true", "cool", "great", "nice"}
)


class TranscriptRulePolicy:
    """VAD gate plus a text rule: stop only for >= ``min_words`` words outside a backchannel list."""

    def __init__(
        self,
        name: str,
        gate: SustainedActivityPolicy,
        transcriber: Transcriber,
        min_words: int = 2,
        ignore_words: frozenset[str] = BACKCHANNEL_PHRASES,
        horizons: tuple[float, ...] = (0.0, 0.5, 1.0),
    ) -> None:
        self.name, self.gate, self.transcriber = name, gate, transcriber
        self.min_words, self.ignore_words, self.horizons = min_words, ignore_words, horizons

    def decisions(self, audio, sample_rate_hz, onset_seconds, horizons):
        caller = caller_channel(audio)
        early, trigger = self.gate.trigger_times(caller, sample_rate_hz, onset_seconds)
        results = []
        for h in horizons:
            end = onset_seconds + h
            if h == 0:
                results.append(_stopped(early, trigger, onset_seconds, h))
                continue
            if trigger is None or trigger > end:
                results.append(False)
                continue
            start = max(onset_seconds, trigger - self.gate.min_active_seconds)
            text = self.transcriber.transcribe(caller[round(start * sample_rate_hz) : round(end * sample_rate_hz)], sample_rate_hz)
            words = [w for w in re.findall(r"[a-z'\-]+", text.lower()) if w.strip("-'") not in self.ignore_words]
            results.append(len(words) >= self.min_words)
        return tuple(results)
