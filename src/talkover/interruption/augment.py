"""Seeded augmentations for stereo clips (ch0 caller, ch1 agent).

Each step mutates only the caller channel and returns the parameters it drew,
so rendered evaluation sets record exactly what was applied.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import fftconvolve, resample_poly

from talkover.interruption.events import Split


Clip = NDArray[np.float32]
Params = dict[str, Any]
SoundKind = Literal["voice", "music", "noise"]
_EPS = 1e-9


@dataclass(frozen=True)
class AugmentContext:
    sample_rate_hz: int
    onset_frame: int
    reference_dbfs: float


class Augmentation(Protocol):
    def __call__(self, clip: Clip, context: AugmentContext, rng: np.random.Generator) -> tuple[Clip, Params]: ...


def rms_dbfs(samples: NDArray[np.floating]) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(samples))) + _EPS))


def scale_to_dbfs(samples: NDArray[np.floating], target_dbfs: float) -> NDArray[np.float32]:
    gain = 10 ** ((target_dbfs - rms_dbfs(samples)) / 20)
    return (samples * gain).astype(np.float32)


def synthetic_rir(rng: np.random.Generator, sample_rate_hz: int, rt60_seconds: float) -> NDArray[np.float32]:
    """Direct path plus exponentially decaying diffuse tail."""

    length = int(rt60_seconds * sample_rate_hz)
    t = np.arange(length) / sample_rate_hz
    tail = rng.standard_normal(length) * np.exp(-6.9 * t / rt60_seconds)
    tail[0] = 0.0
    rir = np.concatenate([[1.0], 0.3 * tail / (np.linalg.norm(tail) + _EPS)])
    return (rir / np.linalg.norm(rir)).astype(np.float32)


def stable_split(name: str, dev_percent: int = 10, test_percent: int = 10) -> Split:
    bucket = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + dev_percent:
        return "dev"
    return "train"


@dataclass
class MusanBank:
    """MUSAN file lists per category, restricted to one split by filename hash."""

    root: Path
    split: Split
    files: dict[str, list[Path]] = field(init=False)

    def __post_init__(self) -> None:
        self.files = {
            category: sorted(p for p in (self.root / category).rglob("*.wav") if stable_split(p.name) == self.split)
            for category in ("noise", "music", "speech")
        }

    def segment(self, category: str, frames: int, sample_rate_hz: int, rng: np.random.Generator) -> NDArray[np.float32]:
        for _ in range(10):
            path = self.files[category][rng.integers(len(self.files[category]))]
            info = sf.info(path)
            if info.samplerate != sample_rate_hz or info.frames < frames:
                continue
            start = int(rng.integers(0, info.frames - frames + 1))
            samples, _ = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
            if rms_dbfs(samples[:, 0]) > -60:
                return samples[:, 0]
        raise RuntimeError(f"no usable {category} segment of {frames} frames")


class BackgroundNoise:
    def __init__(self, bank: MusanBank, snr_db: tuple[float, float] = (0.0, 20.0)) -> None:
        self.bank, self.snr_db = bank, snr_db

    def __call__(self, clip, context, rng):
        category = str(rng.choice(["noise", "music"], p=[0.7, 0.3]))
        snr = float(rng.uniform(*self.snr_db))
        noise = self.bank.segment(category, clip.shape[0], context.sample_rate_hz, rng)
        clip[:, 0] += scale_to_dbfs(noise, context.reference_dbfs - snr)
        return clip, {"background": category, "snr_db": round(snr, 2)}


class AgentEcho:
    """Residual echo of the agent's own audio leaking into the caller mic."""

    def __init__(
        self,
        level_db: tuple[float, float] = (-30.0, -10.0),
        delay_ms: tuple[float, float] = (20.0, 150.0),
        rt60_seconds: tuple[float, float] = (0.2, 0.6),
    ) -> None:
        self.level_db, self.delay_ms, self.rt60_seconds = level_db, delay_ms, rt60_seconds

    def __call__(self, clip, context, rng):
        agent = clip[:, 1]
        level = float(rng.uniform(*self.level_db))
        delay = int(rng.uniform(*self.delay_ms) * context.sample_rate_hz / 1000)
        rir = synthetic_rir(rng, context.sample_rate_hz, float(rng.uniform(*self.rt60_seconds)))
        echo = np.zeros_like(agent)
        wet = fftconvolve(agent, rir)[: agent.shape[0] - delay]
        echo[delay:] = wet
        if echo.any():
            clip[:, 0] += scale_to_dbfs(echo, rms_dbfs(agent) + level)
        return clip, {"echo_db": round(level, 2), "echo_delay_ms": round(delay * 1000 / context.sample_rate_hz)}


class Telephony:
    """8 kHz narrowband path with 8-bit mu-law quantisation."""

    def __call__(self, clip, context, rng):
        caller = clip[:, 0]
        narrow = resample_poly(caller, 8_000, context.sample_rate_hz)
        peak = max(1.0, float(np.max(np.abs(narrow))))
        mu = 255.0
        encoded = np.sign(narrow / peak) * np.log1p(mu * np.abs(narrow / peak)) / np.log1p(mu)
        quantised = np.round(encoded * 127) / 127
        decoded = np.sign(quantised) * np.expm1(np.abs(quantised) * np.log1p(mu)) / mu * peak
        clip[:, 0] = resample_poly(decoded, context.sample_rate_hz, 8_000)[: caller.shape[0]].astype(np.float32)
        return clip, {"telephony": True}


class Gain:
    """Random level change on one channel (0 caller, 1 agent)."""

    def __init__(self, gain_db: tuple[float, float] = (-6.0, 6.0), channel: int = 0) -> None:
        self.gain_db, self.channel = gain_db, channel

    def __call__(self, clip, context, rng):
        gain = float(rng.uniform(*self.gain_db))
        clip[:, self.channel] *= 10 ** (gain / 20)
        return clip, {f"{('caller', 'agent')[self.channel]}_gain_db": round(gain, 2)}


EchoCanceller = Callable[[NDArray[np.float32], NDArray[np.float32]], NDArray[np.float32]]


class CancelledEcho:
    """Loudspeaker-level echo of the agent, then a real echo canceller on the caller channel.

    Reproduces what a deployed agent hears: residual echo plus double-talk suppression of the caller.
    """

    def __init__(
        self,
        canceller: EchoCanceller,
        level_db: tuple[float, float] = (-15.0, 5.0),
        delay_ms: tuple[float, float] = (20.0, 150.0),
        rt60_seconds: tuple[float, float] = (0.2, 0.6),
    ) -> None:
        self.canceller, self.level_db, self.delay_ms, self.rt60_seconds = canceller, level_db, delay_ms, rt60_seconds

    def __call__(self, clip, context, rng):
        clip, params = AgentEcho(self.level_db, self.delay_ms, self.rt60_seconds)(clip, context, rng)
        clip[:, 0] = self.canceller(clip[:, 0], clip[:, 1])
        return clip, {**params, "aec": True}


@dataclass(frozen=True)
class OneOf:
    steps: tuple[Augmentation, ...]

    def __call__(self, clip, context, rng):
        return self.steps[int(rng.integers(len(self.steps)))](clip, context, rng)


class OnsetSound:
    """Insert a non-caller sound starting at the onset; turns silent ignore clips realistic."""

    _CATEGORY = {"voice": "speech", "music": "music", "noise": "noise"}

    def __init__(
        self,
        bank: MusanBank,
        kinds: Sequence[SoundKind] = ("voice", "music", "noise"),
        level_db: dict[SoundKind, tuple[float, float]] | None = None,
        rt60_seconds: tuple[float, float] = (0.3, 0.8),
    ) -> None:
        self.bank, self.kinds, self.rt60_seconds = bank, tuple(kinds), rt60_seconds
        self.level_db = level_db or {"voice": (-20.0, -10.0), "music": (-20.0, -5.0), "noise": (-15.0, 0.0)}

    def __call__(self, clip, context, rng):
        kind: SoundKind = self.kinds[int(rng.integers(len(self.kinds)))]
        frames = clip.shape[0] - context.onset_frame
        sound = self.bank.segment(self._CATEGORY[kind], frames, context.sample_rate_hz, rng)
        if kind == "voice":
            rir = synthetic_rir(rng, context.sample_rate_hz, float(rng.uniform(*self.rt60_seconds)))
            sound = fftconvolve(sound, rir)[:frames].astype(np.float32)
        level = float(rng.uniform(*self.level_db[kind]))
        ramp = min(frames, int(0.01 * context.sample_rate_hz))
        sound[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        clip[context.onset_frame :, 0] += scale_to_dbfs(sound, context.reference_dbfs + level)
        return clip, {"onset_sound": kind, "onset_sound_db": round(level, 2)}


@dataclass(frozen=True)
class Maybe:
    step: Augmentation
    probability: float

    def __call__(self, clip, context, rng):
        if rng.random() < self.probability:
            return self.step(clip, context, rng)
        return clip, {}


class AugmentationPipeline:
    def __init__(self, steps: Sequence[Augmentation]) -> None:
        self.steps = tuple(steps)

    def __call__(self, clip: Clip, context: AugmentContext, rng: np.random.Generator) -> tuple[Clip, Params]:
        out = clip.astype(np.float32, copy=True)
        params: Params = {}
        for step in self.steps:
            out, drawn = step(out, context, rng)
            params.update(drawn)
        return np.clip(out, -1.0, 1.0), params


def event_rng(event_id: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha1(f"{seed}:{event_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def default_pipelines(bank: MusanBank) -> tuple[AugmentationPipeline, AugmentationPipeline]:
    """(all clips, silent ignore clips): the second adds an onset sound first. Used for the rendered eval sets."""

    common = [Maybe(BackgroundNoise(bank), 0.8), Maybe(AgentEcho(), 0.7), Gain(), Maybe(Telephony(), 0.5)]
    return AugmentationPipeline(common), AugmentationPipeline([OnsetSound(bank), *common])


def robust_pipelines(bank: MusanBank, canceller: EchoCanceller) -> tuple[AugmentationPipeline, AugmentationPipeline]:
    """Training pipelines for deployment robustness: wide per-channel levels and real AEC on half the echo."""

    common = [
        Gain((-10.0, 15.0), channel=1),
        Maybe(BackgroundNoise(bank), 0.8),
        Maybe(OneOf((AgentEcho(), CancelledEcho(canceller))), 0.8),
        Gain((-20.0, 6.0), channel=0),
        Maybe(Telephony(), 0.5),
    ]
    return AugmentationPipeline(common), AugmentationPipeline([OnsetSound(bank), *common])
