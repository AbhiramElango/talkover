"""Window geometry shared by training, offline evaluation and the streaming runtime (no torch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CLASSES = ("ignore", "backchannel", "interrupt")
INTERRUPT = CLASSES.index("interrupt")


@dataclass(frozen=True)
class WindowConfig:
    window_seconds: float = 1.0
    onset_seconds: float = 1.0
    sample_rate_hz: int = 16_000
    min_decision_seconds: float = 0.1
    max_decision_seconds: float = 0.5
    pre_onset_probability: float = 0.25
    earliest_pre_onset_seconds: float = -0.9
    mute_agent_after_onset: bool = False
    channels: tuple[int, ...] = (0, 1)
    agent_envelope: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))


def cut_window(clip: np.ndarray, decision_seconds: float, config: WindowConfig) -> np.ndarray:
    """(2, frames) window ending ``decision_seconds`` after the onset, zero-padded on the left."""

    rate = config.sample_rate_hz
    size = round(config.window_seconds * rate)
    end = round((config.onset_seconds + decision_seconds) * rate)
    start = end - size
    window = np.zeros((2, size), dtype=np.float32)
    source = clip[max(0, start) : end].T
    window[:, size - source.shape[1] :] = source
    if config.mute_agent_after_onset:
        onset_in_window = round(config.onset_seconds * rate) - start
        window[1, max(0, onset_in_window) :] = 0.0
    return finalize_window(window, config)


def finalize_window(window: np.ndarray, config: WindowConfig) -> np.ndarray:
    """Agent-channel transform applied identically in training, evaluation and the live runtime."""

    if config.agent_envelope:
        window[1] = envelope_noise(window[1], config.sample_rate_hz)
    return window


_NOISE = np.random.default_rng(0).standard_normal(16_000 * 4).astype(np.float32)


def envelope_noise(agent: np.ndarray, rate: int, frame_seconds: float = 0.02, floor: float = 1e-3) -> np.ndarray:
    """Keep when the agent speaks or pauses, drop its voice and level: fixed noise shaped by a normalised envelope."""

    size = round(frame_seconds * rate)
    frames = agent[: agent.size // size * size].reshape(-1, size)
    envelope = np.sqrt(np.mean(frames**2, axis=1))
    envelope = envelope / max(float(envelope.max()), floor)
    shaped = np.repeat(envelope, size)
    shaped = np.pad(shaped, (0, agent.size - shaped.size), mode="edge")
    return (0.1 * shaped * _NOISE[: agent.size]).astype(np.float32)


def model_inputs(windows, config: WindowConfig):
    """Keep the channels this model consumes; works for numpy and torch batches."""

    return windows[:, list(config.channels)]
