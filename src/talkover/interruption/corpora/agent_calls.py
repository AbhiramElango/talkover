"""Self-recorded agent calls: stereo turns (ch0 raw caller mic, ch1 agent playback) and their labels.

Deployed agents hear the mic after echo cancellation, so candidates and exported
clips use an EchoCanceller (WebRTC AEC3 via LiveKit by default) on the raw mic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import csd, fftconvolve, welch

from talkover.interruption.augment import rms_dbfs


SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.032


@dataclass(frozen=True)
class Candidate:
    session: str
    turn: int
    onset_seconds: float
    instruction: str


class EchoCanceller(Protocol):
    def __call__(self, mic: NDArray[np.float32], agent: NDArray[np.float32]) -> NDArray[np.float32]: ...


class WebRtcEchoCanceller:
    """WebRTC audio processing (AEC3 + high-pass) as used by LiveKit clients, run offline in 10 ms frames."""

    def __init__(self, noise_suppression: bool = False) -> None:
        self.noise_suppression = noise_suppression

    def __call__(self, mic, agent):
        from livekit import rtc

        apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=self.noise_suppression, high_pass_filter=True)
        frame = SAMPLE_RATE // 100
        out = np.zeros_like(mic)

        def pcm(samples: NDArray[np.float32]):
            return rtc.AudioFrame((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes(), SAMPLE_RATE, 1, frame)

        for start in range(0, mic.size // frame * frame, frame):
            apm.process_reverse_stream(pcm(agent[start : start + frame]))
            near = pcm(mic[start : start + frame])
            apm.process_stream(near)
            out[start : start + frame] = np.frombuffer(bytes(near.data), dtype=np.int16) / 32767
        return out


class WienerEchoCanceller:
    def __init__(self, taps: int = 4096) -> None:
        self.taps = taps

    def __call__(self, mic, agent):
        return echo_residual(mic, agent, self.taps)


@dataclass(frozen=True)
class CandidateConfig:
    min_agent_lead_seconds: float = 0.5
    min_agent_remaining_seconds: float = 0.5
    agent_active_dbfs: float = -45.0
    residual_rise_db: float = 12.0
    min_active_seconds: float = 0.1
    merge_seconds: float = 0.7
    agent_gap_seconds: float = 0.3


def echo_residual(mic: NDArray[np.float32], agent: NDArray[np.float32], taps: int = 4096) -> NDArray[np.float32]:
    """Mic minus the agent filtered by a Wiener echo-path estimate (caller speech is uncorrelated with the agent)."""

    nperseg = 2 * taps
    _, cross = csd(agent, mic, fs=SAMPLE_RATE, nperseg=nperseg)
    _, power = welch(agent, fs=SAMPLE_RATE, nperseg=nperseg)
    response = np.fft.irfft(cross / np.maximum(power, 1e-12 * power.max()), n=nperseg)[:taps]
    echo = fftconvolve(agent, response)[: mic.size]
    return (mic - echo).astype(np.float32)


def _frame_levels(audio: NDArray[np.float32]) -> NDArray[np.float64]:
    size = round(FRAME_SECONDS * SAMPLE_RATE)
    frames = audio[: audio.size // size * size].reshape(-1, size)
    return np.array([rms_dbfs(frame) for frame in frames])


def cancel_echo(turn: NDArray[np.float32], canceller: EchoCanceller) -> NDArray[np.float32]:
    """Stereo turn with ch0 replaced by the echo-cancelled mic."""

    return np.stack([canceller(turn[:, 0], turn[:, 1]), turn[:, 1]], axis=1).astype(np.float32)


def find_candidates(
    turn: NDArray[np.float32], session: str, index: int, instruction: str, config: CandidateConfig = CandidateConfig()
) -> list[Candidate]:
    """Onsets where the echo-cancelled mic (``turn[:, 0]``) rises well above its floor while the agent is mid-speech."""

    level = _frame_levels(turn[:, 0])
    agent_active = fill_gaps(_frame_levels(turn[:, 1]) >= config.agent_active_dbfs, round(config.agent_gap_seconds / FRAME_SECONDS))
    active = level >= np.percentile(level, 20) + config.residual_rise_db
    lead = round(config.min_agent_lead_seconds / FRAME_SECONDS)
    remaining = round(config.min_agent_remaining_seconds / FRAME_SECONDS)
    earliest = max(lead, round(1.0 / FRAME_SECONDS))
    candidates: list[Candidate] = []
    for first, length in active_runs(active):
        onset = round(first * FRAME_SECONDS, 3)
        if length * FRAME_SECONDS < config.min_active_seconds or first < earliest:
            continue
        if not agent_active[first - lead : first + remaining].all() or first + remaining > agent_active.size:
            continue
        if not candidates or onset - candidates[-1].onset_seconds >= config.merge_seconds:
            candidates.append(Candidate(session, index, onset, instruction))
    return candidates


def fill_gaps(active: NDArray[np.bool_], max_gap: int) -> NDArray[np.bool_]:
    """Bridge inactive gaps of at most ``max_gap`` frames between active runs (pauses between words)."""

    filled = active.copy()
    runs = active_runs(active)
    for (first, length), (following, _) in zip(runs, runs[1:]):
        if following - (first + length) <= max_gap:
            filled[first + length : following] = True
    return filled


def active_runs(active: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """(first frame, length) of each run of True values."""

    padded = np.concatenate([[False], active, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def read_session(session_dir: Path) -> dict:
    return json.loads((session_dir / "session.json").read_text())


def turn_audio(session_dir: Path, index: int) -> NDArray[np.float32]:
    audio, rate = sf.read(session_dir / f"turn_{index:02d}.wav", dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape[1] != 2:
        raise ValueError("turns must be 16 kHz stereo (caller, agent)")
    return audio


def export_clips(
    labels_path: Path,
    sessions_root: Path,
    out_dir: Path,
    canceller: EchoCanceller,
    split: str = "test",
    sessions: set[str] | None = None,
) -> int:
    """Write 2 s echo-cancelled clips (onset at 1.0 s) and a manifest in the AMI clip format."""

    count = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with labels_path.open(encoding="utf-8") as rows, (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as manifest:
        for line in rows:
            row = json.loads(line)
            if row["label"] not in ("interrupt", "backchannel", "ignore") or (sessions and row["session"] not in sessions):
                continue
            audio = cancel_echo(turn_audio(sessions_root / row["session"], row["turn"]), canceller)
            start = round((row["onset_seconds"] - 1.0) * SAMPLE_RATE)
            clip = audio[start : start + 2 * SAMPLE_RATE]
            if start < 0 or clip.shape[0] < 2 * SAMPLE_RATE:
                continue
            onset_ms = round(row["onset_seconds"] * 1000)
            relative = Path(split) / row["label"] / f"{row['session']}_t{row['turn']:02d}_{onset_ms}.flac"
            (out_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_dir / relative, clip, SAMPLE_RATE, format="FLAC")
            manifest.write(json.dumps({
                "clip": str(relative), "meeting_id": f"{row['session']}/t{row['turn']:02d}", "split": split,
                "label": row["label"], "act": row.get("instruction", ""), "onset_seconds": row["onset_seconds"],
            }) + "\n")
            count += 1
    return count


def candidate_json(candidate: Candidate) -> dict:
    return asdict(candidate)
