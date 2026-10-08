"""Re-voice the agent channel of a clip with TTS while keeping the human agent's word timing."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from talkover.interruption.augment import rms_dbfs

SAMPLE_RATE = 16_000
Word = tuple[float, float, str]


@dataclass(frozen=True)
class Phrase:
    start: float
    end: float
    text: str


class Synthesizer(Protocol):
    def __call__(self, text: str, voice: str, words_per_minute: int) -> NDArray[np.float32]: ...


class MacSay:
    """macOS `say` at 16 kHz float."""

    def __call__(self, text, voice, words_per_minute):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phrase.wav"
            subprocess.run(
                ["say", "-v", voice, "-r", str(words_per_minute), "-o", str(path),
                 "--file-format=WAVE", f"--data-format=LEF32@{SAMPLE_RATE}", text],
                check=True,
            )
            audio, _ = sf.read(path, dtype="float32", always_2d=True)
        return audio[:, 0]


def phrases_in_window(words: Sequence[Word], start: float, end: float, gap_seconds: float = 0.3) -> list[Phrase]:
    """Group words overlapping [start, end) into spurts; times are relative to ``start`` and clipped."""

    inside = [w for w in words if w[1] > start and w[0] < end]
    phrases: list[list[Word]] = []
    for word in inside:
        if phrases and word[0] - phrases[-1][-1][1] <= gap_seconds:
            phrases[-1].append(word)
        else:
            phrases.append([word])
    return [
        Phrase(round(max(0.0, group[0][0] - start), 3), round(min(end, group[-1][1]) - start, 3), " ".join(w[2] for w in group))
        for group in phrases
    ]


def _trim_silence(audio: NDArray[np.float32], threshold_db: float = -50.0) -> NDArray[np.float32]:
    frame = SAMPLE_RATE // 100
    levels = [rms_dbfs(audio[i : i + frame]) for i in range(0, max(1, audio.size - frame), frame)]
    loud = [i for i, level in enumerate(levels) if level > threshold_db]
    if not loud:
        return audio[:0]
    return audio[loud[0] * frame : (loud[-1] + 1) * frame]


def revoice_agent(
    agent: NDArray[np.float32],
    phrases: Sequence[Phrase],
    voice: str,
    synthesize: Synthesizer,
    min_wpm: int = 120,
    max_wpm: int = 320,
) -> NDArray[np.float32]:
    """New agent track: each phrase synthesised at a rate that fits its slot, placed at its start, level-matched."""

    track = np.zeros_like(agent)
    for phrase in phrases:
        slot = phrase.end - phrase.start
        if slot <= 0.05:
            continue
        words = len(phrase.text.split())
        wpm = int(np.clip(words / slot * 60, min_wpm, max_wpm))
        speech = _trim_silence(synthesize(phrase.text, voice, wpm))
        first = round(phrase.start * SAMPLE_RATE)
        length = min(speech.size, round(slot * SAMPLE_RATE), track.size - first)
        if length > 0:
            fade = min(length, SAMPLE_RATE // 100)
            piece = speech[:length].copy()
            piece[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            track[first : first + length] = piece
    if track.any() and agent.any():
        track *= 10 ** ((rms_dbfs(agent[np.abs(agent) > 1e-4]) - rms_dbfs(track[np.abs(track) > 1e-4])) / 20)
    return track.astype(np.float32)
