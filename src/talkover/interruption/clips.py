"""Cut fixed stereo clips (caller, agent) around each event onset."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from talkover.interruption.events import OverlapEvent


ChannelPath = Callable[[str, int], Path]


@dataclass(frozen=True)
class ClipConfig:
    before_seconds: float = 1.0
    after_seconds: float = 1.0
    sample_rate_hz: int = 16_000


def ami_channel_path(audio_dir: Path) -> ChannelPath:
    return lambda meeting_id, channel: audio_dir / meeting_id / f"{meeting_id}.Headset-{channel}.wav"


class ClipExtractor:
    """Read only the needed frames; channel 0 is the caller, channel 1 the agent."""

    def __init__(self, channel_path: ChannelPath, config: ClipConfig | None = None) -> None:
        self.channel_path = channel_path
        self.config = config or ClipConfig()

    def available(self, event: OverlapEvent) -> bool:
        return all(
            self.channel_path(event.meeting_id, channel).exists()
            for channel in (event.caller_channel, event.agent_channel)
        )

    def extract(self, event: OverlapEvent) -> NDArray[np.float32] | None:
        rate = self.config.sample_rate_hz
        start = round((event.onset_seconds - self.config.before_seconds) * rate)
        frames = round((self.config.before_seconds + self.config.after_seconds) * rate)
        if start < 0:
            return None
        channels = []
        for channel in (event.caller_channel, event.agent_channel):
            path = self.channel_path(event.meeting_id, channel)
            if sf.info(path).samplerate != rate:
                raise ValueError(f"{path.name}: expected {rate} Hz")
            samples, _ = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
            if samples.shape[0] < frames:
                return None
            channels.append(samples[:, 0])
        return np.stack(channels, axis=1)


def balance_ignore(
    events: Iterable[OverlapEvent], crosstalk_ratio: float, seed: int = 0
) -> list[OverlapEvent]:
    """Cap crosstalk events per split at ``crosstalk_ratio`` x that split's backchannels."""

    by_split: dict[str, list[OverlapEvent]] = defaultdict(list)
    for event in events:
        by_split[event.split].append(event)
    rng = random.Random(seed)
    kept: list[OverlapEvent] = []
    for split in sorted(by_split):
        group = by_split[split]
        crosstalk = [event for event in group if event.act == "crosstalk"]
        backchannels = sum(event.label == "backchannel" for event in group)
        limit = min(len(crosstalk), round(crosstalk_ratio * backchannels))
        kept.extend(event for event in group if event.act != "crosstalk")
        kept.extend(rng.sample(crosstalk, limit))
    return kept
