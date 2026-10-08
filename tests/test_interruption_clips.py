from pathlib import Path

import numpy as np
import soundfile as sf

from talkover.evaluation.intervals import TimeInterval
from talkover.interruption import OverlapEvent
from talkover.interruption.clips import ClipConfig, ClipExtractor, ami_channel_path, balance_ignore


def _event(onset: float, act: str = "bck", label: str = "backchannel", split: str = "train") -> OverlapEvent:
    return OverlapEvent("M1", split, "A", 0, "B", 1, onset, TimeInterval(0, 9), None, act, label)


def _write_meeting(root: Path, seconds: float = 4.0, rate: int = 16_000) -> None:
    (root / "M1").mkdir(parents=True)
    t = np.arange(int(seconds * rate)) / rate
    sf.write(root / "M1" / "M1.Headset-0.wav", (0.1 * t).astype(np.float32), rate, subtype="FLOAT")
    sf.write(root / "M1" / "M1.Headset-1.wav", (-0.1 * t).astype(np.float32), rate, subtype="FLOAT")


def test_clip_is_caller_then_agent_around_onset(tmp_path: Path) -> None:
    _write_meeting(tmp_path)
    extractor = ClipExtractor(ami_channel_path(tmp_path), ClipConfig(1.0, 1.0))

    clip = extractor.extract(_event(2.0))

    assert clip.shape == (32_000, 2)
    assert np.isclose(clip[0, 1], 0.1, atol=1e-4)
    assert np.isclose(clip[16_000, 0], -0.2, atol=1e-4)


def test_clip_outside_audio_is_skipped(tmp_path: Path) -> None:
    _write_meeting(tmp_path)
    extractor = ClipExtractor(ami_channel_path(tmp_path))
    assert extractor.extract(_event(0.5)) is None
    assert extractor.extract(_event(3.5)) is None
    assert not extractor.available(OverlapEvent("M2", "train", "A", 0, "B", 1, 2.0, TimeInterval(0, 9), None, "bck", "backchannel"))


def test_balance_caps_crosstalk_per_split() -> None:
    events = [_event(i) for i in range(3)] + [_event(i, "crosstalk", "ignore") for i in range(10)]
    events += [_event(i, "laugh", "ignore") for i in range(2)]
    kept = balance_ignore(events, crosstalk_ratio=1.0)
    assert sum(e.act == "crosstalk" for e in kept) == 3
    assert sum(e.act == "laugh" for e in kept) == 2
    assert balance_ignore(events, 1.0) == kept
