import numpy as np
import pytest

from talkover.interruption.augment import rms_dbfs
from talkover.interruption.tts_agent import SAMPLE_RATE, Phrase, phrases_in_window, revoice_agent


def test_phrases_group_by_gap_and_clip_to_window() -> None:
    words = [(9.5, 10.2, "so"), (10.3, 10.6, "the"), (10.7, 11.0, "plan"), (11.6, 12.4, "is"), (12.5, 13.5, "done")]
    phrases = phrases_in_window(words, 10.0, 12.0)
    assert phrases == [Phrase(0.0, 1.0, "so the plan"), Phrase(1.6, 2.0, "is")]


def _fake_tts(text, voice, wpm):
    seconds = len(text.split()) * 60 / wpm
    return (0.3 * np.ones(int(seconds * SAMPLE_RATE))).astype(np.float32)


def test_revoice_places_phrases_in_slots_and_matches_level() -> None:
    agent = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    agent[: SAMPLE_RATE] = 0.05
    track = revoice_agent(agent, [Phrase(0.2, 0.8, "a b c"), Phrase(1.5, 1.9, "d")], "v", _fake_tts)
    assert not track[: int(0.19 * SAMPLE_RATE)].any()
    assert track[int(0.3 * SAMPLE_RATE)] > 0 and not track[int(1.0 * SAMPLE_RATE)]
    assert not track[int(1.95 * SAMPLE_RATE) :].any()
    assert rms_dbfs(track[track > 1e-4]) == pytest.approx(rms_dbfs(agent[agent > 1e-4]), abs=0.5)
