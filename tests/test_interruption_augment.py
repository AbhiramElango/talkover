from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from talkover.interruption.augment import (
    AgentEcho,
    AugmentContext,
    MusanBank,
    OnsetSound,
    Telephony,
    default_pipelines,
    event_rng,
    rms_dbfs,
    stable_split,
)

RATE = 16_000
CONTEXT = AugmentContext(sample_rate_hz=RATE, onset_frame=RATE, reference_dbfs=-30.0)


@pytest.fixture()
def bank(tmp_path: Path) -> MusanBank:
    rng = np.random.default_rng(0)
    for category in ("noise", "music", "speech"):
        folder = tmp_path / category
        folder.mkdir()
        for index in range(40):
            sf.write(folder / f"{category}-{index}.wav", 0.1 * rng.standard_normal(3 * RATE).astype(np.float32), RATE)
    return MusanBank(tmp_path, "train")


def _clip(seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    agent = 0.1 * np.sin(2 * np.pi * 220 * t)
    return np.stack([np.zeros_like(t), agent], axis=1).astype(np.float32)


def test_stable_split_is_deterministic_and_bank_filters(bank: MusanBank) -> None:
    assert stable_split("noise-1.wav") == stable_split("noise-1.wav")
    assert all(stable_split(path.name) == "train" for files in bank.files.values() for path in files)


def test_pipeline_is_seeded_and_never_touches_agent(bank: MusanBank) -> None:
    speech, ignore = default_pipelines(bank)
    for pipeline in (speech, ignore):
        first, params = pipeline(_clip(), CONTEXT, event_rng("e1", 0))
        second, again = pipeline(_clip(), CONTEXT, event_rng("e1", 0))
        assert np.array_equal(first, second) and params == again
        assert np.array_equal(first[:, 1], _clip()[:, 1])


def test_onset_sound_starts_at_onset(bank: MusanBank) -> None:
    clip, params = OnsetSound(bank, kinds=("noise",), level_db={"noise": (-5.0, -5.0)})(
        _clip(), CONTEXT, np.random.default_rng(1)
    )
    assert params["onset_sound"] == "noise"
    assert not clip[: RATE, 0].any()
    assert rms_dbfs(clip[RATE:, 0]) == pytest.approx(-35.0, abs=0.5)


def test_agent_echo_level_is_relative_to_agent() -> None:
    clip, params = AgentEcho(level_db=(-20.0, -20.0), delay_ms=(50.0, 50.0))(
        _clip(), CONTEXT, np.random.default_rng(2)
    )
    assert params["echo_delay_ms"] == 50
    assert not clip[:799, 0].any()
    assert rms_dbfs(clip[RATE:, 0]) - rms_dbfs(clip[RATE:, 1]) == pytest.approx(-20.0, abs=0.5)


def test_telephony_removes_wideband_content() -> None:
    t = np.arange(2 * RATE) / RATE
    clip = np.stack([0.1 * np.sin(2 * np.pi * 6000 * t), np.zeros_like(t)], axis=1).astype(np.float32)
    out, _ = Telephony()(clip.copy(), CONTEXT, np.random.default_rng(3))
    assert rms_dbfs(out[:, 0]) < rms_dbfs(clip[:, 0]) - 30


def test_gain_targets_one_channel() -> None:
    from talkover.interruption.augment import Gain

    clip = np.ones((RATE, 2), dtype=np.float32)
    out, params = Gain((6.0, 6.0), channel=1)(clip.copy(), CONTEXT, np.random.default_rng(0))
    assert np.allclose(out[:, 0], 1.0) and np.allclose(out[:, 1], 10 ** (6 / 20))
    assert params == {"agent_gain_db": 6.0}


def test_cancelled_echo_runs_the_injected_canceller() -> None:
    from talkover.interruption.augment import CancelledEcho

    calls = []

    def fake_canceller(mic, agent):
        calls.append((mic.copy(), agent.copy()))
        return np.zeros_like(mic)

    out, params = CancelledEcho(fake_canceller)(_clip(), CONTEXT, np.random.default_rng(0))
    (mic, agent), = calls
    assert mic.any() and np.array_equal(agent, _clip()[:, 1])
    assert not out[:, 0].any() and params["aec"] is True


def test_robust_pipelines_are_seeded(bank: MusanBank) -> None:
    from talkover.interruption.augment import robust_pipelines

    speech, _ = robust_pipelines(bank, lambda mic, agent: mic * 0.5)
    first, params = speech(_clip(), CONTEXT, event_rng("e1", 0))
    second, again = speech(_clip(), CONTEXT, event_rng("e1", 0))
    assert np.array_equal(first, second) and params == again
