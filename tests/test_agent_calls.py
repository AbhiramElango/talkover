import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from talkover.interruption.augment import rms_dbfs, synthetic_rir
from talkover.interruption.corpora.agent_calls import (
    SAMPLE_RATE,
    WienerEchoCanceller,
    active_runs,
    fill_gaps,
    cancel_echo,
    echo_residual,
    export_clips,
    find_candidates,
)


def _turn(bursts=(3.0, 6.0), seconds=10.0, echo_db=-6.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    agent = (0.1 * rng.standard_normal(t.size) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    agent[: SAMPLE_RATE // 2] = 0
    rir = synthetic_rir(rng, SAMPLE_RATE, 0.3)
    mic = np.convolve(agent, rir)[: t.size] * 10 ** (echo_db / 20) + 1e-4 * rng.standard_normal(t.size)
    for onset in bursts:
        start = int(onset * SAMPLE_RATE)
        mic[start : start + int(0.4 * SAMPLE_RATE)] += 0.2 * rng.standard_normal(int(0.4 * SAMPLE_RATE))
    return np.stack([mic, agent], axis=1).astype(np.float32)


def test_active_runs_and_gap_filling() -> None:
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1], dtype=bool)
    assert active_runs(mask) == [(1, 2), (4, 1), (8, 1)]
    assert fill_gaps(mask, 1).tolist() == [False, True, True, True, True, False, False, False, True]


def test_echo_residual_removes_most_echo() -> None:
    turn = _turn(bursts=())
    residual = echo_residual(turn[:, 0], turn[:, 1])
    assert rms_dbfs(residual) < rms_dbfs(turn[:, 0]) - 15


def test_candidates_find_caller_bursts_not_echo() -> None:
    onsets = [c.onset_seconds for c in find_candidates(cancel_echo(_turn(), WienerEchoCanceller()), "s", 1, "backchannel")]
    assert len(onsets) == 2
    assert onsets[0] == pytest.approx(3.0, abs=0.05) and onsets[1] == pytest.approx(6.0, abs=0.05)


def test_export_writes_ami_style_clips(tmp_path: Path) -> None:
    session = tmp_path / "raw" / "s1"
    session.mkdir(parents=True)
    sf.write(session / "turn_01.wav", _turn(), SAMPLE_RATE)
    labels = tmp_path / "labels.jsonl"
    rows = [{"session": "s1", "turn": 1, "onset_seconds": 3.0, "instruction": "backchannel", "label": "backchannel"},
            {"session": "s1", "turn": 1, "onset_seconds": 6.0, "instruction": "backchannel", "label": "unclear"}]
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert export_clips(labels, tmp_path / "raw", tmp_path / "clips", WienerEchoCanceller()) == 1
    (row,) = [json.loads(line) for line in (tmp_path / "clips" / "test.jsonl").open()]
    clip, rate = sf.read(tmp_path / "clips" / row["clip"])
    assert clip.shape == (2 * SAMPLE_RATE, 2) and rate == SAMPLE_RATE
    assert row["meeting_id"] == "s1/t01" and row["label"] == "backchannel"
