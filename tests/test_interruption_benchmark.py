import numpy as np
import pytest

from talkover.evaluation.bootstrap import cluster_bootstrap_rate
from talkover.interruption.benchmark import EvalItem, noise_band, stop_rates
from talkover.interruption.policies import (
    EnergyScorer,
    SustainedActivityPolicy,
    TranscriptRulePolicy,
    first_sustained,
)

RATE = 16_000


def _caller(onset: float, burst: float, seconds: float = 2.0) -> np.ndarray:
    audio = np.zeros(int(seconds * RATE), dtype=np.float32)
    start = int(onset * RATE)
    audio[start : start + int(burst * RATE)] = 0.1
    return audio


def test_first_sustained_needs_a_full_run() -> None:
    active = np.array([1, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert first_sustained(active, 0.1, 0.3) == pytest.approx(0.6)
    assert first_sustained(active, 0.1, 0.5) is None


def test_energy_policy_is_causal_by_horizon() -> None:
    policy = SustainedActivityPolicy("energy", EnergyScorer(), -40.0, 0.2)
    decisions = policy.decisions(_caller(1.0, 0.6), RATE, 1.0, (0.0, 0.1, 0.3))
    assert decisions == (False, False, True)
    assert policy.decisions(_caller(1.0, 0.1), RATE, 1.0, (0.5,)) == (False,)


def test_policies_rearm_at_onset() -> None:
    policy = SustainedActivityPolicy("energy", EnergyScorer(), -40.0, 0.2)
    caller = _caller(0.5, 1.5)
    assert policy.decisions(caller, RATE, 1.0, (0.0, 0.1, 0.2)) == (True, False, True)


class _FixedText:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio, sample_rate_hz):
        return self.text


def test_transcript_rule_ignores_backchannel_words() -> None:
    gate = SustainedActivityPolicy("energy", EnergyScorer(), -40.0, 0.2)
    caller = _caller(1.0, 0.8)
    assert TranscriptRulePolicy("t", gate, _FixedText("Mm-hmm, yeah.")).decisions(caller, RATE, 1.0, (0.5,)) == (False,)
    assert TranscriptRulePolicy("t", gate, _FixedText("No, wait a second")).decisions(caller, RATE, 1.0, (0.5,)) == (True,)
    assert TranscriptRulePolicy("t", gate, _FixedText("No wait")).decisions(_caller(1.0, 0.05), RATE, 1.0, (0.5,)) == (False,)


def test_cluster_bootstrap_rate_brackets_value() -> None:
    estimate = cluster_bootstrap_rate([True, False, True, True], ["a", "a", "b", "c"])
    assert estimate.value == 0.75 and estimate.total == 4
    assert estimate.low <= estimate.value <= estimate.high


def test_stop_rates_slices_by_label_group_and_noise() -> None:
    items = [
        EvalItem(None, "m1", "interrupt", "interrupt", "clean", 1.0),
        EvalItem(None, "m1", "ignore", "ignore:music", noise_band(3.0), 1.0),
    ]
    results = {r.slice: r.stop_rate for r in stop_rates("p", items, [(True,), (False,)], (0.5,))}
    results = {key: value.value for key, value in results.items()} | {"n_interrupt": results["interrupt"].total}
    assert results["interrupt"] == 1.0 and results["n_interrupt"] == 1
    assert results["ignore:music"] == 0.0
    assert results["ignore@0-5 dB"] == 0.0
    assert results["not_interrupt"] == 0.0
