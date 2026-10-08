import numpy as np
import pytest
import torch

from talkover.interruption.model.data import CLASSES, WindowConfig, class_balanced_weights, cut_window, window_label, ClipRow
from talkover.interruption.model.network import InterruptionClassifier
from talkover.interruption.model.streaming import stop_decisions, threshold_for_recall

RATE = 16_000


def _clip() -> np.ndarray:
    t = np.arange(2 * RATE, dtype=np.float32)
    return np.stack([t, -t], axis=1)


def test_window_ends_at_decision_and_pads_left() -> None:
    config = WindowConfig()
    window = cut_window(_clip(), 0.5, config)
    assert window.shape == (2, RATE)
    assert window[0, -1] == 1.5 * RATE - 1
    early = cut_window(_clip(), -0.5, config)
    assert not early[:, : RATE // 2].any() and early[0, -1] == 0.5 * RATE - 1


def test_mute_agent_zeroes_only_post_onset_agent() -> None:
    window = cut_window(_clip(), 0.3, WindowConfig(mute_agent_after_onset=True))
    onset = RATE - int(0.3 * RATE)
    assert window[1, :onset].all() and not window[1, onset:].any()
    assert window[0, onset:].all()


def test_pre_onset_windows_are_ignore() -> None:
    assert CLASSES[window_label("interrupt", -0.2)] == "ignore"
    assert CLASSES[window_label("interrupt", 0.3)] == "interrupt"


def test_balanced_weights_equalise_classes() -> None:
    rows = [ClipRow(None, label, "x", "m") for label in ["interrupt"] + ["ignore"] * 3 + ["backchannel"] * 2]
    weights = class_balanced_weights(rows)
    sums = {label: float(sum(w for w, r in zip(weights, rows) if r.label == label)) for label in CLASSES}
    assert all(value == pytest.approx(1.0) for value in sums.values())


def test_stop_decisions_rearm_at_onset() -> None:
    steps = (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
    probabilities = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.1])
    assert stop_decisions(probabilities, steps, 0.5, (0.0, 0.1, 0.2, 0.3)) == (True, False, True, True)


def test_threshold_for_recall_hits_target() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    threshold = threshold_for_recall(scores, 0.75)
    assert (scores >= threshold).mean() >= 0.75 and threshold == 0.7


class _TinyEncoder(torch.nn.Module):
    num_layers, dim = 3, 8

    def forward(self, waveforms):
        frames = waveforms.reshape(waveforms.shape[0], -1, 320).mean(-1, keepdim=True)
        return frames.expand(-1, -1, self.dim).unsqueeze(0).repeat(self.num_layers, 1, 1, 1)


def test_classifier_shapes_and_head_parameters() -> None:
    model = InterruptionClassifier(_TinyEncoder(), hidden=16)
    logits = model(torch.randn(4, 2, RATE))
    assert logits.shape == (4, 3)
    assert all(not name.startswith("encoder.") for name, _ in model.named_parameters() if any(p is _ for p in model.head_parameters()))


def test_caller_only_classifier_consumes_one_stream() -> None:
    from talkover.interruption.model.data import model_inputs

    config = WindowConfig(channels=[0])
    windows = torch.randn(4, 2, RATE)
    selected = model_inputs(windows, config)
    assert config.channels == (0,) and selected.shape == (4, 1, RATE)
    assert torch.equal(selected[:, 0], windows[:, 0])
    assert InterruptionClassifier(_TinyEncoder(), streams=1, hidden=16)(selected).shape == (4, 3)


def test_envelope_noise_keeps_timing_and_drops_level() -> None:
    from talkover.interruption.model.data import envelope_noise

    agent = np.zeros(RATE, dtype=np.float32)
    agent[4000:8000] = 0.5 * np.sin(np.arange(4000) / 5)
    quiet = envelope_noise(agent * 0.01, RATE)
    loud = envelope_noise(agent, RATE)
    assert np.allclose(quiet, loud, atol=1e-6)
    assert not loud[:3600].any() and loud[4400:7600].std() > 0.05
