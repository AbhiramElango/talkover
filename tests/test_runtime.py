import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from talkover.interruption.model.windows import WindowConfig, cut_window
from talkover.interruption.runtime import Decision, InterruptionDetector, ModelBundle

RATE = 16_000
CONFIG = WindowConfig()


def _fingerprint(windows: np.ndarray) -> np.ndarray:
    weights = np.linspace(0.0, 1.0, windows.shape[-1], dtype=np.float32)
    return (windows[:, 0] @ weights + 2 * (windows[:, 1] @ weights)).astype(np.float32)


def _clip(seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((int(seconds * RATE), 2)).astype(np.float32) * 0.1


def _stream(detector: InterruptionDetector, clip: np.ndarray, chunk: int = 320) -> list[Decision]:
    decisions = []
    for start in range(0, clip.shape[0], chunk):
        detector.push_agent(clip[start : start + chunk, 1], RATE)
        decisions += detector.push_caller(clip[start : start + chunk, 0], RATE)
    return decisions


def test_streaming_windows_match_offline_windows() -> None:
    clip = _clip()
    decisions = _stream(InterruptionDetector(_fingerprint, CONFIG, threshold=0.0), clip)
    assert [d.time_seconds for d in decisions] == pytest.approx(np.arange(0.1, 2.01, 0.1))
    for decision in decisions:
        offline = _fingerprint(cut_window(clip, decision.time_seconds - CONFIG.onset_seconds, CONFIG)[None])[0]
        assert decision.probability == pytest.approx(float(offline), rel=1e-5)


def test_silent_agent_is_zero_padded_to_stay_aligned() -> None:
    detector = InterruptionDetector(_fingerprint, CONFIG, threshold=0.5)
    for _ in range(50):
        detector.push_caller(np.full(320, 0.1, dtype=np.float32), RATE)
    window = detector.window()
    assert window.shape == (1, 2, RATE) and not window[0, 1].any() and window[0, 0].all()


def test_agent_audio_is_resampled_to_model_rate() -> None:
    detector = InterruptionDetector(_fingerprint, CONFIG, threshold=0.5)
    t = np.arange(24_000) / 24_000
    detector.push_agent(np.sin(2 * np.pi * 200 * t).astype(np.float32), 24_000)
    detector.push_caller(np.zeros(RATE, dtype=np.float32), RATE)
    agent = detector.window()[0, 1]
    assert agent.shape == (RATE,) and abs(np.abs(agent).max() - 1.0) < 0.05


def test_threshold_sets_interrupt_flag() -> None:
    detector = InterruptionDetector(lambda w: np.array([0.7], dtype=np.float32), CONFIG, threshold=0.6)
    (decision,) = detector.push_caller(np.zeros(1600, dtype=np.float32), RATE)
    assert decision == Decision(0.1, pytest.approx(0.7), True)


def test_bundle_round_trip_and_checksum(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"weights")
    ModelBundle(tmp_path, "model.onnx", CONFIG, 0.6).save({"run": "r"})
    ModelBundle(tmp_path, "model.onnx", CONFIG, 0.4).save()
    loaded = ModelBundle.load(tmp_path)
    assert loaded.threshold == 0.4 and loaded.config == CONFIG
    assert json.loads((tmp_path / "bundle.json").read_text())["run"] == "r"
    (tmp_path / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        ModelBundle.load(tmp_path)


RELEASE = Path(__file__).resolve().parents[1] / "data" / "release" / "talkover"
CLIPS = Path(__file__).resolve().parents[1] / "data" / "clips" / "agent_calls" / "test.jsonl"


@pytest.mark.skipif(not (RELEASE.exists() and CLIPS.exists()), reason="needs the packaged model and recorded clips")
def test_released_model_streams_like_offline_scoring() -> None:
    from talkover.interruption.model.streaming import OnnxStreamingScorer

    bundle = ModelBundle.load(RELEASE)
    row = json.loads(CLIPS.read_text().splitlines()[0])
    clip, _ = sf.read(CLIPS.parent / row["clip"], dtype="float32")
    steps = (0.1, 0.2, 0.3, 0.4, 0.5)
    offline = OnnxStreamingScorer(str(RELEASE / bundle.model_file), bundle.config, steps).probabilities(clip)
    live = {d.time_seconds: d.probability for d in _stream(InterruptionDetector.from_bundle(bundle), clip)}
    assert [live[round(1.0 + s, 3)] for s in steps] == pytest.approx(offline.tolist(), abs=1e-4)


def test_pipecat_strategy_defers_to_model_only_while_bot_speaks() -> None:
    pipecat = pytest.importorskip("talkover.integrations.pipecat")
    from pipecat.frames.frames import BotStartedSpeakingFrame, InputAudioRawFrame, VADUserStartedSpeakingFrame
    from pipecat.turns.types import ProcessFrameResult

    class FakeDetector:
        def __init__(self) -> None:
            self.interrupt = False

        def push_caller(self, samples, rate):
            return [Decision(0.1, 0.9 if self.interrupt else 0.1, self.interrupt)]

    detector = FakeDetector()
    strategy = pipecat.TalkoverInterruptionStrategy(detector=detector)
    triggers = []

    async def trigger(**kwargs):
        triggers.append(kwargs)

    strategy.trigger_user_turn_started = trigger
    audio = InputAudioRawFrame(audio=np.zeros(320, dtype=np.int16).tobytes(), sample_rate=RATE, num_channels=1)

    async def run() -> list:
        results = [await strategy.process_frame(VADUserStartedSpeakingFrame())]
        await strategy.process_frame(BotStartedSpeakingFrame())
        results.append(await strategy.process_frame(VADUserStartedSpeakingFrame()))
        results.append(await strategy.process_frame(audio))
        detector.interrupt = True
        results.append(await strategy.process_frame(audio))
        await strategy.handle_user_turn_started()
        results.append(await strategy.process_frame(audio))
        return results

    results = asyncio.run(run())
    assert results == [ProcessFrameResult.STOP, ProcessFrameResult.CONTINUE, ProcessFrameResult.CONTINUE,
                       ProcessFrameResult.STOP, ProcessFrameResult.CONTINUE]
    assert len(triggers) == 2


@pytest.mark.parametrize("interrupt_after", [3, None])
def test_strategy_inside_pipecat_turn_processor(interrupt_after) -> None:
    pipecat = pytest.importorskip("talkover.integrations.pipecat")
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        InputAudioRawFrame,
        UserStartedSpeakingFrame,
        VADUserStartedSpeakingFrame,
    )
    from pipecat.tests.utils import run_test
    from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
    from pipecat.turns.user_turn_processor import UserTurnProcessor
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    class CountingDetector:
        pushes = 0

        def push_caller(self, samples, rate):
            self.pushes += 1
            fire = interrupt_after is not None and self.pushes >= interrupt_after
            return [Decision(self.pushes / 10, 0.9 if fire else 0.1, fire)]

    strategy = pipecat.TalkoverInterruptionStrategy(detector=CountingDetector())
    processor = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(start=[strategy], stop=[SpeechTimeoutUserTurnStopStrategy()])
    )
    audio = [InputAudioRawFrame(audio=np.zeros(320, dtype=np.int16).tobytes(), sample_rate=RATE, num_channels=1) for _ in range(5)]
    down, _ = asyncio.run(run_test(processor, frames_to_send=[BotStartedSpeakingFrame(), VADUserStartedSpeakingFrame(), *audio]))
    started = [f for f in down if isinstance(f, UserStartedSpeakingFrame)]
    assert len(started) == (1 if interrupt_after else 0)
