"""Pipecat integration: interrupt the bot only when the caller really means it.

    detector = InterruptionDetector.from_bundle("path/to/bundle")
    strategies = UserTurnStrategies(start=[TalkoverInterruptionStrategy(detector=detector)])
    pipeline = Pipeline([
        transport.input(), stt, user_aggregator, llm, tts,
        transport.output(),
        AgentAudioTap(detector),   # after output: sees agent audio as it is played
        assistant_aggregator,
    ])

While the bot speaks, the model decides; while it is silent, a VAD speech
start begins the user turn as usual. Use this as the only start strategy:
a VAD or transcription strategy in the same list would bypass the model.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

from talkover.interruption.runtime import InterruptionDetector


def _mono_float(frame: InputAudioRawFrame | OutputAudioRawFrame) -> np.ndarray:
    samples = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
    if frame.num_channels > 1:
        samples = samples.reshape(-1, frame.num_channels).mean(axis=1)
    return samples


class AgentAudioTap(FrameProcessor):
    """Pass-through processor that feeds played bot audio into the detector."""

    def __init__(self, detector: InterruptionDetector, **kwargs) -> None:
        super().__init__(**kwargs)
        self._detector = detector

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            self._detector.push_agent(_mono_float(frame), frame.sample_rate)
        await self.push_frame(frame, direction)


class TalkoverInterruptionStrategy(BaseUserTurnStartStrategy):
    """User turn start strategy backed by the Talkover interruption model."""

    def __init__(self, *, detector: InterruptionDetector, **kwargs) -> None:
        super().__init__(**kwargs)
        self._detector = detector
        self._bot_speaking = False
        self._triggered = False

    async def handle_user_turn_started(self):
        self._triggered = True

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking, self._triggered = True, False
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, InputAudioRawFrame):
            return await self._handle_audio(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame) and not self._bot_speaking:
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE

    async def _handle_audio(self, frame: InputAudioRawFrame) -> ProcessFrameResult:
        decisions = self._detector.push_caller(_mono_float(frame), frame.sample_rate)
        if not self._bot_speaking or self._triggered:
            return ProcessFrameResult.CONTINUE
        for decision in decisions:
            if decision.interrupt:
                logger.debug(f"Talkover: interruption at {decision.time_seconds:.2f}s (p={decision.probability:.2f})")
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE
