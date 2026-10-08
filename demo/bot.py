"""One demo bot: Deepgram STT/TTS + Groq LLM over SmallWebRTC.

Both sides run the same pipeline; only the user turn start policy differs.
UI events go to the browser over the WebRTC data channel.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import aiohttp

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramHttpTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies, default_user_turn_start_strategies

from talkover.integrations.pipecat import AgentAudioTap, TalkoverInterruptionStrategy
from talkover.interruption.runtime import InterruptionDetector, ModelBundle

Emit = Callable[..., None]

SYSTEM_PROMPT = (
    "You are Maya, a fraud-prevention agent at Northwind Bank, on an outbound call to a customer about a "
    "suspicious card transaction. Your words are spoken aloud: plain sentences only, no lists, markdown or "
    "emojis, and numbers said the way a person would say them. "
    "Facts, all fictional: debit card ending in four four one seven; a charge of two thousand four hundred eighty dollars "
    "at Best Electronics Outlet in Denver, Colorado, last night at eleven forty-two PM; an online purchase; "
    "the customer lives in Boston. The card is on a temporary hold until they confirm. "
    "If they did not make it: the card is blocked, a new card arrives in three to five business days, a "
    "dispute is opened, and a provisional credit posts within ten business days. If they did make it: the "
    "hold is lifted within a few minutes and nothing else is needed. "
    "When you explain something, speak in long, detailed turns of six to eight sentences. "
    "Never ask for a full card number, PIN, password, social security number or one-time code; if it comes "
    "up, say the bank will never ask for those on a call. "
    "If the customer only says a short acknowledgement such as 'mm-hmm', 'okay', 'yeah' or 'right', continue "
    "exactly where you left off without starting over. If they interrupt with a question or an objection, "
    "answer it directly in a sentence or two, then continue."
)
OPENING = (
    "Start the call: introduce yourself and Northwind Bank, say you are calling about an unusual transaction "
    "on their debit card, read the transaction details, and explain the temporary hold in detail. Then ask "
    "whether they made this purchase."
)


@dataclass(frozen=True)
class TurnPolicy:
    """What differs between the two sides: start strategies, plus processors after the output transport."""

    start: list[BaseUserTurnStartStrategy]
    taps: list


class StockPolicy:
    name = "stock"

    def build(self, emit: Emit) -> TurnPolicy:
        return TurnPolicy(default_user_turn_start_strategies(), [])


class TalkoverPolicy:
    name = "talkover"

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.model = bundle.onnx_model()

    def build(self, emit: Emit) -> TurnPolicy:
        detector = _ReportingDetector(self.model, self.bundle.config, self.bundle.threshold, emit=emit)
        return TurnPolicy([TalkoverInterruptionStrategy(detector=detector)], [AgentAudioTap(detector)])


class _ReportingDetector(InterruptionDetector):
    def __init__(self, *args, emit: Emit, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._emit = emit

    def push_caller(self, samples, rate):
        decisions = super().push_caller(samples, rate)
        for decision in decisions:
            self._emit("prob", p=round(decision.probability, 3))
        return decisions


class UiEvents(BaseObserver):
    """Translates pipeline frames into a small event stream for the page.

    Captions: the HTTP TTS yields each sentence's audio and then its text, so the
    observer knows every sentence's duration. A sentence starts playing when the
    output transport finishes the one before it; its text can arrive later, so
    the caption carries how much of it has already played.
    """

    def __init__(self, emit: Emit, tts, output) -> None:
        super().__init__()
        self._emit, self._tts, self._output = emit, tts, output
        self._bot_speaking = False
        self._user_speaking = False
        self._interrupted = False
        self._sentences: deque[tuple[str, float]] = deque()
        self._audio_seconds = 0.0
        self._captioned = False
        self._sentence_started = 0.0

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if data.source is self._tts:
            self._on_tts(frame)
        if isinstance(frame, TTSTextFrame) and data.source is self._output:
            if self._sentences:
                self._sentences.popleft()
            self._captioned, self._sentence_started = False, time.monotonic()
            self._caption()
        if not data.first_push:
            return
        if isinstance(frame, BotStartedSpeakingFrame) and not self._bot_speaking:
            self._bot_speaking, self._interrupted = True, False
            self._sentence_started = time.monotonic()
            self._emit("bot", speaking=True)
            self._caption()
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._bot_speaking:
            self._bot_speaking = False
            self._emit("bot", speaking=False)
        elif isinstance(frame, InterruptionFrame):
            self._sentences.clear()
            self._audio_seconds, self._captioned = 0.0, False
        elif isinstance(frame, UserStartedSpeakingFrame) and self._bot_speaking and not self._interrupted:
            self._interrupted = True
            self._emit("interrupted")
        elif isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            speaking = isinstance(frame, VADUserStartedSpeakingFrame)
            if speaking != self._user_speaking:
                self._user_speaking = speaking
                self._emit("user", speaking=speaking)
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            self._emit("transcript", text=frame.text, final=isinstance(frame, TranscriptionFrame))

    def _on_tts(self, frame) -> None:
        if isinstance(frame, TTSAudioRawFrame):
            self._audio_seconds += len(frame.audio) / (2 * frame.sample_rate * frame.num_channels)
        elif isinstance(frame, TTSTextFrame):
            self._sentences.append((frame.text, self._audio_seconds))
            self._audio_seconds = 0.0
            self._caption()

    def _caption(self) -> None:
        if self._bot_speaking and self._sentences and not self._captioned:
            self._captioned = True
            text, seconds = self._sentences[0]
            elapsed = min(time.monotonic() - self._sentence_started, seconds)
            self._emit("caption", text=text, seconds=round(seconds, 2), elapsed=round(elapsed, 2))


async def run_bot(connection: SmallWebRTCConnection, policy: StockPolicy | TalkoverPolicy) -> None:
    def emit(event: str, **data) -> None:
        connection.send_app_message({"event": event, **data})

    turn = policy.build(emit)
    transport = SmallWebRTCTransport(connection, TransportParams(audio_in_enabled=True, audio_out_enabled=True))
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    session = aiohttp.ClientSession()
    tts = DeepgramHttpTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        aiohttp_session=session,
        settings=DeepgramHttpTTSService.Settings(voice="aura-2-thalia-en"),
    )
    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model="openai/gpt-oss-20b", reasoning_effort="low", system_instruction=SYSTEM_PROMPT
        ),
    )
    context = LLMContext([{"role": "user", "content": OPENING}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(), user_turn_strategies=UserTurnStrategies(start=turn.start)
        ),
    )
    pipeline = Pipeline([
        transport.input(), stt, aggregators.user(), llm, tts,
        transport.output(), *turn.taps, aggregators.assistant(),
    ])
    task = PipelineTask(pipeline, observers=[UiEvents(emit, tts, transport.output())], enable_rtvi=False)

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        await task.cancel()

    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        await session.close()
