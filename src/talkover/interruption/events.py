"""Typed values for caller events that happen while the agent-role speaker talks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

from talkover.evaluation.intervals import TimeInterval


Label: TypeAlias = Literal["interrupt", "backchannel", "ignore"]
Split: TypeAlias = Literal["train", "dev", "test"]


@dataclass(frozen=True)
class Utterance:
    """One dialogue act with word-level timing; ``act`` is the corpus tag."""

    span: TimeInterval
    act: str


@dataclass(frozen=True)
class VocalSound:
    span: TimeInterval
    kind: str


@dataclass(frozen=True)
class SpeakerTrack:
    speaker: str
    channel: int
    words: tuple[TimeInterval, ...]
    utterances: tuple[Utterance, ...]
    vocal_sounds: tuple[VocalSound, ...] = ()


@dataclass(frozen=True)
class Meeting:
    meeting_id: str
    split: Split
    tracks: tuple[SpeakerTrack, ...]


@dataclass(frozen=True)
class OverlapEvent:
    """A caller-channel moment during agent speech; ``caller_span`` is None for crosstalk."""

    meeting_id: str
    split: Split
    agent_speaker: str
    agent_channel: int
    caller_speaker: str
    caller_channel: int
    onset_seconds: float
    agent_spurt: TimeInterval
    caller_span: TimeInterval | None
    act: str
    label: Label

    @property
    def event_id(self) -> str:
        onset_ms = round(self.onset_seconds * 1000)
        return f"{self.meeting_id}_{self.agent_speaker}{self.caller_speaker}_{onset_ms}_{self.act}"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OverlapEvent:
        caller_span = data["caller_span"]
        return cls(
            **{
                **data,
                "agent_spurt": TimeInterval(**data["agent_spurt"]),
                "caller_span": TimeInterval(**caller_span) if caller_span else None,
            }
        )
