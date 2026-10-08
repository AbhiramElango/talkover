"""Event sources that label caller-channel moments during agent speech."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from talkover.evaluation.intervals import TimeInterval
from talkover.interruption.events import Label, Meeting, OverlapEvent, SpeakerTrack, Utterance


@dataclass(frozen=True)
class LabelingConfig:
    spurt_gap_seconds: float = 0.3
    min_agent_lead_seconds: float = 0.5
    min_agent_remaining_seconds: float = 0.5
    min_caller_silence_seconds: float = 0.5
    max_agent_yield_seconds: float = 1.5
    min_caller_hold_seconds: float = 1.0
    backchannel_acts: frozenset[str] = frozenset({"bck"})
    excluded_acts: frozenset[str] = frozenset({"stl", "fra", "unlab"})


@dataclass(frozen=True)
class IgnoreConfig:
    vocal_kinds: frozenset[str] = frozenset({"laugh", "cough", "sigh", "sharp inhale", "sharp exhale"})
    speech_clearance_seconds: float = 1.0
    min_crosstalk_spurt_seconds: float = 3.0


class EventSource(Protocol):
    def events(self, meeting: Meeting) -> Iterator[OverlapEvent]: ...


def label_meeting(meeting: Meeting, sources: Iterable[EventSource]) -> Iterator[OverlapEvent]:
    for source in sources:
        yield from source.events(meeting)


def talk_spurts(words: Iterable[TimeInterval], gap_seconds: float) -> tuple[TimeInterval, ...]:
    """Merge words separated by pauses no longer than ``gap_seconds``."""

    spurts: list[TimeInterval] = []
    for word in sorted(words):
        if spurts and word.start_seconds - spurts[-1].end_seconds <= gap_seconds:
            last = spurts[-1]
            spurts[-1] = TimeInterval(last.start_seconds, max(last.end_seconds, word.end_seconds))
        else:
            spurts.append(word)
    return tuple(spurts)


class _Spurts:
    def __init__(self, words: Iterable[TimeInterval], gap_seconds: float) -> None:
        self.items = talk_spurts(words, gap_seconds)
        self.starts = [item.start_seconds for item in self.items]

    def at(self, t: float) -> TimeInterval | None:
        index = bisect_right(self.starts, t) - 1
        if index >= 0 and self.items[index].end_seconds > t:
            return self.items[index]
        return None

    def previous_end(self, t: float) -> float | None:
        index = bisect_right(self.starts, t) - 1
        while index >= 0 and self.items[index].start_seconds >= t:
            index -= 1
        return self.items[index].end_seconds if index >= 0 else None

    def overlaps(self, window: TimeInterval) -> bool:
        index = bisect_right(self.starts, window.end_seconds) - 1
        return index >= 0 and self.items[index].end_seconds > window.start_seconds


class _PairSource:
    """Shared agent/caller pairing and the agent mid-spurt rule."""

    def __init__(self, config: LabelingConfig | None = None) -> None:
        self.config = config or LabelingConfig()

    def events(self, meeting: Meeting) -> Iterator[OverlapEvent]:
        spurts = {
            track.speaker: _Spurts(track.words, self.config.spurt_gap_seconds)
            for track in meeting.tracks
        }
        for agent in meeting.tracks:
            for caller in meeting.tracks:
                if agent.speaker != caller.speaker:
                    yield from self._pair_events(
                        meeting, agent, caller, spurts[agent.speaker], spurts[caller.speaker]
                    )

    def _pair_events(
        self, meeting: Meeting, agent: SpeakerTrack, caller: SpeakerTrack,
        agent_spurts: _Spurts, caller_spurts: _Spurts,
    ) -> Iterator[OverlapEvent]:
        raise NotImplementedError

    def _agent_spurt(self, agent_spurts: _Spurts, onset: float) -> TimeInterval | None:
        spurt = agent_spurts.at(onset)
        if spurt is None:
            return None
        if onset - spurt.start_seconds < self.config.min_agent_lead_seconds:
            return None
        if spurt.end_seconds - onset < self.config.min_agent_remaining_seconds:
            return None
        return spurt

    @staticmethod
    def _event(
        meeting: Meeting, agent: SpeakerTrack, caller: SpeakerTrack, onset: float,
        agent_spurt: TimeInterval, caller_span: TimeInterval | None, act: str, label: Label,
    ) -> OverlapEvent:
        return OverlapEvent(
            meeting_id=meeting.meeting_id,
            split=meeting.split,
            agent_speaker=agent.speaker,
            agent_channel=agent.channel,
            caller_speaker=caller.speaker,
            caller_channel=caller.channel,
            onset_seconds=onset,
            agent_spurt=agent_spurt,
            caller_span=caller_span,
            act=act,
            label=label,
        )


class OverlapLabeler(_PairSource):
    """Caller speech onsets labelled interrupt or backchannel; ambiguous ones are dropped."""

    def _pair_events(self, meeting, agent, caller, agent_spurts, caller_spurts):
        config = self.config
        for utterance in caller.utterances:
            onset = utterance.span.start_seconds
            agent_spurt = self._agent_spurt(agent_spurts, onset)
            caller_spurt = caller_spurts.at(onset)
            if agent_spurt is None or caller_spurt is None or caller_spurt.start_seconds < onset:
                continue
            previous = caller_spurts.previous_end(onset)
            if previous is not None and onset - previous < config.min_caller_silence_seconds:
                continue
            label = self._label(utterance, agent_spurt, caller_spurt)
            if label is not None:
                yield self._event(
                    meeting, agent, caller, onset, agent_spurt, caller_spurt, utterance.act, label
                )

    def _label(
        self, utterance: Utterance, agent_spurt: TimeInterval, caller_spurt: TimeInterval
    ) -> Label | None:
        config = self.config
        onset = utterance.span.start_seconds
        if utterance.act in config.backchannel_acts:
            return "backchannel"
        if utterance.act in config.excluded_acts:
            return None
        yielded = agent_spurt.end_seconds - onset <= config.max_agent_yield_seconds
        held = caller_spurt.end_seconds - onset >= config.min_caller_hold_seconds
        if yielded and held and caller_spurt.end_seconds > agent_spurt.end_seconds:
            return "interrupt"
        return None


class IgnoreMiner(_PairSource):
    """Caller non-speech during agent speech: vocal sounds and crosstalk-only windows."""

    def __init__(
        self, config: LabelingConfig | None = None, ignore: IgnoreConfig | None = None
    ) -> None:
        super().__init__(config)
        self.ignore = ignore or IgnoreConfig()

    def _pair_events(self, meeting, agent, caller, agent_spurts, caller_spurts):
        clearance = self.ignore.speech_clearance_seconds

        def caller_clear(window: TimeInterval) -> bool:
            return not caller_spurts.overlaps(window)

        for sound in caller.vocal_sounds:
            if sound.kind not in self.ignore.vocal_kinds:
                continue
            onset = sound.span.start_seconds
            agent_spurt = self._agent_spurt(agent_spurts, onset)
            window = TimeInterval(max(0.0, onset - clearance), sound.span.end_seconds + clearance)
            if agent_spurt is not None and caller_clear(window):
                yield self._event(
                    meeting, agent, caller, onset, agent_spurt, sound.span, sound.kind, "ignore"
                )

        sounds = [sound.span for sound in caller.vocal_sounds]
        lead = self.config.min_agent_lead_seconds
        for spurt in agent_spurts.items:
            if spurt.duration_seconds < self.ignore.min_crosstalk_spurt_seconds:
                continue
            onset = round((spurt.start_seconds + lead + spurt.end_seconds) / 2, 3)
            window = TimeInterval(max(0.0, onset - clearance), onset + clearance)
            if self._agent_spurt(agent_spurts, onset) is None or not caller_clear(window):
                continue
            if any(s.start_seconds < window.end_seconds and s.end_seconds > window.start_seconds for s in sounds):
                continue
            yield self._event(meeting, agent, caller, onset, spurt, None, "crosstalk", "ignore")
