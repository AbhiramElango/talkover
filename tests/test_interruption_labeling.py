from pathlib import Path

from talkover.evaluation.intervals import TimeInterval
from talkover.interruption import (
    IgnoreMiner,
    Meeting,
    OverlapEvent,
    OverlapLabeler,
    SpeakerTrack,
    Utterance,
    VocalSound,
    talk_spurts,
)
from talkover.interruption.corpora import AmiCorpusReader
from talkover.interruption.corpora.ami import ami_split


def _words(*spans: tuple[float, float]) -> tuple[TimeInterval, ...]:
    return tuple(TimeInterval(start, end) for start, end in spans)


def _track(speaker: str, words, acts, sounds=()) -> SpeakerTrack:
    utterances = tuple(Utterance(TimeInterval(start, end), act) for start, end, act in acts)
    vocal = tuple(VocalSound(TimeInterval(start, end), kind) for start, end, kind in sounds)
    return SpeakerTrack(speaker, ord(speaker) - ord("A"), _words(*words), utterances, vocal)


def _events(source, agent: SpeakerTrack, caller: SpeakerTrack) -> list[OverlapEvent]:
    meeting = Meeting("M", "train", (agent, caller))
    return [event for event in source.events(meeting) if event.agent_speaker == agent.speaker]


def _labels(agent: SpeakerTrack, caller: SpeakerTrack) -> list[str]:
    return [event.label for event in _events(OverlapLabeler(), agent, caller)]


def test_talk_spurts_merge_short_pauses_only() -> None:
    spurts = talk_spurts(_words((0, 1), (1.2, 2), (3, 4)), gap_seconds=0.3)
    assert spurts == _words((0, 2), (3, 4))


def test_backchannel_during_agent_speech() -> None:
    agent = _track("A", [(0, 5)], [(0, 5, "inf")])
    caller = _track("B", [(2, 2.4)], [(2, 2.4, "bck")])
    assert _labels(agent, caller) == ["backchannel"]


def test_floor_taking_overlap_is_interrupt() -> None:
    agent = _track("A", [(0, 3)], [(0, 3, "inf")])
    caller = _track("B", [(2, 5)], [(2, 5, "inf")])
    assert _labels(agent, caller) == ["interrupt"]


def test_overlap_where_agent_keeps_talking_is_dropped() -> None:
    agent = _track("A", [(0, 8)], [(0, 8, "inf")])
    caller = _track("B", [(2, 4)], [(2, 4, "ass")])
    assert _labels(agent, caller) == []


def test_onset_near_agent_end_or_start_is_dropped() -> None:
    near_end = _track("A", [(0, 2.2)], [(0, 2.2, "inf")])
    near_start = _track("A", [(1.8, 5)], [(1.8, 5, "inf")])
    caller = _track("B", [(2, 5)], [(2, 5, "inf")])
    assert _labels(near_end, caller) == []
    assert _labels(near_start, caller) == []


def test_continuation_of_caller_turn_is_dropped() -> None:
    agent = _track("A", [(0, 5)], [(0, 5, "inf")])
    caller = _track("B", [(1.5, 1.9), (2, 2.3)], [(2, 2.3, "bck")])
    assert _labels(agent, caller) == []


def test_caller_laugh_during_agent_speech_is_ignore() -> None:
    agent = _track("A", [(0, 2.0)], [(0, 2.0, "inf")])
    caller = _track("B", [], [], sounds=[(1.0, 1.4, "laugh"), (1.2, 1.3, "other")])
    events = _events(IgnoreMiner(), agent, caller)
    assert [(event.act, event.label) for event in events] == [("laugh", "ignore")]


def test_crosstalk_window_requires_silent_caller() -> None:
    agent = _track("A", [(0, 6)], [(0, 6, "inf")])
    silent = _track("B", [], [])
    talking = _track("B", [(3.0, 3.4)], [(3.0, 3.4, "bck")])
    (event,) = _events(IgnoreMiner(), agent, silent)
    assert (event.act, event.onset_seconds, event.caller_span) == ("crosstalk", 3.25, None)
    assert _events(IgnoreMiner(), agent, talking) == []


def test_event_json_round_trip() -> None:
    agent = _track("A", [(0, 6)], [(0, 6, "inf")])
    for source in (IgnoreMiner(), OverlapLabeler()):
        for event in _events(source, agent, _track("B", [(2, 2.4)], [(2, 2.4, "bck")], [(4.5, 4.8, "cough")])):
            assert OverlapEvent.from_json(event.to_json()) == event


def test_ami_split_by_series() -> None:
    assert ami_split("ES2004a") == "test"
    assert ami_split("IS1008c") == "dev"
    assert ami_split("ES2002a") == "train"


def test_ami_reader_parses_timed_dialogue_acts(tmp_path: Path) -> None:
    nite = 'xmlns:nite="http://nite.sourceforge.net/"'
    files = {
        "ontologies/da-types.xml": f'<da-type nite:id="root" {nite}><da-type nite:id="ami_da_1" name="bck"/></da-type>',
        "corpusResources/meetings.xml": (
            f'<meetings {nite}><meeting observation="ES2002a">'
            '<speaker nxt_agent="A" channel="1"/></meeting></meetings>'
        ),
        "words/ES2002a.A.words.xml": (
            f'<nite:root {nite}><w nite:id="w0" starttime="1.0" endtime="1.3">Mm</w>'
            '<w nite:id="w1" starttime="1.3" endtime="1.3" punc="true">.</w>'
            '<vocalsound nite:id="v0" starttime="1.4" endtime="1.5" type="laugh"/></nite:root>'
        ),
        "dialogueActs/ES2002a.A.dialog-act.xml": (
            f'<nite:root {nite}><dact nite:id="d0">'
            '<nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_1)"/>'
            '<nite:child href="ES2002a.A.words.xml#id(w0)..id(v0)"/></dact></nite:root>'
        ),
    }
    for name, content in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(content, encoding="utf-8")

    (meeting,) = AmiCorpusReader(tmp_path).meetings()

    (track,) = meeting.tracks
    assert (meeting.split, track.speaker, track.channel) == ("train", "A", 1)
    assert track.words == _words((1.0, 1.3))
    assert track.utterances == (Utterance(TimeInterval(1.0, 1.3), "bck"),)
    assert track.vocal_sounds == (VocalSound(TimeInterval(1.4, 1.5), "laugh"),)
