from talkover.interruption.corpora.turnbench import consensus_events


def _row(caller_tracks, agent_track):
    row = {"conversation_id": "c1"}
    for annotator, track in zip("abc", caller_tracks):
        row[f"speaker_1_annotation_{annotator}"] = track
        row[f"speaker_2_annotation_{annotator}"] = agent_track if annotator == "a" else []
    return row


def _event(start, label):
    return {"start_s": start, "end_s": start + 0.5, "label": label, "text": ""}


AGENT = [{"start_s": 0.0, "end_s": 20.0, "label": "Normal Turn", "text": ""}]


def test_two_annotators_agreeing_make_an_event() -> None:
    row = _row([[_event(5.0, "Continuer Backchannel")], [_event(5.2, "Acknowledgement Backchannel")], []], AGENT)
    (event,) = consensus_events(row)
    assert (event.caller, event.label, event.onset_seconds) == (1, "backchannel", 5.1)


def test_single_annotator_or_disagreement_is_dropped() -> None:
    assert consensus_events(_row([[_event(5.0, "Laughter")], [], []], AGENT)) == []
    mixed = [[_event(5.0, "Laughter")], [_event(5.1, "Floor-taking Competitive Interruption")], [_event(5.2, "Filler")]]
    assert consensus_events(_row(mixed, AGENT)) == []


def test_requires_other_speaker_mid_speech() -> None:
    row = _row([[_event(19.8, "Laughter")], [_event(19.8, "Laughter")], []], AGENT)
    assert consensus_events(row) == []
    row = _row([[_event(10.0, "Floor-taking Cooperative Interruption")]] * 3, AGENT)
    assert [e.label for e in consensus_events(row)] == ["interrupt"]
