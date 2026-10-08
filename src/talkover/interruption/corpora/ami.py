"""AMI Meeting Corpus reader for the public manual annotations (v1.6.2)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from talkover.evaluation.intervals import TimeInterval
from talkover.interruption.events import Meeting, SpeakerTrack, Split, Utterance, VocalSound


NITE_ID = "{http://nite.sourceforge.net/}id"
NITE_POINTER = "{http://nite.sourceforge.net/}pointer"
NITE_CHILD = "{http://nite.sourceforge.net/}child"
_RANGE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")

# Standard full-corpus-ASR partition; split by series so speakers never cross splits.
_DEV_SERIES = ("ES2011", "IS1008", "TS3004", "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011")
_TEST_SERIES = ("ES2004", "IS1009", "TS3003", "EN2002")


def ami_split(meeting_id: str) -> Split:
    if meeting_id.startswith(_TEST_SERIES):
        return "test"
    if meeting_id.startswith(_DEV_SERIES):
        return "dev"
    return "train"


class AmiCorpusReader:
    """Yield meetings that have dialogue-act annotations, with timed words."""

    def __init__(self, annotations_dir: Path) -> None:
        self.root = annotations_dir
        self._act_names = self._read_act_names()

    def meeting_ids(self) -> list[str]:
        files = (self.root / "dialogueActs").glob("*.dialog-act.xml")
        return sorted({path.name.split(".")[0] for path in files})

    def meetings(self) -> Iterator[Meeting]:
        channels = self.channels()
        for meeting_id in self.meeting_ids():
            yield self.read_meeting(meeting_id, channels[meeting_id])

    def read_meeting(self, meeting_id: str, channels: dict[str, int]) -> Meeting:
        tracks = []
        for speaker, channel in sorted(channels.items()):
            acts_path = self.root / "dialogueActs" / f"{meeting_id}.{speaker}.dialog-act.xml"
            words_path = self.root / "words" / f"{meeting_id}.{speaker}.words.xml"
            if not words_path.exists():
                continue
            order, times, sounds = _read_words(words_path)
            utterances = _read_acts(acts_path, order, times, self._act_names) if acts_path.exists() else ()
            tracks.append(
                SpeakerTrack(
                    speaker=speaker,
                    channel=channel,
                    words=tuple(interval for interval in times.values()),
                    utterances=utterances,
                    vocal_sounds=sounds,
                )
            )
        return Meeting(meeting_id=meeting_id, split=ami_split(meeting_id), tracks=tuple(tracks))

    def _read_act_names(self) -> dict[str, str]:
        tree = ET.parse(self.root / "ontologies" / "da-types.xml")
        return {node.get(NITE_ID): node.get("name") for node in tree.iter("da-type")}

    def channels(self) -> dict[str, dict[str, int]]:
        tree = ET.parse(self.root / "corpusResources" / "meetings.xml")
        return {
            meeting.get("observation"): {
                speaker.get("nxt_agent"): int(speaker.get("channel"))
                for speaker in meeting.iter("speaker")
            }
            for meeting in tree.iter("meeting")
        }


def _read_words(
    path: Path,
) -> tuple[dict[str, int], dict[str, TimeInterval], tuple[VocalSound, ...]]:
    order: dict[str, int] = {}
    times: dict[str, TimeInterval] = {}
    sounds: list[VocalSound] = []
    for index, node in enumerate(ET.parse(path).getroot()):
        node_id = node.get(NITE_ID)
        order[node_id] = index
        if node.tag not in ("w", "vocalsound") or node.get("punc") == "true":
            continue
        try:
            span = TimeInterval(float(node.get("starttime")), float(node.get("endtime")))
        except (TypeError, ValueError):
            continue
        if node.tag == "w":
            times[node_id] = span
        else:
            sounds.append(VocalSound(span, node.get("type", "other")))
    return order, times, tuple(sorted(sounds, key=lambda item: item.span.start_seconds))


def _read_acts(
    path: Path,
    order: dict[str, int],
    times: dict[str, TimeInterval],
    act_names: dict[str, str],
) -> tuple[Utterance, ...]:
    timed = sorted(((order[node_id], interval) for node_id, interval in times.items()))
    utterances = []
    for act in ET.parse(path).getroot().iter("dact"):
        pointer = act.find(NITE_POINTER)
        child = act.find(NITE_CHILD)
        if child is None:
            continue
        match = _RANGE.search(child.get("href", ""))
        if match is None or match.group(1) not in order:
            continue
        first = order[match.group(1)]
        last = order.get(match.group(2) or match.group(1), first)
        spans = [interval for index, interval in timed if first <= index <= last]
        if not spans:
            continue
        name = "unlab"
        if pointer is not None:
            type_match = _RANGE.search(pointer.get("href", ""))
            if type_match is not None:
                name = act_names.get(type_match.group(1), "unlab")
        utterances.append(
            Utterance(TimeInterval(spans[0].start_seconds, max(s.end_seconds for s in spans)), name)
        )
    return tuple(sorted(utterances, key=lambda item: item.span.start_seconds))


def timed_words(annotations_dir: Path, meeting_id: str, speaker: str) -> list[tuple[float, float, str]]:
    """(start, end, text) for each timed, non-punctuation word of one speaker."""

    path = annotations_dir / "words" / f"{meeting_id}.{speaker}.words.xml"
    words = []
    for node in ET.parse(path).getroot():
        if node.tag != "w" or node.get("punc") == "true" or not (node.text or "").strip():
            continue
        try:
            start, end = float(node.get("starttime")), float(node.get("endtime"))
        except (TypeError, ValueError):
            continue
        if end > start:
            words.append((start, end, node.text.strip()))
    return words
