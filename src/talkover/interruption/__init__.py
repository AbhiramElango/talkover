"""Interrupt vs. backchannel vs. ignore classification for voice agents."""

from talkover.interruption.events import (
    Label,
    Meeting,
    OverlapEvent,
    SpeakerTrack,
    Utterance,
    VocalSound,
)
from talkover.interruption.labeling import (
    EventSource,
    IgnoreConfig,
    IgnoreMiner,
    LabelingConfig,
    OverlapLabeler,
    label_meeting,
    talk_spurts,
)

__all__ = [
    "EventSource",
    "IgnoreConfig",
    "IgnoreMiner",
    "Label",
    "LabelingConfig",
    "Meeting",
    "OverlapEvent",
    "OverlapLabeler",
    "SpeakerTrack",
    "Utterance",
    "VocalSound",
    "label_meeting",
    "talk_spurts",
]
