"""Dataset locations; override the root with ``TALKOVER_DATA_DIR``."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    default = Path(__file__).resolve().parents[3] / "data"
    return Path(os.environ.get("TALKOVER_DATA_DIR", default))


def ami_annotations_dir() -> Path:
    return data_dir() / "raw" / "ami" / "annotations"


def ami_audio_dir() -> Path:
    return data_dir() / "raw" / "ami" / "audio"


def events_path(corpus: str) -> Path:
    return data_dir() / "interim" / f"{corpus}_events.jsonl"


def clips_dir(corpus: str) -> Path:
    return data_dir() / "clips" / corpus
