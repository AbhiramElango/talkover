"""Download AMI individual-headset audio for dev, test and a spread of train meetings."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from talkover.interruption.corpora import AmiCorpusReader
from talkover.interruption.corpora.ami import ami_split
from talkover.interruption.paths import ami_annotations_dir, ami_audio_dir

MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"


def select_meetings(meeting_ids: list[str], train_count: int) -> list[str]:
    """All dev/test meetings plus ``train_count`` train meetings spread evenly."""

    held_out = [m for m in meeting_ids if ami_split(m) != "train"]
    train = [m for m in meeting_ids if ami_split(m) == "train"]
    step = max(1, len(train) / max(1, train_count))
    picked = [train[int(i * step)] for i in range(min(train_count, len(train)))]
    return held_out + picked


def download(url: str, target: Path) -> None:
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    partial.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-meetings", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ami_audio_dir())
    args = parser.parse_args()

    reader = AmiCorpusReader(ami_annotations_dir())
    channels = reader.channels()
    meetings = select_meetings(reader.meeting_ids(), args.train_meetings)
    for index, meeting_id in enumerate(meetings, 1):
        for channel in sorted(channels[meeting_id].values()):
            name = f"{meeting_id}.Headset-{channel}.wav"
            target = args.output / meeting_id / name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                download(f"{MIRROR}/{meeting_id}/audio/{name}", target)
            except OSError as exc:
                print(f"failed {name}: {exc}")
        print(f"[{index}/{len(meetings)}] {meeting_id}", flush=True)


if __name__ == "__main__":
    main()
