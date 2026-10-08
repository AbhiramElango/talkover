"""Label caller onsets during agent speech and write them as JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from talkover.interruption import IgnoreMiner, OverlapLabeler, label_meeting
from talkover.interruption.corpora import AmiCorpusReader
from talkover.interruption.paths import ami_annotations_dir, events_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ami_annotations_dir())
    parser.add_argument("--output", type=Path, default=events_path("ami"))
    args = parser.parse_args()

    sources = (OverlapLabeler(), IgnoreMiner())
    counts: Counter[tuple[str, str, str]] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for meeting in AmiCorpusReader(args.annotations).meetings():
            for event in label_meeting(meeting, sources):
                handle.write(json.dumps(event.to_json()) + "\n")
                kind = event.act if event.label == "ignore" else ""
                counts[(event.split, event.label, kind)] += 1

    for (split, label, kind), count in sorted(counts.items()):
        print(f"{split:5s} {label:11s} {kind:12s} {count}")
    print(f"wrote {sum(counts.values())} events to {args.output}")


if __name__ == "__main__":
    main()
