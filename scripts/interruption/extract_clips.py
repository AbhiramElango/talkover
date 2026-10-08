"""Cut 2 s stereo FLAC clips for every event whose audio is downloaded."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from talkover.interruption.clips import ClipExtractor, ami_channel_path, balance_ignore
from talkover.interruption.events import OverlapEvent
from talkover.interruption.paths import ami_audio_dir, clips_dir, events_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=events_path("ami"))
    parser.add_argument("--output", type=Path, default=clips_dir("ami"))
    parser.add_argument("--crosstalk-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    extractor = ClipExtractor(ami_channel_path(ami_audio_dir()))
    with args.events.open(encoding="utf-8") as handle:
        events = [OverlapEvent.from_json(json.loads(line)) for line in handle]
    events = balance_ignore([event for event in events if extractor.available(event)], args.crosstalk_ratio, args.seed)

    counts: Counter[tuple[str, str]] = Counter()
    manifests = {}
    for event in events:
        clip = extractor.extract(event)
        if clip is None:
            continue
        relative = Path(event.split) / event.label / f"{event.event_id}.flac"
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, clip, extractor.config.sample_rate_hz, format="FLAC")
        if event.split not in manifests:
            manifests[event.split] = (args.output / f"{event.split}.jsonl").open("w", encoding="utf-8")
        manifests[event.split].write(json.dumps({"clip": str(relative), **event.to_json()}) + "\n")
        counts[(event.split, event.label)] += 1
    for handle in manifests.values():
        handle.close()

    for (split, label), count in sorted(counts.items()):
        print(f"{split:5s} {label:11s} {count}")


if __name__ == "__main__":
    main()
