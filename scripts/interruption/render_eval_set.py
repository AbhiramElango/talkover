"""Render fixed, seeded augmented dev/test clips and record what was applied."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from talkover.interruption.augment import AugmentContext, MusanBank, default_pipelines, event_rng, rms_dbfs
from talkover.interruption.clips import ClipConfig
from talkover.interruption.paths import clips_dir, data_dir


def reference_dbfs(clips_root: Path, onset_frame: int) -> float:
    """Median caller level after onset over train speech clips; cached in stats.json."""

    stats_path = clips_root / "stats.json"
    if stats_path.exists():
        return json.loads(stats_path.read_text())["reference_dbfs"]
    levels = []
    with (clips_root / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["label"] != "ignore":
                samples, _ = sf.read(clips_root / row["clip"], dtype="float32")
                levels.append(rms_dbfs(samples[onset_frame:, 0]))
    value = float(np.median(levels))
    stats_path.write_text(json.dumps({"reference_dbfs": value, "train_speech_clips": len(levels)}))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=clips_dir("ami"))
    parser.add_argument("--output", type=Path, default=clips_dir("ami_aug"))
    parser.add_argument("--musan", type=Path, default=data_dir() / "raw" / "musan" / "musan")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = ClipConfig()
    onset_frame = round(config.before_seconds * config.sample_rate_hz)
    context = AugmentContext(config.sample_rate_hz, onset_frame, reference_dbfs(args.clips, onset_frame))
    print(f"reference level {context.reference_dbfs:.1f} dBFS")

    for split in args.splits:
        speech, silent_ignore = default_pipelines(MusanBank(args.musan, split))
        counts: Counter[str] = Counter()
        args.output.joinpath(split).mkdir(parents=True, exist_ok=True)
        with (args.clips / f"{split}.jsonl").open(encoding="utf-8") as rows, (
            args.output / f"{split}.jsonl"
        ).open("w", encoding="utf-8") as manifest:
            for line in rows:
                row = json.loads(line)
                clip, _ = sf.read(args.clips / row["clip"], dtype="float32")
                event_id = Path(row["clip"]).stem
                pipeline = silent_ignore if row["act"] == "crosstalk" else speech
                augmented, params = pipeline(clip, context, event_rng(event_id, args.seed))
                target = args.output / row["clip"]
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, augmented, config.sample_rate_hz, format="FLAC")
                manifest.write(json.dumps({**row, "augmentation": params}) + "\n")
                counts[row["label"]] += 1
        print(split, dict(counts))


if __name__ == "__main__":
    main()
