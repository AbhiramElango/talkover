"""Copy AMI train clips with the agent channel re-voiced by macOS TTS voices (word timing preserved)."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import soundfile as sf

from talkover.interruption.corpora.ami import timed_words
from talkover.interruption.paths import ami_annotations_dir, clips_dir
from talkover.interruption.tts_agent import SAMPLE_RATE, MacSay, phrases_in_window, revoice_agent

VOICES = (
    "Daniel", "Karen", "Moira", "Tessa", "Rishi", "Aman", "Tara",
    "Eddy (English (US))", "Flo (English (US))", "Reed (English (US))",
    "Eddy (English (UK))", "Flo (English (UK))", "Reed (English (UK))",
)


def build_meeting(task: tuple[str, list[dict], str, str, int]) -> list[dict]:
    meeting_id, rows, source, target, seed = task
    source_dir, target_dir = Path(source), Path(target)
    synthesize, words = MacSay(), {}
    out = []
    for row in rows:
        speaker = row["agent_speaker"]
        if speaker not in words:
            words[speaker] = timed_words(ami_annotations_dir(), meeting_id, speaker)
        start = row["onset_seconds"] - 1.0
        phrases = phrases_in_window(words[speaker], start, start + 2.0)
        if not phrases:
            continue
        rng = random.Random(f"{seed}:{row['clip']}")
        voice = rng.choice(VOICES)
        clip, _ = sf.read(source_dir / row["clip"], dtype="float32")
        clip[:, 1] = revoice_agent(clip[:, 1], phrases, voice, synthesize)
        (target_dir / row["clip"]).parent.mkdir(parents=True, exist_ok=True)
        sf.write(target_dir / row["clip"], clip, SAMPLE_RATE, format="FLAC")
        out.append({**row, "agent_voice": voice})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=clips_dir("ami"))
    parser.add_argument("--output", type=Path, default=clips_dir("ami_tts"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-class", type=int, default=0, help="re-voice at most N clips per label (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.source / f"{args.split}.jsonl").open(encoding="utf-8")]
    if args.per_class:
        rng = random.Random(args.seed)
        by_label: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_label[row["label"]].append(row)
        rows = [row for group in by_label.values() for row in rng.sample(group, min(args.per_class, len(group)))]
    by_meeting: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_meeting[row["meeting_id"]].append(row)
    tasks = [(m, rows, str(args.source), str(args.output), args.seed) for m, rows in sorted(by_meeting.items())]
    total = sum(len(rows) for rows in by_meeting.values())
    done, written, start, last = 0, 0, time.monotonic(), 0.0
    args.output.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool, (args.output / f"{args.split}.jsonl").open("w", encoding="utf-8") as manifest:
        for (meeting_id, rows, *_), result in zip(tasks, pool.imap(build_meeting, tasks)):
            for row in result:
                manifest.write(json.dumps(row) + "\n")
            done, written = done + len(rows), written + len(result)
            elapsed = time.monotonic() - start
            if elapsed - last >= 30 or done == total:
                last = elapsed
                print(f"{done:,}/{total:,} clips ({written:,} re-voiced), {elapsed / 60:.1f} min, "
                      f"~{elapsed / done * (total - done) / 60:.1f} min left", flush=True)


if __name__ == "__main__":
    main()
