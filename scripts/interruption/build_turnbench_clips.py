"""Cut 2 s stereo clips (caller, other speaker) around TurnBench consensus events."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from talkover.interruption.corpora.turnbench import consensus_events, decode_channel, read_conversations
from talkover.interruption.paths import clips_dir, data_dir

SAMPLE_RATE = 16_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=data_dir() / "raw" / "turnbench" / "data")
    parser.add_argument("--output", type=Path, default=clips_dir("turnbench"))
    args = parser.parse_args()

    files = sorted(args.source.glob("*.parquet"))
    counts: Counter[str] = Counter()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "test.jsonl").open("w", encoding="utf-8") as manifest:
        for number, row in enumerate(read_conversations(files), 1):
            channels = {1: decode_channel(row["speaker_1_audio"]), 2: decode_channel(row["speaker_2_audio"])}
            for event in consensus_events(row):
                caller, agent = channels[event.caller], channels[3 - event.caller]
                first = round((event.onset_seconds - 1.0) * SAMPLE_RATE)
                if first < 0 or first + 2 * SAMPLE_RATE > min(caller.size, agent.size):
                    continue
                clip = np.stack([caller[first : first + 2 * SAMPLE_RATE], agent[first : first + 2 * SAMPLE_RATE]], axis=1)
                relative = Path("test") / event.label / f"{event.conversation_id}_s{event.caller}_{round(event.onset_seconds * 1000)}.flac"
                (args.output / relative).parent.mkdir(parents=True, exist_ok=True)
                sf.write(args.output / relative, clip, SAMPLE_RATE, format="FLAC")
                manifest.write(json.dumps({
                    "clip": str(relative), "meeting_id": event.conversation_id, "split": "test", "label": event.label,
                    "act": "|".join(event.source_labels), "onset_seconds": event.onset_seconds,
                    "conversation_type": (row.get("metadata") or {}).get("conversation_type", ""),
                }) + "\n")
                counts[event.label] += 1
            print(f"conversation {number}/38 {row['conversation_id'][:12]}: {dict(counts)}, {(time.monotonic() - start) / 60:.1f} min", flush=True)
    print("total", dict(counts))


if __name__ == "__main__":
    main()
