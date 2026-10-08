"""Stream a recorded agent-call turn through the live detector and print when it would stop the agent.

    .venv/bin/python scripts/interruption/replay_call.py --session s1 --turn 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from talkover.interruption.corpora.agent_calls import SAMPLE_RATE, WebRtcEchoCanceller, cancel_echo, turn_audio
from talkover.interruption.paths import data_dir
from talkover.interruption.runtime import InterruptionDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--turn", type=int, required=True)
    parser.add_argument("--bundle", type=Path, default=data_dir() / "release" / "talkover")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    root = data_dir() / "raw" / "agent_calls"
    audio = cancel_echo(turn_audio(root / args.session, args.turn), WebRtcEchoCanceller())
    detector = InterruptionDetector.from_bundle(args.bundle, args.threshold)
    labels = {}
    if (root / "labels.jsonl").exists():
        for line in (root / "labels.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row["session"] == args.session and row["turn"] == args.turn:
                labels[row["onset_seconds"]] = row["label"]

    chunk = SAMPLE_RATE // 50
    print(f"threshold {detector.threshold:.2f}; labelled onsets: " + (", ".join(f"{t:.2f}s {l}" for t, l in sorted(labels.items())) or "none"))
    for start in range(0, audio.shape[0], chunk):
        detector.push_agent(audio[start : start + chunk, 1], SAMPLE_RATE)
        for decision in detector.push_caller(audio[start : start + chunk, 0], SAMPLE_RATE):
            bar = "#" * round(decision.probability * 30)
            flag = "  <- STOP" if decision.interrupt else ""
            print(f"{decision.time_seconds:6.1f}s  p={decision.probability:.2f}  {bar:30s}{flag}")


if __name__ == "__main__":
    main()
