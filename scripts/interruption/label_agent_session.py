"""Label caller onsets in recorded agent sessions, then export 2 s clips for evaluation.

Plays each candidate (1 s before to 1 s after the onset, your mic after echo cancellation). Keys:
  i = interrupt   b = backchannel   n = ignore (noise, echo, other voice, laugh)
  x = unclear (excluded)   r = replay   q = save and quit
Resumable: already-labelled candidates are skipped.

    .venv/bin/python scripts/interruption/label_agent_session.py --session s1
    .venv/bin/python scripts/interruption/label_agent_session.py --export
"""

from __future__ import annotations

import argparse
import json

import sounddevice as sd

from talkover.interruption.corpora.agent_calls import (
    SAMPLE_RATE,
    WebRtcEchoCanceller,
    cancel_echo,
    candidate_json,
    export_clips,
    find_candidates,
    read_session,
    turn_audio,
)
from talkover.interruption.paths import clips_dir, data_dir

KEYS = {"i": "interrupt", "b": "backchannel", "n": "ignore", "x": "unclear"}


def label_session(session: str) -> None:
    root = data_dir() / "raw" / "agent_calls"
    labels_path = root / "labels.jsonl"
    done = set()
    if labels_path.exists():
        for line in labels_path.open(encoding="utf-8"):
            row = json.loads(line)
            done.add((row["session"], row["turn"], row["onset_seconds"]))
    meta = read_session(root / session)
    canceller = WebRtcEchoCanceller()
    with labels_path.open("a", encoding="utf-8") as out:
        for turn in meta["turns"]:
            audio = cancel_echo(turn_audio(root / session, turn["index"]), canceller)
            candidates = find_candidates(audio, session, turn["index"], turn["instruction"])
            print(f"\nturn {turn['index']} [{turn['instruction']}]: {len(candidates)} candidates")
            for number, candidate in enumerate(candidates, 1):
                if (candidate.session, candidate.turn, candidate.onset_seconds) in done:
                    continue
                start = round((candidate.onset_seconds - 1.0) * SAMPLE_RATE)
                clip = audio[start : start + 2 * SAMPLE_RATE, 0]
                while True:
                    sd.play(clip / max(1e-3, float(abs(clip).max())) * 0.8, SAMPLE_RATE)
                    key = input(f"  {number}/{len(candidates)} at {candidate.onset_seconds:.2f}s  [i/b/n/x, r=replay, q=quit]: ").strip().lower()
                    if key == "r":
                        continue
                    if key == "q":
                        return
                    if key in KEYS:
                        break
                out.write(json.dumps({**candidate_json(candidate), "label": KEYS[key]}) + "\n")
                out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session")
    parser.add_argument("--export", action="store_true", help="write labelled clips to data/clips/<name>")
    parser.add_argument("--sessions", nargs="*", help="export only these sessions")
    parser.add_argument("--name", default="agent_calls", help="clip set name")
    args = parser.parse_args()
    if args.session:
        label_session(args.session)
    if args.export:
        root = data_dir() / "raw" / "agent_calls"
        target = clips_dir(args.name)
        count = export_clips(root / "labels.jsonl", root, target, WebRtcEchoCanceller(), sessions=set(args.sessions or []))
        print(f"exported {count} clips to {target}")


if __name__ == "__main__":
    main()
