"""Set a bundle's threshold from your own labelled calls (clip format of label_agent_session.py --export)."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from talkover.interruption.model.data import read_manifest
from talkover.interruption.model.streaming import STEPS, OnnxStreamingScorer, threshold_for_recall
from talkover.interruption.runtime import ModelBundle

HORIZON = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True, help="manifest, e.g. data/clips/agent_calls/test.jsonl")
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--write", action="store_true", help="save the new threshold into the bundle")
    args = parser.parse_args()

    bundle = ModelBundle.load(args.bundle)
    scorer = OnnxStreamingScorer(str(bundle.directory / bundle.model_file), bundle.config, STEPS)
    steps = np.asarray(scorer.steps)
    post = (steps > 1e-9) & (steps <= HORIZON + 1e-9)
    rows = read_manifest(args.clips)
    scores = np.array([scorer.probabilities(sf.read(row.path, dtype="float32")[0])[post].max() for row in rows])
    labels = np.array([row.label for row in rows])
    if not (labels == "interrupt").any():
        raise SystemExit("need at least one labelled interrupt")

    threshold = threshold_for_recall(scores[labels == "interrupt"], args.target_recall)
    for name, value in (("current", bundle.threshold), ("calibrated", threshold)):
        stops = scores >= value
        summary = ", ".join(f"{label} {stops[labels == label].mean():.2f}" for label in ("interrupt", "backchannel", "ignore") if (labels == label).any())
        print(f"{name:10s} threshold {value:.3f}: stop rate {summary}")
    if args.write:
        replace(bundle, threshold=threshold).save({"calibration": f"{args.clips}, {args.target_recall:.0%} recall"})
        print(f"saved threshold {threshold:.3f} to {bundle.directory}")


if __name__ == "__main__":
    main()
