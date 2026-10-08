"""Calibrate a trained model's threshold on dev, then score test with the baseline benchmark."""

from __future__ import annotations

import argparse
import json

import numpy as np
import soundfile as sf

from talkover.interruption.benchmark import console_progress, load_items, run_policy, stop_rates, write_report
from talkover.interruption.model.checkpoint import load_run
from talkover.interruption.model.data import read_manifest
from talkover.interruption.model.streaming import (
    ModelStopPolicy,
    OnnxStreamingScorer,
    StreamingScorer,
    threshold_for_recall,
)
from talkover.interruption.paths import clips_dir, data_dir

TARGET_RECALL = 0.85
HEADLINE_HORIZON = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-clips", default="ami_aug", help="clip set to score (threshold always from ami_aug dev)")
    parser.add_argument("--onnx", help="score with this exported model file inside the run dir, e.g. model.int8.onnx")
    args = parser.parse_args()

    run_dir = data_dir() / "models" / "interruption" / args.run
    model, config, _ = load_run(run_dir, str(data_dir() / "models" / "hf"), "cpu" if args.onnx else args.device)
    scorer = OnnxStreamingScorer(str(run_dir / args.onnx), config) if args.onnx else StreamingScorer(model, args.device, config)
    steps = np.asarray(scorer.steps)
    post = (steps > 1e-9) & (steps <= HEADLINE_HORIZON + 1e-9)

    progress = console_progress("dev calibration")
    dev_rows = read_manifest(clips_dir("ami_aug") / "dev.jsonl")
    interrupt_max = []
    for index, row in enumerate(dev_rows, 1):
        if row.label == "interrupt":
            clip, _ = sf.read(row.path, dtype="float32")
            interrupt_max.append(scorer.probabilities(clip)[post].max())
        progress(index, len(dev_rows))
    threshold = threshold_for_recall(np.array(interrupt_max), TARGET_RECALL)
    print(f"threshold {threshold:.3f} gives {TARGET_RECALL:.0%} dev interrupt recall at {HEADLINE_HORIZON * 1000:.0f} ms", flush=True)

    suffix = f"@{args.onnx.removesuffix('.onnx')}" if args.onnx else ""
    policy = ModelStopPolicy(f"model:{args.run}{suffix}", scorer, threshold)
    items = load_items(clips_dir(args.eval_clips) / f"{args.split}.jsonl")
    decisions = run_policy(policy, items, policy.horizons, console_progress(policy.name))
    results = stop_rates(policy.name, items, decisions, policy.horizons)
    report_dir = args.split if args.eval_clips == "ami_aug" else f"{args.eval_clips}_{args.split}"
    write_report(results, data_dir() / "results" / "baselines" / report_dir / f"{policy.name}.json")
    (run_dir / f"calibration{suffix}.json").write_text(json.dumps({"threshold": threshold, "target_recall": TARGET_RECALL}))
    table = {(r.horizon, r.slice): r.stop_rate for r in results}
    for key in ("interrupt", "backchannel", "ignore"):
        rate = table[(HEADLINE_HORIZON, key)]
        print(f"{key:11s} stop rate @500 ms {rate.value:.3f} [{rate.low:.2f}, {rate.high:.2f}]")
    rate = table[(0.0, "not_interrupt")]
    print(f"stops before onset {rate.value:.3f} [{rate.low:.2f}, {rate.high:.2f}]")


if __name__ == "__main__":
    main()
