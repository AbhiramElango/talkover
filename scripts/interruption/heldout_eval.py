"""Pre-registered agent-call test: thresholds chosen on s1 (dev), applied once to held-out sessions.

Operating points per model, both set on dev against a reference VAD policy:
  match_recall: highest threshold whose dev interrupt recall >= the VAD's dev recall
  match_false_stops: lowest threshold whose dev non-interrupt stop rate <= the VAD's
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from talkover.evaluation.bootstrap import cluster_bootstrap_rate
from talkover.interruption.benchmark import load_items, run_policy
from talkover.interruption.model.checkpoint import load_run
from talkover.interruption.model.streaming import StreamingScorer
from talkover.interruption.paths import clips_dir, data_dir
from talkover.interruption.policies import LevelGatedScorer, SileroScorer, SustainedActivityPolicy

HORIZON = 0.5
CLASSES = ("interrupt", "backchannel", "ignore")


def model_scores(scorer: StreamingScorer, items) -> np.ndarray:
    steps = np.asarray(scorer.steps)
    post = (steps > 1e-9) & (steps <= HORIZON + 1e-9)
    return np.array([scorer.probabilities(sf.read(item.clip, dtype="float32")[0])[post].max() for item in items])


def rates(stops: np.ndarray, items) -> dict[str, dict]:
    out = {}
    for name in CLASSES:
        index = [i for i, item in enumerate(items) if item.label == name]
        estimate = cluster_bootstrap_rate([bool(stops[i]) for i in index], [items[i].meeting_id for i in index])
        out[name] = {"value": estimate.value, "low": estimate.low, "high": estimate.high, "n": estimate.total}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["whisper_tiny_student", "whisper_tiny_tts"])
    parser.add_argument("--dev", default="agent_calls")
    parser.add_argument("--test", default="agent_calls_heldout")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    dev, test = load_items(clips_dir(args.dev) / "test.jsonl"), load_items(clips_dir(args.test) / "test.jsonl")
    reference = json.loads((clips_dir("ami") / "stats.json").read_text())["reference_dbfs"]
    silero = SileroScorer()
    vads = {
        "silero_200ms": SustainedActivityPolicy("silero_200ms", silero, 0.5, 0.2),
        "silero_gated_200ms": SustainedActivityPolicy("silero_gated_200ms", LevelGatedScorer(silero, reference - 15), 0.5, 0.2),
    }
    report = {"dev": args.dev, "test": args.test, "horizon": HORIZON, "results": {}}
    vad_dev = {}
    for name, policy in vads.items():
        dev_stops = np.array([d[0] for d in run_policy(policy, dev, (HORIZON,))])
        test_stops = np.array([d[0] for d in run_policy(policy, test, (HORIZON,))])
        vad_dev[name] = dev_stops
        report["results"][name] = {"test": rates(test_stops, test)}

    labels_dev = np.array([item.label for item in dev])
    reference_stops = vad_dev["silero_200ms"]
    vad_recall = reference_stops[labels_dev == "interrupt"].mean()
    vad_false = reference_stops[labels_dev != "interrupt"].mean()
    for run in args.models:
        model, config, _ = load_run(data_dir() / "models" / "interruption" / run, str(data_dir() / "models" / "hf"), args.device)
        scorer = StreamingScorer(model, args.device, config)
        dev_scores, test_scores = model_scores(scorer, dev), model_scores(scorer, test)
        candidates = np.unique(np.concatenate([dev_scores, [0.0, 1.0]]))
        recall_ok = [t for t in candidates if (dev_scores[labels_dev == "interrupt"] >= t).mean() >= vad_recall]
        false_ok = [t for t in candidates if (dev_scores[labels_dev != "interrupt"] >= t).mean() <= vad_false]
        points = {"match_recall": max(recall_ok), "match_false_stops": min(false_ok)}
        report["results"][run] = {
            point: {"threshold": float(t), "test": rates(test_scores >= t, test)} for point, t in points.items()
        }
    out = data_dir() / "results" / "heldout_agent_calls.json"
    out.write_text(json.dumps(report, indent=1, default=float))

    print(f"held-out {args.test}: " + ", ".join(f"{c} {sum(i.label == c for i in test)}" for c in CLASSES))
    print(f"{'policy':44s} " + "  ".join(f"{c:>20s}" for c in CLASSES))
    for name, result in report["results"].items():
        for point, body in ([("", result)] if "test" in result else result.items()):
            cells = [f"{body['test'][c]['value']:.2f} [{body['test'][c]['low']:.2f}-{body['test'][c]['high']:.2f}]" for c in CLASSES]
            label = f"{name} {point} (t={body['threshold']:.2f})" if point else name
            print(f"{label:44s} " + "  ".join(f"{cell:>20s}" for cell in cells))


if __name__ == "__main__":
    main()
