"""Score baseline stop policies on the rendered eval set and print a summary table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from talkover.interruption.benchmark import (
    HORIZONS,
    console_progress,
    load_items,
    run_policy,
    stop_rates,
    write_report,
)
from talkover.interruption.paths import clips_dir, data_dir
from talkover.interruption.policies import (
    EnergyScorer,
    FasterWhisperTranscriber,
    LevelGatedScorer,
    MlxWhisperTranscriber,
    SileroScorer,
    SustainedActivityPolicy,
    TranscriptRulePolicy,
)


TRANSCRIBERS = {"mlx": MlxWhisperTranscriber, "faster-whisper": FasterWhisperTranscriber}


def build_policies(reference_dbfs: float, asr_backend: str | None) -> list:
    silero = SileroScorer()
    gated = LevelGatedScorer(silero, reference_dbfs - 15)
    policies = [
        SustainedActivityPolicy("energy_200ms", EnergyScorer(), reference_dbfs - 10, 0.2),
        SustainedActivityPolicy("silero_200ms", silero, 0.5, 0.2),
        SustainedActivityPolicy("silero_gated_200ms", gated, 0.5, 0.2),
        SustainedActivityPolicy("silero_gated_500ms", gated, 0.5, 0.5),
    ]
    if asr_backend:
        transcriber = TRANSCRIBERS[asr_backend]()
        policies.append(TranscriptRulePolicy("silero_gated+asr_2words", policies[2], transcriber))
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=clips_dir("ami_aug"))
    parser.add_argument("--tag", default="", help="results subfolder prefix, e.g. agent_calls")
    parser.add_argument("--split", default="test")
    parser.add_argument("--asr", choices=sorted(TRANSCRIBERS), help="include the transcript-rule baseline")
    parser.add_argument("--only", nargs="*", help="run only these policy names")
    parser.add_argument("--output", type=Path, default=data_dir() / "results" / "baselines")
    args = parser.parse_args()

    reference = json.loads((clips_dir("ami") / "stats.json").read_text())["reference_dbfs"]
    items = load_items(args.clips / f"{args.split}.jsonl")
    results = []
    for policy in build_policies(reference, args.asr):
        if args.only and policy.name not in args.only:
            continue
        horizons = policy.horizons or HORIZONS
        decisions = run_policy(policy, items, horizons, console_progress(policy.name))
        policy_results = stop_rates(policy.name, items, decisions, horizons)
        results.extend(policy_results)
        write_report(policy_results, args.output / f"{args.tag + '_' if args.tag else ''}{args.split}" / f"{policy.name}.json")
        print(f"\n{policy.name}  (stop rate, 95% CI by meeting)")
        print(f"{'horizon':>8s} {'interrupt':>18s} {'backchannel':>18s} {'ignore':>18s}")
        table = {(r.horizon, r.slice): r.stop_rate for r in policy_results}
        for h in horizons:
            cells = [table[(h, key)] for key in ("interrupt", "backchannel", "ignore")]
            print(f"{h:7.1f}s " + " ".join(f"{c.value:6.3f} [{c.low:.2f},{c.high:.2f}]" for c in cells), flush=True)
    print(f"\nper-policy results in {args.output / args.split}")


if __name__ == "__main__":
    main()
