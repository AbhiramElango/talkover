"""Render a baseline results JSON as markdown tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from talkover.interruption.paths import data_dir

CLASSES = ("interrupt", "backchannel", "ignore")
BANDS = ("clean", "15-20 dB", "10-15 dB", "5-10 dB", "0-5 dB")


def _cell(rate: dict) -> str:
    return f"{rate['value']:.2f} [{rate['low']:.2f}–{rate['high']:.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=data_dir() / "results" / "baselines" / "test")
    parser.add_argument("--horizon", type=float, default=0.5)
    args = parser.parse_args()

    table: dict[str, dict[tuple[float, str], dict]] = defaultdict(dict)
    order = ("energy_200ms", "silero_200ms", "silero_gated_200ms", "silero_gated_500ms", "silero_gated+asr_2words")
    files = sorted(args.results.glob("*.json"), key=lambda p: order.index(p.stem) if p.stem in order else len(order))
    for path in files:
        for row in json.loads(path.read_text()):
            table[row["policy"]][(row["horizon"], row["slice"])] = row["stop_rate"]
    policies = list(table)
    counts = {name: table[policies[0]][(0.0, name)]["total"] for name in CLASSES}

    lines = [f"Clips: " + ", ".join(f"{name} {count}" for name, count in counts.items()), ""]
    lines += ["### Stop rates at the headline horizon", "",
              f"Decision {args.horizon * 1000:.0f} ms after onset. 95% CI by meeting.", "",
              "| Policy | Stops before onset | Interrupt (want high) | Backchannel (want low) | Ignore (want low) |",
              "|---|---|---|---|---|"]
    for policy in policies:
        rates = table[policy]
        horizon = args.horizon if (args.horizon, "interrupt") in rates else max(h for h, _ in rates)
        cells = [_cell(rates[(0.0, "not_interrupt")])] + [_cell(rates[(horizon, c)]) for c in CLASSES]
        suffix = "" if horizon == args.horizon else f" (@{horizon * 1000:.0f} ms)"
        lines.append(f"| {policy}{suffix} | " + " | ".join(cells) + " |")

    first = table[policies[0]]
    kinds = sorted((k for h, k in first if h == 0.0 and k.startswith("ignore:")), key=lambda k: -first[(0.0, k)]["total"])[:5]
    lines += ["", "### Ignore false stops by sound", "", "| Policy | " + " | ".join(k.split(":", 1)[1] for k in kinds) + " |",
              "|---" * (len(kinds) + 1) + "|"]
    for policy in policies:
        rates = table[policy]
        horizon = args.horizon if (args.horizon, "interrupt") in rates else max(h for h, _ in rates)
        lines.append(f"| {policy} | " + " | ".join(f"{rates[(horizon, k)]['value']:.2f}" if (horizon, k) in rates else "–" for k in kinds) + " |")

    for name in CLASSES:
        lines += ["", f"### {name} stop rate by noise band", "", "| Policy | " + " | ".join(BANDS) + " |",
                  "|---" * (len(BANDS) + 1) + "|"]
        for policy in policies:
            rates = table[policy]
            horizon = args.horizon if (args.horizon, "interrupt") in rates else max(h for h, _ in rates)
            lines.append(f"| {policy} | " + " | ".join(
                f"{rates[(horizon, f'{name}@{b}')]['value']:.2f}" if (horizon, f"{name}@{b}") in rates else "–" for b in BANDS
            ) + " |")

    lines += ["", "### Interrupt recall by horizon", "", "| Policy | " + " | ".join(f"{h * 1000:.0f} ms" for h in (0.1, 0.2, 0.3, 0.4, 0.5, 1.0)) + " |",
              "|---" * 7 + "|"]
    for policy in policies:
        rates = table[policy]
        lines.append(f"| {policy} | " + " | ".join(
            f"{rates[(h, 'interrupt')]['value']:.2f}" if (h, "interrupt") in rates else "–" for h in (0.1, 0.2, 0.3, 0.4, 0.5, 1.0)
        ) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
