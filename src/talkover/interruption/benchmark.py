"""Score stop policies on rendered eval clips: stop rates per group, horizon and noise band."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf

from talkover.evaluation.bootstrap import RateEstimate, cluster_bootstrap_rate
from talkover.interruption.policies import StopPolicy


HORIZONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)
SNR_BANDS = ((15.0, 20.0), (10.0, 15.0), (5.0, 10.0), (0.0, 5.0))


@dataclass(frozen=True)
class EvalItem:
    clip: Path
    meeting_id: str
    label: str
    group: str
    noise_band: str
    onset_seconds: float


def noise_band(snr_db: float | None) -> str:
    if snr_db is None:
        return "clean"
    low, high = next(band for band in SNR_BANDS if band[0] <= snr_db <= band[1])
    return f"{low:g}-{high:g} dB"


def load_items(manifest: Path, onset_seconds: float = 1.0) -> list[EvalItem]:
    items = []
    for line in manifest.open(encoding="utf-8"):
        row = json.loads(line)
        augmentation = row.get("augmentation", {})
        sound = augmentation.get("onset_sound")
        group = row["label"] if row["label"] != "ignore" else f"ignore:{sound or row['act']}"
        items.append(
            EvalItem(
                clip=manifest.parent / row["clip"],
                meeting_id=row["meeting_id"],
                label=row["label"],
                group=group,
                noise_band=noise_band(augmentation.get("snr_db")),
                onset_seconds=onset_seconds,
            )
        )
    return items


Progress = Callable[[int, int], None]


def run_policy(
    policy: StopPolicy,
    items: Sequence[EvalItem],
    horizons: Sequence[float],
    progress: Progress | None = None,
) -> list[tuple[bool, ...]]:
    rows = []
    for index, item in enumerate(items, 1):
        clip, rate = sf.read(item.clip, dtype="float32")
        rows.append(policy.decisions(clip, rate, item.onset_seconds, horizons))
        if progress:
            progress(index, len(items))
    return rows


def console_progress(name: str, every_seconds: float = 300.0) -> Progress:
    start = last = time.monotonic()

    def report(done: int, total: int) -> None:
        nonlocal last
        now = time.monotonic()
        if now - last < every_seconds and done != total:
            return
        last = now
        elapsed = now - start
        eta = elapsed / done * (total - done)
        print(f"  {name}: {done}/{total} clips  {elapsed:5.0f}s elapsed  ~{eta:4.0f}s left", flush=True)

    return report


@dataclass(frozen=True)
class SliceResult:
    policy: str
    horizon: float
    slice: str
    stop_rate: RateEstimate


def stop_rates(
    policy: str,
    items: Sequence[EvalItem],
    decisions: Sequence[tuple[bool, ...]],
    horizons: Sequence[float],
) -> list[SliceResult]:
    """Stop rate per slice; for interrupt it is recall, for the rest a false-stop rate."""

    slices: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        for key in {item.label, item.group, f"{item.label}@{item.noise_band}"}:
            slices.setdefault(key, []).append(index)
    slices["not_interrupt"] = [i for i, item in enumerate(items) if item.label != "interrupt"]
    results = []
    for column, horizon in enumerate(horizons):
        for key, indices in sorted(slices.items()):
            estimate = cluster_bootstrap_rate(
                [decisions[i][column] for i in indices], [items[i].meeting_id for i in indices]
            )
            results.append(SliceResult(policy, horizon, key, estimate))
    return results


def write_report(results: Iterable[SliceResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=1))
