"""Check rendered eval clips: BEATs hears the inserted sound, DNSMOS tracks requested SNR.

Needs the `verify` extra and the BEATs code on the path, e.g.
PYTHONPATH=data/vendor/unilm/beats python scripts/interruption/verify_eval_set.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from talkover.interruption.paths import clips_dir, data_dir
from talkover.verification.beats import CHECKPOINT_FILENAME, CHECKPOINT_REPO, CHECKPOINT_REVISION, BeatsAudioSetTagger
from talkover.verification.dnsmos import DnsmosScorer

SNR_BANDS = ((0, 5), (5, 10), (10, 15), (15, 20))


def audioset_labels(checkpoint: Path) -> list[str]:
    """Display names in the checkpoint's own output order (its ``label_dict``, not CSV order)."""

    import torch

    names = {
        row["mid"]: row["display_name"]
        for row in csv.DictReader((data_dir() / "raw/audioset/class_labels_indices.csv").open())
    }
    label_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)["label_dict"]
    return [names[label_dict[index]] for index in range(len(label_dict))]


def _group(row: dict) -> str:
    sound = row["augmentation"].get("onset_sound")
    return f"onset_{sound}" if sound else row["act"] if row["label"] == "ignore" else row["label"]


def _band(snr: float | None) -> str:
    if snr is None:
        return "no noise"
    low, high = next(band for band in SNR_BANDS if band[0] <= snr <= band[1])
    return f"{low}-{high} dB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=clips_dir("ami_aug"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    checkpoint = Path(hf_hub_download(
        CHECKPOINT_REPO, CHECKPOINT_FILENAME, revision=CHECKPOINT_REVISION, cache_dir=str(data_dir() / "models" / "hf")
    ))
    labels = audioset_labels(checkpoint)
    speech, music = labels.index("Speech"), labels.index("Music")
    tagger = BeatsAudioSetTagger(checkpoint, device=args.device)
    dnsmos = DnsmosScorer(data_dir() / "models/sig_bak_ovr.onnx")

    rows = [json.loads(line) for line in (args.clips / f"{args.split}.jsonl").open(encoding="utf-8")]
    beats: dict[str, list[tuple[float, float, float, float, str]]] = defaultdict(list)
    bak: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clip, rate = sf.read(args.clips / row["clip"], dtype="float32")
        caller = clip[:, 0]
        onset = rate
        pre, post = tagger.tag_waveforms((caller[:onset], caller[onset:]), rate)
        top = labels[int(np.argmax(post))]
        beats[_group(row)].append((pre[speech], post[speech], pre[music], post[music], top))

        bak[_band(row["augmentation"].get("snr_db"))].append(dnsmos.score(caller, rate).background)

    report = {"split": args.split, "beats": {}, "dnsmos_bak_by_snr": {}}
    print(f"{'group':16s} {'n':>5s}  P(speech) pre->post  P(music) pre->post  top post-onset tags")
    for group, values in sorted(beats.items()):
        array = np.array([value[:4] for value in values], dtype=float)
        means = array.mean(axis=0)
        tops = defaultdict(int)
        for value in values:
            tops[value[4]] += 1
        common = sorted(tops.items(), key=lambda item: -item[1])[:3]
        report["beats"][group] = {"n": len(values), "speech": [means[0], means[1]], "music": [means[2], means[3]], "top": common}
        print(f"{group:16s} {len(values):5d}  {means[0]:.2f} -> {means[1]:.2f}         {means[2]:.2f} -> {means[3]:.2f}        {common}")

    print(f"\n{'requested SNR':14s} {'n':>5s}  DNSMOS BAK (higher = cleaner)")
    for band in ["no noise", *(f"{low}-{high} dB" for low, high in reversed(SNR_BANDS))]:
        values = bak.get(band, [])
        if values:
            report["dnsmos_bak_by_snr"][band] = {"n": len(values), "median": float(np.median(values))}
            print(f"{band:14s} {len(values):5d}  {np.median(values):.2f}")
    (args.clips / f"verification_{args.split}.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
