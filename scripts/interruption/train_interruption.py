"""Train the interruption classifier (optionally distilled from a teacher run); keep the best dev checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from talkover.interruption.augment import AugmentContext, MusanBank, default_pipelines, robust_pipelines
from talkover.interruption.corpora.agent_calls import WebRtcEchoCanceller
from talkover.interruption.model.data import (
    INTERRUPT,
    FixedWindows,
    model_inputs,
    TrainWindows,
    WindowConfig,
    class_balanced_weights,
    read_manifest,
)
from talkover.interruption.model.checkpoint import load_run, trainable_state
from talkover.interruption.model.network import InterruptionClassifier, build_encoder
from talkover.interruption.model.streaming import threshold_for_recall
from talkover.interruption.paths import clips_dir, data_dir

DEV_DECISIONS = (-0.5, 0.3, 0.5)
TARGET_RECALL = 0.85


@torch.no_grad()
def evaluate(model, loader, device: str, clip_labels: list[str], config: WindowConfig) -> dict[str, float]:
    model.eval()
    scores: dict[tuple[int, float], float] = {}
    for windows, _, clip_index, decision in loader:
        probs = torch.softmax(model(model_inputs(windows, config).to(device)), dim=-1)[:, INTERRUPT].cpu().numpy()
        for c, d, p in zip(clip_index.tolist(), decision.tolist(), probs):
            scores[(c, round(d, 2))] = float(p)
    post = np.array([max(scores[(c, 0.3)], scores[(c, 0.5)]) for c in range(len(clip_labels))])
    pre = np.array([scores[(c, -0.5)] for c in range(len(clip_labels))])
    labels = np.array(clip_labels)
    threshold = threshold_for_recall(post[labels == "interrupt"], TARGET_RECALL)
    stops = post >= threshold
    return {
        "threshold": threshold,
        "recall": float(stops[labels == "interrupt"].mean()),
        "fsr_backchannel": float(stops[labels == "backchannel"].mean()),
        "fsr_ignore": float(stops[labels == "ignore"].mean()),
        "fsr_non_interrupt": float(stops[labels != "interrupt"].mean()),
        "pre_onset_stop": float((pre >= threshold).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=0, help="use only N train and dev clips (smoke test)")
    parser.add_argument("--mute-agent", action="store_true", help="leakage ablation: zero agent audio after onset")
    parser.add_argument("--channels", choices=("both", "caller"), default="both")
    parser.add_argument("--augment", choices=("default", "robust"), default="default")
    parser.add_argument("--agent-envelope", action="store_true", help="agent channel as envelope-shaped noise")
    parser.add_argument("--extra-train", nargs="*", default=[], help="extra clip sets to add to training, e.g. ami_tts")
    parser.add_argument("--encoder", default="wavlm-base-plus")
    parser.add_argument("--encoder-lr", type=float, default=0.0, help="0 keeps the encoder frozen")
    parser.add_argument("--teacher", help="run name to distill from")
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = data_dir() / "models" / "interruption" / args.run
    out.mkdir(parents=True, exist_ok=True)
    channels = (0, 1) if args.channels == "both" else (0,)
    config = replace(WindowConfig(), mute_agent_after_onset=args.mute_agent, channels=channels, agent_envelope=args.agent_envelope)
    reference = json.loads((clips_dir("ami") / "stats.json").read_text())["reference_dbfs"]
    context = AugmentContext(config.sample_rate_hz, round(config.onset_seconds * config.sample_rate_hz), reference)

    train_rows = read_manifest(clips_dir("ami") / "train.jsonl")
    for extra in args.extra_train:
        train_rows += read_manifest(clips_dir(extra) / "train.jsonl")
    dev_rows = read_manifest(clips_dir("ami_aug") / "dev.jsonl")
    if args.limit:
        rng = np.random.default_rng(args.seed)
        train_rows = [train_rows[i] for i in rng.choice(len(train_rows), min(args.limit, len(train_rows)), replace=False)]
        dev_rows = [dev_rows[i] for i in rng.choice(len(dev_rows), min(args.limit, len(dev_rows)), replace=False)]
    bank = MusanBank(data_dir() / "raw" / "musan" / "musan", "train")
    if args.augment == "robust":
        speech, silent_ignore = robust_pipelines(bank, WebRtcEchoCanceller())
    else:
        speech, silent_ignore = default_pipelines(bank)
    train_set = TrainWindows(train_rows, speech, silent_ignore, context, config, args.seed)
    sampler = WeightedRandomSampler(class_balanced_weights(train_rows), len(train_rows), replacement=True)
    train_loader = DataLoader(train_set, args.batch, sampler=sampler, num_workers=args.workers)
    dev_loader = DataLoader(FixedWindows(dev_rows, DEV_DECISIONS, config), args.batch, num_workers=args.workers)

    cache = str(data_dir() / "models" / "hf")
    model = InterruptionClassifier(build_encoder(args.encoder, cache), streams=len(channels)).to(args.device)
    if not args.encoder_lr:
        model.encoder.requires_grad_(False)
    groups = [{"params": model.head_parameters(), "lr": args.lr}]
    if args.encoder_lr:
        groups.append({"params": model.encoder_parameters(), "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, [group["lr"] for group in groups], total_steps=args.epochs * len(train_loader)
    )
    teacher, teacher_config = None, None
    if args.teacher:
        teacher, teacher_config, _ = load_run(data_dir() / "models" / "interruption" / args.teacher, cache, args.device)
        teacher.requires_grad_(False)

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
        hard = torch.nn.functional.cross_entropy(logits, labels)
        if teacher is None:
            return hard
        with torch.no_grad():
            soft_targets = torch.softmax(teacher(model_inputs(windows, teacher_config)) / args.temperature, dim=-1)
        soft = torch.nn.functional.kl_div(
            torch.log_softmax(logits / args.temperature, dim=-1), soft_targets, reduction="batchmean"
        ) * args.temperature**2
        return (1 - args.distill_alpha) * hard + args.distill_alpha * soft

    meta = {"args": vars(args), "window": asdict(config), "encoder": args.encoder,
            "params_million": sum(p.numel() for p in model.parameters()) / 1e6,
            "train_clips": len(train_rows), "dev_clips": len(dev_rows), "history": []}
    best = float("inf")
    start = last_report = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        train_set.epoch = epoch
        total, count = 0.0, 0
        for step, (windows, labels) in enumerate(train_loader, 1):
            windows = windows.to(args.device)
            logits = model(model_inputs(windows, config))
            loss = loss_fn(logits, labels.to(args.device), windows)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total, count = total + loss.item() * len(labels), count + len(labels)
            if time.monotonic() - last_report >= 300:
                last_report = time.monotonic()
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} loss {total / count:.3f} "
                      f"{(last_report - start) / 60:.1f} min", flush=True)
        metrics = evaluate(model, dev_loader, args.device, [row.label for row in dev_rows], config)
        metrics.update(epoch=epoch + 1, train_loss=total / count, minutes=(time.monotonic() - start) / 60)
        meta["history"].append(metrics)
        improved = metrics["fsr_non_interrupt"] < best
        if improved:
            best = metrics["fsr_non_interrupt"]
            torch.save(trainable_state(model), out / "head.pt")
        print(f"epoch {epoch + 1:2d} loss {metrics['train_loss']:.3f} | dev @85% recall: "
              f"false stops {metrics['fsr_non_interrupt']:.3f} (bc {metrics['fsr_backchannel']:.3f}, "
              f"ign {metrics['fsr_ignore']:.3f}), pre-onset {metrics['pre_onset_stop']:.3f} "
              f"| {metrics['minutes']:.1f} min{' *' if improved else ''}", flush=True)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
