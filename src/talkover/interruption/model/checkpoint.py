"""Save and load trained runs: ``head.pt`` (trainable weights) plus ``meta.json``."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from talkover.interruption.model.data import WindowConfig
from talkover.interruption.model.network import InterruptionClassifier, build_encoder


def encoder_name(meta: dict) -> str:
    encoder = meta["encoder"]
    return "wavlm-base-plus" if isinstance(encoder, list) else encoder


def trainable_state(model: InterruptionClassifier) -> dict[str, torch.Tensor]:
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    return {name: value for name, value in model.state_dict().items() if name not in frozen and "embed_positions" not in name}


def load_run(run_dir: Path, cache_dir: str, device: str) -> tuple[InterruptionClassifier, WindowConfig, dict]:
    meta = json.loads((run_dir / "meta.json").read_text())
    config = replace(WindowConfig(), **meta["window"])
    model = InterruptionClassifier(build_encoder(encoder_name(meta), cache_dir), streams=len(config.channels))
    result = model.load_state_dict(torch.load(run_dir / "head.pt", map_location="cpu"), strict=False)
    trained = {name for name, p in model.named_parameters() if p.requires_grad}
    missing = [key for key in result.missing_keys if key in trained]
    if result.unexpected_keys or missing:
        raise ValueError(f"checkpoint mismatch: missing {missing}, unexpected {result.unexpected_keys}")
    return model.to(device).eval(), config, meta
