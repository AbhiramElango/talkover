"""Export a trained run to ONNX (fp32 and INT8), check parity against PyTorch, and time CPU latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import soundfile as sf
import torch

from talkover.interruption.model.checkpoint import load_run
from talkover.interruption.model.data import INTERRUPT, cut_window, model_inputs, read_manifest
from talkover.interruption.paths import clips_dir, data_dir


class _Probabilities(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(windows), dim=-1)


def _session(path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _latency(path: Path, threads: int, channels: int, runs: int = 200) -> dict[str, float]:
    session = _session(path, threads)
    window = (np.random.default_rng(0).standard_normal((1, channels, 16_000)) * 0.05).astype(np.float32)
    for _ in range(10):
        session.run(None, {"windows": window})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {"windows": window})
        times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--parity-clips", type=int, default=300)
    args = parser.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    run_dir = data_dir() / "models" / "interruption" / args.run
    model, config, _ = load_run(run_dir, str(data_dir() / "models" / "hf"), "cpu")
    wrapped = _Probabilities(model).eval()
    fp32, int8 = run_dir / "model.onnx", run_dir / "model.int8.onnx"
    example = torch.zeros(1, len(config.channels), round(config.window_seconds * config.sample_rate_hz))
    torch.onnx.export(
        wrapped, (example,), str(fp32), input_names=["windows"], output_names=["probabilities"],
        dynamic_axes={"windows": {0: "batch"}, "probabilities": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    front_end = [node.name for node in onnx.load(str(fp32)).graph.node if "/mel/" in node.name]
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, nodes_to_exclude=front_end)

    rows = read_manifest(clips_dir("ami_aug") / "dev.jsonl")
    picks = np.random.default_rng(0).choice(len(rows), min(args.parity_clips, len(rows)), replace=False)
    windows = np.ascontiguousarray(
        model_inputs(np.stack([cut_window(sf.read(rows[i].path, dtype="float32")[0], 0.3, config) for i in picks]), config)
    )
    with torch.no_grad():
        reference = wrapped(torch.from_numpy(windows)).numpy()[:, INTERRUPT]
    report = {"run": args.run, "params_million": sum(p.numel() for p in model.parameters()) / 1e6}
    for name, path in (("fp32", fp32), ("int8", int8)):
        session = _session(path, 4)
        probs = np.concatenate([session.run(None, {"windows": windows[i : i + 32]})[0] for i in range(0, len(windows), 32)])[:, INTERRUPT]
        report[name] = {
            "size_mb": path.stat().st_size / 1e6,
            "max_abs_diff_vs_torch": float(np.abs(probs - reference).max()),
            "mean_abs_diff_vs_torch": float(np.abs(probs - reference).mean()),
            "latency_1_thread": _latency(path, 1, len(config.channels)),
            "latency_4_threads": _latency(path, 4, len(config.channels)),
        }
        print(name, json.dumps(report[name]), flush=True)
    (run_dir / "export.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
