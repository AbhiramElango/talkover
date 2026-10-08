"""Copy a trained run's ONNX export into a self-contained bundle the runtime can load."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from talkover.interruption.model.windows import WindowConfig
from talkover.interruption.paths import data_dir
from talkover.interruption.runtime import ModelBundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="whisper_tiny_tts")
    parser.add_argument("--onnx", default="model.int8.onnx")
    parser.add_argument("--output", type=Path, default=data_dir() / "release" / "talkover")
    args = parser.parse_args()

    run_dir = data_dir() / "models" / "interruption" / args.run
    meta = json.loads((run_dir / "meta.json").read_text())
    calibration = json.loads((run_dir / f"calibration@{args.onnx.removesuffix('.onnx')}.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / args.onnx, args.output / "model.onnx")
    window = {key: meta["window"][key] for key in ("window_seconds", "sample_rate_hz", "channels", "agent_envelope")}
    bundle = ModelBundle(args.output, "model.onnx", WindowConfig(**window), calibration["threshold"])
    bundle.save({
        "run": args.run,
        "source": args.onnx,
        "classes": ["ignore", "backchannel", "interrupt"],
        "calibration": f"AMI dev, {calibration['target_recall']:.0%} interrupt recall at 500 ms",
    })
    print(f"bundle written to {args.output} (threshold {bundle.threshold:.3f})")


if __name__ == "__main__":
    main()
