"""Train and evaluate the interruption classifier on a Modal GPU.

    python infra/modal/upload_musan.py                    # once: MUSAN train split -> volume
    modal run infra/modal/train_app.py::prepare           # once: unpack clips + MUSAN
    modal run --detach infra/modal/train_app.py::train --args "--run NAME --epochs 8"
    modal run --detach infra/modal/train_app.py::evaluate --args "--run NAME"
    modal run infra/modal/train_app.py::export --args "--run NAME"   # x86 CPU latency
    modal volume get talkover-data models/interruption/NAME data/models/interruption/
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

APP_NAME = "talkover-train"
VOLUME_NAME = "talkover-data"
DATA = "/data"
MUSAN_PARTS = "upload/musan_train"


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path("/root")


PROJECT_ROOT = _project_root()

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "curl")
    .pip_install(
        "torch==2.14.1",
        "transformers==4.57.6",
        "huggingface-hub==0.36.2",
        "safetensors==0.8.0",
        "numpy==2.5.3",
        "soundfile==0.14.0",
        "scipy==1.18.1",
        "onnx==1.23.2",
        "onnxruntime==1.30.0",
        "livekit==1.1.20",
    )
    .env({"TALKOVER_DATA_DIR": DATA, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"})
    .add_local_dir(PROJECT_ROOT / "src" / "talkover", "/root/talkover", ignore=["**/__pycache__"])
    .add_local_dir(PROJECT_ROOT / "scripts", "/root/scripts", ignore=["**/__pycache__"])
)
app = modal.App(APP_NAME, image=image)


STAGE = Path("/tmp/stage")
STAGED = ("clips/ami", "clips/ami_tts", "clips/ami_aug", "clips/turnbench", "clips/agent_calls", "raw/musan")
PERSISTENT = ("models", "results")


def _stage() -> Path:
    """Copy training inputs from the network volume to local disk; outputs stay on the volume."""

    files = [path for root in STAGED for path in (Path(DATA) / root).rglob("*") if path.is_file()]
    progress = _Progress("stage to local disk")

    def copy(path: Path) -> None:
        target = STAGE / path.relative_to(DATA)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    with ThreadPoolExecutor(32) as pool:
        for done, _ in enumerate(pool.map(copy, files), 1):
            progress.update(done, len(files), "files")
    for name in PERSISTENT:
        (Path(DATA) / name).mkdir(exist_ok=True)
        (STAGE / name).symlink_to(Path(DATA) / name)
    print(f"staged {len(files):,} files in {(time.monotonic() - progress.start) / 60:.1f} min", flush=True)
    return STAGE


def _run_script(name: str, args: str, device: str | None = "cuda") -> None:
    env = {**os.environ, "TALKOVER_DATA_DIR": str(_stage())}
    command = ["python", f"/root/scripts/interruption/{name}", *shlex.split(args), *(["--device", device] if device else [])]
    print("$", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, env=env)
    finally:
        volume.commit()


class _Progress:
    def __init__(self, label: str, every_seconds: float = 30.0) -> None:
        self.label, self.every = label, every_seconds
        self.start = self.last = time.monotonic()

    def update(self, done: float, total: float | None, unit: str) -> None:
        now = time.monotonic()
        if now - self.last < self.every:
            return
        self.last = now
        elapsed = now - self.start
        rate = done / elapsed if elapsed else 0.0
        if total:
            eta = (total - done) / rate if rate else float("inf")
            print(f"{self.label}: {done:,.0f}/{total:,.0f} {unit} ({done / total:.0%}), {elapsed / 60:.1f} min, ~{eta / 60:.1f} min left", flush=True)
        else:
            print(f"{self.label}: {done:,.0f} {unit}, {elapsed / 60:.1f} min", flush=True)


def _extract(archive: Path, destination: Path) -> None:
    progress, count = _Progress(f"extract {archive.name}"), 0
    with tarfile.open(archive) as tar:
        for member in tar:
            tar.extract(member, destination, filter="data")
            count += 1
            progress.update(count, None, "files")
    print(f"extracted {archive.name}: {count:,} files", flush=True)


@app.function(volumes={DATA: volume}, cpu=4, memory=8192, timeout=3 * 3600)
def prepare() -> None:
    data = Path(DATA)
    musan = data / "raw" / "musan" / "musan"
    marker = musan.parent / ".complete"
    parts = sorted((data / MUSAN_PARTS).glob("part*.tar"))
    if not marker.exists() and parts:
        shutil.rmtree(musan, ignore_errors=True)
        for index, part in enumerate(parts, 1):
            print(f"musan part {index}/{len(parts)}", flush=True)
            _extract(part, musan.parent)
        marker.touch()
        shutil.rmtree(data / MUSAN_PARTS)
        volume.commit()
    for archive in sorted((data / "upload").glob("*_clips.tar")):
        (data / "clips").mkdir(exist_ok=True)
        _extract(archive, data / "clips")
        archive.unlink()
        volume.commit()
    for path in ("clips/ami/train.jsonl", "clips/ami/stats.json", "clips/ami_aug/dev.jsonl", "clips/ami_aug/test.jsonl", "clips/turnbench/test.jsonl"):
        print(path, "ok" if (data / path).exists() else "MISSING")
    print("musan wav files:", sum(1 for _ in musan.rglob("*.wav")))


@app.function(volumes={DATA: volume}, gpu="L4", cpu=16, memory=32768, timeout=6 * 3600)
def train(args: str) -> None:
    _run_script("train_interruption.py", f"{args} --workers 14")


@app.function(volumes={DATA: volume}, gpu="L4", cpu=4, memory=16384, timeout=3 * 3600)
def evaluate(args: str) -> None:
    _run_script("evaluate_interruption.py", args)


@app.function(volumes={DATA: volume}, cpu=4, memory=8192, timeout=3600)
def export(args: str) -> None:
    """ONNX export, parity check and latency on a CPU-only x86 container."""

    _run_script("export_onnx.py", args, device=None)
