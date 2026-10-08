"""Pack MUSAN train-split files into 1 GB tar parts and upload them to the Modal volume with progress.

    python infra/modal/upload_musan.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from talkover.interruption.augment import stable_split
from talkover.interruption.paths import data_dir

VOLUME = "talkover-data"
REMOTE_DIR = "upload/musan_train"


def pack(musan_root: Path, out_dir: Path, part_bytes: int) -> list[Path]:
    files = sorted(p for p in musan_root.rglob("*.wav") if stable_split(p.name) == "train")
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    tar, size, start, last = None, 0, time.monotonic(), time.monotonic()
    for index, path in enumerate(files, 1):
        if tar is None or size >= part_bytes:
            if tar:
                tar.close()
            parts.append(out_dir / f"part{len(parts):03d}.tar")
            tar, size = tarfile.open(parts[-1], "w"), 0
        tar.add(path, arcname=str(path.relative_to(musan_root.parent)))
        size += path.stat().st_size
        if time.monotonic() - last >= 30 or index == len(files):
            last = time.monotonic()
            print(f"packed {index:,}/{len(files):,} files into {len(parts)} parts, {last - start:.0f}s", flush=True)
    if tar:
        tar.close()
    return parts


def upload(parts: list[Path], modal: str) -> None:
    total = sum(p.stat().st_size for p in parts)
    done, start = 0, time.monotonic()
    for index, part in enumerate(parts, 1):
        subprocess.run([modal, "volume", "put", "--force", VOLUME, str(part), f"{REMOTE_DIR}/{part.name}"],
                       check=True, stdout=subprocess.DEVNULL)
        done += part.stat().st_size
        elapsed = time.monotonic() - start
        eta = elapsed / done * (total - done)
        print(f"uploaded {index}/{len(parts)} parts, {done / 1e9:.1f}/{total / 1e9:.1f} GB, "
              f"{done / elapsed / 1e6:.1f} MB/s, {elapsed / 60:.1f} min, ~{eta / 60:.1f} min left", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musan", type=Path, default=data_dir() / "raw" / "musan" / "musan")
    parser.add_argument("--work", type=Path, default=data_dir() / "upload" / "musan_train")
    parser.add_argument("--part-gb", type=float, default=1.0)
    parser.add_argument("--modal", default=shutil.which("modal") or str(Path.home() / ".local/bin/modal"))
    args = parser.parse_args()

    parts = pack(args.musan, args.work, int(args.part_gb * 1e9))
    upload(parts, args.modal)
    shutil.rmtree(args.work)


if __name__ == "__main__":
    main()
