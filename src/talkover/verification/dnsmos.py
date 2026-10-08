"""DNSMOS P.835 quality scores (speech, background, overall) for one waveform.

Model: `sig_bak_ovr.onnx` from microsoft/DNS-Challenge, CC BY 4.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE_HZ = 16_000
INPUT_SAMPLES = 144_160


@dataclass(frozen=True)
class DnsmosScores:
    speech: float
    background: float
    overall: float


class DnsmosScorer:
    """Scores 9 s windows (hop ``hop_seconds``) and averages them; short input is tiled to one window."""

    def __init__(self, model_path: Path | str, hop_seconds: float = 4.0, threads: int = 2) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self._hop = max(1, round(hop_seconds * SAMPLE_RATE_HZ))

    def score(self, waveform: NDArray[np.floating], sample_rate_hz: int) -> DnsmosScores:
        mono = np.asarray(waveform, dtype=np.float32)
        if sample_rate_hz != SAMPLE_RATE_HZ:
            from scipy.signal import resample_poly

            divisor = np.gcd(sample_rate_hz, SAMPLE_RATE_HZ)
            mono = resample_poly(mono, SAMPLE_RATE_HZ // divisor, sample_rate_hz // divisor).astype(np.float32)
        if mono.size < INPUT_SAMPLES:
            mono = np.tile(mono, INPUT_SAMPLES // max(mono.size, 1) + 1)[:INPUT_SAMPLES]
        starts = range(0, mono.size - INPUT_SAMPLES + 1, self._hop)
        windows = np.stack([mono[start : start + INPUT_SAMPLES] for start in starts])
        speech, background, overall = self._session.run(None, {self._input: windows})[0].mean(axis=0)
        return DnsmosScores(float(speech), float(background), float(overall))
