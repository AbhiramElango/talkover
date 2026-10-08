"""BEATs AudioSet tagger used to check what the augmented eval clips contain."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

CHECKPOINT_REPO = "lpepino/beats_ckpts"
CHECKPOINT_REVISION = "5b53b0404df452a3a607d7e67687227730e5bad1"
CHECKPOINT_FILENAME = "BEATs_iter3_finetuned_on_AS2M_cpt1.pt"
CHECKPOINT_SHA256 = "379369a41d0b3749f746cdcea8036de506cb3aedecce84de7db0a75fda2a4fe7"


class BeatsAudioSetTagger:
    """Fine-tuned BEATs checkpoint, returning its 527-class AudioSet posterior.

    Output order follows the checkpoint's ``label_dict``, not the AudioSet CSV.
    """

    sample_rate_hz = 16_000

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        expected_sha256: str = CHECKPOINT_SHA256,
    ) -> None:
        path = Path(checkpoint_path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(
                "AudioSet checkpoint hash does not match the reviewed artifact"
            )
        try:
            import torch
            import torchaudio.functional as audio_functional
            from BEATs import BEATs, BEATsConfig
        except ImportError as exc:  # pragma: no cover - optional runtime gate
            raise RuntimeError(
                "BEATs runtime requires PyTorch, torchaudio, and pinned official code"
            ) from exc
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = BEATs(BEATsConfig(checkpoint["cfg"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        if getattr(model, "predictor", None) is None:
            raise ValueError(
                "checkpoint has no AudioSet head; the fine-tuned build is required"
            )
        self._torch = torch
        self._resample = audio_functional.resample
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = model.to(self._device)

    def tag_waveforms(
        self,
        waveforms: tuple[NDArray[np.floating], ...],
        sample_rate_hz: int,
    ) -> NDArray[np.float64]:
        """One 527-class posterior row per waveform."""

        torch = self._torch
        rows = []
        with torch.inference_mode():
            for waveform in waveforms:
                values = torch.from_numpy(
                    np.ascontiguousarray(waveform, dtype=np.float32)
                ).unsqueeze(0)
                if sample_rate_hz != self.sample_rate_hz:
                    values = self._resample(values, sample_rate_hz, self.sample_rate_hz)
                values = values.to(self._device)
                probabilities, _ = self._model.extract_features(
                    values, padding_mask=torch.zeros_like(values, dtype=torch.bool)
                )
                rows.append(probabilities.squeeze(0).float().cpu().numpy())
        return np.asarray(rows, dtype=np.float64)
