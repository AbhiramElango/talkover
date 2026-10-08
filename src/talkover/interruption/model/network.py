"""Frozen speech encoder plus a small causal head over caller and agent streams."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class LayeredEncoder(Protocol):
    num_layers: int
    dim: int

    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        """(batch, samples) -> (layers, batch, frames, dim)."""
        ...


class FrozenWavLM(nn.Module):
    def __init__(self, model_id: str, revision: str, cache_dir: str | None = None) -> None:
        super().__init__()
        from transformers import WavLMModel

        self.model = WavLMModel.from_pretrained(model_id, revision=revision, cache_dir=cache_dir).eval()
        self.model.requires_grad_(False)
        self.num_layers = self.model.config.num_hidden_layers + 1
        self.dim = self.model.config.hidden_size

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.model(waveforms, output_hidden_states=True).hidden_states)


class LogMel(nn.Module):
    """Whisper's 80-band log-mel front end; the STFT is a fixed Fourier conv so it exports to ONNX."""

    def __init__(self, mel_filters, n_fft: int = 400, hop_length: int = 160) -> None:
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        bins = n_fft // 2 + 1
        time = torch.arange(n_fft, dtype=torch.float64)
        freq = torch.arange(bins, dtype=torch.float64)[:, None]
        angle = 2 * torch.pi * freq * time / n_fft
        window = torch.hann_window(n_fft, dtype=torch.float64)
        basis = torch.cat([torch.cos(angle) * window, -torch.sin(angle) * window])
        self.register_buffer("basis", basis.float()[:, None, :], persistent=False)
        self.register_buffer("filters", torch.as_tensor(mel_filters, dtype=torch.float32).T.contiguous(), persistent=False)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        padded = nn.functional.pad(waveforms[:, None], (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        spectrum = nn.functional.conv1d(padded, self.basis, stride=self.hop_length)
        real, imag = spectrum.chunk(2, dim=1)
        power = (real**2 + imag**2)[..., :-1]
        mel = torch.matmul(self.filters, power).clamp(min=1e-10).log10()
        mel = torch.maximum(mel, mel.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (mel + 4.0) / 4.0


class WhisperTinyEncoder(nn.Module):
    """Trainable Whisper encoder on short windows; positional table sliced to the input length."""

    def __init__(self, model_id: str, revision: str, cache_dir: str | None = None) -> None:
        super().__init__()
        from transformers import WhisperFeatureExtractor, WhisperModel

        whisper = WhisperModel.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        features = WhisperFeatureExtractor.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        self.mel = LogMel(features.mel_filters, features.n_fft, features.hop_length)
        self.encoder = whisper.encoder
        self.num_layers = self.encoder.config.encoder_layers + 1
        self.dim = self.encoder.config.d_model

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        hidden = nn.functional.gelu(encoder.conv1(self.mel(waveforms)))
        hidden = nn.functional.gelu(encoder.conv2(hidden)).transpose(1, 2)
        hidden = hidden + encoder.embed_positions.weight[: hidden.shape[1]]
        states = [hidden]
        for layer in encoder.layers:
            output = layer(hidden, attention_mask=None, layer_head_mask=None)
            hidden = output[0] if isinstance(output, tuple) else output
            states.append(hidden)
        states[-1] = encoder.layer_norm(hidden)
        return torch.stack(states)


ENCODERS = {
    "wavlm-base-plus": (FrozenWavLM, "microsoft/wavlm-base-plus", "4c66d4806a428f2e922ccfa1a962776e232d487b"),
    "whisper-tiny": (WhisperTinyEncoder, "openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"),
}


def build_encoder(name: str, cache_dir: str | None = None) -> nn.Module:
    factory, model_id, revision = ENCODERS[name]
    return factory(model_id, revision, cache_dir)


class InterruptionClassifier(nn.Module):
    """Learned layer mix per stream, projection, GRU over frames, logits from the last frame."""

    def __init__(
        self, encoder: LayeredEncoder, streams: int = 2, hidden: int = 256, classes: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder, self.streams = encoder, streams
        self.layer_logits = nn.Parameter(torch.zeros(streams, encoder.num_layers))
        self.project = nn.ModuleList([nn.Linear(encoder.dim, hidden) for _ in range(streams)])
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(streams * hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, classes)

    def head_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith("encoder.")]

    def encoder_parameters(self):
        return [p for name, p in self.named_parameters() if name.startswith("encoder.") and p.requires_grad]

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch = windows.shape[0]
        layers = self.encoder(windows.reshape(batch * self.streams, -1))
        layers = layers.reshape(layers.shape[0], batch, self.streams, *layers.shape[2:])
        streams = []
        for stream in range(self.streams):
            weights = torch.softmax(self.layer_logits[stream], dim=0)
            mixed = torch.einsum("l,lbfd->bfd", weights, layers[:, :, stream])
            streams.append(torch.relu(self.project[stream](self.dropout(mixed))))
        sequence, _ = self.gru(torch.cat(streams, dim=-1))
        return self.out(self.dropout(sequence[:, -1]))
