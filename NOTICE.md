# Third-party models and datasets

No weights or audio are redistributed by this repository. Revisions are pinned
so a rebuild resolves the same files.

## Models

| Use | Model | Revision | Licence |
|---|---|---|---|
| Shipped classifier encoder (fine-tuned) | [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | `169d4a43` | Apache-2.0 |
| Teacher, development only (never shipped) | [`microsoft/wavlm-base-plus`](https://huggingface.co/microsoft/wavlm-base-plus) | `4c66d480` | Model card states no licence |
| VAD baseline | [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad) 6.2.1 | – | MIT |
| Transcript-rule baseline | [`mlx-community/whisper-small-mlx`](https://huggingface.co/mlx-community/whisper-small-mlx), [`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small) | – | MIT |
| Eval-set check: noise level | [DNSMOS P.835](https://github.com/microsoft/DNS-Challenge) `sig_bak_ovr.onnx` | – | MIT (repository); model file terms not separately audited |
| Eval-set check: sound type | [`lpepino/beats_ckpts`](https://huggingface.co/lpepino/beats_ckpts) `BEATs_iter3_finetuned_on_AS2M_cpt1.pt`, code from [microsoft/unilm](https://github.com/microsoft/unilm) `beats/` | `5b53b040`, `ca43e4cd` | MIT |
| Echo cancellation for recorded calls | WebRTC audio processing via [`livekit`](https://github.com/livekit/python-sdks) | 1.1.20 | Apache-2.0 |

TTS voices used to re-voice training clips and as the test agent are macOS
system voices, used locally to generate training and test audio; the audio is
not redistributed.

## Datasets

| Dataset | Use | Terms |
|---|---|---|
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | Training and in-domain dev/test (headset audio, dialogue acts, vocal sounds) | CC BY 4.0 |
| [MUSAN](https://www.openslr.org/17/) | Noise, music and far-field speech augmentation, split by filename hash | CC BY 4.0 |
| [TurnBench dev](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev) (Mundo AI / Sesame) | Evaluation only; never trained on or redistributed | Dataset Public License v1.0: attribution, non-commercial, no voice cloning |
| AudioSet ontology | Label names for the BEATs check | CC BY 4.0 |
| Self-recorded agent calls | Owner's own sessions; dev and held-out test | Private; not in the repository |
