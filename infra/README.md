# Infrastructure

`modal/train_app.py` runs training, evaluation and ONNX export on Modal
(app `talkover-train`, volume `talkover-data`). Each job first copies its
inputs from the volume to local disk, because many small random reads over
the network volume are slow.

```bash
python infra/modal/upload_musan.py                     # once: MUSAN train split
modal volume put talkover-data <clips.tar> upload/<name>_clips.tar
modal run infra/modal/train_app.py::prepare            # unpack uploads
modal run --detach infra/modal/train_app.py::train --args "--run NAME --encoder whisper-tiny --encoder-lr 1e-4"
modal run infra/modal/train_app.py::evaluate --args "--run NAME --eval-clips turnbench"
modal run infra/modal/train_app.py::export --args "--run NAME"   # x86 CPU latency
modal volume get talkover-data models/interruption/NAME data/models/interruption/
```

Training and evaluation use an L4 GPU (about 1-2 minutes per epoch); export
runs on CPU. Tokens live in `~/.modal.toml` and `~/.cache/huggingface`,
never in the repository.
