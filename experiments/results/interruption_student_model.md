# Interruption classifier: Whisper-tiny student

Date: 2026-10-06. `openai/whisper-tiny` encoder @ 169d4a4 (Apache-2.0),
fine-tuned (lr 1e-4, head 1e-3), same head as the teacher (layer mix, GRU).
Whisper log-mel front end re-implemented as a fixed Fourier conv (matches
`WhisperFeatureExtractor` to ~1e-6) so it exports to ONNX. Distilled from
`wavlm_v2_full` (alpha 0.5, T 2) plus hard labels, 10 epochs on Modal L4
(~2 min/epoch), best dev epoch 9. Both channels, full agent audio (the teacher
ablation showed no meaningful leakage; muting at train time would mismatch
deployment).

## Test at 500 ms (threshold set on dev for 85% recall; 95% CI by meeting)

| Model | Params | Stops before onset | Interrupt | Backchannel | Ignore |
|---|---|---|---|---|---|
| Silero VAD 200 ms + level gate | – | 0.32 | 0.85 | 0.71 | 0.33 |
| Teacher (WavLM-base+) | 95.4M | 0.008 | 0.85 [0.79-0.90] | 0.15 [0.10-0.20] | 0.03 [0.02-0.04] |
| Student, PyTorch | 9.0M | 0.005 | 0.82 [0.77-0.86] | 0.15 [0.10-0.19] | 0.016 [0.01-0.02] |
| Student, ONNX fp32 | 9.0M | 0.005 | 0.82 | 0.15 | 0.016 |
| Student, ONNX INT8 | 9.0M | 0.005 | 0.82 [0.77-0.86] | 0.15 [0.10-0.20] | 0.017 [0.01-0.02] |

## Size and CPU latency per decision (one 1 s window, both channels)

| Format | Size | Apple M-series, 1 / 4 threads | Modal x86 vCPU, 1 / 4 threads |
|---|---|---|---|
| ONNX fp32 | 36.8 MB | 3.9 / 2.8 ms | 25.5 / 12.6 ms |
| ONNX INT8 | 13.4 MB | 4.8 / 3.0 ms | 21.1 / 13.1 ms |

Decisions run every 100 ms, so one x86 thread uses ~20-25% of a core.

## Findings

- 10x smaller than the teacher at ~3 points lower recall, same backchannel
  false stops, and fewer ignore false stops.
- INT8 must exclude the mel front end: quantising it gave up to 0.99
  probability error. With it excluded, INT8 matches fp32 on test.
- INT8 saves size, not time: dynamic quantisation leaves convs in fp32 and adds
  overhead; it is ~15% faster on 1 x86 thread, slower on Apple CPUs.
- The <10 ms target is met on Apple CPUs but not on a single x86 vCPU (21 ms).

## Caller-only variant (`whisper_tiny_caller`)

Same recipe, student sees only the caller channel (teacher still both).
Best dev epoch 6 (10.8% false stops vs 6.3% for both channels).

| Student | Stops before onset | Interrupt | Backchannel | Ignore | x86 1 thread | Size (INT8) |
|---|---|---|---|---|---|---|
| Caller + agent | 0.005 | 0.82 | 0.15 | 0.016 | 21 ms | 13.4 MB |
| Caller only | 0.020 [0.01-0.03] | 0.82 [0.77-0.87] | 0.20 [0.13-0.26] | 0.049 [0.04-0.06] | 9 ms | 12.5 MB |

- Dropping the agent channel halves compute and meets the <10 ms x86 target,
  but backchannel false stops rise 0.15 -> 0.20, ignore 0.016 -> 0.049 and
  pre-onset stops 0.005 -> 0.020.
- So the agent's audio before the onset carries real signal (likely pause and
  prosody cues for where listeners backchannel, plus echo reference). The
  teacher ablation muted only post-onset audio, so it could not show this.
- Both variants remain far better than stock VAD (0.71 / 0.33 / 0.32).

## Caveats

- AMI meetings, not agent calls; validation on agent-style calls is pending.
- x86 timing is a shared Modal vCPU; dedicated cores are usually faster.
