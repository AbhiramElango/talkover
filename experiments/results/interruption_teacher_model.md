# Interruption classifier: WavLM-base+ teacher

Date: 2026-10-06. Frozen `microsoft/wavlm-base-plus` @ 4c66d48 (94M params, not
shipped), learned layer mix per stream (caller, agent), projection, GRU, 3-way
softmax. Trained on clean AMI train clips with on-the-fly augmentation;
decision windows 1 s long ending 0.1-0.5 s after onset, plus 25% pre-onset
windows labelled ignore. Threshold set on dev for 85% interrupt recall at
500 ms, then applied unchanged to test. Streaming evaluation: a decision every
100 ms, same benchmark as `interruption_baselines.md`.

Runs:
- `wavlm_v1_30mtg`: 30 train meetings, 12 epochs, Mac MPS.
- `wavlm_v2_full`: 114 train meetings, 8 epochs, Modal L4 (best epoch 6).
- `wavlm_v2_mute`: as v2, agent channel zeroed after onset (leakage ablation).

## Test results at 500 ms (95% CI by meeting)

| Policy | Stops before onset | Interrupt | Backchannel | Ignore |
|---|---|---|---|---|
| Silero VAD 200 ms + level gate | 0.32 | 0.85 | 0.71 | 0.33 |
| VAD + ASR 2-word rule | 0.32 | 0.38 | 0.11 | 0.12 |
| v1 (30 meetings) | 0.02 | 0.79 [0.73-0.84] | 0.14 [0.10-0.18] | 0.04 [0.03-0.05] |
| v2 full | 0.008 [0.00-0.01] | 0.85 [0.79-0.90] | 0.15 [0.10-0.20] | 0.03 [0.02-0.04] |
| v2 muted agent | 0.000 | 0.82 [0.76-0.88] | 0.13 [0.10-0.16] | 0.03 [0.02-0.04] |

Interrupt recall by horizon (v2 full / muted): 100 ms 0.41/0.44, 200 ms
0.68/0.70, 300 ms 0.77/0.77, 400 ms 0.83/0.81, 500 ms 0.85/0.82.

## Findings

- At the same interrupt recall as gated VAD (0.85), backchannel false stops
  fall from 0.71 to 0.15, ignore from 0.33 to 0.03, and stops before the
  caller speaks from 0.32 to under 0.01.
- 4x more training data mainly fixed calibration: the dev threshold now
  transfers to test (0.85 recall vs 0.79 for v1).
- Leakage ablation: muting the agent channel after onset changes results
  within CI (dev false stops 5.3% vs 6.3%; test differences inside CIs). The
  model relies on the caller audio, not on the AMI agent's reaction. The muted
  model is the conservative headline.
- Hardest residual cases: laughs (~7-10% false stops) and interrupts in
  0-5 dB noise (recall 0.75).

## Caveats

- AMI meetings, not callers talking over a TTS agent; self-labelled agent-style
  calls (build step 4) are the real test.
- WavLM-base+ model card states no licence; used as a research teacher only.
- Teacher is ~94M params; the deployable small model is the next step.
