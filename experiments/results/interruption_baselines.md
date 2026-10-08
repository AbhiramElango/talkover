# Interruption baselines on the AMI test set

Date: 2026-10-05. Rendered, augmented AMI test set (12 meetings). Reproduce with
`scripts/interruption/run_baselines.py --split test --asr mlx` then
`scripts/interruption/summarize_baselines.py`.

Task: the agent is speaking; at 1.0 s the caller makes a sound. A policy sees
only the caller channel, causally. "Stop rate" = share of clips where it
stopped the agent by the horizon. Want high for interrupt (recall), low for
backchannel and ignore (false stops). "Stops before onset" = stopped on echo or
noise before the caller made any sound (non-interrupt clips). Horizons > 0 are
re-armed at the onset, so every policy is judged on the same clips.

Policies:
- `energy_200ms`: caller level >= reference - 10 dB for 200 ms.
- `silero_200ms`: Silero VAD p >= 0.5 for 200 ms (stock VAD barge-in).
- `silero_gated_200ms`: same, plus a level gate at reference - 15 dB.
- `silero_gated_500ms`: 500 ms minimum (LiveKit-style); cannot fire by 500 ms.
- `silero_gated+asr_2words`: gated VAD, then Whisper-small (MLX) must find >= 2
  words outside a backchannel list (Vapi/ElevenLabs-style). Evaluated at 500 ms
  and 1 s only.

Clips: interrupt 223, backchannel 934, ignore 1116

### Stop rates at the headline horizon

Decision 500 ms after onset. 95% CI by meeting.

| Policy | Stops before onset | Interrupt (want high) | Backchannel (want low) | Ignore (want low) |
|---|---|---|---|---|
| energy_200ms | 0.64 [0.51–0.80] | 0.91 [0.85–0.96] | 0.87 [0.80–0.94] | 0.70 [0.63–0.81] |
| silero_200ms | 0.34 [0.16–0.55] | 0.88 [0.82–0.93] | 0.74 [0.62–0.86] | 0.35 [0.24–0.51] |
| silero_gated_200ms | 0.32 [0.13–0.53] | 0.85 [0.78–0.92] | 0.71 [0.57–0.84] | 0.33 [0.21–0.49] |
| silero_gated_500ms | 0.21 [0.06–0.39] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] |
| silero_gated+asr_2words | 0.32 [0.13–0.53] | 0.38 [0.33–0.42] | 0.11 [0.08–0.15] | 0.12 [0.06–0.19] |

### Ignore false stops by sound

| Policy | voice | music | noise | laugh | cough |
|---|---|---|---|---|---|
| energy_200ms | 0.57 | 0.67 | 0.80 | 0.84 | 0.71 |
| silero_200ms | 0.36 | 0.32 | 0.31 | 0.49 | 0.29 |
| silero_gated_200ms | 0.31 | 0.30 | 0.30 | 0.47 | 0.21 |
| silero_gated_500ms | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| silero_gated+asr_2words | 0.12 | 0.09 | 0.12 | 0.15 | 0.00 |

### interrupt stop rate by noise band

| Policy | clean | 15-20 dB | 10-15 dB | 5-10 dB | 0-5 dB |
|---|---|---|---|---|---|
| energy_200ms | 0.85 | 0.90 | 0.82 | 0.95 | 1.00 |
| silero_200ms | 0.98 | 0.95 | 0.89 | 0.80 | 0.80 |
| silero_gated_200ms | 0.93 | 0.88 | 0.86 | 0.80 | 0.78 |
| silero_gated_500ms | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| silero_gated+asr_2words | 0.37 | 0.45 | 0.43 | 0.37 | 0.29 |

### backchannel stop rate by noise band

| Policy | clean | 15-20 dB | 10-15 dB | 5-10 dB | 0-5 dB |
|---|---|---|---|---|---|
| energy_200ms | 0.77 | 0.83 | 0.85 | 0.95 | 0.97 |
| silero_200ms | 0.80 | 0.80 | 0.73 | 0.77 | 0.61 |
| silero_gated_200ms | 0.72 | 0.73 | 0.71 | 0.75 | 0.60 |
| silero_gated_500ms | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| silero_gated+asr_2words | 0.17 | 0.11 | 0.07 | 0.08 | 0.11 |

### ignore stop rate by noise band

| Policy | clean | 15-20 dB | 10-15 dB | 5-10 dB | 0-5 dB |
|---|---|---|---|---|---|
| energy_200ms | 0.53 | 0.57 | 0.72 | 0.77 | 0.91 |
| silero_200ms | 0.47 | 0.38 | 0.36 | 0.28 | 0.29 |
| silero_gated_200ms | 0.39 | 0.37 | 0.34 | 0.27 | 0.29 |
| silero_gated_500ms | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| silero_gated+asr_2words | 0.14 | 0.13 | 0.12 | 0.11 | 0.08 |

### Interrupt recall by horizon

| Policy | 100 ms | 200 ms | 300 ms | 400 ms | 500 ms | 1000 ms |
|---|---|---|---|---|---|---|
| energy_200ms | 0.00 | 0.61 | 0.82 | 0.87 | 0.91 | 0.96 |
| silero_200ms | 0.00 | 0.31 | 0.75 | 0.86 | 0.88 | 0.92 |
| silero_gated_200ms | 0.00 | 0.30 | 0.68 | 0.80 | 0.85 | 0.91 |
| silero_gated_500ms | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.65 |
| silero_gated+asr_2words | – | – | – | – | 0.38 | 0.62 |

## Takeaways

- Stock VAD (Silero 200 ms) stops for 88% of interrupts but also for 74% of
  backchannels and 35% of ignorable sounds, and in 34% of clips it stops
  before the caller says anything (echo/bleed).
- Energy is worse everywhere and degrades with noise.
- Waiting 500 ms removes nothing useful: it delays every decision, and recall
  is only 0.65 at 1 s.
- The transcript rule cuts false stops to ~11% but catches only 38% of
  interrupts at 500 ms (62% at 1 s): words arrive too late.
- No baseline is both fast and selective. Target for the model at 500 ms:
  interrupt recall >= 0.85 with backchannel/ignore false stops well under the
  ASR rule's ~0.11.

## Caveats

- 12 test meetings, so CIs are wide; per-band cells have no CIs here (see JSON).
- AMI interrupts are meeting interrupts, not callers talking over a TTS agent.
- Whisper runs via MLX (same small model family as the pinned faster-whisper);
  the CPU path was too slow (>40 min).
