# Interruption classifier on TurnBench dev (real human conversations)

Date: 2026-10-06. TurnBench dev (Mundo AI / Sesame, Dataset Public License
v1.0, evaluation only): 38 two-channel English conversations. Events = caller
labels where >=2 of 3 annotators agree within 0.3 s and the other speaker's
annotated speech covers onset +-0.5 s. Mapping: floor-taking interruptions ->
interrupt; continuer/acknowledgement/reaction backchannels -> backchannel;
laughter, non-speech noise, channel bleed, non-linguistic speech -> ignore;
non-floor-taking interruptions, fillers and the rest excluded. Clips built by
`scripts/interruption/build_turnbench_clips.py`. Thresholds are the AMI-dev ones; nothing
was tuned on TurnBench. No added noise (all "clean" band).

## Results at 500 ms (95% CI by conversation)

Clips: interrupt 322, backchannel 1482, ignore 1653
| Policy | Stops before onset | Interrupt (want high) | Backchannel (want low) | Ignore (want low) |
|---|---|---|---|---|
| energy_200ms | 0.16 [0.12–0.20] | 0.76 [0.70–0.83] | 0.73 [0.65–0.81] | 0.08 [0.05–0.12] |
| silero_200ms | 0.12 [0.09–0.15] | 0.77 [0.71–0.82] | 0.70 [0.63–0.77] | 0.03 [0.02–0.03] |
| silero_gated_200ms | 0.10 [0.08–0.13] | 0.69 [0.62–0.76] | 0.64 [0.56–0.71] | 0.02 [0.01–0.03] |
| silero_gated_500ms | 0.02 [0.01–0.04] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] |
| silero_gated+asr_2words | 0.10 [0.08–0.13] | 0.28 [0.21–0.35] | 0.03 [0.02–0.04] | 0.01 [0.00–0.01] |
| model:whisper_tiny_student | 0.01 [0.00–0.02] | 0.78 [0.73–0.83] | 0.29 [0.23–0.36] | 0.02 [0.01–0.02] |
| model:whisper_tiny_caller | 0.02 [0.01–0.04] | 0.83 [0.78–0.87] | 0.48 [0.40–0.56] | 0.04 [0.03–0.05] |
| model:wavlm_v2_full | 0.01 [0.01–0.02] | 0.77 [0.72–0.82] | 0.28 [0.22–0.33] | 0.03 [0.01–0.04] |
### Interrupt recall by horizon

| Policy | 100 ms | 200 ms | 300 ms | 400 ms | 500 ms | 1000 ms |
|---|---|---|---|---|---|---|
| energy_200ms | 0.00 | 0.03 | 0.48 | 0.70 | 0.76 | 0.93 |
| silero_200ms | 0.00 | 0.00 | 0.43 | 0.66 | 0.77 | 0.94 |
| silero_gated_200ms | 0.00 | 0.00 | 0.38 | 0.59 | 0.69 | 0.90 |
| silero_gated_500ms | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.53 |
| silero_gated+asr_2words | – | – | – | – | 0.28 | 0.65 |
| model:whisper_tiny_student | 0.22 | 0.55 | 0.69 | 0.74 | 0.78 | – |
| model:whisper_tiny_caller | 0.28 | 0.60 | 0.74 | 0.79 | 0.83 | – |
| model:wavlm_v2_full | 0.20 | 0.49 | 0.65 | 0.71 | 0.77 | – |

## Findings

- Out of domain (new speakers, two-party conversation, real labels), the lead
  student still beats stock VAD at the same recall: interrupt 0.78 vs 0.77,
  backchannel false stops 0.29 vs 0.70, stops before onset 0.009 vs 0.12.
- It is much faster: 55% of interrupts caught by 200 ms (Silero 0%, it needs
  200 ms of speech plus frame alignment).
- Domain gap: backchannel false stops double from AMI (0.15) to TurnBench
  (0.29), and interrupt recall drops 0.82 -> 0.78. Ignore stays low (0.016),
  but TurnBench noise is quiet and Silero also handles it (0.03).
- The ASR word rule is very selective here (backchannel 0.03) but catches only
  28% of interrupts at 500 ms.
- Caller-only is clearly worse on backchannels (0.48), confirming the agent
  channel matters.
- Teacher and student are equivalent, so distillation lost nothing here.

## Not comparable

TurnBench's leaderboard scores interruption detection with its own protocol
(recall vs. false-positive rate over full conversations, with latency). These
numbers use our clip protocol, so they must not be compared to its leaderboard.

## Next

The gap is backchannel handling in conversational (non-meeting) speech.
Options: train on conversational data (e.g. otoSpeech full-duplex, NC), or
report a recall/false-stop trade-off curve instead of one AMI-calibrated point.
