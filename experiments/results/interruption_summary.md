# Interruption classifier: summary of the modelling phase

Closed 2026-10-06. Detail lives in the linked result files.

## Task

While a voice agent speaks, the caller makes a sound. Decide within 500 ms
whether to stop (interrupt) or keep talking (backchannel such as "mm-hmm", or
something to ignore: noise, laughter, echo, other voices). Stock agents use
VAD and stop on almost any sound.

## Shipped model

`whisper_tiny_tts`: Whisper-tiny encoder (Apache-2.0), fine-tuned, caller +
agent streams, layer mix + GRU head. 9.0M params; ONNX INT8 13.4 MB; ~5 ms per
decision on Apple CPU, ~21 ms on one x86 vCPU (decisions every 100 ms).
Trained on AMI clips with noise / echo / telephony augmentation, half of them
with the agent re-voiced by TTS.

## Results at 500 ms (stop rates; interrupt high is good, the rest low)

| Test set | Policy | Interrupt | Backchannel | Ignore | Stops before caller speaks |
|---|---|---|---|---|---|
| AMI test (meetings, noise added) | Silero VAD | 0.88 | 0.74 | 0.35 | 0.34 |
| | Shipped model | 0.80 | 0.18 | 0.04 | 0.01 |
| TurnBench dev (human calls) | Silero VAD | 0.77 | 0.70 | 0.03 | 0.12 |
| | Shipped model | 0.81 | 0.37 | 0.02 | 0.05 |
| Own agent calls, held-out (TTS agent) | Silero VAD | 0.68 | 0.73 | 0.00 | – |
| | Shipped model (threshold from s1) | 0.68 | 0.53 | 0.07 | – |

## What holds

- On human audio the model stops for 2-4x fewer backchannels and far fewer
  noises and echoes than VAD at similar recall, and catches interrupts sooner
  (55% within 200 ms on TurnBench vs 0% for VAD).
- A 9M student matches the 95M teacher; INT8 matches fp32.
- The leakage ablation found no cheating on the AMI agent's reaction.

## What does not (yet)

- On real calls with a TTS agent, the model matches VAD's recall with fewer
  backchannel stops (paired -0.20 [-0.45, 0]) but slightly more noise stops;
  total false stops are about equal. It is not a clear win there.
- Cause, established by ablations: the agent channel's benefit comes from the
  human agent's voice and prosody, which does not transfer to TTS voices.
  Level augmentation, real AEC and envelope-only agent input did not fix it;
  TTS-voiced training fixed calibration but not the gap.
- Next lever: in-domain data (people talking over TTS agents) for training.

## Protocol notes

- Thresholds: AMI dev for AMI/TurnBench; s1 (own dev session) for agent calls,
  then applied once to held-out sessions s2+s3 (pre-registered).
- TurnBench is evaluation-only (non-commercial licence); numbers use our clip
  protocol and are not comparable to its leaderboard.
- Sample sizes on own calls are small (held-out: 25 interrupt, 15 backchannel).

## Files

`interruption_eval_set_verification.md`, `interruption_baselines.md`,
`interruption_teacher_model.md`, `interruption_student_model.md`,
`interruption_turnbench.md`, `interruption_agent_calls.md`.
