# Interruption classifier on self-recorded agent calls (session s1)

Date: 2026-10-06. One session by the owner: macOS TTS agent ("Samantha", gain
0.3) through laptop speakers, owner reacting on the built-in mic. Mic
processed with WebRTC AEC3 (LiveKit `AudioProcessingModule`) as a deployed
agent would hear it. Candidates labelled by the owner: 80 clips (15
interrupt, 10 backchannel, 55 ignore), 24 turns (bootstrap clusters). Small
sample: treat as a sanity check, not a headline.

## Results at 500 ms (thresholds from AMI dev)

| Policy | Stops before onset | Interrupt (want high) | Backchannel (want low) | Ignore (want low) |
|---|---|---|---|---|
| energy_200ms | 0.08 [0.03–0.14] | 0.80 [0.75–0.90] | 0.30 [0.06–0.62] | 0.20 [0.07–0.35] |
| silero_200ms | 0.03 [0.00–0.07] | 0.93 [0.81–1.00] | 0.20 [0.00–0.50] | 0.02 [0.00–0.06] |
| silero_gated_200ms | 0.02 [0.00–0.05] | 0.87 [0.75–1.00] | 0.20 [0.00–0.50] | 0.02 [0.00–0.06] |
| silero_gated_500ms | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] |
| silero_gated+asr_2words | 0.02 [0.00–0.05] | 0.07 [0.00–0.30] | 0.10 [0.00–0.38] | 0.02 [0.00–0.06] |
| model:whisper_tiny_student@model.int8 | 0.00 [0.00–0.00] | 0.47 [0.29–0.67] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] |
| model:whisper_tiny_student | 0.00 [0.00–0.00] | 0.47 [0.29–0.67] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] |
| model:whisper_tiny_caller | 0.00 [0.00–0.00] | 0.67 [0.56–0.83] | 0.30 [0.06–0.62] | 0.15 [0.07–0.24] |
| model:wavlm_v2_full | 0.00 [0.00–0.00] | 0.13 [0.00–0.22] | 0.00 [0.00–0.00] | 0.00 [0.00–0.00] |


## Finding: the two-channel models fail on this domain

Stock VAD catches 0.93 of interrupts; the lead student 0.47, the teacher 0.13.
They become over-conservative (no false stops, few true stops).

Diagnosis (student, 500 ms):

| Input variant | Interrupt | Backchannel | Ignore |
|---|---|---|---|
| As recorded | 0.47 | 0.00 | 0.00 |
| Agent level matched to AMI | 0.53 | 0.00 | 0.00 |
| Caller +10 dB | 0.60 | 0.10 | 0.02 |
| Both matched to AMI (caller +14, agent -15 dB) | 0.67 | 0.10 | 0.09 |
| Raw mic, no AEC | 0.33 | 0.00 | 0.07 |

Median levels: caller after onset -55.5 dBFS (AMI -41.7), agent -26.8 dBFS
(AMI -41.7), so caller/agent ratio is about -29 dB here vs about 0 dB in
training. Two causes:
1. Level sensitivity: training varied caller gain by only +-6 dB and never the
   agent's.
2. Domain shift: a TTS agent voice and real AEC3 artefacts (double-talk
   suppression, high-pass) on the caller were never seen in training.

## Protocol note

s1 is now used for diagnosis, so it is the agent-call dev set. Any fix must be
confirmed on new, held-out sessions (s2+).

## Attempted fix 3c: robustness augmentation (did not help)

Retrained the student with per-channel gain (caller -20..+6 dB, agent
-10..+15 dB) and, for half of the echo, loudspeaker-level echo passed through
real WebRTC AEC3. Two variants: no teacher, and teacher with alpha 0.2.

| Student | AMI test int / bc / ign | TurnBench int / bc / ign | s1 interrupt |
|---|---|---|---|
| Original | 0.82 / 0.15 / 0.02 | 0.78 / 0.29 / 0.02 | 0.47 |
| Robust, no teacher | 0.79 / 0.16 / 0.04 | 0.78 / 0.33 / 0.02 | 0.33 |
| Robust, alpha 0.2 | 0.82 / 0.17 / 0.03 | 0.79 / 0.30 / 0.01 | 0.13 |

## Agent-voice swap diagnostic (original student)

| Agent channel on s1 clips | AUC int vs rest | int @0.58 | bc | ign |
|---|---|---|---|---|
| TTS as recorded | 0.89 | 0.47 | 0.00 | 0.00 |
| Swapped for AMI human agent, same level | 0.89 | 0.73 | 0.30 | 0.02 |
| Human agent + caller +14 / agent -15 dB | 0.88 | 0.87 | 0.50 | 0.22 |

The agent channel shifts the whole score distribution (calibration) without
changing ranking (AUC ~0.89). Recalibrated to VAD's recall (0.93) the student
would false-stop on ~50% of backchannels and ~29% of ignore clips, vs Silero
0.20 / 0.02. On this small sample (15 interrupt, 10 backchannel) the model has
no advantage over VAD in the real TTS-agent setup.

Conclusion: the gap is distribution shift on the agent side (TTS voice), not
caller level or AEC. Fixes need training data whose agent channel looks like
deployment, and a larger held-out agent-call test set.

## Attempted fix B: agent channel as envelope-shaped noise (did not help)

Agent waveform replaced by fixed noise shaped by its normalised 20 ms envelope
(keeps timing, drops voice and level), train and inference. No teacher.

| Student | s1 int / bc / ign | TurnBench int / bc / ign | AMI test int / bc / ign |
|---|---|---|---|
| Original (raw agent audio) | 0.47 / 0.00 / 0.00 | 0.78 / 0.29 / 0.02 | 0.82 / 0.15 / 0.02 |
| Envelope agent | 0.67 / 0.40 / 0.11 | 0.84 / 0.46 / 0.04 | 0.80 / 0.23 / 0.04 |
| Caller only | 0.67 / 0.30 / 0.15 | 0.83 / 0.48 / 0.04 | 0.82 / 0.20 / 0.05 |
| Silero VAD 200 ms | 0.93 / 0.20 / 0.02 | 0.77 / 0.70 / 0.03 | 0.88 / 0.74 / 0.35 |

The envelope model behaves like caller-only: it loses the backchannel benefit
on TurnBench. So the useful agent information is in the voice itself
(prosody), not just its timing, and that does not transfer from human agents
to a TTS agent.

## Fix 1: TTS-voiced agent training data

AMI train clips (3,500 per class) re-voiced with 13 macOS voices (Samantha,
the test agent voice, excluded), each agent spurt synthesised at a rate that
fits its slot so pause timing is preserved (`tts_agent.py`,
`scripts/interruption/build_tts_agent_clips.py`). Trained 50/50 with the original clips,
no teacher (`whisper_tiny_tts`).

| Student (AMI-dev threshold) | s1 int / bc / ign | TurnBench int / bc / ign | AMI test int / bc / ign |
|---|---|---|---|
| Original | 0.47 / 0.00 / 0.00 | 0.78 / 0.29 / 0.02 | 0.82 / 0.15 / 0.02 |
| TTS-agent mix | 0.73 / 0.30 / 0.09 | 0.81 / 0.37 / 0.02 | 0.80 / 0.18 / 0.04 |

## Pre-registered held-out test (s2 + s3)

Owner sessions s2 (12 turns: interrupt / backchannel / natural) and s3 (12
turns: backchannel / interrupt), never looked at before this run: 83 clips
(25 interrupt, 15 backchannel, 43 ignore). Thresholds were set on s1 only
(`scripts/interruption/heldout_eval.py`): match_recall = highest threshold with dev recall
>= Silero's; match_false_stops = lowest threshold with dev non-interrupt
stops <= Silero's. Scored once. 95% CI by turn.

| Policy | Interrupt | Backchannel | Ignore |
|---|---|---|---|
| Silero 200 ms | 0.68 [0.52-0.85] | 0.73 [0.59-0.92] | 0.00 |
| Silero gated | 0.56 [0.31-0.83] | 0.47 [0.25-0.71] | 0.00 |
| Original student, match_recall (t=0.03) | 0.72 [0.50-0.93] | 0.80 [0.65-0.94] | 0.14 [0.04-0.26] |
| Original student, match_false_stops (t=0.17) | 0.52 [0.38-0.71] | 0.33 [0.07-0.62] | 0.00 |
| TTS student, match_recall (t=0.51) | 0.68 [0.48-0.87] | 0.53 [0.29-0.77] | 0.07 [0.00-0.16] |
| TTS student, match_false_stops (t=0.83) | 0.32 [0.12-0.58] | 0.33 [0.13-0.54] | 0.02 [0.00-0.08] |

Paired bootstrap, TTS student (match_recall) minus Silero, same clips:
interrupt +0.00 [-0.10, +0.12]; backchannel -0.20 [-0.45, +0.00] (3 vs 0
discordant clips); ignore +0.07 [+0.00, +0.15] (3 vs 0).

## Conclusion

- TTS-voiced training fixed calibration: the s1-chosen threshold (0.51) is
  close to the AMI one (0.58), whereas the original student needed 0.03.
- On held-out real agent calls the TTS student matches Silero's recall and
  stops for fewer backchannels (borderline significant), but stops for a few
  more noises. Total false stops are about equal (11 vs 11 clips).
- So on real TTS-agent calls the model is not yet a clear win over VAD. Its
  clear wins remain human-human audio (AMI, TurnBench). Closing the gap needs
  in-domain training data (people talking over TTS agents), not more
  augmentation.
