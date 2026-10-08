# Interruption eval set: augmentation verification

Date: 2026-10-05. Split: AMI test, rendered with `scripts/interruption/render_eval_set.py`
(seed 0, reference level -41.5 dBFS from train speech clips). Checked with
`scripts/interruption/verify_eval_set.py` on the caller channel.

## BEATs (AudioSet head, `BEATs_iter3_finetuned_on_AS2M_cpt1.pt` @ 5b53b04)

Mean posterior in the 1 s before vs. after the onset.

| Group | n | P(speech) pre → post | P(music) pre → post |
|---|---|---|---|
| interrupt | 223 | 0.53 → 0.68 | 0.21 → 0.23 |
| backchannel | 934 | 0.57 → 0.66 | 0.22 → 0.21 |
| onset voice (far-field) | 312 | 0.56 → 0.67 | 0.21 → 0.22 |
| onset music | 317 | 0.55 → 0.47 | 0.22 → 0.52 |
| onset noise | 305 | 0.54 → 0.50 | 0.22 → 0.17 |
| real laugh | 168 | 0.58 → 0.60 | 0.18 → 0.17 |

- Inserted music and voice are heard as intended. Far-field voice is as
  speech-like as real caller speech, so it is a hard negative by design.
- Pre-onset speech probability is ~0.55 everywhere because agent echo is in
  the caller channel.
- Noise bursts are not confirmable from these two classes; they are confirmed
  only by construction (level -15 to 0 dB re reference).
- The BEATs output order follows the checkpoint's `label_dict`, not the
  AudioSet CSV order. Mapping by CSV index gives nonsense ("Belly laugh").

## DNSMOS P.835 background score vs. requested SNR

Caller channel tiled to the 9 s model input.

| Requested SNR | n | Median BAK |
|---|---|---|
| no noise | 459 | 2.84 |
| 15–20 dB | 430 | 2.12 |
| 10–15 dB | 474 | 1.88 |
| 5–10 dB | 436 | 1.74 |
| 0–5 dB | 474 | 1.43 |

Monotonic, so requested SNR bands are a valid slicing axis. Absolute values are
low because echo, telephony and tiling are present in every clip.
