"""Record a session of a TTS 'agent' talking through the speakers while you react on the mic.

Each turn saves a 16 kHz stereo WAV: ch0 = your mic (with real room echo), ch1 = the agent audio.
Run it yourself in a terminal (it needs microphone permission):

    .venv/bin/python scripts/interruption/record_agent_session.py --session s1
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample_poly

from talkover.interruption.paths import data_dir

DEVICE_RATE = 48_000
SAMPLE_RATE = 16_000

AGENT_TURNS = (
    "Thanks for calling. I can see your order from last Tuesday. It shipped from our warehouse in Ohio, and the tracking shows it reached the local depot this morning, so you should get it by tomorrow evening at the latest.",
    "Let me explain how the refund works. Once we receive the item, our team inspects it within two business days. After that, the refund goes back to your original payment method, and your bank may take another three to five days.",
    "Your current plan includes ten gigabytes of data, unlimited calls, and five hundred international minutes. If you upgrade to the premium plan, you would get unlimited data and free roaming in about forty countries.",
    "I have checked the appointment calendar for you. Doctor Patel has openings on Thursday at ten thirty and Friday at two fifteen. Both are in-person visits, and you would need to arrive fifteen minutes early to fill in the forms.",
    "To reset your password, first open the login page and select forgot password. We will send a six digit code to your email address. Enter that code, then choose a new password with at least twelve characters.",
    "The warranty covers manufacturing defects for two years from the purchase date. It does not cover accidental damage, water damage, or normal wear. If you would like extra protection, we offer an extended plan for a small monthly fee.",
    "I can see there was a late fee on your last statement. That happened because the payment arrived two days after the due date. Since this is the first time, I can request a one-time waiver, which usually takes one billing cycle.",
    "Our delivery window for your area is between eight in the morning and six in the evening. The driver will call about thirty minutes before arriving. If nobody is home, the package will be left at the nearest pickup point.",
    "The table for four is confirmed for Saturday at seven thirty. We have noted the birthday celebration, so the staff will bring a small dessert. Parking is available behind the restaurant, and the first two hours are free.",
    "Your flight to Chicago departs at nine forty from gate twelve. Boarding starts forty minutes before departure. You have one checked bag included, and the weight limit is twenty three kilograms per bag.",
    "Let me walk you through the setup. Plug the router into the wall socket, then connect the grey cable to the port marked internet. Wait about two minutes until the light turns solid green before you connect your devices.",
    "Based on what you have described, the most likely issue is the battery. These models sometimes lose capacity after about three years. We can replace it in store, and the repair usually takes less than an hour.",
    "The insurance claim has been received and assigned a case number. An adjuster will contact you within five working days to arrange an inspection. Please keep any receipts related to the damage until the claim is closed.",
    "I understand you want to cancel the subscription. Before I do that, I wanted to mention that we can pause it for up to three months instead, and you would keep your saved preferences and your current price.",
    "Your prescription is ready for pickup at the pharmacy on Main Street. They are open until nine tonight. Please bring a photo ID, and if someone else is collecting it for you, they will need a signed note.",
    "The software update fixes several security issues and improves battery life. It takes about twenty minutes to install, and your phone will restart once during the process. Make sure it is charged above fifty percent first.",
    "For the hotel booking, you have a double room with breakfast included for three nights. Check-in starts at three in the afternoon, and check-out is at eleven. Late check-out can be arranged at the front desk if available.",
    "We noticed some unusual activity on your card yesterday afternoon. There were two purchases from an online store that you may not recognise. For your safety, we have temporarily blocked the card until you confirm them.",
    "The course runs for eight weeks with one live session every Wednesday evening. All recordings are available afterwards, and there is a short assignment each week. Your certificate is issued after you complete the final project.",
    "Here is a summary of your energy usage. This month you used about fifteen percent more than last month, mostly in the evenings. Switching to the off-peak tariff could reduce your bill by around twenty dollars.",
    "The replacement part is on back order, unfortunately. The supplier expects new stock in about two weeks. I can reserve one for you now, and we will send a text message as soon as it arrives at the store.",
    "To transfer money abroad, open the payments tab and choose international transfer. You will need the recipient's full name, bank code, and account number. Transfers usually arrive within one to two business days.",
    "Your car is due for its annual service next month. The service includes an oil change, brake inspection, and tyre rotation. We have a courtesy car available if you book at least one week in advance.",
    "Let me confirm the details before we finish. The new address is saved, the next delivery is scheduled for Monday, and your discount code has been applied. Is there anything else I can help you with today?",
)

INSTRUCTIONS = {
    "backchannel": "While the agent talks, give 2-3 short acknowledgements (mm-hmm, yeah, okay, right). Do NOT take over.",
    "interrupt": "Partway through, clearly interrupt: ask a question or correct the agent, and keep talking for 2+ seconds.",
    "background": "Stay silent. Make background sound: TV/music/YouTube near the laptop, typing, moving things.",
    "vocal": "Stay silent except 1-2 non-speech sounds: a laugh, cough, sigh or throat clear.",
    "other_voice": "Stay silent. Someone else in the room talks (not to the agent), or play a podcast nearby.",
    "natural": "React naturally: a couple of backchannels and, if it feels right, one interruption.",
}


def synthesize(text: str, voice: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "turn.wav"
        subprocess.run(["say", "-v", voice, "-o", str(path), "--file-format=WAVE", "--data-format=LEF32@48000", text], check=True)
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != DEVICE_RATE:
        raise ValueError(f"unexpected TTS rate {rate}")
    return audio[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument("--gain", type=float, default=0.3, help="agent playback level (0-1)")
    parser.add_argument("--kinds", nargs="+", choices=sorted(INSTRUCTIONS), default=list(INSTRUCTIONS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = data_dir() / "raw" / "agent_calls" / args.session
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new session name")
    out.mkdir(parents=True)
    rng = random.Random(args.seed)
    texts = rng.sample(AGENT_TURNS, k=min(args.turns, len(AGENT_TURNS)))
    kinds = [args.kinds[i % len(args.kinds)] for i in range(len(texts))]
    rng.shuffle(kinds)

    print(f"Recording {len(texts)} turns to {out}. Use laptop speakers, not headphones.")
    print("Each turn: read the instruction, press Enter, then react while the agent talks.\n")
    session = {"session": args.session, "voice": args.voice, "gain": args.gain,
               "device_rate": DEVICE_RATE, "sample_rate": SAMPLE_RATE, "turns": []}
    for index, (text, kind) in enumerate(zip(texts, kinds), 1):
        agent = synthesize(text, args.voice) * args.gain
        print(f"[{index}/{len(texts)}] {kind.upper()}: {INSTRUCTIONS[kind]}")
        input("    press Enter to start...")
        played = np.concatenate([np.zeros(DEVICE_RATE // 2, np.float32), agent, np.zeros(DEVICE_RATE, np.float32)])
        recorded = sd.playrec(played[:, None], samplerate=DEVICE_RATE, channels=1, dtype="float32")
        sd.wait()
        stereo = np.stack([recorded[:, 0], played], axis=1)
        stereo = resample_poly(stereo, SAMPLE_RATE, DEVICE_RATE, axis=0).astype(np.float32)
        sf.write(out / f"turn_{index:02d}.wav", stereo, SAMPLE_RATE)
        session["turns"].append({"index": index, "instruction": kind, "text": text,
                                 "agent_seconds": round(agent.size / DEVICE_RATE, 2),
                                 "mic_peak_dbfs": round(float(20 * np.log10(np.abs(recorded).max() + 1e-9)), 1)})
        (out / "session.json").write_text(json.dumps(session, indent=1))
        print(f"    saved ({agent.size / DEVICE_RATE:.1f} s agent audio)\n")
        time.sleep(0.3)
    print("Done. Next: .venv/bin/python scripts/interruption/label_agent_session.py --session", args.session)


if __name__ == "__main__":
    main()
