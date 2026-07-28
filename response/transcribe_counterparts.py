#!/usr/bin/env python3
"""Transcribe the agent channel of composites that have no manifest transcript.

The counterpart composites rendered after the human-validation study have no
manifest entry, so pairs.html shows them with no transcript at all. Their audio is
there, though: make_composites.py lays the composite down as stereo with
L = user, R = agent (see its module docstring), so the agent side is recoverable by
downmixing channel 1 alone.

Clips whose agent channel is digital silence are skipped and recorded as silent —
an empty transcript is the correct rendering for those, not a gap to fill.

Writes response/transcripts.json: {clip_stem: {"text": str, "silent": bool}}.
Re-run after new counterparts land; existing entries are kept unless --force.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSITES = os.path.abspath(os.path.join(HERE, "..", "..", "vfdb-study", "data", "composites"))
OUT = os.path.join(HERE, "transcripts.json")

# ffmpeg reports digital silence as -91 dB; anything quieter than this never
# carries speech, and running the model on it only invents text.
SILENCE_DB = -80.0


def agent_peak_db(path):
    """Peak level of the agent channel (stereo right) in dB."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
         "-af", "pan=mono|c0=c1,volumedetect", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"max_volume:\s*(-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else None


def extract_agent_wav(path, wav):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
         "-af", "pan=mono|c0=c1", "-ar", "16000", "-ac", "1", wav],
        check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*", help="composite stems, e.g. p0001-av (default: all *-av/-ao)")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--force", action="store_true", help="re-transcribe stems already present")
    args = ap.parse_args()

    stems = args.stems or sorted(
        f[:-4] for f in os.listdir(COMPOSITES)
        if re.fullmatch(r"p\d+-(av|ao)\.mp4", f))

    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))

    todo = [s for s in stems if args.force or s not in out]
    if not todo:
        print("nothing to do")
        return

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    for stem in todo:
        src = os.path.join(COMPOSITES, stem + ".mp4")
        if not os.path.exists(src):
            print(f"  {stem}: no such composite — skipped")
            continue

        peak = agent_peak_db(src)
        if peak is None or peak <= SILENCE_DB:
            out[stem] = {"text": "", "silent": True}
            print(f"  {stem}: agent channel silent ({peak} dB) — no transcript")
            continue

        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "agent.wav")
            extract_agent_wav(src, wav)
            segs, _ = model.transcribe(wav, language="en", vad_filter=True)
            text = " ".join(s.text.strip() for s in segs).strip()

        out[stem] = {"text": text, "silent": False}
        print(f"  {stem}: {len(text)} chars (peak {peak} dB)")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"wrote {OUT} — {len(out)} entries")


if __name__ == "__main__":
    main()
