#!/usr/bin/env python3
"""
app.py — Studio-quality HD English voiceover with multiple natural male & female voices.

USAGE:
    python app.py list                              # show all voices
    python app.py <voice> script.txt [out.wav|mp3]  # render script.txt

EXAMPLES:
    python app.py brian script.txt voiceover.wav
    python app.py aria script.txt out.mp3
    python app.py jenny tutorial.txt

All voices are tuned for tech YouTube, tutorials, and narration.
No API key, no account, no payment. Internet required when generating (free MS Edge
neural TTS). Auto-installs deps. Robust against WebSocket hiccups & edge-tts quirks.
"""
import asyncio
import html
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------- VOICE LIBRARY ----------------
# (edge_voice_id, rate, pitch, description)
VOICES = {
    # --- MALE VOICES (12) ---
    "brian":        ("en-US-BrianNeural",             "-2%",  "-1Hz", "US — Approachable, casual, sincere (default; most natural tech voice)"),
    "andrew":       ("en-US-AndrewNeural",            "-2%",  "-1Hz", "US — Warm, confident, authentic; slightly deeper than Brian"),
    "brian-multi":  ("en-US-BrianMultilingualNeural", "-2%",  "-1Hz", "US — Brian, multilingual (handles foreign words)"),
    "andrew-multi": ("en-US-AndrewMultilingualNeural","-2%",  "-1Hz", "US — Andrew, multilingual (handles foreign words)"),
    "christopher":  ("en-US-ChristopherNeural",       "-2%",  "-2Hz", "US — Reliable, authoritative news-anchor tone"),
    "eric":         ("en-US-EricNeural",              "-2%",   "0Hz", "US — Rational, clear, straightforward presenter"),
    "guy":          ("en-US-GuyNeural",               "-3%",  "-2Hz", "US — Passionate, deep-voiced classic tech narrator"),
    "roger":        ("en-US-RogerNeural",              "0%",   "0Hz", "US — Lively, energetic, upbeat"),
    "steffan":      ("en-US-SteffanNeural",           "-2%",   "0Hz", "US — Rational, slightly deeper; great for explainers"),
    "ryan":         ("en-GB-RyanNeural",              "-2%",  "-1Hz", "UK — Friendly, positive, smooth British baritone"),
    "thomas":       ("en-GB-ThomasNeural",            "-2%",  "-1Hz", "UK — Friendly, warm, classic British narrator"),
    "william":      ("en-AU-WilliamMultilingualNeural","-2%", "-1Hz", "AU — Warm Australian, friendly & laid-back"),

    # --- FEMALE VOICES (12) ---
    "aria":         ("en-US-AriaNeural",              "-2%",  "-1Hz", "US — Expressive, versatile, articulate studio presenter"),
    "ava":          ("en-US-AvaNeural",               "-2%",  "-1Hz", "US — Warm, bright, natural conversational voice"),
    "ava-multi":    ("en-US-AvaMultilingualNeural",   "-2%",  "-1Hz", "US — Ava, multilingual (handles foreign words & tech terms)"),
    "emma":         ("en-US-EmmaNeural",              "-2%",  "-1Hz", "US — Polished, professional, clear tutorial presenter"),
    "emma-multi":   ("en-US-EmmaMultilingualNeural",  "-2%",  "-1Hz", "US — Emma, multilingual (handles foreign words)"),
    "jenny":        ("en-US-JennyNeural",             "-2%",  "-1Hz", "US — Smooth, friendly, articulate tech narrator"),
    "michelle":     ("en-US-MichelleNeural",          "-2%",  "-1Hz", "US — Crisp, confident, upbeat explainer voice"),
    "sonia":        ("en-GB-SoniaNeural",             "-2%",  "-1Hz", "UK — Sophisticated, elegant British presenter"),
    "libby":        ("en-GB-LibbyNeural",             "-2%",  "-1Hz", "UK — Warm, clear British narrative voice"),
    "clara":        ("en-CA-ClaraNeural",              "-2%",  "-1Hz", "CA — Smooth, natural Canadian presenter"),
    "natasha":      ("en-AU-NatashaNeural",            "-2%",  "-1Hz", "AU — Clear, friendly Australian narrator"),
    "emily":        ("en-IE-EmilyNeural",              "-2%",  "-1Hz", "IE — Warm, expressive Irish presenter"),
}
DEFAULT_VOICE = "brian"

SENTENCE_PAUSE_MS  = 320
PARAGRAPH_PAUSE_MS = 650
ROOM_TONE_DB       = -52
# ------------------------------------------------


def _ensure(pkg: str, import_name: str | None = None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        print(f"⬇  Installing {pkg}...", file=sys.stderr)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg]
        )


def _ffmpeg_bin() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _clean_text(t: str) -> str:
    """Normalize text so it doesn't trip edge-tts."""
    t = t.replace("&amp;", "&").replace("&nbsp;", " ")
    t = html.unescape(t)                       # decode any HTML entities
    t = re.sub(r"[ \t]+", " ", t)              # collapse multiple spaces
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)     # no space before punctuation
    # Replace problematic tech tokens with speakable versions
    replacements = {
        ".profile":   "dot profile",
        ".bashrc":    "dot bash R C",
        "/etc/":      "slash E T C slash",
        "QEMU":       "Q M U",
        "WSL2":       "W S L 2",
        "VMware":     "V M ware",
        "VirtualBox": "Virtual Box",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    return t.strip()


def _split(text: str):
    text = _clean_text(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for p in paragraphs:
        # Split on sentence boundaries
        sents = re.split(r"(?<=[.!?])\s+", p)
        for s in sents:
            s = s.strip()
            if s:
                chunks.append(("sent", s))
        chunks.append(("para", ""))
    while chunks and chunks[-1][0] == "para":
        chunks.pop()
    return chunks


async def _synth_one(text: str, voice_id: str, rate: str, pitch: str, dest: Path):
    """Render one sentence, retrying with safe defaults if the tuned params fail."""
    import edge_tts

    # Attempt 1: tuned params
    for attempt in range(2):
        try:
            use_rate  = rate  if attempt == 0 else "+0%"
            use_pitch = pitch if attempt == 0 else "+0Hz"
            c = edge_tts.Communicate(text, voice_id, rate=use_rate, pitch=use_pitch, volume="+0%")
            await c.save(str(dest))
            # edge-tts sometimes "succeeds" with a 0-byte file — treat as failure
            if dest.exists() and dest.stat().st_size > 500:
                return
            raise RuntimeError("output too small")
        except Exception as e:
            if attempt == 0:
                await asyncio.sleep(0.4)
                continue
            raise RuntimeError(f"edge-tts failed on sentence: {text[:60]!r}... ({e})") from e


def _ff(args):
    ff_path = _ffmpeg_bin()
    subprocess.run([ff_path, "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def _silence(ms: int, sr: int, dest: Path):
    _ff(["-f", "lavfi", "-t", f"{ms/1000:.3f}", "-i", f"anullsrc=r={sr}:cl=mono",
         "-c:a", "pcm_s16le", str(dest)])


def _room(seconds: float, sr: int, dest: Path):
    af = f"anoisesrc=d={seconds:.3f}:c=pink:r={sr}:a=0.01,lowpass=f=400,volume={ROOM_TONE_DB}dB"
    _ff(["-f", "lavfi", "-i", af, "-c:a", "pcm_s16le", "-ac", "1", str(dest)])


def _towav(src: Path, dest: Path, sr: int):
    _ff(["-i", str(src), "-af", "aresample=resampler=soxr",
         "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(dest)])


def _concat(parts: list[Path], dest: Path, sr: int):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        lf = f.name
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")
    try:
        _ff(["-f", "concat", "-safe", "0", "-i", lf,
             "-c:a", "pcm_s16le", "-ar", str(sr), "-ac", "1", str(dest)])
    finally:
        os.unlink(lf)


def _master(src: Path, dest: Path):
    ext = dest.suffix.lower()
    codec = ["-c:a", "libmp3lame", "-b:a", "320k"] if ext == ".mp3" \
            else ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1"]
    af = (
        "highpass=f=70,"
        "equalizer=f=220:t=q:w=1.0:g=-2,"
        "equalizer=f=3800:t=q:w=1.3:g=2.5,"
        "equalizer=f=11000:t=q:w=0.8:g=1.5,"
        "equalizer=f=7200:t=q:w=1.4:g=-3,"
        "acrusher=level_in=1:level_out=1:bits=8:mode=log:aa=1:mix=0.05,"
        "acompressor=threshold=-20dB:ratio=3:attack=8:release=80:makeup=1.5dB,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    _ff(["-i", str(src), "-af", af, *codec, str(dest)])


def _dur(path: Path):
    ff_path = _ffmpeg_bin()
    out = subprocess.run([ff_path, "-hide_banner", "-i", str(path)],
                         stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True).stdout
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", out)
    if not m:
        return None
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


async def _build(chunks, voice_id, rate, pitch, wd, sr):
    parts: list[Path] = []
    ss = wd / "_ss.wav"; ps = wd / "_ps.wav"
    _silence(SENTENCE_PAUSE_MS, sr, ss)
    _silence(PARAGRAPH_PAUSE_MS, sr, ps)
    total = len([c for c in chunks if c[0] == "sent"])
    done = 0
    for i, (kind, text) in enumerate(chunks):
        if kind == "para":
            parts.append(ps)
            continue
        raw = wd / f"s_{i:04d}.mp3"
        wav = wd / f"s_{i:04d}.wav"
        await _synth_one(text, voice_id, rate, pitch, raw)
        _towav(raw, wav, sr)
        parts.append(wav)
        done += 1
        if done % 8 == 0 or done == total:
            print(f"    …segment {done}/{total}", file=sys.stderr)
        if i + 1 < len(chunks) and chunks[i + 1][0] != "para":
            parts.append(ss)
        # Tiny pause to avoid WebSocket throttling
        await asyncio.sleep(0.05)
    return parts


def _list_voices():
    print("\n🎙  Available voices (24 total — Male & Female studio-quality for tech/narration):\n")
    
    print("  --- MALE VOICES ---")
    male_keys = ["brian", "andrew", "brian-multi", "andrew-multi", "christopher", "eric", "guy", "roger", "steffan", "ryan", "thomas", "william"]
    for name in male_keys:
        vid, rate, pitch, desc = VOICES[name]
        tag = "  ← default" if name == DEFAULT_VOICE else ""
        print(f"  {name:<14} {desc}{tag}")

    print("\n  --- FEMALE VOICES ---")
    female_keys = ["aria", "ava", "ava-multi", "emma", "emma-multi", "jenny", "michelle", "sonia", "libby", "clara", "natasha", "emily"]
    for name in female_keys:
        vid, rate, pitch, desc = VOICES[name]
        tag = "  ← default" if name == DEFAULT_VOICE else ""
        print(f"  {name:<14} {desc}{tag}")

    print("\nUsage:  python app.py <voice> script.txt [output.wav|output.mp3]\n")


def main():
    _ensure("edge-tts")
    _ensure("imageio-ffmpeg", "imageio_ffmpeg")

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__); _list_voices(); sys.exit(0)
    if args[0] == "list":
        _list_voices(); sys.exit(0)

    voice_name = args[0]
    if voice_name not in VOICES:
        print(f"❌ Unknown voice: '{voice_name}'\n", file=sys.stderr)
        _list_voices(); sys.exit(1)
    if len(args) < 2:
        print("❌ Missing script file.\nUsage: python app.py <voice> script.txt [output.wav]", file=sys.stderr)
        sys.exit(1)

    inp = Path(args[1])
    if not inp.exists():
        print(f"❌ Script file not found: {inp}", file=sys.stderr); sys.exit(1)

    if len(args) >= 3:
        out = Path(args[2])
        if out.suffix.lower() not in (".wav", ".mp3"):
            out = out.with_suffix(".wav")
    else:
        out = Path("voiceover.wav")

    voice_id, rate, pitch, desc = VOICES[voice_name]
    chunks = _split(inp.read_text(encoding="utf-8"))
    n_sent = sum(1 for c in chunks if c[0] == "sent")
    print(f"🎙  Voice : {voice_name} — {desc}", file=sys.stderr)
    print(f"📝 Script: {n_sent} sentences, {len(chunks)-n_sent} paragraph breaks", file=sys.stderr)

    SR = 48000
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td); raw = wd / "raw.wav"; mixed = wd / "mixed.wav"
        print("🎙  Synthesizing...", file=sys.stderr)
        parts = asyncio.run(_build(chunks, voice_id, rate, pitch, wd, SR))
        print("🔗 Joining segments...", file=sys.stderr)
        _concat(parts, raw, SR)
        d = _dur(raw)
        src = raw
        if d:
            print("🏠 Adding room tone...", file=sys.stderr)
            rt = wd / "room.wav"; _room(d, SR, rt)
            _ff(["-i", str(raw), "-i", str(rt), "-filter_complex",
                 "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0",
                 "-c:a", "pcm_s16le", "-ar", str(SR), "-ac", "1", str(mixed)])
            src = mixed
        print("🎚  Mastering to broadcast loudness...", file=sys.stderr)
        _master(src, out)

    kb = out.stat().st_size / 1024
    print(f"✅ Done: {out}  ({kb/1024:.2f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
