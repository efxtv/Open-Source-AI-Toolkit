# 🎙️ ViralVoice TTS Studio
### Free, Studio-Quality AI Voiceovers — No API Key, No Subscription, No Bullshit


Turn any text script into a broadcast-ready, human-sounding voiceover in **one command**. No API keys, no accounts, no cloud subscriptions — just drop in a script, pick a voice, and get back a mastered WAV/MP3 ready for YouTube, TikTok, courses, or podcasts.

---

## ✨ Features

- **12 natural English voices** — US, UK, and Australian male voices tuned specifically for tech/narration content
- **Sentence-level synthesis** — every sentence gets rendered independently with natural breath pauses (320 ms between sentences, 650 ms between paragraphs)
- **Human prosody** — conversational rewrite + natural pacing = no "robot reader" cadence
- **Broadcast-grade mastering chain** (FFmpeg):
  - 70 Hz high-pass filter (removes sub-bass mud)
  - Smile EQ curve: −2 dB @ 220 Hz (mud cut), +2.5 dB @ 3.8 kHz (presence boost), +1.5 dB @ 11 kHz (air boost)
  - De-essing via narrow EQ cut at 7.2 kHz (−3 dB) to tame harsh S/T consonants
  - Soft harmonic saturation (acrusher, 5% wet) for analog-style warmth
  - Glue compression: 3:1 ratio, −20 dB threshold, fast attack, medium release
  - Pink-noise room tone bed (−52 dB) — creates the psychoacoustic feel of a recorded studio mic
  - EBU R128 loudness normalization to **−16 LUFS** (YouTube's recommended broadcast level) with −1.5 dBTP true-peak limiter
- **48 kHz / 16-bit mono PCM WAV** output (or 320 kbps MP3) — professional video-editing standard
- **Auto-installs dependencies** — no `pip install -r requirements.txt` needed
- **Bulletproof error handling** — per-sentence retries, WebSocket throttle protection, tech-jargon token replacement (`.profile` → "dot profile", `QEMU` → "Q M U", etc.)

---

## 🚀 Quick Start with Docker (recommended)

Zero install on your host machine — Docker does everything.

### Step 1: Pull Python 3.12 and open a shell
```bash
docker pull python:3.12
docker run -it --rm -v $(pwd):/work -w /work -p 8080:8080 python:3.12 bash
```

> 💡 The `-v $(pwd):/work` flag mounts your current folder into the container so the generated audio files appear on your host machine. On Windows PowerShell, replace `$(pwd)` with `${PWD}`. On Windows CMD, replace with `%cd%`.

### Step 2: Inside the container shell, install system + Python dependencies
```bash
# Update package index and install wget for convenience
apt-get update && apt-get install -y --no-install-recommends wget ca-certificates

# Install Python packages (edge-tts for speech, imageio-ffmpeg for built-in FFmpeg binary)
pip install --no-cache-dir edge-tts imageio-ffmpeg
```

### Step 3: Bring in `app.py`
Paste your copy of `app.py` into the working directory (via `nano app.py`, VS Code bind-mount, `wget` from a gist, `docker cp`, etc.). The script is self-contained — no other project files are required.

### Step 4: Write your script
Create a plain text file (e.g., `script.txt`) with the words you want narrated. Use blank lines between paragraphs for natural pauses. Keep sentences short and conversational — write like you talk, not like an essay.

### Step 5: Render
```bash
# List all available voices
python app.py list

# Render with a specific voice
python app.py brian script.txt output.wav

# Or export directly to MP3
python app.py andrew script.txt output.mp3
```

---

## 🎙️ Voice Library

| Voice Key | Accent | Style | Best For |
|---|---|---|---|
| `brian` | 🇺🇸 US | Approachable, casual, sincere | **Default** — tech tutorials, explainers, most natural all-rounder |
| `andrew` | 🇺🇸 US | Warm, confident, authentic | Deeper than Brian; strong YouTube narrator |
| `brian-multi` | 🇺🇸 US | Same as Brian, multilingual | Scripts with foreign words/brand names |
| `andrew-multi` | 🇺🇸 US | Same as Andrew, multilingual | Same as above |
| `christopher` | 🇺🇸 US | Reliable, authoritative | News-style, documentary, formal explainers |
| `eric` | 🇺🇸 US | Rational, clear, straightforward | Educational content, no-nonsense tutorials |
| `guy` | 🇺🇸 US | Passionate, deep-voiced | Classic tech-YouTube narrator vibe |
| `roger` | 🇺🇸 US | Lively, energetic, upbeat | High-energy listicles, intros, calls to action |
| `steffan` | 🇺🇸 US | Rational, slightly deeper | Deep-voiced explainers, walkthroughs |
| `ryan` | 🇬🇧 UK | Friendly, smooth British baritone | UK audience, documentary feel |
| `thomas` | 🇬🇧 UK | Warm, classic British | Friendly UK narrator |
| `william` | 🇦🇺 AU | Warm, laid-back Australian | Australian/NZ audience, casual tutorials |

---

## 🧠 How It Works (Tech Stack)

| Layer | Technology | Purpose |
|---|---|---|
| **TTS Engine** | [`edge-tts`](https://github.com/rany2/edge-tts) (Python) | Uses Microsoft Azure Neural TTS public WebSocket endpoint — same voices powering Microsoft Edge Read Aloud. Zero auth, zero cost. |
| **Voice Models** | Microsoft Azure Cognitive Services Neural Voices (en-US-BrianNeural, en-US-AndrewNeural, etc.) | Deep-learning speech synthesis with prosody prediction |
| **Audio I/O & Processing** | FFmpeg (via `imageio-ffmpeg` wheel — no system FFmpeg needed) | Concatenation, resampling (SoXR high-quality resampler), effects processing, codecs |
| **Audio Effects Chain** | FFmpeg native filters: `highpass`, `equalizer`, `acrusher`, `acompressor`, `loudnorm`, `amix`, `anoisesrc` | Studio mastering — all standard stock filters (no GPL-incompatible plugins) |
| **Room Tone** | Generated pink noise (`anoisesrc=c=pink`) low-passed at 400 Hz | Adds a −52 dB noise floor so the audio feels "recorded in a room" instead of dead-digital |
| **Async Runtime** | Python `asyncio` | Concurrent-safe per-sentence synthesis with 50 ms spacing to avoid WebSocket throttling |
| **Container** | `python:3.12` official Docker image | Reproducible environment, no host pollution |

---

## 📁 CLI Reference

```bash
python app.py list                                      # show voice catalog
python app.py <voice> <script.txt> [output.wav|mp3]     # render a script
python app.py --help                                    # show help
```

**Input:** `.txt` file (UTF-8). Paragraphs = blank line between blocks.
**Output:** `.wav` (48 kHz / 16-bit mono PCM) or `.mp3` (320 kbps CBR).
**Speed:** A 2.5-minute script renders in ~20–40 seconds depending on network.

---

## 💡 Pro Tips for Human-Sounding Results

1. **Write conversationally** — use contractions ("you're", "gonna", "it's"), filler words ("alright", "so", "look"), and short sentences. The voice only sounds as human as the writing it's given.
2. **One idea per paragraph** — the 650 ms paragraph pause mimics a speaker taking a breath between thoughts.
3. **Match voice to audience** — use `brian`/`andrew` for US tech, `ryan`/`thomas` for UK, `william` for AU.
4. **Normalize your whole video** to the same −16 LUFS in your editor so the VO matches music/SFX levels.
5. **Leave 100–200 ms of head/tail silence** in your editor before cutting — don't trim right up to the first word.

---

## ⚖️ Notes

- **Requires internet** when generating (uses Microsoft's neural TTS endpoint). Once generated, audio files are yours.
- **No API key, no account, no payment** — this uses the same public WebSocket endpoint as Edge Read Aloud. Check Microsoft's Terms of Service for commercial-use guidance; output audio is your responsibility.
- **Not 100% offline.** For fully offline HD TTS, look into Coqui XTTS v2, Piper, or Kokoro-82M — but those require model downloads and (for best quality) a GPU. Edge-TTS offers the highest quality-to-friction ratio of any free option.

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `No audio was received` | Script automatically retries with safe params. If persistent, check internet / reduce special characters in script. |
| `No such filter: 'adeesser'` | You have an old `app.py` — grab the latest version, all filters are stock FFmpeg now. |
| File named `[app.py](http://app.py)` | You copy-pasted from rendered markdown — rename it to plain `app.py`. |
| `pip` root user warning | Harmless in Docker — ignore it. |
| Output MP3 not playing | WAV is the most compatible format; MP3 requires libmp3lame (included in imageio-ffmpeg wheels on all platforms). |

---
*<a href="https://buymeacoffee.com/efxtv" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>*

**Made for creators who stop wasting money on overpriced voiceover APIs.** 🎙️✨
