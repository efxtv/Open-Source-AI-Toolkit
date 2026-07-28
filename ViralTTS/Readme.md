# ViralTTS by EFXTv

ViralTTS is a command-line text-to-speech and voiceover generator for Python. It uses Microsoft Edge neural text-to-speech voices and FFmpeg audio processing to turn a UTF-8 text file into a narrated WAV or MP3 file.

The project is designed for:

- YouTube narration
- Tutorials and explainers
- Documentary-style videos
- Educational content
- Audiobook drafts
- Hindi, English, Spanish, and multilingual narration
- Linux desktop or server environments
- Android devices running Termux

An internet connection is required while generating speech. No Microsoft API key or paid account is required.

> **Important quality note:** ViralTTS preserves the natural output of the selected Microsoft voice and applies gentle mastering. Voice realism, emotion, accent, and pronunciation still depend on the underlying Microsoft voice model. Audio processing cannot make every model sound identical or turn synthetic speech into a real studio actor.

---

## Supported platforms

### Linux

The script works on Debian, Ubuntu, Linux Mint, Kali Linux, and other Debian-based distributions. It can also work on other Linux distributions when Python 3, pip, and FFmpeg are installed through the distribution's package manager.

### Termux on Android

The script works in Termux using Termux's native Python and FFmpeg packages. Installing Termux from F-Droid or GitHub is generally recommended because the Google Play version may be outdated.

The script identifies the runtime internally and silently. It does not display operating-system detection messages.

---

## Supported languages

The included catalog contains 322 Microsoft voice models covering 75 languages. Some languages have multiple regional accents.

- Afrikaans
- Albanian
- Amharic
- Arabic
- Azerbaijani
- Bengali
- Bosnian
- Bulgarian
- Burmese
- Catalan
- Chinese
- Croatian
- Czech
- Danish
- Dutch
- English
- Estonian
- Filipino
- Finnish
- French
- Galician
- Georgian
- German
- Greek
- Gujarati
- Hebrew
- Hindi
- Hungarian
- Icelandic
- Indonesian
- Inuktitut
- Irish
- Italian
- Japanese
- Javanese
- Kannada
- Kazakh
- Khmer
- Korean
- Lao
- Latvian
- Lithuanian
- Macedonian
- Malay
- Malayalam
- Maltese
- Marathi
- Mongolian
- Nepali
- Norwegian
- Pashto
- Persian
- Polish
- Portuguese
- Romanian
- Russian
- Serbian
- Sinhala
- Slovak
- Slovenian
- Somali
- Spanish
- Sundanese
- Swahili
- Swedish
- Tamil
- Telugu
- Thai
- Turkish
- Ukrainian
- Urdu
- Uzbek
- Vietnamese
- Welsh
- Zulu

Hindi includes the available native Hindi models and clearly labelled multilingual Hindi/Hinglish alternatives. Native-language choices normally provide the most accurate pronunciation. Multilingual choices can be useful for scripts containing a mixture of Hindi and English.

---

## Project files

A basic project directory should contain:

```text
viraltts/
├── app.py
├── requirements.txt
└── script.txt
```

- `app.py` — the ViralTTS program
- `requirements.txt` — Python package requirements
- `script.txt` — the text that will be narrated

The Python requirements are:

```text
edge-tts
imageio-ffmpeg
```

FFmpeg is also required. On Termux and Linux, the native system FFmpeg package is preferred.

## Installing packages from `requirements.txt`

The `requirements.txt` file allows pip to install every required Python package with one command. Run the command from the directory containing both `app.py` and `requirements.txt`.

Termux:

```bash
python -m pip install -r requirements.txt
```

Linux with an activated virtual environment:

```bash
python -m pip install -r requirements.txt
```

Linux without a virtual environment, when permitted by the distribution:

```bash
python3 -m pip install --user -r requirements.txt
```

A virtual environment is recommended on modern Debian-based distributions because the system Python installation may be externally managed.

If `requirements.txt` does not exist, create it with:

```bash
printf "edge-tts\nimageio-ffmpeg\n" > requirements.txt
```

---

# Complete Termux installation

The following block installs Python, pip, FFmpeg, and all Python requirements:

```bash
pkg update -y
pkg upgrade -y
pkg install -y python ffmpeg
cd ~/tools/tts
python -m pip install -r requirements.txt
python -m py_compile app.py
```

Change `~/tools/tts` if the project is stored in another directory.

Confirm the installations:

```bash
python --version
python -m pip --version
ffmpeg -version
```

---

# Complete Debian/Ubuntu Linux installation

The following block installs Python, pip, virtual-environment support, FFmpeg, and all Python requirements:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg
cd /path/to/viraltts
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m py_compile app.py
```

Replace `/path/to/viraltts` with the actual project directory.

Confirm the installations:

```bash
python --version
python -m pip --version
ffmpeg -version
```

When returning to the project later, reactivate the environment:

```bash
cd /path/to/viraltts
source .venv/bin/activate
```

Leave the environment when finished:

```bash
deactivate
```

---

# Installation on Termux

## 1. Update Termux packages

```bash
pkg update && pkg upgrade
```

## 2. Install system dependencies

```bash
pkg install python ffmpeg
```

These packages provide:

- Python 3
- pip
- FFmpeg audio conversion and mastering

## 3. Open the project directory

For example:

```bash
cd ~/tools/tts
```

## 4. Install Python dependencies

Using `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Or install them directly:

```bash
python -m pip install edge-tts imageio-ffmpeg
```

## 5. Check the script

```bash
python -m py_compile app.py
```

No output means the Python syntax is valid.

## 6. Display supported language choices

```bash
python app.py list
```

To display only one language:

```bash
python app.py list English
```

Replace `English` with any supported language from the list above.

---

# Installation on Debian-based Linux

The following instructions apply to Debian, Ubuntu, Linux Mint, Kali Linux, and similar distributions.

## 1. Update the package index

```bash
sudo apt update
```

## 2. Install system dependencies

```bash
sudo apt install python3 python3-pip python3-venv ffmpeg
```

These packages provide:

- Python 3
- pip
- Python virtual environments
- FFmpeg audio processing

## 3. Open the project directory

```bash
cd /path/to/viraltts
```

## 4. Create a virtual environment

A virtual environment prevents package conflicts with the operating system's Python installation.

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

## 5. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or install the packages directly:

```bash
python -m pip install edge-tts imageio-ffmpeg
```

## 6. Check the script

```bash
python -m py_compile app.py
```

## 7. Display supported language choices

```bash
python app.py list
```

Filter the results by language:

```bash
python app.py list Spanish
```

When finished, leave the virtual environment with:

```bash
deactivate
```

---

# Installing on other Linux distributions

Install these dependencies using your distribution's package manager:

- Python 3
- pip
- FFmpeg
- Python virtual-environment support, if available

Then install the Python packages:

```bash
python3 -m pip install edge-tts imageio-ffmpeg
```

The exact system-package commands differ between Fedora, Arch Linux, openSUSE, Alpine Linux, and other distributions.

---

# Preparing `script.txt`

ViralTTS reads narration from a UTF-8 text file. Put all text that should be spoken inside `script.txt`.

Example structure:

```text
Welcome to today's video.

In this tutorial, we will explain the complete process step by step. First, we will prepare the required files. Then, we will generate the final narration.

Let's get started.
```

## Script-writing recommendations

### Use complete sentences

Complete sentences help the neural model determine natural rhythm and pauses.

Good:

```text
Today, we are going to explore a powerful new method for creating video narration.
```

Less natural:

```text
Today new method video narration powerful explore.
```

### Use punctuation

Commas, periods, question marks, and exclamation marks influence pacing.

```text
Have you ever wondered how this works? Today, we are going to find out.
```

Avoid adding excessive punctuation because it can create unnatural pauses.

### Separate sections with blank lines

A blank line creates a paragraph boundary:

```text
This is the introduction to the video.

Now, let us move to the first section.
```

The program keeps nearby paragraphs together when possible so the voice retains a consistent rhythm and identity.

### Write words as they should be pronounced

Unusual abbreviations, file paths, product names, and technical terms may be pronounced incorrectly. Rewrite difficult terms phonetically when necessary.

For example, an abbreviation can be separated into letters:

```text
W S L
```

### Use the appropriate writing system

For native Hindi narration, Devanagari generally provides better pronunciation than Romanized Hindi. For Hinglish, use a multilingual Hindi/Hinglish choice and test a short sample before generating a long project.

### Save the file as UTF-8

The script supports Unicode text, including Devanagari, Arabic, Chinese, and other writing systems. Make sure the editor saves `script.txt` with UTF-8 encoding.

---

# Selecting a language and voice

Display every available choice:

```bash
python app.py list
```

Display choices for one language:

```bash
python app.py list <language>
```

For example:

```bash
python app.py list Hindi
```

Each result displays a selector inside square brackets. Copy that selector and use it in the generation command.

The general command format is:

```bash
python app.py <voice-selector> script.txt output.wav --style natural
```

This documentation intentionally does not list individual voice names because the program already organizes and displays the complete catalog by language.

---

# Generating a voiceover

## Highest-quality WAV output

```bash
python app.py <voice-selector> script.txt output.wav --style natural
```

WAV is recommended for video editing because it avoids an additional lossy MP3 encode. The output is mastered as mono, 48 kHz, 24-bit PCM audio.

## Smaller MP3 output

```bash
python app.py <voice-selector> script.txt output.mp3 --style natural
```

MP3 output is encoded at 320 kbps.

## Output location

To save the result in another directory, include the path:

```bash
python app.py <voice-selector> script.txt ~/voiceovers/output.wav --style natural
```

The destination directory is created automatically when possible.

---

# Delivery styles

ViralTTS provides the following delivery profiles:

- `natural` — preserves the original model's native delivery
- `narrative` — slightly slower storytelling pace
- `deep` — slower delivery with gentle low-frequency warmth
- `emotional` — natural pacing with less aggressive compression
- `warm` — soft and relaxed presentation
- `cinematic` — measured documentary pacing

Example:

```bash
python app.py <voice-selector> script.txt output.wav --style narrative
```

For the least robotic and most faithful output, start with `natural`. The other profiles modify pacing and mastering, but they cannot add genuine emotions that are not supported by the underlying voice model.

---

# Recommended workflow

1. Write and proofread `script.txt`.
2. Display choices for the required language.
3. Select a choice marked as suitable for narration when available.
4. Generate a short WAV sample using `--style natural`.
5. Check pronunciation, pacing, and accent.
6. Adjust punctuation or phonetic spelling in `script.txt`.
7. Generate the complete WAV voiceover.
8. Import the WAV file into the video editor.

Testing a short sample first saves time and helps identify pronunciation issues before processing a long script.

---

# Troubleshooting

## `FFmpeg was not found`

Termux:

```bash
pkg install ffmpeg
```

Debian-based Linux:

```bash
sudo apt install ffmpeg
```

Confirm the installation:

```bash
ffmpeg -version
```

## `No module named edge_tts`

```bash
python -m pip install edge-tts
```

On Linux, activate the virtual environment first if one was created.

## `No module named imageio_ffmpeg`

```bash
python -m pip install imageio-ffmpeg
```

The program prefers the native system FFmpeg executable. `imageio-ffmpeg` acts as a fallback on supported platforms.

## Voice generation fails

Check the following:

- The device has a working internet connection.
- The selected model is still available from Microsoft.
- `script.txt` is not empty.
- The script is saved as UTF-8.
- The selected language matches the language used in the text.
- The command contains the selector exactly as displayed by `list`.

Try a short test script to determine whether the problem is related to length or specific text.

## The voice sounds robotic

- Use `--style natural`.
- Select a model marked as suitable for narration.
- Use complete sentences and natural punctuation.
- Avoid excessive exclamation marks, ellipses, or abbreviations.
- Use the model's native language whenever possible.
- Generate WAV instead of MP3 when the file will be edited further.
- Test another model from the same language.

Different Microsoft models have different levels of realism. ViralTTS avoids aggressive pitch shifting and metallic effects, but it cannot replace or retrain the underlying model.

## Hindi pronunciation is inaccurate

- Prefer native Hindi choices for Devanagari text.
- Use multilingual alternatives primarily for Hinglish.
- Rewrite uncommon English terms phonetically when necessary.
- Add commas or sentence breaks where natural pauses are needed.
- Test a short passage before generating the full narration.

---

# Privacy and network usage

The text must be sent over the internet to Microsoft's online speech service for synthesis. Do not process confidential, private, regulated, or sensitive text unless this data handling is acceptable for the project.

Temporary audio segments are created during processing and removed automatically after the final file is generated.

---

# Limitations

- Internet access is required for speech generation.
- Microsoft may add, remove, rename, or change voices without notice.
- Genuine acting emotions are not exposed by the free `edge-tts` interface.
- Not every language or regional model has the same realism.
- 48 kHz and 24-bit output prevent additional production loss but cannot add detail missing from the source synthesis.
- Multilingual alternatives may have a non-native accent.
- The service is not a replacement for a human studio narrator when exact emotion, pronunciation, or licensing guarantees are required.

---

## Quick reference

List all supported choices:

```bash
python app.py list
```

List one language:

```bash
python app.py list <language>
```

Generate a natural WAV voiceover from `script.txt`:

```bash
python app.py <voice-selector> script.txt output.wav --style natural
```

Generate an MP3 voiceover:

```bash
python app.py <voice-selector> script.txt output.mp3 --style natural
```

Generate slower narration:

```bash
python app.py <voice-selector> script.txt output.wav --style narrative
```

---

**ViralTTS by EFXTv**
---
*<a href="https://buymeacoffee.com/efxtv" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>*
