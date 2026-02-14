
# 🎙️ Edge-TTS Text-to-Speech

A high-performance, asynchronous web interface for Microsoft Edge's Neural TTS (Text-to-Speech) engine. This project provides a localized, stable, and modernized version of the original [innoai/Edge-TTS-Text-to-Speech](https://huggingface.co/spaces/innoai/Edge-TTS-Text-to-Speech) Hugging Face Space.

---

## 📖 Project Overview
This application leverages Microsoft Edge's online text-to-speech service to generate high-quality, natural-sounding audio files (.mp3) from plain text. It is designed to be lightweight, requiring no GPU and minimal RAM, making it ideal for deployment on small servers, Raspberry Pis, or Docker containers.



## 🛠️ Technical Fixes & Modernization
During the transition from Hugging Face Spaces to a local environment, the following technical "debt" was resolved:

### 1. Dependency Resolution
* **HuggingFace Hub Conflict**: Fixed the critical `ImportError: cannot import name 'HfFolder'`. This was caused by breaking changes in `huggingface_hub` v0.25.0+.
* **Gradio 6.0+ Migration**: Refactored the UI code to comply with the new Gradio 6.0 architecture, specifically moving parameters from `gr.Blocks` to the `.launch()` method.

### 2. Stability & Error Handling
* **NoneType Safeguard**: Implemented a "Fail-Fast" validation system. Instead of the backend crashing when receiving empty strings or unselected voices, it now triggers native Gradio UI `gr.Error` notifications.
* **Asynchronous Execution**: Used `asyncio` to ensure that multiple speech generation requests do not block the main thread, allowing the UI to remain responsive.

### 3. Deployment Optimization
* **Network Binding**: Reconfigured the server to bind to `0.0.0.0`. This allows the application to accept connections from external IP addresses, essential for Docker and Virtual Private Servers (VPS).
* **Dynamic File Management**: Integrated `uuid` and `tempfile` libraries to generate unique, collision-free audio filenames in the system's temporary directory, ensuring data privacy between different users.

---

## 🚀 Features & Capabilities

### 🎤 Voice Selection
* **300+ Voices**: Access all Microsoft neural voices (e.g., `en-US-GuyNeural`, `en-GB-SoniaNeural`).
* **Multi-Gender**: Choice between Male and Female voices for most locales.

### 🌐 Language Support
Extensive global coverage including but not limited to:
| Region | Languages |
| :--- | :--- |
| **Americas** | English (US/CA), Spanish (MX/ES), Portuguese (BR) |
| **Europe** | French, German, Italian, Dutch, Swedish, Polish |
| **Asia** | Chinese (Mandarin/Cantonese), Japanese, Korean, Hindi, Thai |
| **Middle East** | Arabic (multiple dialects), Turkish, Persian |

### 🎚️ Audio Fine-Tuning
* **Rate Control**: Speed up or slow down speech (±100%).
* **Pitch Control**: Adjust the frequency/deepness of the voice (±50Hz).

---

## 🔧 Installation & Usage

### Prerequisites
* Python 3.10 or higher
* A virtual environment (`venv`) is highly recommended

### Setup
1. **Clone and Enter Directory**:
```bash
git clone <repo-link>
cd Edge-TTS-Text-to-Speech
```


2. **Install Requirements**:
```bash
pip install -r requirements.txt

```


3. **Run the App**:
```bash
python3 app.py

```



### Accessing the App

Once running, the terminal will display: `Running on local URL:  http://0.0.0.0:7860`.

* **Local access**: `http://localhost:7860`
* **Network access**: `http://<your-server-ip>:7860`

---

## 📝 Requirements File (`requirements.txt`)


```text
gradio>=5.0.0
edge-tts
asyncio
huggingface_hub>=0.25.0

```

## 📜 License & Acknowledgments

* **Engine**: Powered by `edge-tts` (Microsoft Edge).
* **Original Inspiration**: [innoai](https://huggingface.co/spaces/innoai/Edge-TTS-Text-to-Speech).
* **Disclaimer**: This tool is for educational/personal use. Please refer to Microsoft's service terms for commercial speech synthesis.
