# AI Video Watermark Remover

AI-powered MP4 video watermark remover with **content-aware inpainting**, **LaMa AI**, **smart watermark selection**, and **Frame 1 based mask detection**.

Remove text watermarks, logos, channel branding, and other unwanted overlays from videos using a simple Gradio web interface.

---

## ✨ Features

- 🎬 MP4 video watermark removal
- 🤖 AI-powered content-aware reconstruction
- 🧠 LaMa image inpainting
- 🎯 Frame 1 based watermark selection
- 🖌️ Rough watermark selection — no need to trace perfectly
- 🔍 Automatic watermark mask refinement
- 🟢 Visual auto-selected mask preview
- 🧩 Context-aware reconstruction
- 🎚️ Adjustable automatic selection strength
- ⚡ CPU support
- 🚀 NVIDIA CUDA GPU support
- 🔄 Automatic CPU/GPU detection
- 🔊 Original video audio preservation
- 🎞️ H.264 MP4 output
- 🌐 Gradio web interface
- 🐍 Python based
- 🐧 Linux/Fedora friendly

---

## 🖼️ How It Works

The application uses a simple workflow:

```text
Upload Video
     │
     ▼
Load Frame 1
     │
     ▼
Roughly Paint Watermark
     │
     ▼
Analyze & Auto-Select
     │
     ▼
AI Refines Watermark Mask
     │
     ▼
LaMa Content-Aware Inpainting
     │
     ▼
Process Video Frames
     │
     ▼
Restore Original Audio
     │
     ▼
Final MP4 Video
```

You don't need to precisely trace every pixel of the watermark.

Simply paint approximately around the watermark and let the application refine the selection.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-video-watermark-remover.git
cd ai-video-watermark-remover
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate it

Linux/macOS:

```bash
source .venv/bin/activate
```

Fedora:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🎞️ FFmpeg Installation

FFmpeg is required for video processing and audio preservation.

### Fedora

```bash
sudo dnf install ffmpeg
```

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

Check installation:

```bash
ffmpeg -version
```

---

## ▶️ Run the Application

### Automatic CPU/GPU detection

```bash
python app.py
```

The application automatically uses CUDA when available and otherwise falls back to CPU.

### Force CPU

```bash
python app.py -cpu
```

### Force NVIDIA GPU

```bash
python app.py -gpu
```

If CUDA is unavailable, `-gpu` will stop with an error instead of silently using the CPU.

---

## 🖥️ Hardware

### CPU

CPU processing is supported on systems without a compatible GPU.

However, AI video inpainting can be significantly slower on CPU.

### NVIDIA GPU

For NVIDIA CUDA GPUs, install a compatible PyTorch build for your CUDA environment.

Check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check the detected GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

---

## 🎨 Using the Application

### Step 1 — Upload Video

Upload your MP4 video through the Gradio interface.

### Step 2 — Load Frame 1

Click:

```text
LOAD FRAME 1
```

The first frame of the video will appear in the editor.

### Step 3 — Roughly Select the Watermark

Paint over the watermark.

You don't have to create a perfect mask.

For example:

```text
┌─────────────────────────────┐
│      WATERMARK / LOGO       │
└─────────────────────────────┘
```

Make sure your rough selection covers the entire watermark.

### Step 4 — Analyze & Auto-Select

Click:

```text
ANALYZE & AUTO-SELECT
```

The application analyzes the selected region and creates a refined mask.

The detected region is shown in green.

### Step 5 — Check the Mask

Make sure the green region covers the watermark without unnecessarily covering the subject.

If the selection is too small:

```text
Increase Selection Strength
```

If it covers too much of the character/background:

```text
Reduce Selection Strength
```

### Step 6 — Remove Watermark

Click:

```text
REMOVE WATERMARK
```

The application processes the video and creates a new MP4 file.

---

## 🧠 AI Processing

The project uses **LaMa (Large Mask Inpainting)** for content-aware image reconstruction.

Instead of placing a transparent layer over the watermark, the application attempts to reconstruct the missing image content from surrounding pixels.

Conceptually:

```text
Original Frame

████████████████████████
████  WATERMARK       ███
████████████████████████


        ↓


AI Inpainting

████████████████████████
████ reconstructed    ███
████ content          ███
████████████████████████
```

Only the detected watermark region is replaced.

Pixels outside the mask are preserved from the original frame.

---

## ⚙️ Advanced Settings

### Selection Strength

Controls how aggressively the application refines the rough watermark selection.

Recommended starting value:

```text
50
```

If part of the watermark is missed:

```text
50 → 60 → 70
```

If too much surrounding content is selected:

```text
50 → 40 → 30
```

### AI Context

Controls how much surrounding image information is supplied to the inpainting model.

Recommended:

```text
150–250
```

Higher values provide more surrounding context but increase processing time.

---

## 📁 Project Structure

```text
ai-video-watermark-remover/
│
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
│
├── .gitignore
└── LICENSE
```

---

## 📦 Requirements

Main dependencies:

```text
gradio
simple-lama-inpainting
numpy
opencv-python
Pillow
torch
torchvision
```

See:

```text
requirements.txt
```

for the exact versions.

---

## 🐳 Docker

A Docker deployment can be added for environments such as:

- Hugging Face Spaces
- Docker
- Podman
- Linux servers
- Cloud GPU machines

Example build:

```bash
docker build -t ai-video-watermark-remover .
```

Run:

```bash
docker run --rm -p 7861:7861 ai-video-watermark-remover
```

Then open:

```text
http://localhost:7861
```

---

## 🌐 Gradio

The application runs on:

```text
http://0.0.0.0:7861
```

For local access:

```text
http://localhost:7861
```

You can change the port with:

```bash
GRADIO_SERVER_PORT=7862 python app.py
```

---

## ⚡ Performance

AI video inpainting is computationally expensive.

Processing speed depends on:

- Video resolution
- Number of frames
- Watermark size
- CPU performance
- GPU performance
- Available RAM/VRAM
- AI context size

Processing a small watermark region is significantly more efficient than sending the entire frame to the AI model.

The application therefore crops around the detected watermark and provides surrounding context to LaMa.

---

## ⚠️ Important Limitations

Content-aware inpainting cannot magically recover pixels that were completely hidden by a watermark.

For example, if a watermark completely covers:

```text
face
hand
hair
clothing
object
```

the original pixels may not exist anywhere in the source frame.

The AI generates a plausible reconstruction based on surrounding information.

Results therefore vary depending on the complexity of the underlying scene.

---

## 🎯 Best Results

For the best results:

1. Use the highest-quality source video available.
2. Keep the rough selection reasonably close to the watermark.
3. Make sure the entire watermark is selected.
4. Don't unnecessarily select large portions of the character.
5. Start with Selection Strength `50`.
6. Start with AI Context `180`.
7. Increase context if reconstruction looks unnatural.
8. Increase selection strength only when part of the watermark remains.

---

## 🛠️ Troubleshooting

### `CUDA is not available`

Run:

```bash
python app.py -cpu
```

For NVIDIA GPU support, install a CUDA-compatible PyTorch build.

### `FFmpeg is not installed`

Fedora:

```bash
sudo dnf install ffmpeg
```

Ubuntu:

```bash
sudo apt install ffmpeg
```

### Application says port 7861 is busy

Use another port:

```bash
GRADIO_SERVER_PORT=7862 python app.py
```

### LaMa is slow

CPU-based AI inpainting can be slow.

If you have a compatible NVIDIA CUDA GPU:

```bash
python app.py -gpu
```

Otherwise:

```bash
python app.py -cpu
```

---

## 🧪 Project Status

**Status:** Experimental / Active Development

The project is intended as an AI-assisted video watermark removal tool with a focus on simple selection and content-aware reconstruction.

Future improvements may include:

- Automatic watermark tracking
- Multi-frame temporal reconstruction
- Better moving watermark detection
- Optical-flow assisted masks
- Improved GPU acceleration
- Batch video processing
- Multiple watermark selection
- Automatic watermark detection
- Video preview
- Hardware acceleration improvements

---

## 🤝 Contributing

Contributions, bug reports, ideas, and pull requests are welcome.

Before submitting an issue, include:

- Operating system
- Python version
- GPU model
- PyTorch version
- Gradio version
- Full error message
- Example of the problematic workflow

---

## 📄 License

Add your preferred open-source license before publishing the repository.

For example:

```text
MIT License
```

---

## ⚖️ Responsible Use

Only remove watermarks from videos when you have the right or permission to modify the content.

Do not use this software to misrepresent ownership, bypass licensing restrictions, or remove attribution from content without authorization.

---

## ⭐ Keywords

```text
AI video watermark remover
video watermark remover
MP4 watermark remover
AI watermark removal
remove watermark from video
remove logo from video
content aware video editing
LaMa inpainting
AI video editor
Python video watermark remover
Gradio watermark remover
open source watermark remover
automatic watermark removal
video inpainting
AI inpainting
content aware inpainting
```

---

## ⭐ Star the Project

If this project is useful to you, consider giving it a ⭐ on GitHub.

Contributions and improvements are welcome.
