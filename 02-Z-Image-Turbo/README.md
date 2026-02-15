# 🎨 Z-Image-Turbo (CPU & Local Network Edition)

This project is a modified version of the [Z-Image-Turbo Space by mrfakename](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo), optimized to run on **CPU-only environments** and accessible across a **local network (0.0.0.0)**. 

Z-Image-Turbo is a state-of-the-art 6B-parameter distilled model by Alibaba Tongyi-MAI, capable of generating high-quality images in just 8-9 steps.

## ✨ Key Features & Enhancements

* **CPU Compatibility:** Modified to run without an NVIDIA GPU. It uses `torch.float32` and CPU-based generators for stability on standard processors.
* **Network Accessibility:** Configured to host on `0.0.0.0`, allowing you to access the web interface from any device (phone, tablet, or other PC) on your local Wi-Fi.
* **Persistent Storage:** Added a custom storage configuration (`HF_HOME`) so that the ~15GB of model files are saved to a specific folder (`./model_storage`) rather than hidden system caches.
* **Bilingual Text Rendering:** Superior support for both English and Chinese text within generated images.
* **Ultra-Fast Architecture:** Uses the Single-Stream Diffusion Transformer (S3-DiT) architecture, requiring only 8 inference steps for high-fidelity results.

## 🛠 Changes from Original Source

1.  **Device Mapping:** Removed `pipe.to("cuda")` and replaced with `pipe.to("cpu")`.
2.  **Precision Adjustments:** Changed `torch_dtype` from `bfloat16` to `float32` to ensure compatibility with CPUs that do not support specialized 16-bit instructions.
3.  **Removed ZeroGPU Dependencies:** Stripped `@spaces.GPU` decorators and the `spaces` library imports which are exclusive to Hugging Face infrastructure.
4.  **Launch Settings:** Updated `demo.launch()` with `server_name="0.0.0.0"` and defined a static `server_port=7860`.
5.  **Environment Variables:** Integrated `os.environ` settings to manage model cache paths and Hugging Face tokens within the script.

## 🚀 Getting Started

### Installation
1. Clone this repository or copy the `app.py`.
2. Install the dependencies:
```bash
pip install -r requirements.txt

```

### Running the App

Execute the Python file:

```bash
python app.py

```

### Accessing the Interface

* **Local:** Open `http://localhost:7860` in your browser.
* **Network:** Open `http://<your-computer-ip>:7860` (e.g., `http://192.168.1.5:7860`) from any device on your Wi-Fi.

## ⚙️ Technical Specifications

* **Model:** `Tongyi-MAI/Z-Image-Turbo`
* **Parameters:** 6 Billion
* **Recommended Steps:** 8-12 (for CPU)
* **VRAM Required:** 0MB (Runs on System RAM)
* **Disk Space:** ~20GB (for model weights and encoders)

---

**Note:** Because this runs on a CPU, generation times will be slower than the GPU-powered demo. Expect 1-3 minutes per image depending on your CPU power.

---

### 2. Dependency File (`requirements.txt`)

Create a file named `requirements.txt` in the same directory and paste this:

```text
torch
torchvision
torchaudio
diffusers
transformers
accelerate
gradio
safetensors
sentencepiece
huggingface_hub

```
---
## 📜 Source & Credits

This project is built upon the research and development of the AI community. Special credit to the original creators and contributors:

* **Original Model & Research:** [Tongyi-MAI](https://huggingface.co/Tongyi-MAI) (Alibaba)
* *Model Architecture:* Single-Stream Diffusion Transformer (S3-DiT).


* **Original Gradio Demo:** [@mrfakename](https://huggingface.co/mrfakename)
* **Model Optimizations:** [@multimodalart](https://huggingface.co/multimodalart) (FA3 + AoTI optimizations).
* **UI Redesign:** AnyCoder
* **Base Source Link:** [Hugging Face Space: Z-Image-Turbo](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo)
* **License:** Apache 2.0

---
