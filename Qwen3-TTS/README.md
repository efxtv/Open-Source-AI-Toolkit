# Qwen3-TTS: RAM-Optimized CPU Implementation and GPU Mode

This repository contains a specialized implementation of the **Qwen3-TTS** (Text-to-Speech) architecture. It has been re-engineered to run exclusively on system RAM (CPU) for environments without dedicated GPU resources. GPU based python file is also available.

By implementing custom memory management and lazy-loading protocols, this version successfully runs the **0.6B and 1.7B models** within a **14GB to 24GB RAM** envelope.

---

## 🛠 Technical Optimizations

To overcome the heavy VRAM requirements of the original model, we implemented the following "RAM-Safe" features:

* **Lazy Loading Protocol:** Models are not loaded into memory at startup. Instead, they are initialized only when a specific synthesis tab (Voice Design, Clone, or TTS) is triggered.
* **Active Heap Management:** The script utilizes an explicit `clear_ram()` function that deletes model references and triggers `gc.collect()`. This ensures that when you switch from the 0.6B to the 1.7B model, the previous weights are fully purged from the system RAM.
* **CPU Inference Strategy:**
* **Precision:** Forced `torch.float32` to maintain audio quality without CUDA-specific kernels.
* **Attention:** Uses `sdpa` (Scaled Dot-Product Attention) to optimize Transformer operations for CPU instruction sets.
* **Device Map:** Explicitly set to `"cpu"` to prevent the backend from searching for CUDA drivers.



---

## 🌓 Model Parameters & Clarity Modes

You can toggle between two primary parameter scales depending on your hardware and clarity needs:

### 1. 0.6B Mode (Efficiency)

* **RAM Footprint:** ~14GB - 16GB.
* **Performance:** Faster inference, ideal for standard TTS tasks.
* **Capability:** Reliable voice cloning and standard clarity.

### 2. 1.7B Mode (Clarity & Cloning)

* **RAM Footprint:** ~24GB.
* **Performance:** Higher computational overhead but significantly better results.
* **Cloning Accuracy:** Uses a larger latent space to capture the specific timbre and emotional nuances of a reference speaker.
* **Voice Clarity:** Superior high-frequency reconstruction, reducing "robotic" artifacts common in smaller models.

---

## 🚀 Features

* **Voice Design (Prompt-to-Speech):** Generate a unique voice by describing it (e.g., *"A calm, deep male voice with a professional tone"*).
* **Zero-Shot Voice Cloning:** Upload a 5-10 second audio clip to replicate any voice. Use the **1.7B mode** for the highest similarity.
* **Pre-defined Premium Speakers:** Includes 9 high-quality fixed personas (Aiden, Serena, Ryan, etc.).
* **Multilingual Support:** Supports 10 languages including English, Chinese, Japanese, French, Spanish, and more.

---

## 📦 Installation & Setup

### 1. Dependencies

Create a `requirements.txt` and install:

```text
torch>=2.0.0
gradio
numpy
huggingface_hub
qwen_tts

```

```bash
pip install -r requirements.txt

```

### 2. Authentication

This model uses Hugging Face weights. Ensure your token is set:

```bash
export HF_TOKEN="your_huggingface_token"

```

### 3. Execution

```bash
python app.py

```

---

## 📊 Hardware Requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| **System RAM** | 14GB (for 0.6B) | 24GB+ (for 1.7B) |
| **Processor** | 4-Core CPU | 8-Core+ CPU |
| **Storage** | 10GB Free Space | SSD (for fast weight swapping) |

---

## 🔗 Source

Based on the official Hugging Face Space: [https://huggingface.co/spaces/Qwen/Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS)

---
