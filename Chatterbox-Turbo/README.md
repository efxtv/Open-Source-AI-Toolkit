## 🎙️ [Chatterbox TTS Best Voice Cloning Model, Optimized by EFXtv](https://github.com/resemble-ai/chatterbox?tab=readme-ov-file)
This toolkit is powered by the Chatterbox family of models by Resemble AI. Each file serves a specific purpose, from real-time speed to high-fidelity storytelling.
1. Model Specifications & RAM Usage
> Note: RAM consumption is estimated based on float32 precision (standard for CPU). If using a GPU, these numbers represent VRAM usage.
> 
| File Name | Model Core | Parameters | Estimated RAM | Key Strengths |
|---|---|---|---|---|
| 01Model350M-FastEnglish.py | Turbo | 350 Million | ~1.5 GB | Extreme speed; Supports [chuckle], [laugh] tags. |
| 02Model500M-EnglishBest.py | Standard | 500 Million | ~2.2 GB | Highest English quality; Best for short, clear clips. |
| 03Model500M-Multilingual.py | Multi | 500 Million | ~2.4 GB | Supports 23+ languages (Hindi, French, etc.). |
| 04LongStories500M.py | Standard | 500 Million | ~2.2 GB* | Optimized for stability during long narrations. |
| app.py | Master | All above | ~17 GB* | Optimized for CPU having atleast 16+8GB RAM. |
* RAM remains stable at ~2.2GB because it processes the story sentence-by-sentence rather than all at once.

## Detailed File Breakdown
- As these models reside entirely within the system memory during operation, simultaneous execution may exceed the hardware limitations of standard consumer workstations. For environments with limited RAM, it is recommended to load models sequentially to ensure optimal stability and performance.

- Each model in this suite is designed to be loaded into Random Access Memory (RAM) or Video RAM (VRAM) to ensure high-speed inference. For users operating on hardware with restricted memory capacity, please note the following:

- **Sequential Model Loading**: To prevent system instability or 'Out of Memory' (OOM) errors, users should execute scripts individually rather than concurrently. Closing a model session fully releases the allocated memory back to the system.

- **Hardware Accessibility**: While professional-grade AI often requires significant memory overhead, this toolkit utilizes optimized architectures (350M–500M parameters) to remain accessible. However, system constraints should still be monitored via the Task Manager or Activity Monitor during generation.

- **Scaling Performance**: For long-form narration or multilingual tasks, the application prioritizes memory efficiency by processing data in discrete segments, making professional-quality TTS viable on mid-range hardware.

##  01Model350M-FastEnglish.py (Turbo)
The "Speedster." This model uses a distilled 1-step diffusion process, making it nearly instant even on modest CPUs.
 * Unique Feature: Paralinguistic Tags. You can insert markers like [gasp] or [clear throat] directly into the text.
 * Best For: Real-time assistants, quick memes, and interactive bots.
##  02Model500M-EnglishBestQuality.py (Standard)
The "Narrator." It uses a more complex 10-step diffusion process which takes longer but results in much richer, more human-like tones.
 * Unique Feature: High-fidelity English cloning with deep emotion.
 * Best For: Professional voice-overs and high-quality content.
##  03Model500M-Multilingual.py (Global)
The "Polyglot." Designed for global reach without losing the character of the cloned voice.
 * Unique Feature: Cross-language cloning (e.g., clone an English voice and make it speak perfect Hindi or Japanese).
 * Accuracy Tip: Keep CFG Weight at 0.7+ to prevent the model from adding "gibberish" at the end of clips.
##  04LongStories500M.py (Long-Form)
The "Author." Standard models often "hallucinate" or lose their voice after 20 seconds. This file fixes that.
 * Unique Feature: Auto-Chunking. It splits your 1,000-word story into individual sentences, generates them, and merges them into one long .wav.
 * Best For: Audiobooks and long YouTube scripts.
### 3. Hardware Management Tips
 * The "Address in Use" Error: Since all these files use port 8080, you must close one before opening another. Use Ctrl+C in your terminal to "kill" the active model.
 * Memory Leak Prevention: We included a SuppressOutput class and torch.inference_mode() in these files. This ensures that after a voice is generated, the memory is cleared so your computer doesn't slow down over time.
 * CPU Optimization: If your generations feel slow, lower the Temperature (for Turbo) or Exaggeration (for Standard). This reduces the complexity of the math the CPU has to perform.
 ---
### **Memory Allocation and Resource Management**
As these models reside entirely within the system memory during operation, simultaneous execution may exceed the hardware limitations of standard consumer workstations. For environments with limited RAM, it is recommended to load models sequentially to ensure optimal stability and performance.

### **Technical Support**
Should you encounter technical anomalies or wish to submit optimization proposals, please direct your inquiries to the **[EFXTV](https://t.me/efxtv)**.
