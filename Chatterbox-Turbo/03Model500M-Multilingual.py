import gradio as gr
import torch
import torchaudio as ta
import os
import sys
import warnings
from pathlib import Path

# FIXED FOR LINUX/WSL BASED CPU EFXTv t.me/efxtv

warnings.filterwarnings("ignore")

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

from chatterbox.tts import ChatterboxTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path(__file__).parent / "models" / "chatterbox"

print(f"[*] Hardware: {device.upper()}")
print(f"[*] Loading Standard (500M English)...")

with SuppressOutput():
    model = ChatterboxTTS.from_local(MODEL_PATH, device=device)

def generate(text, audio, exagg, cfg):
    wav = model.generate(text, audio_prompt_path=audio, exaggeration=exagg, cfg_weight=cfg)
    out_file = os.path.abspath("output_standard.wav")
    ta.save(out_file, wav, model.sr)
    return out_file

with gr.Blocks(title="Standard 500M") as demo:
    gr.Markdown("# 🎙️ Chatterbox Standard (500M)")
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Text", value="High quality narration.")
            ref = gr.Audio(type="filepath", label="Reference")
            ex = gr.Slider(0.25, 2, 0.5, label="Exaggeration")
            cf = gr.Slider(0.2, 1, 0.5, label="CFG/Pace")
            btn = gr.Button("Generate")
        with gr.Column():
            out = gr.Audio(label="Output", show_download_button=True)
    btn.click(generate, [txt, ref, ex, cf], out)

print("[*] Studio ready on http://0.0.0.0:8080")
demo.launch(server_port=8080, server_name="0.0.0.0")
(venv) demo@b8ce2453089e:~/chatterbox$ clear;cat 03Model500M-Multilingual.py 
