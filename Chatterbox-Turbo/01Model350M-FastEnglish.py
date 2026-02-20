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

from chatterbox.tts_turbo import ChatterboxTurboTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
TURBO_PATH = Path(__file__).parent / "models" / "chatterbox-turbo"

print(f"[*] Hardware: {device.upper()}")
print(f"[*] Loading Turbo (350M English)...")

with SuppressOutput():
    model = ChatterboxTurboTTS.from_local(str(TURBO_PATH), device)

def generate(text, audio, seed, temp):
    if seed != 0: torch.manual_seed(int(seed))
    # Note: Use absolute path to ensure browser can find it for download
    wav = model.generate(text, audio_prompt_path=audio, temperature=temp)
    out_file = os.path.abspath("output_turbo.wav")
    ta.save(out_file, wav, model.sr)
    return out_file

# --- UI DEFINITION ---
with gr.Blocks(title="Turbo 350M", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Chatterbox Turbo (350M)")
    
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Text", value="I'm super fast! [chuckle]", lines=4)
            
            # --- PARALINGUISTIC TAG BUTTONS ---
            gr.Markdown("### Paralinguistic Tags (Click to add)")
            tags = [
                "[chuckle]", "[laugh]", "[sigh]", "[gasp]", 
                "[clear throat]", "[cough]", "[sniff]", 
                "[groan]", "[shush]", "[mumble]"
            ]
            
            with gr.Row():
                for tag in tags:
                    tag_btn = gr.Button(tag, size="sm", min_width=10)
                    # This appends the tag to the current text in the box
                    tag_btn.click(lambda current_text, t=tag: current_text + f" {t} ", inputs=[txt], outputs=[txt])
            
            ref = gr.Audio(type="filepath", label="Reference")
            btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            out = gr.Audio(label="Output", show_download_button=True)
            sd = gr.Number(label="Seed", value=0)
            tm = gr.Slider(0.1, 1.5, 0.8, label="Temp")

    btn.click(generate, [txt, ref, sd, tm], out)

print("[*] Studio ready on http://0.0.0.0:8080")
demo.launch(server_port=8080, server_name="0.0.0.0")
