import gradio as gr
import torch
import torchaudio as ta
import os
import sys
import warnings
import re
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
print(f"[*] Loading Standard (500M English) for Long-Form Narration...")

with SuppressOutput():
    model = ChatterboxTTS.from_local(MODEL_PATH, device=device)

def split_text(text):
    # Splits text by punctuation to handle long stories sentence-by-sentence
    sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
    return [s.strip() for s in sentences if len(s.strip()) > 2]

def generate_long_story(text, audio_ref, exagg, cfg, progress=gr.Progress()):
    try:
        chunks = split_text(text)
        combined_wavs = []
        
        progress(0, desc="Starting narration...")
        for i, sentence in enumerate(chunks):
            progress((i + 1) / len(chunks), desc=f"Processing sentence {i+1} of {len(chunks)}")
            
            with torch.inference_mode():
                wav = model.generate(
                    sentence, 
                    audio_prompt_path=audio_ref if audio_ref else None,
                    exaggeration=exagg,
                    cfg_weight=cfg
                )
                combined_wavs.append(wav)
        
        # Merge all sentences into one audio tensor
        final_wav = torch.cat(combined_wavs, dim=-1)
        
        out_file = os.path.abspath("long_story_output.wav")
        ta.save(out_file, final_wav, model.sr)
        return out_file
    except Exception as e:
        return str(e)

# --- UI ---
with gr.Blocks(title="Long Story Narrator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📖 Long-Form Story Narrator (500M Standard)")
    gr.Markdown("This version automatically splits long text into sentences for stable, high-quality narration.")
    
    with gr.Row():
        with gr.Column(scale=2):
            story_input = gr.Textbox(
                label="Paste your story here", 
                placeholder="Once upon a time...", 
                lines=15
            )
            ref = gr.Audio(type="filepath", label="Voice Character Reference")
            
        with gr.Column(scale=1):
            out = gr.Audio(label="Final Full Narration", show_download_button=True)
            exagg = gr.Slider(0.25, 2.0, 0.5, label="Emotional Exaggeration")
            cfg = gr.Slider(0.1, 1.0, 0.5, label="Pace/Consistency (CFG)")
            btn = gr.Button("NARRATE ENTIRE STORY", variant="primary", size="lg")
            
    btn.click(generate_long_story, [story_input, ref, exagg, cfg], out)



print("[*] Narrator ready on http://0.0.0.0:8080")
demo.launch(server_port=8080, server_name="0.0.0.0")
