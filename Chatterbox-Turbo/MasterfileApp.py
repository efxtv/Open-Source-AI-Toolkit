import gradio as gr
import torch
import torchaudio as ta
import os
import sys
import traceback
import warnings
from pathlib import Path

# FIXED FOR LINUX/WSL BASED CPU EFXTv t.me/efxtv

# --- HIDE DEPRECATION WARNINGS ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- THE MAGIC CPU FIX ---
original_load = torch.load
def cpu_load(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    return original_load(*args, **kwargs)
torch.load = cpu_load

# --- TERMINAL CLEANER ---
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

torch.set_default_dtype(torch.float32)

from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Hardware detected: {device.upper()}")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "chatterbox"
TURBO_PATH = BASE_DIR / "models" / "chatterbox-turbo"

print("[*] Loading models into memory from local storage...")

try:
    print("    -> Loading Turbo (350M English)...")
    with SuppressOutput():
        model_turbo = ChatterboxTurboTTS.from_local(str(TURBO_PATH), device)
    
    print("    -> Loading Standard (500M English)...")
    with SuppressOutput():
        model_standard = ChatterboxTTS.from_local(MODEL_PATH, device=device)
    
    print("    -> Loading Multilingual (500M 23-Languages)...")
    with SuppressOutput():
        model_multi = ChatterboxMultilingualTTS.from_local(MODEL_PATH, device=device)
    
    print("[*] SUCCESS: All models loaded locally!")
except Exception as e:
    print(f"[!] Loading failed: {e}")
    traceback.print_exc()
    exit()

# --- GENERATION FUNCTIONS ---
def set_seed(seed):
    if seed != 0:
        torch.manual_seed(int(seed))

def generate_turbo(text, reference_audio, seed, temp, top_p, top_k, rep_pen, min_p, norm_loudness):
    print("[*] Generating with Turbo...")
    try:
        set_seed(seed)
        audio_path = reference_audio if reference_audio else None
        with torch.inference_mode():
            wav = model_turbo.generate(text, audio_prompt_path=audio_path, temperature=temp, top_p=top_p, top_k=int(top_k), repetition_penalty=rep_pen, min_p=min_p, norm_loudness=norm_loudness)
        output_filename = os.path.abspath("efxtv_turbo.wav")
        ta.save(output_filename, wav, model_turbo.sr)
        return output_filename
    except Exception as e:
        traceback.print_exc()
        return None

def generate_standard(text, reference_audio, exagg, cfg, seed, temp, vad_trim):
    print("[*] Generating with Standard...")
    try:
        set_seed(seed)
        audio_path = reference_audio if reference_audio else None
        with torch.inference_mode():
            wav = model_standard.generate(text, audio_prompt_path=audio_path, exaggeration=exagg, cfg_weight=cfg, temperature=temp)
        output_filename = os.path.abspath("efxtv_standard.wav")
        ta.save(output_filename, wav, model_standard.sr)
        return output_filename
    except Exception as e:
        traceback.print_exc()
        return None

def generate_multi(text, lang, reference_audio, exagg, cfg, seed, temp):
    print(f"[*] Generating with Multilingual ({lang})...")
    try:
        set_seed(seed)
        audio_path = reference_audio if reference_audio else None
        with torch.inference_mode():
            wav = model_multi.generate(text, audio_prompt_path=audio_path, language_id=lang, exaggeration=exagg, cfg_weight=cfg, temperature=temp)
        output_filename = os.path.abspath("efxtv_multi.wav")
        ta.save(output_filename, wav, model_multi.sr)
        return output_filename
    except Exception as e:
        traceback.print_exc()
        return None

# --- UI DEFINITION ---
with gr.Blocks(title="Chatterbox Fix by EFXTv all 3 models", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Chatterbox Turbo Linux Version By EFXTv")
    
    default_seed = gr.Number(value=0, visible=False)
    default_temp = gr.Number(value=0.8, visible=False)
    default_vad = gr.Checkbox(value=False, visible=False)

    with gr.Tab("Chatterbox Turbo ParalinguisticTAGS "):
        with gr.Row():
            with gr.Column(scale=1):
                t_text = gr.Textbox(label="Text", lines=3, value="Oh, that's hilarious! [chuckle]")
                tags = ["[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]", "[sniff]", "[gasp]", "[chuckle]", "[laugh]"]
                with gr.Row():
                    for tag in tags:
                        btn = gr.Button(tag, size="sm", min_width=10)
                        btn.click(lambda current, t=tag: current + t + " ", inputs=[t_text], outputs=[t_text])
                t_ref = gr.Audio(type="filepath", label="Reference Audio")
                t_btn = gr.Button("Generate ⚡", variant="primary")
            with gr.Column(scale=1):
                # FIXED: show_download_button=True
                t_out = gr.Audio(label="Output Audio", show_download_button=True)
                with gr.Accordion("Options", open=True):
                    t_seed = gr.Number(label="Seed", value=0)
                    t_temp = gr.Slider(0.05, 2, 0.8, label="Temp")
                    t_top_p = gr.Slider(0, 1, 0.95, label="Top P")
                    t_top_k = gr.Slider(0, 1000, 1000, label="Top K")
                    t_rep = gr.Slider(1, 2, 1.2, label="Rep Penalty")
                    t_min_p = gr.Slider(0, 1, 0, label="Min P")
                    t_norm = gr.Checkbox(label="Normalize", value=True)
        t_btn.click(fn=generate_turbo, inputs=[t_text, t_ref, t_seed, t_temp, t_top_p, t_top_k, t_rep, t_min_p, t_norm], outputs=t_out)

    with gr.Tab("Chatterbox-Multilingual-TTS"):
        with gr.Row():
            with gr.Column(scale=1):
                m_text = gr.Textbox(label="Text", lines=3, value="Multilingual tts yes it supports 23 plua languages.")
                m_lang = gr.Dropdown(choices=["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"], value="fr", label="Language")
                m_ref = gr.Audio(type="filepath", label="Reference")
                m_exagg = gr.Slider(0.25, 2, 0.5, label="Exaggeration")
                m_cfg = gr.Slider(0.2, 1, 0.5, label="CFG/Pace")
                m_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                # FIXED: show_download_button=True
                m_out = gr.Audio(label="Output Audio", show_download_button=True)
        m_btn.click(fn=generate_multi, inputs=[m_text, m_lang, m_ref, m_exagg, m_cfg, default_seed, default_temp], outputs=m_out)

    with gr.Tab("Chatterbox Best Exaggeration"):
        with gr.Row():
            with gr.Column(scale=1):
                s_text = gr.Textbox(label="Text", lines=3, value="Please subscribe EFXTv, Try Exaggeration.")
                s_ref = gr.Audio(type="filepath", label="Reference")
                s_exagg = gr.Slider(0.25, 2, 0.5, label="Exaggeration")
                s_cfg = gr.Slider(0.2, 1, 0.5, label="CFG/Pace")
                s_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                # FIXED: show_download_button=True
                s_out = gr.Audio(label="Output Audio", show_download_button=True)
        s_btn.click(fn=generate_standard, inputs=[s_text, s_ref, s_exagg, s_cfg, default_seed, default_temp, default_vad], outputs=s_out)

if __name__ == "__main__":
    print("[*] Starting EFXTv studio on http://0.0.0.0:8080 ...")
    demo.launch(server_name="0.0.0.0", server_port=8080)
