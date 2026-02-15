import gradio as gr
import os
import time

# --- PYTHON 3.12 COMPATIBILITY FIX ---
try:
    import spacy_pkuseg as pkuseg
    import sys
    sys.modules['pkuseg'] = pkuseg  # Redirects any 'import pkuseg' to use spacy_pkuseg
except ImportError:
    # If you haven't installed spacy-pkuseg yet, run: pip install spacy-pkuseg
    pass
# --------------------------------------

from voice_cloning_engine import VoiceCloningManager

# Initialize the manager
manager = VoiceCloningManager()

def clone_and_generate(text, reference_audio, exaggeration, cfg_scale, temperature, seed):
    """Gradio wrapper for voice cloning and generation"""
    if not text or not reference_audio:
        return None, "Error: Please provide both text and a reference audio file."
    
    # Ensure system is initialized
    if not manager.chatterbox_loaded:
        print("🔄 Initializing Chatterbox for Gradio...")
        success = manager.initialize_chatterbox()
        if not success:
            return None, "Error: Failed to initialize AI model. Check logs for details."

    # Unique output path
    timestamp = int(time.time())
    output_path = f"temp_outputs/hf_gen_{timestamp}.wav"
    os.makedirs("temp_outputs", exist_ok=True)

    try:
        # Generate speech directly
        result_path = manager.generate_speech_direct(
            text=text,
            reference_audio_path=reference_audio,
            output_path=output_path,
            exaggeration=exaggeration,
            cfg_scale=cfg_scale,
            temperature=temperature,
            seed=int(seed)
        )
        
        if result_path and os.path.exists(result_path):
            return result_path, "Success! Audio generated."
        else:
            return None, "Error: Speech generation failed."
            
    except Exception as e:
        return None, f"Critical Error: {str(e)}"

# Define the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Voice Cloning Studio (Hugging Face Edition)")
    gr.Markdown("Clone any voice in seconds using the Chatterbox TTS engine.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter the text you want the AI to speak...",
                lines=5
            )
            audio_input = gr.Audio(
                label="Reference Audio (The voice you want to clone)",
                type="filepath"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                exaggeration = gr.Slider(
                    label="Exaggeration",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                    info="Higher values make the emotion more intense."
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                    info="Controls how closely it follows the reference voice."
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=0.8,
                    info="Higher values make the voice more experimental/varied."
                )
                seed = gr.Number(
                    label="Seed",
                    value=0,
                    precision=0,
                    info="Set to a specific number for reproducible results."
                )
            
            generate_btn = gr.Button("🚀 Generate Cloned Voice", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Generated Result")
            status_output = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=clone_and_generate,
        inputs=[text_input, audio_input, exaggeration, cfg_scale, temperature, seed],
        outputs=[audio_output, status_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("### How it works\n1. Upload a clear 10-30 second audio clip of a voice.\n2. Type the text you want them to say.\n3. Click Generate and wait for the AI to process.")

# Launch the app
if __name__ == "__main__":
    # server_name="0.0.0.0" makes it accessible externally
    # server_port=7860 is the default, but you can change it to any port (e.g. 8080)
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True
    )
