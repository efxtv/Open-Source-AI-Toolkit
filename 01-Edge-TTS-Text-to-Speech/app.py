import gradio as gr
import edge_tts
import asyncio
import tempfile
import os
import uuid

# Helper to get voices from Edge TTS
async def get_voices():
    try:
        voices = await edge_tts.list_voices()
        sorted_voices = sorted(voices, key=lambda x: x['ShortName'])
        return {f"{v['ShortName']} - {v['Locale']} ({v['Gender']})": v['ShortName'] for v in sorted_voices}
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return {"Default - en-US-GuyNeural (Male)": "en-US-GuyNeural"}

# Core TTS Logic
async def tts_interface(text, voice, rate, pitch):
    if not text or not text.strip():
        raise gr.Error("Please enter some text to convert!")
    
    if not voice:
        raise gr.Error("Please select a voice from the dropdown.")
    
    try:
        voice_short_name = voice.split(" - ")[0]
        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        
        filename = f"edge_tts_{uuid.uuid4().hex}.mp3"
        tmp_path = os.path.join(tempfile.gettempdir(), filename)
        
        communicate = edge_tts.Communicate(text, voice_short_name, rate=rate_str, pitch=pitch_str)
        await communicate.save(tmp_path)
        
        return tmp_path

    except Exception as e:
        raise gr.Error(f"TTS Generation failed: {str(e)}")

async def create_demo():
    voices_dict = await get_voices()
    voice_choices = list(voices_dict.keys())
    
    with gr.Blocks() as demo:
        gr.Markdown("# 🎙️ Microsoft Edge Text-to-Speech")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Input Text", 
                    lines=8, 
                    placeholder="Enter the text you want to convert to speech..."
                )
                
                voice_dropdown = gr.Dropdown(
                    choices=voice_choices, 
                    label="Select Voice", 
                    value=voice_choices[0] if voice_choices else None
                )
                
                with gr.Row():
                    rate_slider = gr.Slider(minimum=-100, maximum=100, value=0, label="Rate (%)", step=1)
                    pitch_slider = gr.Slider(minimum=-50, maximum=50, value=0, label="Pitch (Hz)", step=1)
                
                generate_btn = gr.Button("Generate Audio", variant="primary")
            
            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Resulting Audio", type="filepath")
                
        generate_btn.click(
            fn=tts_interface,
            inputs=[text_input, voice_dropdown, rate_slider, pitch_slider],
            outputs=[audio_output]
        )
        
    return demo

async def main():
    demo = await create_demo()
    # Using only the most essential arguments to avoid version conflicts
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False
    )

if __name__ == "__main__":
    asyncio.run(main())
