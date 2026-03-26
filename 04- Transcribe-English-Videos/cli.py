import sys
import whisper

def get_subtitles(video_path):
    # Load the model (use "tiny" for speed, "base" for balance)
    model = whisper.load_model("base")
    
    print(f"--- Analyzing {video_path} (This stays 100% on your computer) ---")
    
    # Transcribe the video file directly
    result = model.transcribe(video_path)
    
    # Print the full transcript
    print("\n--- TRANSCRIPT / CC ---\n")
    print(result["text"])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py video.mp4")
    else:
        get_subtitles(sys.argv[1])
