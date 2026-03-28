from transformers import pipeline
import sounddevice as sd
from scipy.io.wavfile import write
import os

# -------------------------------
# STEP 1: Record audio
# -------------------------------
def record_audio(filename="test_audio.wav", duration=5, samplerate=16000):
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    print(f"🎙️ Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(filepath, samplerate, recording)
    print(f"✅ Saved recording as: {filepath}")
    return filepath

# -------------------------------
# STEP 2: Transcribe using Whisper model
# -------------------------------
def transcribe_audio(file_path):
    print("🧠 Loading Whisper model from Hugging Face...")
    stt = pipeline("automatic-speech-recognition", model="openai/whisper-base")

    print(f"🎧 Transcribing: {file_path}")
    result = stt(file_path)

    print("\n🗣️ Transcription Result:")
    print("------------------------")
    print(result["text"])
    print("------------------------")
    return result["text"]

# -------------------------------
# STEP 3: Main
# -------------------------------
if __name__ == "__main__":
    audio_path = record_audio("test_audio.wav", duration=5)
    transcribe_audio(audio_path)

