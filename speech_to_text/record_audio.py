import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os

fs = 16000  # sample rate
seconds = 2  # record length in seconds

os.makedirs("data", exist_ok=True)
filename = input("Enter filename (e.g. yes.wav): ")

print("🎙️ Recording...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
write(f"data/{filename}", fs, recording)
print(f"✅ Saved as data/{filename}")
