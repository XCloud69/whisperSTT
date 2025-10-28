import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue

# Parameters
model_path = "/home/rashad/Desktop/python/whisperModels/faster-whisper-base"
device = "cpu"  # change to "cpu" if no GPU
sample_rate = 16000
block_duration = 5  # seconds

print("Loading model...")
model = WhisperModel(model_path, device=device)
print("Model loaded.")

audio_queue = queue.Queue()


def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())


print("Recording... Press Ctrl+C to stop.")

with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
    try:
        while True:
            audio_block = []
            for _ in range(int(sample_rate * block_duration / 1024)):
                audio_block.append(audio_queue.get())
            audio_data = np.concatenate(audio_block).flatten()

            # Transcribe the current audio block
            segments, _ = model.transcribe(audio_data, beam_size=1, language="en")
            for segment in segments:
                print(segment.text.strip())
    except KeyboardInterrupt:
        print("\nStopped.")
