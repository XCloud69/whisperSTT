import numpy as np
from faster_whisper import WhisperModel
import time
from Configuration import *


# begin timing
begin = time.perf_counter()

print("Loading model...")
model = WhisperModel(model_path, device=device)
print("Model loaded.\nExtracting and transcribing audio...")

# Use ffmpeg to decode audio from the file to 16kHz mono PCM
process = subprocess.Popen(
    [
        "ffmpeg",
        "-loglevel",
        "quiet",
        "-i",
        input_file,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "s16le",
        "-",
    ],
    stdout=subprocess.PIPE,
)

# Read all audio data
audio_bytes, _ = process.communicate()
audio_data = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

# Transcribe with faster-whisper
segments_iter, info = model.transcribe(audio_data, beam_size=1)

# Convert to a list so we can iterate multiple times
segments = list(segments_iter)

# Print results
print("## Transcription")
for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
print("\n---")

# Write to file
with open(file, "w", encoding="utf-8") as f:
    for s in segments:
        start_time = (
            getattr(s, "start", None) if not isinstance(s, dict) else s.get("start")
        )
        end_time = getattr(s, "end", None) if not isinstance(s, dict) else s.get("end")
        text = (
            s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
        ) or ""
        if start_time is not None and end_time is not None:
            f.write(f"({start_time:.2f}s → {end_time:.2f}s) {text}\n")
        else:
            f.write(f"{text}\n")
    f.write("\n---")

# End timing
final = time.perf_counter()

print(f"Execution time: {(final - begin) / 60:.2f} minutes")
