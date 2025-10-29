import subprocess
import sys
import os


def detect_device():
    try:
        subprocess.check_output(["nvidia-smi"])
        return "cuda"
    except subprocess.CalledProcessError:
        return "cpu"


if len(sys.argv) < 2:
    print("Usage: python transcribe_file.py <audio_or_video_file>")
    sys.exit(1)

input_file = sys.argv[1]
# ====== Configuration ======
file_name = os.path.basename(input_file)
model_path = "/home/rashad/Desktop/python/whisperModels/faster-whisper-tiny"
device = "cpu"
sample_rate = 16000
save_path = "/home/rashad/Desktop/"
file = save_path + "/" + file_name + ".md"
# ===========================
