import subprocess
import sys
import os
import platform


def detect_device():
    """
    Detect the best available device for inference.
    Supports: NVIDIA CUDA, AMD ROCm, Intel GPUs, Apple Silicon
    Falls back to CPU if no accelerator is found.
    """
    system = platform.system()

    # Check for NVIDIA CUDA
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        print("✓ NVIDIA CUDA device detected")
        return "cuda"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Check for AMD ROCm
    try:
        subprocess.check_output(["rocm-smi"], stderr=subprocess.DEVNULL)
        print("✓ AMD ROCm device detected")
        return "cuda"  # faster-whisper uses "cuda" for ROCm
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Check for AMD GPUs on Windows (alternative method)
    if system == "Windows":
        try:
            result = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if "AMD" in result or "Radeon" in result:
                print("✓ AMD GPU detected (Windows)")
                return "cuda"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Check for Intel GPUs (Arc, Iris, etc.)
    try:
        # Intel oneAPI/OpenVINO detection
        if system == "Linux":
            result = subprocess.check_output(
                ["lspci"], stderr=subprocess.DEVNULL, text=True
            )
            if "Intel" in result and ("VGA" in result or "Display" in result):
                print("✓ Intel GPU detected")
                # Note: faster-whisper may need special setup for Intel GPUs
                return "cpu"  # Use CPU unless you have Intel GPU support configured
        elif system == "Windows":
            result = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if "Intel" in result and (
                "Arc" in result or "Iris" in result or "UHD" in result
            ):
                print("✓ Intel GPU detected")
                return "cpu"  # Use CPU unless you have Intel GPU support configured
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Check for Apple Silicon (M1, M2, M3, etc.)
    if system == "Darwin":
        machine = platform.machine()
        if machine == "arm64":
            print("✓ Apple Silicon detected (M-series chip)")
            return "cpu"  # faster-whisper uses CPU backend on Apple Silicon
        else:
            print("✓ macOS Intel detected")

    # Fallback to CPU
    print("ℹ Using CPU (no GPU accelerator detected)")
    return "cpu"


if len(sys.argv) < 2:
    print("Usage: python transcribe_file.py <audio_or_video_file>")
    sys.exit(1)

input_file = sys.argv[1]
# ====== Configuration ======
file_name = os.path.basename(input_file)
model_path = "./whisperModels/faster-whisper-tiny"  # edit
device = detect_device()
sample_rate = 16000
save_path = "."  # edit
file = save_path + "/" + file_name + ".md"
# ===========================
