#!/usr/bin/env python3
"""
scripts/environment_setup.py

 - Generates a run_id and creates output directories.
 - Attempts to install PyTorch nightly with CUDA 12.8 (cu128).
 - Falls back to CPU-only PyTorch if that fails.
 - Installs packages from `requirements.txt`.
 - Prints GPU info if available.

Use: `python scripts/environment_setup.py`
"""

import os
import sys
import subprocess
import logging
import importlib
import uuid
from datetime import datetime

def run_command(cmd_list):
    """Run a command list via subprocess.check_call, logs on fail."""
    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd_list)}\n   Error: {e}")
        raise e

# 1) Generate run ID & directories
run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
print(f"🔸 Generated run_id => {run_id}")

run_dir = os.path.join("model_artifacts", f"run_{run_id}")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "visualizations"), exist_ok=True)
print(f"Created/using directory: {run_dir}")

# 2) Attempt to install PyTorch (nightly) for CUDA 12.8
print("🔸 Attempting to install PyTorch nightly (cu128)...")
pytorch_nightly_cmd = [
    sys.executable, "-m", "pip", "install", "--pre",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/nightly/cu128"
]

try:
    run_command(pytorch_nightly_cmd)
except Exception:
    print("⚠️ PyTorch CUDA 12.8 (cu128) install failed. Falling back to CPU-only PyTorch...")
    cpu_install_cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    run_command(cpu_install_cmd)

# 3) Basic logging
logging.basicConfig(level=logging.INFO)

# 4) Check PyTorch + CUDA
def check_cuda_version():
    import torch
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA is available!")
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        for i in range(device_count):
            print(f"--- GPU {i} Details ---")
            print(f"Device Name: {torch.cuda.get_device_name(i)}")
            major, minor = torch.cuda.get_device_capability(i)
            print(f"CUDA Capability (Major, Minor): ({major}, {minor})")
            mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
            print(f"Total GPU Memory: {mem:.2f} MB")
    else:
        print("CUDA NOT available. Using CPU-only PyTorch.")

try:
    check_cuda_version()
except ImportError:
    print("PyTorch not installed or no GPU available.")

# 5) If `requirements.txt` exists, install packages from it
requirements_file = 'requirements.txt'
if os.path.exists(requirements_file):
    print(f"🔸 Installing packages from {requirements_file}...")
    try:
        run_command([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"✅ Successfully installed packages from {requirements_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install from {requirements_file}: {e}")
else:
    print("⚠️ No requirements.txt found. Manually install your other dependencies if needed.")

print("\n✅ Finished environment setup. Ready for training.")

# If you need run_id / run_dir in train.py, for example:
__all__ = ["run_id", "run_dir"]
