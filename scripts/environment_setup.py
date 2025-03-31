#!/usr/bin/env python3
"""
scripts/environment_setup.py

 - Installs or upgrades PyTorch (nightly CUDA 12.8 if possible).
 - Falls back to CPU-only PyTorch if that fails.
 - Installs packages from `requirements.txt`.
 - Checks and logs GPU info if available.
 - No run_id or run_dir generation here, purely environment setup.

Usage:
    python environment_setup.py
"""

import os
import sys
import subprocess
import logging

# Ensure stdout/stderr encode UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# --------------
# Import your rotating logger setup
# --------------
from src.logger_setup import setup_logging


def run_command(cmd_list):
    """Run a command list via subprocess.check_call, logs on fail."""
    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd_list)}\n   Error: {e}")
        raise e


def setup_pytorch_cuda128():
    """Try installing PyTorch nightly (cu128). If it fails, fallback to CPU-only."""
    import logging
    import sys
    logging.info("🔸 Attempting to install PyTorch nightly (cu128) ...")

    # We'll suppress pip install output with "-qq"
    pytorch_nightly_cmd = [
        sys.executable, "-m", "pip", "install", "--pre", "-qq",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/nightly/cu128"
    ]
    try:
        run_command(pytorch_nightly_cmd)
    except Exception:
        logging.warning("⚠️ PyTorch CUDA 12.8 (cu128) install failed. Falling back to CPU-only PyTorch...")
        cpu_install_cmd = [
            sys.executable, "-m", "pip", "install", "-qq",
            "torch", "torchvision", "torchaudio"
        ]
        run_command(cpu_install_cmd)


def check_cuda_version():
    """Log some CUDA/GPU info if available."""
    import torch
    logging.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info("CUDA is available!")
        device_count = torch.cuda.device_count()
        logging.info(f"Number of CUDA devices: {device_count}")
        for i in range(device_count):
            logging.info(f"--- GPU {i} Details ---")
            logging.info(f"Device Name: {torch.cuda.get_device_name(i)}")
            major, minor = torch.cuda.get_device_capability(i)
            logging.info(f"CUDA Capability (Major, Minor): ({major}, {minor})")
            mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
            logging.info(f"Total GPU Memory: {mem:.2f} MB")
    else:
        logging.warning("CUDA NOT available. Using CPU-only PyTorch.")


def install_requirements_if_exists():
    """Install packages from `requirements.txt` if found."""
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        logging.info(f"🔸 Installing packages from {requirements_file} (quietly)...")
        try:
            run_command([sys.executable, "-m", "pip", "install", "-qq", "-r", requirements_file])
            logging.info(f"✅ Successfully installed packages from {requirements_file} (quiet mode).")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Failed to install from {requirements_file}: {e}")
    else:
        logging.warning("⚠️ No requirements.txt found. Manually install dependencies if needed.")


def main():
    # ------------------------------------------------------------------
    # 1) Initialize logging to console or a simple file if you prefer
    # ------------------------------------------------------------------
    # (We can log to console with basic settings or use rotating logs.)
    setup_logging(log_file=None, log_level="INFO")  # logs to console only
    logging.info("🛠 Environment setup script started (no run directories).")

    # ------------------------------------------------------------------
    # 2) Install PyTorch nightly with CUDA 12.8 if possible
    # ------------------------------------------------------------------
    setup_pytorch_cuda128()

    # ------------------------------------------------------------------
    # 3) Check CUDA version
    # ------------------------------------------------------------------
    try:
        check_cuda_version()
    except ImportError:
        logging.warning("PyTorch not installed or no GPU available.")

    # ------------------------------------------------------------------
    # 4) Install from requirements.txt if present
    # ------------------------------------------------------------------
    install_requirements_if_exists()

    logging.info("✅ Environment setup complete. You can now run train_selective_suppression.py.")

if __name__ == "__main__":
    main()
