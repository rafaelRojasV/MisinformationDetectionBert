import os

# Must happen before any `import torch`:
os.environ['PYTHONWARNINGS'] = "ignore::UserWarning:torch.cuda"

import torch
import platform
import psutil
def system_details():
    # PyTorch version and CUDA details
    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        return  # No CUDA, so no GPU details to show

    # Number of GPUs available
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}\n")

    # For each GPU, print out its details
    for i in range(device_count):
        print(f"--- GPU {i} Details ---")
        print(f"Device Name: {torch.cuda.get_device_name(i)}")

        # Device capability (major, minor)
        capability = torch.cuda.get_device_capability(i)
        print(f"CUDA Capability (Major, Minor): {capability}")

        # Get full device properties
        props = torch.cuda.get_device_properties(i)
        print(f"Total GPU Memory: {props.total_memory / (1024 ** 2):.2f} MB")
        print(f"Multi-Processor Count: {props.multi_processor_count}")
        print(f"Max Threads per Multi-Processor: {props.max_threads_per_multi_processor}")
        print(f"Warp Size: {props.warp_size}")
        print()

    # System Details
    print(f"Operating System: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")

    # CPU details
    cpu_count = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max
    print(f"CPU: {cpu_count} physical cores, {logical_cores} logical cores, {cpu_freq} MHz")

    # Memory details
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {memory.used / (1024 ** 3):.2f} GB")
    print(f"Memory Usage: {memory.percent}%")

    # Disk details
    disk = psutil.disk_usage('/')
    print(f"Disk Usage: {disk.percent}% of {disk.total / (1024 ** 3):.2f} GB")
    print(f"Free Disk Space: {disk.free / (1024 ** 3):.2f} GB")


if __name__ == "__main__":
    system_details()
