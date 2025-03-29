import os

# Must happen before any `import torch`:
os.environ['PYTHONWARNINGS'] = "ignore::UserWarning:torch.cuda"

import torch
import platform
import psutil

def system_details():
    print("=" * 60)
    print(" SYSTEM INFORMATION REPORT ".center(60))
    print("=" * 60)

    # -------------------------------
    # 1) PyTorch and CUDA Info
    # -------------------------------
    print("\n[1] PYTORCH & CUDA INFO")
    print("-" * 60)
    print(f"PyTorch Version        : {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available         : {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA Devices : {device_count}")

        for i in range(device_count):
            print(f"\n  -- GPU {i} Details --")
            gpu_name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)

            print(f"  Device Name                   : {gpu_name}")
            print(f"  CUDA Capability (Major,Minor) : {capability}")
            print(f"  Total GPU Memory              : {props.total_memory / (1024 ** 2):.2f} MB")
            print(f"  Multi-Processor Count         : {props.multi_processor_count}")
            print(f"  Max Threads per Multi-Proc    : {props.max_threads_per_multi_processor}")
            print(f"  Warp Size                     : {props.warp_size}")
    else:
        print("No GPU information to show because CUDA is not available.")

    # -------------------------------
    # 2) CPU Info
    # -------------------------------
    print("\n[2] CPU INFO")
    print("-" * 60)
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 0
    print(f"Physical Cores : {physical_cores}")
    print(f"Logical Cores  : {logical_cores}")
    print(f"Max Frequency  : {cpu_freq:.2f} MHz")

    # -------------------------------
    # 3) Memory Info
    # -------------------------------
    print("\n[3] MEMORY INFO")
    print("-" * 60)
    memory = psutil.virtual_memory()
    total_ram = memory.total / (1024 ** 3)
    available_ram = memory.available / (1024 ** 3)
    used_ram = memory.used / (1024 ** 3)
    print(f"Total Memory    : {total_ram:.2f} GB")
    print(f"Available Memory: {available_ram:.2f} GB")
    print(f"Used Memory     : {used_ram:.2f} GB")
    print(f"Memory Usage    : {memory.percent}%")

    # -------------------------------
    # 4) Disk Usage Info
    # -------------------------------
    print("\n[4] DISK USAGE INFO")
    print("-" * 60)
    disk = psutil.disk_usage('/')
    total_disk = disk.total / (1024 ** 3)
    free_disk = disk.free / (1024 ** 3)
    print(f"Total Disk Space : {total_disk:.2f} GB")
    print(f"Used Disk Space  : {(total_disk - free_disk):.2f} GB")
    print(f"Free Disk Space  : {free_disk:.2f} GB")
    print(f"Disk Usage       : {disk.percent}%")

    # -------------------------------
    # 5) Operating System Info
    # -------------------------------
    print("\n[5] OS INFO")
    print("-" * 60)
    os_system = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    architecture = platform.architecture()[0]
    processor = platform.processor()
    print(f"Operating System : {os_system} {os_release} ({os_version})")
    print(f"Architecture     : {architecture}")
    print(f"Processor        : {processor}")

    print("\n" + "=" * 60)
    print(" END OF REPORT ".center(60))
    print("=" * 60)

if __name__ == "__main__":
    system_details()
