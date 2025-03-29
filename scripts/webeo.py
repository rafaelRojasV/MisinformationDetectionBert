import torch

def gpu_details():
    print(f"PyTorch version: {torch.version}")

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
        # This line is no longer valid in latest PyTorch:
        # print(f"Max Threads per Block: {props.max_threads_per_block}")
        print(f"Warp Size: {props.warp_size}")
        print()

if __name__ == "__main__":
    gpu_details()