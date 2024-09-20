import torch

# Function to print GPU utilization and memory statistics
def print_gpu_utilization():
    # Calculate GPU utilization percentage
    utilization = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
    # Get allocated and cached GPU memory in GB
    allocated, cached = torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9
    
    # Print GPU utilization and memory statistics
    print(f"GPU utilization: {utilization:.2f}% / Allocated GPU memory: {allocated:.2f} GB / Cached GPU memory: {cached:.2f} GB")
    
    # Return a dictionary with GPU statistics
    return {"utilization": utilization, "allocated": allocated, "cached": cached}

# Function to get device information
def get_device_info():
    # Determine the device (CUDA if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize info string with device type
    info = f"Device used: {device}\n"
    # If CUDA is available, add GPU-specific information
    if device.type == 'cuda':
        info += f"GPU model: {torch.cuda.get_device_name(0)}, "
        info += f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB, "
        info += f"CUDA version: {torch.version.cuda}"
    return info
