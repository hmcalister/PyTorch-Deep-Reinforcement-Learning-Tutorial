import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def checkGPUMemory():
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)}")
        print(f"Cached:   {torch.cuda.memory_reserved(0)}")
        print()
    else:
        print("Device is not set to cuda")

X = torch.rand(10, device=device)
checkGPUMemory()

X = torch.rand(1000000, device=device)
checkGPUMemory()