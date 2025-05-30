import sys
import torch

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"\nNumber of available GPUs: {device_count}")

    for i in range(device_count):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")

        # Create random tensors to test GPU
        device = torch.device(f"cuda:{i}")
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)

        print("\nPerforming basic matrix multiplication on GPU:")
        print(f"x = \n{x}")
        print(f"y = \n{y}")
        print(f"x @ y = \n{x @ y}")

        # Clear memory
        del x, y
        torch.cuda.empty_cache()
else:
    print("CUDA is not available. Running on CPU only.")