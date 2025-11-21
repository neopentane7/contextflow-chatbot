import torch

print("=" * 60)
print("GPU DETECTION TEST")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test tensor on GPU
    x = torch.rand(5, 3).cuda()
    print(f"\nTest tensor on GPU: {x.device}")
    print("✓ GPU is working correctly!")
else:
    print("\n✗ GPU not detected - will use CPU")

print("=" * 60)
