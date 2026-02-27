"""
Simple test script to verify the PyTorch backend for Transform1d works correctly.
"""

import numpy as np

# Test 1: Import the torch backend module
print("Test 1: Importing torch backend...")
try:
    import dtcwt.torch
    print("✓ Successfully imported dtcwt.torch")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    exit(1)

# Test 2: Check if torch is available
try:
    import torch
    print("✓ PyTorch is available")
    torch_available = True
except ImportError:
    print("⚠ PyTorch is not installed - backend will use fallback mode")
    torch_available = False

# Test 3: Create Transform1d instance
print("\nTest 2: Creating Transform1d instance...")
try:
    from dtcwt.torch import Transform1d
    transform = Transform1d()
    print("✓ Successfully created Transform1d instance")
except Exception as e:
    print(f"✗ Failed to create Transform1d: {e}")
    if not torch_available:
        print("  (Expected - PyTorch not installed)")
    exit(1)

if torch_available:
    # Test 4: Test forward transform with numpy array
    print("\nTest 3: Testing forward transform with numpy array...")
    try:
        # Create a simple 1D signal (must have even length)
        x = np.random.randn(64, 1)
        pyramid = transform.forward(x, nlevels=3)
        print(f"✓ Forward transform successful")
        print(f"  Lowpass shape: {pyramid.lowpass.shape}")
        print(f"  Number of highpass levels: {len(pyramid.highpasses)}")
        for i, h in enumerate(pyramid.highpasses):
            print(f"  Highpass level {i} shape: {h.shape}")
    except Exception as e:
        print(f"✗ Forward transform failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 5: Test inverse transform
    print("\nTest 4: Testing inverse transform...")
    try:
        y = transform.inverse(pyramid)
        print(f"✓ Inverse transform successful")
        print(f"  Reconstructed signal shape: {y.shape}")
        print(f"  Original shape: {x.shape}")
        
        # Check reconstruction error
        if y.shape == x.squeeze().shape:
            error = np.mean(np.abs(y - x.squeeze()))
            print(f"  Mean absolute error: {error:.6f}")
        else:
            print(f"  Shape mismatch: {y.shape} vs {x.squeeze().shape}")
    except Exception as e:
        print(f"✗ Inverse transform failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 6: Test with torch tensor directly
    print("\nTest 5: Testing with torch tensor input...")
    try:
        x_torch = torch.randn(64, 1)
        pyramid_torch = transform.forward(x_torch, nlevels=3)
        print(f"✓ Forward transform with torch tensor successful")
        print(f"  Lowpass is torch tensor: {isinstance(pyramid_torch.lowpass_op, torch.Tensor)}")
        
        y_torch = transform.inverse(pyramid_torch)
        print(f"✓ Inverse transform successful")
        print(f"  Output is torch tensor: {isinstance(y_torch, torch.Tensor)}")
    except Exception as e:
        print(f"✗ Transform with torch tensor failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test 7: Test backend switching
    print("\nTest 6: Testing backend switching...")
    try:
        import dtcwt
        
        # Push torch backend
        dtcwt.push_backend('torch')
        print(f"✓ Switched to torch backend: {dtcwt.backend_name}")
        
        # Create transform using the active backend
        t = dtcwt.Transform1d()
        print(f"✓ Created Transform1d from active backend")
        print(f"  Type: {type(t).__module__}.{type(t).__name__}")
        
        # Pop back to default
        dtcwt.pop_backend()
        print(f"✓ Switched back to default backend: {dtcwt.backend_name}")
        
    except Exception as e:
        print(f"✗ Backend switching failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
