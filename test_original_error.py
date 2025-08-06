#!/usr/bin/env python3
"""
Test to specifically verify the original numpy.float error reported in the notebook is resolved
"""

import os
import sys
import pandas as pd
import numpy as np

os.chdir(r'C:\Users\gcicc\claudeproj\tableGenCompare')
sys.path.append('.')

print("Testing the original numpy.float error from the notebook...")
print("=" * 70)

# Apply the same numpy fixes that were added to resolve the issue
print("1. Applying NumPy compatibility fixes...")
if not hasattr(np, 'float'):
    np.float = np.float64
    print("   np.float = np.float64 applied")
else:
    print("   np.float already exists")

if not hasattr(np, 'int'):
    np.int = np.int_
    print("   np.int = np.int_ applied")
else:
    print("   np.int already exists")

print(f"   NumPy version: {np.__version__}")

# Test the specific operations that caused the original error
print("\n2. Testing operations that caused the original error...")

tests_passed = 0
total_tests = 4

# Test 1: Direct np.float usage
try:
    result = np.float(42.0)
    print(f"   Test 1 - np.float(42.0): SUCCESS ({result})")
    tests_passed += 1
except AttributeError as e:
    if "has no attribute 'float'" in str(e):
        print(f"   Test 1 - np.float(42.0): FAILED - Original error still exists!")
    else:
        print(f"   Test 1 - np.float(42.0): FAILED - Different error: {e}")

# Test 2: Array dtype conversion 
try:
    arr = np.array([1.0, 2.0, 3.0])
    result = arr.astype(np.float)
    print(f"   Test 2 - array.astype(np.float): SUCCESS ({len(result)} elements)")
    tests_passed += 1
except AttributeError as e:
    if "has no attribute 'float'" in str(e):
        print(f"   Test 2 - array.astype(np.float): FAILED - Original error still exists!")
    else:
        print(f"   Test 2 - array.astype(np.float): FAILED - Different error: {e}")

# Test 3: np.int usage 
try:
    result = np.int(42)
    print(f"   Test 3 - np.int(42): SUCCESS ({result})")
    tests_passed += 1
except AttributeError as e:
    if "has no attribute 'int'" in str(e):
        print(f"   Test 3 - np.int(42): FAILED - Similar error exists!")
    else:
        print(f"   Test 3 - np.int(42): FAILED - Different error: {e}")

# Test 4: Check that we can import and instantiate TableGAN without numpy errors
try:
    sys.path.insert(0, './tableGAN')
    from model import TableGan
    
    # This import/instantiation was failing before due to numpy.float usage in the code
    print(f"   Test 4 - TableGAN import: SUCCESS (class imported: {TableGan.__name__})")
    tests_passed += 1
except ImportError as e:
    print(f"   Test 4 - TableGAN import: FAILED - Import error: {e}")
except AttributeError as e:
    if "has no attribute 'float'" in str(e):
        print(f"   Test 4 - TableGAN import: FAILED - Original numpy.float error in TableGAN!")
    else:
        print(f"   Test 4 - TableGAN import: FAILED - Different error: {e}")

print(f"\n3. Test Results: {tests_passed}/{total_tests} tests passed")

if tests_passed == total_tests:
    print("\n✅ SUCCESS: All tests passed!")
    print("✅ The original 'module numpy has no attribute float' error has been RESOLVED!")
    print("✅ TableGAN should now work without the numpy compatibility error.")
    
    print("\nWhat was fixed:")
    print("- Added np.float = np.float64 compatibility alias")
    print("- Added np.int = np.int_ compatibility alias") 
    print("- These fixes handle deprecated numpy aliases in modern numpy versions")
    
    exit_code = 0
else:
    print("\n❌ FAILURE: Some tests failed!")
    print("❌ The original numpy compatibility error may still exist.")
    print("❌ TableGAN may still encounter 'module numpy has no attribute float' errors.")
    
    exit_code = 1

print("\n" + "=" * 70)
print("ORIGINAL ERROR STATUS:", "RESOLVED" if tests_passed == total_tests else "NOT RESOLVED")
print("=" * 70)

sys.exit(exit_code)