#!/usr/bin/env python3
"""
Debug script to check what CTAB-GAN versions are being imported.
"""

import sys
import os

print("Current Python path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

# Test CTAB-GAN import
print("\n" + "="*50)
print("Testing CTAB-GAN import:")
try:
    ctabgan_path = os.path.join(os.path.dirname(__file__), "CTAB-GAN")
    print(f"Adding path: {ctabgan_path}")
    if ctabgan_path not in sys.path:
        sys.path.insert(0, ctabgan_path)
    
    from model.ctabgan import CTABGAN
    import inspect
    
    # Check the source file
    source_file = inspect.getsourcefile(CTABGAN)
    print(f"CTAB-GAN imported from: {source_file}")
    
    # Check constructor signature
    sig = inspect.signature(CTABGAN.__init__)
    print(f"Constructor parameters: {list(sig.parameters.keys())}")
    
except Exception as e:
    print(f"CTAB-GAN import failed: {e}")

# Test CTAB-GAN+ import
print("\n" + "="*50)
print("Testing CTAB-GAN+ import:")
try:
    # Clear the module cache to force fresh import
    if 'model.ctabgan' in sys.modules:
        del sys.modules['model.ctabgan']
    if 'model' in sys.modules:
        del sys.modules['model']
    
    ctabganplus_path = os.path.join(os.path.dirname(__file__), "CTAB-GAN-Plus")
    print(f"Adding path: {ctabganplus_path}")
    if ctabganplus_path not in sys.path:
        sys.path.insert(0, ctabganplus_path)
    
    from model.ctabgan import CTABGAN
    import inspect
    
    # Check the source file
    source_file = inspect.getsourcefile(CTABGAN)
    print(f"CTAB-GAN+ imported from: {source_file}")
    
    # Check constructor signature
    sig = inspect.signature(CTABGAN.__init__)
    print(f"Constructor parameters: {list(sig.parameters.keys())}")
    
except Exception as e:
    print(f"CTAB-GAN+ import failed: {e}")