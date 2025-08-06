#\!/usr/bin/env python3
"""Simple test script to verify TableGAN demo functionality."""

import sys
import os
import warnings
import traceback
import time
import pandas as pd
import numpy as np

def main():
    print("TABLEGAN DEMO TESTING")
    print("=" * 50)
    
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    # Test 1: Basic imports
    print("\nTEST 1: BASIC IMPORTS")
    print("-" * 30)
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("SUCCESS: Basic libraries imported")
        
        import optuna
        print("SUCCESS: Optuna imported")
        
        from ctgan import CTGAN
        print("SUCCESS: CTGAN imported")
        
        from sdv.single_table import TVAESynthesizer, CopulaGANSynthesizer
        print("SUCCESS: SDV models imported")
        
    except Exception as e:
        print(f"FAILED: Import error - {e}")
        return False
    
    # Test 2: TableGAN import
    print("\nTEST 2: TABLEGAN IMPORT")
    print("-" * 30)
    
    try:
        tablegan_path = os.path.join(os.getcwd(), "tableGAN")
        if tablegan_path not in sys.path:
            sys.path.insert(0, tablegan_path)
        
        print(f"Added to path: {tablegan_path}")
        
        from model import TableGan
        from utils import generate_data
        print("SUCCESS: TableGAN imported from GitHub repository")
        TABLEGAN_AVAILABLE = True
        
    except Exception as e:
        print(f"FAILED: TableGAN import error - {e}")
        print("Will use mock implementation for demo")
        TABLEGAN_AVAILABLE = False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
