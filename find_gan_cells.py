#!/usr/bin/env python3
"""
Script to find and analyze GAN model cells in the notebook
"""

import json
import sys
import re

def find_gan_cells(notebook_path):
    """Find cells containing GAN model definitions"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    gan_models = ['CTAB-GAN', 'CTAB_GAN', 'CTABGAN', 'GANerAid', 'CopulaGAN']
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            source_text = ''.join(source_lines)
            
            # Look for GAN model cells
            for model in gan_models:
                if model in source_text and 'ModelFactory' in source_text:
                    print(f"=== Cell {i}: {model} ===")
                    
                    # Print first few lines safely
                    for j, line in enumerate(source_lines[:20]):
                        try:
                            print(f"{j+1:2d}: {line.strip()}")
                        except:
                            print(f"{j+1:2d}: [unicode error in line]")
                    
                    print()
                    break
    
    return True

if __name__ == "__main__":
    notebook_path = "Clinical_Synthetic_Data_Generation_Framework.ipynb"
    find_gan_cells(notebook_path)