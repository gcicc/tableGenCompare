#!/usr/bin/env python3
"""
Script to examine and fix objective functions in the notebook
"""

import json
import sys
import re

def examine_objective_functions(notebook_path):
    """Find and examine objective functions that need fixing"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    objective_functions = ['ctabgan_objective', 'ctabganplus_objective', 'ganeraid_objective', 'copulagan_objective']
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            source_text = ''.join(source_lines)
            
            # Look for objective function cells
            for func_name in objective_functions:
                if f'def {func_name}(' in source_text:
                    print(f"=== Cell {i}: {func_name} ===")
                    
                    # Print all lines safely, looking for train_data
                    for j, line in enumerate(source_lines):
                        line_safe = line.encode('ascii', errors='replace').decode('ascii')
                        if 'train_data' in line or 'model.train(' in line or 'model.generate(' in line:
                            print(f"*** {j+1:2d}: {line_safe.strip()}")
                        else:
                            print(f"    {j+1:2d}: {line_safe.strip()}")
                    
                    print()
                    break
    
    return True

if __name__ == "__main__":
    notebook_path = "Clinical_Synthetic_Data_Generation_Framework.ipynb"
    examine_objective_functions(notebook_path)