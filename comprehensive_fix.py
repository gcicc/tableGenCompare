#!/usr/bin/env python3
"""
Comprehensive script to find and fix any remaining train_data references in source code
"""

import json
import sys

def comprehensive_fix(notebook_path):
    """Find and fix any remaining train_data references in source code"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = 0
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            
            # Check for train_data in source code
            has_train_data = any('train_data' in line for line in source_lines)
            
            if has_train_data:
                print(f"=== Cell {i} has train_data references ===")
                
                # Show the problematic lines
                for j, line in enumerate(source_lines):
                    if 'train_data' in line:
                        print(f"  Line {j+1}: {line.strip()}")
                
                # Fix the lines
                fixed_lines = []
                for line in source_lines:
                    # Replace train_data with data
                    fixed_line = line.replace('train_data', 'data')
                    fixed_lines.append(fixed_line)
                    
                    if line != fixed_line:
                        print(f"  FIXED: {line.strip()}")
                        print(f"      -> {fixed_line.strip()}")
                        changes_made += 1
                
                cell['source'] = fixed_lines
                print()
    
    if changes_made > 0:
        # Create backup
        backup_path = notebook_path + '.backup'
        print(f"BACKUP: Creating backup: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        # Write fixed notebook
        print(f"SAVING: Writing fixed notebook: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"FIXED: Made {changes_made} fixes")
    else:
        print("SUCCESS: No train_data references found in source code - all objective functions appear correct!")
    
    return changes_made

def clear_error_outputs(notebook_path):
    """Clear old error outputs that contain train_data errors"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cleared_cells = 0
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            
            # Check if outputs contain train_data error messages
            has_train_data_errors = False
            for output in outputs:
                if 'text' in output:
                    text_content = ''.join(output['text']) if isinstance(output['text'], list) else str(output['text'])
                    if "name 'train_data' is not defined" in text_content:
                        has_train_data_errors = True
                        break
            
            if has_train_data_errors:
                print(f"Clearing error outputs from cell {i}")
                cell['outputs'] = []
                cell['execution_count'] = None
                cleared_cells += 1
    
    if cleared_cells > 0:
        # Write cleaned notebook
        print(f"SAVING: Writing notebook with cleared error outputs: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"SUCCESS: Cleared error outputs from {cleared_cells} cells")
    
    return cleared_cells

if __name__ == "__main__":
    notebook_path = "Clinical_Synthetic_Data_Generation_Framework.ipynb"
    
    # Fix any remaining train_data references
    fixes = comprehensive_fix(notebook_path)
    
    # Clear old error outputs
    cleared = clear_error_outputs(notebook_path)
    
    print(f"\nSUMMARY:")
    print(f"   - Source code fixes: {fixes}")
    print(f"   - Cleared error outputs: {cleared}")
    print(f"   - Notebook should now be ready for re-execution")