#!/usr/bin/env python3
"""
Script to fix the train_data reference errors in the notebook's objective functions
"""

import json
import sys

def fix_train_data_references(notebook_path):
    """Fix train_data references to data in objective functions"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = 0
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            # Check if this cell contains train_data references
            source_lines = cell.get('source', [])
            
            # Convert to string for easier processing
            source_text = ''.join(source_lines)
            
            # Check for any train_data reference in source
            if 'train_data' in source_text:
                print(f"Found cell {i} with train_data references:")
                print(f"  Preview: {source_text[:200]}...")
                
                # Fix the references
                fixed_lines = []
                for line in source_lines:
                    # Replace train_data references with data
                    fixed_line = line.replace('train_data', 'data')
                    fixed_lines.append(fixed_line)
                    
                    if line != fixed_line:
                        print(f"  Fixed: {line.strip()}")
                        print(f"      -> {fixed_line.strip()}")
                        changes_made += 1
                
                # Update the cell
                cell['source'] = fixed_lines
                print(f"  Cell {i} updated")
                print()
    
    if changes_made > 0:
        # Write the fixed notebook
        backup_path = notebook_path + '.backup'
        print(f"\n💾 Creating backup: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
            
        print(f"💾 Writing fixed notebook: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
            
        print(f"\n✅ Fixed {changes_made} train_data references")
    else:
        print("No train_data references found to fix")
    
    return changes_made

if __name__ == "__main__":
    notebook_path = "Clinical_Synthetic_Data_Generation_Framework.ipynb"
    changes = fix_train_data_references(notebook_path)
    print(f"\nCompleted with {changes} changes made")