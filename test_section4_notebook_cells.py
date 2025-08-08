"""
Test individual Section 4 cells from Clinical_Synthetic_Data_Generation_Framework.ipynb
Following claude6.md protocol - notebook cell validation
"""
import json
import subprocess
import sys
import os
import tempfile
from pathlib import Path

def extract_section4_cells():
    """Extract Section 4 cells from the notebook"""
    notebook_path = "Clinical_Synthetic_Data_Generation_Framework.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    section4_cells = []
    in_section4 = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '# 4.' in source or '## 4.' in source:
                in_section4 = True
                section4_cells.append(('markdown', source))
            elif source.startswith('# ') and not source.startswith('# 4'):
                in_section4 = False
        elif cell['cell_type'] == 'code' and in_section4:
            code = ''.join(cell['source']).strip()
            if code:  # Only add non-empty cells
                section4_cells.append(('code', code))
    
    return section4_cells

def test_individual_cell(cell_code, cell_index):
    """Test execution of individual cell"""
    print(f"\n{'='*60}")
    print(f"TESTING CELL {cell_index}")
    print(f"{'='*60}")
    
    # Show first few lines of cell
    lines = cell_code.split('\n')
    preview = '\n'.join(lines[:3])
    if len(lines) > 3:
        preview += f'\n... ({len(lines)-3} more lines)'
    
    print(f"CELL PREVIEW:\n{preview}")
    print(f"{'='*60}")
    
    # Create temp file for this cell
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Add necessary setup
        setup_code = """
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

# Import common modules that should be available
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set working directory
os.chdir(r'C:\\Users\\gcicc\\claudeproj\\tableGenCompare')

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

"""
        f.write(setup_code)
        f.write('\n# CELL CODE:\n')
        f.write(cell_code)
        temp_file = f.name
    
    try:
        # Execute the cell
        result = subprocess.run(
            [sys.executable, temp_file], 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            cwd=r'C:\Users\gcicc\claudeproj\tableGenCompare'
        )
        
        if result.returncode == 0:
            print(f"✅ CELL {cell_index}: SUCCESS")
            if result.stdout.strip():
                print(f"OUTPUT:\n{result.stdout}")
            return True
        else:
            print(f"❌ CELL {cell_index}: FAILED")
            if result.stderr.strip():
                print(f"ERROR:\n{result.stderr}")
            if result.stdout.strip():
                print(f"STDOUT:\n{result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ CELL {cell_index}: TIMEOUT (>5 minutes)")
        return False
    except Exception as e:
        print(f"💥 CELL {cell_index}: EXCEPTION - {e}")
        return False
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass

def main():
    """Main test execution"""
    print("SECTION 4 NOTEBOOK CELL TESTING")
    print("Following claude6.md protocol")
    print("="*80)
    
    # Extract cells
    try:
        section4_cells = extract_section4_cells()
        print(f"Found {len(section4_cells)} Section 4 cells")
    except Exception as e:
        print(f"FAILED to extract cells: {e}")
        return
    
    # Test each code cell
    code_cells = [(i, code) for i, (cell_type, code) in enumerate(section4_cells) if cell_type == 'code']
    
    print(f"Testing {len(code_cells)} code cells from Section 4...")
    
    results = []
    for cell_index, cell_code in code_cells:
        success = test_individual_cell(cell_code, cell_index)
        results.append((cell_index, success))
        
        # Early exit on first failure for debugging
        if not success:
            print(f"\n🛑 STOPPING at first failure (Cell {cell_index}) for detailed analysis")
            break
    
    # Summary
    print(f"\n{'='*80}")
    print("SECTION 4 CELL TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    print(f"Cells tested: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed cells:")
        for cell_index, success in results:
            if not success:
                print(f"  Cell {cell_index}: FAILED")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)