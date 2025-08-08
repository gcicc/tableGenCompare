#!/usr/bin/env python3
"""
Script to verify that all objective functions are properly defined and use 'data'
"""

import json
import sys

def verify_objective_functions(notebook_path):
    """Verify that all objective functions are correctly using 'data' instead of 'train_data'"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    objective_functions = ['ctabgan_objective', 'ctabganplus_objective', 'ganeraid_objective', 'copulagan_objective']
    
    verification_results = {}
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            source_text = ''.join(source_lines)
            
            # Look for objective function cells
            for func_name in objective_functions:
                if f'def {func_name}(' in source_text:
                    print(f"=== Verifying {func_name} (Cell {i}) ===")
                    
                    # Check for correct patterns
                    has_model_train_data = 'model.train(data' in source_text
                    has_model_generate = 'model.generate(len(data))' in source_text
                    has_enhanced_objective = 'enhanced_objective_function_v2(' in source_text and 'data, synthetic_data' in source_text
                    has_train_data_error = 'train_data' in source_text
                    
                    verification_results[func_name] = {
                        'cell': i,
                        'correct_train_call': has_model_train_data,
                        'correct_generate_call': has_model_generate,
                        'correct_objective_call': has_enhanced_objective,
                        'has_train_data_issues': has_train_data_error
                    }
                    
                    print(f"  - Uses model.train(data, ...): {has_model_train_data}")
                    print(f"  - Uses model.generate(len(data)): {has_model_generate}")
                    print(f"  - Correct enhanced_objective_function_v2 call: {has_enhanced_objective}")
                    print(f"  - Has train_data issues: {has_train_data_error}")
                    
                    if has_model_train_data and has_model_generate and has_enhanced_objective and not has_train_data_error:
                        print(f"  ✓ {func_name} is CORRECTLY implemented")
                    else:
                        print(f"  ✗ {func_name} has issues")
                    print()
                    
                    break
    
    # Summary
    print("=== VERIFICATION SUMMARY ===")
    all_correct = True
    for func_name, results in verification_results.items():
        is_correct = (results['correct_train_call'] and 
                     results['correct_generate_call'] and 
                     results['correct_objective_call'] and 
                     not results['has_train_data_issues'])
        
        status = "CORRECT" if is_correct else "NEEDS FIXING"
        print(f"{func_name}: {status}")
        
        if not is_correct:
            all_correct = False
    
    print(f"\nOverall Status: {'ALL FUNCTIONS CORRECT' if all_correct else 'SOME FUNCTIONS NEED FIXING'}")
    print(f"Functions checked: {len(verification_results)}/{len(objective_functions)}")
    
    return verification_results, all_correct

if __name__ == "__main__":
    notebook_path = "Clinical_Synthetic_Data_Generation_Framework.ipynb"
    results, all_correct = verify_objective_functions(notebook_path)
    
    if all_correct:
        print("\nSUCCESS: All objective functions are correctly implemented!")
        print("The notebook should now execute without 'train_data' errors.")
    else:
        print("\nWARNING: Some objective functions still need fixing.")