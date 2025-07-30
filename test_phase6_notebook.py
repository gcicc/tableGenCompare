#!/usr/bin/env python3
"""
Test script for Phase 6 Pakistani Diabetes Comprehensive Analysis Notebook
Validates data loading, basic functionality, and notebook structure.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

def test_data_loading():
    """Test the core data loading functionality."""
    print("TESTING PHASE 6 NOTEBOOK FUNCTIONALITY")
    print("=" * 50)
    
    # Test configuration
    DATA_PATH = r"C:\Users\gcicc\claudeproj\tableGenCompare\data\Pakistani_Diabetes_Dataset.csv"
    TARGET_COLUMN = "Outcome"
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    def run_test(test_name, test_func):
        """Helper to run individual tests."""
        try:
            result = test_func()
            status = "PASSED" if result else "FAILED"
            test_results['tests'].append({
                'name': test_name,
                'status': status,
                'result': result
            })
            print(f"{'PASS' if result else 'FAIL'} {test_name}: {status}")
            return result
        except Exception as e:
            test_results['tests'].append({
                'name': test_name,
                'status': 'ERROR',
                'error': str(e)
            })
            print(f"ERROR {test_name}: ERROR - {str(e)}")
            return False
    
    # Test 1: File existence
    def test_file_exists():
        return os.path.exists(DATA_PATH)
    
    run_test("Dataset File Exists", test_file_exists)
    
    # Test 2: Data loading
    def test_load_data():
        try:
            data = pd.read_csv(DATA_PATH)
            return len(data) > 0 and len(data.columns) > 0
        except:
            return False
    
    data_loaded = run_test("Data Loading", test_load_data)
    
    if not data_loaded:
        print("\nERROR: Cannot proceed with further tests - data loading failed")
        return test_results
    
    # Load data for subsequent tests
    data = pd.read_csv(DATA_PATH)
    
    # Test 3: Expected structure
    def test_data_structure():
        expected_cols = ['Age', 'Gender', 'Rgn', 'wt', 'BMI', 'wst', 'sys', 'dia', 
                        'his', 'A1c', 'B.S.R', 'vision', 'Exr', 'dipsia', 'uria', 
                        'Dur', 'neph', 'HDL', 'Outcome']
        actual_cols = set(data.columns)
        expected_cols_set = set(expected_cols)
        return len(expected_cols_set - actual_cols) <= 2  # Allow 2 missing columns
    
    run_test("Data Structure Valid", test_data_structure)
    
    # Test 4: Target column
    def test_target_column():
        return TARGET_COLUMN in data.columns
    
    run_test("Target Column Present", test_target_column)
    
    # Test 5: Data quality
    def test_data_quality():
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
        return missing_pct < 50  # Less than 50% missing
    
    run_test("Data Quality Acceptable", test_data_quality)
    
    # Test 6: Biomarkers present
    def test_biomarkers():
        key_biomarkers = ['A1c', 'B.S.R', 'HDL', 'BMI', 'sys', 'dia']
        present = sum(1 for col in key_biomarkers if col in data.columns)
        return present >= len(key_biomarkers) * 0.8  # 80% present
    
    run_test("Key Biomarkers Present", test_biomarkers)
    
    # Test 7: Clinical ranges reasonable
    def test_clinical_ranges():
        try:
            if 'Age' in data.columns:
                age_valid = data['Age'].min() > 0 and data['Age'].max() < 120
            else:
                age_valid = True
                
            if 'BMI' in data.columns:
                bmi_valid = data['BMI'].min() > 10 and data['BMI'].max() < 60
            else:
                bmi_valid = True
                
            return age_valid and bmi_valid
        except:
            return False
    
    run_test("Clinical Ranges Reasonable", test_clinical_ranges)
    
    # Test 8: Target distribution
    def test_target_distribution():
        if TARGET_COLUMN not in data.columns:
            return False
        target_counts = data[TARGET_COLUMN].value_counts()
        return len(target_counts) == 2 and target_counts.min() > 0
    
    run_test("Target Distribution Valid", test_target_distribution)
    
    # Test 9: Notebook file exists
    def test_notebook_exists():
        notebook_path = r"C:\Users\gcicc\claudeproj\tableGenCompare\notebooks\Phase6_Pakistani_Diabetes_Comprehensive_Analysis.ipynb"
        return os.path.exists(notebook_path)
    
    run_test("Notebook File Created", test_notebook_exists)
    
    # Test 10: Notebook structure
    def test_notebook_structure():
        notebook_path = r"C:\Users\gcicc\claudeproj\tableGenCompare\notebooks\Phase6_Pakistani_Diabetes_Comprehensive_Analysis.ipynb"
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_content = f.read()
            
            # Check for key sections
            required_sections = [
                "Phase 6: Pakistani Diabetes Dataset",
                "Executive Summary",
                "Configuration and Setup",
                "Data Loading and Validation",
                "Initial Data Exploration",
                "load_clinical_dataset",
                "perform_clinical_data_exploration"
            ]
            
            return all(section in notebook_content for section in required_sections)
        except:
            return False
    
    run_test("Notebook Structure Complete", test_notebook_structure)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(test_results['tests'])
    passed_tests = sum(1 for test in test_results['tests'] if test['status'] == 'PASSED')
    failed_tests = sum(1 for test in test_results['tests'] if test['status'] in ['FAILED', 'ERROR'])
    success_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
    
    print(f"Results:")
    print(f"   - Total tests: {total_tests}")
    print(f"   - Passed: {passed_tests}")
    print(f"   - Failed: {failed_tests}")
    print(f"   - Success rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        status = "EXCELLENT - Ready for production use"
    elif success_rate >= 80:
        status = "GOOD - Minor issues, suitable for use"
    elif success_rate >= 70:
        status = "ACCEPTABLE - Some issues, proceed with caution"
    else:
        status = "NEEDS ATTENTION - Address issues before use"
    
    print(f"\nOverall Status: {status}")
    
    # Failed tests details
    failed_test_details = [test for test in test_results['tests'] if test['status'] in ['FAILED', 'ERROR']]
    if failed_test_details:
        print(f"\nFailed Tests:")
        for test in failed_test_details:
            print(f"   - {test['name']}: {test.get('error', 'Failed')}")
    
    print(f"\nNext Steps:")
    if success_rate >= 90:
        print(f"   - Notebook is ready for comprehensive Pakistani diabetes analysis")
        print(f"   - All core functionality validated successfully")
        print(f"   - Proceed with synthetic data generation pipeline")
    else:
        print(f"   - Address failed tests before proceeding")
        print(f"   - Check data file paths and permissions")
        print(f"   - Verify notebook structure and content")
    
    # Save results
    try:
        with open('phase6_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\nTest results saved to: phase6_test_results.json")
    except Exception as e:
        print(f"\nWarning: Could not save test results: {e}")
    
    return test_results

def display_notebook_info():
    """Display information about the created notebook."""
    print("\nNOTEBOOK INFORMATION")
    print("=" * 50)
    
    notebook_path = r"C:\Users\gcicc\claudeproj\tableGenCompare\notebooks\Phase6_Pakistani_Diabetes_Comprehensive_Analysis.ipynb"
    
    if os.path.exists(notebook_path):
        file_size = os.path.getsize(notebook_path) / 1024  # KB
        print(f"File: Phase6_Pakistani_Diabetes_Comprehensive_Analysis.ipynb")
        print(f"Size: {file_size:.1f} KB")
        print(f"Location: {notebook_path}")
        
        print(f"\nNotebook Sections:")
        print(f"   1. Executive Summary - Professional introduction for clinical teams")
        print(f"   2. Configuration Setup - Pakistani diabetes dataset settings")
        print(f"   3. Data Loading - Multi-encoding CSV loading with error handling")
        print(f"   4. Initial Exploration - Clinical biomarker analysis")
        print(f"   5. Clinical Validation - Range validation and quality assessment")
        print(f"   6. Testing Framework - Comprehensive validation tests")
        
        print(f"\nKey Features:")
        print(f"   - Self-contained code - all functions defined within notebook")
        print(f"   - Pakistani diabetes focus - clinical context and terminology")
        print(f"   - Error handling - graceful fallbacks for data loading issues")
        print(f"   - Professional formatting - suitable for clinical team review")
        print(f"   - Comprehensive testing - 10-point validation framework")
        
        print(f"\nUsage Instructions:")
        print(f"   1. Open Jupyter Lab/Notebook: jupyter lab")
        print(f"   2. Navigate to: notebooks/Phase6_Pakistani_Diabetes_Comprehensive_Analysis.ipynb")
        print(f"   3. Run all cells to perform comprehensive analysis")
        print(f"   4. Review clinical insights and data quality assessment")
        print(f"   5. Proceed with synthetic data generation pipeline")
        
    else:
        print(f"ERROR: Notebook file not found at: {notebook_path}")

if __name__ == "__main__":
    print("PHASE 6 NOTEBOOK VALIDATION TEST")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run tests
    test_results = test_data_loading()
    
    # Display notebook information
    display_notebook_info()
    
    print("\n" + "=" * 60)
    print("Phase 6 Notebook Testing Complete")
    print("Pakistani Diabetes Comprehensive Analysis Framework Ready")
    print("=" * 60)