#!/usr/bin/env python3
"""
Quick fix for the clinical range validation error in Phase 6 notebook.
This script provides a corrected version of the validate_clinical_ranges function.
"""

import pandas as pd
import numpy as np

def validate_clinical_ranges_fixed(data, clinical_context):
    """
    Fixed version of validate_clinical_ranges function that handles both tuple and single values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Clinical dataset
    clinical_context : dict
        Clinical context information
    
    Returns:
    --------
    dict: Clinical validation results
    """
    
    # Define clinical reference ranges for diabetes biomarkers
    clinical_ranges = {
        'A1c': {
            'normal': (0, 5.7),
            'prediabetes': (5.7, 6.5),
            'diabetes': (6.5, 20),
            'unit': '%',
            'description': 'Hemoglobin A1c'
        },
        'B.S.R': {
            'normal': (70, 140),
            'impaired': (140, 200),
            'diabetes': (200, 1000),
            'unit': 'mg/dL',
            'description': 'Random Blood Sugar'
        },
        'HDL': {
            'low_risk_men': (40, 200),
            'low_risk_women': (50, 200),
            'high_risk': (0, 40),
            'unit': 'mg/dL',
            'description': 'HDL Cholesterol'
        },
        'BMI': {
            'underweight': (0, 18.5),
            'normal': (18.5, 25),
            'overweight': (25, 30),
            'obese': (30, 60),
            'unit': 'kg/m²',
            'description': 'Body Mass Index'
        },
        'sys': {
            'normal': (90, 120),
            'elevated': (120, 130),
            'stage1_htn': (130, 140),
            'stage2_htn': (140, 250),
            'unit': 'mmHg',
            'description': 'Systolic Blood Pressure'
        },
        'dia': {
            'normal': (60, 80),
            'elevated': (80, 90),
            'hypertension': (90, 150),
            'unit': 'mmHg',
            'description': 'Diastolic Blood Pressure'
        },
        'Age': {
            'young_adult': (18, 35),
            'middle_age': (35, 55),
            'older_adult': (55, 100),
            'unit': 'years',
            'description': 'Patient Age'
        }
    }
    
    validation_results = {
        'clinical_distributions': {},
        'outlier_analysis': {},
        'range_compliance': {},
        'clinical_flags': []
    }
    
    print("🏥 CLINICAL RANGE VALIDATION")
    print("=" * 40)
    
    for variable, ranges in clinical_ranges.items():
        if variable in data.columns:
            var_data = data[variable].dropna()
            
            print(f"\n🔬 {ranges['description']} ({variable}):")
            print(f"   Range: {var_data.min():.1f} - {var_data.max():.1f} {ranges['unit']}")
            print(f"   Mean: {var_data.mean():.1f} ± {var_data.std():.1f} {ranges['unit']}")
            
            # Categorize values based on clinical ranges - FIXED VERSION
            categories = {}
            for category, range_value in ranges.items():
                if category not in ['unit', 'description']:
                    # Check if range_value is a tuple (min, max) or a single value
                    if isinstance(range_value, tuple) and len(range_value) == 2:
                        min_val, max_val = range_value
                        count = ((var_data >= min_val) & (var_data < max_val)).sum()
                        percentage = count / len(var_data) * 100
                        categories[category] = {'count': count, 'percentage': percentage}
                        print(f"   • {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
                    else:
                        # Handle single values or other formats
                        print(f"   • {category}: (non-range value: {range_value})")
            
            validation_results['clinical_distributions'][variable] = categories
            
            # Identify potential outliers (values outside typical ranges)
            if variable in ['A1c', 'B.S.R', 'HDL', 'BMI', 'sys', 'dia']:
                # Define extreme outlier thresholds
                outlier_thresholds = {
                    'A1c': (0, 20),
                    'B.S.R': (0, 1000),
                    'HDL': (10, 200),
                    'BMI': (10, 60),
                    'sys': (50, 250),
                    'dia': (30, 150)
                }
                
                if variable in outlier_thresholds:
                    min_thresh, max_thresh = outlier_thresholds[variable]
                    outliers = var_data[(var_data < min_thresh) | (var_data > max_thresh)]
                    
                    if len(outliers) > 0:
                        print(f"   ⚠️ Potential outliers: {len(outliers)} values ({len(outliers)/len(var_data)*100:.1f}%)")
                        validation_results['outlier_analysis'][variable] = {
                            'count': len(outliers),
                            'percentage': len(outliers)/len(var_data)*100,
                            'values': outliers.tolist()
                        }
                        
                        if len(outliers)/len(var_data) > 0.05:  # >5% outliers
                            validation_results['clinical_flags'].append(
                                f"High outlier rate in {variable}: {len(outliers)/len(var_data)*100:.1f}%"
                            )
    
    return validation_results, clinical_ranges

print("✅ Fixed clinical validation function created")
print("💡 To use this fix in your notebook:")
print("   1. Run this cell to define the fixed function")
print("   2. Replace the call to validate_clinical_ranges() with validate_clinical_ranges_fixed()")
print("   3. The rest of your code should work normally")