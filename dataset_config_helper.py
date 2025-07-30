#!/usr/bin/env python3
"""
Dataset Configuration Helper for Phase 6 Clinical Framework

This utility helps adapt the Phase 6 framework to different datasets by:
1. Analyzing dataset structure and suggesting configurations
2. Generating updated configuration cells for Jupyter notebooks
3. Providing dataset switching utilities

Usage:
    python dataset_config_helper.py --analyze dataset.csv
    python dataset_config_helper.py --switch-to liver
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any

class DatasetConfigHelper:
    """Helper for configuring different datasets in the Phase 6 framework."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.known_datasets = {
            'diabetes': {
                'file': 'Pakistani_Diabetes_Dataset.csv',
                'target': 'Outcome',
                'description': 'Pakistani Diabetes Diagnostic Dataset',
                'context': {
                    'population': 'Pakistani patients (age 21-81)',
                    'study_type': 'Cross-sectional diagnostic study',
                    'setting': 'Primary care and diabetes clinics',
                    'primary_outcome': 'Diabetes diagnosis (0=No Diabetes, 1=Diabetes)',
                    'geography': 'Pakistan'
                },
                'clinical_ranges': {
                    'Age': {'normal_range': (18, 65), 'critical_value': 75, 'unit': 'years'},
                    'BMI': {'normal_range': (18.5, 25.0), 'critical_value': 30.0, 'unit': 'kg/m²'},
                    'sys': {'normal_range': (90, 140), 'critical_value': 180, 'unit': 'mmHg'},
                    'A1c': {'normal_range': (4.0, 6.5), 'critical_value': 9.0, 'unit': '%'}
                }
            },
            'liver': {
                'file': 'liver_train.csv',
                'target': 'Result',
                'description': 'Indian Liver Patient Dataset',
                'context': {
                    'population': 'Indian patients with liver disease risk factors',
                    'study_type': 'Case-control study',
                    'setting': 'Tertiary care hospital',
                    'primary_outcome': 'Liver disease diagnosis (1=Disease, 2=No Disease)',
                    'geography': 'India'
                },
                'clinical_ranges': {
                    'Age of the patient': {'normal_range': (18, 65), 'critical_value': 75, 'unit': 'years'},
                    'Total Bilirubin': {'normal_range': (0.2, 1.2), 'critical_value': 3.0, 'unit': 'mg/dL'},
                    'Direct Bilirubin': {'normal_range': (0.0, 0.3), 'critical_value': 1.0, 'unit': 'mg/dL'},
                    'Total Protiens': {'normal_range': (6.3, 8.2), 'critical_value': 5.0, 'unit': 'g/dL'},
                    'ALB Albumin': {'normal_range': (3.5, 5.5), 'critical_value': 2.5, 'unit': 'g/dL'}
                }
            }
        }
    
    def load_dataset_with_encoding_fallback(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load dataset with multiple encoding attempts."""
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"[OK] Successfully loaded with encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"[ERROR] Error with {encoding}: {str(e)}")
                continue
        
        return None
    
    def analyze_dataset(self, file_path: str) -> Dict[str, Any]:
        """Analyze dataset structure and suggest configuration."""
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return {}
        
        print(f"Analyzing dataset: {file_path}")
        print("=" * 60)
        
        # Load data
        df = self.load_dataset_with_encoding_fallback(file_path)
        if df is None:
            print("[ERROR] Could not load dataset with any encoding")
            return {}
        
        # Basic structure
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print()
        
        # Column analysis
        print("Column Analysis:")
        print("-" * 40)
        
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'column_analysis': {}
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'missing': df[col].isnull().sum(),
                'unique': df[col].nunique(),
                'unique_ratio': df[col].nunique() / len(df)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                })
                col_type = 'continuous' if col_info['unique_ratio'] > 0.05 else 'categorical'
            else:
                col_info['sample_values'] = df[col].dropna().unique()[:5].tolist()
                col_type = 'categorical'
            
            col_info['suggested_type'] = col_type
            analysis['column_analysis'][col] = col_info
            
            print(f"{col:25} | {col_type:12} | Missing: {col_info['missing']:3} | Unique: {col_info['unique']:3}")
        
        # Suggest target column
        potential_targets = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['outcome', 'result', 'target', 'label', 'class', 'diagnosis']):
                potential_targets.append(col)
            elif df[col].nunique() == 2 and pd.api.types.is_numeric_dtype(df[col]):
                potential_targets.append(col)
        
        if potential_targets:
            print(f"\nSuggested target columns: {potential_targets}")
            analysis['suggested_targets'] = potential_targets
        else:
            print(f"\nNo obvious target column found. Consider: {df.columns[-1]}")
            analysis['suggested_targets'] = [df.columns[-1]]
        
        return analysis
    
    def generate_config_code(self, dataset_key: str, custom_path: str = None) -> str:
        """Generate configuration code for Jupyter notebook."""
        if dataset_key not in self.known_datasets:
            print(f"[ERROR] Unknown dataset: {dataset_key}")
            return ""
        
        config = self.known_datasets[dataset_key]
        
        # Use custom path if provided
        if custom_path:
            data_path = custom_path
        else:
            data_path = f"C:\\Users\\gcicc\\claudeproj\\tableGenCompare\\data\\{config['file']}"
        
        # Generate configuration code
        code = f'''# ===== UPDATED CONFIGURATION FOR {config['description'].upper()} =====

# Dataset Configuration
DATA_PATH = r"{data_path}"
TARGET_COLUMN = "{config['target']}"

# Clinical Context
DATASET_DESCRIPTION = "{config['description']}"
CLINICAL_CONTEXT = {json.dumps(config['context'], indent=4)}

# Analysis Configuration
RANDOM_STATE = 42
N_OPTIMIZATION_TRIALS = 10  # Reduced for quick testing
EXPORT_RESULTS = True
GENERATE_HTML_REPORT = True

# Clinical Reference Ranges
CLINICAL_RANGES = {{'''

        for var, range_info in config['clinical_ranges'].items():
            code += f'''
    '{var}': {{
        'normal_range': {range_info['normal_range']},
        'critical_value': {range_info['critical_value']},
        'unit': '{range_info['unit']}',
        'interpretation': '{var} measurement'
    }},'''
        
        code += '''
}

# Clinical Labels for Target Variable
CLINICAL_LABELS = {
    0: "Negative",
    1: "Positive"
}

print(f"Configuration updated for {DATASET_DESCRIPTION}")
print(f"Target: {TARGET_COLUMN}")
print(f"Data Path: {DATA_PATH}")
'''
        
        return code
    
    def verify_dataset_compatibility(self, file_path: str) -> Dict[str, Any]:
        """Verify dataset is compatible with the framework."""
        results = {
            'compatible': False,
            'issues': [],
            'recommendations': []
        }
        
        if not os.path.exists(file_path):
            results['issues'].append(f"File not found: {file_path}")
            return results
        
        df = self.load_dataset_with_encoding_fallback(file_path)
        if df is None:
            results['issues'].append("Could not load dataset with any encoding")
            return results
        
        # Check minimum requirements
        if len(df) < 100:
            results['issues'].append(f"Dataset too small ({len(df)} rows). Minimum recommended: 100")
        
        if len(df.columns) < 3:
            results['issues'].append(f"Too few features ({len(df.columns)} columns). Minimum recommended: 3")
        
        # Check missing values
        missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percent > 50:
            results['issues'].append(f"Too many missing values ({missing_percent:.1f}%). Maximum recommended: 50%")
        
        # Check for potential target column
        binary_cols = []
        for col in df.columns:
            if df[col].nunique() == 2:
                binary_cols.append(col)
        
        if not binary_cols:
            results['issues'].append("No binary columns found for classification target")
        
        # Determine compatibility
        results['compatible'] = len(results['issues']) == 0
        
        # Generate recommendations
        if results['compatible']:
            results['recommendations'].append("Dataset appears compatible with the framework")
            if binary_cols:
                results['recommendations'].append(f"Consider using as target: {binary_cols}")
        else:
            results['recommendations'].append("Address the listed issues before using with framework")
        
        return results

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Dataset Configuration Helper for Phase 6 Clinical Framework")
    parser.add_argument('--analyze', type=str, help='Analyze a dataset file')
    parser.add_argument('--switch-to', choices=['diabetes', 'liver'], help='Switch to a known dataset')
    parser.add_argument('--generate-config', type=str, help='Generate config code for dataset')
    parser.add_argument('--verify', type=str, help='Verify dataset compatibility')
    
    args = parser.parse_args()
    
    helper = DatasetConfigHelper()
    
    if args.analyze:
        analysis = helper.analyze_dataset(args.analyze)
        
        if analysis:
            print("\nAnalysis Summary:")
            print("=" * 60)
            print(f"Rows: {analysis['shape'][0]}")
            print(f"Columns: {analysis['shape'][1]}")
            print(f"Missing values: {sum(analysis['missing_values'].values())}")
            print(f"Suggested targets: {analysis.get('suggested_targets', ['Unknown'])}")
    
    elif args.switch_to:
        if args.switch_to in helper.known_datasets:
            config_code = helper.generate_config_code(args.switch_to)
            print("Configuration Code for Jupyter Notebook:")
            print("=" * 60)
            print(config_code)
        else:
            print(f"[ERROR] Unknown dataset: {args.switch_to}")
    
    elif args.generate_config:
        analysis = helper.analyze_dataset(args.generate_config)
        if analysis:
            print("\nSuggested Configuration Template:")
            print("=" * 60)
            # Generate basic template based on analysis
            print("# Add this configuration to your notebook:")
            print(f'DATA_PATH = r"{os.path.abspath(args.generate_config)}"')
            if analysis.get('suggested_targets'):
                print(f'TARGET_COLUMN = "{analysis["suggested_targets"][0]}"')
    
    elif args.verify:
        results = helper.verify_dataset_compatibility(args.verify)
        
        print("Dataset Compatibility Check:")
        print("=" * 60)
        print(f"Compatible: {'[YES]' if results['compatible'] else '[NO]'}")
        
        if results['issues']:
            print("\nIssues Found:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")
    
    else:
        # Default: show available datasets
        print("Available Datasets:")
        print("=" * 60)
        
        for name, config in helper.known_datasets.items():
            file_path = os.path.join("data", config['file'])
            exists = "[OK]" if os.path.exists(file_path) else "[MISSING]"
            print(f"{exists} {name:10} | {config['description']}")
            print(f"   Target: {config['target']} | File: {config['file']}")
            print()
        
        print("Usage examples:")
        print("  python dataset_config_helper.py --switch-to liver")
        print("  python dataset_config_helper.py --analyze data/my_dataset.csv")
        print("  python dataset_config_helper.py --verify data/my_dataset.csv")

if __name__ == "__main__":
    main()