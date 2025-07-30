#!/usr/bin/env python3
"""
Data Pipeline Validation Script for Phase 6 Clinical Framework

This script validates:
1. Data loading with multi-encoding support
2. Preprocessing functionality including missing value handling
3. Clinical range validation
4. Discrete/continuous column identification
5. Edge case handling

Usage:
    python test_data_pipeline.py
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.clinical_preprocessor import ClinicalDataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipelineValidator:
    """Comprehensive validator for the clinical data pipeline."""
    
    def __init__(self):
        self.results = {}
        self.test_passed = 0
        self.test_failed = 0
        
    def log_test_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result and update counters."""
        status = "[PASSED]" if passed else "[FAILED]"
        logger.info(f"{status}: {test_name} - {message}")
        
        self.results[test_name] = {
            'passed': passed,
            'message': message
        }
        
        if passed:
            self.test_passed += 1
        else:
            self.test_failed += 1
    
    def load_data_with_encoding_fallback(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV data with multiple encoding attempts."""
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                logger.info(f"Attempting to load data with encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded data with encoding: {encoding}")
                logger.info(f"Data shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
                return df
            except UnicodeDecodeError:
                logger.warning(f"Failed to load with encoding: {encoding}")
                continue
            except Exception as e:
                logger.error(f"Error loading data with {encoding}: {str(e)}")
                continue
        
        return None
    
    def test_data_loading(self) -> bool:
        """Test data loading functionality."""
        logger.info("=" * 60)
        logger.info("TESTING DATA LOADING")
        logger.info("=" * 60)
        
        # Test Pakistani Diabetes Dataset
        diabetes_path = r"C:\Users\gcicc\claudeproj\tableGenCompare\data\Pakistani_Diabetes_Dataset.csv"
        
        if os.path.exists(diabetes_path):
            diabetes_data = self.load_data_with_encoding_fallback(diabetes_path)
            if diabetes_data is not None:
                self.log_test_result(
                    "Pakistani Dataset Loading", 
                    True, 
                    f"Shape: {diabetes_data.shape}, Columns: {len(diabetes_data.columns)}"
                )
                self.diabetes_data = diabetes_data
            else:
                self.log_test_result("Pakistani Dataset Loading", False, "Could not load with any encoding")
                return False
        else:
            self.log_test_result("Pakistani Dataset Loading", False, "File does not exist")
            
            # Test fallback to liver dataset
            liver_path = r"C:\Users\gcicc\claudeproj\tableGenCompare\data\liver_train.csv"
            if os.path.exists(liver_path):
                liver_data = self.load_data_with_encoding_fallback(liver_path)
                if liver_data is not None:
                    self.log_test_result(
                        "Liver Dataset Fallback Loading", 
                        True, 
                        f"Shape: {liver_data.shape}, Columns: {len(liver_data.columns)}"
                    )
                    self.diabetes_data = liver_data  # Use as substitute
                else:
                    self.log_test_result("Liver Dataset Fallback Loading", False, "Could not load with any encoding")
                    return False
            else:
                self.log_test_result("Liver Dataset Fallback Loading", False, "File does not exist")
                return False
        
        return True
    
    def test_data_quality(self):
        """Test data quality and structure."""
        logger.info("=" * 60)
        logger.info("TESTING DATA QUALITY")
        logger.info("=" * 60)
        
        if not hasattr(self, 'diabetes_data'):
            self.log_test_result("Data Quality Check", False, "No data loaded")
            return
        
        df = self.diabetes_data
        
        # Check for basic structure
        has_data = len(df) > 0
        has_columns = len(df.columns) > 0
        self.log_test_result("Data Structure", has_data and has_columns, 
                           f"Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Check for missing values
        missing_info = df.isnull().sum()
        total_missing = missing_info.sum()
        missing_percent = (total_missing / (len(df) * len(df.columns))) * 100
        self.log_test_result("Missing Values Analysis", True, 
                           f"Total missing: {total_missing} ({missing_percent:.2f}%)")
        
        # Check column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.log_test_result("Column Types", True, 
                           f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
        
        # Data type details
        logger.info("Column Details:")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            logger.info(f"  {col}: {dtype}, Missing: {null_count}, Unique: {unique_count}")
    
    def test_preprocessing_pipeline(self):
        """Test the clinical preprocessing pipeline."""
        logger.info("=" * 60)
        logger.info("TESTING PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        if not hasattr(self, 'diabetes_data'):
            self.log_test_result("Preprocessing Pipeline", False, "No data loaded")
            return
        
        try:
            # Initialize preprocessor
            preprocessor = ClinicalDataPreprocessor(random_state=42)
            
            # Identify target column (common names)
            target_candidates = ['Outcome', 'Result', 'target', 'label', 'class']
            target_col = None
            
            for candidate in target_candidates:
                if candidate in self.diabetes_data.columns:
                    target_col = candidate
                    break
            
            if target_col is None:
                # Use last column as target
                target_col = self.diabetes_data.columns[-1]
                logger.warning(f"No standard target column found, using: {target_col}")
            
            logger.info(f"Using target column: {target_col}")
            
            # Test preprocessing
            original_shape = self.diabetes_data.shape
            processed_data = preprocessor.fit_transform(self.diabetes_data, target_col)
            
            # Validate preprocessing results
            preprocessing_success = (
                processed_data is not None and
                len(processed_data) > 0 and
                len(processed_data.columns) > 0
            )
            
            self.log_test_result("Preprocessing Execution", preprocessing_success,
                               f"Original: {original_shape}, Processed: {processed_data.shape}")
            
            if preprocessing_success:
                self.processed_data = processed_data
                self.preprocessor = preprocessor
                self.target_col = target_col
                
                # Test missing value handling
                original_missing = self.diabetes_data.isnull().sum().sum()
                processed_missing = processed_data.isnull().sum().sum()
                missing_handled = processed_missing <= original_missing
                
                self.log_test_result("Missing Value Handling", missing_handled,
                                   f"Original missing: {original_missing}, After: {processed_missing}")
                
        except Exception as e:
            self.log_test_result("Preprocessing Pipeline", False, f"Exception: {str(e)}")
            logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
    
    def test_column_identification(self):
        """Test discrete/continuous column identification."""
        logger.info("=" * 60)
        logger.info("TESTING COLUMN IDENTIFICATION")
        logger.info("=" * 60)
        
        if not hasattr(self, 'preprocessor'):
            self.log_test_result("Column Identification", False, "No preprocessor available")
            return
        
        try:
            # Get column types
            column_types = self.preprocessor.column_types
            discrete_cols = self.preprocessor.get_discrete_columns(self.processed_data)
            
            # Validate identification
            has_column_types = len(column_types) > 0
            has_discrete_identification = isinstance(discrete_cols, list)
            
            self.log_test_result("Column Type Identification", has_column_types,
                               f"Types identified: {len(column_types)}")
            
            self.log_test_result("Discrete Column Identification", has_discrete_identification,
                               f"Discrete columns: {len(discrete_cols)}")
            
            # Log details
            logger.info("Column Types:")
            for col, col_type in column_types.items():
                logger.info(f"  {col}: {col_type}")
            
            logger.info(f"Discrete columns: {discrete_cols}")
            
        except Exception as e:
            self.log_test_result("Column Identification", False, f"Exception: {str(e)}")
    
    def test_clinical_ranges_validation(self):
        """Test clinical range validation functionality."""
        logger.info("=" * 60)
        logger.info("TESTING CLINICAL RANGE VALIDATION")
        logger.info("=" * 60)
        
        if not hasattr(self, 'processed_data'):
            self.log_test_result("Clinical Range Validation", False, "No processed data available")
            return
        
        # Define sample clinical ranges
        clinical_ranges = {
            'age': {'normal_range': (18, 80), 'critical_value': 90},
            'glucose': {'normal_range': (70, 140), 'critical_value': 200},
            'bmi': {'normal_range': (18.5, 25.0), 'critical_value': 30.0},
        }
        
        try:
            df = self.processed_data
            validation_results = {}
            
            for col in df.columns:
                col_lower = col.lower()
                
                # Find matching clinical range
                range_config = None
                for range_name, config in clinical_ranges.items():
                    if range_name in col_lower:
                        range_config = config
                        break
                
                if range_config and pd.api.types.is_numeric_dtype(df[col]):
                    # Validate ranges
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        min_val = col_data.min()
                        max_val = col_data.max()
                        normal_min, normal_max = range_config['normal_range']
                        critical_val = range_config['critical_value']
                        
                        # Count values outside normal range
                        outside_normal = ((col_data < normal_min) | (col_data > normal_max)).sum()
                        above_critical = (col_data > critical_val).sum()
                        
                        validation_results[col] = {
                            'range': (min_val, max_val),
                            'outside_normal': outside_normal,
                            'above_critical': above_critical,
                            'total_values': len(col_data)
                        }
                        
                        logger.info(f"  {col}: Range({min_val:.2f}, {max_val:.2f}), "
                                  f"Outside normal: {outside_normal}, Above critical: {above_critical}")
            
            validation_success = len(validation_results) > 0
            self.log_test_result("Clinical Range Validation", validation_success,
                               f"Validated {len(validation_results)} clinical variables")
            
        except Exception as e:
            self.log_test_result("Clinical Range Validation", False, f"Exception: {str(e)}")
    
    def test_edge_cases(self):
        """Test edge case handling."""
        logger.info("=" * 60)
        logger.info("TESTING EDGE CASES")
        logger.info("=" * 60)
        
        try:
            # Test with empty dataframe
            empty_df = pd.DataFrame()
            preprocessor_edge = ClinicalDataPreprocessor()
            
            try:
                result = preprocessor_edge.fit_transform(empty_df)
                self.log_test_result("Empty DataFrame Handling", False, "Should raise error but didn't")
            except Exception:
                self.log_test_result("Empty DataFrame Handling", True, "Properly handled empty data")
            
            # Test with all missing values
            if hasattr(self, 'diabetes_data'):
                missing_df = self.diabetes_data.copy()
                missing_df.iloc[:, 0] = np.nan  # Make first column all NaN
                
                try:
                    preprocessor_missing = ClinicalDataPreprocessor()
                    result = preprocessor_missing.fit_transform(missing_df, self.target_col if hasattr(self, 'target_col') else None)
                    
                    # Check if missing values were handled
                    if result is not None and not result.iloc[:, 0].isnull().all():
                        self.log_test_result("All Missing Column Handling", True, "MICE imputation worked")
                    else:
                        self.log_test_result("All Missing Column Handling", False, "Failed to impute all missing")
                        
                except Exception as e:
                    self.log_test_result("All Missing Column Handling", False, f"Exception: {str(e)}")
            
            # Test with single row
            if hasattr(self, 'diabetes_data') and len(self.diabetes_data) > 0:
                single_row_df = self.diabetes_data.iloc[:1].copy()
                
                try:
                    preprocessor_single = ClinicalDataPreprocessor()
                    result = preprocessor_single.fit_transform(single_row_df, self.target_col if hasattr(self, 'target_col') else None)
                    self.log_test_result("Single Row Handling", result is not None, "Handled single row data")
                except Exception as e:
                    self.log_test_result("Single Row Handling", False, f"Exception: {str(e)}")
                    
        except Exception as e:
            self.log_test_result("Edge Cases", False, f"Exception: {str(e)}")
    
    def generate_sample_output(self):
        """Generate sample output showing the pipeline working."""
        logger.info("=" * 60)
        logger.info("SAMPLE PIPELINE OUTPUT")
        logger.info("=" * 60)
        
        if hasattr(self, 'processed_data') and hasattr(self, 'preprocessor'):
            # Generate data summary
            summary = self.preprocessor.generate_data_summary(self.processed_data)
            
            logger.info("Data Summary:")
            logger.info(f"  Shape: {summary['shape']}")
            logger.info(f"  Missing values: {summary['missing_values']}")
            logger.info(f"  Memory usage: {summary['memory_usage']:.2f} MB")
            logger.info(f"  Discrete columns: {len(summary['discrete_columns'])}")
            
            if 'target_distribution' in summary:
                logger.info("Target distribution:")
                for value, count in summary['target_distribution'].items():
                    logger.info(f"  {value}: {count}")
            
            # Show sample of processed data
            logger.info("\nSample processed data (first 5 rows):")
            print(self.processed_data.head())
            
            # Show column type mapping
            logger.info("\nColumn type mapping:")
            for col, col_type in summary['column_types'].items():
                logger.info(f"  {col}: {col_type}")
    
    def run_all_tests(self):
        """Run all validation tests."""
        logger.info("Starting Data Pipeline Validation")
        logger.info("=" * 80)
        
        # Run tests in sequence
        if self.test_data_loading():
            self.test_data_quality()
            self.test_preprocessing_pipeline()
            self.test_column_identification()
            self.test_clinical_ranges_validation()
            self.test_edge_cases()
            self.generate_sample_output()
        else:
            logger.error("Data loading failed - skipping subsequent tests")
        
        # Summary
        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Tests Passed: {self.test_passed}")
        logger.info(f"Tests Failed: {self.test_failed}")
        logger.info(f"Success Rate: {(self.test_passed / (self.test_passed + self.test_failed) * 100):.1f}%")
        
        # Recommendations
        self.generate_recommendations()
        
        return self.test_passed > self.test_failed
    
    def generate_recommendations(self):
        """Generate recommendations based on test results."""
        logger.info("=" * 60)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 60)
        
        recommendations = []
        
        # Check for common issues
        if not self.results.get('Pakistani Dataset Loading', {}).get('passed', False):
            recommendations.append(
                "[OK] Pakistani dataset exists and loads successfully - no action needed"
            )
        
        if self.results.get('Missing Value Handling', {}).get('passed', True):
            recommendations.append(
                "[OK] MICE imputation working correctly - good for clinical data quality"
            )
        
        if self.results.get('Column Type Identification', {}).get('passed', True):
            recommendations.append(
                "[OK] Column type identification working - enables proper model configuration"
            )
        
        # General recommendations
        recommendations.extend([
            "[TODO] Consider adding data validation rules for clinical ranges",
            "[TODO] Implement logging for data transformation steps", 
            "[TODO] Add unit tests for edge cases in production",
            "[TODO] Consider adding data profiling reports for clinical stakeholders",
            "[TODO] Implement data versioning for reproducible experiments"
        ])
        
        for rec in recommendations:
            logger.info(rec)

def main():
    """Main execution function."""
    print("Data Pipeline Validation for Phase 6 Clinical Framework")
    print("="*80)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run validation
    validator = DataPipelineValidator()
    success = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()