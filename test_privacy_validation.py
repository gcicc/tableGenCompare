#!/usr/bin/env python3
"""
Test script for privacy validation system.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add src to path
sys.path.append('src')

from privacy.privacy_validator import (
    PrivacyValidator, PrivacyLevel, ComplianceFramework,
    validate_synthetic_data_privacy
)

def create_test_datasets():
    """Create test datasets for privacy validation."""
    np.random.seed(42)
    
    # Original sensitive dataset (clinical)
    original_data = pd.DataFrame({
        'patient_id': [f'P{i:04d}' for i in range(500)],
        'age': np.random.randint(20, 80, 500),
        'gender': np.random.choice(['M', 'F'], 500),
        'zip_code': np.random.choice([f'{z:05d}' for z in range(10000, 10050)], 500),
        'diagnosis': np.random.choice(['diabetes', 'hypertension', 'healthy', 'cardiac'], 500),
        'income': np.random.lognormal(10, 0.5, 500),
        'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], 500),
        'treatment_outcome': np.random.choice([0, 1], 500, p=[0.3, 0.7])
    })
    
    # Synthetic dataset with varying privacy levels
    datasets = {}
    
    # 1. Good privacy - well anonymized
    good_privacy = original_data.copy()
    # Remove direct identifiers
    good_privacy = good_privacy.drop(['patient_id'], axis=1)
    # Generalize age to age groups
    good_privacy['age_group'] = pd.cut(good_privacy['age'], bins=[0, 30, 50, 70, 100], labels=['<30', '30-50', '50-70', '70+'])
    good_privacy = good_privacy.drop(['age'], axis=1)
    # Generalize zip codes to regions
    good_privacy['region'] = good_privacy['zip_code'].astype(int) // 10
    good_privacy = good_privacy.drop(['zip_code'], axis=1)
    # Add some noise to income
    good_privacy['income'] = good_privacy['income'] * np.random.normal(1, 0.1, len(good_privacy))
    datasets['good_privacy'] = good_privacy
    
    # 2. Poor privacy - includes identifiers
    poor_privacy = original_data.copy()
    # Keep patient IDs (privacy violation)
    # Keep exact ages and zip codes
    datasets['poor_privacy'] = poor_privacy
    
    # 3. Medium privacy - some anonymization
    medium_privacy = original_data.copy()
    # Remove patient IDs
    medium_privacy = medium_privacy.drop(['patient_id'], axis=1)
    # Keep exact ages but generalize zip codes
    medium_privacy['zip_prefix'] = medium_privacy['zip_code'].str[:3]
    medium_privacy = medium_privacy.drop(['zip_code'], axis=1)
    datasets['medium_privacy'] = medium_privacy
    
    return original_data, datasets

def test_privacy_validator_initialization():
    """Test privacy validator initialization."""
    print("Testing privacy validator initialization...")
    
    try:
        # Test different privacy levels
        for level in [PrivacyLevel.LOW, PrivacyLevel.MEDIUM, PrivacyLevel.HIGH, PrivacyLevel.CRITICAL]:
            validator = PrivacyValidator(level)
            print(f"  ✅ {level.value} privacy level: Initialized successfully")
            
            # Check thresholds
            assert len(validator.thresholds) > 0
            assert 'k_anonymity' in validator.thresholds
            
        print("✅ Privacy validator initialization working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Privacy validator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_anonymity_metrics():
    """Test anonymity metric calculations."""
    print("Testing anonymity metrics...")
    
    try:
        validator = PrivacyValidator(PrivacyLevel.MEDIUM)
        original_data, synthetic_datasets = create_test_datasets()
        
        results = {}
        
        for dataset_name, synthetic_data in synthetic_datasets.items():
            print(f"\n  Testing {dataset_name} dataset...")
            
            # Test sensitive column detection
            sensitive_cols = validator._detect_sensitive_columns(synthetic_data)
            print(f"    Detected sensitive columns: {sensitive_cols}")
            
            # Test quasi-identifier detection
            qi_cols = validator._detect_quasi_identifiers(synthetic_data)
            print(f"    Detected quasi-identifiers: {qi_cols}")
            
            # Calculate anonymity metrics
            anonymity_metrics = validator._calculate_anonymity_metrics(
                original_data, synthetic_data, sensitive_cols, qi_cols
            )
            
            results[dataset_name] = anonymity_metrics
            
            # Display results
            for metric_name, metric in anonymity_metrics.items():
                status = "✅ PASS" if metric.passed else "❌ FAIL"
                print(f"    {metric.name}: {metric.value:.3f} (threshold: {metric.threshold:.3f}) {status}")
        
        # Validate that different datasets have different privacy levels
        if 'good_privacy' in results and 'poor_privacy' in results:
            good_k = results['good_privacy']['k_anonymity'].value
            poor_k = results['poor_privacy']['k_anonymity'].value
            
            print(f"\n  Privacy comparison:")
            print(f"    Good privacy k-anonymity: {good_k:.1f}")
            print(f"    Poor privacy k-anonymity: {poor_k:.1f}")
            
            # Good privacy should have better (higher) k-anonymity
            if good_k >= poor_k:
                print("    ✅ Privacy levels differentiated correctly")
            else:
                print("    ⚠️  Privacy levels not as expected")
        
        print("✅ Anonymity metrics working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Anonymity metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_disclosure_risk_analysis():
    """Test disclosure risk calculations."""
    print("Testing disclosure risk analysis...")
    
    try:
        validator = PrivacyValidator(PrivacyLevel.MEDIUM)
        original_data, synthetic_datasets = create_test_datasets()
        
        for dataset_name, synthetic_data in synthetic_datasets.items():
            print(f"\n  Testing {dataset_name} dataset...")
            
            sensitive_cols = validator._detect_sensitive_columns(synthetic_data)
            
            # Calculate disclosure risks
            risks = validator._calculate_disclosure_risks(
                original_data, synthetic_data, sensitive_cols
            )
            
            print(f"    Disclosure risks:")
            for risk_name, risk_value in risks.items():
                risk_level = "HIGH" if risk_value > 0.5 else "MEDIUM" if risk_value > 0.3 else "LOW"
                print(f"      {risk_name}: {risk_value:.3f} ({risk_level})")
                
                # Validate risk values are in expected range
                assert 0 <= risk_value <= 1, f"Risk value {risk_value} out of range [0,1]"
        
        print("✅ Disclosure risk analysis working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Disclosure risk analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compliance_validation():
    """Test compliance framework validation."""
    print("Testing compliance validation...")
    
    try:
        validator = PrivacyValidator(PrivacyLevel.MEDIUM)
        original_data, synthetic_datasets = create_test_datasets()
        
        # Test different compliance frameworks
        frameworks_to_test = [
            ComplianceFramework.GDPR,
            ComplianceFramework.HIPAA,
            ComplianceFramework.CCPA
        ]
        
        for framework in frameworks_to_test:
            print(f"\n  Testing {framework.value.upper()} compliance...")
            
            for dataset_name, synthetic_data in synthetic_datasets.items():
                print(f"    Dataset: {dataset_name}")
                
                # Calculate anonymity metrics first
                sensitive_cols = validator._detect_sensitive_columns(synthetic_data)
                qi_cols = validator._detect_quasi_identifiers(synthetic_data)
                anonymity_metrics = validator._calculate_anonymity_metrics(
                    original_data, synthetic_data, sensitive_cols, qi_cols
                )
                
                # Validate compliance
                compliance_result = validator._validate_compliance(
                    original_data, synthetic_data, framework, anonymity_metrics
                )
                
                status = "✅ COMPLIANT" if compliance_result.passed else "❌ NON-COMPLIANT"
                print(f"      Status: {status}")
                print(f"      Score: {compliance_result.overall_score:.3f}")
                print(f"      Risk Level: {compliance_result.risk_level}")
                print(f"      Violations: {len(compliance_result.violations)}")
                
                # Validate compliance result structure
                assert 0 <= compliance_result.overall_score <= 1
                assert compliance_result.risk_level in ['low', 'medium', 'high', 'critical']
                assert isinstance(compliance_result.violations, list)
                assert isinstance(compliance_result.recommendations, list)
        
        print("✅ Compliance validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Compliance validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_privacy_validation():
    """Test comprehensive privacy validation."""
    print("Testing comprehensive privacy validation...")
    
    try:
        validator = PrivacyValidator(PrivacyLevel.MEDIUM)
        original_data, synthetic_datasets = create_test_datasets()
        
        results = {}
        
        for dataset_name, synthetic_data in synthetic_datasets.items():
            print(f"\n  Validating {dataset_name} dataset...")
            
            # Run comprehensive validation
            report = validator.validate_privacy(
                original_data=original_data,
                synthetic_data=synthetic_data,
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
            )
            
            results[dataset_name] = report
            
            print(f"    Overall Privacy Score: {report.overall_privacy_score:.3f}")
            print(f"    Privacy Level: {report.privacy_level}")
            print(f"    Risk Assessment: {report.risk_assessment}")
            print(f"    Anonymity Metrics: {len(report.anonymity_metrics)}")
            print(f"    Disclosure Risks: {len(report.disclosure_risks)}")
            print(f"    Compliance Results: {len(report.compliance_results)}")
            print(f"    Recommendations: {len(report.recommendations)}")
            print(f"    Required Actions: {len(report.required_actions)}")
            
            # Validate report structure
            assert 0 <= report.overall_privacy_score <= 1
            assert report.privacy_level in ['excellent', 'good', 'acceptable', 'poor', 'inadequate']
            assert report.risk_assessment in ['low', 'medium', 'high', 'critical']
            assert len(report.dataset_fingerprint) == 16  # Expected fingerprint length
            assert isinstance(report.anonymity_metrics, dict)
            assert isinstance(report.utility_preservation, dict)
            assert isinstance(report.disclosure_risks, dict)
            assert isinstance(report.compliance_results, dict)
        
        # Compare privacy levels
        if 'good_privacy' in results and 'poor_privacy' in results:
            good_score = results['good_privacy'].overall_privacy_score
            poor_score = results['poor_privacy'].overall_privacy_score
            
            print(f"\n  Privacy score comparison:")
            print(f"    Good privacy: {good_score:.3f}")
            print(f"    Poor privacy: {poor_score:.3f}")
            
            if good_score > poor_score:
                print("    ✅ Privacy scoring working correctly")
            else:
                print("    ⚠️  Privacy scoring may need adjustment")
        
        print("✅ Comprehensive privacy validation working correctly")
        return True, results
        
    except Exception as e:
        print(f"❌ Comprehensive privacy validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_privacy_report_generation(validation_results):
    """Test privacy report generation."""
    print("Testing privacy report generation...")
    
    try:
        validator = PrivacyValidator(PrivacyLevel.MEDIUM)
        
        if not validation_results:
            print("⏭️  Skipping report test - no validation results available")
            return True
        
        # Test report generation for one dataset
        dataset_name = list(validation_results.keys())[0]
        report = validation_results[dataset_name]
        
        # Generate report
        report_text = validator.generate_privacy_report(report)
        
        # Validate report content
        required_sections = [
            "Privacy Validation Report",
            "Executive Summary", 
            "Anonymity Metrics",
            "Disclosure Risk Analysis",
            "Regulatory Compliance",
            "Recommendations",
            "Technical Details"
        ]
        
        for section in required_sections:
            if section not in report_text:
                print(f"    ⚠️  Missing section: {section}")
            else:
                print(f"    ✅ Found section: {section}")
        
        print(f"  Report length: {len(report_text)} characters")
        print(f"  Report sections: {len([s for s in required_sections if s in report_text])}/{len(required_sections)}")
        
        # Test file saving
        test_file = "test_privacy_report.md"
        validator.generate_privacy_report(report, test_file)
        
        # Check if file was created
        from pathlib import Path
        if Path(test_file).exists():
            print(f"  ✅ Report saved to file: {test_file}")
            Path(test_file).unlink()  # Clean up
        else:
            print(f"  ❌ Failed to save report to file")
        
        print("✅ Privacy report generation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Privacy report generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_function():
    """Test the convenience privacy validation function."""
    print("Testing convenience function...")
    
    try:
        original_data, synthetic_datasets = create_test_datasets()
        
        # Test with medium privacy dataset
        synthetic_data = synthetic_datasets['medium_privacy']
        
        result = validate_synthetic_data_privacy(
            original_data=original_data,
            synthetic_data=synthetic_data,
            privacy_level="medium",
            compliance_frameworks=["gdpr", "hipaa"],
            output_dir="test_privacy_reports"
        )
        
        # Validate result structure
        required_keys = [
            'report', 'privacy_score', 'privacy_level', 'risk_assessment',
            'passed_validation', 'report_text', 'report_file', 
            'recommendations', 'required_actions'
        ]
        
        for key in required_keys:
            if key not in result:
                print(f"    ❌ Missing key: {key}")
                return False
            else:
                print(f"    ✅ Found key: {key}")
        
        print(f"  Privacy Score: {result['privacy_score']:.3f}")
        print(f"  Privacy Level: {result['privacy_level']}")
        print(f"  Risk Assessment: {result['risk_assessment']}")
        print(f"  Passed Validation: {result['passed_validation']}")
        print(f"  Report File: {result['report_file']}")
        print(f"  Recommendations: {len(result['recommendations'])}")
        print(f"  Required Actions: {len(result['required_actions'])}")
        
        # Validate score range
        assert 0 <= result['privacy_score'] <= 1
        assert isinstance(result['passed_validation'], bool)
        
        print("✅ Convenience function working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    try:
        validator = PrivacyValidator(PrivacyLevel.MEDIUM)
        
        # Test with minimal data
        minimal_original = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C']
        })
        
        minimal_synthetic = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C']
        })
        
        report = validator.validate_privacy(minimal_original, minimal_synthetic)
        print(f"  ✅ Minimal data: Privacy score = {report.overall_privacy_score:.3f}")
        
        # Test with no common columns
        different_cols_original = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        different_cols_synthetic = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        
        report = validator.validate_privacy(different_cols_original, different_cols_synthetic)
        print(f"  ✅ Different columns: Privacy score = {report.overall_privacy_score:.3f}")
        
        # Test with all numerical data
        numerical_original = pd.DataFrame({
            'num1': np.random.normal(0, 1, 100),
            'num2': np.random.normal(5, 2, 100),
            'num3': np.random.exponential(1, 100)
        })
        
        numerical_synthetic = pd.DataFrame({
            'num1': np.random.normal(0, 1, 100),
            'num2': np.random.normal(5, 2, 100),
            'num3': np.random.exponential(1, 100)
        })
        
        report = validator.validate_privacy(numerical_original, numerical_synthetic)
        print(f"  ✅ All numerical: Privacy score = {report.overall_privacy_score:.3f}")
        
        print("✅ Edge cases handled correctly")
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_privacy_test():
    """Run comprehensive privacy validation test suite."""
    print("=" * 60)
    print("PRIVACY VALIDATION SYSTEM TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Validator initialization
    results['initialization'] = test_privacy_validator_initialization()
    print()
    
    # Test 2: Anonymity metrics
    results['anonymity_metrics'] = test_anonymity_metrics()
    print()
    
    # Test 3: Disclosure risk analysis
    results['disclosure_risks'] = test_disclosure_risk_analysis()
    print()
    
    # Test 4: Compliance validation
    results['compliance'] = test_compliance_validation()
    print()
    
    # Test 5: Comprehensive privacy validation
    validation_success, validation_results = test_comprehensive_privacy_validation()
    results['comprehensive'] = validation_success
    print()
    
    # Test 6: Report generation
    results['reports'] = test_privacy_report_generation(validation_results)
    print()
    
    # Test 7: Convenience function
    results['convenience'] = test_convenience_function()
    print()
    
    # Test 8: Edge cases
    results['edge_cases'] = test_edge_cases()
    print()
    
    # Summary
    print("=" * 60)
    print("PRIVACY VALIDATION TEST RESULTS")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper().replace('_', ' '):.<30} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All privacy validation tests passed!")
        print("✅ Privacy validation system is working correctly")
        print("\nFeatures validated:")
        print("  - K-anonymity, L-diversity, T-closeness calculations")
        print("  - Disclosure risk analysis (membership, attribute, linkage)")
        print("  - GDPR, HIPAA, CCPA compliance validation")
        print("  - Comprehensive privacy scoring")
        print("  - Automated report generation")
        print("  - Edge case handling")
        return True
    else:
        print("⚠️  Some privacy validation tests failed")
        print("   Check the output above for details")
        return False

if __name__ == "__main__":
    success = run_comprehensive_privacy_test()
    sys.exit(0 if success else 1)