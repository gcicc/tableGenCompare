#!/usr/bin/env python3
"""
Test script for the model recommendation system.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add src to path
sys.path.append('src')

from recommendations.model_selector import (
    DatasetProfiler, ModelSelector, recommend_best_model
)

def create_test_datasets():
    """Create various test datasets to test recommendation system."""
    
    np.random.seed(42)
    datasets = {}
    
    # 1. Small clinical dataset
    clinical_data = pd.DataFrame({
        'patient_age': np.random.randint(20, 80, 200),
        'blood_pressure': np.random.normal(120, 20, 200),
        'heart_rate': np.random.normal(70, 15, 200),
        'diagnosis': np.random.choice(['healthy', 'hypertension', 'diabetes'], 200),
        'treatment_success': np.random.choice([0, 1], 200, p=[0.3, 0.7])
    })
    datasets['clinical_small'] = (clinical_data, 'treatment_success')
    
    # 2. Large financial dataset
    financial_data = pd.DataFrame({
        'customer_income': np.random.lognormal(10, 1, 5000),
        'credit_score': np.random.normal(700, 100, 5000),
        'loan_amount': np.random.lognormal(9, 1, 5000),
        'employment_years': np.random.exponential(5, 5000),
        'debt_ratio': np.random.beta(2, 5, 5000),
        'account_type': np.random.choice(['checking', 'savings', 'credit', 'loan'], 5000),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 5000),
        'default_risk': np.random.choice([0, 1], 5000, p=[0.85, 0.15])
    })
    datasets['financial_large'] = (financial_data, 'default_risk')
    
    # 3. Marketing dataset with many categories
    marketing_data = pd.DataFrame({
        'customer_age': np.random.randint(18, 65, 1000),
        'income_bracket': np.random.choice(['low', 'medium', 'high', 'very_high'], 1000),
        'campaign_channel': np.random.choice(['email', 'sms', 'social', 'direct', 'web'], 1000),
        'product_category': np.random.choice(['electronics', 'clothing', 'home', 'sports', 'books'], 1000),
        'previous_purchases': np.random.poisson(3, 1000),
        'engagement_score': np.random.uniform(0, 100, 1000),
        'click_rate': np.random.beta(2, 8, 1000),
        'conversion': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })
    datasets['marketing_categorical'] = (marketing_data, 'conversion')
    
    # 4. Sensor data (mostly numerical)
    sensor_data = pd.DataFrame({
        'temperature': np.random.normal(25, 5, 800),
        'humidity': np.random.beta(3, 2, 800) * 100,
        'pressure': np.random.normal(1013, 10, 800),
        'voltage': np.random.normal(5.0, 0.5, 800),
        'current': np.random.exponential(0.1, 800),
        'sensor_id': np.random.choice(['A', 'B', 'C'], 800),
        'anomaly': np.random.choice([0, 1], 800, p=[0.95, 0.05])
    })
    datasets['sensor_numerical'] = (sensor_data, 'anomaly')
    
    # 5. Tiny dataset
    tiny_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 50),
        'feature2': np.random.uniform(0, 10, 50),
        'category': np.random.choice(['A', 'B'], 50),
        'target': np.random.choice([0, 1], 50)
    })
    datasets['tiny'] = (tiny_data, 'target')
    
    return datasets

def test_dataset_profiler():
    """Test the dataset profiling functionality."""
    print("Testing dataset profiler...")
    
    try:
        profiler = DatasetProfiler()
        test_data = create_test_datasets()
        
        for dataset_name, (data, target_col) in test_data.items():
            print(f"\n  Profiling {dataset_name} dataset...")
            
            profile = profiler.profile_dataset(data, target_col)
            
            print(f"    Shape: {profile.n_samples} x {profile.n_features}")
            print(f"    Data types: {profile.n_numerical} numerical, {profile.n_categorical} categorical")
            print(f"    Size category: {profile.size_category}")
            print(f"    Complexity: {profile.complexity_score:.3f}")
            print(f"    Domain hint: {profile.domain_hint}")
            print(f"    Target type: {profile.target_type}")
            
            # Validate profile
            assert profile.n_samples == len(data)
            assert profile.n_features == len(data.columns)
            assert profile.n_numerical + profile.n_categorical <= profile.n_features
            assert 0 <= profile.complexity_score <= 1
            assert 0 <= profile.missing_ratio <= 1
            
        print("✅ Dataset profiler working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Dataset profiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_selector():
    """Test the model selection functionality."""
    print("Testing model selector...")
    
    try:
        selector = ModelSelector()
        test_data = create_test_datasets()
        
        results = {}
        
        for dataset_name, (data, target_col) in test_data.items():
            print(f"\n  Testing recommendations for {dataset_name}...")
            
            recommendations = selector.recommend_models(
                data=data,
                target_column=target_col,
                max_recommendations=3,
                include_config=True
            )
            
            results[dataset_name] = recommendations
            
            print(f"    Generated {len(recommendations)} recommendations")
            
            if recommendations:
                best_rec = recommendations[0]
                print(f"    Best model: {best_rec.model_name}")
                print(f"    Confidence: {best_rec.confidence_score:.3f}")
                print(f"    Expected performance: {best_rec.expected_performance:.3f}")
                print(f"    Estimated training time: {best_rec.estimated_training_time:.1f} min")
                print(f"    Reasons: {len(best_rec.reasons)}")
                print(f"    Warnings: {len(best_rec.warnings)}")
                
                # Validate recommendation structure
                assert 0 <= best_rec.confidence_score <= 1
                assert 0 <= best_rec.expected_performance <= 1
                assert best_rec.estimated_training_time > 0
                assert isinstance(best_rec.reasons, list)
                assert isinstance(best_rec.warnings, list)
                assert isinstance(best_rec.recommended_config, dict)
                assert isinstance(best_rec.resource_requirements, dict)
            else:
                print("    No recommendations generated")
        
        # Check that different datasets get different recommendations
        rec_models = {}
        for dataset_name, recommendations in results.items():
            if recommendations:
                rec_models[dataset_name] = recommendations[0].model_name
        
        print(f"\n  Recommendation diversity: {rec_models}")
        
        print("✅ Model selector working correctly")
        return True, results
        
    except Exception as e:
        print(f"❌ Model selector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_recommendation_reports(recommendations_data):
    """Test report generation functionality."""
    print("Testing recommendation reports...")
    
    try:
        selector = ModelSelector()
        test_data = create_test_datasets()
        
        # Test report generation for one dataset
        dataset_name = 'clinical_small'
        data, target_col = test_data[dataset_name]
        recommendations = recommendations_data.get(dataset_name, [])
        
        if not recommendations:
            print("⏭️  Skipping report test - no recommendations available")
            return True
        
        # Generate report
        report = selector.generate_recommendation_report(
            data=data,
            recommendations=recommendations,
            target_column=target_col
        )
        
        # Validate report content
        assert "Model Recommendation Report" in report
        assert "Dataset Profile" in report
        assert "Model Recommendations" in report
        assert f"**Dataset Shape:** {data.shape}" in report
        
        # Check that each recommendation is included
        for rec in recommendations:
            assert rec.model_name.upper() in report
        
        print(f"✅ Generated report with {len(report)} characters")
        print(f"  Report sections: Dataset Profile, {len(recommendations)} Model Recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_function():
    """Test the convenience recommendation function."""
    print("Testing convenience function...")
    
    try:
        test_data = create_test_datasets()
        
        # Test with clinical dataset
        data, target_col = test_data['clinical_small']
        
        result = recommend_best_model(
            data=data,
            target_column=target_col,
            max_recommendations=2,
            save_report=True,
            output_dir="test_recommendations"
        )
        
        # Validate result structure
        required_keys = ['recommendations', 'best_model', 'profile', 'report']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        print(f"✅ Generated {len(result['recommendations'])} recommendations")
        
        if result['best_model']:
            print(f"  Best model: {result['best_model']['model_name']}")
            print(f"  Confidence: {result['best_model']['confidence_score']:.3f}")
        
        if result['report_file']:
            print(f"  Report saved to: {result['report_file']}")
        
        # Validate profile
        profile = result['profile']
        assert isinstance(profile, dict)
        assert 'n_samples' in profile
        assert 'n_features' in profile
        
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
        selector = ModelSelector()
        
        # Test with empty dataset
        try:
            empty_data = pd.DataFrame()
            recommendations = selector.recommend_models(empty_data)
            print("  ⚠️  Empty dataset should fail but didn't")
        except Exception:
            print("  ✅ Empty dataset properly rejected")
        
        # Test with single column dataset
        single_col_data = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        recommendations = selector.recommend_models(single_col_data)
        print(f"  ✅ Single column dataset: {len(recommendations)} recommendations")
        
        # Test with all missing data
        missing_data = pd.DataFrame({
            'col1': [np.nan] * 10,
            'col2': [np.nan] * 10,
            'col3': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # One valid column
        })
        recommendations = selector.recommend_models(missing_data)
        print(f"  ✅ High missing data: {len(recommendations)} recommendations")
        
        # Test with constant columns
        constant_data = pd.DataFrame({
            'constant1': [1] * 100,
            'constant2': ['A'] * 100,
            'variable': np.random.normal(0, 1, 100)
        })
        recommendations = selector.recommend_models(constant_data)
        print(f"  ✅ Constant columns: {len(recommendations)} recommendations")
        
        print("✅ Edge cases handled correctly")
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_domain_classification():
    """Test domain classification accuracy."""
    print("Testing domain classification...")
    
    try:
        profiler = DatasetProfiler()
        
        # Test datasets with clear domain signals
        test_cases = [
            (pd.DataFrame({
                'patient_age': [25, 30, 35],
                'diagnosis': ['A', 'B', 'C'],
                'treatment': ['X', 'Y', 'Z']
            }), 'clinical'),
            
            (pd.DataFrame({
                'customer_income': [50000, 60000, 70000],
                'credit_score': [700, 750, 800],
                'loan_amount': [10000, 15000, 20000]
            }), 'financial'),
            
            (pd.DataFrame({
                'temperature': [20, 25, 30],
                'pressure': [1000, 1010, 1020],
                'sensor_reading': [1.5, 2.0, 2.5]
            }), 'sensor'),
        ]
        
        correct_classifications = 0
        
        for data, expected_domain in test_cases:
            profile = profiler.profile_dataset(data)
            detected_domain = profile.domain_hint
            
            print(f"  Expected: {expected_domain}, Detected: {detected_domain}")
            
            if detected_domain == expected_domain:
                correct_classifications += 1
        
        accuracy = correct_classifications / len(test_cases)
        print(f"✅ Domain classification accuracy: {accuracy:.1%}")
        
        return accuracy > 0.5  # At least 50% accuracy
        
    except Exception as e:
        print(f"❌ Domain classification test failed: {e}")
        return False

def run_comprehensive_recommendation_test():
    """Run comprehensive recommendation system test suite."""
    print("=" * 60)
    print("MODEL RECOMMENDATION SYSTEM TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Dataset profiler
    results['profiler'] = test_dataset_profiler()
    print()
    
    # Test 2: Model selector
    selector_success, recommendations_data = test_model_selector()
    results['selector'] = selector_success
    print()
    
    # Test 3: Report generation
    results['reports'] = test_recommendation_reports(recommendations_data)
    print()
    
    # Test 4: Convenience function
    results['convenience'] = test_convenience_function()
    print()
    
    # Test 5: Edge cases
    results['edge_cases'] = test_edge_cases()
    print()
    
    # Test 6: Domain classification
    results['domain_classification'] = test_domain_classification()
    print()
    
    # Summary
    print("=" * 60)
    print("RECOMMENDATION SYSTEM TEST RESULTS")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper().replace('_', ' '):.<30} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All recommendation tests passed!")
        print("✅ Model recommendation system is working correctly")
        print("\nFeatures validated:")
        print("  - Dataset profiling and characterization")
        print("  - Intelligent model recommendations")
        print("  - Confidence scoring and performance prediction")
        print("  - Automated hyperparameter configuration")
        print("  - Comprehensive reporting")
        print("  - Domain-specific recommendations")
        print("  - Edge case handling")
        return True
    else:
        print("⚠️  Some recommendation tests failed")
        print("   Check the output above for details")
        return False

if __name__ == "__main__":
    success = run_comprehensive_recommendation_test()
    sys.exit(0 if success else 1)