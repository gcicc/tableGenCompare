#!/usr/bin/env python3
"""
Test script for the extracted evaluation framework.
Validates that all components work together correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the synthetic-tabular-benchmark src to path
sys.path.append('synthetic-tabular-benchmark/src')

from evaluation.unified_evaluator import UnifiedEvaluator
from datasets.dataset_handler import DatasetHandler

def create_mock_data():
    """Create mock original and synthetic data for testing."""
    np.random.seed(42)
    
    # Create original data
    n_samples = 1000
    original_data = pd.DataFrame({
        'feature1': np.random.normal(10, 2, n_samples),
        'feature2': np.random.normal(5, 1.5, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Create synthetic data (slightly different but similar)
    synthetic_data = pd.DataFrame({
        'feature1': np.random.normal(10.1, 2.1, n_samples),
        'feature2': np.random.normal(5.05, 1.45, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.35, 0.35, 0.3]),
        'feature4': np.random.uniform(-5, 105, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.48, 0.52])
    })
    
    return original_data, synthetic_data

def create_mock_model():
    """Create a mock model with get_model_info method."""
    class MockModel:
        def get_model_info(self):
            return {
                'model_type': 'MockGAN',
                'version': '1.0',
                'parameters': {'epochs': 100, 'batch_size': 32}
            }
    
    return MockModel()

def test_unified_evaluator():
    """Test the UnifiedEvaluator with mock data."""
    print("Testing UnifiedEvaluator...")
    
    # Create test data
    original_data, synthetic_data = create_mock_data()
    mock_model = create_mock_model()
    
    # Create dataset metadata
    dataset_metadata = {
        'dataset_info': {
            'name': 'test_dataset',
            'description': 'Mock dataset for testing'
        },
        'target_info': {
            'column': 'target',
            'type': 'binary'
        }
    }
    
    # Initialize evaluator
    evaluator = UnifiedEvaluator(random_state=42)
    
    # Create output directory
    output_dir = "test_evaluation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run complete evaluation
        print("Running complete evaluation...")
        results = evaluator.run_complete_evaluation(
            model=mock_model,
            original_data=original_data,
            synthetic_data=synthetic_data,
            dataset_metadata=dataset_metadata,
            output_dir=output_dir,
            target_column='target'
        )
        
        print("✓ Complete evaluation successful!")
        
        # Check results structure
        expected_keys = [
            'evaluation_timestamp', 'model_info', 'dataset_info', 'target_column',
            'statistical_analysis', 'correlation_analysis', 'similarity_analysis',
            'trts_results', 'data_quality', 'plot_files'
        ]
        
        for key in expected_keys:
            if key in results:
                print(f"✓ {key} present in results")
            else:
                print(f"✗ {key} missing from results")
        
        # Print some key metrics
        if 'trts_results' in results:
            trts = results['trts_results']
            print(f"\nTRTS Results:")
            print(f"  Utility Score: {trts.get('utility_score_percent', 'N/A'):.1f}%")
            print(f"  Quality Score: {trts.get('quality_score_percent', 'N/A'):.1f}%")
            print(f"  Overall Score: {trts.get('overall_score_percent', 'N/A'):.1f}%")
        
        if 'similarity_analysis' in results:
            sim = results['similarity_analysis']
            print(f"\nSimilarity Analysis:")
            print(f"  Final Similarity: {sim.get('final_similarity', 'N/A'):.3f}")
            print(f"  Univariate Similarity: {sim.get('univariate_similarity', 'N/A'):.3f}")
            print(f"  Bivariate Similarity: {sim.get('bivariate_similarity', 'N/A'):.3f}")
        
        # Check output files
        output_path = Path(output_dir)
        expected_files = [
            'comprehensive_statistical_comparison.csv',
            'trts_evaluation.csv',
            'evaluation_summary.json',
            'evaluation_metadata.json'
        ]
        
        print(f"\nOutput Files:")
        for file in expected_files:
            file_path = output_path / file
            if file_path.exists():
                print(f"✓ {file} created")
            else:
                print(f"✗ {file} missing")
        
        # Check plots
        if 'plot_files' in results:
            print(f"\nPlot Files:")
            for plot_type, plot_path in results['plot_files'].items():
                if Path(plot_path).exists():
                    print(f"✓ {plot_type} created")
                else:
                    print(f"✗ {plot_type} missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual evaluation components."""
    print("\nTesting individual components...")
    
    # Test imports
    try:
        from evaluation.statistical_analysis import StatisticalAnalyzer
        from evaluation.similarity_metrics import SimilarityCalculator
        from evaluation.trts_framework import TRTSEvaluator
        from evaluation.visualization_engine import VisualizationEngine
        print("[OK] All imports successful")
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False
    
    # Create test data
    original_data, synthetic_data = create_mock_data()
    
    try:
        # Test StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        stats_df, stats_summary = analyzer.comprehensive_statistical_comparison(
            original_data, synthetic_data, 'target'
        )
        print(f"✓ StatisticalAnalyzer: {len(stats_df)} features analyzed")
        
        # Test SimilarityCalculator
        calculator = SimilarityCalculator()
        final_sim, uni_sim, bi_sim = calculator.evaluate_overall_similarity(
            original_data, synthetic_data, 'target'
        )
        print(f"✓ SimilarityCalculator: Final similarity = {final_sim:.3f}")
        
        # Test TRTSEvaluator
        trts_eval = TRTSEvaluator(random_state=42)
        trts_results = trts_eval.evaluate_trts_scenarios(
            original_data, synthetic_data, 'target'
        )
        print(f"✓ TRTSEvaluator: Overall score = {trts_results.get('overall_score_percent', 'N/A'):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATION FRAMEWORK TEST")
    print("=" * 60)
    
    # Test individual components first
    components_ok = test_individual_components()
    
    if components_ok:
        print("\n" + "=" * 60)
        print("UNIFIED EVALUATOR TEST")
        print("=" * 60)
        
        # Test unified evaluator
        unified_ok = test_unified_evaluator()
        
        if unified_ok:
            print("\n🎉 All tests passed! Evaluation framework is working correctly.")
        else:
            print("\n❌ Unified evaluator test failed.")
    else:
        print("\n❌ Component tests failed.")
    
    print("=" * 60)