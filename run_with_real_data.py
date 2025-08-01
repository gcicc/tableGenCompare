#!/usr/bin/env python3
"""
Run the framework with a real dataset.
You can replace the data loading section with your own dataset.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add the synthetic-tabular-benchmark src to path
sys.path.append('synthetic-tabular-benchmark/src')

from models.model_factory import ModelFactory
from evaluation.unified_evaluator import UnifiedEvaluator

def load_data():
    """
    Load your dataset here. 
    For demo, we'll create a realistic clinical dataset.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate clinical data
    data = pd.DataFrame({
        'age': np.random.normal(55, 15, n_samples).astype(int).clip(18, 90),
        'bmi': np.random.normal(26, 4, n_samples).clip(15, 50),
        'systolic_bp': np.random.normal(130, 20, n_samples).clip(90, 200),
        'glucose': np.random.normal(95, 15, n_samples).clip(70, 300),
        'cholesterol': np.random.normal(200, 40, n_samples).clip(120, 350),
        'diabetes_risk': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    return data

def main():
    print("=" * 60)
    print("RUNNING FRAMEWORK WITH REAL DATA")
    print("=" * 60)
    
    # Load data
    print("Loading dataset...")
    data = load_data()
    print(f"Dataset loaded: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Data types: {data.dtypes.tolist()}")
    
    # Create model
    print("\nCreating GANerAid model...")
    model = ModelFactory.create('ganeraid', device='cpu', random_state=42)
    
    # Train model
    print("Training model (this will take a minute)...")
    training_result = model.train(data, epochs=200, verbose=True)
    print(f"Training completed in {training_result.get('training_duration_seconds', 'N/A'):.2f} seconds")
    
    # Generate synthetic data
    print(f"\nGenerating {len(data)} synthetic samples...")
    synthetic_data = model.generate(len(data))
    print(f"Generated synthetic data: {synthetic_data.shape}")
    
    # Run evaluation
    print("\nRunning comprehensive evaluation...")
    evaluator = UnifiedEvaluator(random_state=42)
    
    dataset_metadata = {
        'dataset_info': {
            'name': 'clinical_demo',
            'description': 'Clinical demo dataset for framework testing'
        },
        'target_info': {
            'column': 'diabetes_risk',
            'type': 'binary'
        }
    }
    
    results = evaluator.run_complete_evaluation(
        model=model,
        original_data=data,
        synthetic_data=synthetic_data,
        dataset_metadata=dataset_metadata,
        output_dir="real_data_results",
        target_column='diabetes_risk'
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {results['model_info']['model_type']}")
    print(f"Dataset: {results['dataset_info']['name']}")
    print(f"Original Data Shape: {data.shape}")
    print(f"Synthetic Data Shape: {synthetic_data.shape}")
    
    print(f"\nTRTS Framework Results:")
    trts = results['trts_results']
    print(f"  Utility Score: {trts['utility_score_percent']:.1f}%")
    print(f"  Quality Score: {trts['quality_score_percent']:.1f}%")
    print(f"  Overall Score: {trts['overall_score_percent']:.1f}%")
    
    print(f"\nSimilarity Analysis:")
    sim = results['similarity_analysis']
    print(f"  Final Similarity: {sim['final_similarity']:.3f}")
    print(f"  Univariate Similarity: {sim['univariate_similarity']:.3f}")
    print(f"  Bivariate Similarity: {sim['bivariate_similarity']:.3f}")
    
    print(f"\nData Quality:")
    quality = results['data_quality']
    print(f"  Type Consistency: {quality['data_type_consistency']:.1f}%")
    print(f"  Range Validity: {quality['range_validity_percentage']:.1f}%")
    print(f"  Column Match: {quality['column_match']}")
    
    overall = results.get('overall_assessment', {})
    print(f"\nOverall Assessment:")
    print(f"  Score: {overall.get('score', 'N/A'):.1f}")
    print(f"  Assessment: {overall.get('assessment', 'N/A')}")
    
    print(f"\nOutput files saved to: real_data_results/")
    print("- comprehensive_statistical_comparison.csv")
    print("- trts_evaluation.csv")
    print("- evaluation_summary.json")
    print("- distribution_comparison.png")
    print("- evaluation_dashboard.png")
    print("- correlation_analysis.png")
    print("- statistical_comparison.png")
    
    print("\n[SUCCESS] Complete framework run finished!")

if __name__ == "__main__":
    main()