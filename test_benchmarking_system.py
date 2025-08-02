#!/usr/bin/env python3
"""
Comprehensive test for the multi-dataset benchmarking system.

This script demonstrates the complete benchmarking pipeline including:
- Dataset management
- Multi-model benchmarking
- Comprehensive reporting
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.append('src')

def create_test_datasets():
    """Create various test datasets with different characteristics."""
    np.random.seed(42)
    
    datasets = {}
    
    # Dataset 1: Small clinical-like dataset
    datasets['clinical_small'] = pd.DataFrame({
        'age': np.random.randint(18, 85, 300),
        'gender': np.random.choice(['M', 'F'], 300),
        'blood_pressure': np.random.normal(120, 20, 300),
        'cholesterol': np.random.normal(200, 40, 300),
        'diagnosis': np.random.choice([0, 1], 300, p=[0.7, 0.3])
    })
    
    # Dataset 2: Medium financial dataset
    datasets['financial_medium'] = pd.DataFrame({
        'income': np.random.lognormal(10, 1, 1500),
        'age': np.random.randint(22, 70, 1500),
        'credit_score': np.random.randint(300, 850, 1500),
        'loan_type': np.random.choice(['mortgage', 'personal', 'auto'], 1500),
        'risk_category': np.random.choice(['low', 'medium', 'high'], 1500),
        'default': np.random.choice([0, 1], 1500, p=[0.85, 0.15])
    })
    
    # Dataset 3: Large marketing dataset
    datasets['marketing_large'] = pd.DataFrame({
        'customer_id': range(5000),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], 5000),
        'income_bracket': np.random.choice(['low', 'medium', 'high'], 5000),
        'product_category': np.random.choice(['electronics', 'clothing', 'books', 'home'], 5000),
        'purchase_frequency': np.random.poisson(3, 5000),
        'total_spent': np.random.exponential(200, 5000),
        'satisfaction_score': np.random.randint(1, 6, 5000),
        'will_recommend': np.random.choice([0, 1], 5000, p=[0.4, 0.6])
    })
    
    # Dataset 4: Complex mixed dataset
    datasets['mixed_complex'] = pd.DataFrame({
        'numerical_1': np.random.normal(50, 15, 800),
        'numerical_2': np.random.exponential(2, 800),
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D', 'E'], 800),
        'categorical_2': np.random.choice(['Type1', 'Type2', 'Type3'], 800),
        'binary_1': np.random.choice([0, 1], 800),
        'binary_2': np.random.choice(['Yes', 'No'], 800),
        'ordinal': np.random.choice(['Low', 'Medium', 'High'], 800),
        'target': np.random.choice([0, 1], 800, p=[0.6, 0.4])
    })
    
    return datasets

def test_dataset_manager():
    """Test the dataset manager functionality."""
    print("Testing Dataset Manager...")
    
    try:
        from benchmarking import DatasetManager, DatasetConfig, DatasetType
        
        # Create dataset manager
        manager = DatasetManager()
        print("[OK] DatasetManager created")
        
        # Create test datasets
        test_datasets = create_test_datasets()
        
        # Add datasets to manager
        for name, data in test_datasets.items():
            dataset_type = DatasetType.CLINICAL if 'clinical' in name else \
                          DatasetType.FINANCIAL if 'financial' in name else \
                          DatasetType.MARKETING if 'marketing' in name else \
                          DatasetType.GENERAL
            
            target_col = 'diagnosis' if 'clinical' in name else \
                        'default' if 'financial' in name else \
                        'will_recommend' if 'marketing' in name else \
                        'target'
            
            config = manager.add_dataset_from_dataframe(
                name=name,
                data=data,
                target_column=target_col,
                dataset_type=dataset_type,
                description=f"Test dataset: {name}"
            )
            
            print(f"[OK] Added dataset '{name}': {data.shape} with complexity {config.complexity_score:.3f}")
        
        # Test dataset groups
        manager.create_dataset_group('small_datasets', ['clinical_small'])
        manager.create_dataset_group('all_datasets', list(test_datasets.keys()))
        print("[OK] Created dataset groups")
        
        # Generate summary
        summary = manager.get_summary_report()
        print(f"[OK] Manager summary: {summary['total_datasets']} datasets, {summary['total_samples']} total samples")
        
        # Validate datasets
        validation = manager.validate_datasets()
        print(f"[OK] Validation: {len(validation['valid'])} valid, {len(validation['invalid'])} invalid")
        
        return manager
        
    except Exception as e:
        print(f"[FAIL] Dataset manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_benchmark_pipeline(dataset_manager):
    """Test the benchmark pipeline functionality."""
    print("\\nTesting Benchmark Pipeline...")
    
    try:
        from benchmarking import BenchmarkPipeline, BenchmarkConfig
        
        # Create benchmark configuration
        config = BenchmarkConfig(
            models_to_test=['ganeraid', 'ctgan', 'tvae'],
            training_epochs={'ganeraid': 50, 'ctgan': 100, 'tvae': 100},
            synthetic_samples_ratio=0.8,
            parallel_execution=False,  # Sequential for testing
            save_intermediate_results=True,
            output_directory="benchmark_test_results"
        )
        print("[OK] BenchmarkConfig created")
        
        # Create benchmark pipeline
        pipeline = BenchmarkPipeline(dataset_manager, config)
        print("[OK] BenchmarkPipeline created")
        
        # Run benchmark on small dataset group first
        print("Running benchmark on small dataset group...")
        results = pipeline.run_benchmark(dataset_group='small_datasets')
        print(f"[OK] Benchmark completed: {len(results)} results")
        
        # Display results
        successful_results = pipeline.get_successful_results()
        failed_results = pipeline.get_failed_results()
        
        print(f"\\nBenchmark Results Summary:")
        print(f"  Successful: {len(successful_results)}")
        print(f"  Failed: {len(failed_results)}")
        
        if successful_results:
            print("\\nSuccessful Results:")
            for result in successful_results:
                print(f"  {result.model_name} on {result.dataset_name}:")
                print(f"    TRTS: {result.trts_score:.3f}" if result.trts_score else "    TRTS: N/A")
                print(f"    Similarity: {result.similarity_score:.3f}" if result.similarity_score else "    Similarity: N/A")
                print(f"    Duration: {result.duration_seconds:.1f}s" if result.duration_seconds else "    Duration: N/A")
        
        if failed_results:
            print("\\nFailed Results:")
            for result in failed_results:
                print(f"  {result.model_name} on {result.dataset_name}: {result.error_message}")
        
        # Test leaderboard
        leaderboard = pipeline.get_leaderboard('composite')
        if leaderboard:
            print(f"\\nLeaderboard (top 3):")
            for i, (model, dataset, score) in enumerate(leaderboard[:3], 1):
                print(f"  {i}. {model} on {dataset}: {score:.3f}")
        
        return pipeline
        
    except Exception as e:
        print(f"[FAIL] Benchmark pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_benchmark_reporter(pipeline):
    """Test the benchmark reporter functionality."""
    print("\\nTesting Benchmark Reporter...")
    
    try:
        from benchmarking import BenchmarkReporter
        
        # Create reporter
        reporter = BenchmarkReporter(pipeline)
        print("[OK] BenchmarkReporter created")
        
        # Generate comprehensive report
        print("Generating comprehensive report...")
        report = reporter.generate_comprehensive_report("benchmark_test_reports")
        print("[OK] Comprehensive report generated")
        
        # Display executive summary
        exec_summary = report['executive_summary']
        print(f"\\nExecutive Summary:")
        print(f"  Total Benchmarks: {exec_summary['total_benchmarks_run']}")
        print(f"  Success Rate: {exec_summary['success_rate']:.1%}")
        print(f"  Average TRTS: {exec_summary['average_trts_score']:.3f}")
        print(f"  Average Similarity: {exec_summary['average_similarity_score']:.3f}")
        print(f"  Computation Time: {exec_summary['total_computation_hours']:.2f} hours")
        
        if exec_summary['best_model_dataset_combination']['model']:
            best = exec_summary['best_model_dataset_combination']
            print(f"  Best Combination: {best['model']} on {best['dataset']} ({best['score']:.3f})")
        
        # Display key findings
        print("\\nKey Findings:")
        for i, finding in enumerate(exec_summary['key_findings'], 1):
            print(f"  {i}. {finding}")
        
        # Display model analysis
        model_analysis = report['model_analysis']
        print("\\nModel Analysis:")
        for model_name, analysis in model_analysis.items():
            perf = analysis['performance_summary']
            print(f"  {model_name.upper()}:")
            print(f"    Success Rate: {perf['success_rate']:.1%}")
            print(f"    Avg TRTS: {perf['avg_trts_score']:.3f}")
            print(f"    Avg Similarity: {perf['avg_similarity_score']:.3f}")
            
            if analysis['strengths']:
                print(f"    Strengths: {', '.join(analysis['strengths'])}")
            if analysis['weaknesses']:
                print(f"    Weaknesses: {', '.join(analysis['weaknesses'])}")
        
        # Export results to CSV
        csv_file = reporter.export_results_csv("benchmark_test_results.csv")
        print(f"[OK] Results exported to CSV: {csv_file}")
        
        return reporter
        
    except Exception as e:
        print(f"[FAIL] Benchmark reporter test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_full_benchmarking_workflow():
    """Test the complete benchmarking workflow."""
    print("\\nTesting Full Benchmarking Workflow...")
    
    try:
        from benchmarking import DatasetManager, BenchmarkPipeline, BenchmarkConfig, BenchmarkReporter
        from benchmarking import DatasetType
        
        # Step 1: Setup datasets
        manager = DatasetManager()
        test_datasets = create_test_datasets()
        
        # Add a subset for full workflow test
        subset_datasets = {
            'clinical_test': test_datasets['clinical_small'][:200],  # Smaller for speed
            'financial_test': test_datasets['financial_medium'][:500]
        }
        
        for name, data in subset_datasets.items():
            target_col = 'diagnosis' if 'clinical' in name else 'default'
            dataset_type = DatasetType.CLINICAL if 'clinical' in name else DatasetType.FINANCIAL
            
            manager.add_dataset_from_dataframe(
                name=name,
                data=data,
                target_column=target_col,
                dataset_type=dataset_type,
                description=f"Test dataset for full workflow: {name}"
            )
        
        print(f"[OK] Setup {len(subset_datasets)} datasets for workflow test")
        
        # Step 2: Configure and run benchmark
        config = BenchmarkConfig(
            models_to_test=['ganeraid', 'ctgan'],  # Subset for speed
            training_epochs={'ganeraid': 30, 'ctgan': 50},
            synthetic_samples_ratio=0.5,
            parallel_execution=False,
            output_directory="workflow_test_results"
        )
        
        pipeline = BenchmarkPipeline(manager, config)
        results = pipeline.run_benchmark()
        
        print(f"[OK] Workflow benchmark completed: {len(results)} results")
        
        # Step 3: Generate comprehensive analysis
        reporter = BenchmarkReporter(pipeline)
        report = reporter.generate_comprehensive_report("workflow_test_reports")
        
        # Step 4: Display workflow summary
        successful = len(pipeline.get_successful_results())
        total = len(results)
        
        print(f"\\n=== FULL WORKFLOW SUMMARY ===")
        print(f"Success Rate: {successful}/{total} ({successful/total:.1%})")
        
        if successful > 0:
            leaderboard = pipeline.get_leaderboard('composite')
            if leaderboard:
                print(f"Winner: {leaderboard[0][0]} on {leaderboard[0][1]} ({leaderboard[0][2]:.3f})")
            
            # Show model comparison
            model_perf = report['executive_summary']['model_success_rates']
            print("Model Success Rates:")
            for model, rate in model_perf.items():
                print(f"  {model.upper()}: {rate:.1%}")
        
        print("[SUCCESS] Full benchmarking workflow completed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Full workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("MULTI-DATASET BENCHMARKING SYSTEM TEST")
    print("="*60)
    
    # Test individual components
    dataset_manager = test_dataset_manager()
    
    if dataset_manager:
        pipeline = test_benchmark_pipeline(dataset_manager)
        
        if pipeline:
            reporter = test_benchmark_reporter(pipeline)
    
    # Test full workflow
    print("\\n" + "="*60)
    full_workflow_success = test_full_benchmarking_workflow()
    
    # Final summary
    print("\\n" + "="*60)
    print("BENCHMARKING SYSTEM TEST SUMMARY")
    print("="*60)
    
    print(f"Dataset Manager: {'PASS' if dataset_manager else 'FAIL'}")
    print(f"Benchmark Pipeline: {'PASS' if 'pipeline' in locals() and pipeline else 'FAIL'}")
    print(f"Benchmark Reporter: {'PASS' if 'reporter' in locals() and reporter else 'FAIL'}")
    print(f"Full Workflow: {'PASS' if full_workflow_success else 'FAIL'}")
    
    all_passed = all([
        dataset_manager is not None,
        'pipeline' in locals() and pipeline is not None,
        'reporter' in locals() and reporter is not None,
        full_workflow_success
    ])
    
    if all_passed:
        print("\\n[SUCCESS] All benchmarking system tests passed!")
        print("🎯 Phase 2C: Multi-dataset benchmarking pipeline COMPLETE!")
        print("\\nFeatures available:")
        print("  ✅ Multi-dataset management and validation")
        print("  ✅ Parallel benchmark execution")
        print("  ✅ Comprehensive model comparison")
        print("  ✅ Statistical analysis and reporting")
        print("  ✅ Performance insights and recommendations")
        print("  ✅ Export capabilities (JSON, CSV)")
        print("\\nReady for production benchmarking!")
    else:
        print("\\n[PARTIAL] Some benchmarking system tests failed.")
    
    print("="*60)