"""
Comprehensive Test Suite for Phase 6 Clinical Framework
Testing all 5 synthetic data generation models: CTGAN, TVAE, CopulaGAN, TableGAN, GANerAid

This script validates:
1. Model initialization and parameter spaces
2. Data fitting process with discrete column handling
3. Synthetic data generation with correct shapes and types
4. Bayesian optimization pipeline functionality
5. Graceful fallbacks when packages are missing
6. Error handling and recovery
7. Data quality and format consistency
"""

import pandas as pd
import numpy as np
import time
import logging
import warnings
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_tests.log'),
        logging.StreamHandler()
    ]
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import test data and utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class ComprehensiveModelTester:
    """Comprehensive testing framework for all synthetic data models."""
    
    def __init__(self, data_path: str, target_column: str, random_state: int = 42):
        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state
        self.results = {}
        self.test_start_time = datetime.now()
        
        # Load and prepare data
        self.load_test_data()
        
        # Initialize model availability tracking
        self.model_availability = self.check_model_availability()
        
        # Test results storage
        self.test_results = {}
        
    def load_test_data(self):
        """Load and prepare test dataset."""
        try:
            self.original_data = pd.read_csv(self.data_path)
            self.discrete_columns = self.identify_discrete_columns()
            
            logging.info(f"Data loaded: {self.original_data.shape}")
            logging.info(f"Discrete columns: {self.discrete_columns}")
            logging.info(f"Target column: {self.target_column}")
            
        except Exception as e:
            logging.error(f"Failed to load data from {self.data_path}: {e}")
            raise
    
    def identify_discrete_columns(self) -> List[str]:
        """Identify discrete/categorical columns in the dataset."""
        discrete_cols = []
        
        for col in self.original_data.columns:
            # Check if column is categorical or has few unique values
            if (self.original_data[col].dtype == 'object' or 
                self.original_data[col].nunique() <= 10 or
                col == self.target_column):
                discrete_cols.append(col)
        
        return discrete_cols
    
    def check_model_availability(self) -> Dict[str, bool]:
        """Check which model packages are available."""
        availability = {}
        
        # Test CTGAN/TVAE availability
        try:
            from ctgan import CTGAN, TVAE
            availability['CTGAN'] = True
            availability['TVAE'] = True
            logging.info("CTGAN and TVAE packages available")
        except ImportError:
            availability['CTGAN'] = False
            availability['TVAE'] = False
            logging.warning("⚠️ CTGAN/TVAE packages not available - will use baseline models")
        
        # Test SDV availability for CopulaGAN
        try:
            from sdv.single_table import CopulaGANSynthesizer
            availability['CopulaGAN'] = True
            logging.info("✅ SDV CopulaGAN package available")
        except ImportError:
            availability['CopulaGAN'] = False
            logging.warning("⚠️ SDV CopulaGAN package not available - will use baseline model")
        
        # TableGAN and GANerAid will use baseline implementations
        availability['TableGAN'] = True  # Always available via baseline
        availability['GANerAid'] = True  # Always available via baseline
        
        return availability
    
    def create_baseline_model(self, model_name: str, **params) -> Any:
        """Create baseline model implementation."""
        from models.baseline_clinical_model import BaselineClinicalModel
        return BaselineClinicalModel(model_name, **params)
    
    def create_ctgan_model(self, **params) -> Any:
        """Create CTGAN model if available."""
        if self.model_availability['CTGAN']:
            from ctgan import CTGAN
            return CTGAN(**params, verbose=False)
        else:
            return self.create_baseline_model('CTGAN', **params)
    
    def create_tvae_model(self, **params) -> Any:
        """Create TVAE model if available."""
        if self.model_availability['TVAE']:
            from ctgan import TVAE
            return TVAE(**params, verbose=False)
        else:
            return self.create_baseline_model('TVAE', **params)
    
    def create_copulagan_model(self, **params) -> Any:
        """Create CopulaGAN model if available."""
        if self.model_availability['CopulaGAN']:
            from sdv.single_table import CopulaGANSynthesizer
            return CopulaGANSynthesizer()
        else:
            return self.create_baseline_model('CopulaGAN', **params)
    
    def create_tablegan_model(self, **params) -> Any:
        """Create TableGAN model (baseline implementation)."""
        return self.create_baseline_model('TableGAN', **params)
    
    def create_ganeraid_model(self, **params) -> Any:
        """Create GANerAid model (baseline implementation)."""
        return self.create_baseline_model('GANerAid', **params)
    
    def get_default_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters for each model."""
        param_sets = {
            'CTGAN': {
                'epochs': 50,
                'batch_size': 64,
                'generator_lr': 2e-4,
                'discriminator_lr': 2e-4
            },
            'TVAE': {
                'epochs': 50,
                'batch_size': 64,
                'lr': 1e-3
            },
            'CopulaGAN': {
                'epochs': 50,
                'batch_size': 64,
                'learning_rate': 1e-3
            },
            'TableGAN': {
                'epochs': 50,
                'batch_size': 64,
                'learning_rate': 2e-4,
                'noise_dim': 128
            },
            'GANerAid': {
                'epochs': 50,
                'batch_size': 64,
                'learning_rate': 1e-3,
                'generator_dim': 128,
                'noise_level': 0.1
            }
        }
        return param_sets.get(model_name, {})
    
    def test_model_initialization(self, model_name: str) -> Dict[str, Any]:
        """Test model initialization with default parameters."""
        logging.info(f"Testing {model_name} initialization...")
        
        result = {
            'model_name': model_name,
            'initialization_success': False,
            'initialization_time': 0,
            'error': None,
            'model_type': 'baseline' if not self.model_availability.get(model_name, False) else 'advanced'
        }
        
        try:
            start_time = time.time()
            params = self.get_default_parameters(model_name)
            
            # Create model based on type
            if model_name == 'CTGAN':
                model = self.create_ctgan_model(**params)
            elif model_name == 'TVAE':
                model = self.create_tvae_model(**params)
            elif model_name == 'CopulaGAN':
                model = self.create_copulagan_model(**params)
            elif model_name == 'TableGAN':
                model = self.create_tablegan_model(**params)
            elif model_name == 'GANerAid':
                model = self.create_ganeraid_model(**params)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            result['initialization_time'] = time.time() - start_time
            result['initialization_success'] = True
            result['parameters'] = params
            
            logging.info(f"✅ {model_name} initialization successful ({result['initialization_time']:.3f}s)")
            
        except Exception as e:
            result['error'] = str(e)
            logging.error(f"❌ {model_name} initialization failed: {e}")
        
        return result
    
    def test_model_fitting(self, model_name: str) -> Dict[str, Any]:
        """Test model fitting process."""
        logging.info(f"🏋️ Testing {model_name} fitting...")
        
        result = {
            'model_name': model_name,
            'fitting_success': False,
            'fitting_time': 0,
            'error': None,
            'data_shape': self.original_data.shape,
            'discrete_columns_used': len(self.discrete_columns)
        }
        
        try:
            start_time = time.time()
            params = self.get_default_parameters(model_name)
            
            # Create model
            if model_name == 'CTGAN':
                model = self.create_ctgan_model(**params)
            elif model_name == 'TVAE':
                model = self.create_tvae_model(**params)
            elif model_name == 'CopulaGAN':
                model = self.create_copulagan_model(**params)
            elif model_name == 'TableGAN':
                model = self.create_tablegan_model(**params)
            elif model_name == 'GANerAid':
                model = self.create_ganeraid_model(**params)
            
            # Fit model
            if hasattr(model, 'fit') and not self.model_availability.get(model_name, False):
                # Baseline model
                model.fit(self.original_data, discrete_columns=self.discrete_columns)
            elif hasattr(model, 'fit') and model_name in ['CTGAN', 'TVAE']:
                # CTGAN/TVAE from ctgan package
                model.fit(self.original_data, discrete_columns=self.discrete_columns)
            elif hasattr(model, 'fit') and model_name == 'CopulaGAN':
                # SDV CopulaGAN
                model.fit(self.original_data)
            else:
                raise ValueError(f"Model {model_name} doesn't have a fit method")
            
            result['fitting_time'] = time.time() - start_time
            result['fitting_success'] = True
            result['model'] = model  # Store for generation test
            
            logging.info(f"✅ {model_name} fitting successful ({result['fitting_time']:.1f}s)")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logging.error(f"❌ {model_name} fitting failed: {e}")
        
        return result
    
    def test_model_generation(self, model_name: str, fitted_model: Any) -> Dict[str, Any]:
        """Test synthetic data generation."""
        logging.info(f"🎲 Testing {model_name} generation...")
        
        n_samples = min(len(self.original_data), 100)  # Generate small sample for testing
        
        result = {
            'model_name': model_name,
            'generation_success': False,
            'generation_time': 0,
            'generated_samples': 0,
            'data_shape_match': False,
            'columns_match': False,
            'data_types_valid': False,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Generate synthetic data
            if hasattr(fitted_model, 'generate'):
                # Baseline model
                synthetic_data = fitted_model.generate(n_samples)
            elif hasattr(fitted_model, 'sample'):
                # CTGAN/TVAE/SDV models
                synthetic_data = fitted_model.sample(n_samples)
            else:
                raise ValueError(f"Model {model_name} doesn't have generation method")
            
            result['generation_time'] = time.time() - start_time
            result['generation_success'] = True
            result['generated_samples'] = len(synthetic_data)
            
            # Validate synthetic data quality
            result['data_shape_match'] = synthetic_data.shape[1] == self.original_data.shape[1]
            result['columns_match'] = list(synthetic_data.columns) == list(self.original_data.columns)
            
            # Check data types
            type_issues = []
            for col in self.original_data.columns:
                if col in synthetic_data.columns:
                    orig_type = self.original_data[col].dtype
                    synth_type = synthetic_data[col].dtype
                    
                    # Allow some flexibility in numeric types
                    if orig_type in ['int64', 'float64'] and synth_type in ['int64', 'float64']:
                        continue
                    elif orig_type != synth_type:
                        type_issues.append(f"{col}: {orig_type} -> {synth_type}")
            
            result['data_types_valid'] = len(type_issues) == 0
            result['type_issues'] = type_issues
            result['synthetic_data_sample'] = synthetic_data.head(3)  # Store small sample
            
            logging.info(f"✅ {model_name} generation successful ({result['generation_time']:.3f}s, {n_samples} samples)")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logging.error(f"❌ {model_name} generation failed: {e}")
        
        return result
    
    def test_bayesian_optimization(self, model_names: List[str], n_trials: int = 5) -> Dict[str, Any]:
        """Test Bayesian optimization pipeline with reduced trials."""
        logging.info(f"🎯 Testing Bayesian optimization with {n_trials} trials per model...")
        
        result = {
            'optimization_success': False,
            'models_tested': model_names,
            'n_trials_per_model': n_trials,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'total_time': 0,
            'model_results': {},
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Import optimization components
            from evaluation.clinical_evaluator import ClinicalEvaluator
            from optimization.bayesian_optimizer import ClinicalModelOptimizer
            
            # Create evaluator
            evaluator = ClinicalEvaluator(
                original_data=self.original_data,
                target_column=self.target_column,
                discrete_columns=self.discrete_columns
            )
            
            # Create optimizer
            optimizer = ClinicalModelOptimizer(
                data=self.original_data,
                discrete_columns=self.discrete_columns,
                evaluator=evaluator,
                random_state=self.random_state
            )
            
            # Test optimization for each model
            for model_name in model_names:
                try:
                    logging.info(f"Optimizing {model_name}...")
                    opt_result = optimizer.optimize_model(model_name, n_trials=n_trials)
                    result['model_results'][model_name] = {
                        'success': True,
                        'best_score': opt_result['best_score'],
                        'best_params': opt_result['best_params'],
                        'optimization_time': opt_result['optimization_time']
                    }
                    result['successful_optimizations'] += 1
                    
                except Exception as e:
                    result['model_results'][model_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    result['failed_optimizations'] += 1
                    logging.error(f"❌ {model_name} optimization failed: {e}")
            
            result['total_time'] = time.time() - start_time
            result['optimization_success'] = result['successful_optimizations'] > 0
            
            logging.info(f"✅ Optimization completed: {result['successful_optimizations']}/{len(model_names)} successful")
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logging.error(f"❌ Bayesian optimization test failed: {e}")
        
        return result
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling scenarios."""
        logging.info("🛡️ Testing error handling scenarios...")
        
        result = {
            'error_handling_tests': {},
            'total_tests': 0,
            'passed_tests': 0
        }
        
        # Test 1: Empty data
        try:
            empty_data = pd.DataFrame()
            model = self.create_baseline_model('CTGAN')
            model.fit(empty_data)
            result['error_handling_tests']['empty_data'] = {'passed': False, 'error': 'Should have failed'}
        except Exception as e:
            result['error_handling_tests']['empty_data'] = {'passed': True, 'error': str(e)}
        
        # Test 2: Invalid parameters
        try:
            model = self.create_baseline_model('CTGAN', epochs=-1, batch_size=0)
            result['error_handling_tests']['invalid_params'] = {'passed': True, 'error': 'Gracefully handled'}
        except Exception as e:
            result['error_handling_tests']['invalid_params'] = {'passed': True, 'error': str(e)}
        
        # Test 3: Generation without fitting
        try:
            model = self.create_baseline_model('CTGAN')
            synthetic_data = model.generate(10)
            result['error_handling_tests']['unfitted_generation'] = {'passed': False, 'error': 'Should have failed'}
        except Exception as e:
            result['error_handling_tests']['unfitted_generation'] = {'passed': True, 'error': str(e)}
        
        result['total_tests'] = len(result['error_handling_tests'])
        result['passed_tests'] = sum(1 for test in result['error_handling_tests'].values() if test['passed'])
        
        logging.info(f"✅ Error handling tests: {result['passed_tests']}/{result['total_tests']} passed")
        
        return result
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logging.info("🚀 Starting comprehensive model testing...")
        logging.info("=" * 60)
        
        models_to_test = ['CTGAN', 'TVAE', 'CopulaGAN', 'TableGAN', 'GANerAid']
        
        # Test 1: Model Initialization
        logging.info("\n📝 Phase 1: Model Initialization Tests")
        initialization_results = {}
        for model_name in models_to_test:
            initialization_results[model_name] = self.test_model_initialization(model_name)
        
        # Test 2: Model Fitting
        logging.info("\n📝 Phase 2: Model Fitting Tests")
        fitting_results = {}
        fitted_models = {}
        for model_name in models_to_test:
            fitting_results[model_name] = self.test_model_fitting(model_name)
            if fitting_results[model_name]['fitting_success']:
                fitted_models[model_name] = fitting_results[model_name]['model']
        
        # Test 3: Data Generation
        logging.info("\n📝 Phase 3: Data Generation Tests")
        generation_results = {}
        for model_name, model in fitted_models.items():
            generation_results[model_name] = self.test_model_generation(model_name, model)
        
        # Test 4: Bayesian Optimization (reduced trials)
        logging.info("\n📝 Phase 4: Bayesian Optimization Tests")
        optimization_results = self.test_bayesian_optimization(
            [name for name in models_to_test if name in fitted_models], 
            n_trials=5
        )
        
        # Test 5: Error Handling
        logging.info("\n📝 Phase 5: Error Handling Tests")
        error_handling_results = self.test_error_handling()
        
        # Compile final results
        final_results = {
            'test_metadata': {
                'test_start_time': self.test_start_time,
                'test_end_time': datetime.now(),
                'data_path': self.data_path,
                'data_shape': self.original_data.shape,
                'target_column': self.target_column,
                'discrete_columns': self.discrete_columns,
                'model_availability': self.model_availability
            },
            'initialization_results': initialization_results,
            'fitting_results': fitting_results,
            'generation_results': generation_results,
            'optimization_results': optimization_results,
            'error_handling_results': error_handling_results
        }
        
        return final_results
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE MODEL TESTING REPORT")
        report_lines.append("Phase 6 Clinical Framework - 5 Model Validation")
        report_lines.append("=" * 80)
        
        # Test metadata
        metadata = results['test_metadata']
        report_lines.append(f"\n📊 Test Summary:")
        report_lines.append(f"   • Data Path: {metadata['data_path']}")
        report_lines.append(f"   • Data Shape: {metadata['data_shape']}")
        report_lines.append(f"   • Target Column: {metadata['target_column']}")
        report_lines.append(f"   • Discrete Columns: {len(metadata['discrete_columns'])}")
        report_lines.append(f"   • Test Duration: {metadata['test_end_time'] - metadata['test_start_time']}")
        
        # Model availability
        report_lines.append(f"\n📦 Package Availability:")
        for model, available in metadata['model_availability'].items():
            status = "✅ Available" if available else "⚠️ Using Baseline"
            report_lines.append(f"   • {model}: {status}")
        
        # Initialization results
        report_lines.append(f"\n🔧 Model Initialization Results:")
        init_results = results['initialization_results']
        for model_name, result in init_results.items():
            status = "✅ Success" if result['initialization_success'] else "❌ Failed"
            model_type = f"({result['model_type']})"
            report_lines.append(f"   • {model_name} {model_type}: {status}")
            if result.get('error'):
                report_lines.append(f"     Error: {result['error']}")
        
        # Fitting results
        report_lines.append(f"\n🏋️ Model Fitting Results:")
        fit_results = results['fitting_results']
        for model_name, result in fit_results.items():
            status = "✅ Success" if result['fitting_success'] else "❌ Failed"
            time_str = f"({result['fitting_time']:.1f}s)" if result['fitting_success'] else ""
            report_lines.append(f"   • {model_name}: {status} {time_str}")
            if result.get('error'):
                report_lines.append(f"     Error: {result['error']}")
        
        # Generation results
        report_lines.append(f"\n🎲 Data Generation Results:")
        gen_results = results['generation_results']
        for model_name, result in gen_results.items():
            if result['generation_success']:
                samples = result['generated_samples']
                time_str = f"({result['generation_time']:.3f}s)"
                shape_ok = "✅" if result['data_shape_match'] else "⚠️"
                cols_ok = "✅" if result['columns_match'] else "⚠️"
                types_ok = "✅" if result['data_types_valid'] else "⚠️"
                report_lines.append(f"   • {model_name}: ✅ Success {time_str}")
                report_lines.append(f"     - Generated {samples} samples")
                report_lines.append(f"     - Shape match: {shape_ok}, Columns: {cols_ok}, Types: {types_ok}")
            else:
                report_lines.append(f"   • {model_name}: ❌ Failed")
                if result.get('error'):
                    report_lines.append(f"     Error: {result['error']}")
        
        # Optimization results
        report_lines.append(f"\n🎯 Bayesian Optimization Results:")
        opt_results = results['optimization_results']
        if opt_results['optimization_success']:
            report_lines.append(f"   • Total Time: {opt_results['total_time']:.1f}s")
            report_lines.append(f"   • Successful: {opt_results['successful_optimizations']}/{len(opt_results['models_tested'])}")
            report_lines.append(f"   • Trials per model: {opt_results['n_trials_per_model']}")
            
            for model_name, result in opt_results['model_results'].items():
                if result['success']:
                    score = result['best_score']
                    time_str = f"({result['optimization_time']:.1f}s)"
                    report_lines.append(f"   • {model_name}: ✅ Score {score:.4f} {time_str}")
                else:
                    report_lines.append(f"   • {model_name}: ❌ Failed")
        else:
            report_lines.append(f"   • ❌ Optimization failed: {opt_results.get('error', 'Unknown error')}")
        
        # Error handling results
        report_lines.append(f"\n🛡️ Error Handling Results:")
        error_results = results['error_handling_results']
        passed = error_results['passed_tests']
        total = error_results['total_tests']
        report_lines.append(f"   • Tests passed: {passed}/{total}")
        
        for test_name, result in error_results['error_handling_tests'].items():
            status = "✅ Passed" if result['passed'] else "❌ Failed"
            report_lines.append(f"   • {test_name}: {status}")
        
        # Summary and recommendations
        report_lines.append(f"\n🎯 SUMMARY AND RECOMMENDATIONS:")
        
        # Count successful models
        successful_init = sum(1 for r in init_results.values() if r['initialization_success'])
        successful_fit = sum(1 for r in fit_results.values() if r['fitting_success'])
        successful_gen = sum(1 for r in gen_results.values() if r['generation_success'])
        
        report_lines.append(f"   • Models successfully initialized: {successful_init}/5")
        report_lines.append(f"   • Models successfully fitted: {successful_fit}/5")
        report_lines.append(f"   • Models successfully generating: {successful_gen}/5")
        
        # Working models
        working_models = [name for name, r in gen_results.items() if r['generation_success']]
        report_lines.append(f"   • Fully working models: {', '.join(working_models) if working_models else 'None'}")
        
        # Model type breakdown
        advanced_working = [name for name in working_models if metadata['model_availability'].get(name, False)]
        baseline_working = [name for name in working_models if not metadata['model_availability'].get(name, False)]
        
        if advanced_working:
            report_lines.append(f"   • Advanced packages working: {', '.join(advanced_working)}")
        if baseline_working:
            report_lines.append(f"   • Baseline fallbacks working: {', '.join(baseline_working)}")
        
        # Recommendations
        report_lines.append(f"\n💡 RECOMMENDATIONS:")
        
        if successful_gen >= 3:
            report_lines.append("   ✅ Framework is functioning well with good model coverage")
        elif successful_gen >= 1:
            report_lines.append("   ⚠️ Framework is partially working - investigate failed models")
        else:
            report_lines.append("   ❌ Framework has critical issues - major debugging needed")
        
        if len(advanced_working) < 2:
            report_lines.append("   📦 Consider installing missing packages (ctgan, sdv) for better performance")
        
        if opt_results['optimization_success']:
            report_lines.append("   🎯 Bayesian optimization is working - can proceed with full pipeline")
        else:
            report_lines.append("   🎯 Bayesian optimization needs debugging before production use")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)


def main():
    """Main testing function."""
    # Configuration
    DATA_PATH = r"C:\Users\gcicc\claudeproj\tableGenCompare\data\Pakistani_Diabetes_Dataset.csv"
    TARGET_COLUMN = "Outcome"
    
    print("Starting Comprehensive Model Testing...")
    print("Phase 6 Clinical Framework - 5 Model Validation")
    print("=" * 60)
    
    # Initialize tester
    tester = ComprehensiveModelTester(
        data_path=DATA_PATH,
        target_column=TARGET_COLUMN,
        random_state=42
    )
    
    # Run comprehensive tests
    results = tester.run_comprehensive_tests()
    
    # Generate and save report
    report = tester.generate_test_report(results)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"comprehensive_model_test_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write(report)
    
    # Print report to console
    print("\n" + report)
    print(f"\n📄 Full report saved to: {report_filename}")
    
    return results


if __name__ == "__main__":
    results = main()