"""
Bayesian Hyperparameter Optimization Framework

Comprehensive optimization module for clinical synthetic data generation models.
Uses Optuna for Bayesian optimization with clinical-specific objective functions.
"""

import optuna
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import model classes (with fallbacks)
try:
    from ctgan import CTGAN, TVAE
    ctgan_available = True
except ImportError:
    ctgan_available = False

try:
    from sdv.single_table import CopulaGANSynthesizer, GaussianCopulaSynthesizer
    sdv_available = True
except ImportError:
    sdv_available = False

# Local imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.baseline_clinical_model import BaselineClinicalModel


class ClinicalModelOptimizer:
    """Bayesian optimization for clinical synthetic data models."""
    
    def __init__(self, data, discrete_columns, evaluator, random_state=42):
        self.data = data
        self.discrete_columns = discrete_columns
        self.evaluator = evaluator
        self.random_state = random_state
        self.optimization_results = {}
    
    def get_parameter_spaces(self):
        """Define parameter spaces for all supported models."""
        return {
            'CTGAN': {
                'epochs': {'type': 'int', 'low': 50, 'high': 300},
                'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
                'generator_lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
                'discriminator_lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
                'generator_dim': {'type': 'categorical', 'choices': [(128, 128), (256, 256), (512, 512)]},
                'discriminator_dim': {'type': 'categorical', 'choices': [(128, 128), (256, 256), (512, 512)]}
            },
            'TVAE': {
                'epochs': {'type': 'int', 'low': 50, 'high': 300},
                'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
                'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
                'compress_dims': {'type': 'categorical', 'choices': [(64, 32), (128, 64), (256, 128)]},
                'decompress_dims': {'type': 'categorical', 'choices': [(32, 64), (64, 128), (128, 256)]}
            },
            'CopulaGAN': {
                'epochs': {'type': 'int', 'low': 50, 'high': 200},
                'batch_size': {'type': 'categorical', 'choices': [32, 64, 128]},
                'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
                'noise_level': {'type': 'float', 'low': 0.01, 'high': 0.3}
            },
            'TableGAN': {
                'epochs': {'type': 'int', 'low': 50, 'high': 250},
                'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
                'learning_rate': {'type': 'float', 'low': 5e-5, 'high': 5e-3, 'log': True},
                'noise_dim': {'type': 'int', 'low': 32, 'high': 256}
            },
            'GANerAid': {
                'epochs': {'type': 'int', 'low': 50, 'high': 200},
                'batch_size': {'type': 'categorical', 'choices': [32, 64, 128]},
                'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
                'generator_dim': {'type': 'int', 'low': 64, 'high': 512},
                'noise_level': {'type': 'float', 'low': 0.05, 'high': 0.25}
            }
        }
    
    def create_objective_function(self, model_name, param_space):
        """Create objective function for Optuna optimization."""
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'], 
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            try:
                # Create and train model
                training_time = time.time()
                
                if ctgan_available and model_name in ['CTGAN', 'TVAE']:
                    if model_name == 'CTGAN':
                        model = CTGAN(**params, verbose=False)
                    else:  # TVAE
                        model = TVAE(**params, verbose=False)
                    
                    model.fit(self.data, discrete_columns=self.discrete_columns)
                    synthetic_data = model.sample(len(self.data))
                    
                elif sdv_available and model_name == 'CopulaGAN':
                    # Note: SDV CopulaGAN has different API
                    model = CopulaGANSynthesizer()
                    model.fit(self.data)
                    synthetic_data = model.sample(len(self.data))
                    
                else:
                    # Use baseline model
                    model = BaselineClinicalModel(model_name, **params)
                    model.fit(self.data, discrete_columns=self.discrete_columns)
                    synthetic_data = model.generate(len(self.data))
                
                training_time = time.time() - training_time
                
                # Evaluate model
                similarity_metrics = self.evaluator.evaluate_similarity(synthetic_data)
                classification_metrics = self.evaluator.evaluate_classification(synthetic_data)
                clinical_utility = self.evaluator.evaluate_clinical_utility(synthetic_data)
                
                # Clinical-focused composite objective
                objective_score = self.calculate_clinical_objective(
                    similarity_metrics, classification_metrics, clinical_utility
                )
                
                # Store additional metrics for analysis
                trial.set_user_attr('similarity_overall', similarity_metrics['overall'])
                trial.set_user_attr('classification_ratio', classification_metrics.get('accuracy_ratio', 0))
                trial.set_user_attr('clinical_utility', clinical_utility['overall_utility'])
                trial.set_user_attr('training_time', training_time)
                
                return objective_score
                
            except Exception as e:
                print(f"Trial failed for {model_name}: {e}")
                return 0.0
        
        return objective
    
    def calculate_clinical_objective(self, similarity, classification, utility):
        """Calculate clinical-focused composite objective score."""
        # Weighted combination emphasizing clinical utility
        weights = {
            'similarity': 0.3,
            'classification': 0.3,
            'utility': 0.4
        }
        
        similarity_score = similarity['overall']
        classification_score = classification.get('accuracy_ratio', 0)
        utility_score = utility['overall_utility']
        
        # Penalize very low scores in any category
        min_threshold = 0.1
        if any(score < min_threshold for score in [similarity_score, classification_score, utility_score]):
            penalty = 0.5
        else:
            penalty = 1.0
        
        composite_score = (
            weights['similarity'] * similarity_score +
            weights['classification'] * classification_score +
            weights['utility'] * utility_score
        ) * penalty
        
        return composite_score
    
    def optimize_model(self, model_name, n_trials=100, timeout=None):
        """Optimize hyperparameters for a specific model."""
        parameter_spaces = self.get_parameter_spaces()
        
        if model_name not in parameter_spaces:
            raise ValueError(f"Model {model_name} not supported. Available: {list(parameter_spaces.keys())}")
        
        param_space = parameter_spaces[model_name]
        objective_func = self.create_objective_function(model_name, param_space)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"{model_name}_clinical_optimization",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        )
        
        # Run optimization
        print(f"Optimizing {model_name} ({n_trials} trials)...")
        start_time = time.time()
        
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        optimization_time = time.time() - start_time
        
        # Store results
        result = {
            'study': study,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time': optimization_time,
            'best_trial_attrs': study.best_trial.user_attrs if study.best_trial else {}
        }
        
        self.optimization_results[model_name] = result
        
        print(f"{model_name} optimization completed:")
        print(f"   • Best score: {study.best_value:.4f}")
        print(f"   • Time: {optimization_time:.1f}s")
        print(f"   • Trials completed: {len(study.trials)}")
        
        return result
    
    def optimize_all_models(self, models=None, n_trials=100, timeout_per_model=None):
        """Optimize all supported models."""
        if models is None:
            models = ['CTGAN', 'TVAE', 'CopulaGAN', 'TableGAN', 'GANerAid']
        
        print(f"Starting comprehensive optimization for {len(models)} models...")
        print(f"Estimated time: ~{len(models) * n_trials * 0.5 / 60:.1f} minutes")
        print("=" * 60)
        
        total_start_time = time.time()
        successful_optimizations = 0
        
        for model_name in models:
            try:
                self.optimize_model(model_name, n_trials, timeout_per_model)
                successful_optimizations += 1
            except Exception as e:
                print(f"{model_name} optimization failed: {e}")
                self.optimization_results[model_name] = {
                    'best_score': 0.0,
                    'best_params': {},
                    'error': str(e),
                    'n_trials': 0,
                    'optimization_time': 0
                }
        
        total_time = time.time() - total_start_time
        
        print(f"\nOptimization completed in {total_time/60:.1f} minutes")
        print(f"Successful optimizations: {successful_optimizations}/{len(models)}")
        
        return self.optimization_results
    
    def get_optimization_summary(self):
        """Generate optimization summary DataFrame."""
        summary_data = []
        
        for model_name, result in self.optimization_results.items():
            summary_data.append({
                'Model': model_name,
                'Best_Score': result.get('best_score', 0),
                'N_Trials': result.get('n_trials', 0),
                'Optimization_Time_min': result.get('optimization_time', 0) / 60,
                'Success': 'error' not in result,
                'Similarity_Score': result.get('best_trial_attrs', {}).get('similarity_overall', 0),
                'Classification_Score': result.get('best_trial_attrs', {}).get('classification_ratio', 0),
                'Clinical_Utility': result.get('best_trial_attrs', {}).get('clinical_utility', 0)
            })
        
        return pd.DataFrame(summary_data).sort_values('Best_Score', ascending=False)
    
    def generate_best_models(self):
        """Generate synthetic data using best parameters for each model."""
        best_synthetic_data = {}
        
        for model_name, result in self.optimization_results.items():
            if 'error' in result:
                continue
            
            try:
                best_params = result['best_params']
                
                # Create model with best parameters
                if ctgan_available and model_name in ['CTGAN', 'TVAE']:
                    if model_name == 'CTGAN':
                        model = CTGAN(**best_params, verbose=False)
                    else:
                        model = TVAE(**best_params, verbose=False)
                    model.fit(self.data, discrete_columns=self.discrete_columns)
                    synthetic_data = model.sample(len(self.data))
                    
                elif sdv_available and model_name == 'CopulaGAN':
                    model = CopulaGANSynthesizer()
                    model.fit(self.data)
                    synthetic_data = model.sample(len(self.data))
                    
                else:
                    model = BaselineClinicalModel(model_name, **best_params)
                    model.fit(self.data, discrete_columns=self.discrete_columns)
                    synthetic_data = model.generate(len(self.data))
                
                best_synthetic_data[model_name] = synthetic_data
                
            except Exception as e:
                print(f"Error generating synthetic data for {model_name}: {e}")
        
        return best_synthetic_data
    
    def plot_optimization_history(self, model_name, save_path=None):
        """Plot optimization history for a specific model."""
        if model_name not in self.optimization_results:
            print(f"No optimization results found for {model_name}")
            return
        
        study = self.optimization_results[model_name].get('study')
        if not study:
            print(f"No study object found for {model_name}")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Optimization history
        trials = study.trials
        values = [trial.value for trial in trials if trial.value is not None]
        
        ax1.plot(values)
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Objective Value')
        ax1.set_title(f'{model_name} - Optimization History')
        ax1.grid(True, alpha=0.3)
        
        # Best value over time
        best_values = []
        current_best = float('-inf')
        for value in values:
            if value > current_best:
                current_best = value
            best_values.append(current_best)
        
        ax2.plot(best_values, color='red')
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Best Objective Value')
        ax2.set_title(f'{model_name} - Best Value Progress')
        ax2.grid(True, alpha=0.3)
        
        # Parameter importance (if available)
        try:
            param_importance = optuna.importance.get_param_importances(study)
            if param_importance:
                params, importance = zip(*param_importance.items())
                ax3.barh(params, importance)
                ax3.set_xlabel('Importance')
                ax3.set_title(f'{model_name} - Parameter Importance')
        except:
            ax3.text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'{model_name} - Parameter Importance')
        
        # Objective components (if available)
        if hasattr(study.best_trial, 'user_attrs'):
            attrs = study.best_trial.user_attrs
            components = ['similarity_overall', 'classification_ratio', 'clinical_utility']
            values = [attrs.get(comp, 0) for comp in components]
            labels = ['Similarity', 'Classification', 'Clinical Utility']
            
            ax4.bar(labels, values, color=['blue', 'orange', 'green'], alpha=0.7)
            ax4.set_ylabel('Score')
            ax4.set_title(f'{model_name} - Best Trial Components')
            ax4.set_ylim(0, 1)
            
            # Add value labels
            for i, v in enumerate(values):
                ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle(f'{model_name} Optimization Analysis', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()