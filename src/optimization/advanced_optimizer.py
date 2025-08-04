#!/usr/bin/env python3
"""
Advanced hyperparameter optimization with Optuna integration.
Extends the existing optimization framework with production-ready features.
"""

import os
import sys
import logging
import json
import pickle
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna import Trial
from optuna.samplers import (
    TPESampler, RandomSampler, CmaEsSampler, 
    NSGAIISampler, QMCSampler
)
from optuna.pruners import (
    MedianPruner, SuccessiveHalvingPruner, 
    HyperbandPruner, ThresholdPruner
)
from optuna.storages import RDBStorage
from optuna.integration import SklearnEvaluator
import joblib

# Internal imports
from models.model_factory import ModelFactory
from evaluation.unified_evaluator import UnifiedEvaluator
from optimization.objective_functions import (
    TRTSObjective, SimilarityObjective, DataQualityObjective,
    CompositeObjective, MultiObjectiveFunction
)

logger = logging.getLogger(__name__)

class AdvancedOptunaOptimizer:
    """
    Advanced hyperparameter optimizer with production features:
    - Multi-objective optimization
    - Early stopping and pruning
    - Distributed optimization
    - Study persistence and resumption
    - Advanced sampling strategies
    """
    
    def __init__(
        self,
        model_name: str,
        storage: Optional[str] = None,
        sampler_name: str = "tpe",
        pruner_name: str = "median",
        n_startup_trials: int = 10,
        random_state: int = 42
    ):
        self.model_name = model_name.lower()
        self.random_state = random_state
        self.storage = storage
        
        # Initialize model factory and evaluator
        self.model_factory = ModelFactory()
        self.evaluator = UnifiedEvaluator(random_state=random_state)
        
        # Validate model availability
        available_models = self.model_factory.list_available_models()
        if not available_models.get(self.model_name, False):
            raise ValueError(f"Model '{self.model_name}' is not available")
        
        # Get model hyperparameter space
        self.model = self.model_factory.create(self.model_name, random_state=random_state)
        self.hyperparameter_space = self.model.get_hyperparameter_space()
        
        if not self.hyperparameter_space:
            raise ValueError(f"No hyperparameter space defined for {self.model_name}")
        
        # Initialize sampler and pruner
        self.sampler = self._create_sampler(sampler_name, n_startup_trials)
        self.pruner = self._create_pruner(pruner_name)
        
        # Optimization state
        self.study = None
        self.objective_function = None
        self.best_trials = []
        self.optimization_history = []
        
        logger.info(f"AdvancedOptunaOptimizer initialized for {self.model_name}")
    
    def _create_sampler(self, sampler_name: str, n_startup_trials: int):
        """Create Optuna sampler based on configuration."""
        sampler_name = sampler_name.lower()
        
        samplers = {
            'tpe': lambda: TPESampler(
                n_startup_trials=n_startup_trials,
                seed=self.random_state,
                multivariate=True,
                group=True
            ),
            'random': lambda: RandomSampler(seed=self.random_state),
            'cmaes': lambda: CmaEsSampler(
                seed=self.random_state,
                n_startup_trials=n_startup_trials
            ),
            'nsgaii': lambda: NSGAIISampler(
                seed=self.random_state,
                population_size=50
            ),
            'qmc': lambda: QMCSampler(seed=self.random_state)
        }
        
        if sampler_name not in samplers:
            logger.warning(f"Unknown sampler '{sampler_name}', using TPE")
            sampler_name = 'tpe'
        
        return samplers[sampler_name]()
    
    def _create_pruner(self, pruner_name: str):
        """Create Optuna pruner based on configuration."""
        pruner_name = pruner_name.lower()
        
        pruners = {
            'median': lambda: MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            ),
            'successive_halving': lambda: SuccessiveHalvingPruner(
                min_resource=10,
                reduction_factor=4
            ),
            'hyperband': lambda: HyperbandPruner(
                min_resource=10,
                reduction_factor=3
            ),
            'threshold': lambda: ThresholdPruner(lower=0.1),
            'none': lambda: optuna.pruners.NopPruner()
        }
        
        if pruner_name not in pruners:
            logger.warning(f"Unknown pruner '{pruner_name}', using median")
            pruner_name = 'median'
        
        return pruners[pruner_name]()
    
    def create_study(
        self,
        study_name: str,
        directions: List[str] = None,
        load_if_exists: bool = True
    ) -> optuna.Study:
        """Create or load an Optuna study."""
        
        # Default to single-objective maximization
        if directions is None:
            directions = ["maximize"]
        
        # Create storage if specified
        storage = None
        if self.storage:
            if self.storage.startswith('sqlite'):
                storage = optuna.storages.RDBStorage(self.storage)
            else:
                storage = self.storage
        
        try:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                directions=directions,
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=load_if_exists
            )
            
            logger.info(f"Created/loaded study '{study_name}' with {len(self.study.trials)} existing trials")
            return self.study
            
        except Exception as e:
            logger.error(f"Error creating study: {e}")
            # Fallback to in-memory study
            self.study = optuna.create_study(
                directions=directions,
                sampler=self.sampler,
                pruner=self.pruner
            )
            return self.study
    
    def define_hyperparameter_space(self, trial: Trial) -> Dict[str, Any]:
        """Define the hyperparameter search space for the trial."""
        params = {}
        
        for param_name, param_config in self.hyperparameter_space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'float':
                low = param_config.get('low', 0.0001)
                high = param_config.get('high', 0.1)
                log = param_config.get('log', False)
                
                params[param_name] = trial.suggest_float(
                    param_name, low, high, log=log
                )
                
            elif param_type == 'int':
                low = param_config.get('low', 1)
                high = param_config.get('high', 1000)
                log = param_config.get('log', False)
                
                params[param_name] = trial.suggest_int(
                    param_name, low, high, log=log
                )
                
            elif param_type == 'categorical':
                choices = param_config.get('choices', [])
                if choices:
                    params[param_name] = trial.suggest_categorical(
                        param_name, choices
                    )
                    
            elif param_type == 'uniform':
                low = param_config.get('low', 0.0)
                high = param_config.get('high', 1.0)
                params[param_name] = trial.suggest_uniform(param_name, low, high)
                
            elif param_type == 'loguniform':
                low = param_config.get('low', 1e-5)
                high = param_config.get('high', 1e-1)
                params[param_name] = trial.suggest_loguniform(param_name, low, high)
        
        return params
    
    def create_objective_function(
        self,
        training_data: pd.DataFrame,
        target_column: Optional[str] = None,
        objective_type: str = "composite",
        custom_weights: Optional[Dict[str, float]] = None,
        validation_split: float = 0.2
    ) -> Callable:
        """Create the objective function for optimization."""
        
        # Split data for training and validation
        if validation_split > 0:
            split_idx = int(len(training_data) * (1 - validation_split))
            train_data = training_data.iloc[:split_idx].copy()
            val_data = training_data.iloc[split_idx:].copy()
        else:
            train_data = training_data.copy()
            val_data = None
        
        def objective(trial: Trial) -> Union[float, List[float]]:
            try:
                # Get hyperparameters for this trial
                params = self.define_hyperparameter_space(trial)
                
                # Create and configure model
                model = self.model_factory.create(self.model_name, random_state=self.random_state)
                model.set_config(params)
                
                # Train model with early stopping support
                start_time = time.time()
                
                # Add pruning callback if supported
                training_params = params.copy()
                if hasattr(model, 'supports_early_stopping') and model.supports_early_stopping():
                    training_params['pruning_callback'] = lambda epoch, loss: trial.report(loss, epoch)
                
                training_result = model.train(train_data, **training_params)
                training_time = time.time() - start_time
                
                # Generate synthetic data
                generation_start = time.time()
                synthetic_data = model.generate(len(val_data) if val_data is not None else len(train_data))
                generation_time = time.time() - generation_start
                
                # Evaluate quality
                eval_data = val_data if val_data is not None else train_data
                
                dataset_metadata = {
                    'dataset_info': {'name': f'optimization_trial_{trial.number}'},
                    'target_info': {'column': target_column, 'type': 'auto'} if target_column else None
                }
                
                evaluation_results = self.evaluator.run_complete_evaluation(
                    model=model,
                    original_data=eval_data,
                    synthetic_data=synthetic_data,
                    dataset_metadata=dataset_metadata,
                    output_dir=None,  # Don't save outputs during optimization
                    target_column=target_column
                )
                
                # Calculate objective value(s)
                if objective_type == "trts":
                    objective_value = evaluation_results['trts_results']['overall_score_percent'] / 100.0
                    
                elif objective_type == "similarity":
                    objective_value = evaluation_results['similarity_analysis']['final_similarity']
                    
                elif objective_type == "quality":
                    objective_value = evaluation_results['data_quality']['data_type_consistency'] / 100.0
                    
                elif objective_type == "composite":
                    weights = custom_weights or {'trts': 0.4, 'similarity': 0.4, 'quality': 0.2}
                    
                    trts_score = evaluation_results['trts_results']['overall_score_percent'] / 100.0
                    similarity_score = evaluation_results['similarity_analysis']['final_similarity']
                    quality_score = evaluation_results['data_quality']['data_type_consistency'] / 100.0
                    
                    objective_value = (
                        weights.get('trts', 0) * trts_score +
                        weights.get('similarity', 0) * similarity_score +
                        weights.get('quality', 0) * quality_score
                    )
                    
                elif objective_type == "multi":
                    # Multi-objective: return list of objectives
                    trts_score = evaluation_results['trts_results']['overall_score_percent'] / 100.0
                    similarity_score = evaluation_results['similarity_analysis']['final_similarity']
                    quality_score = evaluation_results['data_quality']['data_type_consistency'] / 100.0
                    
                    objective_value = [trts_score, similarity_score, quality_score]
                
                else:
                    raise ValueError(f"Unknown objective type: {objective_type}")
                
                # Store additional metrics in trial user attributes
                trial.set_user_attr('trts_score', evaluation_results['trts_results']['overall_score_percent'])
                trial.set_user_attr('similarity_score', evaluation_results['similarity_analysis']['final_similarity'])
                trial.set_user_attr('quality_score', evaluation_results['data_quality']['data_type_consistency'])
                trial.set_user_attr('training_time', training_time)
                trial.set_user_attr('generation_time', generation_time)
                
                if 'final_loss' in training_result:
                    trial.set_user_attr('final_training_loss', training_result['final_loss'])
                
                return objective_value
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                # Return worst possible score for failed trials
                if objective_type == "multi":
                    return [0.0, 0.0, 0.0]
                else:
                    return 0.0
        
        return objective
    
    def optimize(
        self,
        training_data: pd.DataFrame,
        study_name: str,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        target_column: Optional[str] = None,
        objective_type: str = "composite",
        custom_weights: Optional[Dict[str, float]] = None,
        validation_split: float = 0.2,
        callbacks: Optional[List[Callable]] = None,
        show_progress: bool = True
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Args:
            training_data: Training dataset
            study_name: Name for the optimization study
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            target_column: Target column for ML evaluation
            objective_type: Type of objective ('trts', 'similarity', 'quality', 'composite', 'multi')
            custom_weights: Custom weights for composite objective
            validation_split: Fraction of data to use for validation
            callbacks: List of callback functions
            show_progress: Whether to show progress bar
            
        Returns:
            Completed Optuna study
        """
        
        # Determine study directions
        if objective_type == "multi":
            directions = ["maximize", "maximize", "maximize"]  # TRTS, Similarity, Quality
        else:
            directions = ["maximize"]
        
        # Create study
        self.create_study(study_name, directions)
        
        # Create objective function
        objective_fn = self.create_objective_function(
            training_data=training_data,
            target_column=target_column,
            objective_type=objective_type,
            custom_weights=custom_weights,
            validation_split=validation_split
        )
        
        # Setup callbacks
        all_callbacks = callbacks or []
        
        if show_progress:
            # Add progress callback
            def progress_callback(study: optuna.Study, trial: optuna.Trial):
                completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                logger.info(f"Trial {trial.number}: {completed}/{n_trials} completed")
                
                if objective_type != "multi" and trial.value is not None:
                    logger.info(f"  Objective value: {trial.value:.4f}")
                
                # Log best values so far
                if objective_type != "multi":
                    best_value = study.best_value
                    logger.info(f"  Best value so far: {best_value:.4f}")
            
            all_callbacks.append(progress_callback)
        
        # Run optimization
        logger.info(f"Starting optimization: {n_trials} trials, objective={objective_type}")
        start_time = time.time()
        
        try:
            self.study.optimize(
                objective_fn,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=all_callbacks,
                show_progress_bar=show_progress
            )
            
            optimization_time = time.time() - start_time
            logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
            
            # Analyze results
            self._analyze_results(objective_type)
            
            return self.study
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _analyze_results(self, objective_type: str):
        """Analyze optimization results and log key findings."""
        
        if not self.study or not self.study.trials:
            logger.warning("No trials to analyze")
            return
        
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            logger.warning("No completed trials to analyze")
            return
        
        logger.info(f"Optimization Analysis:")
        logger.info(f"  Total trials: {len(self.study.trials)}")
        logger.info(f"  Completed trials: {len(completed_trials)}")
        logger.info(f"  Failed trials: {len(self.study.trials) - len(completed_trials)}")
        
        if objective_type == "multi":
            # Multi-objective analysis
            pareto_trials = self.study.best_trials
            logger.info(f"  Pareto optimal solutions: {len(pareto_trials)}")
            
            for i, trial in enumerate(pareto_trials[:5]):  # Show top 5
                logger.info(f"    Solution {i+1}: TRTS={trial.values[0]:.3f}, "
                          f"Similarity={trial.values[1]:.3f}, Quality={trial.values[2]:.3f}")
        else:
            # Single-objective analysis
            best_trial = self.study.best_trial
            logger.info(f"  Best objective value: {best_trial.value:.4f}")
            logger.info(f"  Best parameters:")
            
            for param, value in best_trial.params.items():
                logger.info(f"    {param}: {value}")
            
            # Log additional metrics from best trial
            if hasattr(best_trial, 'user_attrs'):
                for attr, value in best_trial.user_attrs.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"    {attr}: {value:.4f}")
        
        # Parameter importance analysis
        try:
            if len(completed_trials) >= 10:  # Need sufficient trials
                importances = optuna.importance.get_param_importances(self.study)
                logger.info("  Parameter importance:")
                for param, importance in importances.items():
                    logger.info(f"    {param}: {importance:.3f}")
        except Exception as e:
            logger.debug(f"Could not calculate parameter importance: {e}")
    
    def get_best_hyperparameters(self, n_best: int = 1) -> List[Dict[str, Any]]:
        """Get the best hyperparameters from optimization."""
        
        if not self.study:
            raise ValueError("No optimization study available")
        
        if self.study.directions and len(self.study.directions) > 1:
            # Multi-objective: return Pareto optimal solutions
            pareto_trials = self.study.best_trials[:n_best]
            return [trial.params for trial in pareto_trials]
        else:
            # Single-objective: return top N trials
            completed_trials = [
                t for t in self.study.trials 
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            
            if not completed_trials:
                return []
            
            # Sort by objective value (descending for maximization)
            sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
            return [trial.params for trial in sorted_trials[:n_best]]
    
    def save_study(self, filepath: str):
        """Save the optimization study to file."""
        
        if not self.study:
            raise ValueError("No study to save")
        
        study_data = {
            'study_name': self.study.study_name,
            'directions': [d.name for d in self.study.directions],
            'trials': [],
            'best_params': self.get_best_hyperparameters(1)[0] if self.get_best_hyperparameters(1) else None,
            'metadata': {
                'model_name': self.model_name,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'n_trials': len(self.study.trials)
            }
        }
        
        # Save trial data
        for trial in self.study.trials:
            trial_data = {
                'number': trial.number,
                'state': trial.state.name,
                'params': trial.params,
                'value': trial.value if hasattr(trial, 'value') else None,
                'values': trial.values if hasattr(trial, 'values') else None,
                'user_attrs': dict(trial.user_attrs) if hasattr(trial, 'user_attrs') else {}
            }
            study_data['trials'].append(trial_data)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(study_data, f, indent=2, default=str)
        
        logger.info(f"Study saved to {filepath}")
    
    def load_study(self, filepath: str) -> Dict[str, Any]:
        """Load optimization study from file."""
        
        with open(filepath, 'r') as f:
            study_data = json.load(f)
        
        logger.info(f"Loaded study with {len(study_data['trials'])} trials")
        return study_data
    
    def create_optimization_report(self, output_dir: str = "optimization_results") -> str:
        """Create comprehensive optimization report."""
        
        if not self.study:
            raise ValueError("No study available for reporting")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(output_dir) / f"optimization_report_{self.model_name}_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Hyperparameter Optimization Report\n\n")
            f.write(f"**Model:** {self.model_name}\n")
            f.write(f"**Study Name:** {self.study.study_name}\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            
            # Study summary
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            f.write(f"## Summary\n\n")
            f.write(f"- Total trials: {len(self.study.trials)}\n")
            f.write(f"- Completed trials: {len(completed_trials)}\n")
            f.write(f"- Success rate: {len(completed_trials)/len(self.study.trials)*100:.1f}%\n\n")
            
            # Best results
            if len(self.study.directions) > 1:
                f.write(f"## Pareto Optimal Solutions\n\n")
                for i, trial in enumerate(self.study.best_trials[:10]):
                    f.write(f"### Solution {i+1}\n")
                    f.write(f"- TRTS: {trial.values[0]:.3f}\n")
                    f.write(f"- Similarity: {trial.values[1]:.3f}\n")
                    f.write(f"- Quality: {trial.values[2]:.3f}\n")
                    f.write(f"- Parameters: {trial.params}\n\n")
            else:
                best_trial = self.study.best_trial
                f.write(f"## Best Configuration\n\n")
                f.write(f"**Objective Value:** {best_trial.value:.4f}\n\n")
                f.write(f"**Parameters:**\n")
                for param, value in best_trial.params.items():
                    f.write(f"- {param}: {value}\n")
                f.write(f"\n")
            
            # Parameter importance
            try:
                if len(completed_trials) >= 10:
                    importances = optuna.importance.get_param_importances(self.study)
                    f.write(f"## Parameter Importance\n\n")
                    for param, importance in importances.items():
                        f.write(f"- {param}: {importance:.3f}\n")
                    f.write(f"\n")
            except:
                pass
            
            # Trial history
            f.write(f"## Trial History\n\n")
            f.write(f"| Trial | State | Objective | Parameters |\n")
            f.write(f"|-------|-------|-----------|------------|\n")
            
            for trial in self.study.trials[-20:]:  # Last 20 trials
                state = trial.state.name
                value = f"{trial.value:.4f}" if hasattr(trial, 'value') and trial.value else "N/A"
                params = str(trial.params)[:50] + "..." if len(str(trial.params)) > 50 else str(trial.params)
                f.write(f"| {trial.number} | {state} | {value} | {params} |\n")
        
        logger.info(f"Optimization report saved to {report_file}")
        return str(report_file)

def optimize_model_hyperparameters(
    model_name: str,
    training_data: pd.DataFrame,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    target_column: Optional[str] = None,
    objective_type: str = "composite",
    study_name: Optional[str] = None,
    output_dir: str = "optimization_results"
) -> Dict[str, Any]:
    """
    Convenience function for hyperparameter optimization.
    
    Args:
        model_name: Name of the model to optimize
        training_data: Training dataset
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds
        target_column: Target column for ML evaluation
        objective_type: Optimization objective type
        study_name: Name for the optimization study
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing optimization results
    """
    
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{model_name}_optimization_{timestamp}"
    
    # Create optimizer
    optimizer = AdvancedOptunaOptimizer(
        model_name=model_name,
        sampler_name="tpe",
        pruner_name="median"
    )
    
    # Run optimization
    study = optimizer.optimize(
        training_data=training_data,
        study_name=study_name,
        n_trials=n_trials,
        timeout=timeout,
        target_column=target_column,
        objective_type=objective_type,
        show_progress=True
    )
    
    # Get results
    best_params = optimizer.get_best_hyperparameters(1)
    
    # Save study and create report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    study_file = Path(output_dir) / f"{study_name}.json"
    optimizer.save_study(str(study_file))
    
    report_file = optimizer.create_optimization_report(output_dir)
    
    return {
        'study': study,
        'best_parameters': best_params[0] if best_params else {},
        'study_file': str(study_file),
        'report_file': report_file,
        'n_trials': len(study.trials),
        'best_objective_value': study.best_value if hasattr(study, 'best_value') else None
    }