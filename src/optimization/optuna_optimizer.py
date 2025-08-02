"""
Optuna-based hyperparameter optimization engine for synthetic data models.

This module provides comprehensive hyperparameter optimization capabilities
using Optuna's state-of-the-art optimization algorithms, including multi-objective
optimization for balancing TRTS performance and similarity metrics.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler, NSGAIISampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from datetime import datetime
import logging
import json
from pathlib import Path
import time

try:
    from ..models.base_model import SyntheticDataModel
    from ..evaluation.unified_evaluator import UnifiedEvaluator
except ImportError:
    # Fallback for when running as standalone script
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.base_model import SyntheticDataModel
    from evaluation.unified_evaluator import UnifiedEvaluator

logger = logging.getLogger(__name__)

# Set Optuna logging level to reduce noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer for synthetic data models.
    
    Supports single and multi-objective optimization with various sampling
    strategies and pruning techniques. Based on Optuna 4.4+ with latest
    multi-objective optimization capabilities.
    """
    
    def __init__(
        self,
        model_name: str,
        optimization_objectives: List[str] = ["trts_overall", "similarity"],
        sampler_type: str = "tpe",
        pruner_type: str = "median",
        random_state: int = 42
    ):
        """
        Initialize Optuna optimizer.
        
        Args:
            model_name: Name of the model to optimize ("ganeraid", "ctgan", "tvae")
            optimization_objectives: List of objectives to optimize
            sampler_type: Type of sampler ("tpe", "random", "nsga2", "cmaes")
            pruner_type: Type of pruner ("median", "hyperband", "none")
            random_state: Random seed for reproducible optimization
        """
        self.model_name = model_name.lower()
        self.optimization_objectives = optimization_objectives
        self.sampler_type = sampler_type.lower()
        self.pruner_type = pruner_type.lower()
        self.random_state = random_state
        
        # Initialize components
        self.evaluator = UnifiedEvaluator(random_state=random_state)
        self.study = None
        self.optimization_history = []
        
        # Optimization configuration
        self.is_multi_objective = len(optimization_objectives) > 1
        
        logger.info(f"OptunaOptimizer initialized for {model_name}")
        logger.info(f"Objectives: {optimization_objectives}")
        logger.info(f"Multi-objective: {self.is_multi_objective}")
    
    def optimize(
        self,
        train_data: pd.DataFrame,
        target_column: str,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        custom_objective: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            train_data: Training dataset
            target_column: Name of target column
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            dataset_metadata: Dataset metadata for evaluation
            custom_objective: Custom objective function (overrides default)
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Starting optimization with {n_trials} trials")
        optimization_start = datetime.now()
        
        # Create study
        self.study = self._create_study()
        
        # Set up objective function
        if custom_objective is not None:
            objective_func = custom_objective
        else:
            objective_func = self._create_objective_function(
                train_data, target_column, dataset_metadata
            )
        
        try:
            # Run optimization
            self.study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[self._trial_callback]
            )
            
            optimization_end = datetime.now()
            optimization_duration = (optimization_end - optimization_start).total_seconds()
            
            # Compile results
            results = self._compile_optimization_results(
                optimization_start, optimization_end, optimization_duration
            )
            
            logger.info(f"Optimization completed in {optimization_duration:.2f} seconds")
            logger.info(f"Best trial: {self.study.best_trial.number}")
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get the best hyperparameters found during optimization.
        
        Returns:
            Dictionary of best hyperparameters
        """
        if self.study is None:
            raise ValueError("No optimization study available. Run optimize() first.")
        
        if self.is_multi_objective:
            # For multi-objective, return Pareto-optimal solutions
            pareto_trials = self.study.best_trials
            return {
                'pareto_solutions': [
                    {
                        'trial_number': trial.number,
                        'parameters': trial.params,
                        'values': trial.values
                    }
                    for trial in pareto_trials
                ],
                'recommended_solution': pareto_trials[0].params if pareto_trials else {}
            }
        else:
            # Single objective
            return self.study.best_params
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get complete optimization history.
        
        Returns:
            List of trial results with parameters and objectives
        """
        if self.study is None:
            return self.optimization_history
        
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_info = {
                    'trial_number': trial.number,
                    'parameters': trial.params,
                    'objectives': trial.values if self.is_multi_objective else [trial.value],
                    'duration': trial.duration.total_seconds() if trial.duration else 0,
                    'datetime': trial.datetime_complete.isoformat() if trial.datetime_complete else None
                }
                history.append(trial_info)
        
        return history
    
    def create_visualizations(self, output_dir: str) -> Dict[str, str]:
        """
        Create optimization visualization plots.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        if self.study is None:
            raise ValueError("No optimization study available. Run optimize() first.")
        
        try:
            import optuna.visualization as vis
            import plotly
        except ImportError:
            logger.warning("Optuna visualization dependencies not available. Install with: pip install plotly")
            return {}
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        try:
            # Optimization history
            fig = vis.plot_optimization_history(self.study)
            history_path = output_path / "optimization_history.html"
            fig.write_html(str(history_path))
            plot_files['optimization_history'] = str(history_path)
            
            # Parameter importance
            if len(self.study.trials) > 10:  # Need sufficient trials
                fig = vis.plot_param_importances(self.study)
                importance_path = output_path / "parameter_importance.html"
                fig.write_html(str(importance_path))
                plot_files['parameter_importance'] = str(importance_path)
            
            # Parallel coordinate plot
            fig = vis.plot_parallel_coordinate(self.study)
            parallel_path = output_path / "parallel_coordinate.html"
            fig.write_html(str(parallel_path))
            plot_files['parallel_coordinate'] = str(parallel_path)
            
            # Multi-objective specific plots
            if self.is_multi_objective:
                # Pareto front
                fig = vis.plot_pareto_front(
                    self.study, 
                    target_names=self.optimization_objectives
                )
                pareto_path = output_path / "pareto_front.html"
                fig.write_html(str(pareto_path))
                plot_files['pareto_front'] = str(pareto_path)
            
            logger.info(f"Visualization plots saved to {output_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to create some visualizations: {e}")
        
        return plot_files
    
    def save_study(self, filepath: str) -> None:
        """
        Save the optimization study to disk.
        
        Args:
            filepath: Path to save the study
        """
        if self.study is None:
            raise ValueError("No optimization study to save.")
        
        try:
            # Save study as pickle (Optuna native format)
            import pickle
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(self.study, f)
            
            # Save study summary as JSON
            study_summary = {
                'model_name': self.model_name,
                'optimization_objectives': self.optimization_objectives,
                'sampler_type': self.sampler_type,
                'pruner_type': self.pruner_type,
                'n_trials': len(self.study.trials),
                'best_parameters': self.get_best_parameters(),
                'optimization_history': self.get_optimization_history()
            }
            
            with open(f"{filepath}_summary.json", 'w') as f:
                json.dump(study_summary, f, indent=2, default=str)
            
            logger.info(f"Study saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save study: {e}")
            raise
    
    def load_study(self, filepath: str) -> None:
        """
        Load a saved optimization study.
        
        Args:
            filepath: Path to the saved study
        """
        try:
            import pickle
            with open(f"{filepath}.pkl", 'rb') as f:
                self.study = pickle.load(f)
            
            logger.info(f"Study loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load study: {e}")
            raise
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with appropriate sampler and pruner."""
        # Create sampler
        if self.sampler_type == "tpe":
            sampler = TPESampler(seed=self.random_state)
        elif self.sampler_type == "random":
            sampler = RandomSampler(seed=self.random_state)
        elif self.sampler_type == "nsga2":
            sampler = NSGAIISampler(seed=self.random_state)
        elif self.sampler_type == "cmaes":
            sampler = CmaEsSampler(seed=self.random_state)
        else:
            logger.warning(f"Unknown sampler type: {self.sampler_type}, using TPE")
            sampler = TPESampler(seed=self.random_state)
        
        # Create pruner
        if self.pruner_type == "median":
            pruner = MedianPruner()
        elif self.pruner_type == "hyperband":
            pruner = HyperbandPruner()
        elif self.pruner_type == "none":
            pruner = optuna.pruners.NopPruner()
        else:
            logger.warning(f"Unknown pruner type: {self.pruner_type}, using median")
            pruner = MedianPruner()
        
        # Create study
        if self.is_multi_objective:
            # Multi-objective optimization
            directions = []
            for obj in self.optimization_objectives:
                if obj in ["trts_overall", "trts_utility", "trts_quality", "similarity", "data_quality"]:
                    directions.append("maximize")  # Higher is better
                else:
                    directions.append("minimize")  # Assume minimize for unknown objectives
            
            study = optuna.create_study(
                directions=directions,
                sampler=sampler,
                pruner=pruner
            )
        else:
            # Single objective optimization
            direction = "maximize" if self.optimization_objectives[0] in [
                "trts_overall", "trts_utility", "trts_quality", "similarity", "data_quality"
            ] else "minimize"
            
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner
            )
        
        return study
    
    def _create_objective_function(
        self, 
        train_data: pd.DataFrame, 
        target_column: str,
        dataset_metadata: Optional[Dict[str, Any]]
    ) -> Callable:
        """Create objective function for optimization."""
        
        def objective(trial: optuna.Trial) -> Union[float, List[float]]:
            trial_start = time.time()
            
            try:
                # Import model factory
                from ..models.model_factory import ModelFactory
                
                # Create model with suggested hyperparameters
                model = ModelFactory.create(self.model_name, random_state=self.random_state)
                
                # Get hyperparameter space and suggest parameters
                hyperparameter_space = model.get_hyperparameter_space()
                suggested_params = {}
                
                for param_name, param_config in hyperparameter_space.items():
                    if param_config['type'] == 'int':
                        suggested_params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            step=param_config.get('step', 1)
                        )
                    elif param_config['type'] == 'float':
                        suggested_params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'categorical':
                        suggested_params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # Set model configuration
                model.set_config(suggested_params)
                
                # Train model
                logger.debug(f"Trial {trial.number}: Training with params {suggested_params}")
                model.train(train_data, **suggested_params)
                
                # Generate synthetic data
                synthetic_data = model.generate(len(train_data))
                
                # Run evaluation
                if dataset_metadata is None:
                    dataset_metadata = {
                        'dataset_info': {'name': f'optimization_trial_{trial.number}'},
                        'target_info': {'column': target_column, 'type': 'auto'}
                    }
                
                results = self.evaluator.run_complete_evaluation(
                    model=model,
                    original_data=train_data,
                    synthetic_data=synthetic_data,
                    dataset_metadata=dataset_metadata,
                    output_dir=f"temp_optimization_trial_{trial.number}",
                    target_column=target_column
                )
                
                # Extract objective values
                objective_values = []
                for obj_name in self.optimization_objectives:
                    if obj_name == "trts_overall":
                        value = results['trts_results']['overall_score_percent'] / 100.0
                    elif obj_name == "trts_utility":
                        value = results['trts_results']['utility_score_percent'] / 100.0
                    elif obj_name == "trts_quality":
                        value = results['trts_results']['quality_score_percent'] / 100.0
                    elif obj_name == "similarity":
                        value = results['similarity_analysis']['final_similarity']
                    elif obj_name == "data_quality":
                        value = results['data_quality']['data_type_consistency'] / 100.0
                    else:
                        logger.warning(f"Unknown objective: {obj_name}, using 0.5")
                        value = 0.5
                    
                    objective_values.append(value)
                
                trial_duration = time.time() - trial_start
                
                # Log trial results
                logger.debug(f"Trial {trial.number} completed in {trial_duration:.2f}s")
                logger.debug(f"Objectives: {dict(zip(self.optimization_objectives, objective_values))}")
                
                # Store trial info
                trial_info = {
                    'trial_number': trial.number,
                    'parameters': suggested_params,
                    'objectives': dict(zip(self.optimization_objectives, objective_values)),
                    'duration': trial_duration
                }
                self.optimization_history.append(trial_info)
                
                # Return single value for single objective, list for multi-objective
                return objective_values if self.is_multi_objective else objective_values[0]
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                # Return worst possible values
                if self.is_multi_objective:
                    return [0.0] * len(self.optimization_objectives)
                else:
                    return 0.0
        
        return objective
    
    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback function called after each trial."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if self.is_multi_objective:
                logger.info(f"Trial {trial.number} completed with values: {trial.values}")
            else:
                logger.info(f"Trial {trial.number} completed with value: {trial.value}")
    
    def _compile_optimization_results(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        duration: float
    ) -> Dict[str, Any]:
        """Compile comprehensive optimization results."""
        
        results = {
            'optimization_summary': {
                'model_name': self.model_name,
                'objectives': self.optimization_objectives,
                'is_multi_objective': self.is_multi_objective,
                'sampler_type': self.sampler_type,
                'pruner_type': self.pruner_type,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_trials': len(self.study.trials),
                'completed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'failed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
            },
            'best_parameters': self.get_best_parameters(),
            'optimization_history': self.get_optimization_history()
        }
        
        # Add multi-objective specific results
        if self.is_multi_objective:
            pareto_trials = self.study.best_trials
            results['pareto_analysis'] = {
                'n_pareto_solutions': len(pareto_trials),
                'pareto_front_values': [trial.values for trial in pareto_trials]
            }
        
        return results