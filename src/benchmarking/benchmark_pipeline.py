"""
Multi-dataset benchmarking pipeline for synthetic data generation models.

This module provides a comprehensive pipeline for running systematic benchmarks
across multiple datasets and models, with configurable evaluation criteria.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, field
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .dataset_manager import DatasetManager, DatasetConfig

try:
    from ..models.model_factory import ModelFactory
    from ..evaluation.unified_evaluator import UnifiedEvaluator
    from ..optimization.objective_functions import TRTSObjective, SimilarityObjective, CompositeObjective
except ImportError:
    # Fallback for when running as standalone script
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.model_factory import ModelFactory
    from evaluation.unified_evaluator import UnifiedEvaluator
    from optimization.objective_functions import TRTSObjective, SimilarityObjective, CompositeObjective

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    # Models to benchmark
    models_to_test: List[str] = field(default_factory=lambda: ['ganeraid', 'ctgan', 'tvae'])
    
    # Evaluation configuration
    evaluation_metrics: List[str] = field(default_factory=lambda: ['trts', 'similarity', 'quality'])
    
    # Generation configuration
    synthetic_samples_ratio: float = 1.0  # Ratio of synthetic to original samples
    min_synthetic_samples: int = 100
    max_synthetic_samples: int = 10000
    
    # Training configuration
    training_epochs: Dict[str, int] = field(default_factory=lambda: {
        'ganeraid': 100,
        'ctgan': 200,
        'tvae': 200
    })
    
    # Optimization configuration
    enable_optimization: bool = False
    optimization_trials: int = 10
    optimization_timeout_minutes: int = 30
    
    # Execution configuration
    parallel_execution: bool = True
    max_workers: Optional[int] = None
    timeout_per_model_minutes: int = 60
    
    # Output configuration
    output_directory: str = "benchmark_results"
    save_intermediate_results: bool = True
    save_synthetic_data: bool = False
    
    # Random state for reproducibility
    random_state: int = 42
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.max_workers is None:
            self.max_workers = min(4, multiprocessing.cpu_count())
        
        # Validate model names
        available_models = ModelFactory.list_available_models()
        invalid_models = [m for m in self.models_to_test if m not in available_models or not available_models[m]]
        if invalid_models:
            logger.warning(f"Invalid/unavailable models will be skipped: {invalid_models}")
            self.models_to_test = [m for m in self.models_to_test if m not in invalid_models]


class BenchmarkResult:
    """Container for benchmark results from a single model-dataset combination."""
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        dataset_config: DatasetConfig,
        success: bool = False
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.success = success
        
        # Timing information
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.duration_seconds: Optional[float] = None
        
        # Model training information
        self.training_duration: Optional[float] = None
        self.training_epochs: Optional[int] = None
        self.model_config: Optional[Dict[str, Any]] = None
        
        # Generation information
        self.generation_duration: Optional[float] = None
        self.samples_generated: Optional[int] = None
        self.synthetic_data_shape: Optional[Tuple[int, int]] = None
        
        # Evaluation results
        self.evaluation_results: Optional[Dict[str, Any]] = None
        self.trts_score: Optional[float] = None
        self.similarity_score: Optional[float] = None
        self.quality_score: Optional[float] = None
        
        # Objective scores
        self.objective_scores: Dict[str, float] = {}
        
        # Error information
        self.error_message: Optional[str] = None
        self.error_type: Optional[str] = None
    
    def set_timing(self, start_time: datetime, end_time: datetime):
        """Set timing information."""
        self.start_time = start_time
        self.end_time = end_time
        self.duration_seconds = (end_time - start_time).total_seconds()
    
    def set_training_info(self, duration: float, epochs: int, config: Dict[str, Any]):
        """Set model training information."""
        self.training_duration = duration
        self.training_epochs = epochs
        self.model_config = config
    
    def set_generation_info(self, duration: float, samples: int, shape: Tuple[int, int]):
        """Set data generation information."""
        self.generation_duration = duration
        self.samples_generated = samples
        self.synthetic_data_shape = shape
    
    def set_evaluation_results(self, results: Dict[str, Any]):
        """Set evaluation results and extract key metrics."""
        self.evaluation_results = results
        
        # Extract key metrics
        if 'trts_results' in results:
            self.trts_score = results['trts_results'].get('overall_score_percent', 0) / 100
        
        if 'similarity_analysis' in results:
            self.similarity_score = results['similarity_analysis'].get('final_similarity', 0)
        
        if 'data_quality' in results:
            self.quality_score = results['data_quality'].get('data_type_consistency', 0) / 100
    
    def set_error(self, error: Exception):
        """Set error information."""
        self.success = False
        self.error_message = str(error)
        self.error_type = type(error).__name__
    
    def calculate_objective_scores(self):
        """Calculate various objective scores."""
        if not self.evaluation_results:
            return
        
        try:
            # TRTS objective
            trts_obj = TRTSObjective(metric='overall')
            self.objective_scores['trts_overall'] = trts_obj.evaluate(self.evaluation_results)
            
            # Similarity objective
            sim_obj = SimilarityObjective(metric='final')
            self.objective_scores['similarity'] = sim_obj.evaluate(self.evaluation_results)
            
            # Composite objective (balanced)
            composite_obj = CompositeObjective([trts_obj, sim_obj], weights=[0.6, 0.4])
            self.objective_scores['composite'] = composite_obj.evaluate(self.evaluation_results)
            
        except Exception as e:
            logger.warning(f"Failed to calculate objective scores: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'success': self.success,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'training_duration': self.training_duration,
            'training_epochs': self.training_epochs,
            'generation_duration': self.generation_duration,
            'samples_generated': self.samples_generated,
            'synthetic_data_shape': self.synthetic_data_shape,
            'trts_score': self.trts_score,
            'similarity_score': self.similarity_score,
            'quality_score': self.quality_score,
            'objective_scores': self.objective_scores,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'dataset_summary': self.dataset_config.get_summary()
        }


class BenchmarkPipeline:
    """
    Comprehensive benchmarking pipeline for synthetic data generation models.
    
    Provides systematic evaluation across multiple datasets and models with
    configurable evaluation criteria and parallel execution support.
    """
    
    def __init__(
        self,
        dataset_manager: DatasetManager,
        config: Optional[BenchmarkConfig] = None
    ):
        """
        Initialize benchmark pipeline.
        
        Args:
            dataset_manager: Dataset manager with loaded datasets
            config: Benchmark configuration
        """
        self.dataset_manager = dataset_manager
        self.config = config or BenchmarkConfig()
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.benchmark_metadata: Dict[str, Any] = {}
        
        # Setup output directory
        self.output_path = Path(self.config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized benchmark pipeline with {len(self.config.models_to_test)} models")
    
    def run_benchmark(
        self,
        dataset_names: Optional[List[str]] = None,
        dataset_group: Optional[str] = None
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark across specified datasets.
        
        Args:
            dataset_names: Specific datasets to benchmark (if None, uses all)
            dataset_group: Dataset group to benchmark
            
        Returns:
            List of benchmark results
        """
        # Determine datasets to benchmark
        if dataset_group:
            datasets_to_test = self.dataset_manager.get_dataset_group(dataset_group)
        elif dataset_names:
            datasets_to_test = [self.dataset_manager.get_dataset(name) for name in dataset_names]
        else:
            datasets_to_test = list(self.dataset_manager.datasets.values())
        
        logger.info(f"Starting benchmark with {len(datasets_to_test)} datasets and {len(self.config.models_to_test)} models")
        
        # Store benchmark metadata
        self.benchmark_metadata = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'datasets': [d.get_summary() for d in datasets_to_test],
            'models': self.config.models_to_test
        }
        
        # Clear previous results
        self.results.clear()
        
        benchmark_start = datetime.now()
        
        try:
            if self.config.parallel_execution and len(datasets_to_test) * len(self.config.models_to_test) > 1:
                self._run_parallel_benchmark(datasets_to_test)
            else:
                self._run_sequential_benchmark(datasets_to_test)
            
            benchmark_end = datetime.now()
            self.benchmark_metadata['end_time'] = benchmark_end.isoformat()
            self.benchmark_metadata['total_duration_seconds'] = (benchmark_end - benchmark_start).total_seconds()
            
            # Generate summary statistics
            self._generate_benchmark_summary()
            
            # Save results
            if self.config.save_intermediate_results:
                self._save_results()
            
            logger.info(f"Benchmark completed. {len([r for r in self.results if r.success])} successful, {len([r for r in self.results if not r.success])} failed")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
    
    def _run_sequential_benchmark(self, datasets: List[DatasetConfig]):
        """Run benchmark sequentially."""
        total_combinations = len(datasets) * len(self.config.models_to_test)
        completed = 0
        
        for dataset in datasets:
            for model_name in self.config.models_to_test:
                logger.info(f"Running benchmark {completed + 1}/{total_combinations}: {model_name} on {dataset.name}")
                
                result = self._run_single_benchmark(model_name, dataset)
                self.results.append(result)
                completed += 1
                
                # Save intermediate results
                if self.config.save_intermediate_results and completed % 5 == 0:
                    self._save_results()
    
    def _run_parallel_benchmark(self, datasets: List[DatasetConfig]):
        """Run benchmark with parallel execution."""
        # Create list of all model-dataset combinations
        benchmark_tasks = []
        for dataset in datasets:
            for model_name in self.config.models_to_test:
                benchmark_tasks.append((model_name, dataset))
        
        logger.info(f"Running {len(benchmark_tasks)} benchmark tasks in parallel with {self.config.max_workers} workers")
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_single_benchmark, model_name, dataset): (model_name, dataset.name)
                for model_name, dataset in benchmark_tasks
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task, timeout=self.config.timeout_per_model_minutes * 60 * len(benchmark_tasks)):
                model_name, dataset_name = future_to_task[future]
                completed += 1
                
                try:
                    result = future.result()
                    self.results.append(result)
                    logger.info(f"Completed benchmark {completed}/{len(benchmark_tasks)}: {model_name} on {dataset_name}")
                except Exception as e:
                    logger.error(f"Benchmark failed for {model_name} on {dataset_name}: {e}")
                    # Create failed result
                    failed_result = BenchmarkResult(model_name, dataset_name, None)
                    failed_result.set_error(e)
                    self.results.append(failed_result)
                
                # Save intermediate results
                if self.config.save_intermediate_results and completed % 10 == 0:
                    self._save_results()
    
    def _run_single_benchmark(self, model_name: str, dataset_config: DatasetConfig) -> BenchmarkResult:
        """
        Run benchmark for a single model-dataset combination.
        
        Args:
            model_name: Name of the model to test
            dataset_config: Dataset configuration
            
        Returns:
            Benchmark result
        """
        result = BenchmarkResult(model_name, dataset_config.name, dataset_config)
        start_time = datetime.now()
        
        try:
            # Create model
            model = ModelFactory.create(model_name, random_state=self.config.random_state)
            
            # Determine training epochs
            epochs = self.config.training_epochs.get(model_name, 100)
            
            # Train model
            logger.debug(f"Training {model_name} on {dataset_config.name} for {epochs} epochs")
            training_result = model.train(dataset_config.data, epochs=epochs, verbose=False)
            
            result.set_training_info(
                training_result.get('training_duration_seconds', 0),
                epochs,
                model.get_model_info()
            )
            
            # Calculate number of synthetic samples to generate
            n_original = len(dataset_config.data)
            n_synthetic = int(n_original * self.config.synthetic_samples_ratio)
            n_synthetic = max(self.config.min_synthetic_samples, 
                            min(n_synthetic, self.config.max_synthetic_samples))
            
            # Generate synthetic data
            logger.debug(f"Generating {n_synthetic} synthetic samples with {model_name}")
            gen_start = time.time()
            synthetic_data = model.generate(n_synthetic)
            gen_duration = time.time() - gen_start
            
            result.set_generation_info(gen_duration, len(synthetic_data), synthetic_data.shape)
            
            # Save synthetic data if requested
            if self.config.save_synthetic_data:
                synthetic_path = self.output_path / f"{model_name}_{dataset_config.name}_synthetic.csv"
                synthetic_data.to_csv(synthetic_path, index=False)
            
            # Run evaluation
            logger.debug(f"Evaluating {model_name} on {dataset_config.name}")
            evaluator = UnifiedEvaluator(random_state=self.config.random_state)
            
            evaluation_results = evaluator.run_complete_evaluation(
                model=model,
                original_data=dataset_config.data,
                synthetic_data=synthetic_data,
                dataset_metadata={
                    'dataset_info': {'name': dataset_config.name},
                    'target_info': {
                        'column': dataset_config.target_column,
                        'type': 'binary' if dataset_config.target_column else None
                    }
                },
                output_dir=str(self.output_path / f"{model_name}_{dataset_config.name}"),
                target_column=dataset_config.target_column
            )
            
            result.set_evaluation_results(evaluation_results)
            result.calculate_objective_scores()
            
            end_time = datetime.now()
            result.set_timing(start_time, end_time)
            result.success = True
            
            logger.debug(f"Successfully completed benchmark: {model_name} on {dataset_config.name}")
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name} on {dataset_config.name}: {e}")
            end_time = datetime.now()
            result.set_timing(start_time, end_time)
            result.set_error(e)
        
        return result
    
    def _generate_benchmark_summary(self):
        """Generate summary statistics for the benchmark."""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        summary = {
            'total_benchmarks': len(self.results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(self.results) if self.results else 0,
        }
        
        if successful_results:
            # Calculate average metrics
            trts_scores = [r.trts_score for r in successful_results if r.trts_score is not None]
            similarity_scores = [r.similarity_score for r in successful_results if r.similarity_score is not None]
            durations = [r.duration_seconds for r in successful_results if r.duration_seconds is not None]
            
            if trts_scores:
                summary['average_trts_score'] = np.mean(trts_scores)
                summary['std_trts_score'] = np.std(trts_scores)
            
            if similarity_scores:
                summary['average_similarity_score'] = np.mean(similarity_scores)
                summary['std_similarity_score'] = np.std(similarity_scores)
            
            if durations:
                summary['average_duration_seconds'] = np.mean(durations)
                summary['total_computation_time_hours'] = sum(durations) / 3600
            
            # Model performance comparison
            model_performance = {}
            for model_name in self.config.models_to_test:
                model_results = [r for r in successful_results if r.model_name == model_name]
                if model_results:
                    model_trts = [r.trts_score for r in model_results if r.trts_score is not None]
                    model_sim = [r.similarity_score for r in model_results if r.similarity_score is not None]
                    
                    model_performance[model_name] = {
                        'success_count': len(model_results),
                        'average_trts': np.mean(model_trts) if model_trts else 0,
                        'average_similarity': np.mean(model_sim) if model_sim else 0
                    }
            
            summary['model_performance'] = model_performance
        
        self.benchmark_metadata['summary'] = summary
    
    def _save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_data = {
            'metadata': self.benchmark_metadata,
            'results': [r.to_dict() for r in self.results]
        }
        
        results_file = self.output_path / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary CSV
        if self.results:
            results_df = pd.DataFrame([r.to_dict() for r in self.results])
            summary_file = self.output_path / f"benchmark_summary_{timestamp}.csv"
            results_df.to_csv(summary_file, index=False)
        
        logger.info(f"Saved benchmark results to {results_file}")
    
    def get_results_by_model(self, model_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific model."""
        return [r for r in self.results if r.model_name == model_name]
    
    def get_results_by_dataset(self, dataset_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific dataset."""
        return [r for r in self.results if r.dataset_name == dataset_name]
    
    def get_successful_results(self) -> List[BenchmarkResult]:
        """Get all successful benchmark results."""
        return [r for r in self.results if r.success]
    
    def get_failed_results(self) -> List[BenchmarkResult]:
        """Get all failed benchmark results."""
        return [r for r in self.results if not r.success]
    
    def get_leaderboard(self, metric: str = 'composite') -> List[Tuple[str, str, float]]:
        """
        Get leaderboard of model-dataset combinations by specified metric.
        
        Args:
            metric: Metric to rank by ('trts_score', 'similarity_score', 'composite', etc.)
            
        Returns:
            List of (model_name, dataset_name, score) tuples sorted by score
        """
        successful_results = self.get_successful_results()
        leaderboard = []
        
        for result in successful_results:
            if metric == 'composite' and 'composite' in result.objective_scores:
                score = result.objective_scores['composite']
            elif metric == 'trts_score' and result.trts_score is not None:
                score = result.trts_score
            elif metric == 'similarity_score' and result.similarity_score is not None:
                score = result.similarity_score
            elif metric == 'quality_score' and result.quality_score is not None:
                score = result.quality_score
            else:
                continue
            
            leaderboard.append((result.model_name, result.dataset_name, score))
        
        # Sort by score (descending)
        leaderboard.sort(key=lambda x: x[2], reverse=True)
        return leaderboard