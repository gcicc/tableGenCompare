"""
Benchmark reporting system for generating comprehensive analysis reports.

This module provides functionality to generate detailed reports from benchmark
results, including statistical analysis, model comparisons, and insights.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json
from dataclasses import dataclass
import statistics

from .benchmark_pipeline import BenchmarkResult, BenchmarkPipeline

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceStats:
    """Statistical summary for a model's performance."""
    model_name: str
    total_benchmarks: int
    successful_benchmarks: int
    success_rate: float
    
    # TRTS statistics
    avg_trts_score: float
    std_trts_score: float
    min_trts_score: float
    max_trts_score: float
    
    # Similarity statistics
    avg_similarity_score: float
    std_similarity_score: float
    min_similarity_score: float
    max_similarity_score: float
    
    # Performance statistics
    avg_training_duration: float
    avg_generation_duration: float
    avg_total_duration: float
    
    # Composite scores
    avg_composite_score: float
    std_composite_score: float


@dataclass
class DatasetDifficultyStats:
    """Statistical summary for dataset difficulty analysis."""
    dataset_name: str
    complexity_score: float
    n_features: int
    n_samples: int
    n_categorical: int
    
    # Model performance on this dataset
    best_model: str
    best_trts_score: float
    worst_model: str
    worst_trts_score: float
    
    # Difficulty indicators
    avg_model_performance: float
    performance_variance: float
    is_challenging: bool  # High variance or low average performance


class BenchmarkReporter:
    """
    Comprehensive benchmark reporting system.
    
    Generates detailed analysis reports from benchmark results including
    model comparisons, dataset difficulty analysis, and performance insights.
    """
    
    def __init__(self, benchmark_pipeline: BenchmarkPipeline):
        """
        Initialize benchmark reporter.
        
        Args:
            benchmark_pipeline: Completed benchmark pipeline with results
        """
        self.pipeline = benchmark_pipeline
        self.results = benchmark_pipeline.results
        self.successful_results = benchmark_pipeline.get_successful_results()
        
        # Cached analysis results
        self._model_stats: Optional[Dict[str, ModelPerformanceStats]] = None
        self._dataset_stats: Optional[Dict[str, DatasetDifficultyStats]] = None
        
        logger.info(f"Initialized benchmark reporter with {len(self.results)} total results ({len(self.successful_results)} successful)")
    
    def generate_comprehensive_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.
        
        Args:
            output_path: Optional path to save report files
            
        Returns:
            Complete report dictionary
        """
        logger.info("Generating comprehensive benchmark report...")
        
        report = {
            'metadata': self._generate_report_metadata(),
            'executive_summary': self._generate_executive_summary(),
            'model_analysis': self._generate_model_analysis(),
            'dataset_analysis': self._generate_dataset_analysis(),
            'comparative_analysis': self._generate_comparative_analysis(),
            'performance_insights': self._generate_performance_insights(),
            'recommendations': self._generate_recommendations()
        }
        
        if output_path:
            self._save_report(report, output_path)
        
        logger.info("Comprehensive report generation completed")
        return report
    
    def _generate_report_metadata(self) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            'report_generated': datetime.now().isoformat(),
            'benchmark_metadata': self.pipeline.benchmark_metadata,
            'total_results': len(self.results),
            'successful_results': len(self.successful_results),
            'failed_results': len(self.results) - len(self.successful_results),
            'models_tested': list(set(r.model_name for r in self.results)),
            'datasets_tested': list(set(r.dataset_name for r in self.results))
        }
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of benchmark results."""
        if not self.successful_results:
            return {'message': 'No successful benchmark results available'}
        
        # Overall statistics
        trts_scores = [r.trts_score for r in self.successful_results if r.trts_score is not None]
        similarity_scores = [r.similarity_score for r in self.successful_results if r.similarity_score is not None]
        durations = [r.duration_seconds for r in self.successful_results if r.duration_seconds is not None]
        
        # Find best performing model overall
        leaderboard = self.pipeline.get_leaderboard('composite')
        best_combination = leaderboard[0] if leaderboard else None
        
        # Model success rates
        model_success_rates = {}
        for model_name in set(r.model_name for r in self.results):
            model_results = [r for r in self.results if r.model_name == model_name]
            successful = [r for r in model_results if r.success]
            model_success_rates[model_name] = len(successful) / len(model_results) if model_results else 0
        
        return {
            'total_benchmarks_run': len(self.results),
            'success_rate': len(self.successful_results) / len(self.results) if self.results else 0,
            'average_trts_score': np.mean(trts_scores) if trts_scores else 0,
            'average_similarity_score': np.mean(similarity_scores) if similarity_scores else 0,
            'total_computation_hours': sum(durations) / 3600 if durations else 0,
            'best_model_dataset_combination': {
                'model': best_combination[0] if best_combination else None,
                'dataset': best_combination[1] if best_combination else None,
                'score': best_combination[2] if best_combination else None
            },
            'model_success_rates': model_success_rates,
            'key_findings': self._generate_key_findings()
        }
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from the benchmark results."""
        findings = []
        
        if not self.successful_results:
            findings.append("No successful benchmark results to analyze")
            return findings
        
        # Model performance comparison
        model_scores = {}
        for model_name in set(r.model_name for r in self.successful_results):
            model_results = [r for r in self.successful_results if r.model_name == model_name]
            composite_scores = [r.objective_scores.get('composite', 0) for r in model_results]
            if composite_scores:
                model_scores[model_name] = np.mean(composite_scores)
        
        if model_scores:
            best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
            worst_model = min(model_scores.keys(), key=lambda k: model_scores[k])
            
            findings.append(f"{best_model.upper()} achieved the highest average performance ({model_scores[best_model]:.3f})")
            
            if len(model_scores) > 1:
                performance_gap = model_scores[best_model] - model_scores[worst_model]
                findings.append(f"Performance gap between best ({best_model}) and worst ({worst_model}) model: {performance_gap:.3f}")
        
        # Dataset difficulty analysis
        dataset_performances = {}
        for dataset_name in set(r.dataset_name for r in self.successful_results):
            dataset_results = [r for r in self.successful_results if r.dataset_name == dataset_name]
            trts_scores = [r.trts_score for r in dataset_results if r.trts_score is not None]
            if trts_scores:
                dataset_performances[dataset_name] = {
                    'avg_performance': np.mean(trts_scores),
                    'performance_variance': np.var(trts_scores)
                }
        
        if dataset_performances:
            # Find most challenging dataset
            challenging_dataset = max(dataset_performances.keys(), 
                                    key=lambda k: dataset_performances[k]['performance_variance'])
            findings.append(f"{challenging_dataset} appears to be the most challenging dataset (highest performance variance)")
            
            # Find easiest dataset
            easiest_dataset = max(dataset_performances.keys(),
                                key=lambda k: dataset_performances[k]['avg_performance'])
            findings.append(f"{easiest_dataset} achieved the highest average model performance")
        
        # Performance trends
        trts_scores = [r.trts_score for r in self.successful_results if r.trts_score]
        if trts_scores:
            high_performers = [s for s in trts_scores if s > 0.9]  # 90%+ TRTS
            if high_performers:
                findings.append(f"{len(high_performers)}/{len(trts_scores)} results achieved >90% TRTS score")
        
        return findings
    
    def _generate_model_analysis(self) -> Dict[str, Any]:
        """Generate detailed model performance analysis."""
        if self._model_stats is None:
            self._calculate_model_statistics()
        
        model_analysis = {}
        
        for model_name, stats in self._model_stats.items():
            model_analysis[model_name] = {
                'performance_summary': {
                    'success_rate': stats.success_rate,
                    'avg_trts_score': stats.avg_trts_score,
                    'avg_similarity_score': stats.avg_similarity_score,
                    'avg_composite_score': stats.avg_composite_score
                },
                'reliability_metrics': {
                    'trts_consistency': 1 - (stats.std_trts_score / max(stats.avg_trts_score, 0.001)),
                    'similarity_consistency': 1 - (stats.std_similarity_score / max(stats.avg_similarity_score, 0.001)),
                    'performance_range': {
                        'trts': (stats.min_trts_score, stats.max_trts_score),
                        'similarity': (stats.min_similarity_score, stats.max_similarity_score)
                    }
                },
                'efficiency_metrics': {
                    'avg_training_time_minutes': stats.avg_training_duration / 60,
                    'avg_generation_time_seconds': stats.avg_generation_duration,
                    'avg_total_time_minutes': stats.avg_total_duration / 60
                },
                'strengths': self._identify_model_strengths(model_name, stats),
                'weaknesses': self._identify_model_weaknesses(model_name, stats)
            }
        
        return model_analysis
    
    def _generate_dataset_analysis(self) -> Dict[str, Any]:
        """Generate detailed dataset difficulty analysis."""
        if self._dataset_stats is None:
            self._calculate_dataset_statistics()
        
        dataset_analysis = {}
        
        for dataset_name, stats in self._dataset_stats.items():
            dataset_analysis[dataset_name] = {
                'characteristics': {
                    'complexity_score': stats.complexity_score,
                    'n_features': stats.n_features,
                    'n_samples': stats.n_samples,
                    'n_categorical': stats.n_categorical,
                    'size_category': 'large' if stats.n_samples > 10000 else 'medium' if stats.n_samples > 1000 else 'small'
                },
                'difficulty_assessment': {
                    'is_challenging': stats.is_challenging,
                    'avg_model_performance': stats.avg_model_performance,
                    'performance_variance': stats.performance_variance,
                    'difficulty_score': stats.performance_variance / max(stats.avg_model_performance, 0.001)
                },
                'model_performance': {
                    'best_model': stats.best_model,
                    'best_score': stats.best_trts_score,
                    'worst_model': stats.worst_model,
                    'worst_score': stats.worst_trts_score,
                    'performance_gap': stats.best_trts_score - stats.worst_trts_score
                }
            }
        
        return dataset_analysis
    
    def _generate_comparative_analysis(self) -> Dict[str, Any]:
        """Generate comparative analysis between models."""
        if not self.successful_results:
            return {}
        
        models = list(set(r.model_name for r in self.successful_results))
        
        # Head-to-head comparisons
        head_to_head = {}
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                comparison = self._compare_models(model1, model2)
                head_to_head[f"{model1}_vs_{model2}"] = comparison
        
        # Statistical significance tests (simplified)
        significance_tests = self._perform_significance_tests()
        
        # Model rankings by different metrics
        rankings = {
            'trts_score': self._rank_models_by_metric('trts_score'),
            'similarity_score': self._rank_models_by_metric('similarity_score'),
            'composite_score': self._rank_models_by_metric('composite'),
            'efficiency': self._rank_models_by_efficiency()
        }
        
        return {
            'head_to_head_comparisons': head_to_head,
            'statistical_significance': significance_tests,
            'model_rankings': rankings,
            'overall_winner': self._determine_overall_winner()
        }
    
    def _generate_performance_insights(self) -> Dict[str, Any]:
        """Generate performance insights and patterns."""
        insights = {
            'data_size_impact': self._analyze_data_size_impact(),
            'complexity_impact': self._analyze_complexity_impact(),
            'categorical_data_impact': self._analyze_categorical_impact(),
            'training_time_analysis': self._analyze_training_times(),
            'scalability_analysis': self._analyze_scalability()
        }
        
        return insights
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations based on benchmark results."""
        recommendations = {
            'model_selection': self._recommend_model_selection(),
            'dataset_considerations': self._recommend_dataset_considerations(),
            'optimization_opportunities': self._recommend_optimizations(),
            'future_benchmarking': self._recommend_future_benchmarks()
        }
        
        return recommendations
    
    def _calculate_model_statistics(self):
        """Calculate detailed statistics for each model."""
        self._model_stats = {}
        
        for model_name in set(r.model_name for r in self.results):
            model_results = [r for r in self.results if r.model_name == model_name]
            successful_results = [r for r in model_results if r.success]
            
            if not successful_results:
                continue
            
            # Extract metrics
            trts_scores = [r.trts_score for r in successful_results if r.trts_score is not None]
            similarity_scores = [r.similarity_score for r in successful_results if r.similarity_score is not None]
            composite_scores = [r.objective_scores.get('composite', 0) for r in successful_results]
            training_durations = [r.training_duration for r in successful_results if r.training_duration is not None]
            generation_durations = [r.generation_duration for r in successful_results if r.generation_duration is not None]
            total_durations = [r.duration_seconds for r in successful_results if r.duration_seconds is not None]
            
            self._model_stats[model_name] = ModelPerformanceStats(
                model_name=model_name,
                total_benchmarks=len(model_results),
                successful_benchmarks=len(successful_results),
                success_rate=len(successful_results) / len(model_results),
                
                avg_trts_score=np.mean(trts_scores) if trts_scores else 0,
                std_trts_score=np.std(trts_scores) if trts_scores else 0,
                min_trts_score=min(trts_scores) if trts_scores else 0,
                max_trts_score=max(trts_scores) if trts_scores else 0,
                
                avg_similarity_score=np.mean(similarity_scores) if similarity_scores else 0,
                std_similarity_score=np.std(similarity_scores) if similarity_scores else 0,
                min_similarity_score=min(similarity_scores) if similarity_scores else 0,
                max_similarity_score=max(similarity_scores) if similarity_scores else 0,
                
                avg_training_duration=np.mean(training_durations) if training_durations else 0,
                avg_generation_duration=np.mean(generation_durations) if generation_durations else 0,
                avg_total_duration=np.mean(total_durations) if total_durations else 0,
                
                avg_composite_score=np.mean(composite_scores) if composite_scores else 0,
                std_composite_score=np.std(composite_scores) if composite_scores else 0
            )
    
    def _calculate_dataset_statistics(self):
        """Calculate difficulty statistics for each dataset."""
        self._dataset_stats = {}
        
        for dataset_name in set(r.dataset_name for r in self.successful_results):
            dataset_results = [r for r in self.successful_results if r.dataset_name == dataset_name]
            
            if not dataset_results:
                continue
            
            # Get dataset characteristics from first result
            dataset_config = dataset_results[0].dataset_config
            
            # Calculate performance statistics
            trts_scores = [r.trts_score for r in dataset_results if r.trts_score is not None]
            
            if not trts_scores:
                continue
            
            avg_performance = np.mean(trts_scores)
            performance_variance = np.var(trts_scores)
            
            # Find best and worst performing models
            model_performances = {}
            for result in dataset_results:
                if result.trts_score is not None:
                    if result.model_name not in model_performances:
                        model_performances[result.model_name] = []
                    model_performances[result.model_name].append(result.trts_score)
            
            model_avg_performances = {
                model: np.mean(scores) for model, scores in model_performances.items()
            }
            
            best_model = max(model_avg_performances.keys(), key=lambda k: model_avg_performances[k])
            worst_model = min(model_avg_performances.keys(), key=lambda k: model_avg_performances[k])
            
            self._dataset_stats[dataset_name] = DatasetDifficultyStats(
                dataset_name=dataset_name,
                complexity_score=dataset_config.complexity_score,
                n_features=dataset_config.n_features or 0,
                n_samples=dataset_config.n_samples or 0,
                n_categorical=dataset_config.n_categorical or 0,
                
                best_model=best_model,
                best_trts_score=model_avg_performances[best_model],
                worst_model=worst_model,
                worst_trts_score=model_avg_performances[worst_model],
                
                avg_model_performance=avg_performance,
                performance_variance=performance_variance,
                is_challenging=performance_variance > 0.01 or avg_performance < 0.7  # Threshold for "challenging"
            )
    
    def _identify_model_strengths(self, model_name: str, stats: ModelPerformanceStats) -> List[str]:
        """Identify strengths of a specific model."""
        strengths = []
        
        if stats.success_rate > 0.95:
            strengths.append("High reliability (>95% success rate)")
        
        if stats.avg_trts_score > 0.9:
            strengths.append("Excellent TRTS performance (>90%)")
        
        if stats.avg_similarity_score > 0.9:
            strengths.append("High similarity to original data (>90%)")
        
        if stats.std_trts_score < 0.05:
            strengths.append("Consistent TRTS performance (low variance)")
        
        if stats.avg_training_duration < 60:  # Less than 1 minute
            strengths.append("Fast training time")
        
        return strengths
    
    def _identify_model_weaknesses(self, model_name: str, stats: ModelPerformanceStats) -> List[str]:
        """Identify weaknesses of a specific model."""
        weaknesses = []
        
        if stats.success_rate < 0.8:
            weaknesses.append("Low reliability (<80% success rate)")
        
        if stats.avg_trts_score < 0.7:
            weaknesses.append("Poor TRTS performance (<70%)")
        
        if stats.avg_similarity_score < 0.8:
            weaknesses.append("Low similarity to original data (<80%)")
        
        if stats.std_trts_score > 0.1:
            weaknesses.append("Inconsistent TRTS performance (high variance)")
        
        if stats.avg_training_duration > 300:  # More than 5 minutes
            weaknesses.append("Slow training time")
        
        return weaknesses
    
    def _compare_models(self, model1: str, model2: str) -> Dict[str, Any]:
        """Compare two models head-to-head."""
        model1_results = [r for r in self.successful_results if r.model_name == model1]
        model2_results = [r for r in self.successful_results if r.model_name == model2]
        
        comparison = {
            'model1': model1,
            'model2': model2,
            'sample_sizes': {
                model1: len(model1_results),
                model2: len(model2_results)
            }
        }
        
        # Compare on common datasets
        common_datasets = set(r.dataset_name for r in model1_results) & set(r.dataset_name for r in model2_results)
        
        if common_datasets:
            head_to_head_wins = {model1: 0, model2: 0, 'ties': 0}
            
            for dataset in common_datasets:
                model1_score = np.mean([r.trts_score for r in model1_results 
                                     if r.dataset_name == dataset and r.trts_score is not None])
                model2_score = np.mean([r.trts_score for r in model2_results 
                                     if r.dataset_name == dataset and r.trts_score is not None])
                
                if abs(model1_score - model2_score) < 0.01:  # Tie threshold
                    head_to_head_wins['ties'] += 1
                elif model1_score > model2_score:
                    head_to_head_wins[model1] += 1
                else:
                    head_to_head_wins[model2] += 1
            
            comparison['head_to_head_results'] = head_to_head_wins
            comparison['common_datasets'] = len(common_datasets)
        
        return comparison
    
    def _perform_significance_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests (simplified)."""
        # This is a simplified version - in practice you'd use proper statistical tests
        return {
            'note': 'Statistical significance testing would require proper statistical analysis',
            'recommendation': 'Use paired t-tests or Wilcoxon signed-rank tests for formal comparison'
        }
    
    def _rank_models_by_metric(self, metric: str) -> List[Tuple[str, float]]:
        """Rank models by specified metric."""
        model_scores = {}
        
        for model_name in set(r.model_name for r in self.successful_results):
            model_results = [r for r in self.successful_results if r.model_name == model_name]
            
            if metric == 'trts_score':
                scores = [r.trts_score for r in model_results if r.trts_score is not None]
            elif metric == 'similarity_score':
                scores = [r.similarity_score for r in model_results if r.similarity_score is not None]
            elif metric == 'composite':
                scores = [r.objective_scores.get('composite', 0) for r in model_results]
            else:
                continue
            
            if scores:
                model_scores[model_name] = np.mean(scores)
        
        return sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _rank_models_by_efficiency(self) -> List[Tuple[str, float]]:
        """Rank models by efficiency (performance per time)."""
        model_efficiency = {}
        
        for model_name in set(r.model_name for r in self.successful_results):
            model_results = [r for r in self.successful_results if r.model_name == model_name]
            
            performances = [r.trts_score for r in model_results if r.trts_score is not None]
            durations = [r.duration_seconds for r in model_results if r.duration_seconds is not None]
            
            if performances and durations:
                avg_performance = np.mean(performances)
                avg_duration = np.mean(durations)
                efficiency = avg_performance / (avg_duration / 3600)  # Performance per hour
                model_efficiency[model_name] = efficiency
        
        return sorted(model_efficiency.items(), key=lambda x: x[1], reverse=True)
    
    def _determine_overall_winner(self) -> Dict[str, Any]:
        """Determine overall winner based on multiple criteria."""
        composite_ranking = self._rank_models_by_metric('composite')
        
        if not composite_ranking:
            return {'winner': None, 'reason': 'No successful results to analyze'}
        
        winner = composite_ranking[0]
        return {
            'winner': winner[0],
            'score': winner[1],
            'reason': 'Highest average composite score across all datasets'
        }
    
    def _analyze_data_size_impact(self) -> Dict[str, Any]:
        """Analyze impact of dataset size on model performance."""
        size_analysis = {}
        
        # Group results by dataset size
        for result in self.successful_results:
            n_samples = result.dataset_config.n_samples or 0
            
            if n_samples < 1000:
                size_category = 'small'
            elif n_samples < 10000:
                size_category = 'medium'
            else:
                size_category = 'large'
            
            if size_category not in size_analysis:
                size_analysis[size_category] = []
            
            if result.trts_score is not None:
                size_analysis[size_category].append(result.trts_score)
        
        # Calculate statistics for each size category
        size_stats = {}
        for size_cat, scores in size_analysis.items():
            if scores:
                size_stats[size_cat] = {
                    'count': len(scores),
                    'avg_performance': np.mean(scores),
                    'std_performance': np.std(scores)
                }
        
        return size_stats
    
    def _analyze_complexity_impact(self) -> Dict[str, Any]:
        """Analyze impact of dataset complexity on performance."""
        # This would analyze correlation between complexity score and performance
        complexities = []
        performances = []
        
        for result in self.successful_results:
            if result.trts_score is not None and result.dataset_config.complexity_score:
                complexities.append(result.dataset_config.complexity_score)
                performances.append(result.trts_score)
        
        if len(complexities) > 2:
            correlation = np.corrcoef(complexities, performances)[0, 1]
            return {
                'correlation': correlation,
                'interpretation': 'negative correlation suggests performance decreases with complexity' if correlation < -0.3 else 
                               'positive correlation suggests performance increases with complexity' if correlation > 0.3 else
                               'weak correlation between complexity and performance'
            }
        
        return {'correlation': None, 'interpretation': 'Insufficient data for correlation analysis'}
    
    def _analyze_categorical_impact(self) -> Dict[str, Any]:
        """Analyze impact of categorical features on performance."""
        # Group by number of categorical features
        categorical_analysis = {'low': [], 'medium': [], 'high': []}
        
        for result in self.successful_results:
            n_categorical = result.dataset_config.n_categorical or 0
            n_features = result.dataset_config.n_features or 1
            
            categorical_ratio = n_categorical / n_features
            
            if categorical_ratio < 0.2:
                category = 'low'
            elif categorical_ratio < 0.5:
                category = 'medium'
            else:
                category = 'high'
            
            if result.trts_score is not None:
                categorical_analysis[category].append(result.trts_score)
        
        # Calculate statistics
        categorical_stats = {}
        for cat_level, scores in categorical_analysis.items():
            if scores:
                categorical_stats[cat_level] = {
                    'count': len(scores),
                    'avg_performance': np.mean(scores),
                    'std_performance': np.std(scores)
                }
        
        return categorical_stats
    
    def _analyze_training_times(self) -> Dict[str, Any]:
        """Analyze training time patterns."""
        training_analysis = {}
        
        for model_name in set(r.model_name for r in self.successful_results):
            model_results = [r for r in self.successful_results if r.model_name == model_name]
            training_times = [r.training_duration for r in model_results if r.training_duration is not None]
            
            if training_times:
                training_analysis[model_name] = {
                    'avg_training_time_seconds': np.mean(training_times),
                    'min_training_time': min(training_times),
                    'max_training_time': max(training_times),
                    'std_training_time': np.std(training_times)
                }
        
        return training_analysis
    
    def _analyze_scalability(self) -> Dict[str, Any]:
        """Analyze scalability patterns."""
        # This would analyze how performance/time scales with data size
        return {
            'note': 'Scalability analysis requires more detailed timing and memory measurements',
            'recommendation': 'Implement detailed resource monitoring for comprehensive scalability analysis'
        }
    
    def _recommend_model_selection(self) -> List[str]:
        """Generate model selection recommendations."""
        recommendations = []
        
        if not self.successful_results:
            return ["No successful results available for recommendations"]
        
        # Overall best performer
        composite_ranking = self._rank_models_by_metric('composite')
        if composite_ranking:
            best_model = composite_ranking[0][0]
            recommendations.append(f"For general use: {best_model.upper()} showed the best overall performance")
        
        # Efficiency recommendation
        efficiency_ranking = self._rank_models_by_efficiency()
        if efficiency_ranking:
            efficient_model = efficiency_ranking[0][0]
            recommendations.append(f"For time-critical applications: {efficient_model.upper()} offers the best performance per time")
        
        # Reliability recommendation
        if self._model_stats:
            reliable_models = [(name, stats.success_rate) for name, stats in self._model_stats.items()]
            reliable_models.sort(key=lambda x: x[1], reverse=True)
            if reliable_models:
                recommendations.append(f"For reliability: {reliable_models[0][0].upper()} has the highest success rate ({reliable_models[0][1]:.1%})")
        
        return recommendations
    
    def _recommend_dataset_considerations(self) -> List[str]:
        """Generate dataset-specific recommendations."""
        recommendations = []
        
        if self._dataset_stats:
            challenging_datasets = [name for name, stats in self._dataset_stats.items() if stats.is_challenging]
            if challenging_datasets:
                recommendations.append(f"Challenging datasets requiring extra attention: {', '.join(challenging_datasets)}")
        
        # Size-based recommendations
        size_impact = self._analyze_data_size_impact()
        if 'small' in size_impact and 'large' in size_impact:
            small_perf = size_impact['small']['avg_performance']
            large_perf = size_impact['large']['avg_performance']
            if small_perf > large_perf:
                recommendations.append("Small datasets tend to achieve better synthetic data quality")
            else:
                recommendations.append("Larger datasets tend to achieve better synthetic data quality")
        
        return recommendations
    
    def _recommend_optimizations(self) -> List[str]:
        """Generate optimization recommendations."""
        return [
            "Consider hyperparameter optimization for underperforming model-dataset combinations",
            "Implement early stopping for models with long training times",
            "Consider ensemble approaches for challenging datasets",
            "Investigate preprocessing techniques for categorical data handling"
        ]
    
    def _recommend_future_benchmarks(self) -> List[str]:
        """Generate recommendations for future benchmarking."""
        return [
            "Include more diverse datasets to improve generalizability",
            "Add memory usage and computational resource monitoring",
            "Implement cross-validation for more robust performance estimates",
            "Consider domain-specific evaluation metrics",
            "Add privacy preservation metrics for sensitive datasets"
        ]
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _save_report(self, report: Dict[str, Any], output_path: str):
        """Save comprehensive report to files."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full report as JSON with numpy type conversion
        report_json = self._convert_numpy_types(report)
        report_file = output_dir / f"comprehensive_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report_json, f, indent=2)
        
        # Save executive summary as text
        summary = report['executive_summary']
        summary_file = output_dir / f"executive_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("BENCHMARK EXECUTIVE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Benchmarks: {summary['total_benchmarks_run']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1%}\n")
            f.write(f"Average TRTS Score: {summary['average_trts_score']:.3f}\n")
            f.write(f"Average Similarity Score: {summary['average_similarity_score']:.3f}\n")
            f.write(f"Total Computation Time: {summary['total_computation_hours']:.2f} hours\n\n")
            
            if summary['best_model_dataset_combination']['model']:
                f.write("Best Model-Dataset Combination:\n")
                f.write(f"  Model: {summary['best_model_dataset_combination']['model']}\n")
                f.write(f"  Dataset: {summary['best_model_dataset_combination']['dataset']}\n")
                f.write(f"  Score: {summary['best_model_dataset_combination']['score']:.3f}\n\n")
            
            f.write("Key Findings:\n")
            for i, finding in enumerate(summary['key_findings'], 1):
                f.write(f"  {i}. {finding}\n")
        
        logger.info(f"Saved comprehensive report to {output_dir}")
    
    def export_results_csv(self, output_path: str) -> str:
        """Export results to CSV format."""
        if not self.results:
            raise ValueError("No results to export")
        
        # Convert results to DataFrame
        results_data = []
        for result in self.results:
            row = result.to_dict()
            # Flatten nested dictionaries
            if 'dataset_summary' in row and row['dataset_summary']:
                for key, value in row['dataset_summary'].items():
                    row[f'dataset_{key}'] = value
                del row['dataset_summary']
            
            if 'objective_scores' in row and row['objective_scores']:
                for key, value in row['objective_scores'].items():
                    row[f'objective_{key}'] = value
                del row['objective_scores']
            
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        
        output_file = Path(output_path)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported results to CSV: {output_file}")
        return str(output_file)