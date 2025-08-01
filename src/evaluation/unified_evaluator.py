"""
Unified evaluator for comprehensive synthetic data assessment.

This module orchestrates all evaluation components to provide comprehensive
synthetic data quality assessment with statistical analysis, visualization,
and reporting capabilities.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

from .statistical_analysis import StatisticalAnalyzer
from .similarity_metrics import SimilarityCalculator
from .trts_framework import TRTSEvaluator
from .visualization_engine import VisualizationEngine

logger = logging.getLogger(__name__)


class UnifiedEvaluator:
    """
    Unified evaluator for comprehensive synthetic data assessment.
    
    This class orchestrates all evaluation components to provide a single
    interface for comprehensive synthetic data quality assessment, matching
    all functionality from Phase 1 Section 4.4 and 5.5.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize unified evaluator.
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        
        # Initialize evaluation components
        self.statistical_analyzer = StatisticalAnalyzer()
        self.similarity_calculator = SimilarityCalculator()
        self.trts_evaluator = TRTSEvaluator(random_state=random_state)
        self.visualization_engine = VisualizationEngine()
        
        logger.info("UnifiedEvaluator initialized with all evaluation components")
    
    def run_complete_evaluation(
        self,
        model: Any,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        dataset_metadata: Dict[str, Any],
        output_dir: str,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline - equivalent to Phase 1 Section 4.4.
        
        This single method produces ALL outputs from Phase 1 Section 4.4:
        - Statistical comparisons
        - TRTS framework evaluation  
        - Distribution comparisons
        - Performance dashboards
        - All visualizations
        - All CSV exports
        
        Args:
            model: Trained synthetic data model
            original_data: Original training dataset
            synthetic_data: Generated synthetic dataset
            dataset_metadata: Dataset metadata from DatasetHandler
            output_dir: Directory to save all outputs
            target_column: Target column name (auto-detected if None)
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting complete evaluation pipeline")
        
        # Auto-detect target column if not provided
        if target_column is None and 'target_info' in dataset_metadata:
            target_column = dataset_metadata['target_info']['column']
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        evaluation_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
            'dataset_info': dataset_metadata.get('dataset_info', {}),
            'target_column': target_column
        }
        
        # 1. STATISTICAL ANALYSIS (Section 4.4 statistical comparison tables)
        logger.info("Running statistical analysis...")
        stats_comparison_df, stats_summary = self.statistical_analyzer.comprehensive_statistical_comparison(
            original_data, synthetic_data, target_column
        )
        
        evaluation_results['statistical_analysis'] = {
            'stats_comparison_df': stats_comparison_df,
            'summary_statistics': stats_summary
        }
        
        # Export statistical comparison CSV (matches Phase 1 export)
        stats_csv_path = output_path / 'comprehensive_statistical_comparison.csv'
        stats_comparison_df.to_csv(stats_csv_path, index=False)
        logger.info(f"Statistical comparison exported: {stats_csv_path}")
        
        # 2. CORRELATION ANALYSIS
        logger.info("Running correlation analysis...")
        correlation_results = self.statistical_analyzer.correlation_analysis(
            original_data, synthetic_data
        )
        evaluation_results['correlation_analysis'] = correlation_results
        
        # 3. SIMILARITY METRICS (Advanced metrics from Section 5)
        logger.info("Calculating similarity metrics...")
        final_similarity, univariate_sim, bivariate_sim = self.similarity_calculator.evaluate_overall_similarity(
            original_data, synthetic_data, target_column
        )
        
        feature_similarities = self.similarity_calculator.calculate_feature_wise_similarities(
            original_data, synthetic_data, target_column
        )
        
        evaluation_results['similarity_analysis'] = {
            'final_similarity': final_similarity,
            'univariate_similarity': univariate_sim,
            'bivariate_similarity': bivariate_sim,
            'feature_similarities': feature_similarities
        }
        
        # 4. TRTS FRAMEWORK EVALUATION (Section 4.4 TRTS implementation)
        logger.info("Running TRTS framework evaluation...")
        trts_results = self.trts_evaluator.evaluate_trts_scenarios(
            original_data, synthetic_data, target_column
        )
        
        evaluation_results['trts_results'] = trts_results
        
        # Create and export TRTS summary table (matches Phase 1 export)
        trts_summary_df = self.trts_evaluator.create_trts_summary_table(trts_results['trts_scores'])
        trts_csv_path = output_path / 'trts_evaluation.csv'
        trts_summary_df.to_csv(trts_csv_path, index=False)
        logger.info(f"TRTS evaluation exported: {trts_csv_path}")
        
        # 5. DATA QUALITY ASSESSMENT
        logger.info("Assessing data quality...")
        data_quality = self.statistical_analyzer.data_quality_assessment(
            original_data, synthetic_data
        )
        evaluation_results['data_quality'] = data_quality
        
        # 6. GENERATE ALL VISUALIZATIONS (All Phase 1 Section 4.4 plots)
        logger.info("Generating all visualizations...")
        
        # Prepare visualization data
        viz_stats_results = {
            'stats_comparison_df': stats_comparison_df,
            'data_quality': data_quality,
            'utility_score': trts_results['utility_score_percent'],
            'quality_score': trts_results['quality_score_percent']
        }
        
        # Generate all plots (matches Phase 1 outputs exactly)
        plot_files = self.visualization_engine.save_all_plots(
            original_data=original_data,
            synthetic_data=synthetic_data,
            trts_results=trts_results['trts_scores'],
            stats_results=viz_stats_results,
            output_dir=str(output_path),
            dataset_name=dataset_metadata.get('dataset_info', {}).get('name', ''),
            model_name=evaluation_results['model_info'].get('model_type', '')
        )
        
        evaluation_results['plot_files'] = plot_files
        
        # 7. CREATE COMPREHENSIVE SUMMARY REPORT
        logger.info("Creating comprehensive summary report...")
        summary_report = self._create_summary_report(evaluation_results)
        
        # Export summary report
        summary_path = output_path / 'evaluation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        # Export metadata
        metadata_path = output_path / 'evaluation_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Complete evaluation finished - all outputs saved to {output_dir}")
        
        return evaluation_results
    
    def evaluate_model_comparison(
        self,
        default_results: Dict[str, Any],
        optimized_results: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Compare results between default and optimized models (Section 5.5 functionality).
        
        Args:
            default_results: Results from default model evaluation
            optimized_results: Results from optimized model evaluation
            output_dir: Directory to save comparison results
            
        Returns:
            Dictionary containing comparison analysis
        """
        logger.info("Running model comparison analysis")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract statistical comparisons
        default_stats = default_results.get('statistical_analysis', {}).get('stats_comparison_df')
        optimized_stats = optimized_results.get('statistical_analysis', {}).get('stats_comparison_df')
        
        comparison_results = {
            'comparison_timestamp': datetime.now().isoformat(),
            'models_compared': ['default', 'optimized']
        }
        
        if default_stats is not None and optimized_stats is not None:
            # Calculate improvement metrics
            default_mean_error = np.mean(default_stats['Mean_Diff'].values)
            optimized_mean_error = np.mean(optimized_stats['Mean_Diff'].values)
            
            default_std_error = np.mean(default_stats['Std_Diff'].values)
            optimized_std_error = np.mean(optimized_stats['Std_Diff'].values)
            
            default_similar_features = default_stats['KS_Similar'].value_counts().get('Yes', 0)
            optimized_similar_features = optimized_stats['KS_Similar'].value_counts().get('Yes', 0)
            
            # Create improvement summary
            improvement_summary = pd.DataFrame({
                'Metric': [
                    'Average Mean Error',
                    'Average Std Error',
                    'Similar Distributions (KS test)',
                    'Total Features Evaluated'
                ],
                'Default Model': [
                    f"{default_mean_error:.4f}",
                    f"{default_std_error:.4f}",
                    f"{default_similar_features}/{len(default_stats)}",
                    f"{len(default_stats)}"
                ],
                'Optimized Model': [
                    f"{optimized_mean_error:.4f}",
                    f"{optimized_std_error:.4f}",
                    f"{optimized_similar_features}/{len(optimized_stats)}",
                    f"{len(optimized_stats)}"
                ],
                'Improvement': [
                    f"{((default_mean_error - optimized_mean_error) / default_mean_error * 100):+.1f}%",
                    f"{((default_std_error - optimized_std_error) / default_std_error * 100):+.1f}%",
                    f"{optimized_similar_features - default_similar_features:+d} features",
                    "Same"
                ]
            })
            
            comparison_results['improvement_summary'] = improvement_summary
            
            # Export comparison summary
            comparison_csv_path = output_path / 'model_comparison_summary.csv'
            improvement_summary.to_csv(comparison_csv_path, index=False)
            logger.info(f"Model comparison summary exported: {comparison_csv_path}")
        
        # Compare TRTS results
        default_trts = default_results.get('trts_results', {}).get('trts_scores', {})
        optimized_trts = optimized_results.get('trts_results', {}).get('trts_scores', {})
        
        if default_trts and optimized_trts:
            trts_comparison = {}
            for scenario in ['TRTR', 'TSTS', 'TRTS', 'TSTR']:
                if scenario in default_trts and scenario in optimized_trts:
                    improvement = ((optimized_trts[scenario] - default_trts[scenario]) / default_trts[scenario]) * 100
                    trts_comparison[scenario] = {
                        'default': default_trts[scenario],
                        'optimized': optimized_trts[scenario],
                        'improvement_percent': improvement
                    }
            
            comparison_results['trts_comparison'] = trts_comparison
        
        # Export full comparison results
        comparison_path = output_path / 'full_model_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        logger.info(f"Model comparison completed - results saved to {output_dir}")
        
        return comparison_results
    
    def _create_summary_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive summary report from evaluation results.
        
        Args:
            evaluation_results: Complete evaluation results
            
        Returns:
            Summary report dictionary
        """
        summary = {
            'evaluation_summary': {
                'timestamp': evaluation_results['evaluation_timestamp'],
                'model': evaluation_results.get('model_info', {}).get('model_type', 'Unknown'),
                'dataset': evaluation_results.get('dataset_info', {}).get('name', 'Unknown')
            }
        }
        
        # Statistical summary
        if 'statistical_analysis' in evaluation_results:
            stats_summary = evaluation_results['statistical_analysis'].get('summary_statistics', {})
            summary['statistical_summary'] = {
                'average_mean_error': stats_summary.get('average_mean_error', 0),
                'similarity_percentage': stats_summary.get('similarity_percentage', 0),
                'features_evaluated': stats_summary.get('total_features', 0)
            }
        
        # TRTS summary
        if 'trts_results' in evaluation_results:
            trts_data = evaluation_results['trts_results']
            summary['trts_summary'] = {
                'utility_score_percent': trts_data.get('utility_score_percent', 0),
                'quality_score_percent': trts_data.get('quality_score_percent', 0),
                'overall_score_percent': trts_data.get('overall_score_percent', 0)
            }
        
        # Similarity summary
        if 'similarity_analysis' in evaluation_results:
            sim_data = evaluation_results['similarity_analysis']
            summary['similarity_summary'] = {
                'final_similarity': sim_data.get('final_similarity', 0),
                'univariate_similarity': sim_data.get('univariate_similarity', 0),
                'bivariate_similarity': sim_data.get('bivariate_similarity', 0)
            }
        
        # Data quality summary
        if 'data_quality' in evaluation_results:
            quality_data = evaluation_results['data_quality']
            summary['data_quality_summary'] = {
                'data_type_consistency': quality_data.get('data_type_consistency', 0),
                'range_validity_percentage': quality_data.get('range_validity_percentage', 0),
                'shape_match': quality_data.get('shape_match', False),
                'column_match': quality_data.get('column_match', False)
            }
        
        # Overall assessment
        trts_score = summary.get('trts_summary', {}).get('overall_score_percent', 0)
        similarity_score = summary.get('similarity_summary', {}).get('final_similarity', 0) * 100
        
        overall_score = (trts_score + similarity_score) / 2
        
        if overall_score > 90:
            assessment = "Excellent synthetic data quality"
        elif overall_score > 80:
            assessment = "Good synthetic data quality"
        elif overall_score > 70:
            assessment = "Moderate synthetic data quality"
        else:
            assessment = "Poor synthetic data quality - improvements needed"
        
        summary['overall_assessment'] = {
            'score': overall_score,
            'assessment': assessment
        }
        
        return summary