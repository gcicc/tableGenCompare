#!/usr/bin/env python3
"""
Intelligent model selection and auto-recommendation system.
Analyzes dataset characteristics and recommends optimal models with configurations.
"""

import logging
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports
from models.model_factory import ModelFactory
from evaluation.unified_evaluator import UnifiedEvaluator

logger = logging.getLogger(__name__)

@dataclass
class DatasetProfile:
    """Profile of a dataset's characteristics."""
    
    # Basic properties
    n_samples: int
    n_features: int
    n_numerical: int
    n_categorical: int
    n_binary: int
    
    # Size categories
    size_category: str  # tiny, small, medium, large, huge
    complexity_score: float  # 0-1 score
    
    # Data quality
    missing_ratio: float
    duplicate_ratio: float
    
    # Distribution properties
    numerical_skewness: float
    categorical_cardinality: Dict[str, int]
    feature_correlation: float
    
    # Target properties (if available)
    target_type: Optional[str] = None
    target_balance: Optional[float] = None
    target_entropy: Optional[float] = None
    
    # Domain classification
    domain_hint: Optional[str] = None  # clinical, financial, marketing, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class ModelRecommendation:
    """Recommendation for a specific model."""
    
    model_name: str
    confidence_score: float  # 0-1 confidence in recommendation
    expected_performance: float  # Predicted performance score
    reasons: List[str]  # Why this model was recommended
    warnings: List[str]  # Potential issues or limitations
    
    # Recommended configuration
    recommended_config: Dict[str, Any]
    estimated_training_time: float  # In minutes
    resource_requirements: Dict[str, str]  # memory, compute, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class DatasetProfiler:
    """Analyzes datasets and creates comprehensive profiles."""
    
    def __init__(self):
        self.domain_keywords = {
            'clinical': ['age', 'patient', 'diagnosis', 'treatment', 'symptom', 'medical', 'health', 'disease'],
            'financial': ['income', 'salary', 'credit', 'loan', 'debt', 'balance', 'payment', 'transaction'],
            'marketing': ['customer', 'campaign', 'click', 'conversion', 'revenue', 'segment', 'channel'],
            'hr': ['employee', 'salary', 'department', 'performance', 'experience', 'promotion'],
            'retail': ['product', 'price', 'sales', 'inventory', 'customer', 'purchase', 'category'],
            'sensor': ['temperature', 'pressure', 'voltage', 'sensor', 'measurement', 'timestamp'],
            'social': ['user', 'post', 'like', 'share', 'comment', 'social', 'network']
        }
    
    def profile_dataset(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> DatasetProfile:
        """Create comprehensive dataset profile."""
        
        logger.info(f"Profiling dataset: {data.shape}")
        
        # Basic properties
        n_samples, n_features = data.shape
        
        # Data types analysis
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        n_numerical = len(numerical_cols)
        n_categorical = len(categorical_cols)
        
        # Binary columns (categorical with 2 unique values or boolean)
        binary_cols = []
        for col in data.columns:
            if data[col].nunique() == 2:
                binary_cols.append(col)
        n_binary = len(binary_cols)
        
        # Data quality metrics
        missing_ratio = data.isnull().sum().sum() / (n_samples * n_features)
        duplicate_ratio = data.duplicated().sum() / n_samples
        
        # Size categorization
        size_category = self._categorize_size(n_samples, n_features)
        
        # Complexity score (combination of features, cardinality, interactions)
        complexity_score = self._calculate_complexity(data)
        
        # Numerical distribution analysis
        if n_numerical > 0:
            numerical_skewness = data[numerical_cols].skew().abs().mean()
        else:
            numerical_skewness = 0.0
        
        # Categorical cardinality
        categorical_cardinality = {}
        for col in categorical_cols:
            categorical_cardinality[col] = data[col].nunique()
        
        # Feature correlation (for numerical features)
        if n_numerical > 1:
            corr_matrix = data[numerical_cols].corr().abs()
            # Average correlation excluding diagonal
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            feature_correlation = corr_matrix.where(mask).stack().mean()
        else:
            feature_correlation = 0.0
        
        # Target analysis
        target_type = None
        target_balance = None
        target_entropy = None
        
        if target_column and target_column in data.columns:
            target_type = self._analyze_target_type(data[target_column])
            if target_type == 'binary':
                target_balance = min(data[target_column].value_counts(normalize=True))
            elif target_type == 'multiclass':
                # Calculate entropy
                probs = data[target_column].value_counts(normalize=True)
                target_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Domain classification
        domain_hint = self._classify_domain(data)
        
        return DatasetProfile(
            n_samples=int(n_samples),
            n_features=int(n_features),
            n_numerical=int(n_numerical),
            n_categorical=int(n_categorical),
            n_binary=int(n_binary),
            size_category=size_category,
            complexity_score=float(complexity_score),
            missing_ratio=float(missing_ratio),
            duplicate_ratio=float(duplicate_ratio),
            numerical_skewness=float(numerical_skewness),
            categorical_cardinality=categorical_cardinality,
            feature_correlation=float(feature_correlation),
            target_type=target_type,
            target_balance=float(target_balance) if target_balance is not None else None,
            target_entropy=float(target_entropy) if target_entropy is not None else None,
            domain_hint=domain_hint
        )
    
    def _categorize_size(self, n_samples: int, n_features: int) -> str:
        """Categorize dataset size."""
        total_cells = n_samples * n_features
        
        if total_cells < 1000:
            return "tiny"
        elif total_cells < 10000:
            return "small"
        elif total_cells < 100000:
            return "medium"
        elif total_cells < 1000000:
            return "large"
        else:
            return "huge"
    
    def _calculate_complexity(self, data: pd.DataFrame) -> float:
        """Calculate dataset complexity score (0-1)."""
        
        # Feature complexity
        feature_complexity = min(len(data.columns) / 50, 1.0)  # Normalize to 50 features
        
        # Cardinality complexity (for categorical features)
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            avg_cardinality = np.mean([data[col].nunique() for col in categorical_cols])
            cardinality_complexity = min(avg_cardinality / 20, 1.0)  # Normalize to 20 categories
        else:
            cardinality_complexity = 0.0
        
        # Missing data complexity
        missing_complexity = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        # Overall complexity
        complexity = (
            0.4 * feature_complexity +
            0.3 * cardinality_complexity +
            0.3 * missing_complexity
        )
        
        return min(complexity, 1.0)
    
    def _analyze_target_type(self, target_series: pd.Series) -> str:
        """Analyze target column type."""
        
        n_unique = target_series.nunique()
        
        if n_unique == 2:
            return "binary"
        elif n_unique <= 10 and target_series.dtype == 'object':
            return "multiclass"
        elif n_unique <= 10 and np.issubdtype(target_series.dtype, np.integer):
            return "multiclass"
        elif np.issubdtype(target_series.dtype, np.number):
            return "regression"
        else:
            return "multiclass"
    
    def _classify_domain(self, data: pd.DataFrame) -> Optional[str]:
        """Classify the likely domain of the dataset."""
        
        # Convert column names to lowercase for matching
        column_text = ' '.join(data.columns).lower()
        
        # Score each domain
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in column_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            # Only return if we have reasonable confidence
            if domain_scores[best_domain] >= 2:
                return best_domain
        
        return None

class ModelSelector:
    """Intelligent model selection system."""
    
    def __init__(self):
        self.profiler = DatasetProfiler()
        self.model_factory = ModelFactory()
        
        # Model performance database (could be loaded from historical data)
        self.performance_db = self._initialize_performance_db()
        
        # Model characteristics
        self.model_characteristics = self._initialize_model_characteristics()
    
    def _initialize_performance_db(self) -> Dict[str, Dict[str, float]]:
        """Initialize model performance database with research-backed priors."""
        
        # Based on research and benchmarking results
        return {
            'ctgan': {
                'base_performance': 0.85,
                'mixed_data_bonus': 0.1,
                'large_data_bonus': 0.05,
                'categorical_penalty': -0.05,
                'time_multiplier': 2.0
            },
            'tvae': {
                'base_performance': 0.82,
                'mixed_data_bonus': 0.08,
                'small_data_bonus': 0.03,
                'numerical_bonus': 0.05,
                'time_multiplier': 1.8
            },
            'ganeraid': {
                'base_performance': 0.88,
                'categorical_bonus': 0.1,
                'fast_training_bonus': 0.05,
                'small_data_bonus': 0.05,
                'time_multiplier': 0.5
            }
        }
    
    def _initialize_model_characteristics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model characteristics."""
        
        return {
            'ctgan': {
                'strengths': ['mixed_data', 'large_datasets', 'numerical_data', 'high_quality'],
                'weaknesses': ['training_time', 'memory_usage', 'small_datasets'],
                'best_for': ['financial', 'clinical', 'sensor'],
                'min_samples': 1000,
                'optimal_samples': (5000, 50000),
                'memory_multiplier': 3.0,
                'supports_categorical': True,
                'supports_mixed': True
            },
            'tvae': {
                'strengths': ['numerical_data', 'stable_training', 'robust_to_outliers'],
                'weaknesses': ['categorical_data', 'training_time'],
                'best_for': ['sensor', 'financial', 'scientific'],
                'min_samples': 500,
                'optimal_samples': (2000, 20000),
                'memory_multiplier': 2.5,
                'supports_categorical': True,
                'supports_mixed': True
            },
            'ganeraid': {
                'strengths': ['fast_training', 'categorical_data', 'small_datasets', 'interpretable'],
                'weaknesses': ['complex_distributions', 'very_large_datasets'],
                'best_for': ['clinical', 'hr', 'marketing', 'retail'],
                'min_samples': 100,
                'optimal_samples': (500, 10000),
                'memory_multiplier': 1.0,
                'supports_categorical': True,
                'supports_mixed': True
            }
        }
    
    def recommend_models(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        max_recommendations: int = 3,
        include_config: bool = True
    ) -> List[ModelRecommendation]:
        """Generate model recommendations for the given dataset."""
        
        logger.info(f"Generating model recommendations for dataset: {data.shape}")
        
        # Profile the dataset
        profile = self.profiler.profile_dataset(data, target_column)
        
        # Get available models
        available_models = self.model_factory.list_available_models()
        available_model_names = [name for name, available in available_models.items() if available]
        
        if not available_model_names:
            logger.warning("No models available for recommendation")
            return []
        
        # Generate recommendations for each available model
        recommendations = []
        
        for model_name in available_model_names:
            try:
                recommendation = self._evaluate_model_for_dataset(model_name, profile, data)
                if recommendation:
                    recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Error evaluating {model_name}: {e}")
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Add recommended configurations if requested
        if include_config:
            for rec in recommendations[:max_recommendations]:
                rec.recommended_config = self._generate_config_recommendation(
                    rec.model_name, profile, data
                )
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations[:max_recommendations]
    
    def _evaluate_model_for_dataset(
        self,
        model_name: str,
        profile: DatasetProfile,
        data: pd.DataFrame
    ) -> Optional[ModelRecommendation]:
        """Evaluate how well a model fits the dataset."""
        
        if model_name not in self.model_characteristics:
            logger.warning(f"No characteristics defined for {model_name}")
            return None
        
        model_char = self.model_characteristics[model_name]
        model_perf = self.performance_db.get(model_name, {})
        
        # Base performance score
        base_score = model_perf.get('base_performance', 0.5)
        
        # Calculate bonuses and penalties
        score_adjustments = []
        reasons = []
        warnings = []
        
        # Size-based adjustments
        if profile.n_samples < model_char.get('min_samples', 100):
            score_adjustments.append(-0.2)
            warnings.append(f"Dataset size ({profile.n_samples}) below recommended minimum ({model_char['min_samples']})")
        
        optimal_range = model_char.get('optimal_samples', (1000, 10000))
        if optimal_range[0] <= profile.n_samples <= optimal_range[1]:
            score_adjustments.append(0.1)
            reasons.append(f"Dataset size ({profile.n_samples}) in optimal range")
        
        # Data type bonuses
        if profile.n_categorical > 0 and 'categorical_data' in model_char['strengths']:
            bonus = model_perf.get('categorical_bonus', 0.05)
            score_adjustments.append(bonus)
            reasons.append("Good categorical data support")
        
        if profile.n_numerical / profile.n_features > 0.7 and 'numerical_data' in model_char['strengths']:
            bonus = model_perf.get('numerical_bonus', 0.05)
            score_adjustments.append(bonus)
            reasons.append("Strong numerical data support")
        
        if profile.n_categorical > 0 and profile.n_numerical > 0 and 'mixed_data' in model_char['strengths']:
            bonus = model_perf.get('mixed_data_bonus', 0.08)
            score_adjustments.append(bonus)
            reasons.append("Excellent mixed data type support")
        
        # Size category bonuses
        if profile.size_category == 'small' and 'small_datasets' in model_char['strengths']:
            bonus = model_perf.get('small_data_bonus', 0.05)
            score_adjustments.append(bonus)
            reasons.append("Optimized for small datasets")
        
        if profile.size_category in ['large', 'huge'] and 'large_datasets' in model_char['strengths']:
            bonus = model_perf.get('large_data_bonus', 0.05)
            score_adjustments.append(bonus)
            reasons.append("Handles large datasets well")
        
        # Domain matching
        if profile.domain_hint and profile.domain_hint in model_char.get('best_for', []):
            score_adjustments.append(0.1)
            reasons.append(f"Well-suited for {profile.domain_hint} domain")
        
        # Training time considerations
        if 'fast_training' in model_char['strengths']:
            score_adjustments.append(0.05)
            reasons.append("Fast training time")
        
        # Complexity penalties
        if profile.complexity_score > 0.7:
            if 'complex_distributions' in model_char.get('weaknesses', []):
                score_adjustments.append(-0.1)
                warnings.append("May struggle with complex data distributions")
        
        # Calculate final confidence score
        confidence_score = base_score + sum(score_adjustments)
        confidence_score = np.clip(confidence_score, 0.0, 1.0)
        
        # Estimate performance
        expected_performance = confidence_score * 0.9  # Conservative estimate
        
        # Estimate training time
        base_time = 60  # Base 60 minutes
        time_multiplier = model_perf.get('time_multiplier', 1.0)
        size_factor = np.log10(profile.n_samples) / 3  # Log scale with sample size
        complexity_factor = 1 + profile.complexity_score
        
        estimated_time = base_time * time_multiplier * size_factor * complexity_factor
        
        # Resource requirements
        memory_multiplier = model_char.get('memory_multiplier', 1.0)
        base_memory = max(1, profile.n_samples * profile.n_features / 100000)  # GB
        estimated_memory = base_memory * memory_multiplier
        
        resource_requirements = {
            'memory': f"{estimated_memory:.1f} GB",
            'compute': "Medium" if estimated_time < 60 else "High",
            'storage': f"{profile.n_samples * profile.n_features / 1000:.1f} MB"
        }
        
        return ModelRecommendation(
            model_name=model_name,
            confidence_score=float(confidence_score),
            expected_performance=float(expected_performance),
            reasons=reasons,
            warnings=warnings,
            recommended_config={},  # Will be filled later if requested
            estimated_training_time=float(estimated_time),
            resource_requirements=resource_requirements
        )
    
    def _generate_config_recommendation(
        self,
        model_name: str,
        profile: DatasetProfile,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate recommended hyperparameter configuration."""
        
        try:
            # Get model's hyperparameter space
            model = self.model_factory.create(model_name)
            hyperparam_space = model.get_hyperparameter_space()
            
            if not hyperparam_space:
                return {}
            
            config = {}
            
            # Epochs based on dataset size
            if 'epochs' in hyperparam_space:
                if profile.size_category == 'small':
                    config['epochs'] = 200
                elif profile.size_category == 'medium':
                    config['epochs'] = 150
                else:
                    config['epochs'] = 100
            
            # Batch size based on dataset size
            if 'batch_size' in hyperparam_space:
                if profile.n_samples < 1000:
                    config['batch_size'] = min(64, profile.n_samples // 4)
                elif profile.n_samples < 10000:
                    config['batch_size'] = 256
                else:
                    config['batch_size'] = 512
            
            # Learning rates based on dataset characteristics
            if 'lr' in hyperparam_space:
                if profile.complexity_score > 0.7:
                    config['lr'] = 0.0001  # Lower for complex data
                else:
                    config['lr'] = 0.001
            
            if 'lr_g' in hyperparam_space and 'lr_d' in hyperparam_space:
                if profile.complexity_score > 0.7:
                    config['lr_g'] = 0.0002
                    config['lr_d'] = 0.0002
                else:
                    config['lr_g'] = 0.001
                    config['lr_d'] = 0.001
            
            # Model-specific configurations
            if model_name == 'ctgan':
                # CTGAN specific
                config['generator_dim'] = (128, 128) if profile.n_features < 10 else (256, 256)
                config['discriminator_dim'] = (128, 128) if profile.n_features < 10 else (256, 256)
                
            elif model_name == 'tvae':
                # TVAE specific
                config['compress_dims'] = (64, 32) if profile.n_features < 10 else (128, 64)
                config['decompress_dims'] = (32, 64) if profile.n_features < 10 else (64, 128)
                
            elif model_name == 'ganeraid':
                # GANerAid specific
                config['hidden_feature_space'] = min(200, max(100, profile.n_features * 10))
            
            return config
            
        except Exception as e:
            logger.warning(f"Error generating config for {model_name}: {e}")
            return {}
    
    def generate_recommendation_report(
        self,
        data: pd.DataFrame,
        recommendations: List[ModelRecommendation],
        target_column: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> str:
        """Generate a comprehensive recommendation report."""
        
        profile = self.profiler.profile_dataset(data, target_column)
        
        report_lines = [
            "# Model Recommendation Report",
            "",
            f"**Dataset Shape:** {data.shape}",
            f"**Generated:** {pd.Timestamp.now().isoformat()}",
            "",
            "## Dataset Profile",
            "",
            f"- **Size Category:** {profile.size_category}",
            f"- **Features:** {profile.n_features} total ({profile.n_numerical} numerical, {profile.n_categorical} categorical, {profile.n_binary} binary)",
            f"- **Samples:** {profile.n_samples:,}",
            f"- **Complexity Score:** {profile.complexity_score:.3f}",
            f"- **Missing Data:** {profile.missing_ratio:.1%}",
            f"- **Domain:** {profile.domain_hint or 'Unknown'}",
            ""
        ]
        
        if profile.target_type:
            report_lines.extend([
                f"- **Target Type:** {profile.target_type}",
                f"- **Target Balance:** {profile.target_balance:.3f}" if profile.target_balance else "",
                ""
            ])
        
        report_lines.extend([
            "## Model Recommendations",
            ""
        ])
        
        for i, rec in enumerate(recommendations, 1):
            report_lines.extend([
                f"### {i}. {rec.model_name.upper()}",
                "",
                f"**Confidence:** {rec.confidence_score:.1%}",
                f"**Expected Performance:** {rec.expected_performance:.1%}",
                f"**Estimated Training Time:** {rec.estimated_training_time:.1f} minutes",
                "",
                "**Strengths for this dataset:**"
            ])
            
            for reason in rec.reasons:
                report_lines.append(f"- {reason}")
            
            if rec.warnings:
                report_lines.extend(["", "**Potential Issues:**"])
                for warning in rec.warnings:
                    report_lines.append(f"- ⚠️ {warning}")
            
            report_lines.extend([
                "",
                "**Resource Requirements:**"
            ])
            
            for resource, value in rec.resource_requirements.items():
                report_lines.append(f"- {resource.title()}: {value}")
            
            if rec.recommended_config:
                report_lines.extend([
                    "",
                    "**Recommended Configuration:**",
                    "```json"
                ])
                report_lines.append(json.dumps(rec.recommended_config, indent=2))
                report_lines.append("```")
            
            report_lines.append("")
        
        # Add usage recommendations
        report_lines.extend([
            "## Usage Recommendations",
            "",
            f"1. **Primary Choice:** {recommendations[0].model_name.upper()} - Highest confidence ({recommendations[0].confidence_score:.1%})",
            ""
        ])
        
        if len(recommendations) > 1:
            report_lines.append(f"2. **Alternative:** {recommendations[1].model_name.upper()} - Good fallback option")
            report_lines.append("")
        
        # Add implementation example
        if recommendations:
            best_rec = recommendations[0]
            report_lines.extend([
                "## Implementation Example",
                "",
                "```python",
                "from models.model_factory import ModelFactory",
                "",
                f"# Create {best_rec.model_name} model",
                f"model = ModelFactory.create('{best_rec.model_name}')",
                ""
            ])
            
            if best_rec.recommended_config:
                report_lines.append("# Apply recommended configuration")
                report_lines.append(f"model.set_config({best_rec.recommended_config})")
                report_lines.append("")
            
            report_lines.extend([
                "# Train the model",
                "training_result = model.train(your_data)",
                "",
                "# Generate synthetic data", 
                "synthetic_data = model.generate(1000)",
                "```",
                ""
            ])
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Recommendation report saved to {output_file}")
        
        return report_text

def recommend_best_model(
    data: pd.DataFrame,
    target_column: Optional[str] = None,
    max_recommendations: int = 3,
    save_report: bool = True,
    output_dir: str = "recommendations"
) -> Dict[str, Any]:
    """
    Convenience function to get model recommendations.
    
    Args:
        data: Input dataset
        target_column: Target column name (if any)
        max_recommendations: Maximum number of recommendations
        save_report: Whether to save detailed report
        output_dir: Directory to save report
        
    Returns:
        Dictionary with recommendations and analysis
    """
    
    selector = ModelSelector()
    
    # Generate recommendations
    recommendations = selector.recommend_models(
        data=data,
        target_column=target_column,
        max_recommendations=max_recommendations,
        include_config=True
    )
    
    if not recommendations:
        return {
            'recommendations': [],
            'best_model': None,
            'profile': selector.profiler.profile_dataset(data, target_column).to_dict(),
            'report': "No model recommendations available."
        }
    
    # Generate report
    report = selector.generate_recommendation_report(
        data=data,
        recommendations=recommendations,
        target_column=target_column
    )
    
    # Save report if requested
    report_file = None
    if save_report:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(output_dir) / f"model_recommendations_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
    
    return {
        'recommendations': [rec.to_dict() for rec in recommendations],
        'best_model': recommendations[0].to_dict(),
        'profile': selector.profiler.profile_dataset(data, target_column).to_dict(),
        'report': report,
        'report_file': str(report_file) if report_file else None
    }