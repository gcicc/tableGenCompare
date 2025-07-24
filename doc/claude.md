# Clinical Synthetic Data Generation Framework - Multi-Agent Development Guide

## Overview

This guide outlines a multi-agent approach to building a comprehensive Jupyter notebook for clinical synthetic data generation, using the liver disease dataset as an example. The framework will compare multiple models including CTGAN, TVAE, CopulaGAN, and GANerAid.
Critical: Read what is here first to understand what is intended. 
Create a github branch.  Agents can review what's hear and make changes for improvement so long as agents are working together.  
In first phase, let's use the liver dataset as input to assess initial visualizations, create mock results from models and let's see what output visualizations and tables look like.
In next phase, we will work on running small tests on each of the models.


## Architecture Overview

```
├── notebooks/
│   └── clinical_synth_comparison.ipynb    # Main user-facing notebook
├── src/
│   ├── preprocessing/
│   │   ├── clinical_preprocessor.py       # Data preprocessing pipeline
│   │   └── data_validator.py              # Data validation utilities
│   ├── models/
│   │   ├── base_model.py                  # Abstract base class
│   │   ├── ctgan_model.py                 # CTGAN wrapper
│   │   ├── tvae_model.py                  # TVAE wrapper
│   │   ├── copulagan_model.py             # CopulaGAN wrapper
│   │   └── ganeraid_model.py              # GANerAid wrapper
│   ├── evaluation/
│   │   ├── similarity_metrics.py          # Statistical similarity evaluation
│   │   ├── utility_metrics.py             # TRTS/TSTS/TRTR/TSTR evaluation
│   │   └── privacy_metrics.py             # Privacy evaluation (future)
│   ├── optimization/
│   │   ├── hyperopt_engine.py             # Bayesian optimization engine
│   │   └── param_spaces.py                # Model-specific parameter spaces
│   └── reporting/
│       ├── html_reporter.py               # HTML report generation
│       ├── visualization.py               # Plotting utilities
│       └── comparison_dashboard.py        # Interactive dashboard
└── config/
    ├── model_configs.yaml                 # Model configurations
    └── evaluation_configs.yaml            # Evaluation settings
```

## Agent Responsibilities

### Agent 1: Core Framework Development
**Deliverable**: `src/` directory with all utility functions

### Agent 2: Data Preprocessing & Validation
**Deliverable**: Robust preprocessing pipeline for clinical data

### Agent 3: Model Integration & Optimization
**Deliverable**: Model wrappers and hyperparameter optimization

### Agent 4: Evaluation & Reporting
**Deliverable**: Comprehensive evaluation metrics and reporting

### Agent 5: Notebook Development
**Deliverable**: User-friendly Jupyter notebook with clear guidance

## Detailed Agent Instructions

## Agent 1: Core Framework Development

Create the base infrastructure with the following key components:

### File: `src/models/base_model.py`
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

class BaseSyntheticModel(ABC):
    \"\"\"Abstract base class for synthetic data generation models.\"\"\"
    
    def __init__(self, name: str, random_state: Optional[int] = None):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.training_time = 0.0
        self.generation_time = 0.0
        
    @abstractmethod
    def get_param_space(self) -> Dict[str, Any]:
        \"\"\"Return Optuna parameter space for hyperparameter optimization.\"\"\"
        pass
        
    @abstractmethod
    def create_model(self, hyperparams: Dict[str, Any]) -> Any:
        \"\"\"Create model instance with given hyperparameters.\"\"\"
        pass
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, discrete_columns: list = None) -> None:
        \"\"\"Fit the model to training data.\"\"\"
        pass
        
    @abstractmethod
    def generate(self, n_samples: int) -> pd.DataFrame:
        \"\"\"Generate synthetic samples.\"\"\"
        pass
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        \"\"\"Validate input data format.\"\"\"
        if data.empty:
            raise ValueError(\"Input data is empty\")
        if data.isnull().all().any():
            raise ValueError(\"Input data contains columns with all NaN values\")
        return True
        
    def get_model_info(self) -> Dict[str, Any]:
        \"\"\"Return model information and statistics.\"\"\"
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'generation_time': self.generation_time,
            'random_state': self.random_state
        }
```

### File: `src/models/ganeraid_model.py`
```python
\"\"\"
GANerAid Model Wrapper

Based on the paper: \"GANerAid: Realistic synthetic patient data for clinical trials\"
GANerAid uses LSTM-based GAN architecture specifically designed for clinical data.

Installation: pip install GANerAid
\"\"\"

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import time
import logging

try:
    from GANerAid import GANerAid  # Assuming this is the import pattern
    GANERAID_AVAILABLE = True
except ImportError:
    GANERAID_AVAILABLE = False
    logging.warning(\"GANerAid not available. Install with: pip install GANerAid\")

from .base_model import BaseSyntheticModel

class GANerAidModel(BaseSyntheticModel):
    \"\"\"GANerAid wrapper for the clinical synthetic data framework.\"\"\"
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__(\"GANerAid\", random_state)
        
        if not GANERAID_AVAILABLE:
            raise ImportError(\"GANerAid package not available. Install with: pip install GANerAid\")
    
    def get_param_space(self) -> Dict[str, Any]:
        \"\"\"GANerAid hyperparameter space for Optuna optimization.
        
        Based on the paper, GANerAid uses LSTM layers with specific parameters.
        Adjust these ranges based on actual GANerAid API documentation.
        \"\"\"
        return {
            'epochs': ('int', 100, 1000, 100),
            'batch_size': ('categorical', [32, 64, 128, 256]),
            'learning_rate': ('float', 1e-4, 1e-2, True),
            'lstm_units': ('int', 32, 256, 32),
            'dropout_rate': ('float', 0.1, 0.5, False),
            'noise_dim': ('int', 32, 128, 16),
            # Add more parameters based on actual GANerAid API
        }
    
    def create_model(self, hyperparams: Dict[str, Any]) -> Any:
        \"\"\"Create GANerAid model with hyperparameters.
        
        Note: Adjust parameters based on actual GANerAid API
        \"\"\"
        # Example initialization - adjust based on actual API
        model_params = {
            'epochs': hyperparams.get('epochs', 500),
            'batch_size': hyperparams.get('batch_size', 128),
            'learning_rate': hyperparams.get('learning_rate', 0.001),
            'lstm_units': hyperparams.get('lstm_units', 128),
            'dropout_rate': hyperparams.get('dropout_rate', 0.2),
            'noise_dim': hyperparams.get('noise_dim', 64),
            'random_state': self.random_state
        }
        
        # This is a placeholder - adjust based on actual GANerAid constructor
        return GANerAid(**model_params)
    
    def fit(self, data: pd.DataFrame, discrete_columns: list = None) -> None:
        \"\"\"Fit GANerAid to clinical data.\"\"\"
        self.validate_data(data)
        
        start_time = time.time()
        
        # GANerAid may have specific requirements for discrete columns
        if discrete_columns is None:
            discrete_columns = self._identify_discrete_columns(data)
        
        # Fit the model - adjust based on actual GANerAid API
        self.model.fit(data, discrete_columns=discrete_columns)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logging.info(f\"GANerAid training completed in {self.training_time:.2f} seconds\")
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        \"\"\"Generate synthetic samples using GANerAid.\"\"\"
        if not self.is_fitted:
            raise ValueError(\"Model must be fitted before generating samples\")
        
        start_time = time.time()
        
        # Generate samples - adjust based on actual GANerAid API
        synthetic_data = self.model.sample(n_samples)
        
        self.generation_time = time.time() - start_time
        
        logging.info(f\"Generated {n_samples} samples in {self.generation_time:.2f} seconds\")
        
        return synthetic_data
    
    def _identify_discrete_columns(self, data: pd.DataFrame) -> list:
        \"\"\"Identify discrete/categorical columns in the data.\"\"\"
        discrete_cols = []
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].nunique() <= 10:
                discrete_cols.append(col)
        return discrete_cols

# Placeholder class if GANerAid is not available
class GANerAidModelPlaceholder(BaseSyntheticModel):
    \"\"\"Placeholder for GANerAid when package is not available.\"\"\"
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__(\"GANerAid (Not Available)\", random_state)
    
    def get_param_space(self) -> Dict[str, Any]:
        return {}
    
    def create_model(self, hyperparams: Dict[str, Any]) -> Any:
        raise NotImplementedError(\"GANerAid package not available\")
    
    def fit(self, data: pd.DataFrame, discrete_columns: list = None) -> None:
        raise NotImplementedError(\"GANerAid package not available\")
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        raise NotImplementedError(\"GANerAid package not available\")

# Export the appropriate class
if GANERAID_AVAILABLE:
    GANerAidModelWrapper = GANerAidModel
else:
    GANerAidModelWrapper = GANerAidModelPlaceholder
```

---

## Agent 2: Data Preprocessing & Validation

### File: `src/preprocessing/clinical_preprocessor.py`
```python
\"\"\"
Clinical Data Preprocessing Pipeline

Specialized preprocessing for clinical trial data including:
- Missing value imputation using MICE
- Categorical encoding
- Data validation
- Column type inference
\"\"\"

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import logging

class ClinicalDataPreprocessor:
    \"\"\"Preprocessing pipeline for clinical trial data.\"\"\"
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.imputer = None
        self.label_encoders = {}
        self.column_types = {}
        self.target_column = None
        self.is_fitted = False
        
    def fit_transform(self, data: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        \"\"\"Fit preprocessor and transform data.
        
        Args:
            data: Input clinical data
            target_col: Name of target column (if any)
            
        Returns:
            Preprocessed data ready for synthetic data generation
        \"\"\"
        # USER MODIFICATION POINT: Update target column name for your dataset
        # For liver dataset: target_col = 'result'
        # For your dataset: target_col = 'your_target_column_name'
        
        self.target_column = target_col
        data_clean = self._clean_column_names(data.copy())
        
        # Remove rows with missing target values
        if target_col and target_col in data_clean.columns:
            data_clean = data_clean.dropna(subset=[target_col])
            logging.info(f\"Removed rows with missing target values. Shape: {data_clean.shape}\")
        
        # Identify column types
        self.column_types = self._identify_column_types(data_clean, target_col)
        
        # Encode categorical variables
        data_encoded = self._encode_categorical_variables(data_clean)
        
        # Handle missing values with MICE
        data_imputed = self._impute_missing_values(data_encoded)
        
        self.is_fitted = True
        return data_imputed
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Transform new data using fitted preprocessor.\"\"\"
        if not self.is_fitted:
            raise ValueError(\"Preprocessor must be fitted before transform\")
        
        data_clean = self._clean_column_names(data.copy())
        data_encoded = self._encode_categorical_variables(data_clean, fit=False)
        data_imputed = self._impute_missing_values(data_encoded, fit=False)
        
        return data_imputed
    
    def _clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Clean and standardize column names.\"\"\"
        data.columns = (data.columns
                       .str.strip()
                       .str.replace(r'\\xa0', '', regex=True)
                       .str.replace(r'\\s+', '_', regex=True)
                       .str.lower())
        return data
    
    def _identify_column_types(self, data: pd.DataFrame, target_col: str = None) -> Dict[str, str]:
        \"\"\"Identify continuous and categorical columns.
        
        USER MODIFICATION POINT: Adjust logic for your specific dataset
        \"\"\"
        column_types = {}
        
        for col in data.columns:
            if col == target_col:
                column_types[col] = 'target'
            elif data[col].dtype in ['int64', 'float64']:
                # Consider as categorical if few unique values
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.05 and data[col].nunique() < 20:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'continuous'
            else:
                column_types[col] = 'categorical'
        
        logging.info(f\"Column types identified: {column_types}\")
        return column_types
    
    def _encode_categorical_variables(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        \"\"\"Encode categorical variables.\"\"\"
        data_encoded = data.copy()
        
        categorical_cols = [col for col, col_type in self.column_types.items() 
                          if col_type == 'categorical' and col != self.target_column]
        
        if categorical_cols:
            # One-hot encoding for non-target categorical columns
            data_encoded = pd.get_dummies(data_encoded, columns=categorical_cols, drop_first=True)
        
        # Label encode target column if categorical
        if (self.target_column and self.target_column in data_encoded.columns 
            and data_encoded[self.target_column].dtype == 'object'):
            
            if fit:
                le = LabelEncoder()
                data_encoded[self.target_column] = le.fit_transform(
                    data_encoded[self.target_column].astype(str)
                )
                self.label_encoders[self.target_column] = le
            else:
                le = self.label_encoders[self.target_column]
                data_encoded[self.target_column] = le.transform(
                    data_encoded[self.target_column].astype(str)
                )
        
        return data_encoded
    
    def _impute_missing_values(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        \"\"\"Impute missing values using MICE.\"\"\"
        if data.isnull().sum().sum() == 0:
            return data
        
        # Separate target column for imputation
        target_data = None
        if self.target_column and self.target_column in data.columns:
            target_data = data[self.target_column].copy()
            data_for_imputation = data.drop(columns=[self.target_column])
        else:
            data_for_imputation = data.copy()
        
        if fit:
            self.imputer = IterativeImputer(max_iter=10, random_state=self.random_state)
            imputed_array = self.imputer.fit_transform(data_for_imputation)
        else:
            imputed_array = self.imputer.transform(data_for_imputation)
        
        data_imputed = pd.DataFrame(imputed_array, columns=data_for_imputation.columns)
        
        # Reattach target column
        if target_data is not None:
            data_imputed[self.target_column] = target_data.values
        
        logging.info(f\"Missing value imputation completed. Shape: {data_imputed.shape}\")
        return data_imputed
    
    def get_discrete_columns(self, data: pd.DataFrame) -> List[str]:
        \"\"\"Identify discrete columns for synthetic data models.\"\"\"
        discrete_cols = []
        for col in data.columns:
            if (col in self.column_types and self.column_types[col] == 'categorical') or \\
               col.endswith('_1') or col.endswith('_True'):  # One-hot encoded columns
                discrete_cols.append(col)
        return discrete_cols
    
    def generate_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Generate comprehensive data summary for reporting.\"\"\"
        summary = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().sum(),
            'column_types': self.column_types,
            'discrete_columns': self.get_discrete_columns(data),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        if self.target_column and self.target_column in data.columns:
            summary['target_distribution'] = data[self.target_column].value_counts().to_dict()
        
        return summary
```

---

## Agent 3: Model Integration & Optimization

### File: `src/optimization/hyperopt_engine.py`
```python
\"\"\"
Bayesian Hyperparameter Optimization Engine

Uses Optuna for efficient hyperparameter optimization of synthetic data models.
\"\"\"

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import time
import logging
import pickle
import os

class HyperoptEngine:
    \"\"\"Bayesian optimization engine for synthetic data models.\"\"\"
    
    def __init__(self, output_dir: str = \"results\", random_state: int = 42):
        self.output_dir = output_dir
        self.random_state = random_state
        self.studies = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def optimize_model(self, 
                      model_wrapper,
                      data: pd.DataFrame,
                      target_col: str,
                      discrete_columns: List[str],
                      n_trials: int = 25,
                      similarity_weight: float = 0.6,
                      evaluation_func: Optional[Callable] = None) -> optuna.Study:
        \"\"\"Optimize hyperparameters for a synthetic data model.
        
        Args:
            model_wrapper: Model wrapper instance
            data: Training data
            target_col: Target column name
            discrete_columns: List of discrete column names
            n_trials: Number of optimization trials
            similarity_weight: Weight for similarity vs utility (0-1)
            evaluation_func: Custom evaluation function
            
        Returns:
            Optuna study object with optimization results
        \"\"\"
        
        def objective(trial):
            try:
                # Sample hyperparameters
                hyperparams = self._sample_hyperparameters(trial, model_wrapper.get_param_space())
                
                # Create and train model
                start_time = time.time()
                
                model_wrapper.model = model_wrapper.create_model(hyperparams)
                model_wrapper.fit(data, discrete_columns)
                
                training_time = time.time() - start_time
                
                # Generate synthetic data
                gen_start = time.time()
                synthetic_data = model_wrapper.generate(len(data))
                generation_time = time.time() - gen_start
                
                # Evaluate synthetic data quality
                if evaluation_func:
                    evaluation_results = evaluation_func(data, synthetic_data, target_col)
                else:
                    evaluation_results = self._default_evaluation(data, synthetic_data, target_col)
                
                # Compute combined score
                similarity_score = evaluation_results.get('similarity_score', 0.0)
                utility_score = evaluation_results.get('utility_score', 0.0)
                
                combined_score = (similarity_weight * similarity_score + 
                                (1 - similarity_weight) * utility_score)
                
                # Log results
                trial.set_user_attr('training_time', training_time)
                trial.set_user_attr('generation_time', generation_time)
                trial.set_user_attr('similarity_score', similarity_score)
                trial.set_user_attr('utility_score', utility_score)
                trial.set_user_attr('evaluation_results', evaluation_results)
                
                logging.info(f\"Trial {trial.number} ({model_wrapper.name}): \"\n",
                           f\"Combined Score = {combined_score:.4f}\")
                
                return combined_score
                
            except Exception as e:
                logging.error(f\"Trial {trial.number} ({model_wrapper.name}) failed: {e}\")
                return 0.0
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=HyperbandPruner()
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Save study
        study_path = os.path.join(self.output_dir, f\"{model_wrapper.name}_study.pkl\")
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        self.studies[model_wrapper.name] = study
        
        logging.info(f\"Optimization complete for {model_wrapper.name}. \"\n",
                   f\"Best score: {study.best_value:.4f}\")
        
        return study
    
    def _sample_hyperparameters(self, trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Sample hyperparameters from parameter space.\"\"\"
        hyperparams = {}
        
        for param_name, param_config in param_space.items():
            param_type = param_config[0]
            
            if param_type == 'int':
                hyperparams[param_name] = trial.suggest_int(
                    param_name, param_config[1], param_config[2], 
                    step=param_config[3] if len(param_config) > 3 else 1
                )
            elif param_type == 'float':
                hyperparams[param_name] = trial.suggest_float(
                    param_name, param_config[1], param_config[2], 
                    log=param_config[3] if len(param_config) > 3 else False
                )
            elif param_type == 'categorical':
                hyperparams[param_name] = trial.suggest_categorical(
                    param_name, param_config[1]
                )
        
        return hyperparams
    
    def _default_evaluation(self, original: pd.DataFrame, synthetic: pd.DataFrame, 
                          target_col: str) -> Dict[str, float]:
        \"\"\"Default evaluation function (placeholder).\"\"\"
        # This would be replaced by actual evaluation metrics
        return {
            'similarity_score': np.random.random(),  # Placeholder
            'utility_score': np.random.random(),     # Placeholder
        }
    
    def compare_models(self, study_names: List[str]) -> pd.DataFrame:
        \"\"\"Compare optimization results across models.\"\"\"
        comparison_data = []
        
        for name in study_names:
            if name in self.studies:
                study = self.studies[name]
                best_trial = study.best_trial
                
                comparison_data.append({
                    'model_name': name,
                    'best_score': study.best_value,
                    'n_trials': len(study.trials),
                    'best_params': str(best_trial.params),
                    'training_time': best_trial.user_attrs.get('training_time', 0),
                    'generation_time': best_trial.user_attrs.get('generation_time', 0),
                    'similarity_score': best_trial.user_attrs.get('similarity_score', 0),
                    'utility_score': best_trial.user_attrs.get('utility_score', 0),
                })
        
        return pd.DataFrame(comparison_data)
```

---

## Agent 4: Evaluation & Reporting

### File: `src/evaluation/similarity_metrics.py`
```python
\"\"\"
Statistical Similarity Evaluation Metrics

Comprehensive similarity evaluation including:
- Univariate similarity (Wasserstein, Jensen-Shannon)
- Bivariate similarity (Optimal Transport)
- Correlation preservation
- Distribution matching
\"\"\"

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon
import ot  # Optimal transport
from sklearn.preprocessing import LabelEncoder
import logging

class SimilarityEvaluator:
    \"\"\"Comprehensive similarity evaluation between original and synthetic data.\"\"\"
    
    def __init__(self, max_bivariate_pairs: int = 10):
        self.max_bivariate_pairs = max_bivariate_pairs
    
    def evaluate_similarity(self, 
                          original: pd.DataFrame, 
                          synthetic: pd.DataFrame,
                          target_col: Optional[str] = None,
                          weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        \"\"\"Comprehensive similarity evaluation.
        
        Args:
            original: Original dataset
            synthetic: Synthetic dataset
            target_col: Target column name (excluded from similarity)
            weights: Weights for different similarity components
            
        Returns:
            Dictionary of similarity scores
        \"\"\"
        if weights is None:
            weights = {'univariate': 0.4, 'bivariate': 0.4, 'correlation': 0.2}
        
        # Univariate similarity
        univariate_scores = self._univariate_similarity(original, synthetic, target_col)
        univariate_avg = np.mean(list(univariate_scores.values())) if univariate_scores else 0.0
        
        # Bivariate similarity
        bivariate_scores = self._bivariate_similarity(original, synthetic, target_col)
        bivariate_avg = np.mean(list(bivariate_scores.values())) if bivariate_scores else 0.0
        
        # Correlation similarity
        correlation_score = self._correlation_similarity(original, synthetic, target_col)
        
        # Overall similarity
        overall_similarity = (
            weights['univariate'] * univariate_avg +
            weights['bivariate'] * bivariate_avg +
            weights['correlation'] * correlation_score
        )
        
        return {
            'overall_similarity': overall_similarity,
            'univariate_similarity': univariate_avg,
            'bivariate_similarity': bivariate_avg,
            'correlation_similarity': correlation_score,
            'individual_univariate': univariate_scores,
            'individual_bivariate': bivariate_scores
        }
    
    def _univariate_similarity(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                             target_col: Optional[str] = None) -> Dict[str, float]:
        """Compute univariate similarity for each column."""
        similarity_scores = {}
        
        for column in original.columns:
            if column == target_col:
                continue
                
            try:
                if np.issubdtype(original[column].dtype, np.number):
                    # Wasserstein distance for continuous variables
                    distance = wasserstein_distance(original[column], synthetic[column])
                    similarity_scores[column] = 1 / (1 + distance)
                else:
                    # Jensen-Shannon divergence for categorical variables
                    similarity_scores[column] = self._js_similarity(
                        original[column], synthetic[column]
                    )
                    
            except Exception as e:
                logging.warning(f"Error computing similarity for column {column}: {e}")
                similarity_scores[column] = 0.0
                
        return similarity_scores
    
    def _bivariate_similarity(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                            target_col: Optional[str] = None) -> Dict[str, float]:
        """Compute bivariate similarity using optimal transport."""
        similarity_scores = {}
        continuous_cols = [col for col in original.select_dtypes(include=[np.number]).columns
                          if col != target_col]
        
        if len(continuous_cols) < 2:
            return similarity_scores
        
        # Sample pairs to avoid combinatorial explosion
        pairs = self._select_column_pairs(continuous_cols)
        
        for col_x, col_y in pairs:
            try:
                original_2d = original[[col_x, col_y]].values
                synthetic_2d = synthetic[[col_x, col_y]].values
                
                # Compute optimal transport distance
                cost_matrix = ot.dist(original_2d, synthetic_2d)
                n, m = cost_matrix.shape
                
                p_real = np.ones(n) / n
                p_synthetic = np.ones(m) / m
                
                ot_distance = ot.emd2(p_real, p_synthetic, cost_matrix)
                similarity_scores[f"{col_x}_{col_y}"] = 1 / (1 + ot_distance)
                
            except Exception as e:
                logging.warning(f"Error computing bivariate similarity for {col_x}, {col_y}: {e}")
                similarity_scores[f"{col_x}_{col_y}"] = 0.0
                
        return similarity_scores
    
    def _correlation_similarity(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                              target_col: Optional[str] = None) -> float:
        """Compute correlation matrix similarity."""
        try:
            numeric_cols = [col for col in original.select_dtypes(include=[np.number]).columns
                           if col != target_col]
            
            if len(numeric_cols) < 2:
                return 1.0
            
            orig_corr = original[numeric_cols].corr().values
            synth_corr = synthetic[numeric_cols].corr().values
            
            # Frobenius norm of difference
            diff_norm = np.linalg.norm(orig_corr - synth_corr, 'fro')
            max_norm = np.linalg.norm(orig_corr, 'fro')
            
            return 1 - (diff_norm / max_norm) if max_norm > 0 else 1.0
            
        except Exception as e:
            logging.warning(f"Error computing correlation similarity: {e}")
            return 0.0
    
    def _js_similarity(self, original_col: pd.Series, synthetic_col: pd.Series) -> float:
        """Compute Jensen-Shannon similarity for categorical columns."""
        try:
            categories = np.union1d(original_col.unique(), synthetic_col.unique())
            
            orig_probs = (original_col.value_counts(normalize=True)
                         .reindex(categories, fill_value=0).values)
            synth_probs = (synthetic_col.value_counts(normalize=True)
                          .reindex(categories, fill_value=0).values)
            
            js_divergence = jensenshannon(orig_probs, synth_probs)
            return 1 - js_divergence
            
        except Exception:
            return 0.0
    
    def _select_column_pairs(self, columns: List[str]) -> List[Tuple[str, str]]:
        """Select representative column pairs for bivariate analysis."""
        if len(columns) <= 4:
            return [(columns[i], columns[j]) 
                   for i in range(len(columns)) 
                   for j in range(i+1, len(columns))]
        
        # Sample pairs for larger datasets
        np.random.seed(42)
        pairs = []
        for i in range(min(self.max_bivariate_pairs, len(columns))):
            for j in range(i+1, min(i+3, len(columns))):
                pairs.append((columns[i], columns[j]))
        
        return pairs[:self.max_bivariate_pairs]


class UtilityEvaluator:
    """Evaluate synthetic data utility through classification tasks."""
    
    def __init__(self, classifier=None, test_size: float = 0.2, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = classifier or RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.test_size = test_size
        self.random_state = random_state
    
    def evaluate_utility(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                        target_col: str) -> Dict[str, float]:
        """Comprehensive utility evaluation using TRTS/TSTS/TRTR/TSTR framework."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        results = {}
        
        try:
            # Prepare data splits
            X_real = original.drop(columns=[target_col])
            y_real = original[target_col]
            X_synth = synthetic.drop(columns=[target_col])
            y_synth = synthetic[target_col]
            
            # Split both datasets
            X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
                X_real, y_real, test_size=self.test_size, random_state=self.random_state,
                stratify=y_real if len(np.unique(y_real)) > 1 else None
            )
            
            X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
                X_synth, y_synth, test_size=self.test_size, random_state=self.random_state,
                stratify=y_synth if len(np.unique(y_synth)) > 1 else None
            )
            
            # TRTR: Train Real, Test Real
            results['TRTR'] = self._train_test_score(X_real_train, y_real_train, X_real_test, y_real_test)
            
            # TSTS: Train Synthetic, Test Synthetic
            results['TSTS'] = self._train_test_score(X_synth_train, y_synth_train, X_synth_test, y_synth_test)
            
            # TRTS: Train Real, Test Synthetic
            results['TRTS'] = self._train_test_score(X_real_train, y_real_train, X_synth_test, y_synth_test)
            
            # TSTR: Train Synthetic, Test Real
            results['TSTR'] = self._train_test_score(X_synth_train, y_synth_train, X_real_test, y_real_test)
            
            # Compute average utility
            results['average_utility'] = np.mean([results['TRTR'], results['TSTS'], 
                                                results['TRTS'], results['TSTR']])
            
        except Exception as e:
            logging.error(f"Utility evaluation failed: {e}")
            results = {'TRTR': 0.0, 'TSTS': 0.0, 'TRTS': 0.0, 'TSTR': 0.0, 'average_utility': 0.0}
        
        return results
    
    def _train_test_score(self, X_train, y_train, X_test, y_test) -> float:
        """Train classifier and return test accuracy."""
        try:
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            from sklearn.metrics import accuracy_score
            return accuracy_score(y_test, y_pred)
        except Exception as e:
            logging.warning(f"Classification failed: {e}")
            return 0.0
```

### File: `src/reporting/html_reporter.py`
```python
"""
HTML Report Generation

Creates comprehensive HTML reports with visualizations and comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional
import os
import base64
from io import BytesIO
import logging

class HTMLReporter:
    """Generate comprehensive HTML reports for synthetic data evaluation."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comparison_report(self, 
                                 comparison_results: pd.DataFrame,
                                 original_data: pd.DataFrame,
                                 target_col: str,
                                 synthetic_samples: Optional[Dict[str, pd.DataFrame]] = None) -> str:
        """Generate comprehensive comparison report."""
        
        report_path = os.path.join(self.output_dir, "model_comparison_report.html")
        
        # Generate visualizations
        plots = self._create_comparison_plots(comparison_results, original_data, target_col, synthetic_samples)
        
        # Generate HTML content
        html_content = self._generate_html_template(comparison_results, original_data, target_col, plots)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"HTML report generated: {report_path}")
        return report_path
    
    def _create_comparison_plots(self, 
                               results_df: pd.DataFrame,
                               original_data: pd.DataFrame,
                               target_col: str,
                               synthetic_samples: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, str]:
        """Create comparison plots and return as base64 strings."""
        plots = {}
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Combined Scores
        axes[0, 0].bar(results_df['model_name'], results_df['best_combined_score'])
        axes[0, 0].set_title('Best Combined Scores by Model')
        axes[0, 0].set_ylabel('Combined Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Similarity vs Utility
        axes[0, 1].scatter(results_df['overall_similarity'], results_df['average_utility'], 
                          s=100, alpha=0.7, c='blue')
        for i, model in enumerate(results_df['model_name']):
            axes[0, 1].annotate(model, 
                               (results_df.iloc[i]['overall_similarity'], 
                                results_df.iloc[i]['average_utility']))
        axes[0, 1].set_xlabel('Overall Similarity')
        axes[0, 1].set_ylabel('Average Utility')
        axes[0, 1].set_title('Similarity vs Utility Trade-off')
        
        # TRTS/TSTS/TRTR/TSTR Comparison
        utility_cols = ['TRTR', 'TSTS', 'TRTS', 'TSTR']
        x = np.arange(len(results_df))
        width = 0.2
        
        for i, col in enumerate(utility_cols):
            if col in results_df.columns:
                axes[1, 0].bar(x + i*width, results_df[col], width, label=col)
        
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Utility Metrics Comparison')
        axes[1, 0].set_xticks(x + width * 1.5)
        axes[1, 0].set_xticklabels(results_df['model_name'], rotation=45)
        axes[1, 0].legend()
        
        # Training Time Comparison
        if 'training_time_sec' in results_df.columns:
            axes[1, 1].bar(results_df['model_name'], results_df['training_time_sec'])
            axes[1, 1].set_title('Training Time Comparison')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plots['performance_comparison'] = self._fig_to_base64(fig)
        plt.close()
        
        # 2. Data Distribution Comparison (if synthetic samples available)
        if synthetic_samples:
            plots['distribution_comparison'] = self._create_distribution_plots(
                original_data, synthetic_samples, target_col
            )
        
        return plots
    
    def _create_distribution_plots(self, 
                                 original_data: pd.DataFrame,
                                 synthetic_samples: Dict[str, pd.DataFrame],
                                 target_col: str) -> str:
        """Create distribution comparison plots."""
        
        # Select a few key columns for visualization
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Limit to 4 columns for visualization
        cols_to_plot = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        
        if not cols_to_plot:
            return ""
        
        n_models = len(synthetic_samples)
        fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(12, 4 * len(cols_to_plot)))
        
        if len(cols_to_plot) == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_models + 1))
        
        for i, col in enumerate(cols_to_plot):
            # Plot original data
            axes[i].hist(original_data[col], alpha=0.6, density=True, 
                        label='Original', color=colors[0], bins=30)
            
            # Plot synthetic data for each model
            for j, (model_name, synth_data) in enumerate(synthetic_samples.items()):
                if col in synth_data.columns:
                    axes[i].hist(synth_data[col], alpha=0.6, density=True, 
                               label=f'Synthetic ({model_name})', 
                               color=colors[j+1], bins=30)
            
            axes[i].set_title(f'Distribution Comparison: {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].legend()
        
        plt.tight_layout()
        plot_b64 = self._fig_to_base64(fig)
        plt.close()
        
        return plot_b64
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')
        
        return graphic
    
    def _generate_html_template(self, 
                              results_df: pd.DataFrame,
                              original_data: pd.DataFrame,
                              target_col: str,
                              plots: Dict[str, str]) -> str:
        """Generate HTML template with results and plots."""
        
        # Rank results by combined score
        ranked_df = results_df.sort_values('best_combined_score', ascending=False)
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Clinical Synthetic Data Generation - Model Comparison Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 40px; 
                    background-color: #f8f9fa;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{ 
                    color: #2c3e50; 
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{ 
                    color: #34495e; 
                    margin-top: 30px;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #3498db; 
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{ 
                    background-color: #f8f9fa; 
                }}
                .best {{ 
                    background-color: #d5f4e6 !important; 
                    font-weight: bold;
                }}
                .summary {{ 
                    background-color: #e8f4f8; 
                    padding: 20px; 
                    margin: 20px 0; 
                    border-radius: 8px;
                    border-left: 5px solid #3498db;
                }}
                .plot-container {{ 
                    text-align: center; 
                    margin: 30px 0;
                }}
                .plot-container img {{ 
                    max-width: 100%; 
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .metric-box {{
                    display: inline-block;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    margin: 10px;
                    border-radius: 8px;
                    text-align: center;
                    min-width: 120px;
                }}
                .metric-title {{
                    font-weight: bold;
                    font-size: 0.9em;
                }}
                .metric-value {{
                    font-size: 1.2em;
                    margin-top: 5px;
                }}
                .alert {{
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                    border-left: 5px solid;
                }}
                .alert-info {{
                    background-color: #d1ecf1;
                    border-color: #bee5eb;
                    color: #0c5460;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 Clinical Synthetic Data Generation - Model Comparison Report</h1>
                
                <div class="summary">
                    <h2>📊 Dataset Summary</h2>
                    <div class="metric-box">
                        <div class="metric-title">Dataset Shape</div>
                        <div class="metric-value">{original_data.shape}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">Target Column</div>
                        <div class="metric-value">{target_col}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">Missing Values</div>
                        <div class="metric-value">{original_data.isnull().sum().sum()}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">Memory Usage</div>
                        <div class="metric-value">{original_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB</div>
                    </div>
                </div>
                
                <div class="alert alert-info">
                    <strong>💡 Target Distribution:</strong> {dict(original_data[target_col].value_counts()) if target_col in original_data.columns else 'N/A'}
                </div>
                
                <h2>🏆 Model Performance Ranking</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Combined Score</th>
                        <th>Overall Similarity</th>
                        <th>Average Utility</th>
                        <th>Training Time (s)</th>
                        <th>Status</th>
                    </tr>
        """
        
        # Add ranking table
        for i, (_, row) in enumerate(ranked_df.iterrows()):
            best_class = "best" if i == 0 else ""
            status = "🥇 Best" if i == 0 else f"#{i+1}"
            training_time = row.get('training_time_sec', 0)
            
            html_template += f"""
                    <tr class="{best_class}">
                        <td>{i+1}</td>
                        <td><strong>{row['model_name']}</strong></td>
                        <td>{row.get('best_combined_score', 0):.4f}</td>
                        <td>{row.get('overall_similarity', 0):.4f}</td>
                        <td>{row.get('average_utility', 0):.4f}</td>
                        <td>{training_time:.2f}</td>
                        <td>{status}</td>
                    </tr>
            """
        
        html_template += """
                </table>
                
                <h2>📈 Performance Visualizations</h2>
        """
        
        # Add plots
        if 'performance_comparison' in plots:
            html_template += f"""
                <div class="plot-container">
                    <h3>Model Performance Comparison</h3>
                    <img src="data:image/png;base64,{plots['performance_comparison']}" alt="Performance Comparison">
                </div>
            """
        
        if 'distribution_comparison' in plots:
            html_template += f"""
                <div class="plot-container">
                    <h3>Data Distribution Comparison</h3>
                    <img src="data:image/png;base64,{plots['distribution_comparison']}" alt="Distribution Comparison">
                </div>
            """
        
        # Add detailed results table
        html_template += f"""
                <h2>📋 Detailed Results</h2>
                <div style="overflow-x: auto;">
                    {results_df.to_html(classes='table', table_id='detailed_results', escape=False)}
                </div>
                
                <div class="summary">
                    <h2>🔍 Key Findings</h2>
                    <ul style="line-height: 1.8;">
        """
        
        if not ranked_df.empty:
            best_model = ranked_df.iloc[0]
            best_sim_idx = results_df['overall_similarity'].idxmax() if 'overall_similarity' in results_df.columns else 0
            best_util_idx = results_df['average_utility'].idxmax() if 'average_utility' in results_df.columns else 0
            fastest_idx = results_df['training_time_sec'].idxmin() if 'training_time_sec' in results_df.columns else 0
            
            html_template += f"""
                        <li><strong>🏆 Best Overall Model:</strong> {best_model['model_name']} 
                            (Combined Score: {best_model.get('best_combined_score', 0):.4f})</li>
                        <li><strong>📊 Best Similarity:</strong> {results_df.loc[best_sim_idx, 'model_name']} 
                            ({results_df.loc[best_sim_idx, 'overall_similarity']:.4f})</li>
                        <li><strong>🎯 Best Utility:</strong> {results_df.loc[best_util_idx, 'model_name']} 
                            ({results_df.loc[best_util_idx, 'average_utility']:.4f})</li>
                        <li><strong>⚡ Fastest Training:</strong> {results_df.loc[fastest_idx, 'model_name']} 
                            ({results_df.loc[fastest_idx, 'training_time_sec']:.2f}s)</li>
            """
        
        html_template += """
                    </ul>
                </div>
                
                <div class="alert alert-info">
                    <h3>🔬 Methodology Notes</h3>
                    <p><strong>Evaluation Framework:</strong> Models were evaluated using a combination of statistical similarity metrics and downstream utility performance.</p>
                    <ul>
                        <li><strong>Similarity Metrics:</strong> Univariate (Wasserstein distance, Jensen-Shannon divergence) + Bivariate (Optimal Transport) + Correlation preservation</li>
                        <li><strong>Utility Metrics:</strong> TRTR, TSTS, TRTS, TSTR classification performance</li>
                        <li><strong>Optimization:</strong> Bayesian hyperparameter optimization using Optuna</li>
                    </ul>
                </div>
                
                <div style="text-align: center; margin-top: 40px; color: #7f8c8d;">
                    <p>Generated by Clinical Synthetic Data Generation Framework</p>
                    <p><em>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
```

---

## Agent 5: Notebook Development

### File: `notebooks/clinical_synth_comparison.ipynb`

Create a comprehensive Jupyter notebook with the following structure:

```python
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Clinical Synthetic Data Generation - Model Comparison Framework\n",
    "\n",
    "This notebook provides a comprehensive comparison of synthetic data generation models for clinical trial data.\n",
    "\n",
    "## 🎯 Objectives\n",
    "- Compare multiple synthetic data generation models (CTGAN, TVAE, CopulaGAN, GANerAid)\n",
    "- Evaluate models using statistical similarity and downstream utility metrics\n",
    "- Optimize hyperparameters using Bayesian optimization\n",
    "- Generate comprehensive evaluation reports\n",
    "\n",
    "## 📊 Example Dataset\n",
    "We use the **liver disease dataset** as an example, but this framework can be adapted to any clinical dataset.\n",
    "\n",
    "### ⚠️ USER MODIFICATION POINTS\n",
    "Throughout this notebook, look for **USER MODIFICATION POINT** comments indicating where you need to adapt the code for your specific dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "from pathlib import Path\n",
    "import sys\n",
    "import logging\n",
    "\n",
    "# Suppress warnings for cleaner output\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Set up logging\n",
    "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')\n",
    "\n",
    "# Add src directory to path\n",
    "sys.path.append('../src')\n",
    "\n",
    "# Import our custom modules\n",
    "from preprocessing.clinical_preprocessor import ClinicalDataPreprocessor\n",
    "from models.ctgan_model import CTGANModel\n",
    "from models.tvae_model import TVAEModel\n",
    "from models.copulagan_model import CopulaGANModel\n",
    "from models.ganeraid_model import GANerAidModelWrapper\n",
    "from evaluation.similarity_metrics import SimilarityEvaluator, UtilityEvaluator\n",
    "from optimization.hyperopt_engine import HyperoptEngine\n",
    "from reporting.html_reporter import HTMLReporter\n",
    "\n",
    "# Set random seeds for reproducibility\n",
    "np.random.seed(42)\n",
    "import random\n",
    "random.seed(42)\n",
    "\n",
    "print(\"✅ All imports successful!\")\n",
    "print(\"📁 Framework structure loaded\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📂 1. Data Loading and Initial Exploration\n",
    "\n",
    "### USER MODIFICATION POINT 1: Update file path and target column\n",
    "Modify the following cell to load your specific dataset:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==========================================\n",
    "# USER MODIFICATION POINT 1: Data Loading\n",
    "# ==========================================\n",
    "\n",
    "# For liver disease dataset (example)\n",
    "DATA_FILE = \"../data/liver_train.csv\"  # Update this path for your dataset\n",
    "TARGET_COLUMN = \"result\"                # Update this for your target column\n",
    "DATASET_NAME = \"Liver Disease\"          # Update this for your dataset name\n",
    "\n",
    "# For your dataset, modify as needed:\n",
    "# DATA_FILE = \"../data/your_dataset.csv\"\n",
    "# TARGET_COLUMN = \"your_target_column\"\n",
    "# DATASET_NAME = \"Your Dataset Name\"\n",
    "\n",
    "try:\n",
    "    # Load the dataset\n",
    "    df_original = pd.read_csv(DATA_FILE, encoding='ISO-8859-1')\n",
    "    print(f\"✅ Dataset loaded successfully!\")\n",
    "    print(f\"📊 Dataset: {DATASET_NAME}\")\n",
    "    print(f\"🎯 Target Column: {TARGET_COLUMN}\")\n",
    "    print(f\"📏 Shape: {df_original.shape}\")\n",
    "    \n",
    "except FileNotFoundError:\n",
    "    print(f\"❌ Error: Could not find file {DATA_FILE}\")\n",
    "    print(\"Please update the DATA_FILE path in the cell above\")\n",
    "    raise\n",
    "except Exception as e:\n",
    "    print(f\"❌ Error loading dataset: {e}\")\n",
    "    raise"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Initial Data Exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display basic information about the dataset\n",
    "print(\"=\" * 50)\n",
    "print(\"📋 DATASET OVERVIEW\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "print(f\"Dataset Shape: {df_original.shape}\")\n",
    "print(f\"Memory Usage: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB\")\n",
    "print(f\"Number of Features: {df_original.shape[1]}\")\n",
    "print(f\"Number of Samples: {df_original.shape[0]}\")\n",
    "\n",
    "print(\"\\n📊 Column Information:\")\n",
    "print(f\"{'Column Name':<30} {'Data Type':<15} {'Unique Values':<15} {'Missing Values':<15}\")\n",
    "print(\"-\" * 75)\n",
    "\n",
    "for col in df_original.columns:\n",
    "    dtype = str(df_original[col].dtype)\n",
    "    unique_vals = df_original[col].nunique()\n",
    "    missing_vals = df_original[col].isnull().sum()\n",
    "    print(f\"{col:<30} {dtype:<15} {unique_vals:<15} {missing_vals:<15}\")\n",
    "\n",
    "print(\"\\n🎯 Target Variable Distribution:\")\n",
    "if TARGET_COLUMN in df_original.columns:\n",
    "    target_counts = df_original[TARGET_COLUMN].value_counts()\n",
    "    print(target_counts)\n",
    "    print(f\"\\nTarget Balance Ratio: {target_counts.min() / target_counts.max():.3f}\")\n",
    "else:\n",
    "    print(f\"⚠️ Warning: Target column '{TARGET_COLUMN}' not found in dataset!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Quality Assessment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assess data quality\n",
    "print(\"=\" * 50)\n",
    "print(\"🔍 DATA QUALITY ASSESSMENT\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "# Missing values analysis\n",
    "missing_analysis = pd.DataFrame({\n",
    "    'Missing_Count': df_original.isnull().sum(),\n",
    "    'Missing_Percentage': (df_original.isnull().sum() / len(df_original)) * 100\n",
    "}).sort_values('Missing_Percentage', ascending=False)\n",
    "\n",
    "print(\"📊 Missing Values Analysis:\")\n",
    "print(missing_analysis[missing_analysis.Missing_Count > 0])\n",
    "\n",
    "# Data type analysis\n",
    "print(\"\\n📈 Data Types Distribution:\")\n",
    "dtype_counts = df_original.dtypes.value_counts()\n",
    "print(dtype_counts)\n",
    "\n",
    "# Identify potential issues\n",
    "issues = []\n",
    "if missing_analysis.Missing_Percentage.max() > 50:\n",
    "    issues.append(\"High missing values (>50%) detected\")\n",
    "if df_original.duplicated().sum() > 0:\n",
    "    issues.append(f\"Duplicate rows detected: {df_original.duplicated().sum()}\")\n",
    "if len(df_original) < 1000:\n",
    "    issues.append(\"Small dataset size (<1000 samples)\")\n",
    "\n",
    "if issues:\n",
    "    print(\"\\n⚠️ Potential Issues Detected:\")\n",
    "    for issue in issues:\n",
    "        print(f\"  • {issue}\")\n",
    "else:\n",
    "    print(\"\\n✅ No major data quality issues detected\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualize Data Distributions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create visualizations for data exploration\n",
    "# Identify numeric and categorical columns\n",
    "numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()\n",
    "categorical_cols = df_original.select_dtypes(include=['object']).columns.tolist()\n",
    "\n",
    "if TARGET_COLUMN in numeric_cols:\n",
    "    numeric_cols.remove(TARGET_COLUMN)\n",
    "if TARGET_COLUMN in categorical_cols:\n",
    "    categorical_cols.remove(TARGET_COLUMN)\n",
    "\n",
    "print(f\"📊 Identified {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns\")\n",
    "\n",
    "# Plot numeric variables\n",
    "if numeric_cols:\n",
    "    n_cols = min(4, len(numeric_cols))\n",
    "    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols\n",
    "    \n",
    "    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))\n",
    "    if n_rows * n_cols == 1:\n",
    "        axes = [axes]\n",
    "    else:\n",
    "        axes = axes.flatten()\n",
    "    \n",
    "    for i, col in enumerate(numeric_cols[:n_rows*n_cols]):\n",
    "        df_original[col].hist(bins=30, ax=axes[i], alpha=0.7, edgecolor='black')\n",
    "        axes[i].set_title(f'Distribution of {col}')\n",
    "        axes[i].set_xlabel(col)\n",
    "        axes[i].set_ylabel('Frequency')\n",
    "    \n",
    "    # Remove empty subplots\n",
    "    for j in range(len(numeric_cols), len(axes)):\n",
    "        fig.delaxes(axes[j])\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Plot categorical variables\n",
    "if categorical_cols:\n",
    "    n_cols = min(3, len(categorical_cols))\n",
    "    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols\n",
    "    \n",
    "    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))\n",
    "    if n_rows * n_cols == 1:\n",
    "        axes = [axes]\n",
    "    else:\n",
    "        axes = axes.flatten()\n",
    "    \n",
    "    for i, col in enumerate(categorical_cols[:n_rows*n_cols]):\n",
    "        value_counts = df_original[col].value_counts()\n",
    "        value_counts.plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')\n",
    "        axes[i].set_title(f'Distribution of {col}')\n",
    "        axes[i].set_xlabel(col)\n",
    "        axes[i].set_ylabel('Count')\n",
    "        axes[i].tick_params(axis='x', rotation=45)\n",
    "    \n",
    "    # Remove empty subplots\n",
    "    for j in range(len(categorical_cols), len(axes)):\n",
    "        fig.delaxes(axes[j])\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Correlation heatmap for numeric variables\n",
    "if len(numeric_cols) > 1:\n",
    "    plt.figure(figsize=(10, 8))\n",
    "    correlation_matrix = df_original[numeric_cols].corr()\n",
    "    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, \n",
    "                square=True, linewidths=0.5)\n",
    "    plt.title('Correlation Matrix of Numeric Variables')\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔧 2. Data Preprocessing\n",
    "\n",
    "Clean and prepare the data for synthetic data generation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize the preprocessor\n",
    "preprocessor = ClinicalDataPreprocessor(random_state=42)\n",
    "\n",
    "print(\"🔧 Starting data preprocessing...\")\n",
    "print(f\"Original data shape: {df_original.shape}\")\n",
    "\n",
    "# Preprocess the data\n",
    "df_processed = preprocessor.fit_transform(df_original, target_col=TARGET_COLUMN)\n",
    "\n",
    "print(f\"✅ Preprocessing complete!\")\n",
    "print(f\"Processed data shape: {df_processed.shape}\")\n",
    "print(f\"Missing values after preprocessing: {df_processed.isnull().sum().sum()}\")\n",
    "\n",
    "# Generate preprocessing summary\n",
    "preprocessing_summary = preprocessor.generate_data_summary(df_processed)\n",
    "print(\"\\n📋 Preprocessing Summary:\")\n",
    "for key, value in preprocessing_summary.items():\n",
    "    if key != 'column_types':\n",
    "        print(f\"  {key}: {value}\")\n",
    "\n",
    "# Identify discrete columns for synthetic data models\n",
    "discrete_columns = preprocessor.get_discrete_columns(df_processed)\n",
    "print(f\"\\n🏷️ Discrete columns identified: {len(discrete_columns)}\")\n",
    "print(f\"Discrete columns: {discrete_columns[:5]}{'...' if len(discrete_columns) > 5 else ''}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Optional: Subsample for Faster Experimentation\n",
    "\n",
    "For large datasets or initial testing, you may want to use a subset of the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==========================================\n",
    "# USER MODIFICATION POINT 2: Subsampling\n",
    "# ==========================================\n",
    "\n",
    "# Set to True if you want to use a subset for faster experimentation\n",
    "USE_SUBSET = True  # Change to False to use full dataset\n",
    "SUBSET_SIZE = 5000  # Adjust subset size as needed\n",
    "\n",
    "if USE_SUBSET and len(df_processed) > SUBSET_SIZE:\n",
    "    print(f\"🎯 Using subset of {SUBSET_SIZE} samples for faster experimentation\")\n",
    "    df_working = df_processed.sample(n=SUBSET_SIZE, random_state=42)\n",
    "    print(f\"Working dataset shape: {df_working.shape}\")\n",
    "else:\n",
    "    print(f\"📊 Using full dataset for analysis\")\n",
    "    df_working = df_processed.copy()\n",
    "    print(f\"Working dataset shape: {df_working.shape}\")\n",
    "\n",
    "# Update discrete columns for working dataset\n",
    "discrete_columns_working = [col for col in discrete_columns if col in df_working.columns]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🤖 3. Model Setup and Configuration\n",
    "\n",
    "Initialize synthetic data generation models and configure optimization settings."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==========================================\n",
    "# USER MODIFICATION POINT 3: Model Selection\n",
    "# ==========================================\n",
    "\n",
    "# Choose which models to compare (set to False to skip)\n",
    "MODELS_TO_COMPARE = {\n",
    "    'CTGAN': True,       # Conditional Tabular GAN\n",
    "    'TVAE': True,        # Tabular Variational Autoencoder  \n",
    "    'CopulaGAN': True,   # Copula-based GAN\n",
    "    'GANerAid': True,    # LSTM-based GAN (if available)\n",
    "}\n",
    "\n",
    "# Optimization settings\n",
    "N_TRIALS = 20           # Number of hyperparameter optimization trials per model\n",
    "SIMILARITY_WEIGHT = 0.6  # Weight for similarity vs utility (0-1)\n",
    "\n",
    "print(\"🤖 Initializing synthetic data generation models...\")\n",
    "\n",
    "# Initialize models\n",
    "available_models = {}\n",
    "\n",
    "if MODELS_TO_COMPARE['CTGAN']:\n",
    "    try:\n",
    "        available_models['CTGAN'] = CTGANModel()\n",
    "        print(\"✅ CTGAN model initialized\")\n",
    "    except Exception as e:\n",
    "        print(f\"❌ CTGAN initialization failed: {e}\")\n",
    "\n",
    "if MODELS_TO_COMPARE['TVAE']:\n",
    "    try:\n",
    "        available_models['TVAE'] = TVAEModel()\n",
    "        print(\"✅ TVAE model initialized\")\n",
    "    except Exception as e:\n",
    "        print(f\"❌ TVAE initialization failed: {e}\")\n",
    "\n",
    "if MODELS_TO_COMPARE['CopulaGAN']:\n",
    "    try:\n",
    "        available_models['CopulaGAN'] = CopulaGANModel()\n",
    "        print(\"✅ CopulaGAN model initialized\")\n",
    "    except Exception as e:\n",
    "        print(f\"❌ CopulaGAN initialization failed: {e}\")\n",
    "\n",
    "if MODELS_TO_COMPARE['GANerAid']:\n",
    "    try:\n",
    "        available_models['GANerAid'] = GANerAidModelWrapper()\n",
    "        print(\"✅ GANerAid model initialized\")\n",
    "    except Exception as e:\n",
    "        print(f\"❌ GANerAid initialization failed: {e}\")\n",
    "        print(\"💡 Install GANerAid with: pip install GANerAid\")\n",
    "\n",
    "print(f\"\\n🎯 {len(available_models)} models ready for comparison\")\n",
    "print(f\"Available models: {list(available_models.keys())}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Initialize Evaluation Components"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize evaluation components\n",
    "print(\"🔍 Initializing evaluation components...\")\n",
    "\n",
    "similarity_evaluator = SimilarityEvaluator(max_bivariate_pairs=10)\n",
    "utility_evaluator = UtilityEvaluator(random_state=42)\n",
    "hyperopt_engine = HyperoptEngine(output_dir=\"results\", random_state=42)\n",
    "html_reporter = HTMLReporter(output_dir=\"results\")\n",
    "\n",
    "print(\"✅ Evaluation components initialized\")\n",
    "\n",
    "# Create evaluation function\n",
    "def comprehensive_evaluation(original_data, synthetic_data, target_col):\n",
    "    \"\"\"Comprehensive evaluation combining similarity and utility metrics.\"\"\"\n",
    "    \n",
    "    # Similarity evaluation\n",
    "    similarity_results = similarity_evaluator.evaluate_similarity(\n",
    "        original_data, synthetic_data, target_col\n",
    "    )\n",
    "    \n",
    "    # Utility evaluation\n",
    "    utility_results = utility_evaluator.evaluate_utility(\n",
    "        original_data, synthetic_data, target_col\n",
    "    )\n",
    "    \n",
    "    # Combine results\n",
    "    combined_results = {\n",
    "        'similarity_score': similarity_results['overall_similarity'],\n",
    "        'utility_score': utility_results['average_utility'],\n",
    "        'similarity_details': similarity_results,\n",
    "        'utility_details': utility_results\n",
    "    }\n",
    "    \n",
    "    return combined_results\n",
    "\n",
    "print(\"🎯 Evaluation pipeline ready\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🚀 4. Model Optimization and Comparison\n",
    "\n",
    "Run Bayesian hyperparameter optimization for each model and compare results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==========================================\n",
    "# MAIN OPTIMIZATION LOOP\n",
    "# ==========================================\n",
    "\n",
    "print(\"🚀 Starting model optimization and comparison...\")\n",
    "print(f\"⏱️ This may take some time with {N_TRIALS} trials per model\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "optimization_results = {}\n",
    "best_synthetic_samples = {}\n",
    "\n",
    "for model_name, model_wrapper in available_models.items():\n",
    "    print(f\"\\n🔧 Optimizing {model_name}...\")\n",
    "    print(f\"📊 Dataset shape: {df_working.shape}\")\n",
    "    print(f\"🎯 Target column: {TARGET_COLUMN}\")\n",
    "    print(f\"🏷️ Discrete columns: {len(discrete_columns_working)}\")\n",
    "    \n",
    "    try:\n",
    "        # Run optimization\n",
    "        study = hyperopt_engine.optimize_model(\n",
    "            model_wrapper=model_wrapper,\n",
    "            data=df_working,\n",
    "            target_col=TARGET_COLUMN,\n",
    "            discrete_columns=discrete_columns_working,\n",
    "            n_trials=N_TRIALS,\n",
    "            similarity_weight=SIMILARITY_WEIGHT,\n",
    "            evaluation_func=comprehensive_evaluation\n",
    "        )\n",
    "        \n",
    "        optimization_results[model_name] = study\n",
    "        \n",
    "        # Generate synthetic data with best parameters\n",
    "        print(f\"🏆 Best score for {model_name}: {study.best_value:.4f}\")\n",
    "        print(f\"⚙️ Best parameters: {study.best_params}\")\n",
    "        \n",
    "        # Create model with best parameters and generate samples\n",
    "        best_model = model_wrapper.create_model(study.best_params)\n",
    "        model_wrapper.model = best_model\n",
    "        model_wrapper.fit(df_working, discrete_columns_working)\n",
    "        synthetic_sample = model_wrapper.generate(len(df_working))\n",
    "        best_synthetic_samples[model_name] = synthetic_sample\n",
    "        \n",
    "        print(f\"✅ {model_name} optimization complete\")\n",
    "        \n",
    "    except Exception as e:\n",
    "        print(f\"❌ {model_name} optimization failed: {e}\")\n",
    "        continue\n",
    "\n",
    "print(\"\\n\" + \"=\" * 60)\n",
    "print(f\"🎉 Optimization complete for {len(optimization_results)} models!\")\n",
    "print(f\"📊 Models optimized: {list(optimization_results.keys())}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📊 5. Results Analysis and Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create comprehensive comparison results\n",
    "if optimization_results:\n",
    "    print(\"📈 Generating comparison results...\")\n",
    "    \n",
    "    comparison_data = []\n",
    "    \n",
    "    for model_name, study in optimization_results.items():\n",
    "        best_trial = study.best_trial\n",
    "        evaluation_results = best_trial.user_attrs.get('evaluation_results', {})\n",
    "        \n",
    "        similarity_details = evaluation_results.get('similarity_details', {})\n",
    "        utility_details = evaluation_results.get('utility_details', {})\n",
    "        \n",
    "        row = {\n",
    "            'model_name': model_name,\n",
    "            'best_combined_score': study.best_value,\n",
    "            'overall_similarity': similarity_details.get('overall_similarity', 0),\n",
    "            'univariate_similarity': similarity_details.get('univariate_similarity', 0),\n",
    "            'bivariate_similarity': similarity_details.get('bivariate_similarity', 0),\n",
    "            'correlation_similarity': similarity_details.get('correlation_similarity', 0),\n",
    "            'average_utility': utility_details.get('average_utility', 0),\n",
    "            'TRTR': utility_details.get('TRTR', 0),\n",
    "            'TSTS': utility_details.get('TSTS', 0),\n",
    "            'TRTS': utility_details.get('TRTS', 0),\n",
    "            'TSTR': utility_details.get('TSTR', 0),\n",
    "            'training_time_sec': best_trial.user_attrs.get('training_time', 0),\n",
    "            'generation_time_sec': best_trial.user_attrs.get('generation_time', 0),\n",
    "            'n_trials': len(study.trials)\n",
    "        }\n",
    "        \n",
    "        # Add hyperparameters\n",
    "        for param, value in best_trial.params.items():\n",
    "            row[f'param_{param}'] = value\n",
    "            \n",
    "        comparison_data.append(row)\n",
    "    \n",
    "    comparison_df = pd.DataFrame(comparison_data).round(4)\n",
    "    \n",
    "    # Sort by combined score\n",
    "    comparison_df = comparison_df.sort_values('best_combined_score', ascending=False)\n",
    "    \n",
    "    print(\"✅ Comparison results generated\")\n",
    "    print(f\"📊 Results shape: {comparison_df.shape}\")\n",
    "    \n",
    "    # Display summary\n",
    "    print(\"\\n\" + \"=\" * 60)\n",
    "    print(\"🏆 MODEL RANKING SUMMARY\")\n",
    "    print(\"=\" * 60)\n",
    "    \n",
    "    summary_cols = ['model_name', 'best_combined_score', 'overall_similarity', \n",
    "                   'average_utility', 'training_time_sec']\n",
    "    print(comparison_df[summary_cols].to_string(index=False))\n",
    "    \n",
    "else:\n",
    "    print(\"❌ No optimization results available\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Quick Visualization of Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(comparison_df) > 0:\n",
    "    # Create quick comparison plots\n",
    "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
    "    \n",
    "    # Combined scores\n",
    "    axes[0, 0].bar(comparison_df['model_name'], comparison_df['best_combined_score'], \n",
    "                   color='skyblue', edgecolor='black')\n",
    "    axes[0, 0].set_title('Combined Scores by Model')\n",
    "    axes[0, 0].set_ylabel('Combined Score')\n",
    "    axes[0, 0].tick_params(axis='x', rotation=45)\n",
    "    \n",
    "    # Similarity vs Utility scatter\n",
    "    axes[0, 1].scatter(comparison_df['overall_similarity'], comparison_df['average_utility'], \n",
    "                      s=100, alpha=0.7, c='red')\n",
    "    for i, model in enumerate(comparison_df['model_name']):\n",
    "        axes[0, 1].annotate(model, \n",
    "                           (comparison_df.iloc[i]['overall_similarity'], \n",
    "                            comparison_df.iloc[i]['average_utility']))\n",
    "
