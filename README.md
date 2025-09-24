# Clinical Synthetic Data Generation Framework

A comprehensive benchmarking suite for evaluating synthetic tabular data generation methods across multiple healthcare datasets.

## Overview

This project implements and compares the performance of state-of-the-art synthetic data generation methods specifically designed for clinical and healthcare tabular data. The framework provides a standardized pipeline for training, evaluating, and optimizing generative models across four distinct medical datasets.

## Synthetic Data Generation Methods

The framework evaluates the following generative models:

- **CTGAN** (Conditional Tabular GAN) - Standard GAN approach for tabular data
- **CTAB-GAN** (Conditional Tabular GAN with advanced preprocessing) - Enhanced preprocessing pipeline
- **CTAB-GAN+** (Enhanced version with WGAN-GP losses, general transforms, and improved stability) - Advanced stability improvements
- **GANerAid** (Custom implementation) - Purpose-built clinical data generator
- **CopulaGAN** (Copula-based GAN) - Statistical approach using copula functions
- **TVAE** (Variational Autoencoder) - Variational approach for tabular synthesis

## Datasets

The framework is designed to work with four healthcare datasets:

1. **Alzheimer's Disease** - Neurological condition classification
2. **Breast Cancer** - Cancer diagnosis and prognosis data
3. **Liver Disease** - Hepatic condition assessment
4. **Pakistani Liver Patient** - Regional liver disease dataset

Each dataset undergoes standardized preprocessing including missing value imputation using MICE (Multiple Imputation by Chained Equations) and appropriate encoding for categorical variables.

## Notebook Structure

Each dataset notebook follows a standardized 5-section pipeline with harmonized chunk identifiers using the `CHUNK_{Major}_{Minor}_{Patch}_{Seq}` naming scheme:

### Section 1: Setup and Data Loading
- Environment configuration and library imports
- Dataset loading and initial exploration

### Section 2: Data Preprocessing and Analysis
- Comprehensive dataset overview and statistics
- Missing value analysis and MICE imputation
- Categorical encoding and feature preparation
- Data quality assessment

### Section 3: Model Configuration
- Model factory setup and configuration management
- Enhanced objective functions with dynamic target column support
- Hyperparameter space definitions for each generative model

### Section 4: Hyperparameter Optimization
- Optuna-based Bayesian optimization for each model
- Performance evaluation using enhanced objective functions
- Model training with optimized parameters
- Comprehensive logging and progress tracking

### Section 5: Model Evaluation and Comparison
- Synthetic data generation using best parameters
- Statistical fidelity assessment
- Utility preservation analysis
- Cross-model performance comparison
- Visualization and reporting

## Key Features

### Advanced Preprocessing Pipeline
- **MICE Imputation**: Sophisticated missing value handling
- **Dynamic Encoding**: Automatic categorical variable processing
- **Target Column Support**: Flexible objective functions for different prediction tasks

### Hyperparameter Optimization
- **Bayesian Optimization**: Optuna-powered efficient parameter search
- **Multi-Objective Evaluation**: Balancing fidelity, utility, and privacy metrics
- **Timeout Management**: Robust optimization with time constraints

### Comprehensive Evaluation
- **Statistical Fidelity**: Distribution matching and correlation preservation
- **Utility Preservation**: Downstream task performance maintenance
- **Privacy Assessment**: Membership inference and attribute disclosure analysis

### Reproducible Framework
- **Harmonized Chunk IDs**: Consistent code organization across all notebooks
- **Standardized Pipeline**: Identical methodology across different datasets
- **Version Control**: Systematic tracking of model configurations and results

## Usage

1. **Environment Setup**: Install required dependencies and configure Python environment
2. **Dataset Preparation**: Place datasets in the appropriate data directory
3. **Model Training**: Execute notebooks section by section or run complete pipeline
4. **Hyperparameter Optimization**: Customize n_trials parameter based on computational resources
5. **Evaluation**: Review generated synthetic data quality and utility metrics

## Results and Outputs

Each notebook generates:
- **Optimized Model Parameters**: Best hyperparameter configurations for each method
- **Synthetic Datasets**: High-quality generated data maintaining statistical properties
- **Performance Metrics**: Comprehensive evaluation scores and comparisons
- **Visualization**: Distribution plots, correlation matrices, and performance charts
- **Detailed Reports**: Statistical analysis and model comparison summaries

## Future Considerations

Based on current development roadmap:

### Infrastructure Scaling
- **AWS Deployment**: Migration to cloud environment for enhanced computational resources
- **Increased Optimization Trials**: Scale n_trials to 50-100 for more robust hyperparameter search
- **Global Parameter Management**: Centralized n_trials configuration across all models

### Advanced Dataset Support
- **Multi-Level Categorical Endpoints**: Support for complex categorical target variables
- **One-Hot Encoding Verification**: Enhanced validation for categorical variable handling

### Enhanced Missing Data Handling
- **Alternative Imputation Strategies**: Beyond MICE, explore indicator-based missingness encoding
- **Missingness Pattern Analysis**: Deep analysis of missing data mechanisms and their preservation

### Evaluation Enhancement
- **Extended Optimization Assessment**: Comprehensive analysis of hyperparameter optimization impact on downstream performance
- **Production-Scale Validation**: Large-scale trials (n_trials = 100+) for definitive model comparison

## Technical Requirements

- Python 3.8+
- PyTorch/TensorFlow for deep learning models
- Optuna for hyperparameter optimization
- Scikit-learn for preprocessing and evaluation
- Pandas/NumPy for data manipulation
- Matplotlib/Seaborn for visualization

## Contributing

This framework follows a systematic approach to synthetic data generation research, emphasizing reproducibility, comprehensive evaluation, and practical applicability to healthcare data challenges.