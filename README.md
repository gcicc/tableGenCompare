# Clinical Synthetic Data Generation Framework

A comprehensive multi-model comparison and hyperparameter optimization framework for generating high-quality synthetic clinical data using state-of-the-art Generative Adversarial Networks (GANs).

## Overview

This framework provides a systematic approach to evaluating and optimizing multiple synthetic data generation models, enabling researchers and practitioners to select the best-performing model for their specific clinical datasets.

## Key Features

- **Multi-Model Support**: Comprehensive comparison of 6 leading synthetic data generation models
- **Advanced Hyperparameter Optimization**: Automated tuning using Optuna framework
- **Enhanced Evaluation Metrics**: Combined similarity and accuracy scoring with TRTS methodology
- **Production-Ready Implementation**: Robust error handling and computational efficiency
- **Comprehensive Analysis**: Detailed visualizations and statistical assessments

## Supported Models

- **CTGAN**: Conditional Tabular GAN for mixed-type data
- **CTAB-GAN**: Conditional Tabular GAN with advanced conditioning
- **CTAB-GAN+**: Enhanced version with improved performance
- **GANerAid**: Medical data-focused GAN implementation  
- **CopulaGAN**: Copula-based approach for complex distributions
- **TVAE**: Tabular Variational Autoencoder

## Framework Structure

### 1. Setup and Configuration
- Environment initialization
- Library imports and dependencies
- Configuration parameters

### 2. Data Loading and Pre-processing
- **2.1 Data Loading**: Initial data ingestion and preprocessing
- **2.2 Visual Summaries**: Comprehensive dataset exploration and visualization

### 3. Demo All Models with Default Parameters
- **3.1 CTGAN Demo**: Baseline performance assessment with quality evaluation
- **3.2 CTAB-GAN Demo**: Standard CTAB-GAN implementation
- **3.3 CTAB-GAN+ Demo**: Enhanced CTAB-GAN variant
- **3.4 GANerAid Demo**: Medical data specialized model
- **3.5 CopulaGAN Demo**: Copula-based generation approach
- **3.6 TVAE Demo**: Variational autoencoder baseline

### 4. Hyperparameter Tuning for Each Model
Comprehensive optimization for all models with detailed analysis:
- **4.1 CTGAN Optimization**: Advanced parameter tuning with PAC compatibility
- **4.2 CTAB-GAN Optimization**: Model-specific parameter optimization
- **4.3 CTAB-GAN+ Optimization**: Enhanced variant tuning
- **4.4 GANerAid Optimization**: Medical data focused optimization
- **4.5 CopulaGAN Optimization**: Distribution-aware parameter tuning
- **4.6 TVAE Optimization**: Autoencoder architecture optimization
- **4.7 Optimization Summary**: Cross-model performance comparison

### 5. Best Model Analysis
- **5.1 Comprehensive Model Evaluation**: Detailed comparison and selection
- **PCA Analysis**: Principal component analysis for best performing model
- **Final Summary**: Conclusions and recommendations

## Methodology

### Enhanced Objective Function
The framework uses a sophisticated evaluation approach combining:
- **Similarity Score (60%)**: Univariate and bivariate distribution matching using Earth Mover's Distance and correlation analysis
- **Accuracy Score (40%)**: TRTS (Train-on-Real-Test-on-Synthetic) methodology for downstream task performance

### Optimization Framework
- **Optuna Integration**: Advanced hyperparameter optimization with pruning
- **Production-Ready Ranges**: Carefully selected parameter spaces for computational efficiency
- **Cross-Validation**: Robust validation methodology for parameter selection

## Technical Appendices

### Appendix 1: Model Descriptions
Conceptual overview of each synthetic data generation model and their theoretical foundations.

### Appendix 2: Optuna Optimization
Detailed explanation of the hyperparameter optimization methodology with CTGAN example implementation.

### Appendix 3: Enhanced Objective Function
Theoretical foundation and mathematical formulation of the evaluation framework.

### Appendix 4: Hyperparameter Design
Rationale behind parameter space design and validation methodology.

## Getting Started

1. **Data Preparation**: Load your clinical dataset following the preprocessing guidelines
2. **Model Selection**: Run all models with default parameters for initial comparison
3. **Optimization**: Execute hyperparameter optimization for promising models
4. **Evaluation**: Compare results using comprehensive metrics and visualizations
5. **Production**: Deploy the best-performing model for synthetic data generation

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Required libraries: SDV, CTGAN, Optuna, scikit-learn, pandas, numpy, matplotlib, seaborn

## Performance Considerations

- **Computational Efficiency**: Optimized parameter ranges for reasonable execution times
- **Memory Management**: Batch processing and memory-efficient implementations
- **Scalability**: Framework designed for datasets of varying sizes

## Contributing

This framework is designed for extensibility. New models can be integrated by following the established evaluation and optimization patterns.

## License

This project is licensed under the MIT License - see the LICENSE file for details.