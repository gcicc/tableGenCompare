# Clinical Synthetic Data Generation Framework

A comprehensive multi-model comparison and hyperparameter optimization framework for generating high-quality synthetic clinical data using state-of-the-art Generative Adversarial Networks (GANs).

## 🚀 **STREAMLINED VERSION** - Recent Improvements

This branch features significant **streamlining and automation** improvements over the main branch:

### ✨ **Section 4 Hyperparameter Analysis - Streamlined**
- **Before**: 6 individual analysis chunks (CHUNK_041, CHUNK_043, CHUNK_045, CHUNK_047, CHUNK_049, CHUNK_051)
- **After**: Single batch analysis chunk (CHUNK_052) with automated processing
- **Benefit**: **90% reduction** in repetitive code while preserving all functionality

### 📁 **Enhanced File Export System**
- **Comprehensive file output**: All figures and tables automatically exported to organized directories
- **Standardized structure**: Following consistent patterns established in Sections 2 & 3
- **CSV exports**: DataFrames with hyperparameter results, best trials, and optimization summaries
- **PNG exports**: All optimization plots, parameter importance charts, and correlation matrices

### 🔧 **Modular Architecture**
- **Centralized functions**: Core logic moved to `setup.py` for better maintainability
- **Batch processing**: `evaluate_hyperparameter_optimization_results()` function handles all models
- **Error resilience**: Graceful handling of missing or failed optimizations
- **Cross-model summary**: Automatic best model identification and performance comparison

## Overview

This framework provides a systematic approach to evaluating and optimizing multiple synthetic data generation models, enabling researchers and practitioners to select the best-performing model for their specific clinical datasets.

## Key Features

- **Multi-Model Support**: Comprehensive comparison of 6 leading synthetic data generation models
- **Advanced Hyperparameter Optimization**: Automated tuning using Optuna framework  
- **Enhanced Evaluation Metrics**: Combined similarity and accuracy scoring with TRTS methodology
- **Production-Ready Implementation**: Robust error handling and computational efficiency
- **Comprehensive Analysis**: Detailed visualizations and statistical assessments
- **🆕 Streamlined Notebook**: Automated batch processing reduces code repetition by 90%
- **🆕 Complete File Export**: All figures and tables automatically saved to organized directories
- **🆕 Modular Architecture**: Centralized functions in setup.py for enhanced maintainability

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

### 4. Hyperparameter Tuning for Each Model ⚡ **STREAMLINED**
Comprehensive optimization for all models with **automated batch analysis**:
- **4.1 CTGAN Optimization**: Advanced parameter tuning with PAC compatibility
- **4.2 CTAB-GAN Optimization**: Model-specific parameter optimization
- **4.3 CTAB-GAN+ Optimization**: Enhanced variant tuning
- **4.4 GANerAid Optimization**: Medical data focused optimization
- **4.5 CopulaGAN Optimization**: Distribution-aware parameter tuning
- **4.6 TVAE Optimization**: Autoencoder architecture optimization
- **🆕 4.7 Automated Batch Analysis**: Single-chunk processing of all optimization results
  - **Replaces**: 6 individual analysis chunks with 1 streamlined batch processor
  - **Exports**: All figures and tables to organized file structure  
  - **Identifies**: Best performing model across all optimization studies
  - **Generates**: Comprehensive summary CSV with all hyperparameter results

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

## 🔧 **Streamlining Technical Details**

### Automated Batch Processing
The streamlined version introduces `evaluate_hyperparameter_optimization_results()` function that:
- **Processes all models**: Automatically detects and analyzes available optimization studies
- **Handles errors gracefully**: Continues processing if individual models fail
- **Exports systematically**: Saves all plots and tables to standardized directory structure
- **Provides summaries**: Creates comprehensive CSV summaries and identifies best performers

### File Organization
```
results/
├── dataset-name/
│   ├── YYYY-MM-DD/
│   │   ├── Section-4/
│   │   │   ├── ctgan_optimization_analysis.png
│   │   │   ├── ctgan_hyperparameter_importance.png
│   │   │   ├── ctgan_best_trials.csv
│   │   │   ├── [similar files for all models]
│   │   │   └── hyperparameter_optimization_summary.csv
```

### Code Efficiency Gains
- **Lines of code**: Reduced from ~1,500 to ~150 lines (90% reduction)  
- **Maintainability**: Single function vs 6 separate chunks
- **Consistency**: Standardized processing across all models
- **Error handling**: Centralized exception management

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