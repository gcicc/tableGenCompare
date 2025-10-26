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

## Installation

### Prerequisites

- **Python 3.11+** (recommended)
- **Git** for cloning the repository
- **10+ GB disk space** for models and dependencies
- **8+ GB RAM** (16+ GB recommended for larger datasets)

### Step 1: Clone the Repository

```bash
git clone https://github.com/gcicc/tableGenCompare.git
cd tableGenCompare
```

### Step 2: Create Virtual Environment (Recommended)

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Installation may take 10-15 minutes due to large packages (PyTorch ~200MB, etc.)

### Step 4: Verify Installation

Test that critical imports work:

```python
python -c "import pandas, numpy, torch, sklearn, optuna, sdv; print('✓ All core packages installed successfully')"
```

### Quick Start

1. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open a dataset notebook:**
   - `SynthethicTableGenerator-Alzheimer.ipynb`
   - `SynthethicTableGenerator-BreastCancer.ipynb`
   - `SynthethicTableGenerator-Liver.ipynb`
   - `SynthethicTableGenerator-Pakistani.ipynb`

3. **Run sections sequentially:**
   - Section 1: Setup and Data Loading
   - Section 2: Data Preprocessing
   - Section 3: Model Configuration
   - Section 4: Hyperparameter Optimization
   - Section 5: Model Evaluation

### AWS Deployment

For cloud deployment on AWS:

```bash
# On your AWS instance (Amazon Linux 2 / Ubuntu):
sudo yum install python3.11 git -y  # Amazon Linux
# OR
sudo apt-get install python3.11 git -y  # Ubuntu

# Clone and install
git clone https://github.com/gcicc/tableGenCompare.git
cd tableGenCompare
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**AWS SageMaker Notebook:**
```python
!git clone https://github.com/gcicc/tableGenCompare.git
%cd tableGenCompare
!pip install -r requirements.txt
```

### Troubleshooting

**Issue: PyTorch installation fails**
- Solution: Ensure you have sufficient disk space (10+ GB)
- Alternative: Install CPU-only PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

**Issue: CTAB-GAN import errors**
- Solution: Verify Git repositories were cloned during installation
- Check: `ls CTAB-GAN CTAB-GAN-Plus` should show directories

**Issue: Out of memory during training**
- Solution: Reduce `n_trials` parameter in Section 4 hyperparameter optimization
- Recommended: Start with `n_trials=5` for testing

### Package Versions

The framework uses flexible version constraints (`>=`) to ensure compatibility while allowing updates. See `requirements.txt` for complete dependency list.

**Key Dependencies:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- torch >= 2.0.0
- scikit-learn >= 1.3.0
- optuna >= 3.0.0
- sdv >= 1.2.0 (includes CTGAN, CopulaGAN, TVAE)

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

**System Requirements:**
- Python 3.11+ (recommended for optimal compatibility)
- 10+ GB disk space for dependencies and models
- 8+ GB RAM (16+ GB recommended for larger datasets)
- GPU optional but recommended for faster training

**Software Dependencies:**
All dependencies are specified in `requirements.txt` and installed automatically via pip. See the [Installation](#installation) section for detailed setup instructions.

**Key Libraries:**
- PyTorch for deep learning models
- SDV (Synthetic Data Vault) for CTGAN, CopulaGAN, TVAE
- CTAB-GAN and CTAB-GAN+ (installed from GitHub)
- Optuna for hyperparameter optimization
- Scikit-learn for preprocessing and evaluation
- Pandas/NumPy for data manipulation
- Matplotlib/Seaborn for visualization

## Contributing

This framework follows a systematic approach to synthetic data generation research, emphasizing reproducibility, comprehensive evaluation, and practical applicability to healthcare data challenges.