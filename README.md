# Clinical Synthetic Data Generation Framework

A comprehensive framework for generating and evaluating synthetic clinical data using multiple state-of-the-art generative models. This project provides a systematic approach to comparing different synthetic data generation techniques with rigorous evaluation metrics.

## 🚀 Features

### Supported Models
- **CTGAN** - Conditional Generative Adversarial Networks
- **CTAB-GAN** - Conditional Tabular GAN with two-stage training
- **CTAB-GAN+** - Enhanced version with privacy-preserving features
- **GANerAid** - Specialized GAN for clinical data
- **CopulaGAN** - Copula-based generative adversarial networks
- **TVAE** - Tabular Variational Autoencoder

### Key Capabilities
- **Automated Hyperparameter Optimization** using Optuna
- **Comprehensive Evaluation Framework** with multiple metrics
- **TRTS Framework** (Train Real Test Synthetic) for ML utility assessment
- **Modular Architecture** with factory pattern for easy model extensibility
- **Clinical Data Focus** with specialized handling for medical datasets

## 📊 Evaluation Metrics

### Statistical Similarity
- **Earth Mover's Distance (EMD)** for distribution comparison
- **Correlation Structure Preservation** analysis
- **Statistical Moment Matching** (mean, std, skewness, kurtosis)

### Machine Learning Utility
- **TRTS Framework**: Train Real, Test Synthetic
- **Cross-Accuracy Evaluation** between real and synthetic data
- **Classification Performance** preservation

### Comprehensive Quality Assessment
- **Overall Quality Score** (0.6 × similarity + 0.4 × accuracy)
- **PCA-based Analysis** with outcome variable visualization
- **Distribution Similarity** using Jensen-Shannon divergence

## 🛠 Installation

### Prerequisites
```bash
# Core dependencies
pip install pandas numpy scikit-learn
pip install torch torchvision torchaudio
pip install optuna plotly seaborn matplotlib

# Synthetic data libraries
pip install sdv ctgan
pip install rdt

# Additional dependencies
pip install scipy statsmodels
pip install jupyter ipykernel
```

### Model-Specific Setup

#### CTAB-GAN Setup
```bash
# Clone CTAB-GAN repository
git clone https://github.com/Team-TUD/CTAB-GAN.git
cd CTAB-GAN
pip install -e .
```

#### CTAB-GAN+ Setup
```bash
# Clone CTAB-GAN+ repository  
git clone https://github.com/Team-TUD/CTAB-GAN-Plus.git
cd CTAB-GAN-Plus
pip install -e .
```

## 📁 Project Structure

```
├── src/
│   ├── models/
│   │   ├── base_model.py              # Abstract base class
│   │   ├── model_factory.py           # Factory for model creation
│   │   └── implementations/
│   │       ├── ctgan_model.py         # CTGAN implementation
│   │       ├── ctabgan_model.py       # CTAB-GAN implementation
│   │       ├── ctabganplus_model.py   # CTAB-GAN+ implementation
│   │       ├── ganeraid_model.py      # GANerAid implementation
│   │       ├── copulagan_model.py     # CopulaGAN implementation
│   │       └── tvae_model.py          # TVAE implementation
│   └── evaluation/
│       └── trts_framework.py          # TRTS evaluation framework
├── data/
│   ├── alzheimers_disease_data.csv
│   ├── Breast_cancer_data.csv
│   ├── liver_train.csv
│   └── Pakistani_Diabetes_Dataset.csv
├── doc/
│   ├── Model-descriptions.md          # Detailed model descriptions
│   └── Objective-function.md          # Evaluation methodology
├── results/                           # Generated results and plots
├── setup.py                           # Core framework functions
└── Clinical_Synthetic_Data_Generation_Framework_Generalized.ipynb
```

## 🏃 Quick Start

### 1. Basic Usage

```python
from src.models.model_factory import ModelFactory
import pandas as pd

# Load your data
data = pd.read_csv('data/Pakistani_Diabetes_Dataset.csv')

# Create and train a model
model = ModelFactory.create("ctgan", random_state=42)
model.train(data, target_column='Outcome', epochs=100)

# Generate synthetic data
synthetic_data = model.generate(n_samples=1000)
```

### 2. Hyperparameter Optimization

```python
import optuna
from setup import enhanced_objective_function_v2

def objective(trial):
    # Define hyperparameter search space
    params = {
        'epochs': trial.suggest_int('epochs', 100, 1000),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'generator_lr': trial.suggest_loguniform('generator_lr', 1e-5, 1e-3)
    }
    
    # Train model with trial parameters
    model = ModelFactory.create("ctgan")
    model.set_config(params)
    model.train(data, target_column='Outcome')
    
    # Generate and evaluate
    synthetic_data = model.generate(len(data))
    score, similarity, accuracy = enhanced_objective_function_v2(
        data, synthetic_data, 'Outcome'
    )
    
    return score

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 3. Comprehensive Evaluation

```python
from setup import evaluate_synthetic_data_quality

# Evaluate synthetic data quality
results = evaluate_synthetic_data_quality(
    real_data=data,
    synthetic_data=synthetic_data,
    model_name="CTGAN",
    target_column="Outcome",
    section_number=3,
    save_files=True,
    display_plots=False
)

print(f"Overall Quality Score: {results['overall_quality']:.4f}")
print(f"Statistical Similarity: {results['avg_similarity']:.4f}")
print(f"ML Utility: {results['ml_utility']:.4f}")
```

## 📈 Using the Jupyter Notebook

The main notebook `Clinical_Synthetic_Data_Generation_Framework_Generalized.ipynb` provides:

1. **Section 1-2**: Setup and Data Loading
2. **Section 3**: Model Demonstrations and Basic Evaluation
3. **Section 4**: Hyperparameter Optimization for all models
4. **Section 5**: Advanced Analysis and Comparison

### Running the Complete Framework

1. **Configure your dataset**:
```python
# User Configuration
DATASET_NAME = "Pakistani Diabetes Dataset"
DATA_FILE = "data/Pakistani_Diabetes_Dataset.csv"
TARGET_COLUMN = "Outcome"
CATEGORICAL_COLUMNS = ['Gender', 'his', 'vision', 'Exr', 'dipsia', 'uria', 'neph']
MISSING_STRATEGY = "median"  # or "mice", "drop"
```

2. **Execute sections sequentially** for complete analysis

## 🔬 Evaluation Framework

### TRTS (Train Real, Test Synthetic)
The framework implements a comprehensive TRTS evaluation:

- **TRTR**: Train Real → Test Real (baseline)
- **TRTS**: Train Real → Test Synthetic (forward utility)
- **TSTR**: Train Synthetic → Test Real (reverse utility)  
- **TSTS**: Train Synthetic → Test Synthetic (consistency)

### Enhanced Objective Function
Combines multiple evaluation criteria:
```
Objective Score = 0.6 × Similarity Score + 0.4 × Accuracy Score
```

Where:
- **Similarity Score**: EMD + Correlation preservation
- **Accuracy Score**: TRTS/TRTR ratio

## 📊 Available Datasets

The framework includes several clinical datasets:
- **Pakistani Diabetes Dataset** - Diabetes prediction
- **Alzheimer's Disease Data** - Cognitive assessment
- **Breast Cancer Data** - Cancer diagnosis
- **Liver Disease Data** - Hepatic condition prediction

## 🎛 Configuration Options

### Model Parameters
Each model supports extensive hyperparameter customization:
- Training epochs and batch size
- Learning rates (generator/discriminator)
- Network architectures
- Regularization parameters

### Evaluation Settings
- Verbose output control
- File saving preferences  
- Plot generation options
- Result organization by dataset/timestamp

## 📝 Results and Outputs

The framework generates comprehensive results:
- **CSV files** with detailed metrics
- **Visualization plots** (PCA, correlation heatmaps, distributions)
- **Parameter optimization history**
- **Model comparison tables**

Results are organized by dataset and timestamp:
```
results/
└── pakistani-diabetes-dataset/
    └── 2025-09-12/
        ├── Section-3/
        ├── Section-4/
        └── Section-5/
```

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional generative models
- New evaluation metrics
- Dataset-specific optimizations
- Performance improvements

## 📚 References

- **CTGAN**: [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503)
- **CTAB-GAN**: [CTAB-GAN: Effective Table Data Synthesizing](https://arxiv.org/abs/2102.08369)
- **SDV**: [The Synthetic Data Vault](https://sdv.dev/)

## 📄 License

This project is open source. Please see individual model repositories for their specific licensing terms.

## 🔧 Troubleshooting

### Common Issues

1. **CTAB-GAN Import Error**:
   - Ensure CTAB-GAN is properly installed and in Python path
   - Check sklearn version compatibility

2. **Memory Issues**:
   - Reduce batch size or dataset size for initial testing
   - Use GPU if available for large datasets

3. **Hyperparameter Optimization Timeout**:
   - Reduce number of trials or epochs for faster testing
   - Use pruning to eliminate poor trials early

### Getting Help

- Check the documentation in `doc/`
- Review the example notebook for usage patterns
- Ensure all dependencies are properly installed

---

**Note**: This framework is designed for research and educational purposes. Ensure proper validation before using synthetic data in production scenarios.