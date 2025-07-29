import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

class SyntheticDataEvaluator:
    """
    Comprehensive evaluation framework for synthetic data generation models
    """
    
    def __init__(self, real_data, target_column=None, random_state=42):
        """
        Initialize the evaluator
        
        Parameters:
        -----------
        real_data : pandas.DataFrame
            The original dataset
        target_column : str, optional
            Column name for classification tasks
        random_state : int
            Random seed for reproducibility
        """
        self.real_data = real_data.copy()
        self.target_column = target_column
        self.random_state = random_state
        self.numeric_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in self.numeric_columns:
            self.numeric_columns.remove(target_column)
        
        # Store evaluation results
        self.eda_results = {}
        self.model_results = {}
        
    def perform_eda(self, save_plots=True, figsize=(12, 8)):
        """
        Perform comprehensive exploratory data analysis
        """
        print("=== EXPLORATORY DATA ANALYSIS ===\n")
        
        # Basic info
        print("Dataset Shape:", self.real_data.shape)
        print("\nMissing Values:")
        print(self.real_data.isnull().sum())
        
        # Univariate analysis
        print("\n=== UNIVARIATE ANALYSIS ===")
        univariate_stats = {}
        
        for col in self.numeric_columns:
            data = self.real_data[col].dropna()
            stats_dict = {
                'mean': data.mean(),
                'std': data.std(),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'min': data.min(),
                'max': data.max(),
                'median': data.median(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75)
            }
            univariate_stats[col] = stats_dict
            
        self.eda_results['univariate_stats'] = pd.DataFrame(univariate_stats).T
        print("\nUnivariate Statistics:")
        print(self.eda_results['univariate_stats'].round(4))
        
        # Correlation analysis
        print("\n=== BIVARIATE ANALYSIS ===")
        correlation_matrix = self.real_data[self.numeric_columns].corr()
        self.eda_results['correlation_matrix'] = correlation_matrix
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(4))
        
        # Visualizations
        if save_plots:
            self._create_eda_plots(figsize)
            
        return self.eda_results
    
    def _create_eda_plots(self, figsize):
        """Create EDA visualizations"""
        n_numeric = len(self.numeric_columns)
        
        # Distribution plots
        fig, axes = plt.subplots(nrows=(n_numeric + 2) // 3, ncols=3, figsize=figsize)
        axes = axes.flatten() if n_numeric > 1 else [axes]
        
        for i, col in enumerate(self.numeric_columns):
            if i < len(axes):
                self.real_data[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Remove empty subplots
        for i in range(len(self.numeric_columns), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.eda_results['correlation_matrix'], 
                   annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def calculate_similarity_metrics(self, synthetic_data):
        """
        Calculate comprehensive similarity metrics between real and synthetic data
        """
        similarity_scores = {}
        
        # Univariate metrics
        univariate_scores = {}
        for col in self.numeric_columns:
            real_col = self.real_data[col].dropna()
            synth_col = synthetic_data[col].dropna()
            
            # Earth Mover's Distance (Wasserstein)
            emd = wasserstein_distance(real_col, synth_col)
            
            # Jensen-Shannon Divergence
            # Create histograms with same bins
            min_val = min(real_col.min(), synth_col.min())
            max_val = max(real_col.max(), synth_col.max())
            bins = np.linspace(min_val, max_val, 50)
            
            real_hist, _ = np.histogram(real_col, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_col, bins=bins, density=True)
            
            # Normalize to make them probability distributions
            real_hist = real_hist / real_hist.sum()
            synth_hist = synth_hist / synth_hist.sum()
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            real_hist = real_hist + epsilon
            synth_hist = synth_hist + epsilon
            
            js_div = jensenshannon(real_hist, synth_hist)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = ks_2samp(real_col, synth_col)
            
            univariate_scores[col] = {
                'wasserstein_distance': emd,
                'jensen_shannon_divergence': js_div,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval
            }
        
        similarity_scores['univariate'] = univariate_scores
        
        # Bivariate metrics - correlation matrix comparison
        real_corr = self.real_data[self.numeric_columns].corr()
        synth_corr = synthetic_data[self.numeric_columns].corr()
        
        # Frobenius norm of difference
        corr_diff = np.linalg.norm(real_corr - synth_corr, 'fro')
        
        similarity_scores['bivariate'] = {
            'correlation_frobenius_distance': corr_diff
        }
        
        # Calculate overall similarity score (60% weight)
        # Convert distances to similarity scores (higher is better)
        avg_emd = np.mean([scores['wasserstein_distance'] for scores in univariate_scores.values()])
        avg_js = np.mean([scores['jensen_shannon_divergence'] for scores in univariate_scores.values()])
        
        # Normalize and convert to similarity (1 - normalized_distance)
        emd_similarity = 1 / (1 + avg_emd)  # Higher EMD = lower similarity
        js_similarity = 1 - avg_js  # JS divergence is already 0-1, where 0 is identical
        corr_similarity = 1 / (1 + corr_diff)  # Higher correlation difference = lower similarity
        
        overall_similarity = (emd_similarity + js_similarity + corr_similarity) / 3
        similarity_scores['overall'] = overall_similarity
        
        return similarity_scores
    
    def evaluate_classification_performance(self, synthetic_data):
        """
        Evaluate classification performance using TRTS/TRTR, TSTR/TRTR, TSTS/TRTR ratios
        """
        if not self.target_column:
            print("No target column specified for classification evaluation")
            return {}
        
        # Prepare data
        X_real = self.real_data[self.numeric_columns]
        y_real = self.real_data[self.target_column]
        X_synth = synthetic_data[self.numeric_columns]
        y_synth = synthetic_data[self.target_column]
        
        # Encode labels if necessary
        le = LabelEncoder()
        y_real_encoded = le.fit_transform(y_real)
        y_synth_encoded = le.transform(y_synth)
        
        # Split real data
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real_encoded, test_size=0.3, random_state=self.random_state, stratify=y_real_encoded
        )
        
        # Split synthetic data
        X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
            X_synth, y_synth_encoded, test_size=0.3, random_state=self.random_state, stratify=y_synth_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_real_train_scaled = scaler.fit_transform(X_real_train)
        X_real_test_scaled = scaler.transform(X_real_test)
        X_synth_train_scaled = scaler.transform(X_synth_train)
        X_synth_test_scaled = scaler.transform(X_synth_test)
        
        # Train models and evaluate
        models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        performance_results = {}
        
        for model_name, model in models.items():
            # TRTR - Train Real, Test Real (baseline)
            model.fit(X_real_train_scaled, y_real_train)
            y_pred_trtr = model.predict(X_real_test_scaled)
            y_pred_proba_trtr = model.predict_proba(X_real_test_scaled)[:, 1] if len(np.unique(y_real_encoded)) == 2 else None
            
            trtr_metrics = self._calculate_classification_metrics(y_real_test, y_pred_trtr, y_pred_proba_trtr)
            
            # TRTS - Train Real, Test Synthetic
            y_pred_trts = model.predict(X_synth_test_scaled)
            y_pred_proba_trts = model.predict_proba(X_synth_test_scaled)[:, 1] if len(np.unique(y_real_encoded)) == 2 else None
            
            trts_metrics = self._calculate_classification_metrics(y_synth_test, y_pred_trts, y_pred_proba_trts)
            
            # TSTR - Train Synthetic, Test Real
            model.fit(X_synth_train_scaled, y_synth_train)
            y_pred_tstr = model.predict(X_real_test_scaled)
            y_pred_proba_tstr = model.predict_proba(X_real_test_scaled)[:, 1] if len(np.unique(y_real_encoded)) == 2 else None
            
            tstr_metrics = self._calculate_classification_metrics(y_real_test, y_pred_tstr, y_pred_proba_tstr)
            
            # TSTS - Train Synthetic, Test Synthetic
            y_pred_tsts = model.predict(X_synth_test_scaled)
            y_pred_proba_tsts = model.predict_proba(X_synth_test_scaled)[:, 1] if len(np.unique(y_real_encoded)) == 2 else None
            
            tsts_metrics = self._calculate_classification_metrics(y_synth_test, y_pred_tsts, y_pred_proba_tsts)
            
            # Calculate ratios
            performance_results[model_name] = {
                'TRTR': trtr_metrics,
                'TRTS': trts_metrics,
                'TSTR': tstr_metrics,
                'TSTS': tsts_metrics,
                'TRTS_TRTR_ratio': {k: trts_metrics[k] / trtr_metrics[k] if trtr_metrics[k] != 0 else 0 
                                   for k in trts_metrics.keys()},
                'TSTR_TRTR_ratio': {k: tstr_metrics[k] / trtr_metrics[k] if trtr_metrics[k] != 0 else 0 
                                   for k in tstr_metrics.keys()},
                'TSTS_TRTR_ratio': {k: tsts_metrics[k] / trtr_metrics[k] if trtr_metrics[k] != 0 else 0 
                                   for k in tsts_metrics.keys()}
            }
        
        # Calculate overall accuracy score (40% weight)
        accuracy_scores = []
        for model_results in performance_results.values():
            trts_acc = model_results['TRTS_TRTR_ratio']['accuracy']
            tstr_acc = model_results['TSTR_TRTR_ratio']['accuracy']
            tsts_acc = model_results['TSTS_TRTR_ratio']['accuracy']
            accuracy_scores.append(np.mean([trts_acc, tstr_acc, tsts_acc]))
        
        overall_accuracy = np.mean(accuracy_scores)
        performance_results['overall_accuracy'] = overall_accuracy
        
        return performance_results
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['auc'] = 0
            
        return metrics

class BayesianHyperparameterOptimizer:
    """
    Bayesian optimization for synthetic data generation models
    """
    
    def __init__(self, evaluator, similarity_weight=0.6, accuracy_weight=0.4):
        """
        Initialize optimizer
        
        Parameters:
        -----------
        evaluator : SyntheticDataEvaluator
            The data evaluator instance
        similarity_weight : float
            Weight for similarity metrics in objective function
        accuracy_weight : float
            Weight for accuracy metrics in objective function
        """
        self.evaluator = evaluator
        self.similarity_weight = similarity_weight
        self.accuracy_weight = accuracy_weight
        self.optimization_results = {}
    
    def optimize_model(self, model_class, search_space, n_calls=50, model_name="Model"):
        """
        Optimize hyperparameters for a given model
        
        Parameters:
        -----------
        model_class : class
            The synthetic data generation model class
        search_space : list
            List of skopt dimension objects defining the search space
        n_calls : int
            Number of optimization iterations
        model_name : str
            Name identifier for the model
        """
        print(f"\n=== OPTIMIZING {model_name.upper()} ===")
        
        @use_named_args(search_space)
        def objective(**params):
            try:
                # Initialize and train model with current hyperparameters
                model = model_class(**params, random_state=self.evaluator.random_state)
                
                # This is a placeholder - you'll need to implement the actual training
                # For now, we'll simulate synthetic data generation
                synthetic_data = self._generate_synthetic_data_placeholder(model, params)
                
                # Calculate similarity metrics
                similarity_results = self.evaluator.calculate_similarity_metrics(synthetic_data)
                similarity_score = similarity_results['overall']
                
                # Calculate classification performance if target column exists
                if self.evaluator.target_column:
                    performance_results = self.evaluator.evaluate_classification_performance(synthetic_data)
                    accuracy_score = performance_results['overall_accuracy']
                else:
                    accuracy_score = 0
                
                # Composite objective (we minimize, so negate the score)
                composite_score = (self.similarity_weight * similarity_score + 
                                 self.accuracy_weight * accuracy_score)
                
                return -composite_score  # Negative because we minimize
                
            except Exception as e:
                print(f"Error in objective function: {e}")
                return 1.0  # Return poor score on error
        
        # Perform Bayesian optimization
        result = gp_minimize(objective, search_space, n_calls=n_calls, 
                           random_state=self.evaluator.random_state, verbose=True)
        
        # Store results
        self.optimization_results[model_name] = {
            'best_params': dict(zip([dim.name for dim in search_space], result.x)),
            'best_score': -result.fun,  # Convert back to maximization
            'optimization_result': result
        }
        
        print(f"Best parameters for {model_name}:")
        for param, value in self.optimization_results[model_name]['best_params'].items():
            print(f"  {param}: {value}")
        print(f"Best composite score: {self.optimization_results[model_name]['best_score']:.4f}")
        
        return self.optimization_results[model_name]
    
    def _generate_synthetic_data_placeholder(self, model, params):
        """
        Placeholder for synthetic data generation
        Replace this with actual model training and generation
        """
        # This is a simulation - replace with actual synthetic data generation
        # For demonstration, we'll add some noise to the original data
        noise_level = params.get('noise_level', 0.1)
        synthetic_data = self.evaluator.real_data.copy()
        
        for col in self.evaluator.numeric_columns:
            noise = np.random.normal(0, noise_level * synthetic_data[col].std(), 
                                   size=len(synthetic_data))
            synthetic_data[col] = synthetic_data[col] + noise
            
        return synthetic_data

# Example usage and model templates
class ExampleSyntheticModel:
    """
    Example synthetic data generation model template
    You'll replace this with your actual models (CTGAN, TVAE, etc.)
    """
    
    def __init__(self, noise_level=0.1, n_components=10, learning_rate=0.001, 
                 batch_size=64, epochs=100, random_state=42):
        self.noise_level = noise_level
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
    def fit(self, data):
        """Fit the model to data"""
        # Placeholder implementation
        self.fitted = True
        self.data_stats = data.describe()
        
    def generate(self, n_samples):
        """Generate synthetic samples"""
        # Placeholder implementation
        if not hasattr(self, 'fitted'):
            raise ValueError("Model must be fitted before generating data")
        
        # This is just a placeholder - implement actual generation logic
        synthetic_data = pd.DataFrame()
        for col in self.data_stats.columns:
            mean = self.data_stats.loc['mean', col]
            std = self.data_stats.loc['std', col]
            synthetic_data[col] = np.random.normal(
                mean, std * (1 + self.noise_level), n_samples
            )
            
        return synthetic_data

def create_example_search_spaces():
    """
    Create example search spaces for different model types
    Customize these based on your actual models
    """
    search_spaces = {
        'Model1_CTGAN': [
            Real(0.01, 0.5, name='noise_level'),
            Integer(5, 50, name='n_components'),
            Real(1e-5, 1e-2, name='learning_rate'),
            Integer(32, 256, name='batch_size'),
            Integer(50, 500, name='epochs')
        ],
        'Model2_TVAE': [
            Real(0.01, 0.3, name='noise_level'),
            Integer(10, 100, name='n_components'),
            Real(1e-4, 1e-2, name='learning_rate'),
            Integer(16, 128, name='batch_size'),
            Integer(100, 1000, name='epochs')
        ],
        'Model3_GaussianCopula': [
            Real(0.01, 0.2, name='noise_level'),
            Integer(5, 20, name='n_components')
        ],
        'Model4_BayesianNetwork': [
            Real(0.01, 0.3, name='noise_level'),
            Integer(2, 10, name='n_components')
        ],
        'Model5_VAE': [
            Real(0.01, 0.4, name='noise_level'),
            Integer(8, 64, name='n_components'),
            Real(1e-5, 1e-2, name='learning_rate'),
            Integer(32, 256, name='batch_size'),
            Integer(50, 300, name='epochs')
        ]
    }
    
    return search_spaces

# Main execution example
if __name__ == "__main__":
    # Example usage with synthetic dataset
    np.random.seed(42)
    
    # Create example dataset
    n_samples = 1000
    data = pd.DataFrame({
        'feature1': np.random.normal(10, 2, n_samples),
        'feature2': np.random.exponential(2, n_samples),
        'feature3': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
    
    # Add some correlation
    data['feature4'] = 0.7 * data['feature1'] + 0.3 * np.random.normal(0, 1, n_samples)
    
    print("=== SYNTHETIC DATA GENERATION EVALUATION FRAMEWORK ===")
    print("This is a comprehensive framework for evaluating synthetic data generation models")
    print("Replace the ExampleSyntheticModel with your actual models (CTGAN, TVAE, etc.)")
    print("\nExample dataset created with shape:", data.shape)
    
    # Initialize evaluator
    evaluator = SyntheticDataEvaluator(data, target_column='target')
    
    # Perform EDA
    eda_results = evaluator.perform_eda()
    
    # Initialize optimizer
    optimizer = BayesianHyperparameterOptimizer(evaluator)
    
    # Get search spaces
    search_spaces = create_example_search_spaces()
    
    # Example optimization for one model
    print("\n" + "="*50)
    print("EXAMPLE OPTIMIZATION (using placeholder model)")
    print("Replace ExampleSyntheticModel with your actual model classes")
    print("="*50)
    
    result = optimizer.optimize_model(
        ExampleSyntheticModel, 
        search_spaces['Model1_CTGAN'][:2],  # Use first 2 params for quick demo
        n_calls=10,  # Reduced for demo
        model_name="Example_Model"
    )
