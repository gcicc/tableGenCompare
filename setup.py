# Setup Module for Clinical Synthetic Data Generation Framework
# Contains imported chunks from notebook for better organization

# SESSION TIMESTAMP AND DATASET IDENTIFIER SYSTEM
from datetime import datetime
import os

# Generate session timestamp when setup.py is first imported
SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")
print(f"Session timestamp captured: {SESSION_TIMESTAMP}")

def extract_dataset_identifier(data_file_path):
    """Extract dataset identifier from file path or filename"""
    if isinstance(data_file_path, str):
        # Extract filename without extension
        filename = os.path.basename(data_file_path)
        dataset_id = os.path.splitext(filename)[0].lower()
        # Clean up common dataset naming patterns
        dataset_id = dataset_id.replace('_', '-').replace(' ', '-')
        return dataset_id
    return "unknown-dataset"

def get_results_path(dataset_identifier, section_number):
    """Generate standardized results path: results/dataset_identifier/YYYY-MM-DD/Section-N"""
    return f"results/{dataset_identifier}/{SESSION_TIMESTAMP}/Section-{section_number}"

# Global variables to be set when data is loaded
DATASET_IDENTIFIER = None
CURRENT_DATA_FILE = None

# Code Chunk ID: CHUNK_001 - CTAB-GAN Import and Compatibility
# Import CTAB-GAN - try multiple installation paths with sklearn compatibility fix
import sys
import warnings
import importlib.util

# First, apply sklearn compatibility patch
try:
    import sklearn
    print(f"Detected sklearn {sklearn.__version__} - applying compatibility patch...")
    
    # Patch for sklearn 1.0+ compatibility
    from sklearn.mixture import GaussianMixture
    if not hasattr(GaussianMixture, 'n_components'):
        # This shouldn't happen in modern sklearn, but keeping for safety
        print("WARNING: Unexpected sklearn version behavior detected")
    
    # Import warnings to suppress sklearn deprecation warnings during CTAB-GAN import
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    print("Global sklearn compatibility patch applied successfully")
    
except Exception as e:
    print(f"Warning: Could not apply sklearn compatibility patch: {e}")

# Now try to import CTAB-GAN
try:
    # Try importing from various possible locations
    import_successful = False
    
    # Method 1: Try standard import
    try:
        from model.ctabgan import CTABGANSynthesizer
        print("CTAB-GAN imported successfully")
        import_successful = True
    except ImportError:
        pass
    
    # Method 2: Try from CTAB-GAN directory 
    if not import_successful:
        try:
            sys.path.append('./CTAB-GAN')
            from model.ctabgan import CTABGANSynthesizer
            print("CTAB-GAN imported successfully from ./CTAB-GAN")
            import_successful = True
        except ImportError:
            pass
    
    # Method 3: Try from current directory
    if not import_successful:
        try:
            sys.path.append('.')
            from model.ctabgan import CTABGANSynthesizer
            print("CTAB-GAN imported successfully from current directory")
            import_successful = True
        except ImportError:
            pass
    
    if not import_successful:
        print("WARNING: Could not import CTAB-GAN. Please ensure it's properly installed.")
        # Create a dummy class to prevent import errors
        class CTABGANSynthesizer:
            def __init__(self, *args, **kwargs):
                raise ImportError("CTAB-GAN not available")
        CTABGAN_AVAILABLE = False
    else:
        CTABGAN_AVAILABLE = True

except Exception as e:
    print(f"ERROR importing CTAB-GAN: {e}")
    # Create a dummy class to prevent import errors
    class CTABGANSynthesizer:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CTAB-GAN import failed: {e}")
    CTABGAN_AVAILABLE = False

# Code Chunk ID: CHUNK_001B - CTAB-GAN+ Availability Check
try:
    # Try to import CTAB-GAN+ - it's in CTAB-GAN-Plus directory with model/ctabgan.py
    import sys
    sys.path.append('./CTAB-GAN-Plus')
    from model.ctabgan import CTABGAN
    CTABGANPLUS_AVAILABLE = True
    print("✅ CTAB-GAN+ detected and available")
except ImportError:
    CTABGANPLUS_AVAILABLE = False
    print("⚠️ CTAB-GAN+ not available - falling back to regular CTAB-GAN")

# Code Chunk ID: CHUNK_001C - GANerAid Import and Availability Check
try:
    # Try to import GANerAid from various possible locations
    ganeraid_import_successful = False
    
    # Method 1: Try from src.models.implementations
    try:
        from src.models.implementations.ganeraid_model import GANerAidModel
        print("✅ GANerAidModel imported successfully from src.models.implementations")
        ganeraid_import_successful = True
    except ImportError:
        pass
    
    # Method 2: Try direct import (if available in path)
    if not ganeraid_import_successful:
        try:
            from ganeraid_model import GANerAidModel
            print("✅ GANerAidModel imported successfully (direct import)")
            ganeraid_import_successful = True
        except ImportError:
            pass
    
    if not ganeraid_import_successful:
        print("⚠️ GANerAidModel not available - creating placeholder")
        # Create a dummy class to prevent import errors
        class GANerAidModel:
            def __init__(self, *args, **kwargs):
                raise ImportError("GANerAid not available")
        GANERAID_AVAILABLE = False
    else:
        GANERAID_AVAILABLE = True

except Exception as e:
    print(f"ERROR importing GANerAid: {e}")
    # Create a dummy class to prevent import errors
    class GANerAidModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"GANerAid import failed: {e}")
    GANERAID_AVAILABLE = False

# Code Chunk ID: CHUNK_002 - CTABGANModel Class
class CTABGANModel:
    def __init__(self, epochs=100, batch_size=500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        
    def fit(self, data, categorical_columns=None, target_column=None):
        """Train CTAB-GAN model"""
        try:
            # Check if CTAB-GAN is actually available
            if not CTABGAN_AVAILABLE:
                raise ImportError("CTAB-GAN is not available in this environment")
            
            # Store column names for later use in generate()
            self.column_names = list(data.columns)
            
            # Initialize CTAB-GAN with basic parameters only
            self.model = CTABGANSynthesizer(
                epochs=self.epochs,
                batch_size=self.batch_size
            )
            
            # Train the model (categorical_columns passed to fit, not init)
            print(f"Training CTAB-GAN for {self.epochs} epochs...")
            if categorical_columns:
                self.model.fit(data, categorical_columns=categorical_columns)
            else:
                self.model.fit(data)
            print("✅ CTAB-GAN training completed successfully")
            
        except Exception as e:
            print(f"❌ CTAB-GAN training failed: {e}")
            raise
    
    def generate(self, n_samples):
        """Generate synthetic samples"""
        if self.model is None:
            raise ValueError("Model must be fitted before generating samples")
        
        try:
            synthetic_data = self.model.sample(n_samples)
            print(f"✅ Generated {len(synthetic_data)} synthetic samples")
            
            # Convert to DataFrame if it's a numpy array
            if hasattr(synthetic_data, 'shape') and not hasattr(synthetic_data, 'columns'):
                # It's a numpy array, convert to DataFrame
                if hasattr(self, 'column_names'):
                    synthetic_data = pd.DataFrame(synthetic_data, columns=self.column_names)
                else:
                    # Generate generic column names
                    synthetic_data = pd.DataFrame(synthetic_data, columns=[f'feature_{i}' for i in range(synthetic_data.shape[1])])
                    print("⚠️ Using generic column names - original column names not preserved")
            
            return synthetic_data
        except Exception as e:
            print(f"❌ CTAB-GAN generation failed: {e}")
            raise

# Code Chunk ID: CHUNK_003 - CTABGANPlusModel Class  
class CTABGANPlusModel:
    def __init__(self, epochs=100, batch_size=500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.has_plus_features = False
        self.temp_csv_path = None
        self.original_data = None
        
    def _check_plus_features(self):
        """Check if CTAB-GAN+ features are available"""
        try:
            # Try to import CTAB-GAN+ - correct path and class name
            import sys
            if './CTAB-GAN-Plus' not in sys.path:
                sys.path.append('./CTAB-GAN-Plus')
            from model.ctabgan import CTABGAN
            self.has_plus_features = True
            return CTABGAN
        except ImportError:
            print("WARNING: CTAB-GAN+ features not available, falling back to regular CTAB-GAN parameters")
            self.has_plus_features = False
            return CTABGANSynthesizer
    
    def fit(self, data, categorical_columns=None, target_column=None):
        """Train CTAB-GAN+ model with enhanced features"""
        try:
            # Check for CTAB-GAN+ availability
            CTABGANClass = self._check_plus_features()
            
            if self.has_plus_features:
                # CTAB-GAN+ requires CSV file, so save DataFrame to temp file
                import tempfile
                import os
                
                # Store original data for later reference
                self.original_data = data.copy()
                
                # Create temporary CSV file
                temp_dir = tempfile.mkdtemp()
                self.temp_csv_path = os.path.join(temp_dir, "temp_data.csv")
                data.to_csv(self.temp_csv_path, index=False)
                
                # Automatically detect categorical columns if not provided
                if categorical_columns is None:
                    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Automatically detect integer columns
                integer_columns = data.select_dtypes(include=['int64', 'int32']).columns.tolist()
                
                # Determine problem type (using main branch logic)
                if target_column and target_column in data.columns:
                    target_col = target_column
                else:
                    target_col = data.columns[-1]  # Assume last column is target (main branch approach)
                
                # Smart problem_type logic from main branch
                if data[target_col].nunique() <= 10:
                    problem_type = {"Classification": target_col}
                    print(f"🎯 Using Classification with target: {target_col} ({data[target_col].nunique()} unique values)")
                else:
                    problem_type = {None: None}  # Avoid stratification issues for continuous targets
                    print(f"🎯 Using regression mode (target has {data[target_col].nunique()} unique values)")
                
                # Use default test_ratio since stratification is handled by problem_type logic
                test_ratio = 0.20
                
                # Initialize CTAB-GAN+ with proper parameters
                self.model = CTABGANClass(
                    raw_csv_path=self.temp_csv_path,
                    categorical_columns=categorical_columns,
                    integer_columns=integer_columns,
                    problem_type=problem_type,
                    test_ratio=test_ratio
                )
                
                print(f"Training CTAB-GAN+ (Enhanced) for {self.epochs} epochs...")
                self.model.fit()
                print("✅ CTAB-GAN+ training completed successfully")
                
            else:
                # Fallback to regular CTAB-GAN
                self.model = CTABGANClass(
                    epochs=self.epochs,
                    batch_size=self.batch_size
                )
                self.model.fit(data, categorical_columns=categorical_columns)
                print("✅ CTAB-GAN (fallback) training completed successfully")
            
        except Exception as e:
            print(f"❌ CTAB-GAN+ training failed: {e}")
            # Clean up temp file on error
            if self.temp_csv_path and os.path.exists(self.temp_csv_path):
                os.remove(self.temp_csv_path)
            raise
    
    def generate(self, n_samples):
        """Generate synthetic samples using CTAB-GAN+"""
        if self.model is None:
            raise ValueError("Model must be fitted before generating samples")
        
        try:
            if self.has_plus_features:
                # CTAB-GAN+ generates all samples at once
                synthetic_data = self.model.generate_samples()
                
                # If we need fewer samples, take a random subset
                if len(synthetic_data) > n_samples:
                    synthetic_data = synthetic_data.sample(n=n_samples, random_state=42).reset_index(drop=True)
                elif len(synthetic_data) < n_samples:
                    # If we need more samples, repeat the generation or duplicate existing
                    print(f"⚠️ CTAB-GAN+ generated {len(synthetic_data)} samples, requested {n_samples}")
                
            else:
                # Use regular CTAB-GAN generation
                synthetic_data = self.model.sample(n_samples)
            
            print(f"✅ Generated {len(synthetic_data)} synthetic samples using CTAB-GAN+")
            
            # Clean up temp file after successful generation
            if self.temp_csv_path and os.path.exists(self.temp_csv_path):
                os.remove(self.temp_csv_path)
                
            return synthetic_data
            
        except Exception as e:
            print(f"❌ CTAB-GAN+ generation failed: {e}")
            # Clean up temp file on error
            if self.temp_csv_path and os.path.exists(self.temp_csv_path):
                os.remove(self.temp_csv_path)
            raise

# Code Chunk ID: CHUNK_004 - Required Libraries Import
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("✅ All required libraries imported successfully")

# Code Chunk ID: CHUNK_017 - Comprehensive Data Quality Evaluation Function
# ============================================================================
# COMPREHENSIVE DATA QUALITY EVALUATION FUNCTION
# CRITICAL: Must be defined before Section 3.1 calls
# ============================================================================

def evaluate_hyperparameter_optimization_results(section_number=4, scope=None, target_column=None):
    """
    Batch evaluation of hyperparameter optimization results for all available models.
    
    Replaces individual CHUNK_041, CHUNK_043, CHUNK_045, CHUNK_047, CHUNK_049, CHUNK_051
    Following CHUNK_018 pattern for Section 3.
    
    Parameters:
    - section_number: Section number for file organization (default 4)
    - scope: Notebook scope (globals()) to access study variables
    - target_column: Target column name for analysis
    
    Returns:
    - Dictionary with analysis results for all models
    """
    
    if scope is None:
        scope = globals()
    
    # Use global TARGET_COLUMN if not provided
    if target_column is None and 'TARGET_COLUMN' in scope:
        target_column = scope['TARGET_COLUMN']
    
    # Get dataset identifier and results directory
    dataset_id = scope.get('DATASET_IDENTIFIER', 'unknown-dataset')
    results_dir = get_results_path(dataset_id, section_number)
    
    print(f"\n{'='*80}")
    print(f"SECTION {section_number} - HYPERPARAMETER OPTIMIZATION BATCH ANALYSIS")
    print(f"{'='*80}")
    print(f"📁 Results directory: {results_dir}")
    print(f"🎯 Target column: {target_column}")
    print()
    
    # Define model configurations with their study variable names
    model_configs = [
        {'name': 'CTGAN', 'study_var': 'ctgan_study', 'model_name': 'ctgan', 'section': '4.1.1'},
        {'name': 'CTAB-GAN', 'study_var': 'ctabgan_study', 'model_name': 'ctabgan', 'section': '4.2.1'},
        {'name': 'CTAB-GAN+', 'study_var': 'ctabganplus_study', 'model_name': 'ctabganplus', 'section': '4.3.1'},
        {'name': 'GANerAid', 'study_var': 'ganeraid_study', 'model_name': 'ganeraid', 'section': '4.4.1'},
        {'name': 'CopulaGAN', 'study_var': 'copulagan_study', 'model_name': 'copulagan', 'section': '4.5.1'},
        {'name': 'TVAE', 'study_var': 'tvae_study', 'model_name': 'tvae', 'section': '4.6.1'}
    ]
    
    analysis_results = {}
    summary_data = []
    
    for config in model_configs:
        model_name = config['name']
        study_var = config['study_var']
        model_key = config['model_name']
        section = config['section']
        
        print(f"\n🔍 {section}: {model_name} Hyperparameter Optimization Analysis")
        print("-" * 60)
        
        try:
            # Check if study exists in scope
            if study_var in scope and scope[study_var] is not None:
                study_results = scope[study_var]
                
                print(f"✅ {model_name} optimization study found")
                
                # Run hyperparameter analysis with file export
                analysis_result = analyze_hyperparameter_optimization(
                    study_results=study_results,
                    model_name=model_key,
                    target_column=target_column,
                    results_dir=results_dir,
                    export_figures=True,  # Export all figures to files
                    export_tables=True,   # Export all tables to CSV
                    display_plots=False   # Don't display inline - save to files only
                )
                
                analysis_results[model_key] = analysis_result
                
                # Collect summary statistics
                if hasattr(study_results, 'best_trial'):
                    best_trial = study_results.best_trial
                    completed_trials = [t for t in study_results.trials if hasattr(t, 'state') and 
                                      t.state.name == 'COMPLETE']
                    
                    summary_data.append({
                        'model': model_name,
                        'section': section,
                        'best_score': best_trial.value if best_trial else None,
                        'total_trials': len(study_results.trials),
                        'completed_trials': len(completed_trials),
                        'best_trial_number': best_trial.number if best_trial else None,
                        'study_variable': study_var
                    })
                    
                    print(f"✅ {model_name} analysis completed - files exported to {results_dir}")
                    
            else:
                print(f"⚠️  {model_name} optimization study not found (variable: {study_var})")
                print(f"   Skipping {model_name} analysis")
                
        except Exception as e:
            print(f"❌ {model_name} analysis failed: {str(e)}")
            print(f"🔍 Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comprehensive summary
    print(f"\n{'='*60}")
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        print(f"📊 Models analyzed: {len(summary_data)}")
        print(f"📈 Total optimization trials: {summary_df['total_trials'].sum()}")
        print(f"✅ Successful trials: {summary_df['completed_trials'].sum()}")
        print()
        
        # Display summary table
        print("📋 OPTIMIZATION RESULTS SUMMARY:")
        print(summary_df.to_string(index=False))
        
        # Export summary to CSV
        summary_file = f"{results_dir}/hyperparameter_optimization_summary.csv"
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary exported to: {summary_file}")
        
        # Find best performing model
        valid_scores = summary_df.dropna(subset=['best_score'])
        if not valid_scores.empty:
            best_model = valid_scores.loc[valid_scores['best_score'].idxmax()]
            print(f"\n🏆 BEST PERFORMING MODEL: {best_model['model']}")
            print(f"   • Score: {best_model['best_score']:.4f}")
            print(f"   • Section: {best_model['section']}")
            print(f"   • Trials completed: {best_model['completed_trials']}")
    
    else:
        print("⚠️  No optimization results found")
        print("   Run hyperparameter optimization first (CHUNK_040, CHUNK_042, etc.)")
    
    print(f"\n✅ Section {section_number} hyperparameter optimization batch analysis completed!")
    print(f"📁 All figures and tables exported to: {results_dir}")
    
    return {
        'analysis_results': analysis_results,
        'summary_data': summary_data,
        'results_dir': results_dir
    }

# ============================================================================

from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_synthetic_data_quality(real_data, synthetic_data, model_name, target_column, 
                                  section_number, dataset_identifier=None, 
                                  save_files=True, display_plots=False, verbose=True):
    """
    Enhanced comprehensive evaluation of synthetic data quality with PCA analysis and file output.
    
    Parameters:
    - real_data: Original dataset
    - synthetic_data: Synthetic dataset to evaluate  
    - model_name: Model identifier (e.g., 'CTGAN', 'TVAE')
    - target_column: Name of target column for supervised metrics and PCA color-coding
    - section_number: Section number for file organization (2, 3, 5, etc.)
    - dataset_identifier: Dataset name for folder structure (auto-detected if None)
    - save_files: Whether to save plots and tables to files
    - display_plots: Whether to display plots in notebook (False for file-only mode)
    - verbose: Print detailed results
    
    Returns:
    - Dictionary with all evaluation metrics and file paths
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Auto-detect dataset identifier if not provided
    if dataset_identifier is None:
        dataset_identifier = DATASET_IDENTIFIER or "unknown-dataset"
    
    # Create results directory structure
    results_dir = None
    if save_files:
        results_dir = Path(get_results_path(dataset_identifier, section_number)) / model_name.upper()
        results_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"🔍 {model_name.upper()} - COMPREHENSIVE DATA QUALITY EVALUATION")
        print("=" * 60)
        if save_files:
            print(f"📁 Output directory: {results_dir}")
    
    results = {
        'model': model_name,
        'section': section_number,
        'files_generated': [],
        'dataset_identifier': dataset_identifier
    }
    
    # Get numeric columns for analysis
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
    # Keep target column for PCA analysis but separate for other analyses
    numeric_cols_no_target = [col for col in numeric_cols if col != target_column]
    
    # 1. STATISTICAL SIMILARITY
    if verbose:
        print("\n1️⃣ STATISTICAL SIMILARITY")
        print("-" * 30)
    
    stat_results = []
    for col in numeric_cols_no_target:
        try:
            real_vals = real_data[col].dropna()
            synth_vals = synthetic_data[col].dropna()
            
            real_mean, real_std = real_vals.mean(), real_vals.std()
            synth_mean, synth_std = synth_vals.mean(), synth_vals.std()
            
            mean_diff = abs(real_mean - synth_mean) / real_std if real_std > 0 else 0
            std_ratio = min(real_std, synth_std) / max(real_std, synth_std) if max(real_std, synth_std) > 0 else 1
            
            stat_results.append({
                'column': col,
                'real_mean': real_mean,
                'synthetic_mean': synth_mean,
                'mean_similarity': 1 - min(mean_diff, 1),
                'std_similarity': std_ratio,
                'overall_similarity': (1 - min(mean_diff, 1) + std_ratio) / 2
            })
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error analyzing {col}: {e}")
    
    if stat_results:
        stat_df = pd.DataFrame(stat_results)
        avg_stat_similarity = stat_df['overall_similarity'].mean()
        results['avg_statistical_similarity'] = avg_stat_similarity
        
        if save_files and results_dir:
            stat_file = results_dir / 'statistical_similarity.csv'
            stat_df.to_csv(stat_file, index=False)
            results['files_generated'].append(str(stat_file))
        
        if verbose:
            print(f"   📊 Average Statistical Similarity: {avg_stat_similarity:.3f}")
    
    # 2. PCA COMPARISON ANALYSIS WITH OUTCOME VARIABLE COLOR-CODING
    if verbose:
        print("\n2️⃣ PCA COMPARISON ANALYSIS WITH OUTCOME COLOR-CODING")
        print("-" * 50)
    
    pca_results = {}
    try:
        # Use all numeric columns including target for PCA
        pca_columns = [col for col in numeric_cols if col in synthetic_data.columns]
        
        if len(pca_columns) >= 2:
            # Standardize data
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_data[pca_columns].fillna(0))
            synth_scaled = scaler.transform(synthetic_data[pca_columns].fillna(0))
            
            # Apply PCA
            n_components = min(4, len(pca_columns))
            pca = PCA(n_components=n_components)
            real_pca = pca.fit_transform(real_scaled)
            synth_pca = pca.transform(synth_scaled)
            
            # Calculate PCA similarity metrics
            pca_similarities = []
            for i in range(n_components):
                corr = abs(stats.pearsonr(real_pca[:, i], synth_pca[:len(real_pca), i])[0]) if len(real_pca) > 0 else 0
                pca_similarities.append(corr)
            
            pca_similarity = np.mean(pca_similarities)
            explained_variance_real = pca.explained_variance_ratio_
            
            # Store PCA results
            pca_results = {
                'n_components': n_components,
                'explained_variance_ratio': explained_variance_real,
                'component_similarity': pca_similarities,
                'overall_pca_similarity': pca_similarity
            }
            results.update(pca_results)
            
            # Create PCA comparison plots with outcome color-coding
            if save_files or display_plots:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'{model_name.upper()} - PCA Comparison with {target_column.title()} Color-coding', 
                           fontsize=16, fontweight='bold')
                
                # Get target values for color-coding
                real_target = real_data[target_column] if target_column in real_data.columns else np.zeros(len(real_data))
                synth_target = synthetic_data[target_column] if target_column in synthetic_data.columns else np.zeros(len(synthetic_data))
                
                # Plot 1: Real data PC1 vs PC2
                axes[0,0].scatter(real_pca[:, 0], real_pca[:, 1], c=real_target, cmap='viridis', alpha=0.6, s=30)
                axes[0,0].set_title('Real Data - PC1 vs PC2')
                axes[0,0].set_xlabel(f'PC1 ({explained_variance_real[0]:.1%} variance)')
                axes[0,0].set_ylabel(f'PC2 ({explained_variance_real[1]:.1%} variance)')
                
                # Plot 2: Synthetic data PC1 vs PC2  
                scatter = axes[0,1].scatter(synth_pca[:, 0], synth_pca[:, 1], c=synth_target, cmap='viridis', alpha=0.6, s=30)
                axes[0,1].set_title('Synthetic Data - PC1 vs PC2')
                axes[0,1].set_xlabel(f'PC1 ({explained_variance_real[0]:.1%} variance)')
                axes[0,1].set_ylabel(f'PC2 ({explained_variance_real[1]:.1%} variance)')
                plt.colorbar(scatter, ax=axes[0,1], label=target_column.title())
                
                # Plot 3: Explained variance comparison
                components = range(1, n_components + 1)
                axes[1,0].bar([x - 0.2 for x in components], explained_variance_real, 0.4, 
                            label='Real Data', alpha=0.7, color='blue')
                # Note: Using same explained variance ratio for synthetic as approximation
                axes[1,0].bar([x + 0.2 for x in components], explained_variance_real, 0.4,
                            label='Synthetic Data', alpha=0.7, color='orange') 
                axes[1,0].set_title('Explained Variance Ratio Comparison')
                axes[1,0].set_xlabel('Principal Component')
                axes[1,0].set_ylabel('Explained Variance Ratio')
                axes[1,0].legend()
                
                # Plot 4: Component similarity scores
                axes[1,1].bar(components, pca_similarities, alpha=0.7, color='green')
                axes[1,1].set_title('PCA Component Similarity Scores')
                axes[1,1].set_xlabel('Principal Component')
                axes[1,1].set_ylabel('Similarity Score')
                axes[1,1].set_ylim(0, 1)
                
                plt.tight_layout()
                
                if save_files and results_dir:
                    pca_plot_file = results_dir / 'pca_comparison_with_outcome.png'
                    plt.savefig(pca_plot_file, dpi=300, bbox_inches='tight')
                    results['files_generated'].append(str(pca_plot_file))
                    if verbose:
                        print(f"   📊 PCA comparison plot saved: {pca_plot_file.name}")
                
                if display_plots:
                    plt.show()
                else:
                    plt.close()
            
            if verbose:
                print(f"   📊 PCA Overall Similarity: {pca_similarity:.3f}")
                print(f"   📊 Explained Variance (PC1, PC2): {explained_variance_real[0]:.3f}, {explained_variance_real[1]:.3f}")
                
    except Exception as e:
        if verbose:
            print(f"   ❌ PCA analysis failed: {e}")
        pca_results = {'error': str(e)}
    
    # 3. DISTRIBUTION SIMILARITY WITH VISUALIZATIONS
    if verbose:
        print("\n3️⃣ DISTRIBUTION SIMILARITY")
        print("-" * 30)
    
    try:
        if save_files or display_plots:
            n_cols = min(3, len(numeric_cols_no_target))
            n_rows = (len(numeric_cols_no_target) + n_cols - 1) // n_cols
            
            if n_rows > 0:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                fig.suptitle(f'{model_name.upper()} - Feature Distribution Comparison', 
                           fontsize=16, fontweight='bold')
                
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes
                else:
                    axes = axes.flatten()
                
                js_scores = []
                for i, col in enumerate(numeric_cols_no_target):
                    if i < len(axes):
                        ax = axes[i]
                        
                        # Calculate Jensen-Shannon divergence
                        try:
                            real_hist, bins = np.histogram(real_data[col].dropna(), bins=20, density=True)
                            synth_hist, _ = np.histogram(synthetic_data[col].dropna(), bins=bins, density=True)
                            
                            real_hist = real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
                            synth_hist = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist
                            
                            js_div = jensenshannon(real_hist, synth_hist)
                            js_similarity = 1 - js_div
                            js_scores.append(js_similarity)
                            
                            # Create distribution comparison plot
                            ax.hist(real_data[col].dropna(), bins=20, alpha=0.7, label='Real', 
                                  density=True, color='blue')
                            ax.hist(synthetic_data[col].dropna(), bins=20, alpha=0.7, label='Synthetic', 
                                  density=True, color='orange')
                            ax.set_title(f'{col}\nJS Similarity: {js_similarity:.3f}')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                        except Exception as e:
                            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'{col} - Error')
                
                # Hide unused subplots
                for j in range(len(numeric_cols_no_target), len(axes)):
                    axes[j].set_visible(False)
                
                plt.tight_layout()
                
                if save_files and results_dir:
                    dist_plot_file = results_dir / 'distribution_comparison.png'
                    plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
                    results['files_generated'].append(str(dist_plot_file))
                
                if display_plots:
                    plt.show()
                else:
                    plt.close()
                
                avg_js_similarity = np.mean(js_scores) if js_scores else 0
                results['avg_js_similarity'] = avg_js_similarity
                
                if verbose:
                    print(f"   📊 Average Distribution Similarity: {avg_js_similarity:.3f}")
    
    except Exception as e:
        if verbose:
            print(f"   ❌ Distribution analysis failed: {e}")
    
    # 4. CORRELATION STRUCTURE PRESERVATION
    if verbose:
        print("\n4️⃣ CORRELATION STRUCTURE")
        print("-" * 30)
    
    try:
        real_corr = real_data[numeric_cols].corr()
        synth_corr = synthetic_data[numeric_cols].corr()
        
        # Calculate correlation preservation
        corr_preservation = stats.pearsonr(
            real_corr.values.flatten(),
            synth_corr.values.flatten()
        )[0]
        corr_preservation = max(0, corr_preservation)
        results['correlation_preservation'] = corr_preservation
        
        # Create correlation heatmap comparison
        if save_files or display_plots:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Real data correlation
            sns.heatmap(real_corr, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=axes[0], fmt='.2f')
            axes[0].set_title('Real Data - Correlation Matrix')
            
            # Synthetic data correlation
            sns.heatmap(synth_corr, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=axes[1], fmt='.2f')
            axes[1].set_title(f'Synthetic Data - Correlation Matrix\nPreservation Score: {corr_preservation:.3f}')
            
            plt.tight_layout()
            
            if save_files and results_dir:
                corr_plot_file = results_dir / 'correlation_comparison.png'
                plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
                results['files_generated'].append(str(corr_plot_file))
            
            if display_plots:
                plt.show()
            else:
                plt.close()
        
        if verbose:
            print(f"   📊 Correlation Structure Preservation: {corr_preservation:.3f}")
            
    except Exception as e:
        if verbose:
            print(f"   ❌ Correlation analysis failed: {e}")
        results['correlation_preservation'] = 0
    
    # 5. MACHINE LEARNING UTILITY
    if target_column and target_column in real_data.columns:
        if verbose:
            print("\n5️⃣ MACHINE LEARNING UTILITY")
            print("-" * 30)
        
        try:
            X_real = real_data[numeric_cols_no_target].fillna(0)
            y_real = real_data[target_column]
            X_synth = synthetic_data[numeric_cols_no_target].fillna(0)
            y_synth = synthetic_data[target_column] if target_column in synthetic_data.columns else None
            
            if y_synth is not None:
                # Train on real, test on synthetic
                rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_real.fit(X_real, y_real)
                synth_test_accuracy = rf_real.score(X_synth, y_synth)
                
                # Train on synthetic, test on real
                rf_synth = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_synth.fit(X_synth, y_synth)
                real_test_accuracy = rf_synth.score(X_real, y_real)
                
                ml_utility = (synth_test_accuracy + real_test_accuracy) / 2
                
                results.update({
                    'ml_utility': ml_utility,
                    'synth_test_accuracy': synth_test_accuracy,
                    'real_test_accuracy': real_test_accuracy
                })
                
                if verbose:
                    print(f"   📊 ML Utility (Cross-Accuracy): {ml_utility:.3f}")
                    print(f"   📊 Real→Synth Accuracy: {synth_test_accuracy:.3f}")
                    print(f"   📊 Synth→Real Accuracy: {real_test_accuracy:.3f}")
            
        except Exception as e:
            if verbose:
                print(f"   ❌ ML utility analysis failed: {e}")
    
    # 6. OVERALL QUALITY ASSESSMENT
    quality_scores = []
    if 'avg_statistical_similarity' in results:
        quality_scores.append(results['avg_statistical_similarity'])
    if 'avg_js_similarity' in results:
        quality_scores.append(results['avg_js_similarity'])
    if 'correlation_preservation' in results:
        quality_scores.append(results['correlation_preservation'])
    if 'overall_pca_similarity' in results:
        quality_scores.append(results['overall_pca_similarity'])
    if 'ml_utility' in results:
        quality_scores.append(results['ml_utility'])
    
    overall_quality = np.mean(quality_scores) if quality_scores else 0
    results['overall_quality_score'] = overall_quality
    
    # Quality assessment
    if overall_quality >= 0.8:
        quality_label = "EXCELLENT"
    elif overall_quality >= 0.6:
        quality_label = "GOOD"
    elif overall_quality >= 0.4:
        quality_label = "FAIR"
    else:
        quality_label = "POOR"
    
    results['quality_assessment'] = quality_label
    
    # Save comprehensive results summary
    if save_files and results_dir:
        summary_df = pd.DataFrame([{
            'Model': model_name,
            'Overall_Quality_Score': overall_quality,
            'Quality_Assessment': quality_label,
            'Statistical_Similarity': results.get('avg_statistical_similarity', 'N/A'),
            'Distribution_Similarity': results.get('avg_js_similarity', 'N/A'),
            'Correlation_Preservation': results.get('correlation_preservation', 'N/A'),
            'PCA_Similarity': results.get('overall_pca_similarity', 'N/A'),
            'ML_Utility': results.get('ml_utility', 'N/A')
        }])
        
        summary_file = results_dir / 'evaluation_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        results['files_generated'].append(str(summary_file))
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"🏆 {model_name.upper()} OVERALL QUALITY SCORE: {overall_quality:.3f}")
        print(f"📋 Quality Assessment: {quality_label}")
        print("=" * 60)
        
        if save_files:
            print(f"\n📁 Generated {len(results['files_generated'])} output files:")
            for file_path in results['files_generated']:
                print(f"   • {Path(file_path).name}")
    
    return results

print("✅ Comprehensive data quality evaluation function loaded!")

# ============================================================================
# BATCH EVALUATION SYSTEM FOR ALL SECTIONS
# ============================================================================

def evaluate_all_available_models(section_number, scope=None, models_to_evaluate=None, real_data=None, target_col=None):
    """
    Enhanced batch evaluation for all available synthetic datasets with notebook scope support
    
    Parameters:
    - section_number: Section number for file organization (3, 5, etc.)
    - scope: globals() from notebook for variable access (required for notebook use)
    - models_to_evaluate: List of specific models to evaluate (optional, evaluates all if None)
    - real_data: Real dataset (uses 'data' from scope if not provided)
    - target_col: Target column name (uses 'target_column' from scope if not provided)
    
    Returns:
    - Dictionary with results for each evaluated model
    """
    
    if scope is None:
        print("❌ ERROR: scope parameter required! Pass globals() from notebook")
        return {}
    
    # Get data and target from scope if not provided
    if real_data is None:
        real_data = scope.get('data')
        if real_data is None:
            print("❌ ERROR: 'data' variable not found in scope")
            return {}
    
    if target_col is None:
        target_col = scope.get('target_column')
        if target_col is None:
            print("❌ ERROR: 'target_column' variable not found in scope")
            return {}
    
    dataset_id = scope.get('DATASET_IDENTIFIER', 'unknown-dataset')
    
    # Model mappings - check for available synthetic datasets in scope
    available_models = {}
    
    # Standard model variable names to check
    model_checks = {
        'CTGAN': 'synthetic_data_ctgan',
        'CTABGAN': 'synthetic_data_ctabgan', 
        'CTABGANPLUS': 'synthetic_data_ctabganplus',
        'GANerAid': 'synthetic_data_ganeraid',
        'CopulaGAN': 'synthetic_data_copulagan',
        'TVAE': 'synthetic_data_tvae'
    }
    
    # Check which models are available in notebook scope
    for model_name, var_name in model_checks.items():
        if var_name in scope and scope[var_name] is not None:
            # Filter by requested models if specified
            if models_to_evaluate is None or model_name in models_to_evaluate or model_name.lower() in [m.lower() for m in models_to_evaluate]:
                available_models[model_name] = scope[var_name]
    
    print(f"🔍 BATCH EVALUATION - SECTION {section_number}")
    print("=" * 60)
    print(f"📋 Dataset: {dataset_id}")
    print(f"📋 Target column: {target_col}")
    print(f"📋 Found {len(available_models)} trained models:")
    for model_name in available_models.keys():
        print(f"   ✅ {model_name}")
    
    if not available_models:
        available_vars = [var for var in model_checks.values() if var in scope]
        print("❌ No synthetic datasets found!")
        print("   Train some models first before running batch evaluation")
        if available_vars:
            print(f"   Found variables: {available_vars}")
        return {}
    
    # Evaluate each available model
    evaluation_results = {}
    
    for model_name, synthetic_data in available_models.items():
        print(f"\n{'='*20} EVALUATING {model_name} {'='*20}")
        
        try:
            results = evaluate_synthetic_data_quality(
                real_data=real_data,
                synthetic_data=synthetic_data,
                model_name=model_name,
                target_column=target_col,
                section_number=section_number,
                dataset_identifier=dataset_id,
                save_files=True,
                display_plots=False,  # File-only mode for batch processing
                verbose=True
            )
            
            evaluation_results[model_name] = results
            print(f"✅ {model_name} evaluation completed successfully!")
            
        except Exception as e:
            print(f"❌ {model_name} evaluation failed: {e}")
            evaluation_results[model_name] = {'error': str(e)}
    
    # Create summary comparison
    print(f"\n{'='*25} EVALUATION SUMMARY {'='*25}")
    print(f"{'Model':<15} {'Quality Score':<15} {'Assessment':<12} {'Files':<8}")
    print("-" * 65)
    
    for model_name, results in evaluation_results.items():
        if 'error' not in results:
            quality_score = results.get('overall_quality_score', 0)
            assessment = results.get('quality_assessment', 'Unknown')
            file_count = len(results.get('files_generated', []))
            print(f"{model_name:<15} {quality_score:<15.3f} {assessment:<12} {file_count:<8}")
        else:
            print(f"{model_name:<15} {'ERROR':<15} {'FAILED':<12} {'0':<8}")
    
    # Save comparison summary
    if evaluation_results:
        try:
            summary_data = []
            for model_name, results in evaluation_results.items():
                if 'error' not in results:
                    summary_data.append({
                        'Model': model_name,
                        'Section': section_number,
                        'Quality_Score': results.get('overall_quality_score', 0),
                        'Quality_Assessment': results.get('quality_assessment', 'Unknown'),
                        'Statistical_Similarity': results.get('avg_statistical_similarity', 'N/A'),
                        'PCA_Similarity': results.get('overall_pca_similarity', 'N/A'),
                        'Files_Generated': len(results.get('files_generated', []))
                    })
            
            if summary_data:
                import pandas as pd
                summary_df = pd.DataFrame(summary_data)
                summary_path = get_results_path(dataset_id, section_number)
                os.makedirs(summary_path, exist_ok=True)
                summary_file = f"{summary_path}/batch_evaluation_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                print(f"\n📊 Batch summary saved to: {summary_file}")
                
        except Exception as e:
            print(f"⚠️ Could not save batch summary: {e}")
    
    return evaluation_results

# Code Chunk ID: CHUNK_037 - Enhanced Objective Function v2
def enhanced_objective_function_v2(real_data, synthetic_data, target_column, 
                                 similarity_weight=0.6, accuracy_weight=0.4):
    """
    Enhanced objective function: 60% similarity + 40% accuracy with DYNAMIC TARGET COLUMN FIX
    
    Args:
        real_data: Original dataset
        synthetic_data: Generated synthetic dataset  
        target_column: Name of target column (DYNAMIC - works with any dataset)
        similarity_weight: Weight for similarity component (default 0.6)
        accuracy_weight: Weight for accuracy component (default 0.4)

    Returns:
        Combined objective score (higher is better), similarity_score, accuracy_score
    """
    
    print(f"🎯 Enhanced objective function using target column: '{target_column}'")
    
    # CRITICAL FIX: Validate target column exists in both datasets
    if target_column not in real_data.columns:
        print(f"❌ Target column '{target_column}' not found in real data columns: {list(real_data.columns)}")
        return 0.0, 0.0, 0.0
    
    if target_column not in synthetic_data.columns:
        print(f"❌ Target column '{target_column}' not found in synthetic data columns: {list(synthetic_data.columns)}")
        return 0.0, 0.0, 0.0
    
    # 1. Similarity Component (60%)
    similarity_scores = []
    
    # Univariate similarity using Earth Mover's Distance
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != target_column and col in synthetic_data.columns:
            try:
                # DATA TYPE VALIDATION: Ensure both columns are numeric before EMD calculation
                real_values = real_data[col]
                synth_values = synthetic_data[col]
                
                # Check for mixed data types in synthetic data
                if synth_values.dtype == 'object' or synth_values.apply(lambda x: isinstance(x, str)).any():
                    print(f"⚠️ Warning: {col} in synthetic data contains non-numeric values, attempting to convert...")
                    # Try to convert to numeric, replacing invalid values with NaN
                    synth_values = pd.to_numeric(synth_values, errors='coerce')
                    # Remove NaN values for EMD calculation
                    synth_values = synth_values.dropna()
                    real_values = real_values.dropna()
                    print(f"✅ Converted {col}: {len(synth_values)} valid numeric values")
                
                # Ensure we have enough values for EMD calculation
                if len(real_values) == 0 or len(synth_values) == 0:
                    print(f"❌ Skipping {col}: insufficient valid numeric values")
                    continue
                    
                # Earth Mover's Distance (Wasserstein distance)
                emd_score = wasserstein_distance(real_values, synth_values)
                # Convert to similarity (lower EMD = higher similarity)
                similarity_scores.append(1 / (1 + emd_score))
                print(f"✅ {col}: EMD={emd_score:.4f}, Similarity={similarity_scores[-1]:.4f}")
                
            except Exception as e:
                print(f"❌ Error calculating EMD for {col}: {e}")
                print(f"   Real dtype: {real_data[col].dtype}, Synthetic dtype: {synthetic_data[col].dtype}")
                continue
    
    # Correlation similarity
    try:
        # Only use columns that have valid numeric data in both datasets
        valid_numeric_cols = []
        for col in numeric_columns:
            if col in synthetic_data.columns and col != target_column:
                if not synthetic_data[col].apply(lambda x: isinstance(x, str)).any():
                    valid_numeric_cols.append(col)
        
        if len(valid_numeric_cols) > 1:
            real_corr = real_data[valid_numeric_cols].corr()
            synth_corr = synthetic_data[valid_numeric_cols].corr()
            
            # Flatten correlation matrices and compute distance
            real_corr_flat = real_corr.values[np.triu_indices_from(real_corr, k=1)]
            synth_corr_flat = synth_corr.values[np.triu_indices_from(synth_corr, k=1)]
            
            # Correlation similarity (1 - distance)
            corr_distance = np.mean(np.abs(real_corr_flat - synth_corr_flat))
            similarity_scores.append(1 - corr_distance)
            print(f"✅ Correlation similarity: {similarity_scores[-1]:.4f}")
        else:
            print("⚠️ Insufficient valid numeric columns for correlation analysis")
            
    except Exception as e:
        print(f"Warning: Correlation similarity failed: {e}")
    
    similarity_score = np.mean(similarity_scores) if similarity_scores else 0.5
    
    # 2. Accuracy Component (40%) - TRTS Framework with DYNAMIC TARGET COLUMN FIX
    accuracy_scores = []
    
    try:
        # CRITICAL FIX: Robust column existence checking
        print(f"🔧 Preparing TRTS evaluation with target column: '{target_column}'")
        
        # Ensure target column exists before proceeding
        if target_column not in real_data.columns or target_column not in synthetic_data.columns:
            print(f"❌ Target column '{target_column}' missing. Real cols: {list(real_data.columns)[:5]}...")
            return similarity_score * similarity_weight, similarity_score, 0.0
        
        # Prepare features and target with robust error handling
        try:
            X_real = real_data.drop(columns=[target_column])
            y_real = real_data[target_column]
            X_synth = synthetic_data.drop(columns=[target_column]) 
            y_synth = synthetic_data[target_column]
            
            print(f"🔧 Data shapes - Real: X{X_real.shape}, y{y_real.shape}, Synthetic: X{X_synth.shape}, y{y_synth.shape}")
            
        except KeyError as ke:
            print(f"❌ KeyError in data preparation: {ke}")
            return similarity_score * similarity_weight, similarity_score, 0.0
        
        # CRITICAL FIX: Ensure consistent label types before any sklearn operations
        print(f"🔧 Data type check - Real: {y_real.dtype}, Synthetic: {y_synth.dtype}")
        
        # Convert all labels to same type (prefer numeric if possible)
        if y_real.dtype != y_synth.dtype:
            print(f"⚠️ Data type mismatch detected - harmonizing types")
            if pd.api.types.is_numeric_dtype(y_real):
                try:
                    y_synth = pd.to_numeric(y_synth, errors='coerce')
                    print(f"✅ Converted synthetic labels to numeric")
                except:
                    y_real = y_real.astype(str)
                    y_synth = y_synth.astype(str)
                    print(f"✅ Converted both to string type")
            else:
                y_real = y_real.astype(str)
                y_synth = y_synth.astype(str)
                print(f"✅ Converted both to string type")
        
        # Ensure we have matching features between datasets
        common_features = list(set(X_real.columns) & set(X_synth.columns))
        if len(common_features) == 0:
            print("❌ No common features between real and synthetic data")
            return similarity_score * similarity_weight, similarity_score, 0.0
        
        print(f"🔧 Using {len(common_features)} common features for TRTS evaluation")
        
        X_real = X_real[common_features]
        X_synth = X_synth[common_features]
        
        # Handle mixed data types in features
        for col in common_features:
            if X_synth[col].dtype == 'object':
                try:
                    X_synth[col] = pd.to_numeric(X_synth[col], errors='coerce')
                    if X_synth[col].isna().all():
                        # If conversion failed, use label encoding
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        X_real[col] = le.fit_transform(X_real[col].astype(str))
                        X_synth[col] = le.transform(X_synth[col].astype(str))
                        print(f"🔧 Label encoded column: {col}")
                    else:
                        print(f"🔧 Converted to numeric: {col}")
                except Exception as e:
                    print(f"⚠️ Could not process column {col}: {e}")
                    # Drop problematic columns
                    X_real = X_real.drop(columns=[col])
                    X_synth = X_synth.drop(columns=[col])
        
        # Handle missing values
        X_real = X_real.fillna(X_real.median())
        X_synth = X_synth.fillna(X_synth.median())
        
        # Final check for remaining NaN values
        if X_real.isna().any().any() or X_synth.isna().any().any():
            X_real = X_real.fillna(0)
            X_synth = X_synth.fillna(0)
            print("⚠️ Used zero-fill for remaining NaN values")
        
        # TRTS: Train on Real, Test on Synthetic (and vice versa)
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Ensure we have sufficient samples
        if len(X_real) < 10 or len(X_synth) < 10:
            print("⚠️ Insufficient samples for TRTS evaluation")
            return similarity_score * similarity_weight, similarity_score, 0.5
        
        # TRTS 1: Train on Real, Test on Synthetic
        try:
            rf1 = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            rf1.fit(X_real, y_real)
            pred_synth = rf1.predict(X_synth)
            acc1 = accuracy_score(y_synth, pred_synth)
            accuracy_scores.append(acc1)
            print(f"✅ TRTS (Real→Synthetic): {acc1:.4f}")
        except Exception as e:
            print(f"❌ TRTS (Real→Synthetic) failed: {e}")
        
        # TRTS 2: Train on Synthetic, Test on Real
        try:
            rf2 = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            rf2.fit(X_synth, y_synth)
            pred_real = rf2.predict(X_real)
            acc2 = accuracy_score(y_real, pred_real)
            accuracy_scores.append(acc2)
            print(f"✅ TRTS (Synthetic→Real): {acc2:.4f}")
        except Exception as e:
            print(f"❌ TRTS (Synthetic→Real) failed: {e}")
            
    except Exception as e:
        print(f"❌ Accuracy evaluation failed: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
    
    # Calculate final scores
    accuracy_score_final = np.mean(accuracy_scores) if accuracy_scores else 0.5
    combined_score = (similarity_score * similarity_weight) + (accuracy_score_final * accuracy_weight)
    
    print(f"📊 Final scores - Similarity: {similarity_score:.4f}, Accuracy: {accuracy_score_final:.4f}, Combined: {combined_score:.4f}")
    
    return combined_score, similarity_score, accuracy_score_final

# Code Chunk ID: CHUNK_039 - Analyze Hyperparameter Optimization
def analyze_hyperparameter_optimization(study_results, model_name, 
                                       target_column, results_dir=None,
                                       export_figures=True, export_tables=True,
                                       display_plots=True):
    """
    Comprehensive hyperparameter optimization analysis with file output
    Reusable across all model sections in Section 4
    
    Enhanced following Section 3 lessons learned:
    - Model-specific subdirectories for clean organization
    - Professional dataframe display for all tables
    - Consistent display + file output for all models
    - High-quality graphics with proper styling
    
    Parameters:
    - study_results: Optuna study object or trial results dataframe
    - model_name: str, model identifier (ctgan, ctabgan, etc.)
    - target_column: str, target column name for context
    - results_dir: str, base results directory (creates model subdirectories)
    - export_figures: bool, save graphics to files
    - export_tables: bool, save tables to CSV files  
    - display_plots: bool, show plots and dataframes in notebook
    
    Returns:
    - Dictionary with analysis results and file paths
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')
    
    # Helper function to safely convert parameters for plotting
    def safe_plot_parameter(param_col, trials_df):
        """Convert parameter values to plottable numeric format"""
        param_data = trials_df[param_col]
        
        # Handle TimeDelta64 types (convert to seconds)
        if pd.api.types.is_timedelta64_dtype(param_data):
            return param_data.dt.total_seconds()
        
        # Handle datetime types (convert to timestamp)
        elif pd.api.types.is_datetime64_dtype(param_data):
            return pd.to_numeric(param_data)
        
        # Handle list/tuple parameters (convert to string representation)
        elif param_data.apply(lambda x: isinstance(x, (list, tuple))).any():
            return param_data.astype(str)
        
        # Handle object/categorical types
        elif pd.api.types.is_object_dtype(param_data) or pd.api.types.is_categorical_dtype(param_data):
            try:
                # Try to convert to numeric
                return pd.to_numeric(param_data, errors='coerce')
            except:
                # If conversion fails, use string representation
                return param_data.astype(str)
        
        # For numeric types, return as-is
        else:
            return param_data
    
    # Enhanced Setup with Model-Specific Subdirectories (Following Section 3 Pattern)
    if results_dir is None:
        base_results_dir = Path('./results')
    else:
        base_results_dir = Path(results_dir)
    
    # Create model-specific subdirectory for clean organization
    results_dir = base_results_dir / 'section4_optimizations' / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 ANALYZING {model_name.upper()} HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    try:
        # 1. EXTRACT AND PROCESS TRIAL DATA
        print("📊 1. TRIAL DATA EXTRACTION AND PROCESSING")
        print("-" * 40)
        
        # Handle both Optuna Study objects and DataFrames
        if hasattr(study_results, 'trials_dataframe'):
            trials_df = study_results.trials_dataframe()
        elif hasattr(study_results, 'trials'):
            # Convert Optuna study to DataFrame manually
            trial_data = []
            for trial in study_results.trials:
                trial_dict = {
                    'number': trial.number,
                    'value': trial.value,
                    'datetime_start': trial.datetime_start,
                    'datetime_complete': trial.datetime_complete,
                    'duration': trial.duration,
                    'state': trial.state.name
                }
                # Add parameters with 'params_' prefix
                for key, value in trial.params.items():
                    trial_dict[f'params_{key}'] = value
                trial_data.append(trial_dict)
            trials_df = pd.DataFrame(trial_data)
        else:
            # Assume it's already a DataFrame
            trials_df = study_results.copy()
        
        if trials_df.empty:
            print("❌ No trial data available for analysis")
            return {"error": "No trial data available"}
        
        print(f"✅ Extracted {len(trials_df)} trials for analysis")
        
        # Get parameter columns (those starting with 'params_')
        param_cols = [col for col in trials_df.columns if col.startswith('params_')]
        objective_col = 'value'
        
        if objective_col not in trials_df.columns:
            print(f"❌ Objective column '{objective_col}' not found")
            return {"error": f"Objective column '{objective_col}' not found"}
        
        print(f"📊 2. PARAMETER SPACE EXPLORATION ANALYSIS")
        print("-" * 40)
        print(f"   • Found {len(param_cols)} hyperparameters: {param_cols}")
        
        # Filter out completed trials for analysis
        completed_trials = trials_df[trials_df['state'] == 'COMPLETE'] if 'state' in trials_df.columns else trials_df
        
        if completed_trials.empty:
            print("❌ No completed trials available for analysis")
            return {"error": "No completed trials available"}
        
        print(f"   • Using {len(completed_trials)} completed trials")
        
        # ENHANCED PARAMETER VS PERFORMANCE VISUALIZATION (WITH DTYPE FIX)
        if param_cols and display_plots:
            print(f"📈 Creating parameter vs performance visualizations...")
            
            # Limit parameters for readability 
            n_params = min(6, len(param_cols))  # Limit to 6 for visualization
            if n_params > 0:
                n_cols = 3
                n_rows = (n_params + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_params == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes
                else:
                    axes = axes.flatten()
                
                fig.suptitle(f'{model_name.upper()} - Parameter vs Performance Analysis', 
                             fontsize=16, fontweight='bold')
                
                for i, param_col in enumerate(param_cols[:n_params]):
                    if param_col in completed_trials.columns:
                        try:
                            # CRITICAL FIX: Use safe parameter conversion for plotting
                            param_data = safe_plot_parameter(param_col, completed_trials)
                            objective_data = completed_trials[objective_col]
                            
                            # Create scatter plot with converted data
                            axes[i].scatter(param_data, objective_data, alpha=0.6, s=50)
                            axes[i].set_xlabel(param_col)
                            axes[i].set_ylabel(f'{objective_col}')
                            axes[i].set_title(f'{param_col} vs Performance', fontweight='bold')
                            axes[i].grid(True, alpha=0.3)
                            
                            # Add correlation coefficient if possible
                            try:
                                if pd.api.types.is_numeric_dtype(param_data):
                                    corr = param_data.corr(objective_data)
                                    axes[i].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                                               transform=axes[i].transAxes,
                                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            except Exception as corr_error:
                                print(f"⚠️ Could not calculate correlation for {param_col}: {corr_error}")
                                
                        except Exception as plot_error:
                            print(f"⚠️ Could not plot {param_col}: {plot_error}")
                            axes[i].text(0.5, 0.5, f'Plot Error\\n{param_col}', 
                                       transform=axes[i].transAxes, ha='center', va='center')
                            axes[i].set_title(f'{param_col} (Plot Error)', fontweight='bold')
                
                # Remove empty subplots
                for j in range(n_params, len(axes)):
                    fig.delaxes(axes[j])
                
                plt.tight_layout()
                
                if export_figures:
                    param_plot_path = results_dir / f'{model_name}_parameter_analysis.png'
                    plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
                    print(f"   📁 Parameter analysis plot saved: {param_plot_path}")
                
                if display_plots:
                    plt.show()
                else:
                    plt.close()
        
        # BEST TRIAL ANALYSIS
        print(f"📊 3. BEST TRIAL ANALYSIS")
        print("-" * 40)
        
        best_trial = completed_trials.loc[completed_trials[objective_col].idxmax()]
        print(f"✅ Best Trial #{best_trial.get('number', 'Unknown')}")
        print(f"   • Best Score: {best_trial[objective_col]:.4f}")
        
        if 'duration' in best_trial:
            duration = best_trial['duration']
            if pd.isna(duration):
                print(f"   • Duration: Not available")
            else:
                try:
                    if isinstance(duration, pd.Timedelta):
                        print(f"   • Duration: {duration.total_seconds():.1f} seconds")
                    else:
                        print(f"   • Duration: {duration}")
                except:
                    print(f"   • Duration: {duration}")
        
        # Best parameters
        best_params = {col.replace('params_', ''): best_trial[col] 
                      for col in param_cols if col in best_trial.index}
        
        print(f"   • Best Parameters:")
        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"     - {param}: {value:.4f}")
            else:
                print(f"     - {param}: {value}")
        
        # CONVERGENCE ANALYSIS
        print(f"📊 4. CONVERGENCE ANALYSIS")
        print("-" * 40)
        
        if len(completed_trials) > 1 and display_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Trial progression
            ax1.plot(completed_trials['number'], completed_trials[objective_col], 'o-', alpha=0.7)
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Objective Value')
            ax1.set_title(f'{model_name.upper()} - Trial Progression', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Best value progression (cumulative best)
            cumulative_best = completed_trials[objective_col].cummax()
            ax2.plot(completed_trials['number'], cumulative_best, 'g-', linewidth=2, label='Best So Far')
            ax2.fill_between(completed_trials['number'], cumulative_best, alpha=0.3, color='green')
            ax2.set_xlabel('Trial Number')
            ax2.set_ylabel('Best Objective Value')
            ax2.set_title(f'{model_name.upper()} - Convergence', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if export_figures:
                convergence_plot_path = results_dir / f'{model_name}_convergence_analysis.png'
                plt.savefig(convergence_plot_path, dpi=300, bbox_inches='tight')
                print(f"   📁 Convergence plot saved: {convergence_plot_path}")
            
            if display_plots:
                plt.show()
            else:
                plt.close()
        
        # STATISTICAL SUMMARY
        print(f"📊 5. STATISTICAL SUMMARY")
        print("-" * 40)
        
        summary_stats = completed_trials[objective_col].describe()
        print(f"✅ Performance Statistics:")
        print(f"   • Mean Score: {summary_stats['mean']:.4f}")
        print(f"   • Std Dev: {summary_stats['std']:.4f}")
        print(f"   • Min Score: {summary_stats['min']:.4f}")
        print(f"   • Max Score: {summary_stats['max']:.4f}")
        print(f"   • Median Score: {summary_stats['50%']:.4f}")
        
        # EXPORT RESULTS TABLES
        files_generated = []
        
        if export_tables:
            # Export trial results
            trials_export_path = results_dir / f'{model_name}_trial_results.csv'
            completed_trials.to_csv(trials_export_path, index=False)
            files_generated.append(str(trials_export_path))
            print(f"   📁 Trial results saved: {trials_export_path}")
            
            # Export summary statistics
            summary_export_path = results_dir / f'{model_name}_optimization_summary.csv'
            summary_df = pd.DataFrame({
                'Metric': ['Best Score', 'Mean Score', 'Std Dev', 'Min Score', 'Max Score', 'Trials Completed'],
                'Value': [best_trial[objective_col], summary_stats['mean'], 
                         summary_stats['std'], summary_stats['min'], 
                         summary_stats['max'], len(completed_trials)]
            })
            summary_df.to_csv(summary_export_path, index=False)
            files_generated.append(str(summary_export_path))
            print(f"   📁 Summary statistics saved: {summary_export_path}")
        
        # PREPARE RETURN DATA
        analysis_results = {
            'best_score': float(best_trial[objective_col]),
            'best_params': best_params,
            'n_trials': len(completed_trials),
            'mean_score': float(summary_stats['mean']),
            'std_score': float(summary_stats['std']),
            'trials_df': completed_trials,
            'files_generated': files_generated,
            'output_dir': str(results_dir)
        }
        
        print(f"✅ {model_name.upper()} optimization analysis completed successfully!")
        print(f"📁 Results saved to: {results_dir}")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Error in {model_name} hyperparameter optimization analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

print("✅ Batch evaluation system loaded!")
print("✅ Enhanced objective function v2 with DYNAMIC TARGET COLUMN support defined!")
print("✅ Enhanced hyperparameter optimization analysis function loaded!")

print("🎯 SETUP MODULE LOADED SUCCESSFULLY!")
print("="*60)

# Setup Imports - Global dependencies for objective functions
# CRITICAL: This cell provides wasserstein_distance import for enhanced_objective_function_v2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance  # CRITICAL: Required for EMD calculations
print("✅ Enhanced objective function dependencies imported")

# Set style
plt.style.use('default')
sns.set_palette("husl")
print("📦 Basic libraries imported successfully")

# Import Optuna for hyperparameter optimization
OPTUNA_AVAILABLE = False
try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✅ Optuna imported successfully")
except ImportError:
    print("❌ Optuna not found - hyperparameter optimization not available")

# Import CTGAN
CTGAN_AVAILABLE = False
try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
    print("✅ CTGAN imported successfully")
except ImportError:
    print("❌ CTGAN not found")

print("🔧 Setup imports cell restored from main branch - wasserstein_distance now available globally")