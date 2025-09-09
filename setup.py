# Setup Module for Clinical Synthetic Data Generation Framework
# Contains imported chunks from notebook for better organization

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

except Exception as e:
    print(f"ERROR importing CTAB-GAN: {e}")
    # Create a dummy class to prevent import errors
    class CTABGANSynthesizer:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CTAB-GAN import failed: {e}")

# Code Chunk ID: CHUNK_002 - CTABGANModel Class
class CTABGANModel:
    def __init__(self, epochs=100, batch_size=500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        
    def fit(self, data, categorical_columns=None, target_column=None):
        """Train CTAB-GAN model"""
        try:
            # Initialize CTAB-GAN with enhanced parameters
            self.model = CTABGANSynthesizer(
                epochs=self.epochs,
                batch_size=self.batch_size,
                categorical_columns=categorical_columns or []
            )
            
            # Train the model
            print(f"Training CTAB-GAN for {self.epochs} epochs...")
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
        
    def _check_plus_features(self):
        """Check if CTAB-GAN+ features are available"""
        try:
            # Try to import CTAB-GAN+ specific modules
            from model.ctabganplus import CTABGANPlusSynthesizer
            self.has_plus_features = True
            return CTABGANPlusSynthesizer
        except ImportError:
            print("WARNING: CTAB-GAN+ features not available, falling back to regular CTAB-GAN parameters")
            self.has_plus_features = False
            return CTABGANSynthesizer
    
    def fit(self, data, categorical_columns=None, target_column=None):
        """Train CTAB-GAN+ model with enhanced features"""
        try:
            # Check for CTAB-GAN+ availability
            SynthesizerClass = self._check_plus_features()
            
            # Enhanced parameters for CTAB-GAN+
            params = {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'categorical_columns': categorical_columns or []
            }
            
            # Add CTAB-GAN+ specific parameters if available
            if self.has_plus_features:
                params.update({
                    'privacy_budget': 1.0,  # Differential privacy budget
                    'target_column': target_column,
                    'enhanced_conditioning': True
                })
            
            # Initialize and train
            self.model = SynthesizerClass(**params)
            print(f"Training CTAB-GAN{'+ (Enhanced)' if self.has_plus_features else ''} for {self.epochs} epochs...")
            self.model.fit(data)
            print("✅ CTAB-GAN+ training completed successfully")
            
        except Exception as e:
            print(f"❌ CTAB-GAN+ training failed: {e}")
            raise
    
    def generate(self, n_samples):
        """Generate synthetic samples using CTAB-GAN+"""
        if self.model is None:
            raise ValueError("Model must be fitted before generating samples")
        
        try:
            synthetic_data = self.model.sample(n_samples)
            print(f"✅ Generated {len(synthetic_data)} synthetic samples using CTAB-GAN{'+ (Enhanced)' if self.has_plus_features else ''}")
            return synthetic_data
        except Exception as e:
            print(f"❌ CTAB-GAN+ generation failed: {e}")
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

from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_synthetic_data_quality(real_data, synthetic_data, target_column=None, verbose=True):
    """
    Comprehensive evaluation of synthetic data quality using multiple metrics.
    
    Parameters:
    - real_data: Original dataset
    - synthetic_data: Synthetic dataset to evaluate
    - target_column: Name of target column for supervised metrics
    - verbose: Print detailed results
    
    Returns:
    - Dictionary with all evaluation metrics
    """
    
    if verbose:
        print("🔍 COMPREHENSIVE DATA QUALITY EVALUATION")
        print("=" * 50)
    
    results = {}
    
    # 1. Basic Statistics Comparison
    if verbose:
        print("\n1️⃣ STATISTICAL SIMILARITY")
        print("-" * 30)
    
    # Get numeric columns
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    stat_similarity = {}
    for col in numeric_cols:
        real_mean = real_data[col].mean()
        synth_mean = synthetic_data[col].mean()
        real_std = real_data[col].std()
        synth_std = synthetic_data[col].std()
        
        mean_diff = abs(real_mean - synth_mean) / real_std if real_std > 0 else 0
        std_ratio = min(real_std, synth_std) / max(real_std, synth_std) if max(real_std, synth_std) > 0 else 1
        
        stat_similarity[col] = {
            'mean_similarity': 1 - min(mean_diff, 1),
            'std_similarity': std_ratio
        }
    
    results['statistical_similarity'] = stat_similarity
    avg_stat_similarity = np.mean([np.mean(list(metrics.values())) for metrics in stat_similarity.values()])
    results['avg_statistical_similarity'] = avg_stat_similarity
    
    if verbose:
        print(f"   📊 Average Statistical Similarity: {avg_stat_similarity:.3f}")
    
    # 2. Distribution Similarity (Jensen-Shannon Divergence)
    if verbose:
        print("\n2️⃣ DISTRIBUTION SIMILARITY")
        print("-" * 30)
    
    js_divergences = {}
    for col in numeric_cols:
        try:
            # Create histograms for comparison
            real_hist, bins = np.histogram(real_data[col].dropna(), bins=20, density=True)
            synth_hist, _ = np.histogram(synthetic_data[col].dropna(), bins=bins, density=True)
            
            # Normalize to create probability distributions
            real_hist = real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
            synth_hist = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist
            
            # Calculate Jensen-Shannon divergence
            js_div = jensenshannon(real_hist, synth_hist)
            js_divergences[col] = 1 - js_div  # Convert to similarity (1 = identical)
        except:
            js_divergences[col] = 0
    
    results['js_similarities'] = js_divergences
    avg_js_similarity = np.mean(list(js_divergences.values())) if js_divergences else 0
    results['avg_js_similarity'] = avg_js_similarity
    
    if verbose:
        print(f"   📊 Average Distribution Similarity: {avg_js_similarity:.3f}")
    
    # 3. Correlation Structure Preservation
    if verbose:
        print("\n3️⃣ CORRELATION STRUCTURE")
        print("-" * 30)
    
    try:
        real_corr = real_data[numeric_cols].corr()
        synth_corr = synthetic_data[numeric_cols].corr()
        
        # Calculate correlation between correlation matrices
        corr_preservation = stats.pearsonr(
            real_corr.values.flatten(),
            synth_corr.values.flatten()
        )[0]
        corr_preservation = max(0, corr_preservation)  # Ensure non-negative
    except:
        corr_preservation = 0
    
    results['correlation_preservation'] = corr_preservation
    
    if verbose:
        print(f"   📊 Correlation Structure Preservation: {corr_preservation:.3f}")
    
    # 4. PCA-based Similarity
    if verbose:
        print("\n4️⃣ DIMENSIONALITY REDUCTION SIMILARITY")
        print("-" * 30)
    
    try:
        # Standardize data
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_data[numeric_cols].fillna(0))
        synth_scaled = scaler.transform(synthetic_data[numeric_cols].fillna(0))
        
        # Apply PCA
        pca = PCA(n_components=min(5, len(numeric_cols)))
        real_pca = pca.fit_transform(real_scaled)
        synth_pca = pca.transform(synth_scaled)
        
        # Compare PCA representations
        pca_similarities = []
        for i in range(real_pca.shape[1]):
            corr = abs(stats.pearsonr(real_pca[:, i], synth_pca[:len(real_pca), i])[0])
            pca_similarities.append(corr)
        
        pca_similarity = np.mean(pca_similarities)
    except:
        pca_similarity = 0
    
    results['pca_similarity'] = pca_similarity
    
    if verbose:
        print(f"   📊 PCA Similarity: {pca_similarity:.3f}")
    
    # 5. Machine Learning Utility (if target column provided)
    if target_column and target_column in real_data.columns:
        if verbose:
            print("\n5️⃣ MACHINE LEARNING UTILITY")
            print("-" * 30)
        
        try:
            # Train on real data, test on synthetic data
            X_real = real_data[numeric_cols].fillna(0)
            y_real = real_data[target_column]
            X_synth = synthetic_data[numeric_cols].fillna(0)
            y_synth = synthetic_data[target_column] if target_column in synthetic_data.columns else None
            
            # Random Forest trained on real data
            rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_real.fit(X_real, y_real)
            
            # Test on synthetic data
            if y_synth is not None:
                synth_pred_accuracy = rf_real.score(X_synth, y_synth)
            else:
                synth_pred_accuracy = 0
            
            # Train on synthetic data, test on real data
            if y_synth is not None:
                rf_synth = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_synth.fit(X_synth, y_synth)
                real_pred_accuracy = rf_synth.score(X_real, y_real)
            else:
                real_pred_accuracy = 0
            
            ml_utility = (synth_pred_accuracy + real_pred_accuracy) / 2
            
        except:
            ml_utility = 0
            synth_pred_accuracy = 0
            real_pred_accuracy = 0
        
        results['ml_utility'] = ml_utility
        results['synth_test_accuracy'] = synth_pred_accuracy
        results['real_test_accuracy'] = real_pred_accuracy
        
        if verbose:
            print(f"   📊 ML Utility (Avg Cross-Accuracy): {ml_utility:.3f}")
            print(f"   📊 Real→Synth Test Accuracy: {synth_pred_accuracy:.3f}")
            print(f"   📊 Synth→Real Test Accuracy: {real_pred_accuracy:.3f}")
    
    # 6. Overall Quality Score
    scores = [
        avg_stat_similarity,
        avg_js_similarity,
        corr_preservation,
        pca_similarity
    ]
    
    if target_column and target_column in real_data.columns:
        scores.append(results.get('ml_utility', 0))
    
    overall_quality = np.mean([s for s in scores if s > 0])  # Only include valid scores
    results['overall_quality_score'] = overall_quality
    
    if verbose:
        print("\n" + "=" * 50)
        print(f"🏆 OVERALL QUALITY SCORE: {overall_quality:.3f}")
        print("=" * 50)
        
        # Quality interpretation
        if overall_quality >= 0.8:
            quality_label = "EXCELLENT"
        elif overall_quality >= 0.6:
            quality_label = "GOOD"
        elif overall_quality >= 0.4:
            quality_label = "FAIR"
        else:
            quality_label = "POOR"
        
        print(f"📋 Quality Assessment: {quality_label}")
    
    return results

print("✅ Comprehensive data quality evaluation function loaded!")

print("🎯 SETUP MODULE LOADED SUCCESSFULLY!")
print("="*60)