"""
XAI (Explainability) Metrics for SDAC Evaluation

Compares feature importance and SHAP value distributions between
models trained on real vs synthetic data.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def compute_feature_importance_correlation(real_df, synth_df, target_col, verbose=True):
    """
    Compare Random Forest feature importances trained on real vs synthetic data.

    Returns Pearson correlation between the two importance vectors.
    Higher correlation means synthetic data preserves feature relevance structure.

    Parameters
    ----------
    real_df, synth_df : pd.DataFrame
    target_col : str
    verbose : bool

    Returns
    -------
    dict : {'Feature_Importance_Corr': float}
    """
    try:
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col and c in synth_df.columns]

        if len(feature_cols) < 2:
            return {'Feature_Importance_Corr': np.nan}

        X_real = real_df[feature_cols].fillna(0)
        X_synth = synth_df[feature_cols].fillna(0)

        y_real = real_df[target_col]
        y_synth = synth_df[target_col]

        # Encode targets if needed
        if y_real.dtype == 'object':
            le = LabelEncoder()
            combined = pd.concat([y_real.astype(str), y_synth.astype(str)])
            le.fit(combined)
            y_real = le.transform(y_real.astype(str))
            y_synth = le.transform(y_synth.astype(str))

        rf_real = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_real.fit(X_real, y_real)

        rf_synth = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_synth.fit(X_synth, y_synth)

        imp_real = rf_real.feature_importances_
        imp_synth = rf_synth.feature_importances_

        corr, _ = pearsonr(imp_real, imp_synth)
        corr = max(0.0, corr)  # Clip negative to 0

        if verbose:
            print(f"   [XAI] Feature Importance Correlation: {corr:.4f}")

        return {'Feature_Importance_Corr': float(corr)}

    except Exception as e:
        if verbose:
            print(f"   [XAI] Feature importance failed: {e}")
        return {'Feature_Importance_Corr': np.nan}


def compute_shap_distance(real_df, synth_df, target_col, verbose=True):
    """
    Compare mean absolute SHAP values between real-trained and synth-trained models.

    Returns cosine distance between the two SHAP vectors.
    Lower distance means better preservation of feature explanations.

    Gracefully returns NaN if SHAP is not installed.

    Parameters
    ----------
    real_df, synth_df : pd.DataFrame
    target_col : str
    verbose : bool

    Returns
    -------
    dict : {'SHAP_Distance': float}
    """
    try:
        import shap
    except ImportError:
        if verbose:
            print("   [XAI] SHAP library not installed; skipping SHAP distance")
        return {'SHAP_Distance': np.nan}

    try:
        from scipy.spatial.distance import cosine

        numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col and c in synth_df.columns]

        if len(feature_cols) < 2:
            return {'SHAP_Distance': np.nan}

        X_real = real_df[feature_cols].fillna(0)
        X_synth = synth_df[feature_cols].fillna(0)

        y_real = real_df[target_col]
        y_synth = synth_df[target_col]

        if y_real.dtype == 'object':
            le = LabelEncoder()
            combined = pd.concat([y_real.astype(str), y_synth.astype(str)])
            le.fit(combined)
            y_real = le.transform(y_real.astype(str))
            y_synth = le.transform(y_synth.astype(str))

        # Train models
        rf_real = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
        rf_real.fit(X_real, y_real)

        rf_synth = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
        rf_synth.fit(X_synth, y_synth)

        # Sample for speed
        n_sample = min(100, len(X_real), len(X_synth))
        X_real_sample = X_real.sample(n=n_sample, random_state=42)
        X_synth_sample = X_synth.sample(n=n_sample, random_state=42)

        # SHAP values
        explainer_real = shap.TreeExplainer(rf_real)
        explainer_synth = shap.TreeExplainer(rf_synth)

        shap_real = explainer_real.shap_values(X_real_sample)
        shap_synth = explainer_synth.shap_values(X_synth_sample)

        # Handle multi-class SHAP output (list of arrays)
        if isinstance(shap_real, list):
            shap_real = np.mean([np.abs(s) for s in shap_real], axis=0)
        else:
            shap_real = np.abs(shap_real)

        if isinstance(shap_synth, list):
            shap_synth = np.mean([np.abs(s) for s in shap_synth], axis=0)
        else:
            shap_synth = np.abs(shap_synth)

        # Mean absolute SHAP per feature
        mean_shap_real = np.mean(shap_real, axis=0)
        mean_shap_synth = np.mean(shap_synth, axis=0)

        # Cosine distance
        dist = cosine(mean_shap_real, mean_shap_synth)

        if verbose:
            print(f"   [XAI] SHAP Distance (cosine): {dist:.4f}")

        return {'SHAP_Distance': float(dist)}

    except Exception as e:
        if verbose:
            print(f"   [XAI] SHAP distance failed: {e}")
        return {'SHAP_Distance': np.nan}
