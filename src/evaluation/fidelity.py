"""
Fidelity Metrics for SDAC Evaluation

Statistical distribution and structure metrics comparing real and synthetic data.
Covers: KS Statistic, KL Divergence, Wasserstein Distance, Detection AUC,
Contingency Similarity.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data.eda import mixed_association_matrix


def compute_ks_statistic(real_df, synth_df, target_col=None):
    """
    Per-column Kolmogorov-Smirnov statistic, return mean across numeric columns.

    Parameters
    ----------
    real_df, synth_df : pd.DataFrame
    target_col : str, optional
        Column to exclude from computation.

    Returns
    -------
    dict : {'KS_Mean': float, 'KS_Per_Column': dict}
    """
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    common = [c for c in numeric_cols if c in synth_df.columns]
    if not common:
        return {'KS_Mean': np.nan, 'KS_Per_Column': {}}

    ks_scores = {}
    for col in common:
        try:
            stat, _ = stats.ks_2samp(
                real_df[col].dropna().values,
                synth_df[col].dropna().values
            )
            ks_scores[col] = stat
        except Exception:
            ks_scores[col] = np.nan

    valid = [v for v in ks_scores.values() if not np.isnan(v)]
    return {
        'KS_Mean': np.mean(valid) if valid else np.nan,
        'KS_Per_Column': ks_scores
    }


def compute_kl_divergence(real_df, synth_df, target_col=None, n_bins=30):
    """
    Per-column KL divergence via histogram binning, return mean.

    Uses add-one (Laplace) smoothing to avoid division by zero.

    Parameters
    ----------
    real_df, synth_df : pd.DataFrame
    target_col : str, optional
    n_bins : int

    Returns
    -------
    dict : {'KL_Mean': float, 'KL_Per_Column': dict}
    """
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    common = [c for c in numeric_cols if c in synth_df.columns]
    if not common:
        return {'KL_Mean': np.nan, 'KL_Per_Column': {}}

    kl_scores = {}
    for col in common:
        try:
            real_vals = real_df[col].dropna().values
            synth_vals = synth_df[col].dropna().values
            if len(real_vals) == 0 or len(synth_vals) == 0:
                kl_scores[col] = np.nan
                continue

            # Common bin edges
            combined = np.concatenate([real_vals, synth_vals])
            bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

            real_hist, _ = np.histogram(real_vals, bins=bins)
            synth_hist, _ = np.histogram(synth_vals, bins=bins)

            # Laplace smoothing
            real_hist = (real_hist + 1).astype(float)
            synth_hist = (synth_hist + 1).astype(float)

            real_prob = real_hist / real_hist.sum()
            synth_prob = synth_hist / synth_hist.sum()

            kl = stats.entropy(real_prob, synth_prob)
            kl_scores[col] = kl
        except Exception:
            kl_scores[col] = np.nan

    valid = [v for v in kl_scores.values() if not np.isnan(v)]
    return {
        'KL_Mean': np.mean(valid) if valid else np.nan,
        'KL_Per_Column': kl_scores
    }


def compute_wasserstein_mean(real_df, synth_df, target_col=None):
    """
    Per-column Earth Mover's Distance (Wasserstein-1), return mean.

    Parameters
    ----------
    real_df, synth_df : pd.DataFrame
    target_col : str, optional

    Returns
    -------
    dict : {'WD_Mean': float, 'WD_Per_Column': dict}
    """
    from scipy.stats import wasserstein_distance

    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    common = [c for c in numeric_cols if c in synth_df.columns]
    if not common:
        return {'WD_Mean': np.nan, 'WD_Per_Column': {}}

    wd_scores = {}
    for col in common:
        try:
            real_vals = real_df[col].dropna().values
            synth_vals = synth_df[col].dropna().values
            if len(real_vals) == 0 or len(synth_vals) == 0:
                wd_scores[col] = np.nan
                continue
            # Normalize to [0,1] for comparability across columns
            combined_min = min(real_vals.min(), synth_vals.min())
            combined_max = max(real_vals.max(), synth_vals.max())
            rng = combined_max - combined_min
            if rng > 0:
                real_norm = (real_vals - combined_min) / rng
                synth_norm = (synth_vals - combined_min) / rng
            else:
                real_norm = real_vals
                synth_norm = synth_vals
            wd_scores[col] = wasserstein_distance(real_norm, synth_norm)
        except Exception:
            wd_scores[col] = np.nan

    valid = [v for v in wd_scores.values() if not np.isnan(v)]
    return {
        'WD_Mean': np.mean(valid) if valid else np.nan,
        'WD_Per_Column': wd_scores
    }


def compute_detection_auc(real_df, synth_df, target_col=None):
    """
    Logistic Detection AUC: train a classifier to distinguish real vs synthetic.

    Lower AUC (closer to 0.5) means synthetic data is harder to distinguish
    from real data, which indicates better fidelity.

    Parameters
    ----------
    real_df, synth_df : pd.DataFrame
    target_col : str, optional

    Returns
    -------
    dict : {'Detection_AUC': float}
    """
    try:
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        common = [c for c in numeric_cols if c in synth_df.columns]

        if len(common) < 1:
            return {'Detection_AUC': np.nan}

        X_real = real_df[common].fillna(0).values
        X_synth = synth_df[common].fillna(0).values

        # Balance the datasets
        n = min(len(X_real), len(X_synth))
        if n < 20:
            return {'Detection_AUC': np.nan}

        X_real_sub = X_real[:n]
        X_synth_sub = X_synth[:n]

        X = np.vstack([X_real_sub, X_synth_sub])
        y = np.array([0] * n + [1] * n)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
        return {'Detection_AUC': float(np.mean(scores))}
    except Exception:
        return {'Detection_AUC': np.nan}


def compute_contingency_similarity(real_df, synth_df, target_col=None):
    """
    Contingency table similarity for categorical column pairs.

    For each categorical column, computes the Total Variation Distance
    between category frequency distributions.

    Parameters
    ----------
    real_df, synth_df : pd.DataFrame
    target_col : str, optional

    Returns
    -------
    dict : {'Contingency_Sim_Mean': float, 'Contingency_Per_Column': dict}
    """
    cat_cols = real_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)
    common = [c for c in cat_cols if c in synth_df.columns]

    if not common:
        return {'Contingency_Sim_Mean': np.nan, 'Contingency_Per_Column': {}}

    sim_scores = {}
    for col in common:
        try:
            real_freq = real_df[col].value_counts(normalize=True)
            synth_freq = synth_df[col].value_counts(normalize=True)

            # Union of all categories
            all_cats = set(real_freq.index) | set(synth_freq.index)
            real_vec = np.array([real_freq.get(c, 0) for c in all_cats])
            synth_vec = np.array([synth_freq.get(c, 0) for c in all_cats])

            # Total Variation Distance, converted to similarity
            tvd = 0.5 * np.sum(np.abs(real_vec - synth_vec))
            sim_scores[col] = 1.0 - tvd
        except Exception:
            sim_scores[col] = np.nan

    valid = [v for v in sim_scores.values() if not np.isnan(v)]
    return {
        'Contingency_Sim_Mean': np.mean(valid) if valid else np.nan,
        'Contingency_Per_Column': sim_scores
    }


# ---- association preservation (table-level bivariate) ---------------------
# Copied verbatim from multi-table-gen-compare/src/evaluation/fidelity.py:175
# (keep in sync — paired-PR rule).

def association_preservation(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    *,
    sample_rows: int = 5_000,
    random_state: int = 42,
    signal_threshold: float = 0.3,
) -> float:
    """1 - mean |A_real - A_synth| over pairs where |A_real| > signal_threshold.

    Restricting to real-signal pairs (default > 0.3) prevents the metric
    from being dominated by noise-level off-diagonal cells where the real
    association is ~0 and any synth jitter reads as "preservation loss" —
    which rewarded marginal-independent samplers artificially. Now we only
    measure preservation on pairs where there's actually something to
    preserve.

    Bounded in [0, 1]; 1 = identical association strength across all
    above-threshold pairs. NaN if no pair clears ``signal_threshold``.
    """
    if real.shape[0] < 10 or synth.shape[0] < 10:
        return float("nan")
    common = [c for c in real.columns if c in synth.columns]
    if len(common) < 2:
        return float("nan")
    r = real[common]
    s = synth[common]
    if len(r) > sample_rows:
        r = r.sample(sample_rows, random_state=random_state)
    if len(s) > sample_rows:
        s = s.sample(sample_rows, random_state=random_state)
    try:
        A_r = mixed_association_matrix(r).reindex(index=common, columns=common)
        A_s = mixed_association_matrix(s).reindex(index=common, columns=common)
    except Exception:  # noqa: BLE001
        return float("nan")
    mask = ~np.eye(len(common), dtype=bool)
    abs_real = np.abs(A_r.to_numpy())
    abs_synth = np.abs(A_s.to_numpy())
    signal_mask = mask & (abs_real > signal_threshold) & np.isfinite(abs_real)
    if not signal_mask.any():
        return float("nan")
    diffs = np.abs(abs_real[signal_mask] - abs_synth[signal_mask])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    return float(1.0 - diffs.mean())
