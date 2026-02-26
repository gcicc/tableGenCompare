"""
Fairness Metrics for SDAC Evaluation

Measures demographic parity, equalized odds, and disparate impact
between real and synthetic data with respect to a protected attribute.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def compute_fairness_metrics(real_df, synth_df, target_col, protected_col=None,
                             verbose=True):
    """
    Compute fairness metrics comparing real and synthetic data.

    Returns empty dict if protected_col is None or not present in data.

    Parameters
    ----------
    real_df, synth_df : pd.DataFrame
    target_col : str
        Target/outcome column.
    protected_col : str or None
        Protected attribute column (e.g., 'sex', 'race').
    verbose : bool

    Returns
    -------
    dict : {
        'Demographic_Parity_Diff': float,   # |P(Y=1|A=0) - P(Y=1|A=1)| on synth
        'Equalized_Odds_Diff': float,        # max |TPR_diff|, |FPR_diff|
        'Disparate_Impact': float,           # min(P(Y=1|A=0), P(Y=1|A=1)) / max(...)
    }
    """
    if protected_col is None:
        if verbose:
            print("[FAIRNESS] No protected_col specified; skipping fairness metrics")
        return {}

    for df, label in [(real_df, 'real'), (synth_df, 'synthetic')]:
        if protected_col not in df.columns:
            if verbose:
                print(f"[FAIRNESS] protected_col '{protected_col}' not found in {label} data")
            return {}
        if target_col not in df.columns:
            if verbose:
                print(f"[FAIRNESS] target_col '{target_col}' not found in {label} data")
            return {}

    try:
        result = {}

        # --- Prepare synthetic data for fairness analysis ---
        synth = synth_df.copy()
        y_synth = synth[target_col]
        a_synth = synth[protected_col]

        # Binarize protected attribute (most common vs rest)
        if a_synth.nunique() > 2:
            majority = a_synth.mode().iloc[0]
            a_binary = (a_synth == majority).astype(int)
        else:
            le = LabelEncoder()
            a_binary = pd.Series(le.fit_transform(a_synth.astype(str)), index=a_synth.index)

        # Binarize target (positive class = 1)
        if y_synth.nunique() > 2:
            majority_target = y_synth.mode().iloc[0]
            y_binary = (y_synth == majority_target).astype(int)
        else:
            le_t = LabelEncoder()
            y_binary = pd.Series(le_t.fit_transform(y_synth.astype(str)), index=y_synth.index)

        # --- 1. Demographic Parity Difference ---
        # |P(Y_hat=1 | A=0) - P(Y_hat=1 | A=1)|
        # Using observed outcome rates in synthetic data as proxy
        group_0_mask = a_binary == 0
        group_1_mask = a_binary == 1

        rate_0 = y_binary[group_0_mask].mean() if group_0_mask.sum() > 0 else np.nan
        rate_1 = y_binary[group_1_mask].mean() if group_1_mask.sum() > 0 else np.nan

        if not np.isnan(rate_0) and not np.isnan(rate_1):
            result['Demographic_Parity_Diff'] = abs(rate_0 - rate_1)
        else:
            result['Demographic_Parity_Diff'] = np.nan

        # --- 2. Disparate Impact Ratio ---
        # min(rate_0, rate_1) / max(rate_0, rate_1)
        if not np.isnan(rate_0) and not np.isnan(rate_1) and max(rate_0, rate_1) > 0:
            result['Disparate_Impact'] = min(rate_0, rate_1) / max(rate_0, rate_1)
        else:
            result['Disparate_Impact'] = np.nan

        # --- 3. Equalized Odds Difference ---
        # Train classifier on synthetic, predict on synthetic, compare TPR/FPR by group
        numeric_cols = synth.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col and c != protected_col]
        common_features = [c for c in feature_cols if c in real_df.columns]

        if len(common_features) >= 1 and len(synth) >= 30:
            X = synth[common_features].fillna(0).values
            y = y_binary.values

            clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
            clf.fit(X, y)
            y_pred = clf.predict(X)

            # TPR and FPR per group
            tpr_list = []
            fpr_list = []
            for mask in [group_0_mask, group_1_mask]:
                y_g = y[mask.values]
                yp_g = y_pred[mask.values]
                tp = ((yp_g == 1) & (y_g == 1)).sum()
                fn = ((yp_g == 0) & (y_g == 1)).sum()
                fp = ((yp_g == 1) & (y_g == 0)).sum()
                tn = ((yp_g == 0) & (y_g == 0)).sum()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
                tpr_list.append(tpr)
                fpr_list.append(fpr_val)

            tpr_diff = abs(tpr_list[0] - tpr_list[1])
            fpr_diff = abs(fpr_list[0] - fpr_list[1])
            result['Equalized_Odds_Diff'] = max(tpr_diff, fpr_diff)
        else:
            result['Equalized_Odds_Diff'] = np.nan

        if verbose:
            for k, v in result.items():
                print(f"   [FAIRNESS] {k}: {v:.4f}" if not np.isnan(v) else f"   [FAIRNESS] {k}: N/A")

        return result

    except Exception as e:
        if verbose:
            print(f"[FAIRNESS] Error computing fairness metrics: {e}")
        return {
            'Demographic_Parity_Diff': np.nan,
            'Equalized_Odds_Diff': np.nan,
            'Disparate_Impact': np.nan,
        }
