"""
SDAC Tabular Metrics Orchestrator

Unified entry point that computes all SDAC-aligned metrics across five categories:
Privacy, Fidelity, Utility, Fairness, XAI.

Returns a flat dict suitable for one row in sdac_evaluation_summary.csv.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .privacy import calculate_privacy_metrics, compute_mia_auc
from .fidelity import (
    compute_ks_statistic,
    compute_kl_divergence,
    compute_wasserstein_mean,
    compute_detection_auc,
    compute_contingency_similarity,
)
from .fairness import compute_fairness_metrics
from .xai_metrics import (
    compute_feature_importance_correlation,
    compute_shap_distance,
)


def _compute_jsd_mean(real_df, synth_df, target_col=None):
    """Compute mean Jensen-Shannon similarity across numeric columns."""
    from scipy.spatial.distance import jensenshannon

    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    common = [c for c in numeric_cols if c in synth_df.columns]

    js_scores = []
    for col in common:
        try:
            real_vals = real_df[col].dropna().values
            synth_vals = synth_df[col].dropna().values
            if len(real_vals) == 0 or len(synth_vals) == 0:
                continue
            real_hist, bins = np.histogram(real_vals, bins=30, density=True)
            synth_hist, _ = np.histogram(synth_vals, bins=bins, density=True)
            real_hist = real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
            synth_hist = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist
            js_div = jensenshannon(real_hist, synth_hist)
            js_scores.append(1 - js_div)  # similarity
        except Exception:
            continue
    return np.mean(js_scores) if js_scores else np.nan


def _compute_correlation_similarity(real_df, synth_df, target_col=None):
    """Pearson correlation between flattened mixed-association matrices."""
    from scipy.stats import pearsonr
    from src.evaluation.association import compute_mixed_association_matrix

    common = [c for c in real_df.columns if c in synth_df.columns]
    if len(common) < 2:
        return np.nan
    try:
        real_corr = compute_mixed_association_matrix(real_df[common]).values.flatten()
        synth_corr = compute_mixed_association_matrix(synth_df[common]).values.flatten()
        # Remove NaN pairs before computing correlation
        mask = ~(np.isnan(real_corr) | np.isnan(synth_corr))
        if mask.sum() < 2:
            return np.nan
        r, _ = pearsonr(real_corr[mask], synth_corr[mask])
        return max(0.0, r)
    except Exception:
        return np.nan


def _compute_tstr_multi_model(real_df, synth_df, target_col, n_estimators=100,
                              verbose=True):
    """
    TSTR across RF, LogisticRegression, and XGBoost.
    Returns dict of accuracy, F1, AUROC per classifier + ML Efficacy + SRA.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler, label_binarize

    results = {}

    # Prepare data
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col and c in synth_df.columns]
    if len(feature_cols) < 1:
        return results

    X_real = real_df[feature_cols].fillna(0)
    y_real = real_df[target_col].copy()
    X_synth = synth_df[feature_cols].fillna(0)
    y_synth = synth_df[target_col].copy()

    # Encode target if needed
    if y_real.dtype == 'object' or y_synth.dtype == 'object':
        le = LabelEncoder()
        combined = pd.concat([y_real.astype(str), y_synth.astype(str)])
        le.fit(combined)
        y_real = pd.Series(le.transform(y_real.astype(str)), index=y_real.index)
        y_synth = pd.Series(le.transform(y_synth.astype(str)), index=y_synth.index)

    # Split real data for testing
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )

    # Scaler for LR
    scaler = StandardScaler()
    X_synth_scaled = scaler.fit_transform(X_synth)
    X_real_test_scaled = scaler.transform(X_real_test)

    # Classifiers: train on synthetic, test on real
    classifiers = {
        'RF': RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=10),
        'LR': LogisticRegression(max_iter=500, random_state=42, solver='lbfgs'),
    }

    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        classifiers['XGB'] = XGBClassifier(
            n_estimators=n_estimators, random_state=42, max_depth=6,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
    except ImportError:
        if verbose:
            print("   [UTILITY] XGBoost not available; skipping XGB TSTR")

    n_classes = y_real.nunique()
    model_accuracies_real = {}  # For SRA: trained on real
    model_accuracies_synth = {}  # For SRA: trained on synth

    for name, clf in classifiers.items():
        try:
            # Use scaled data for LR, raw for tree models
            if name == 'LR':
                clf.fit(X_synth_scaled, y_synth)
                y_pred = clf.predict(X_real_test_scaled)
                if hasattr(clf, 'predict_proba'):
                    y_proba = clf.predict_proba(X_real_test_scaled)
                else:
                    y_proba = None
            else:
                clf.fit(X_synth, y_synth)
                y_pred = clf.predict(X_real_test)
                if hasattr(clf, 'predict_proba'):
                    y_proba = clf.predict_proba(X_real_test)
                else:
                    y_proba = None

            acc = accuracy_score(y_real_test, y_pred)
            f1 = f1_score(y_real_test, y_pred, average='macro', zero_division=0)

            results[f'TSTR_Acc_{name}'] = acc
            results[f'TSTR_F1_{name}'] = f1

            # AUROC
            if y_proba is not None:
                try:
                    if n_classes == 2:
                        auroc = roc_auc_score(y_real_test, y_proba[:, 1])
                    else:
                        y_bin = label_binarize(y_real_test, classes=np.arange(n_classes))
                        auroc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
                    results[f'TSTR_AUROC_{name}'] = auroc
                except Exception:
                    results[f'TSTR_AUROC_{name}'] = np.nan
            else:
                results[f'TSTR_AUROC_{name}'] = np.nan

            model_accuracies_synth[name] = acc

            if verbose:
                print(f"   [UTILITY] TSTR {name}: Acc={acc:.4f}, F1={f1:.4f}")

        except Exception as e:
            if verbose:
                print(f"   [UTILITY] TSTR {name} failed: {e}")
            results[f'TSTR_Acc_{name}'] = np.nan
            results[f'TSTR_F1_{name}'] = np.nan
            results[f'TSTR_AUROC_{name}'] = np.nan

    # ML Efficacy: average of TSTR accuracies
    tstr_accs = [v for k, v in results.items() if k.startswith('TSTR_Acc_') and not np.isnan(v)]
    results['ML_Efficacy'] = np.mean(tstr_accs) if tstr_accs else np.nan

    # SRA: Synthetic Ranking Agreement
    # Train same classifiers on REAL data to get real-data rankings, then correlate
    for name, clf_class in [('RF', RandomForestClassifier), ('LR', LogisticRegression)]:
        try:
            if name == 'LR':
                clf_real = clf_class(max_iter=500, random_state=42, solver='lbfgs')
                scaler_real = StandardScaler()
                X_real_train_scaled = scaler_real.fit_transform(X_real_train)
                X_real_test_scaled2 = scaler_real.transform(X_real_test)
                clf_real.fit(X_real_train_scaled, y_real_train)
                acc_real = accuracy_score(y_real_test, clf_real.predict(X_real_test_scaled2))
            else:
                clf_real = clf_class(n_estimators=n_estimators, random_state=42, max_depth=10)
                clf_real.fit(X_real_train, y_real_train)
                acc_real = accuracy_score(y_real_test, clf_real.predict(X_real_test))
            model_accuracies_real[name] = acc_real
        except Exception:
            pass

    try:
        from xgboost import XGBClassifier
        xgb_real = XGBClassifier(
            n_estimators=n_estimators, random_state=42, max_depth=6,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        xgb_real.fit(X_real_train, y_real_train)
        model_accuracies_real['XGB'] = accuracy_score(y_real_test, xgb_real.predict(X_real_test))
    except Exception:
        pass

    # SRA = Spearman correlation of model rankings
    common_models = sorted(set(model_accuracies_real.keys()) & set(model_accuracies_synth.keys()))
    if len(common_models) >= 2:
        from scipy.stats import spearmanr
        real_ranks = [model_accuracies_real[m] for m in common_models]
        synth_ranks = [model_accuracies_synth[m] for m in common_models]
        sra, _ = spearmanr(real_ranks, synth_ranks)
        results['SRA'] = float(sra) if not np.isnan(sra) else np.nan
        if verbose:
            print(f"   [UTILITY] SRA (Synthetic Ranking Agreement): {results['SRA']:.4f}" if not np.isnan(results['SRA']) else "   [UTILITY] SRA: N/A")
    else:
        results['SRA'] = np.nan

    return results


def compute_sdac_tabular_metrics(real_df, synthetic_df, target_col,
                                  protected_col=None, compute_mia=False,
                                  verbose=True):
    """
    Compute all SDAC-aligned tabular metrics.

    Parameters
    ----------
    real_df, synthetic_df : pd.DataFrame
    target_col : str
    protected_col : str or None
        Protected attribute for fairness metrics. Fairness section is NaN if None.
    compute_mia : bool
        Whether to run MIA evaluation (expensive). Default False.
    verbose : bool

    Returns
    -------
    dict : Flat dict with SDAC category-prefixed keys, suitable for a single
           row in sdac_evaluation_summary.csv. Keys like:
           Privacy_DCR, Privacy_NNDR, Privacy_IMS, Privacy_ReID_Risk, Privacy_MIA_AUC,
           Fidelity_JSD, Fidelity_KS, Fidelity_KL, Fidelity_Corr_Sim, Fidelity_WD,
           Fidelity_Detection_AUC, Fidelity_Contingency_Sim,
           Utility_TSTR_Acc_RF, Utility_TSTR_F1_RF, ... Utility_ML_Efficacy, Utility_SRA,
           Fairness_Dem_Parity, Fairness_Eq_Odds, Fairness_Disp_Impact,
           XAI_Feat_Imp_Corr, XAI_SHAP_Dist
    """
    if verbose:
        print("\n[SDAC] Computing SDAC Tabular Metrics")
        print("=" * 60)

    row = {}

    # ===== PRIVACY =====
    if verbose:
        print("\n[SDAC] === PRIVACY ===")
    privacy = calculate_privacy_metrics(
        real_df, synthetic_df, target_column=target_col, verbose=verbose
    )
    row['Privacy_DCR'] = privacy.get('dcr_mean', np.nan)
    row['Privacy_NNDR'] = privacy.get('nndr_mean', np.nan)
    row['Privacy_IMS'] = privacy.get('memorization_score', np.nan)
    row['Privacy_ReID_Risk'] = privacy.get('reidentification_risk', np.nan)
    row['Privacy_Score'] = privacy.get('privacy_score', np.nan)

    if compute_mia:
        if verbose:
            print("   [SDAC] Running MIA evaluation...")
        mia = compute_mia_auc(real_df, synthetic_df, target_col=target_col, verbose=verbose)
        row['Privacy_MIA_AUC'] = mia.get('MIA_AUC', np.nan)
    else:
        row['Privacy_MIA_AUC'] = np.nan

    # ===== FIDELITY =====
    if verbose:
        print("\n[SDAC] === FIDELITY ===")

    # JSD
    jsd = _compute_jsd_mean(real_df, synthetic_df, target_col)
    row['Fidelity_JSD'] = jsd
    if verbose:
        print(f"   [FIDELITY] JSD Mean Similarity: {jsd:.4f}" if not np.isnan(jsd) else "   [FIDELITY] JSD: N/A")

    # KS
    ks = compute_ks_statistic(real_df, synthetic_df, target_col)
    row['Fidelity_KS'] = ks['KS_Mean']
    if verbose:
        print(f"   [FIDELITY] KS Mean: {ks['KS_Mean']:.4f}" if not np.isnan(ks['KS_Mean']) else "   [FIDELITY] KS: N/A")

    # KL
    kl = compute_kl_divergence(real_df, synthetic_df, target_col)
    row['Fidelity_KL'] = kl['KL_Mean']
    if verbose:
        print(f"   [FIDELITY] KL Mean: {kl['KL_Mean']:.4f}" if not np.isnan(kl['KL_Mean']) else "   [FIDELITY] KL: N/A")

    # Correlation Similarity
    corr_sim = _compute_correlation_similarity(real_df, synthetic_df, target_col)
    row['Fidelity_Corr_Sim'] = corr_sim
    if verbose:
        print(f"   [FIDELITY] Correlation Similarity: {corr_sim:.4f}" if not np.isnan(corr_sim) else "   [FIDELITY] Corr Sim: N/A")

    # Wasserstein Distance
    wd = compute_wasserstein_mean(real_df, synthetic_df, target_col)
    row['Fidelity_WD'] = wd['WD_Mean']
    if verbose:
        print(f"   [FIDELITY] Wasserstein Mean: {wd['WD_Mean']:.4f}" if not np.isnan(wd['WD_Mean']) else "   [FIDELITY] WD: N/A")

    # Detection AUC
    det = compute_detection_auc(real_df, synthetic_df, target_col)
    row['Fidelity_Detection_AUC'] = det['Detection_AUC']
    if verbose:
        print(f"   [FIDELITY] Detection AUC: {det['Detection_AUC']:.4f}" if not np.isnan(det['Detection_AUC']) else "   [FIDELITY] Detection AUC: N/A")

    # Contingency Similarity
    cont = compute_contingency_similarity(real_df, synthetic_df, target_col)
    row['Fidelity_Contingency_Sim'] = cont['Contingency_Sim_Mean']

    # ===== UTILITY =====
    if verbose:
        print("\n[SDAC] === UTILITY ===")
    utility = _compute_tstr_multi_model(real_df, synthetic_df, target_col, verbose=verbose)
    for k, v in utility.items():
        row[f'Utility_{k}'] = v

    # ===== FAIRNESS =====
    if verbose:
        print("\n[SDAC] === FAIRNESS ===")
    fairness = compute_fairness_metrics(
        real_df, synthetic_df, target_col, protected_col=protected_col, verbose=verbose
    )
    row['Fairness_Dem_Parity'] = fairness.get('Demographic_Parity_Diff', np.nan)
    row['Fairness_Eq_Odds'] = fairness.get('Equalized_Odds_Diff', np.nan)
    row['Fairness_Disp_Impact'] = fairness.get('Disparate_Impact', np.nan)

    # ===== XAI =====
    if verbose:
        print("\n[SDAC] === XAI ===")
    fi = compute_feature_importance_correlation(real_df, synthetic_df, target_col, verbose=verbose)
    row['XAI_Feat_Imp_Corr'] = fi.get('Feature_Importance_Corr', np.nan)

    shap_d = compute_shap_distance(real_df, synthetic_df, target_col, verbose=verbose)
    row['XAI_SHAP_Dist'] = shap_d.get('SHAP_Distance', np.nan)

    if verbose:
        print("\n[SDAC] Metrics computation complete")
        n_valid = sum(1 for v in row.values() if not (isinstance(v, float) and np.isnan(v)))
        print(f"   [SDAC] {n_valid}/{len(row)} metrics populated")

    return row
