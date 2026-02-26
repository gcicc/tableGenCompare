"""
TRTS (Train Real Test Synthetic) Analysis Functions

This module contains functions for comprehensive TRTS analysis,
including expanded classification metrics (15+) and privacy-aware evaluation.
Runs both RandomForest and XGBoost classifiers for each scenario.
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from .privacy import calculate_privacy_metrics

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def calculate_comprehensive_classification_metrics(y_true, y_pred, y_pred_proba=None, verbose=False):
    """
    Calculate 15+ comprehensive classification metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for AUROC/AUPRC calculation
    verbose : bool
        Print debug information

    Returns:
    --------
    dict : Comprehensive metrics including:
        - accuracy: Overall accuracy
        - balanced_accuracy: Balanced accuracy (accounts for class imbalance)
        - precision: Precision (macro average)
        - recall: Recall / Sensitivity (macro average)
        - f1_score: F1 score (macro average)
        - specificity: Specificity (macro average)
        - sensitivity: Sensitivity / Recall (macro average)
        - npv: Negative Predictive Value (macro average)
        - fpr: False Positive Rate (macro average)
        - fnr: False Negative Rate (macro average)
        - mcc: Matthews Correlation Coefficient
        - cohen_kappa: Cohen's Kappa
        - auroc: Area Under ROC Curve (if y_pred_proba provided)
        - auprc: Area Under Precision-Recall Curve (if y_pred_proba provided)
    """
    try:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # Precision, recall, F1 (macro average)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)

        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)

        # Confusion matrix for specificity, NPV, FPR, FNR
        cm = confusion_matrix(y_true, y_pred)
        n_classes = len(np.unique(y_true))

        # Calculate per-class metrics
        specificity_list = []
        npv_list = []
        fpr_list = []
        fnr_list = []
        fdr_list = []  # Phase 2: False Discovery Rate
        for_omission_list = []  # Phase 2: False Omission Rate (avoid 'for' keyword)

        for i in range(n_classes):
            # True/False Positives/Negatives for class i (one-vs-rest)
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            # Specificity = TN / (TN + FP)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_list.append(spec)

            # NPV = TN / (TN + FN)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            npv_list.append(npv)

            # FPR = FP / (FP + TN)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_list.append(fpr)

            # FNR = FN / (FN + TP)
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fnr_list.append(fnr)

            # FDR = FP / (FP + TP) = False Discovery Rate
            fdr_val = fp / (fp + tp) if (fp + tp) > 0 else 0
            fdr_list.append(fdr_val)

            # FOR = FN / (FN + TN) = False Omission Rate
            for_val = fn / (fn + tn) if (fn + tn) > 0 else 0
            for_omission_list.append(for_val)

        # Macro average
        specificity = np.mean(specificity_list)
        npv = np.mean(npv_list)
        fpr = np.mean(fpr_list)
        fnr = np.mean(fnr_list)
        fdr = np.mean(fdr_list)
        false_omission_rate = np.mean(for_omission_list)  # Not 'for' (Python keyword)

        # Phase 3: Additional metrics from dev-plan.md task 3.5

        # Youden's J Statistic = Sensitivity + Specificity - 1
        youden_j = recall + specificity - 1

        # Fowlkes-Mallows Index = sqrt(Precision x Recall)
        fmi = np.sqrt(precision * recall) if (precision > 0 and recall > 0) else 0

        # F-beta scores (beta=0.5 precision-weighted, beta=2.0 recall-weighted)
        beta = 0.5
        denominator = beta**2 * precision + recall
        f_beta_0_5 = (1 + beta**2) * (precision * recall) / denominator if denominator > 0 else 0

        beta = 2
        denominator = beta**2 * precision + recall
        f_beta_2 = (1 + beta**2) * (precision * recall) / denominator if denominator > 0 else 0

        # TPR and TNR aliases (for medical/clinical users)
        tpr = recall  # True Positive Rate = Sensitivity = Recall
        tnr = specificity  # True Negative Rate = Specificity

        # Prevalence and Predicted Positive Rate (using confusion matrix, not label assumptions)
        prevalence_list = []
        predicted_positive_rate_list = []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            total = cm.sum()

            # Prevalence = (TP + FN) / Total (actual positive rate per class)
            prevalence_list.append((tp + fn) / total if total > 0 else 0)

            # Predicted Positive Rate = (TP + FP) / Total (predicted positive rate per class)
            predicted_positive_rate_list.append((tp + fp) / total if total > 0 else 0)

        prevalence = np.mean(prevalence_list)  # Macro average across classes
        predicted_positive_rate = np.mean(predicted_positive_rate_list)

        # AUROC, AUPRC, and Brier Score (if probabilities provided)
        auroc = np.nan
        auprc = np.nan
        brier = np.nan

        if y_pred_proba is not None:
            try:
                if n_classes == 2:
                    # Binary classification
                    auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    auprc = average_precision_score(y_true, y_pred_proba[:, 1])
                    brier = brier_score_loss(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class (use one-vs-rest macro average)
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    auroc = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
                    auprc = average_precision_score(y_true_bin, y_pred_proba, average='macro')

                    # Brier Score for multi-class (handle missing classes gracefully)
                    all_classes = np.arange(n_classes)
                    y_true_bin_full = label_binarize(y_true, classes=all_classes)
                    brier_scores = []
                    for i in range(n_classes):
                        # Only calculate if class exists in test set
                        if y_true_bin_full[:, i].sum() > 0:
                            brier_scores.append(brier_score_loss(y_true_bin_full[:, i], y_pred_proba[:, i]))
                    brier = np.mean(brier_scores) if brier_scores else np.nan
            except Exception as e:
                if verbose:
                    print(f"   [WARNING] Could not calculate AUROC/AUPRC/Brier: {e}")
                auroc = np.nan
                auprc = np.nan
                brier = np.nan

        return {
            # Core metrics
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,

            # Precision/Recall family
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f_beta_0_5': f_beta_0_5,  # Phase 3: F-beta (precision-weighted)
            'f_beta_2': f_beta_2,  # Phase 3: F-beta (recall-weighted)

            # Specificity family
            'specificity': specificity,
            'sensitivity': recall,  # Sensitivity = Recall (synonym)
            'tpr': tpr,  # Phase 3: True Positive Rate = Sensitivity (alias)
            'tnr': tnr,  # Phase 3: True Negative Rate = Specificity (alias)

            # Predictive values
            'npv': npv,
            'ppv': precision,  # Phase 3: Positive Predictive Value = Precision (synonym)

            # Error rates
            'fpr': fpr,
            'fnr': fnr,
            'fdr': fdr,  # Phase 2: False Discovery Rate
            'false_omission_rate': false_omission_rate,  # Phase 2: False Omission Rate

            # Combined metrics
            'mcc': mcc,
            'cohen_kappa': kappa,
            'youden_j': youden_j,  # Phase 3: Youden's J Statistic
            'fmi': fmi,  # Phase 3: Fowlkes-Mallows Index

            # ROC/PR metrics
            'auroc': auroc,
            'auprc': auprc,

            # Probability metrics
            'brier_score': brier,  # Phase 3: Brier Score

            # Population metrics
            'prevalence': prevalence,  # Phase 3: Base rate
            'predicted_positive_rate': predicted_positive_rate  # Phase 3: Predicted positive rate
        }

    except Exception as e:
        if verbose:
            print(f"   [ERROR] Metric calculation failed: {e}")
        # Return NaN for all metrics on error
        return {
            # Core metrics
            'accuracy': np.nan,
            'balanced_accuracy': np.nan,

            # Precision/Recall family
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'f_beta_0_5': np.nan,  # Phase 3
            'f_beta_2': np.nan,  # Phase 3

            # Specificity family
            'specificity': np.nan,
            'sensitivity': np.nan,
            'tpr': np.nan,  # Phase 3
            'tnr': np.nan,  # Phase 3

            # Predictive values
            'npv': np.nan,
            'ppv': np.nan,  # Phase 3

            # Error rates
            'fpr': np.nan,
            'fnr': np.nan,
            'fdr': np.nan,  # Phase 2
            'false_omission_rate': np.nan,  # Phase 2

            # Combined metrics
            'mcc': np.nan,
            'cohen_kappa': np.nan,
            'youden_j': np.nan,  # Phase 3
            'fmi': np.nan,  # Phase 3

            # ROC/PR metrics
            'auroc': np.nan,
            'auprc': np.nan,

            # Probability metrics
            'brier_score': np.nan,  # Phase 3

            # Population metrics
            'prevalence': np.nan,  # Phase 3
            'predicted_positive_rate': np.nan  # Phase 3
        }


def _run_scenario(clf, X_train, y_train, X_test, y_test, scenario_name,
                  store_predictions=False, verbose=True, clf_label="RF"):
    """
    Run a single TRTS scenario with a given classifier.

    Parameters
    ----------
    clf : sklearn-compatible classifier
    X_train, y_train : training data
    X_test, y_test : test data
    scenario_name : str
        Human-readable scenario name
    store_predictions : bool
    verbose : bool
    clf_label : str
        Classifier label for logging (e.g., "RF", "XGB")

    Returns
    -------
    dict : scenario results with metrics, timing, and optional predictions
    """
    start_time = time.time()
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
        elapsed = time.time() - start_time

        metrics = calculate_comprehensive_classification_metrics(
            y_test, y_pred, y_pred_proba, verbose=verbose
        )

        result = {
            'scenario': scenario_name,
            'status': 'success',
            'training_time': elapsed,
            **metrics
        }

        if store_predictions:
            result['predictions'] = {
                'y_true': np.array(y_test).copy(),
                'y_pred': y_pred.copy(),
                'y_pred_proba': y_pred_proba.copy() if y_pred_proba is not None else None,
                'classes': clf.classes_.tolist() if hasattr(clf, 'classes_') else []
            }

        if verbose:
            print(f"   [{clf_label}] Accuracy: {metrics['accuracy']:.4f} | "
                  f"Balanced Acc: {metrics['balanced_accuracy']:.4f} | "
                  f"MCC: {metrics['mcc']:.4f} (Time: {elapsed:.3f}s)")

        return result

    except Exception as e:
        if verbose:
            print(f"   [{clf_label}] {scenario_name} failed: {e}")
        return {'accuracy': 0.0, 'error': str(e), 'status': 'failed'}


def comprehensive_trts_analysis(real_data, synthetic_data, target_column,
                               test_size=0.2, random_state=42, n_estimators=100,
                               verbose=True, store_predictions=False):
    """
    Comprehensive TRTS framework analysis with all four scenarios and 15+ metrics:
    - TRTR: Train Real, Test Real
    - TRTS: Train Real, Test Synthetic
    - TSTR: Train Synthetic, Test Real
    - TSTS: Train Synthetic, Test Synthetic

    Runs both XGBoost (primary) and RandomForest for each scenario.
    XGB results are stored as primary keys (TRTR, TRTS, TSTR, TSTS).
    RF results are stored under '{SCENARIO}_RF' keys.
    Falls back to RF as primary if XGBoost is not available.

    Parameters:
    -----------
    real_data : pd.DataFrame
        Original dataset
    synthetic_data : pd.DataFrame
        Generated synthetic dataset
    target_column : str
        Target column name
    test_size : float
        Test split ratio (default 0.2)
    random_state : int
        Random seed for reproducibility
    n_estimators : int
        Number of trees in RandomForest / XGBoost
    verbose : bool
        Print detailed results
    store_predictions : bool, default=False
        If True, stores y_true, y_pred, y_pred_proba arrays in results for
        ROC/PR/Calibration curve generation. WARNING: Increases memory usage.
        Only enable when visualization generation is needed.

    Returns:
    --------
    dict : Dictionary with detailed TRTS results including 15+ metrics per scenario.
          Primary (XGB if available, else RF) under TRTR/TRTS/TSTR/TSTS keys.
          Secondary under TRTR_RF/TRTS_RF/TSTR_RF/TSTS_RF keys (or _XGB if RF is primary).
          If store_predictions=True, each scenario dict also contains 'predictions' key
          with y_true, y_pred, y_pred_proba, and class labels.
    """
    if verbose:
        print("[ANALYSIS] COMPREHENSIVE TRTS FRAMEWORK ANALYSIS (RF + XGBoost)")
        print("=" * 60)
        if XGBOOST_AVAILABLE:
            print("   [OK] XGBoost available - running dual-classifier analysis")
        else:
            print("   [WARNING] XGBoost not available - running RF only")

    # Prepare data
    X_real = real_data.drop(columns=[target_column])
    y_real = real_data[target_column]
    X_synth = synthetic_data.drop(columns=[target_column])
    y_synth = synthetic_data[target_column]

    if verbose:
        print(f"[CHART] Data shapes:")
        print(f"   - Real: {X_real.shape}, Target unique values: {y_real.nunique()}")
        print(f"   - Synthetic: {X_synth.shape}, Target unique values: {y_synth.nunique()}")

    # Ensure common features
    common_features = list(set(X_real.columns) & set(X_synth.columns))
    if len(common_features) == 0:
        if verbose:
            print("[ERROR] No common features between datasets")
        return {'error': 'No common features'}

    X_real = X_real[common_features]
    X_synth = X_synth[common_features]

    # Handle categorical features with label encoding
    for col in common_features:
        if X_real[col].dtype == 'object' or X_synth[col].dtype == 'object':
            le = LabelEncoder()
            # Fit on combined data to ensure consistent encoding
            combined_values = pd.concat([X_real[col].astype(str), X_synth[col].astype(str)])
            le.fit(combined_values)
            X_real[col] = le.transform(X_real[col].astype(str))
            X_synth[col] = le.transform(X_synth[col].astype(str))

    # Handle target column encoding if needed
    if y_real.dtype == 'object' or y_synth.dtype == 'object':
        le_target = LabelEncoder()
        combined_targets = pd.concat([y_real.astype(str), y_synth.astype(str)])
        le_target.fit(combined_targets)
        y_real = le_target.transform(y_real.astype(str))
        y_synth = le_target.transform(y_synth.astype(str))

    # Fill missing values
    X_real = X_real.fillna(X_real.median())
    X_synth = X_synth.fillna(X_synth.median())

    if verbose:
        print(f"   - Using {len(common_features)} common features")

    # Split real data for TRTR scenario
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=test_size, random_state=random_state, stratify=y_real
    )

    # Split synthetic data for TSTS scenario
    X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
        X_synth, y_synth, test_size=test_size, random_state=random_state, stratify=y_synth
    )

    results = {}

    # Define the 4 TRTS scenarios
    scenarios = [
        ('TRTR', 'Train Real, Test Real',
         X_real_train, y_real_train, X_real_test, y_real_test),
        ('TRTS', 'Train Real, Test Synthetic',
         X_real_train, y_real_train, X_synth_test, y_synth_test),
        ('TSTR', 'Train Synthetic, Test Real',
         X_synth_train, y_synth_train, X_real_test, y_real_test),
        ('TSTS', 'Train Synthetic, Test Synthetic',
         X_synth_train, y_synth_train, X_synth_test, y_synth_test),
    ]

    for idx, (key, name, X_tr, y_tr, X_te, y_te) in enumerate(scenarios, 1):
        if verbose:
            print(f"\n[PROCESS] {idx}. {key} - {name}")

        if XGBOOST_AVAILABLE:
            # --- XGBoost (primary) ---
            xgb = XGBClassifier(
                n_estimators=n_estimators, random_state=random_state, max_depth=6,
                eval_metric='logloss', verbosity=0
            )
            results[key] = _run_scenario(
                xgb, X_tr, y_tr, X_te, y_te, name,
                store_predictions=store_predictions, verbose=verbose, clf_label="XGB"
            )

            # --- RandomForest (secondary) ---
            rf = RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_state, max_depth=10
            )
            results[f'{key}_RF'] = _run_scenario(
                rf, X_tr, y_tr, X_te, y_te, name,
                store_predictions=store_predictions, verbose=verbose, clf_label="RF"
            )
        else:
            # --- RandomForest only (fallback when XGBoost not installed) ---
            rf = RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_state, max_depth=10
            )
            results[key] = _run_scenario(
                rf, X_tr, y_tr, X_te, y_te, name,
                store_predictions=store_predictions, verbose=verbose, clf_label="RF"
            )

    # Calculate summary metrics (primary classifier: XGB if available, else RF)
    primary_scenarios = ['TRTR', 'TRTS', 'TSTR', 'TSTS']
    primary_label = "XGB" if XGBOOST_AVAILABLE else "RF"
    successful_scenarios = [k for k in primary_scenarios if results.get(k, {}).get('status') == 'success']
    if successful_scenarios:
        accuracies = [results[k]['accuracy'] for k in successful_scenarios]
        times = [results[k]['training_time'] for k in successful_scenarios]

        results['summary'] = {
            'average_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'total_training_time': sum(times),
            'successful_scenarios': len(successful_scenarios),
            'baseline_accuracy': results.get('TRTR', {}).get('accuracy', 0.0),
            'primary_classifier': primary_label
        }

        if verbose:
            print(f"\n[CHART] {primary_label} Summary Statistics (Primary):")
            print(f"   - Successful scenarios: {len(successful_scenarios)}/4")
            print(f"   - Average accuracy: {np.mean(accuracies):.4f} (+/-{np.std(accuracies):.4f})")
            print(f"   - Total training time: {sum(times):.3f}s")

    # Secondary classifier summary (RF when XGB is primary)
    if XGBOOST_AVAILABLE:
        rf_scenarios = [f'{k}_RF' for k in primary_scenarios]
        successful_rf = [k for k in rf_scenarios if results.get(k, {}).get('status') == 'success']
        if successful_rf:
            rf_accs = [results[k]['accuracy'] for k in successful_rf]
            rf_times = [results[k]['training_time'] for k in successful_rf]

            results['summary_rf'] = {
                'average_accuracy': np.mean(rf_accs),
                'accuracy_std': np.std(rf_accs),
                'total_training_time': sum(rf_times),
                'successful_scenarios': len(successful_rf),
                'baseline_accuracy': results.get('TRTR_RF', {}).get('accuracy', 0.0)
            }

            if verbose:
                print(f"\n[CHART] RF Summary Statistics (Secondary):")
                print(f"   - Successful scenarios: {len(successful_rf)}/4")
                print(f"   - Average accuracy: {np.mean(rf_accs):.4f} (+/-{np.std(rf_accs):.4f})")
                print(f"   - Total training time: {sum(rf_times):.3f}s")

    # Calculate privacy metrics
    privacy_metrics = calculate_privacy_metrics(
        real_data, synthetic_data, target_column, verbose=verbose
    )
    results['privacy'] = privacy_metrics

    return results
