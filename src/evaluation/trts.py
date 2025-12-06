"""
TRTS (Train Real Test Synthetic) Analysis Functions

This module contains functions for comprehensive TRTS analysis,
including expanded classification metrics (15+) and privacy-aware evaluation.
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
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from .privacy import calculate_privacy_metrics


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

        # Macro average
        specificity = np.mean(specificity_list)
        npv = np.mean(npv_list)
        fpr = np.mean(fpr_list)
        fnr = np.mean(fnr_list)

        # AUROC and AUPRC (if probabilities provided)
        auroc = np.nan
        auprc = np.nan

        if y_pred_proba is not None:
            try:
                if n_classes == 2:
                    # Binary classification
                    auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    auprc = average_precision_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class (use one-vs-rest macro average)
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    auroc = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
                    auprc = average_precision_score(y_true_bin, y_pred_proba, average='macro')
            except Exception as e:
                if verbose:
                    print(f"   [WARNING] Could not calculate AUROC/AUPRC: {e}")
                auroc = np.nan
                auprc = np.nan

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': recall,  # Sensitivity = Recall
            'npv': npv,
            'fpr': fpr,
            'fnr': fnr,
            'mcc': mcc,
            'cohen_kappa': kappa,
            'auroc': auroc,
            'auprc': auprc
        }

    except Exception as e:
        if verbose:
            print(f"   [ERROR] Metric calculation failed: {e}")
        # Return NaN for all metrics on error
        return {
            'accuracy': np.nan,
            'balanced_accuracy': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'specificity': np.nan,
            'sensitivity': np.nan,
            'npv': np.nan,
            'fpr': np.nan,
            'fnr': np.nan,
            'mcc': np.nan,
            'cohen_kappa': np.nan,
            'auroc': np.nan,
            'auprc': np.nan
        }


def comprehensive_trts_analysis(real_data, synthetic_data, target_column,
                               test_size=0.2, random_state=42, n_estimators=100,
                               verbose=True):
    """
    Comprehensive TRTS framework analysis with all four scenarios and 15+ metrics:
    - TRTR: Train Real, Test Real
    - TRTS: Train Real, Test Synthetic
    - TSTR: Train Synthetic, Test Real
    - TSTS: Train Synthetic, Test Synthetic

    Enhanced with comprehensive classification metrics (balanced accuracy, precision,
    recall, F1, specificity, sensitivity, NPV, FPR, FNR, MCC, Cohen's Kappa, AUROC, AUPRC).

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
        Number of trees in RandomForest
    verbose : bool
        Print detailed results

    Returns:
    --------
    dict : Dictionary with detailed TRTS results including 15+ metrics per scenario
    """
    if verbose:
        print("[ANALYSIS] COMPREHENSIVE TRTS FRAMEWORK ANALYSIS (15+ Metrics)")
        print("=" * 60)

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

    # SCENARIO 1: TRTR - Train Real, Test Real (Baseline)
    if verbose:
        print(f"\n[PROCESS] 1. TRTR - Train Real, Test Real (Baseline)")

    start_time = time.time()
    try:
        rf_trtr = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10)
        rf_trtr.fit(X_real_train, y_real_train)
        pred_trtr = rf_trtr.predict(X_real_test)
        pred_proba_trtr = rf_trtr.predict_proba(X_real_test)
        trtr_time = time.time() - start_time

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_classification_metrics(
            y_real_test, pred_trtr, pred_proba_trtr, verbose=verbose
        )

        results['TRTR'] = {
            'scenario': 'Train Real, Test Real',
            'status': 'success',
            'training_time': trtr_time,
            **metrics  # Unpack all 15+ metrics
        }

        if verbose:
            print(f"   [OK] TRTR Accuracy: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | MCC: {metrics['mcc']:.4f} (Time: {trtr_time:.3f}s)")
    except Exception as e:
        results['TRTR'] = {'accuracy': 0.0, 'error': str(e), 'status': 'failed'}
        if verbose:
            print(f"   [ERROR] TRTR failed: {e}")

    # SCENARIO 2: TRTS - Train Real, Test Synthetic
    if verbose:
        print(f"[PROCESS] 2. TRTS - Train Real, Test Synthetic")

    start_time = time.time()
    try:
        rf_trts = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10)
        rf_trts.fit(X_real_train, y_real_train)
        pred_trts = rf_trts.predict(X_synth_test)
        pred_proba_trts = rf_trts.predict_proba(X_synth_test)
        trts_time = time.time() - start_time

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_classification_metrics(
            y_synth_test, pred_trts, pred_proba_trts, verbose=verbose
        )

        results['TRTS'] = {
            'scenario': 'Train Real, Test Synthetic',
            'status': 'success',
            'training_time': trts_time,
            **metrics
        }

        if verbose:
            print(f"   [OK] TRTS Accuracy: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | MCC: {metrics['mcc']:.4f} (Time: {trts_time:.3f}s)")
    except Exception as e:
        results['TRTS'] = {'accuracy': 0.0, 'error': str(e), 'status': 'failed'}
        if verbose:
            print(f"   [ERROR] TRTS failed: {e}")

    # SCENARIO 3: TSTR - Train Synthetic, Test Real
    if verbose:
        print(f"[PROCESS] 3. TSTR - Train Synthetic, Test Real")

    start_time = time.time()
    try:
        rf_tstr = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10)
        rf_tstr.fit(X_synth_train, y_synth_train)
        pred_tstr = rf_tstr.predict(X_real_test)
        pred_proba_tstr = rf_tstr.predict_proba(X_real_test)
        tstr_time = time.time() - start_time

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_classification_metrics(
            y_real_test, pred_tstr, pred_proba_tstr, verbose=verbose
        )

        results['TSTR'] = {
            'scenario': 'Train Synthetic, Test Real',
            'status': 'success',
            'training_time': tstr_time,
            **metrics
        }

        if verbose:
            print(f"   [OK] TSTR Accuracy: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | MCC: {metrics['mcc']:.4f} (Time: {tstr_time:.3f}s)")
    except Exception as e:
        results['TSTR'] = {'accuracy': 0.0, 'error': str(e), 'status': 'failed'}
        if verbose:
            print(f"   [ERROR] TSTR failed: {e}")

    # SCENARIO 4: TSTS - Train Synthetic, Test Synthetic
    if verbose:
        print(f"[PROCESS] 4. TSTS - Train Synthetic, Test Synthetic")

    start_time = time.time()
    try:
        rf_tsts = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10)
        rf_tsts.fit(X_synth_train, y_synth_train)
        pred_tsts = rf_tsts.predict(X_synth_test)
        pred_proba_tsts = rf_tsts.predict_proba(X_synth_test)
        tsts_time = time.time() - start_time

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_classification_metrics(
            y_synth_test, pred_tsts, pred_proba_tsts, verbose=verbose
        )

        results['TSTS'] = {
            'scenario': 'Train Synthetic, Test Synthetic',
            'status': 'success',
            'training_time': tsts_time,
            **metrics
        }

        if verbose:
            print(f"   [OK] TSTS Accuracy: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | MCC: {metrics['mcc']:.4f} (Time: {tsts_time:.3f}s)")
    except Exception as e:
        results['TSTS'] = {'accuracy': 0.0, 'error': str(e), 'status': 'failed'}
        if verbose:
            print(f"   [ERROR] TSTS failed: {e}")

    # Calculate summary metrics
    successful_scenarios = [k for k, v in results.items() if v.get('status') == 'success']
    if successful_scenarios:
        accuracies = [results[k]['accuracy'] for k in successful_scenarios]
        times = [results[k]['training_time'] for k in successful_scenarios]

        results['summary'] = {
            'average_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'total_training_time': sum(times),
            'successful_scenarios': len(successful_scenarios),
            'baseline_accuracy': results.get('TRTR', {}).get('accuracy', 0.0)
        }

        if verbose:
            print(f"\n[CHART] Summary Statistics:")
            print(f"   - Successful scenarios: {len(successful_scenarios)}/4")
            print(f"   - Average accuracy: {np.mean(accuracies):.4f} (+/-{np.std(accuracies):.4f})")
            print(f"   - Total training time: {sum(times):.3f}s")

    # Calculate privacy metrics
    privacy_metrics = calculate_privacy_metrics(
        real_data, synthetic_data, target_column, verbose=verbose
    )
    results['privacy'] = privacy_metrics

    return results
