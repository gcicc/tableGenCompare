"""
Hyperparameter Optimization Objective Functions

This module contains enhanced objective functions for Optuna-based hyperparameter
optimization, including early stopping support via pruning.
"""

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import Optuna for pruning (optional dependency)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def enhanced_objective_function_v2(real_data, synthetic_data, target_column,
                                 similarity_weight=0.6, accuracy_weight=0.4,
                                 trial=None):
    """
    Enhanced objective function: 60% similarity + 40% accuracy with Optuna pruning support.

    Evaluates synthetic data quality by combining:
    - Similarity metrics (Earth Mover's Distance, correlation)
    - Accuracy metrics (TRTS framework)

    Supports Optuna early stopping through intermediate value reporting and pruning.

    Parameters:
    -----------
    real_data : pd.DataFrame
        Original dataset
    synthetic_data : pd.DataFrame
        Generated synthetic dataset
    target_column : str
        Name of target column (DYNAMIC - works with any dataset)
    similarity_weight : float
        Weight for similarity component (default 0.6)
    accuracy_weight : float
        Weight for accuracy component (default 0.4)
    trial : optuna.Trial, optional
        Optuna trial object for pruning support. If None, no pruning occurs.

    Returns:
    --------
    tuple : (combined_score, similarity_score, accuracy_score)
        - combined_score: Weighted combination of similarity and accuracy
        - similarity_score: Pure similarity score
        - accuracy_score: Pure accuracy score

    Raises:
    -------
    optuna.TrialPruned : If trial should be pruned based on intermediate values
    """

    print(f"[TARGET] Enhanced objective function using target column: '{target_column}'")

    # Validate target column exists in both datasets
    if target_column not in real_data.columns:
        print(f"[ERROR] Target column '{target_column}' not found in real data")
        return 0.0, 0.0, 0.0

    if target_column not in synthetic_data.columns:
        print(f"[ERROR] Target column '{target_column}' not found in synthetic data")
        return 0.0, 0.0, 0.0

    # ========================================
    # 1. SIMILARITY COMPONENT (60%)
    # ========================================
    similarity_scores = []

    # Univariate similarity using Earth Mover's Distance
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != target_column and col in synthetic_data.columns:
            try:
                real_values = real_data[col]
                synth_values = synthetic_data[col]

                # Handle categorical data that was inverse transformed to strings
                if synth_values.dtype == 'object' or synth_values.apply(lambda x: isinstance(x, str)).any():
                    synth_numeric = pd.to_numeric(synth_values, errors='coerce')

                    if synth_numeric.isna().sum() > len(synth_numeric) * 0.5:
                        # Treat as categorical and re-encode
                        unique_synth_values = synth_values.dropna().unique()
                        value_mapping = {val: idx for idx, val in enumerate(unique_synth_values)}
                        synth_values = synth_values.map(value_mapping).fillna(-1)
                    else:
                        synth_values = synth_numeric.dropna()
                        real_values = real_values.dropna()

                # Ensure we have enough values for EMD calculation
                if len(real_values) == 0 or len(synth_values) == 0:
                    continue

                # Earth Mover's Distance (Wasserstein distance)
                emd_score = wasserstein_distance(real_values, synth_values)

                # Handle nan/inf values from EMD calculation
                if np.isnan(emd_score) or np.isinf(emd_score):
                    # Use fallback similarity based on basic statistics
                    real_mean, synth_mean = np.mean(real_values), np.mean(synth_values)
                    real_std, synth_std = np.std(real_values), np.std(synth_values)
                    mean_diff = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-8)
                    std_diff = abs(real_std - synth_std) / (abs(real_std) + 1e-8)
                    fallback_similarity = 1 / (1 + mean_diff + std_diff)
                    similarity_scores.append(fallback_similarity)
                else:
                    # Convert to similarity (lower EMD = higher similarity)
                    similarity_scores.append(1 / (1 + emd_score))

            except Exception as e:
                print(f"[ERROR] Error calculating EMD for {col}: {e}")
                continue

    # Correlation similarity
    try:
        valid_numeric_cols = []
        for col in numeric_columns:
            if col in synthetic_data.columns and col != target_column:
                col_is_numeric = (
                    pd.api.types.is_numeric_dtype(synthetic_data[col]) and
                    not synthetic_data[col].apply(lambda x: isinstance(x, str)).any()
                )
                if col_is_numeric:
                    valid_numeric_cols.append(col)

        if len(valid_numeric_cols) > 1:
            real_corr = real_data[valid_numeric_cols].corr()
            synth_corr = synthetic_data[valid_numeric_cols].corr()

            # Flatten correlation matrices and compute distance
            real_corr_flat = real_corr.values[np.triu_indices_from(real_corr, k=1)]
            synth_corr_flat = synth_corr.values[np.triu_indices_from(synth_corr, k=1)]

            # Handle nan values in correlation calculation
            real_corr_flat = real_corr_flat[~np.isnan(real_corr_flat)]
            synth_corr_flat = synth_corr_flat[~np.isnan(synth_corr_flat)]

            if len(real_corr_flat) > 0 and len(synth_corr_flat) > 0:
                min_len = min(len(real_corr_flat), len(synth_corr_flat))
                real_corr_flat = real_corr_flat[:min_len]
                synth_corr_flat = synth_corr_flat[:min_len]

                corr_distance = np.mean(np.abs(real_corr_flat - synth_corr_flat))

                if not (np.isnan(corr_distance) or np.isinf(corr_distance)):
                    similarity_scores.append(1 - corr_distance)
                else:
                    similarity_scores.append(0.5)
            else:
                similarity_scores.append(0.5)

    except Exception as e:
        print(f"Warning: Correlation similarity failed: {e}")

    # Robust similarity score aggregation
    if similarity_scores:
        valid_scores = [score for score in similarity_scores if not np.isnan(score)]
        if valid_scores:
            similarity_score = np.mean(valid_scores)
            print(f"[OK] Similarity Analysis: {len(valid_scores)}/{len(similarity_scores)} valid metrics, Average: {similarity_score:.4f}")
            if np.isnan(similarity_score) or np.isinf(similarity_score):
                similarity_score = 0.5
        else:
            similarity_score = 0.5
    else:
        similarity_score = 0.5

    # OPTUNA PRUNING CHECKPOINT 1: After similarity calculation
    if trial is not None and OPTUNA_AVAILABLE:
        trial.report(similarity_score, step=0)
        if trial.should_prune():
            print(f"[PRUNED] Trial pruned after similarity calculation (score: {similarity_score:.4f})")
            raise optuna.TrialPruned()

    # ========================================
    # 2. ACCURACY COMPONENT (40%) - TRTS Framework
    # ========================================
    accuracy_scores = []

    try:
        # Ensure target column exists
        if target_column not in real_data.columns or target_column not in synthetic_data.columns:
            print(f"[ERROR] Target column '{target_column}' missing")
            return similarity_score * similarity_weight, similarity_score, 0.0

        # Prepare features and target
        try:
            X_real = real_data.drop(columns=[target_column])
            y_real = real_data[target_column]
            X_synth = synthetic_data.drop(columns=[target_column])
            y_synth = synthetic_data[target_column]
        except KeyError as ke:
            print(f"[ERROR] KeyError in data preparation: {ke}")
            return similarity_score * similarity_weight, similarity_score, 0.0

        # Ensure consistent label types
        if y_real.dtype != y_synth.dtype:
            if pd.api.types.is_numeric_dtype(y_real):
                try:
                    y_synth = pd.to_numeric(y_synth, errors='coerce')
                except:
                    y_real = y_real.astype(str)
                    y_synth = y_synth.astype(str)
            else:
                y_real = y_real.astype(str)
                y_synth = y_synth.astype(str)

        # Ensure matching features
        common_features = list(set(X_real.columns) & set(X_synth.columns))
        if len(common_features) == 0:
            print("[ERROR] No common features between real and synthetic data")
            return similarity_score * similarity_weight, similarity_score, 0.0

        X_real = X_real[common_features]
        X_synth = X_synth[common_features]

        # Handle mixed data types in features
        for col in common_features:
            if X_synth[col].dtype == 'object':
                try:
                    X_synth[col] = pd.to_numeric(X_synth[col], errors='coerce')
                    if X_synth[col].isna().all():
                        le = LabelEncoder()
                        X_real[col] = le.fit_transform(X_real[col].astype(str))
                        X_synth[col] = le.transform(X_synth[col].astype(str))
                except Exception:
                    X_real = X_real.drop(columns=[col])
                    X_synth = X_synth.drop(columns=[col])

        # Handle missing values
        X_real = X_real.fillna(X_real.median())
        X_synth = X_synth.fillna(X_synth.median())

        if X_real.isna().any().any() or X_synth.isna().any().any():
            X_real = X_real.fillna(0)
            X_synth = X_synth.fillna(0)

        # Ensure sufficient samples
        if len(X_real) < 10 or len(X_synth) < 10:
            print("[WARNING] Insufficient samples for TRTS evaluation")
            return similarity_score * similarity_weight, similarity_score, 0.5

        # TRTS 1: Train on Real, Test on Synthetic
        try:
            rf1 = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            rf1.fit(X_real, y_real)
            pred_synth = rf1.predict(X_synth)
            acc1 = accuracy_score(y_synth, pred_synth)
            accuracy_scores.append(acc1)
        except Exception as e:
            print(f"[ERROR] TRTS (Real->Synthetic) failed: {e}")

        # TRTS 2: Train on Synthetic, Test on Real
        try:
            rf2 = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            rf2.fit(X_synth, y_synth)
            pred_real = rf2.predict(X_real)
            acc2 = accuracy_score(y_real, pred_real)
            accuracy_scores.append(acc2)
        except Exception as e:
            print(f"[ERROR] TRTS (Synthetic->Real) failed: {e}")

    except Exception as e:
        print(f"[ERROR] Accuracy evaluation failed: {e}")

    # Calculate final accuracy score
    accuracy_score_final = np.mean(accuracy_scores) if accuracy_scores else 0.5

    if np.isnan(accuracy_score_final) or np.isinf(accuracy_score_final):
        accuracy_score_final = 0.5

    # OPTUNA PRUNING CHECKPOINT 2: After accuracy calculation
    if trial is not None and OPTUNA_AVAILABLE:
        trial.report(accuracy_score_final, step=1)
        if trial.should_prune():
            print(f"[PRUNED] Trial pruned after accuracy calculation (score: {accuracy_score_final:.4f})")
            raise optuna.TrialPruned()

    # ========================================
    # 3. COMBINED SCORE
    # ========================================
    combined_score = (similarity_score * similarity_weight) + (accuracy_score_final * accuracy_weight)

    if np.isnan(combined_score) or np.isinf(combined_score):
        combined_score = 0.5

    # Print summary
    if accuracy_scores:
        print(f"[OK] TRTS Evaluation: {len(accuracy_scores)} scenarios, Average: {accuracy_score_final:.4f}")
    print(f"[CHART] Combined Score: {combined_score:.4f} (Similarity: {similarity_score:.4f}, Accuracy: {accuracy_score_final:.4f})")

    return combined_score, similarity_score, accuracy_score_final
