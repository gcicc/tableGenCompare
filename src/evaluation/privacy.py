"""
Privacy Metrics Evaluation Functions

This module contains functions for evaluating privacy risks in synthetic data,
including memorization detection and re-identification risk assessment.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def calculate_privacy_metrics(real_data, synthetic_data, target_column=None,
                              n_neighbors=5, memorization_threshold=0.01,
                              verbose=True):
    """
    Calculate comprehensive privacy risk metrics for synthetic data.

    Evaluates multiple privacy dimensions:
    1. Nearest Neighbor Distance Ratio (NNDR): Detects memorization
    2. Distance to Closest Record (DCR): Identifies exact/near-exact copies
    3. Memorization Score: Percentage of synthetic records too close to real data
    4. Re-identification Risk: Potential for linking synthetic to real records

    Parameters:
    -----------
    real_data : pd.DataFrame
        Original dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset to evaluate
    target_column : str, optional
        Target column to exclude from distance calculations
    n_neighbors : int
        Number of neighbors for NNDR calculation (default: 5)
    memorization_threshold : float
        DCR threshold below which a record is considered memorized (default: 0.01)
    verbose : bool
        Print detailed results

    Returns:
    --------
    dict : {
        'nndr_mean': float,              # Average NNDR across synthetic records
        'nndr_std': float,                # Standard deviation of NNDR
        'nndr_distribution': list,        # Full NNDR distribution (for visualization)
        'dcr_mean': float,                # Average distance to closest real record
        'dcr_min': float,                 # Minimum DCR (most suspicious record)
        'dcr_max': float,                 # Maximum DCR (most private record)
        'memorization_score': float,      # Percentage of memorized records (0-1)
        'memorized_count': int,           # Number of memorized records
        'reidentification_risk': float,   # Re-identification risk score (0-1)
        'privacy_score': float            # Overall privacy score (0-1, higher = better)
    }

    Notes:
    ------
    - NNDR < 1.0 indicates potential memorization (synthetic closer to real than to other synthetic)
    - Low DCR values indicate near-exact copies of real records
    - Privacy score combines multiple metrics (higher is better)
    """
    if verbose:
        print(f"\n[PRIVACY] Calculating privacy risk metrics...")

    try:
        # Prepare data - remove target column if specified
        X_real = real_data.drop(columns=[target_column]) if target_column else real_data.copy()
        X_synth = synthetic_data.drop(columns=[target_column]) if target_column else synthetic_data.copy()

        # Ensure common features
        common_features = list(set(X_real.columns) & set(X_synth.columns))
        if len(common_features) == 0:
            if verbose:
                print("[ERROR] No common features for privacy analysis")
            return {
                'nndr_mean': np.nan,
                'nndr_std': np.nan,
                'nndr_distribution': [],
                'dcr_mean': np.nan,
                'dcr_min': np.nan,
                'dcr_max': np.nan,
                'memorization_score': np.nan,
                'memorized_count': 0,
                'reidentification_risk': np.nan,
                'privacy_score': np.nan,
                'error': 'No common features'
            }

        X_real = X_real[common_features]
        X_synth = X_synth[common_features]

        # Select only numeric columns for distance calculations
        numeric_cols = X_real.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            if verbose:
                print("[ERROR] No numeric features for privacy analysis")
            return {
                'nndr_mean': np.nan,
                'nndr_std': np.nan,
                'nndr_distribution': [],
                'dcr_mean': np.nan,
                'dcr_min': np.nan,
                'dcr_max': np.nan,
                'memorization_score': np.nan,
                'memorized_count': 0,
                'reidentification_risk': np.nan,
                'privacy_score': np.nan,
                'error': 'No numeric features'
            }

        X_real_numeric = X_real[numeric_cols].fillna(0)
        X_synth_numeric = X_synth[numeric_cols].fillna(0)

        # Standardize features for fair distance comparison
        scaler = StandardScaler()
        X_real_scaled = scaler.fit_transform(X_real_numeric)
        X_synth_scaled = scaler.transform(X_synth_numeric)

        if verbose:
            print(f"   [PRIVACY] Using {len(numeric_cols)} numeric features")
            print(f"   [PRIVACY] Real: {X_real_scaled.shape[0]} records, Synthetic: {X_synth_scaled.shape[0]} records")

        # 1. Calculate Distance to Closest Record (DCR)
        # For each synthetic record, find distance to nearest real record
        nbrs_real = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_real_scaled)
        dcr_distances, _ = nbrs_real.kneighbors(X_synth_scaled)
        dcr_distances = dcr_distances.flatten()

        dcr_mean = np.mean(dcr_distances)
        dcr_min = np.min(dcr_distances)
        dcr_max = np.max(dcr_distances)

        # 2. Calculate memorization score
        memorized_mask = dcr_distances < memorization_threshold
        memorized_count = int(np.sum(memorized_mask))
        memorization_score = memorized_count / len(dcr_distances) if len(dcr_distances) > 0 else 0

        # 3. Calculate Nearest Neighbor Distance Ratio (NNDR)
        # For each synthetic record:
        # - Find average distance to k nearest synthetic neighbors
        # - Find distance to nearest real record
        # - NNDR = dist_to_real / avg_dist_to_synth
        # NNDR < 1 means synthetic record is closer to real data than to other synthetic data

        nndr_list = []
        if len(X_synth_scaled) > n_neighbors:
            nbrs_synth = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(X_synth_scaled)
            synth_distances, _ = nbrs_synth.kneighbors(X_synth_scaled)

            for i, dcr in enumerate(dcr_distances):
                # Average distance to k nearest synthetic neighbors (exclude self at index 0)
                avg_synth_dist = np.mean(synth_distances[i, 1:])

                # NNDR = distance_to_real / average_distance_to_synthetic
                if avg_synth_dist > 0:
                    nndr = dcr / avg_synth_dist
                    nndr_list.append(nndr)
                else:
                    nndr_list.append(np.nan)

            nndr_list = [x for x in nndr_list if not np.isnan(x)]
            nndr_mean = np.mean(nndr_list) if nndr_list else np.nan
            nndr_std = np.std(nndr_list) if nndr_list else np.nan
        else:
            if verbose:
                print(f"   [WARNING] Not enough synthetic records ({len(X_synth_scaled)}) for NNDR calculation (need > {n_neighbors})")
            nndr_mean = np.nan
            nndr_std = np.nan
            nndr_list = []

        # 4. Calculate re-identification risk
        # Based on percentage of synthetic records with very low DCR
        # Higher percentage = higher risk
        high_risk_threshold = 0.05  # Records with DCR < 0.05 are high re-id risk
        high_risk_count = np.sum(dcr_distances < high_risk_threshold)
        reidentification_risk = high_risk_count / len(dcr_distances) if len(dcr_distances) > 0 else 0

        # 5. Calculate overall privacy score (0-1, higher is better)
        # Combine multiple metrics:
        # - Low memorization score is good (inverted)
        # - High DCR mean is good (normalized)
        # - NNDR > 1 is good (synthetic more similar to each other than to real)
        # - Low re-identification risk is good (inverted)

        privacy_components = []

        # Component 1: Memorization (inverted - lower is better)
        privacy_components.append(1 - memorization_score)

        # Component 2: Average DCR (normalized to 0-1, capped at 1.0)
        dcr_normalized = min(dcr_mean, 1.0) if not np.isnan(dcr_mean) else 0.5
        privacy_components.append(dcr_normalized)

        # Component 3: NNDR (>1 is good, scale to 0-1)
        if not np.isnan(nndr_mean):
            # NNDR around 1.0 means balanced, >1.0 is better
            # Scale: 0.5 -> 0, 1.0 -> 0.5, 2.0 -> 1.0
            nndr_score = min((nndr_mean - 0.5) / 1.5, 1.0) if nndr_mean > 0.5 else 0
            privacy_components.append(max(0, nndr_score))

        # Component 4: Re-identification risk (inverted)
        privacy_components.append(1 - reidentification_risk)

        privacy_score = np.mean(privacy_components)

        if verbose:
            print(f"   [METRIC] DCR Mean: {dcr_mean:.4f} (min: {dcr_min:.4f}, max: {dcr_max:.4f})")
            print(f"   [METRIC] Memorization: {memorization_score:.2%} ({memorized_count}/{len(dcr_distances)} records)")
            if not np.isnan(nndr_mean):
                print(f"   [METRIC] NNDR: {nndr_mean:.4f} (+/- {nndr_std:.4f})")
            print(f"   [METRIC] Re-identification Risk: {reidentification_risk:.2%}")
            print(f"   [METRIC] Overall Privacy Score: {privacy_score:.3f}")

        return {
            'nndr_mean': nndr_mean,
            'nndr_std': nndr_std,
            'nndr_distribution': nndr_list,
            'dcr_mean': dcr_mean,
            'dcr_min': dcr_min,
            'dcr_max': dcr_max,
            'memorization_score': memorization_score,
            'memorized_count': memorized_count,
            'reidentification_risk': reidentification_risk,
            'privacy_score': privacy_score
        }

    except Exception as e:
        if verbose:
            print(f"   [ERROR] Privacy calculation failed: {e}")
        return {
            'nndr_mean': np.nan,
            'nndr_std': np.nan,
            'nndr_distribution': [],
            'dcr_mean': np.nan,
            'dcr_min': np.nan,
            'dcr_max': np.nan,
            'memorization_score': np.nan,
            'memorized_count': 0,
            'reidentification_risk': np.nan,
            'privacy_score': np.nan,
            'error': str(e)
        }
