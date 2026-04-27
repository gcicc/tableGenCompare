"""
SDMetrics Tabular Report

Independent third-party evaluation of synthetic data using the public
SDMetrics library (https://docs.sdv.dev/sdmetrics). Mirrors the row shape of
``compute_sdac_tabular_metrics`` so the result can sit alongside the SDAC
table in Section 5.

Each metric is computed in a defensive try/except — failures yield NaN so a
single bad column or type mismatch does not break the whole table.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


SDMETRICS_CATALOG: list[dict[str, Any]] = [
    {"metric": "Coverage_Range",          "category": "Coverage",      "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Coverage_Category",       "category": "Coverage",      "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Validity_Boundary",       "category": "Validity",      "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Validity_CategoryAdherence", "category": "Validity",   "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Validity_MissingValueSim","category": "Validity",      "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Shape_KSComplement",      "category": "Shapes",        "sdac_overlap": True,  "sdac_counterpart": "Fidelity_KS"},
    {"metric": "Shape_TVComplement",      "category": "Shapes",        "sdac_overlap": True,  "sdac_counterpart": "Fidelity_JSD (partial)"},
    {"metric": "Shape_StatisticSim_Mean", "category": "Shapes",        "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Shape_StatisticSim_Median","category": "Shapes",       "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Shape_StatisticSim_Std",  "category": "Shapes",        "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Pair_CorrelationSim",     "category": "Pair-Trends",   "sdac_overlap": True,  "sdac_counterpart": "Fidelity_Corr_Sim"},
    {"metric": "Pair_ContingencySim",     "category": "Pair-Trends",   "sdac_overlap": True,  "sdac_counterpart": "Fidelity_Contingency_Sim"},
    {"metric": "Pair_ContinuousKL",       "category": "Pair-Trends",   "sdac_overlap": True,  "sdac_counterpart": "Fidelity_KL (partial)"},
    {"metric": "Detection_Logistic",      "category": "Detection",     "sdac_overlap": True,  "sdac_counterpart": "Fidelity_Detection_AUC"},
    {"metric": "Detection_SVC",           "category": "Detection",     "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Privacy_NewRowSynthesis", "category": "Privacy",       "sdac_overlap": True,  "sdac_counterpart": "Privacy_IMS (partial)"},
    {"metric": "Privacy_DCRBaseline",     "category": "Privacy",       "sdac_overlap": True,  "sdac_counterpart": "Privacy_DCR"},
    {"metric": "Privacy_DCROverfitting",  "category": "Privacy",       "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Privacy_Disclosure",      "category": "Privacy",       "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Privacy_CategoricalCAP",  "category": "Privacy",       "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Privacy_NumericalLR",     "category": "Privacy",       "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "MLEff_BinaryDT",          "category": "ML-Efficacy",   "sdac_overlap": True,  "sdac_counterpart": "Utility_TSTR_Acc_RF (different model)"},
    {"metric": "MLEff_BinaryLR",          "category": "ML-Efficacy",   "sdac_overlap": True,  "sdac_counterpart": "Utility_TSTR_Acc_LR"},
    {"metric": "MLEff_BinaryAdaBoost",    "category": "ML-Efficacy",   "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Quality_Overall",         "category": "Aggregate",     "sdac_overlap": False, "sdac_counterpart": ""},
    {"metric": "Diagnostic_Overall",      "category": "Aggregate",     "sdac_overlap": False, "sdac_counterpart": ""},
]


def get_sdmetrics_catalog_df() -> pd.DataFrame:
    """Return the metric catalog as a DataFrame."""
    return pd.DataFrame(SDMETRICS_CATALOG)


def _build_metadata(df: pd.DataFrame) -> dict:
    """Build SDMetrics single-table metadata dict from a DataFrame."""
    columns = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            sdtype = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            sdtype = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            sdtype = "numerical"
        else:
            sdtype = "categorical"
        columns[col] = {"sdtype": sdtype}
    return {"columns": columns}


def _safe(fn, *args, default=np.nan, **kwargs):
    """Run a metric and return NaN on any failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = fn(*args, **kwargs)
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _mean_per_column(metric_cls, real_df, synth_df, columns):
    """Apply a single-column metric over a list of columns and return the mean."""
    scores = []
    for col in columns:
        if col not in synth_df.columns:
            continue
        score = _safe(metric_cls.compute, real_df[col], synth_df[col])
        if not np.isnan(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else np.nan


def _column_pair_mean(metric_cls, real_df, synth_df, columns, **kwargs):
    """Apply a column-pair metric to the full subset and return the score."""
    if len(columns) < 2:
        return np.nan
    real_sub = real_df[columns]
    synth_sub = synth_df[[c for c in columns if c in synth_df.columns]]
    if synth_sub.shape[1] < 2:
        return np.nan
    return _safe(metric_cls.compute, real_sub, synth_sub, **kwargs)


def compute_sdmetrics_tabular_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    target_col: str,
    protected_col: str | None = None,
    metadata: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Compute the SDMetrics single-table cross-check.

    Parameters
    ----------
    real_df, synthetic_df : pd.DataFrame
    target_col : str
        Used for ML-Efficacy metrics. NaN if target type isn't binary.
    protected_col : str or None
        Currently unused; kept for signature parity with SDAC.
    metadata : dict or None
        SDMetrics single-table metadata. Auto-detected from real_df if None.
    verbose : bool

    Returns
    -------
    dict : Flat dict with keys matching ``SDMETRICS_CATALOG``.
    """
    if verbose:
        print("\n[SDMetrics] Computing SDMetrics single-table report")
        print("=" * 60)

    try:
        from sdmetrics.single_column import (
            BoundaryAdherence,
            CategoryAdherence,
            CategoryCoverage,
            KSComplement,
            MissingValueSimilarity,
            RangeCoverage,
            StatisticSimilarity,
            TVComplement,
        )
        from sdmetrics.column_pairs import (
            ContingencySimilarity,
            ContinuousKLDivergence,
            CorrelationSimilarity,
        )
        from sdmetrics.single_table import (
            BinaryAdaBoostClassifier,
            BinaryDecisionTreeClassifier,
            BinaryLogisticRegression,
            CategoricalCAP,
            DCRBaselineProtection,
            DCROverfittingProtection,
            DisclosureProtection,
            LogisticDetection,
            NewRowSynthesis,
            NumericalLR,
            SVCDetection,
        )
        from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
    except ImportError as e:
        raise ImportError(
            "sdmetrics is required for compute_sdmetrics_tabular_report. "
            "Install with: pip install sdmetrics"
        ) from e

    if metadata is None:
        metadata = _build_metadata(real_df)

    numeric_cols = [
        c for c, m in metadata["columns"].items()
        if m["sdtype"] == "numerical" and c in synthetic_df.columns
    ]
    categorical_cols = [
        c for c, m in metadata["columns"].items()
        if m["sdtype"] in ("categorical", "boolean") and c in synthetic_df.columns
    ]
    all_cols = [c for c in real_df.columns if c in synthetic_df.columns]

    row: dict[str, float] = {}

    # ---- Coverage ----
    if verbose:
        print("[SDMetrics] === COVERAGE ===")
    row["Coverage_Range"] = _mean_per_column(RangeCoverage, real_df, synthetic_df, numeric_cols)
    row["Coverage_Category"] = _mean_per_column(CategoryCoverage, real_df, synthetic_df, categorical_cols)

    # ---- Validity ----
    if verbose:
        print("[SDMetrics] === VALIDITY ===")
    row["Validity_Boundary"] = _mean_per_column(BoundaryAdherence, real_df, synthetic_df, numeric_cols)
    row["Validity_CategoryAdherence"] = _mean_per_column(CategoryAdherence, real_df, synthetic_df, categorical_cols)
    row["Validity_MissingValueSim"] = _mean_per_column(MissingValueSimilarity, real_df, synthetic_df, all_cols)

    # ---- Column Shapes ----
    if verbose:
        print("[SDMetrics] === COLUMN SHAPES ===")
    row["Shape_KSComplement"] = _mean_per_column(KSComplement, real_df, synthetic_df, numeric_cols)
    row["Shape_TVComplement"] = _mean_per_column(TVComplement, real_df, synthetic_df, categorical_cols)
    for stat in ("mean", "median", "std"):
        scores = []
        for col in numeric_cols:
            s = _safe(StatisticSimilarity.compute, real_df[col], synthetic_df[col], statistic=stat)
            if not np.isnan(s):
                scores.append(s)
        row[f"Shape_StatisticSim_{stat.capitalize()}"] = float(np.mean(scores)) if scores else np.nan

    # ---- Column Pair Trends ----
    if verbose:
        print("[SDMetrics] === COLUMN PAIR TRENDS ===")
    # CorrelationSimilarity / ContingencySimilarity / ContinuousKL handle their own column pairs;
    # pass full numeric / categorical subsets.
    if len(numeric_cols) >= 2:
        scores = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                pair_real = real_df[[c1, c2]]
                pair_synth = synthetic_df[[c1, c2]]
                s = _safe(CorrelationSimilarity.compute, pair_real, pair_synth, coefficient="Pearson")
                if not np.isnan(s):
                    scores.append(s)
        row["Pair_CorrelationSim"] = float(np.mean(scores)) if scores else np.nan
    else:
        row["Pair_CorrelationSim"] = np.nan

    if len(categorical_cols) >= 2:
        scores = []
        for i, c1 in enumerate(categorical_cols):
            for c2 in categorical_cols[i + 1:]:
                pair_real = real_df[[c1, c2]]
                pair_synth = synthetic_df[[c1, c2]]
                s = _safe(ContingencySimilarity.compute, pair_real, pair_synth)
                if not np.isnan(s):
                    scores.append(s)
        row["Pair_ContingencySim"] = float(np.mean(scores)) if scores else np.nan
    else:
        row["Pair_ContingencySim"] = np.nan

    if len(numeric_cols) >= 2:
        scores = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                pair_real = real_df[[c1, c2]]
                pair_synth = synthetic_df[[c1, c2]]
                s = _safe(ContinuousKLDivergence.compute, pair_real, pair_synth)
                if not np.isnan(s):
                    scores.append(s)
        row["Pair_ContinuousKL"] = float(np.mean(scores)) if scores else np.nan
    else:
        row["Pair_ContinuousKL"] = np.nan

    # ---- Detection ----
    if verbose:
        print("[SDMetrics] === DETECTION ===")
    row["Detection_Logistic"] = _safe(LogisticDetection.compute, real_df, synthetic_df, metadata=metadata)
    row["Detection_SVC"] = _safe(SVCDetection.compute, real_df, synthetic_df, metadata=metadata)

    # ---- Privacy ----
    if verbose:
        print("[SDMetrics] === PRIVACY ===")
    row["Privacy_NewRowSynthesis"] = _safe(
        NewRowSynthesis.compute, real_df, synthetic_df, metadata=metadata
    )
    row["Privacy_DCRBaseline"] = _safe(
        DCRBaselineProtection.compute, real_df, synthetic_df, metadata
    )
    # DCROverfittingProtection requires a held-out validation real set: split real_df 80/20.
    if len(real_df) >= 20:
        rng = np.random.default_rng(42)
        idx = np.arange(len(real_df))
        rng.shuffle(idx)
        cut = max(1, int(0.8 * len(idx)))
        real_train = real_df.iloc[idx[:cut]].reset_index(drop=True)
        real_val = real_df.iloc[idx[cut:]].reset_index(drop=True)
        row["Privacy_DCROverfitting"] = _safe(
            DCROverfittingProtection.compute, real_train, synthetic_df, real_val, metadata
        )
    else:
        row["Privacy_DCROverfitting"] = np.nan

    # DisclosureProtection: known = features, sensitive = target
    feature_cols = [c for c in all_cols if c != target_col]
    if feature_cols and target_col in real_df.columns and target_col in synthetic_df.columns:
        cont_cols = [c for c in feature_cols if c in numeric_cols]
        row["Privacy_Disclosure"] = _safe(
            DisclosureProtection.compute,
            real_df, synthetic_df,
            known_column_names=feature_cols,
            sensitive_column_names=[target_col],
            continuous_column_names=cont_cols if cont_cols else None,
        )
    else:
        row["Privacy_Disclosure"] = np.nan

    # CategoricalCAP: needs categorical sensitive field; skip if target is numerical
    if target_col in categorical_cols:
        cat_features = [c for c in categorical_cols if c != target_col]
        if cat_features:
            row["Privacy_CategoricalCAP"] = _safe(
                CategoricalCAP.compute,
                real_df, synthetic_df, metadata=metadata,
                key_fields=cat_features,
                sensitive_fields=[target_col],
            )
        else:
            row["Privacy_CategoricalCAP"] = np.nan
    else:
        row["Privacy_CategoricalCAP"] = np.nan

    # NumericalLR: needs numeric sensitive field; skip if target is categorical
    if target_col in numeric_cols:
        num_features = [c for c in numeric_cols if c != target_col]
        if num_features:
            row["Privacy_NumericalLR"] = _safe(
                NumericalLR.compute,
                real_df, synthetic_df, metadata=metadata,
                key_fields=num_features,
                sensitive_fields=[target_col],
            )
        else:
            row["Privacy_NumericalLR"] = np.nan
    else:
        row["Privacy_NumericalLR"] = np.nan

    # ---- ML-Efficacy (binary classification only) ----
    if verbose:
        print("[SDMetrics] === ML EFFICACY ===")
    is_binary = (
        target_col in real_df.columns
        and real_df[target_col].nunique() == 2
        and target_col in synthetic_df.columns
    )
    if is_binary:
        row["MLEff_BinaryDT"] = _safe(
            BinaryDecisionTreeClassifier.compute,
            real_df, synthetic_df, metadata=metadata, target=target_col,
        )
        row["MLEff_BinaryLR"] = _safe(
            BinaryLogisticRegression.compute,
            real_df, synthetic_df, metadata=metadata, target=target_col,
        )
        row["MLEff_BinaryAdaBoost"] = _safe(
            BinaryAdaBoostClassifier.compute,
            real_df, synthetic_df, metadata=metadata, target=target_col,
        )
    else:
        row["MLEff_BinaryDT"] = np.nan
        row["MLEff_BinaryLR"] = np.nan
        row["MLEff_BinaryAdaBoost"] = np.nan

    # ---- Aggregate reports ----
    if verbose:
        print("[SDMetrics] === AGGREGATE REPORTS ===")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = QualityReport()
            qr.generate(real_df, synthetic_df, metadata, verbose=False)
            row["Quality_Overall"] = float(qr.get_score())
    except Exception:
        row["Quality_Overall"] = np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dr = DiagnosticReport()
            dr.generate(real_df, synthetic_df, metadata, verbose=False)
            row["Diagnostic_Overall"] = float(dr.get_score())
    except Exception:
        row["Diagnostic_Overall"] = np.nan

    if verbose:
        n_valid = sum(1 for v in row.values() if not (isinstance(v, float) and np.isnan(v)))
        print(f"[SDMetrics] {n_valid}/{len(row)} metrics populated")

    return row
