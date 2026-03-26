"""
Mixed-Association Matrix

Replaces Pearson-only correlation with a metric appropriate for each
column-pair type:
  num–num   → Pearson correlation  (range [-1, 1])
  cat–cat   → Cramér's V           (range [0, 1])
  num–cat   → Correlation ratio η  (range [0, 1])

Uses dython.nominal.associations() under the hood.
"""

import numpy as np
import pandas as pd


def compute_mixed_association_matrix(df, nominal_columns=None):
    """Compute a mixed-association matrix for a DataFrame with mixed types.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (numeric and/or categorical columns).
    nominal_columns : list[str] or None
        Columns to treat as categorical.  If *None*, columns with dtype
        ``object``, ``category``, or ``bool`` are treated as categorical,
        and integer columns with ≤10 unique values are also treated as
        categorical (a common heuristic for encoded labels).

    Returns
    -------
    pd.DataFrame
        Square association matrix with column/row labels matching *df*.
    """
    from dython.nominal import associations

    df = df.copy()

    # Auto-detect nominal columns when not specified
    if nominal_columns is None:
        nominal_columns = []
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype.name == 'category' or df[col].dtype == bool:
                nominal_columns.append(col)
            elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 10:
                nominal_columns.append(col)

    # dython expects nominal columns as a list of column names
    try:
        result = associations(
            df,
            nominal_columns=nominal_columns if nominal_columns else 'auto',
            compute_only=True,
            nan_replace_value='nan',
            mark_columns=False,
        )
        corr_matrix = result['corr']
    except Exception:
        # Fallback: numeric-only Pearson if dython fails
        corr_matrix = df.select_dtypes(include=[np.number]).corr()

    # Ensure the matrix is a DataFrame with proper labels
    if not isinstance(corr_matrix, pd.DataFrame):
        corr_matrix = pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)

    return corr_matrix
