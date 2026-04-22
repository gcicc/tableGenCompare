"""Runtime compatibility shim for dython / tab_gan_metrics.

``tab-gan-metrics`` (pinned at 1.1.4) imports ``compute_associations`` and
``numerical_encoding`` from ``dython.nominal`` — names that existed in
dython ≤0.5.1. From dython 0.7.x onward those names were renamed with a
leading underscore (``_compute_associations``) or refactored, breaking
``tab_gan_metrics.helpers``.

SageMaker's ``setup_env.sh`` works around this by hand-editing
``site-packages/tab_gan_metrics/*.py``. For the Windows (Positron /
pytorch conda env) install we avoid mutating third-party sources and
instead install the missing symbols back onto ``dython.nominal`` at
import time, so ``from dython.nominal import compute_associations, ...``
resolves.

Imported unconditionally by ``setup.py`` so that any subsequent
``from GANerAid.ganeraid import GANerAid`` — which transitively imports
``tab_gan_metrics.helpers`` — finds the compatibility names already in
place.
"""

from __future__ import annotations


def _apply() -> None:
    try:
        import dython.nominal as _dn  # noqa: WPS433
    except ImportError:
        return

    # ``compute_associations`` — old public name, now private ``_compute_associations``.
    if not hasattr(_dn, "compute_associations") and hasattr(_dn, "_compute_associations"):
        _dn.compute_associations = _dn._compute_associations

    # ``numerical_encoding`` was removed in dython 0.7.x. Provide a
    # minimal stand-in (label-encode non-numeric columns, pass numerics
    # through) — tab_gan_metrics uses it for categorical preprocessing
    # before distance calculations, and that is the semantics the old
    # helper offered.
    if not hasattr(_dn, "numerical_encoding"):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        def numerical_encoding(
            dataset,
            nominal_columns="all",
            drop_single_label=False,
            drop_fact_dict=False,
        ):
            df = dataset.copy()
            if nominal_columns == "all":
                cols = df.select_dtypes(
                    include=["object", "category", "bool"]
                ).columns.tolist()
            elif nominal_columns is None:
                cols = []
            else:
                cols = list(nominal_columns)
            for c in cols:
                if c in df.columns:
                    df[c] = LabelEncoder().fit_transform(df[c].astype(str))
            return df

        _dn.numerical_encoding = numerical_encoding


_apply()
