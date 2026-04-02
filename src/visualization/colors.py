"""
Shared Color Definitions for All Visualizations

Centralizes model colors, real/synthetic colors, and semantic threshold colors
to ensure consistency across Sections 2–5.
"""

# ============================================================================
# Model-specific colors (consistent across all multi-model plots)
# Based on matplotlib tab10 — well-separated and colorblind-friendly.
# ============================================================================
MODEL_COLORS = {
    'CTGAN': '#1f77b4',       # Blue
    'TVAE': '#ff7f0e',        # Orange
    'CTABGAN': '#2ca02c',     # Green
    'CTABGANPLUS': '#d62728', # Red
    'CTABGAN+': '#d62728',    # Red (alias)
    'COPULAGAN': '#9467bd',   # Purple
    'GANERAID': '#8c564b',    # Brown
    'PATEGAN': '#e377c2',     # Pink
    'PATE-GAN': '#e377c2',    # Pink (alias)
    'MEDGAN': '#17becf',      # Cyan
    'TABDIFFUSION': '#bcbd22', # Olive/yellow-green (Phase 5 - April 2026)
    'GREAT': '#f0027f',        # Magenta (Phase 5 - April 2026)
}

DEFAULT_MODEL_COLOR = '#7f7f7f'  # Gray fallback for unknown models


def get_model_color(model_name):
    """Get consistent color for a model name."""
    return MODEL_COLORS.get(model_name.upper(), DEFAULT_MODEL_COLOR)


def get_model_colors_for_list(model_names):
    """Return a list of colors for a list of model names."""
    return [get_model_color(name) for name in model_names]


# ============================================================================
# Real vs. Synthetic data colors (used in distribution overlays, MI bars, etc.)
# ============================================================================
REAL_COLOR = '#4a90d9'      # Steel blue
SYNTH_COLOR = '#f5a623'     # Amber/orange

# ============================================================================
# SDAC category colors (for heatmap headers, radar chart grouping)
# ============================================================================
SDAC_CATEGORY_COLORS = {
    'Privacy': '#e74c3c',    # Red
    'Fidelity': '#3498db',   # Blue
    'Utility': '#2ecc71',    # Green
    'Fairness': '#f39c12',   # Orange
    'XAI': '#9b59b6',        # Purple
}

SDAC_CATEGORY_FALLBACK = '#95a5a6'  # Gray

# ============================================================================
# TRTS scenario colors (for grouped bar charts)
# ============================================================================
TRTS_SCENARIO_COLORS = {
    'TRTR': '#FFB6C1',  # Light Pink
    'TSTS': '#90EE90',  # Light Green
    'TRTS': '#DEB887',  # Burlywood/Tan
    'TSTR': '#87CEEB',  # Sky Blue
}
