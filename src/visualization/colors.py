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
#
# Hues are deliberately picked OUTSIDE the RdYlGn cell colormap used by the
# heatmap (red → yellow → green) so the legend swatches cannot be confused
# with score colors. All five sit in the cool / blue–purple–magenta region.
# ============================================================================
SDAC_CATEGORY_COLORS = {
    'Privacy':  '#1a237e',   # Deep Indigo
    'Fidelity': '#00838f',   # Cyan
    'Utility':  '#6a1b9a',   # Purple
    'Fairness': '#ad1457',   # Magenta-Pink
    'XAI':      '#37474f',   # Slate
}

SDAC_CATEGORY_FALLBACK = '#90a4ae'  # Blue-Grey (also outside RdYlGn)

# ============================================================================
# SDMetrics category colors — same RdYlGn-avoidance principle as SDAC, with
# enough hue separation across 8 categories to keep the legend readable.
# Palette is intentionally distinct from SDAC so the two heatmap color
# strips also read as separate evaluations at a glance.
# ============================================================================
SDMETRICS_CATEGORY_COLORS = {
    'Coverage':    '#0097a7',  # Cyan-Teal
    'Validity':    '#1976d2',  # Blue
    'Shapes':      '#283593',  # Indigo
    'Pair-Trends': '#7b1fa2',  # Purple
    'Detection':   '#c2185b',  # Pink
    'Privacy':     '#5e35b1',  # Deep Purple
    'ML-Efficacy': '#455a64',  # Blue-Grey
    'Aggregate':   '#212121',  # Near Black
}

SDMETRICS_CATEGORY_FALLBACK = '#9e9e9e'  # Mid Grey

# ============================================================================
# TRTS scenario colors (for grouped bar charts)
# ============================================================================
TRTS_SCENARIO_COLORS = {
    'TRTR': '#FFB6C1',  # Light Pink
    'TSTS': '#90EE90',  # Light Green
    'TRTS': '#DEB887',  # Burlywood/Tan
    'TSTR': '#87CEEB',  # Sky Blue
}
