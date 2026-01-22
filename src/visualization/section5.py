"""
Section 5 Visualization Functions

This module contains visualization functions for Final Comparison (Section 5),
including TRTS analysis visualizations and privacy metric dashboards.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# Model-specific color mapping for consistent visualization across all plots
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


def _get_model_color(model_name):
    """Get consistent color for a model name."""
    return MODEL_COLORS.get(model_name.upper(), '#7f7f7f')  # Gray fallback


def create_trts_visualizations(trts_results_dict, model_names, results_dir,
                              dataset_name="Dataset", save_files=True, display_plots=False):
    """
    Create comprehensive TRTS visualizations with dynamic y-axis adjustment.

    Parameters:
    -----------
    trts_results_dict : dict
        Dict of {model_name: trts_results} from comprehensive_trts_analysis
    model_names : list
        List of model names
    results_dir : str or Path
        Directory to save plots
    dataset_name : str
        Dataset name for plot titles
    save_files : bool
        Whether to save plots to files
    display_plots : bool
        Whether to display plots

    Returns:
    --------
    dict : Dictionary with plot file paths and summary statistics
    """
    if save_files:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

    # Prepare data for visualizations
    model_data = []
    scenario_data = []

    for model_name, trts_results in trts_results_dict.items():
        if 'error' in trts_results:
            continue

        # Extract individual scenario results
        scenarios = ['TRTR', 'TRTS', 'TSTR', 'TSTS']
        model_accuracies = []
        model_times = []

        for scenario in scenarios:
            if scenario in trts_results and trts_results[scenario].get('status') == 'success':
                result = trts_results[scenario]

                # Extract all comprehensive metrics (Phase 1-3: 30 metrics total)
                scenario_data.append({
                    'Model': model_name,
                    'Scenario': scenario,
                    # Core metrics
                    'Accuracy': result.get('accuracy', np.nan),
                    'Balanced_Accuracy': result.get('balanced_accuracy', np.nan),
                    # Precision/Recall family
                    'Precision': result.get('precision', np.nan),
                    'Recall': result.get('recall', np.nan),
                    'F1_Score': result.get('f1_score', np.nan),
                    'F_Beta_0_5': result.get('f_beta_0_5', np.nan),  # Phase 3
                    'F_Beta_2': result.get('f_beta_2', np.nan),  # Phase 3
                    # Specificity family
                    'Specificity': result.get('specificity', np.nan),
                    'Sensitivity': result.get('sensitivity', np.nan),
                    'TPR': result.get('tpr', np.nan),  # Phase 3: True Positive Rate
                    'TNR': result.get('tnr', np.nan),  # Phase 3: True Negative Rate
                    # Predictive values
                    'NPV': result.get('npv', np.nan),
                    'PPV': result.get('ppv', np.nan),  # Phase 3: Positive Predictive Value
                    # Error rates
                    'FPR': result.get('fpr', np.nan),
                    'FNR': result.get('fnr', np.nan),
                    'FDR': result.get('fdr', np.nan),  # Phase 2: False Discovery Rate
                    'FOR': result.get('false_omission_rate', np.nan),  # Phase 2: False Omission Rate
                    # Combined metrics
                    'MCC': result.get('mcc', np.nan),
                    'Cohen_Kappa': result.get('cohen_kappa', np.nan),
                    'Youden_J': result.get('youden_j', np.nan),  # Phase 3
                    'FMI': result.get('fmi', np.nan),  # Phase 3: Fowlkes-Mallows Index
                    # ROC/PR metrics
                    'AUROC': result.get('auroc', np.nan),
                    'AUPRC': result.get('auprc', np.nan),
                    # Probability metrics
                    'Brier_Score': result.get('brier_score', np.nan),  # Phase 3
                    # Population metrics
                    'Prevalence': result.get('prevalence', np.nan),  # Phase 3
                    'Predicted_Positive_Rate': result.get('predicted_positive_rate', np.nan),  # Phase 3
                    # Timing
                    'Training_Time': result.get('training_time', 0)
                })
                model_accuracies.append(result.get('accuracy', 0))
                model_times.append(result.get('training_time', 0))

        if model_accuracies:  # Only include if we have data
            # Calculate summary metrics
            avg_accuracy = np.mean(model_accuracies)
            total_time = sum(model_times)

            # Calculate similarity and utility scores
            similarity_score = (trts_results.get('TRTR', {}).get('accuracy', 0) +
                              trts_results.get('TSTS', {}).get('accuracy', 0)) / 2
            utility_score = (trts_results.get('TRTS', {}).get('accuracy', 0) +
                           trts_results.get('TSTR', {}).get('accuracy', 0)) / 2
            combined_score = (similarity_score + utility_score) / 2

            model_data.append({
                'Model': model_name,
                'Combined_Score': combined_score,
                'Overall_Similarity': similarity_score,
                'Average_Utility': utility_score,
                'Training_Time_Sec': total_time,
                'TRTR': trts_results.get('TRTR', {}).get('accuracy', 0),
                'TRTS': trts_results.get('TRTS', {}).get('accuracy', 0),
                'TSTR': trts_results.get('TSTR', {}).get('accuracy', 0),
                'TSTS': trts_results.get('TSTS', {}).get('accuracy', 0)
            })

    if not model_data:
        print("[ERROR] No valid TRTS data for visualization")
        return {'error': 'No valid data'}

    # Convert to DataFrame for easier plotting
    model_df = pd.DataFrame(model_data)
    scenario_df = pd.DataFrame(scenario_data)

    # Calculate dynamic y-axis limit based on max accuracy
    max_combined_score = model_df['Combined_Score'].max()
    max_scenario_accuracy = model_df[['TRTR', 'TRTS', 'TSTR', 'TSTS']].max().max()

    # Dynamic y-limit for subplots 1 and 3
    if max_combined_score > 0.95 or max_scenario_accuracy > 0.95:
        y_max = 1.1  # Add 10% headroom when scores are very high
    elif max_combined_score > 0.85 or max_scenario_accuracy > 0.85:
        y_max = 1.05  # Add 5% headroom for high scores
    else:
        y_max = 1.0  # Standard limit for lower scores

    # Create comprehensive visualization (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name} - Model Comparison Results', fontsize=16, fontweight='bold')

    # 1. Overall Model Performance (Combined Score) - Top Left
    ax1 = axes[0, 0]
    model_colors = [_get_model_color(m) for m in model_df['Model']]
    bars = ax1.bar(model_df['Model'], model_df['Combined_Score'], color=model_colors)
    ax1.set_title('Overall Model Performance (Combined Score)', fontweight='bold')
    ax1.set_ylabel('Combined Score')
    ax1.set_ylim(0, y_max)  # Dynamic y-axis limit

    # Add value labels on bars
    for bar, score in zip(bars, model_df['Combined_Score']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Similarity vs Utility Trade-off - Top Right
    ax2 = axes[0, 1]
    for _, row in model_df.iterrows():
        ax2.scatter(row['Overall_Similarity'], row['Average_Utility'],
                   s=100, color=_get_model_color(row['Model']), label=row['Model'], alpha=0.7)
    ax2.set_title('Similarity vs Utility Trade-off', fontweight='bold')
    ax2.set_xlabel('Overall Similarity')
    ax2.set_ylabel('Average Utility')
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Utility Metrics Comparison (TRTS Framework) - Bottom Left
    ax3 = axes[1, 0]
    x = np.arange(len(model_df))
    width = 0.2

    scenarios = ['TRTR', 'TSTS', 'TRTS', 'TSTR']
    colors_bar = ['#FFB6C1', '#90EE90', '#DEB887', '#87CEEB']

    for i, scenario in enumerate(scenarios):
        values = model_df[scenario].values
        ax3.bar(x + i*width, values, width, label=scenario, color=colors_bar[i], alpha=0.8)

    ax3.set_title('Utility Metrics Comparison (TRTS Framework)', fontweight='bold')
    ax3.set_ylabel('Accuracy Score')
    ax3.set_xlabel('Models')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(model_df['Model'])
    ax3.legend()
    ax3.set_ylim(0, y_max)  # Dynamic y-axis limit

    # 4. Model Training Time Comparison - Bottom Right
    ax4 = axes[1, 1]
    bars_time = ax4.bar(model_df['Model'], model_df['Training_Time_Sec'], color=model_colors)
    ax4.set_title('Model Training Time Comparison', fontweight='bold')
    ax4.set_ylabel('Training Time (seconds)')

    # Add value labels on bars
    for bar, time_val in zip(bars_time, model_df['Training_Time_Sec']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(model_df['Training_Time_Sec'])*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    files_generated = []

    if save_files:
        # Save comprehensive plot
        plot_file = results_path / 'trts_comprehensive_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        files_generated.append(str(plot_file))
        print(f"[VIZ] Saved: trts_comprehensive_analysis.png")

        # Save summary tables as CSV
        summary_file = results_path / 'trts_summary_metrics.csv'
        model_df.to_csv(summary_file, index=False)
        files_generated.append(str(summary_file))

        detailed_file = results_path / 'trts_detailed_results.csv'
        scenario_df.to_csv(detailed_file, index=False)
        files_generated.append(str(detailed_file))

        print(f"[VIZ] Generated {len(files_generated)} TRTS files")

    if display_plots:
        plt.show()
    else:
        plt.close()

    return {
        'files_generated': files_generated,
        'summary_stats': {
            'models_analyzed': len(model_df),
            'avg_combined_score': model_df['Combined_Score'].mean(),
            'best_model': model_df.loc[model_df['Combined_Score'].idxmax(), 'Model'],
            'total_scenarios_tested': len(scenario_df)
        }
    }


def create_privacy_dashboard(trts_results_dict, model_names, results_dir,
                            dataset_name="Dataset", save_files=True, display_plots=False,
                            verbose=True):
    """
    Create 4-panel privacy analysis dashboard.

    Visualizes privacy metrics across models:
    - Privacy Score Comparison (bar chart)
    - NNDR Distribution (box plot)
    - Memorization Risk (stacked bar chart)
    - Re-identification Risk (horizontal bar chart)

    Parameters:
    -----------
    trts_results_dict : dict
        Dict of {model_name: trts_results} with privacy metrics
    model_names : list
        List of model names
    results_dir : str or Path
        Directory to save plots
    dataset_name : str
        Dataset name for plot titles
    save_files : bool
        Whether to save plots to files
    display_plots : bool
        Whether to display plots
    verbose : bool
        Print messages

    Returns:
    --------
    dict : Dictionary with plot file path and summary statistics
    """
    if save_files:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

    # Prepare privacy data
    privacy_data = []

    for model_name, trts_results in trts_results_dict.items():
        if 'privacy' in trts_results and 'error' not in trts_results['privacy']:
            privacy = trts_results['privacy']
            privacy_data.append({
                'Model': model_name,
                'Privacy_Score': privacy.get('privacy_score', np.nan),
                'NNDR_Mean': privacy.get('nndr_mean', np.nan),
                'NNDR_Std': privacy.get('nndr_std', np.nan),
                'NNDR_Distribution': privacy.get('nndr_distribution', []),
                'Memorization_Score': privacy.get('memorization_score', np.nan),
                'Memorized_Count': privacy.get('memorized_count', 0),
                'Reidentification_Risk': privacy.get('reidentification_risk', np.nan),
                'DCR_Mean': privacy.get('dcr_mean', np.nan)
            })

    if not privacy_data:
        if verbose:
            print("[WARNING] No privacy data available for visualization")
        return {'error': 'No privacy data'}

    privacy_df = pd.DataFrame(privacy_data)

    # Create 4-panel dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name} - Privacy Analysis Dashboard', fontsize=16, fontweight='bold')

    # Panel 1 (Top Left): Privacy Score Comparison
    ax1 = axes[0, 0]
    colors_privacy = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red'
                     for x in privacy_df['Privacy_Score']]
    bars = ax1.bar(privacy_df['Model'], privacy_df['Privacy_Score'], color=colors_privacy, alpha=0.7)
    ax1.set_title('Overall Privacy Score (Higher = Better)', fontweight='bold')
    ax1.set_ylabel('Privacy Score')
    ax1.set_ylim(0, 1.15)  # Increased from 1.0 to provide headroom for labels
    ax1.axhline(y=0.7, color='green', linestyle='--', linewidth=1, label='Good (>0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='Moderate (>0.5)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, score in zip(bars, privacy_df['Privacy_Score']):
        if not np.isnan(score):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Panel 2 (Top Right): NNDR Distribution (Box Plot)
    ax2 = axes[0, 1]
    nndr_data = [row['NNDR_Distribution'] for _, row in privacy_df.iterrows()
                 if len(row['NNDR_Distribution']) > 0]
    nndr_labels = [row['Model'] for _, row in privacy_df.iterrows()
                   if len(row['NNDR_Distribution']) > 0]

    if nndr_data:
        bp = ax2.boxplot(nndr_data, labels=nndr_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.6)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                   label='NNDR=1.0 (Threshold)')
        ax2.set_title('NNDR Distribution (>1.0 = Good)', fontweight='bold')
        ax2.set_ylabel('NNDR Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'NNDR data not available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('NNDR Distribution', fontweight='bold')

    # Panel 3 (Bottom Left): Memorization Risk (Stacked Bar)
    ax3 = axes[1, 0]
    memorized_pct = privacy_df['Memorization_Score'] * 100
    safe_pct = 100 - memorized_pct

    x = np.arange(len(privacy_df))
    width = 0.6

    ax3.bar(x, safe_pct, width, label='Safe Records', color='green', alpha=0.7)
    ax3.bar(x, memorized_pct, width, bottom=safe_pct, label='Memorized Records', color='red', alpha=0.7)

    ax3.set_title('Memorization Risk (Lower = Better)', fontweight='bold')
    ax3.set_ylabel('Percentage of Records (%)')
    ax3.set_xlabel('Models')
    ax3.set_xticks(x)
    ax3.set_xticklabels(privacy_df['Model'], rotation=45, ha='right')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (mem, safe) in enumerate(zip(memorized_pct, safe_pct)):
        if mem > 5:  # Only show if > 5%
            ax3.text(i, safe + mem/2, f'{mem:.1f}%', ha='center', va='center',
                    fontweight='bold', color='white')

    # Panel 4 (Bottom Right): Re-identification Risk
    ax4 = axes[1, 1]
    reid_risk_pct = privacy_df['Reidentification_Risk'] * 100
    colors_reid = ['green' if x < 5 else 'orange' if x < 20 else 'red'
                   for x in reid_risk_pct]

    bars_reid = ax4.barh(privacy_df['Model'], reid_risk_pct, color=colors_reid, alpha=0.7)
    ax4.set_title('Re-identification Risk (Lower = Better)', fontweight='bold')
    ax4.set_xlabel('Risk Percentage (%)')
    ax4.axvline(x=5, color='green', linestyle='--', linewidth=1, label='Low (<5%)')
    ax4.axvline(x=20, color='orange', linestyle='--', linewidth=1, label='Moderate (<20%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, risk in zip(bars_reid, reid_risk_pct):
        if not np.isnan(risk):
            ax4.text(risk + 1, bar.get_y() + bar.get_height()/2,
                    f'{risk:.1f}%', ha='left', va='center', fontweight='bold')

    plt.tight_layout()

    files_generated = []

    if save_files:
        plot_file = results_path / 'privacy_dashboard.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        files_generated.append(str(plot_file))

        # Save summary CSV
        summary_file = results_path / 'privacy_summary.csv'
        privacy_df_export = privacy_df.drop(columns=['NNDR_Distribution'])  # Remove list column
        privacy_df_export.to_csv(summary_file, index=False)
        files_generated.append(str(summary_file))

        if verbose:
            print(f"[VIZ] Saved: privacy_dashboard.png")
            print(f"[VIZ] Saved: privacy_summary.csv")

    if display_plots:
        plt.show()
    else:
        plt.close()

    return {
        'files_generated': files_generated,
        'summary_stats': {
            'models_analyzed': len(privacy_df),
            'avg_privacy_score': privacy_df['Privacy_Score'].mean(),
            'avg_memorization': privacy_df['Memorization_Score'].mean(),
            'avg_reid_risk': privacy_df['Reidentification_Risk'].mean(),
            'best_privacy_model': privacy_df.loc[privacy_df['Privacy_Score'].idxmax(), 'Model'] if len(privacy_df) > 0 else None
        }
    }


def create_trts_roc_curves(trts_results_dict, model_names, results_dir,
                          dataset_name="Dataset", save_files=True,
                          display_plots=False, verbose=True):
    """
    Generate ROC (Receiver Operating Characteristic) curves for all TRTS scenarios.

    Creates a 2×2 subplot grid showing ROC curves for:
    - TRTR (Train Real, Test Real)
    - TRTS (Train Real, Test Synthetic)
    - TSTR (Train Synthetic, Test Real)
    - TSTS (Train Synthetic, Test Synthetic)

    Parameters:
    -----------
    trts_results_dict : dict
        Dictionary with model names as keys, each containing TRTS results with
        'predictions' key (y_true, y_pred, y_pred_proba)
    model_names : list
        List of model names to include in visualization
    results_dir : str
        Directory to save output PNG file
    dataset_name : str, default="Dataset"
        Dataset identifier for title
    save_files : bool, default=True
        Whether to save PNG file to disk
    display_plots : bool, default=False
        Whether to display plot interactively
    verbose : bool, default=True
        Whether to print progress messages

    Returns:
    --------
    str or None : Path to saved PNG file, or None if no valid data

    Notes:
    ------
    - Requires store_predictions=True when calling comprehensive_trts_analysis()
    - Gracefully skips models/scenarios without prediction data
    - Binary classification: Single ROC curve per model
    - Multi-class: Macro-average ROC curve
    - Displays AUROC scores in legend
    """
    import os

    if verbose:
        print("\n" + "="*70)
        print("GENERATING ROC CURVES")
        print("="*70)

    scenarios = ['TRTR', 'TRTS', 'TSTR', 'TSTS']
    scenario_titles = {
        'TRTR': 'Train Real, Test Real',
        'TRTS': 'Train Real, Test Synthetic',
        'TSTR': 'Train Synthetic, Test Real',
        'TSTS': 'Train Synthetic, Test Synthetic'
    }

    # Check if any model has prediction data
    has_predictions = False
    for model_name in model_names:
        if model_name in trts_results_dict:
            for scenario in scenarios:
                if (scenario in trts_results_dict[model_name] and
                    trts_results_dict[model_name][scenario].get('status') == 'success' and
                    'predictions' in trts_results_dict[model_name][scenario]):
                    has_predictions = True
                    break
            if has_predictions:
                break

    if not has_predictions:
        if verbose:
            print("[WARNING] No prediction data available for ROC curves")
            print("         Call comprehensive_trts_analysis() with store_predictions=True")
        return None

    # Create 2×2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]

        for model_name in model_names:
            if model_name not in trts_results_dict:
                continue

            scenario_results = trts_results_dict[model_name].get(scenario, {})

            if (scenario_results.get('status') != 'success' or
                'predictions' not in scenario_results):
                continue

            preds = scenario_results['predictions']
            y_true = preds['y_true']
            y_pred_proba = preds['y_pred_proba']
            classes = preds.get('classes', np.unique(y_true))

            if y_pred_proba is None:
                continue

            n_classes = len(classes)

            try:
                if n_classes == 2:
                    # Binary classification: Single curve
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})', linewidth=2)
                else:
                    # Multi-class: Macro-average curve
                    y_true_bin = label_binarize(y_true, classes=classes)

                    # Compute micro-average ROC curve
                    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
                    roc_auc_micro = auc(fpr_micro, tpr_micro)

                    # Compute macro-average ROC curve
                    fpr_macro = np.linspace(0, 1, 100)
                    tpr_macro_list = []

                    for i in range(n_classes):
                        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                        tpr_macro_list.append(np.interp(fpr_macro, fpr_i, tpr_i))

                    tpr_macro = np.mean(tpr_macro_list, axis=0)
                    roc_auc_macro = auc(fpr_macro, tpr_macro)

                    ax.plot(fpr_macro, tpr_macro,
                           label=f'{model_name} (macro AUC={roc_auc_macro:.3f})',
                           linewidth=2)
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Could not generate ROC curve for {model_name} {scenario}: {e}")
                continue

        # Diagonal reference line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{scenario_titles[scenario]}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

    plt.suptitle(f'ROC Curves - {dataset_name}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = None
    if save_files:
        output_path = os.path.join(results_dir, 'trts_roc_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"[SUCCESS] ROC curves saved: {output_path}")

    if display_plots:
        plt.show()
    else:
        plt.close()

    return output_path


def create_trts_pr_curves(trts_results_dict, model_names, results_dir,
                         dataset_name="Dataset", save_files=True,
                         display_plots=False, verbose=True):
    """
    Generate Precision-Recall curves for all TRTS scenarios.

    Creates a 2×2 subplot grid showing PR curves for all scenarios.
    Particularly useful for imbalanced datasets where AUROC may be misleading.

    Parameters:
    -----------
    trts_results_dict : dict
        Dictionary with model names as keys, each containing TRTS results with
        'predictions' key (y_true, y_pred, y_pred_proba)
    model_names : list
        List of model names to include in visualization
    results_dir : str
        Directory to save output PNG file
    dataset_name : str, default="Dataset"
        Dataset identifier for title
    save_files : bool, default=True
        Whether to save PNG file to disk
    display_plots : bool, default=False
        Whether to display plot interactively
    verbose : bool, default=True
        Whether to print progress messages

    Returns:
    --------
    str or None : Path to saved PNG file, or None if no valid data
    """
    import os

    if verbose:
        print("\n" + "="*70)
        print("GENERATING PRECISION-RECALL CURVES")
        print("="*70)

    scenarios = ['TRTR', 'TRTS', 'TSTR', 'TSTS']
    scenario_titles = {
        'TRTR': 'Train Real, Test Real',
        'TRTS': 'Train Real, Test Synthetic',
        'TSTR': 'Train Synthetic, Test Real',
        'TSTS': 'Train Synthetic, Test Synthetic'
    }

    # Check for prediction data
    has_predictions = False
    for model_name in model_names:
        if model_name in trts_results_dict:
            for scenario in scenarios:
                if (scenario in trts_results_dict[model_name] and
                    trts_results_dict[model_name][scenario].get('status') == 'success' and
                    'predictions' in trts_results_dict[model_name][scenario]):
                    has_predictions = True
                    break
            if has_predictions:
                break

    if not has_predictions:
        if verbose:
            print("[WARNING] No prediction data available for PR curves")
        return None

    # Create 2×2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]

        for model_name in model_names:
            if model_name not in trts_results_dict:
                continue

            scenario_results = trts_results_dict[model_name].get(scenario, {})

            if (scenario_results.get('status') != 'success' or
                'predictions' not in scenario_results):
                continue

            preds = scenario_results['predictions']
            y_true = preds['y_true']
            y_pred_proba = preds['y_pred_proba']
            classes = preds.get('classes', np.unique(y_true))

            if y_pred_proba is None:
                continue

            n_classes = len(classes)

            try:
                if n_classes == 2:
                    # Binary classification
                    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
                    ap = average_precision_score(y_true, y_pred_proba[:, 1])
                    ax.plot(recall, precision, label=f'{model_name} (AP={ap:.3f})', linewidth=2)
                else:
                    # Multi-class: Macro-average
                    y_true_bin = label_binarize(y_true, classes=classes)

                    ap_scores = []
                    for i in range(n_classes):
                        ap_i = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
                        ap_scores.append(ap_i)

                    ap_macro = np.mean(ap_scores)

                    # Plot macro-average curve
                    recall_macro = np.linspace(0, 1, 100)
                    precision_list = []

                    for i in range(n_classes):
                        precision_i, recall_i, _ = precision_recall_curve(
                            y_true_bin[:, i], y_pred_proba[:, i]
                        )
                        precision_list.append(np.interp(recall_macro[::-1], recall_i[::-1],
                                                       precision_i[::-1])[::-1])

                    precision_macro = np.mean(precision_list, axis=0)
                    ax.plot(recall_macro, precision_macro,
                           label=f'{model_name} (macro AP={ap_macro:.3f})',
                           linewidth=2)
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Could not generate PR curve for {model_name} {scenario}: {e}")
                continue

        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'{scenario_titles[scenario]}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

    plt.suptitle(f'Precision-Recall Curves - {dataset_name}',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = None
    if save_files:
        output_path = os.path.join(results_dir, 'trts_pr_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"[SUCCESS] PR curves saved: {output_path}")

    if display_plots:
        plt.show()
    else:
        plt.close()

    return output_path


def create_trts_calibration_curves(trts_results_dict, model_names, results_dir,
                                   dataset_name="Dataset", save_files=True,
                                   display_plots=False, verbose=True):
    """
    Generate calibration (reliability) curves for all TRTS scenarios.

    Shows how well predicted probabilities match actual frequencies.
    Perfect calibration = diagonal line (predicted probability = observed frequency).

    Parameters:
    -----------
    trts_results_dict : dict
        Dictionary with model names as keys, each containing TRTS results with
        'predictions' key (y_true, y_pred, y_pred_proba)
    model_names : list
        List of model names to include in visualization
    results_dir : str
        Directory to save output PNG file
    dataset_name : str, default="Dataset"
        Dataset identifier for title
    save_files : bool, default=True
        Whether to save PNG file to disk
    display_plots : bool, default=False
        Whether to display plot interactively
    verbose : bool, default=True
        Whether to print progress messages

    Returns:
    --------
    str or None : Path to saved PNG file, or None if no valid data
    """
    import os

    if verbose:
        print("\n" + "="*70)
        print("GENERATING CALIBRATION CURVES")
        print("="*70)

    scenarios = ['TRTR', 'TRTS', 'TSTR', 'TSTS']
    scenario_titles = {
        'TRTR': 'Train Real, Test Real',
        'TRTS': 'Train Real, Test Synthetic',
        'TSTR': 'Train Synthetic, Test Real',
        'TSTS': 'Train Synthetic, Test Synthetic'
    }

    # Check for prediction data
    has_predictions = False
    for model_name in model_names:
        if model_name in trts_results_dict:
            for scenario in scenarios:
                if (scenario in trts_results_dict[model_name] and
                    trts_results_dict[model_name][scenario].get('status') == 'success' and
                    'predictions' in trts_results_dict[model_name][scenario]):
                    has_predictions = True
                    break
            if has_predictions:
                break

    if not has_predictions:
        if verbose:
            print("[WARNING] No prediction data available for calibration curves")
        return None

    # Create 2×2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]

        for model_name in model_names:
            if model_name not in trts_results_dict:
                continue

            scenario_results = trts_results_dict[model_name].get(scenario, {})

            if (scenario_results.get('status') != 'success' or
                'predictions' not in scenario_results):
                continue

            preds = scenario_results['predictions']
            y_true = preds['y_true']
            y_pred_proba = preds['y_pred_proba']
            classes = preds.get('classes', np.unique(y_true))

            if y_pred_proba is None:
                continue

            n_classes = len(classes)

            try:
                if n_classes == 2:
                    # Binary classification
                    prob_true, prob_pred = calibration_curve(
                        y_true, y_pred_proba[:, 1], n_bins=10, strategy='uniform'
                    )
                    ax.plot(prob_pred, prob_true, marker='o', label=model_name, linewidth=2)
                else:
                    # Multi-class: Average calibration across all classes
                    y_true_bin = label_binarize(y_true, classes=classes)

                    prob_true_list = []
                    prob_pred_list = []

                    for i in range(n_classes):
                        if y_true_bin[:, i].sum() > 0:  # Only if class exists
                            prob_true_i, prob_pred_i = calibration_curve(
                                y_true_bin[:, i], y_pred_proba[:, i],
                                n_bins=10, strategy='uniform'
                            )
                            prob_true_list.append(prob_true_i)
                            prob_pred_list.append(prob_pred_i)

                    if prob_true_list:
                        # Average across classes
                        max_len = max(len(p) for p in prob_true_list)
                        prob_true_avg = np.mean([
                            np.pad(p, (0, max_len - len(p)), constant_values=np.nan)
                            for p in prob_true_list
                        ], axis=0)
                        prob_pred_avg = np.mean([
                            np.pad(p, (0, max_len - len(p)), constant_values=np.nan)
                            for p in prob_pred_list
                        ], axis=0)

                        # Remove NaN values
                        mask = ~np.isnan(prob_true_avg) & ~np.isnan(prob_pred_avg)
                        ax.plot(prob_pred_avg[mask], prob_true_avg[mask],
                               marker='o', label=model_name, linewidth=2)

            except Exception as e:
                if verbose:
                    print(f"[WARNING] Could not generate calibration curve for "
                          f"{model_name} {scenario}: {e}")
                continue

        # Perfect calibration reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')

        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Positives', fontsize=11)
        ax.set_title(f'{scenario_titles[scenario]}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

    plt.suptitle(f'Calibration Curves - {dataset_name}',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = None
    if save_files:
        output_path = os.path.join(results_dir, 'trts_calibration_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"[SUCCESS] Calibration curves saved: {output_path}")

    if display_plots:
        plt.show()
    else:
        plt.close()

    return output_path
