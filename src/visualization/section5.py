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
                accuracy = trts_results[scenario]['accuracy']
                time_val = trts_results[scenario]['training_time']

                scenario_data.append({
                    'Model': model_name,
                    'Scenario': scenario,
                    'Accuracy': accuracy,
                    'Training_Time': time_val
                })
                model_accuracies.append(accuracy)
                model_times.append(time_val)

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
    bars = ax1.bar(model_df['Model'], model_df['Combined_Score'],
                   color=['#FFD700', '#C0C0C0', '#CD853F', '#87CEEB'][:len(model_df)])
    ax1.set_title('Overall Model Performance (Combined Score)', fontweight='bold')
    ax1.set_ylabel('Combined Score')
    ax1.set_ylim(0, y_max)  # Dynamic y-axis limit

    # Add value labels on bars
    for bar, score in zip(bars, model_df['Combined_Score']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Similarity vs Utility Trade-off - Top Right
    ax2 = axes[0, 1]
    colors = ['#9966CC', '#32CD32', '#FF6347', '#FFD700']
    for i, (_, row) in enumerate(model_df.iterrows()):
        ax2.scatter(row['Overall_Similarity'], row['Average_Utility'],
                   s=100, color=colors[i % len(colors)], label=row['Model'], alpha=0.7)
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
    bars_time = ax4.bar(model_df['Model'], model_df['Training_Time_Sec'],
                       color=['#87CEEB', '#4682B4', '#C0C0C0', '#FFD700'][:len(model_df)])
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
