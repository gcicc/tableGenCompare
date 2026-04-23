"""
Data Quality Evaluation Functions

This module contains functions for evaluating synthetic data quality,
including statistical similarity metrics and higher-order feature interactions.

Migrated from setup.py Phase 3 (Task 4.3 Migration Plan).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def calculate_mutual_information(real_data, synthetic_data, target_column,
                                 max_features=10, verbose=True):
    """
    Calculate mutual information preservation between real and synthetic data.

    Mutual information captures non-linear dependencies between features and the target,
    going beyond simple correlation to measure shared information content.

    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    synthetic_data : pd.DataFrame
        Synthetic dataset
    target_column : str
        Target column name
    max_features : int
        Maximum number of features to analyze (MI is computationally expensive)
    verbose : bool
        Print progress messages

    Returns:
    --------
    dict : {
        'mi_preservation': float,
        'mi_real': np.array,
        'mi_synth': np.array,
        'mi_features': list,
        'mi_correlation': float
    }
    """
    if verbose:
        print(f"\n[MI] Calculating Mutual Information preservation...")

    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        # Get numeric columns excluding target
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        # Limit to max_features for performance
        mi_cols = numeric_cols[:min(max_features, len(numeric_cols))]

        if not mi_cols:
            return {
                'mi_preservation': np.nan,
                'mi_real': np.array([]),
                'mi_synth': np.array([]),
                'mi_features': [],
                'mi_correlation': np.nan,
                'error': 'No numeric features to analyze'
            }

        if target_column not in real_data.columns:
            return {
                'mi_preservation': np.nan,
                'mi_real': np.array([]),
                'mi_synth': np.array([]),
                'mi_features': [],
                'mi_correlation': np.nan,
                'error': 'Target column not found'
            }

        # Prepare data
        X_real = real_data[mi_cols].fillna(0)
        X_synth = synthetic_data[mi_cols].fillna(0)
        y_real = real_data[target_column]
        y_synth = synthetic_data[target_column] if target_column in synthetic_data.columns else None

        if y_synth is None:
            return {
                'mi_preservation': np.nan,
                'mi_real': np.array([]),
                'mi_synth': np.array([]),
                'mi_features': [],
                'mi_correlation': np.nan,
                'error': 'Target column not in synthetic data'
            }

        # Determine if classification or regression
        is_classification = y_real.nunique() <= 10

        if verbose:
            print(f"   [MI] Analyzing {len(mi_cols)} features...")
            print(f"   [MI] Task type: {'Classification' if is_classification else 'Regression'}")

        # Calculate MI
        if is_classification:
            mi_real = mutual_info_classif(X_real, y_real, random_state=42)
            mi_synth = mutual_info_classif(X_synth, y_synth, random_state=42)
        else:
            mi_real = mutual_info_regression(X_real, y_real, random_state=42)
            mi_synth = mutual_info_regression(X_synth, y_synth, random_state=42)

        # Calculate preservation score (correlation between MI vectors)
        from scipy.stats import pearsonr
        if len(mi_real) > 1:
            mi_correlation = pearsonr(mi_real, mi_synth)[0]
            mi_correlation = max(0, mi_correlation)  # Clip to [0, 1]
        else:
            mi_correlation = 0.0

        if verbose:
            print(f"   [METRIC] MI Preservation: {mi_correlation:.3f}")

        return {
            'mi_preservation': mi_correlation,
            'mi_real': mi_real,
            'mi_synth': mi_synth,
            'mi_features': mi_cols,
            'mi_correlation': mi_correlation
        }

    except ImportError as e:
        if verbose:
            print(f"   [ERROR] sklearn.feature_selection not available: {e}")
        return {
            'mi_preservation': np.nan,
            'mi_real': np.array([]),
            'mi_synth': np.array([]),
            'mi_features': [],
            'mi_correlation': np.nan,
            'error': 'sklearn not available'
        }
    except Exception as e:
        if verbose:
            print(f"   [ERROR] MI calculation failed: {e}")
        return {
            'mi_preservation': np.nan,
            'mi_real': np.array([]),
            'mi_synth': np.array([]),
            'mi_features': [],
            'mi_correlation': np.nan,
            'error': str(e)
        }


def evaluate_synthetic_data_quality(real_data, synthetic_data, model_name, target_column,
                                  section_number, dataset_identifier=None,
                                  save_files=True, display_plots=False, verbose=True,
                                  collin_ctx=None):
    """
    Enhanced comprehensive evaluation of synthetic data quality with PCA analysis and file output.
    
    Parameters:
    - real_data: Original dataset
    - synthetic_data: Synthetic dataset to evaluate  
    - model_name: Model identifier (e.g., 'CTGAN', 'TVAE')
    - target_column: Name of target column for supervised metrics and PCA color-coding
    - section_number: Section number for file organization (2, 3, 5, etc.)
    - dataset_identifier: Dataset name for folder structure (auto-detected if None)
    - save_files: Whether to save plots and tables to files
    - display_plots: Whether to display plots in notebook (False for file-only mode)
    - verbose: Print detailed results
    
    Returns:
    - Dictionary with all evaluation metrics and file paths
    """
    # Import dependencies
    from src.config import DATASET_IDENTIFIER as GLOBAL_DATASET_ID
    from src.utils.paths import get_results_path
    
    # Auto-detect dataset identifier if not provided
    if dataset_identifier is None:
        dataset_identifier = GLOBAL_DATASET_ID or "unknown-dataset"
    # Create results directory structure
    results_dir = None
    if save_files:
        results_dir = Path(get_results_path(dataset_identifier, section_number)) / model_name.upper()
        results_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"[SEARCH] {model_name.upper()} - COMPREHENSIVE DATA QUALITY EVALUATION")
        print("=" * 60)
        if save_files:
            print(f"[FOLDER] Output directory: {results_dir}")
    
    results = {
        'model': model_name,
        'section': section_number,
        'files_generated': [],
        'dataset_identifier': dataset_identifier
    }
    
    # Get numeric columns for analysis
    numeric_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
    # Keep target column for PCA analysis but separate for other analyses
    numeric_cols_no_target = [col for col in numeric_cols if col != target_column]
    
    # 1. STATISTICAL SIMILARITY
    if verbose:
        print("\n[1] STATISTICAL SIMILARITY")
        print("-" * 30)
    
    stat_results = []
    for col in numeric_cols_no_target:
        try:
            real_vals = real_data[col].dropna()
            synth_vals = synthetic_data[col].dropna()
            
            real_mean, real_std = real_vals.mean(), real_vals.std()
            synth_mean, synth_std = synth_vals.mean(), synth_vals.std()
            
            mean_diff = abs(real_mean - synth_mean) / real_std if real_std > 0 else 0
            std_ratio = min(real_std, synth_std) / max(real_std, synth_std) if max(real_std, synth_std) > 0 else 1
            
            stat_results.append({
                'column': col,
                'real_mean': real_mean,
                'synthetic_mean': synth_mean,
                'mean_similarity': 1 - min(mean_diff, 1),
                'std_similarity': std_ratio,
                'overall_similarity': (1 - min(mean_diff, 1) + std_ratio) / 2
            })
        except Exception as e:
            if verbose:
                print(f"   [WARNING] Error analyzing {col}: {e}")
    
    if stat_results:
        stat_df = pd.DataFrame(stat_results)
        avg_stat_similarity = stat_df['overall_similarity'].mean()
        results['avg_statistical_similarity'] = avg_stat_similarity
        
        if save_files and results_dir:
            stat_file = results_dir / 'statistical_similarity.csv'
            stat_df.to_csv(stat_file, index=False)
            results['files_generated'].append(str(stat_file))
        
        if verbose:
            print(f"   [CHART] Average Statistical Similarity: {avg_stat_similarity:.3f}")
    
    # 2. PCA COMPARISON ANALYSIS WITH OUTCOME VARIABLE COLOR-CODING
    if verbose:
        print("\n[2] PCA COMPARISON ANALYSIS WITH OUTCOME COLOR-CODING")
        print("-" * 50)
    
    pca_results = {}
    try:
        # Use all numeric columns including target for PCA
        pca_columns = [col for col in numeric_cols if col in synthetic_data.columns]
        
        if len(pca_columns) >= 2:
            # Standardize data
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_data[pca_columns].fillna(0))
            synth_scaled = scaler.transform(synthetic_data[pca_columns].fillna(0))
            
            # Apply PCA
            n_components = min(4, len(pca_columns))
            pca = PCA(n_components=n_components)
            real_pca = pca.fit_transform(real_scaled)
            synth_pca = pca.transform(synth_scaled)
            
            # Calculate PCA similarity metrics
            pca_similarities = []
            for i in range(n_components):
                corr = abs(stats.pearsonr(real_pca[:, i], synth_pca[:len(real_pca), i])[0]) if len(real_pca) > 0 else 0
                pca_similarities.append(corr)
            
            pca_similarity = np.mean(pca_similarities)
            explained_variance_real = pca.explained_variance_ratio_
            
            # Store PCA results
            pca_results = {
                'n_components': n_components,
                'explained_variance_ratio': explained_variance_real,
                'component_similarity': pca_similarities,
                'overall_pca_similarity': pca_similarity
            }
            results.update(pca_results)
            
            # Create PCA comparison plots with outcome color-coding
            if save_files or display_plots:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'{model_name.upper()} - PCA Comparison with {target_column.title()} Color-coding', 
                           fontsize=16, fontweight='bold')
                
                # Get target values for color-coding
                real_target = real_data[target_column] if target_column in real_data.columns else np.zeros(len(real_data))
                synth_target = synthetic_data[target_column] if target_column in synthetic_data.columns else np.zeros(len(synthetic_data))
                
                # Plot 1: Real data PC1 vs PC2
                axes[0,0].scatter(real_pca[:, 0], real_pca[:, 1], c=real_target, cmap='viridis', alpha=0.6, s=30)
                axes[0,0].set_title('Real Data - PC1 vs PC2')
                axes[0,0].set_xlabel(f'PC1 ({explained_variance_real[0]:.1%} variance)')
                axes[0,0].set_ylabel(f'PC2 ({explained_variance_real[1]:.1%} variance)')
                
                # Plot 2: Synthetic data PC1 vs PC2  
                scatter = axes[0,1].scatter(synth_pca[:, 0], synth_pca[:, 1], c=synth_target, cmap='viridis', alpha=0.6, s=30)
                axes[0,1].set_title('Synthetic Data - PC1 vs PC2')
                axes[0,1].set_xlabel(f'PC1 ({explained_variance_real[0]:.1%} variance)')
                axes[0,1].set_ylabel(f'PC2 ({explained_variance_real[1]:.1%} variance)')
                plt.colorbar(scatter, ax=axes[0,1], label=target_column.title())
                
                # Plot 3: Explained variance comparison
                components = range(1, n_components + 1)
                axes[1,0].bar([x - 0.2 for x in components], explained_variance_real, 0.4, 
                            label='Real Data', alpha=0.7, color='blue')
                # Note: Using same explained variance ratio for synthetic as approximation
                axes[1,0].bar([x + 0.2 for x in components], explained_variance_real, 0.4,
                            label='Synthetic Data', alpha=0.7, color='orange') 
                axes[1,0].set_title('Explained Variance Ratio Comparison')
                axes[1,0].set_xlabel('Principal Component')
                axes[1,0].set_ylabel('Explained Variance Ratio')
                axes[1,0].legend()
                
                # Plot 4: Component similarity scores
                axes[1,1].bar(components, pca_similarities, alpha=0.7, color='green')
                axes[1,1].set_title('PCA Component Similarity Scores')
                axes[1,1].set_xlabel('Principal Component')
                axes[1,1].set_ylabel('Similarity Score')
                axes[1,1].set_ylim(0, 1)
                
                plt.tight_layout()
                
                if save_files and results_dir:
                    pca_plot_file = results_dir / 'pca_comparison_with_outcome.png'
                    plt.savefig(pca_plot_file, dpi=300, bbox_inches='tight')
                    results['files_generated'].append(str(pca_plot_file))
                    if verbose:
                        print(f"   [CHART] PCA comparison plot saved: {pca_plot_file.name}")
                
                if display_plots:
                    plt.show()
                else:
                    plt.close()
            
            if verbose:
                print(f"   [CHART] PCA Overall Similarity: {pca_similarity:.3f}")
                print(f"   [CHART] Explained Variance (PC1, PC2): {explained_variance_real[0]:.3f}, {explained_variance_real[1]:.3f}")
                
    except Exception as e:
        if verbose:
            print(f"   [ERROR] PCA analysis failed: {e}")
        pca_results = {'error': str(e)}
    
    # 3. DISTRIBUTION SIMILARITY WITH VISUALIZATIONS
    if verbose:
        print("\n[3] DISTRIBUTION SIMILARITY")
        print("-" * 30)
    
    try:
        if save_files or display_plots:
            n_cols = min(3, len(numeric_cols_no_target))
            n_rows = (len(numeric_cols_no_target) + n_cols - 1) // n_cols
            
            if n_rows > 0:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                fig.suptitle(f'{model_name.upper()} - Feature Distribution Comparison', 
                           fontsize=16, fontweight='bold')
                
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes
                else:
                    axes = axes.flatten()
                
                js_scores = []
                for i, col in enumerate(numeric_cols_no_target):
                    if i < len(axes):
                        ax = axes[i]
                        
                        # Calculate Jensen-Shannon divergence
                        try:
                            real_hist, bins = np.histogram(real_data[col].dropna(), bins=20, density=True)
                            synth_hist, _ = np.histogram(synthetic_data[col].dropna(), bins=bins, density=True)
                            
                            real_hist = real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
                            synth_hist = synth_hist / synth_hist.sum() if synth_hist.sum() > 0 else synth_hist
                            
                            js_div = jensenshannon(real_hist, synth_hist)
                            js_similarity = 1 - js_div
                            js_scores.append(js_similarity)
                            
                            # Create distribution comparison plot
                            ax.hist(real_data[col].dropna(), bins=20, alpha=0.7, label='Real', 
                                  density=True, color='blue')
                            ax.hist(synthetic_data[col].dropna(), bins=20, alpha=0.7, label='Synthetic', 
                                  density=True, color='orange')
                            ax.set_title(f'{col}\nJS Similarity: {js_similarity:.3f}')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                        except Exception as e:
                            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'{col} - Error')
                
                # Hide unused subplots
                for j in range(len(numeric_cols_no_target), len(axes)):
                    axes[j].set_visible(False)
                
                plt.tight_layout()
                
                if save_files and results_dir:
                    dist_plot_file = results_dir / 'distribution_comparison.png'
                    plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
                    results['files_generated'].append(str(dist_plot_file))
                
                if display_plots:
                    plt.show()
                else:
                    plt.close()
                
                avg_js_similarity = np.mean(js_scores) if js_scores else 0
                results['avg_js_similarity'] = avg_js_similarity
                
                if verbose:
                    print(f"   [CHART] Average Distribution Similarity: {avg_js_similarity:.3f}")
    
    except Exception as e:
        if verbose:
            print(f"   [ERROR] Distribution analysis failed: {e}")
    
    # 4. CORRELATION STRUCTURE PRESERVATION
    if verbose:
        print("\n[4] CORRELATION STRUCTURE")
        print("-" * 30)
    
    try:
        from src.evaluation.association import compute_mixed_association_matrix

        def _association_comparison(real_df, synth_df, filename, title_suffix=""):
            """Build real-vs-synthetic heatmap pair and return preservation score."""
            common = [c for c in real_df.columns if c in synth_df.columns]
            r_corr = compute_mixed_association_matrix(real_df[common])
            s_corr = compute_mixed_association_matrix(synth_df[common])

            r_flat = r_corr.values.flatten()
            s_flat = s_corr.values.flatten()
            m = ~(np.isnan(r_flat) | np.isnan(s_flat))
            score = max(0, stats.pearsonr(r_flat[m], s_flat[m])[0]) if m.sum() > 1 else 0

            if save_files or display_plots:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                show_annot = len(r_corr.columns) <= 6
                sns.heatmap(r_corr, annot=show_annot, cmap='RdBu_r', center=0,
                            vmin=-1, vmax=1, square=True, ax=axes[0], fmt='.2f')
                axes[0].set_title(f'Real Data - Association Matrix{title_suffix}')
                sns.heatmap(s_corr, annot=show_annot, cmap='RdBu_r', center=0,
                            vmin=-1, vmax=1, square=True, ax=axes[1], fmt='.2f')
                axes[1].set_title(
                    f'Synthetic Data - Association Matrix{title_suffix}\n'
                    f'Preservation Score: {score:.3f}'
                )
                fig.text(0.5, -0.02,
                         'Pearson (num\u2013num): [\u22121, 1]  |  Cram\u00e9r\u2019s V (cat\u2013cat): [0, 1]  |  '
                         'Correlation ratio \u03b7 (num\u2013cat): [0, 1]',
                         ha='center', va='top', fontsize=9, style='italic', color='0.4')
                plt.tight_layout()
                if save_files and results_dir:
                    out = results_dir / filename
                    plt.savefig(out, dpi=300, bbox_inches='tight')
                    results['files_generated'].append(str(out))
                if display_plots:
                    plt.show()
                else:
                    plt.close()
            return score

        # When collin_ctx is provided, emit two standalone files (full + reduced).
        # Otherwise, emit the single legacy file on the schema as passed in.
        if collin_ctx is not None and getattr(collin_ctx, 'ops', None):
            from src.data.collinearity import apply_reducer
            corr_preservation = _association_comparison(
                real_data, synthetic_data,
                filename='association_comparison_full.png',
                title_suffix=' (full)',
            )
            real_reduced = apply_reducer(real_data, collin_ctx)
            synth_reduced = apply_reducer(synthetic_data, collin_ctx)
            corr_preservation_reduced = _association_comparison(
                real_reduced, synth_reduced,
                filename='association_comparison_reduced.png',
                title_suffix=' (reduced)',
            )
            results['correlation_preservation_reduced'] = corr_preservation_reduced
        else:
            corr_preservation = _association_comparison(
                real_data, synthetic_data,
                filename='association_comparison.png',
            )

        results['correlation_preservation'] = corr_preservation

        if verbose:
            print(f"   [CHART] Correlation Structure Preservation: {corr_preservation:.3f}")
            if 'correlation_preservation_reduced' in results:
                print(
                    f"   [CHART] Correlation Structure Preservation (reduced): "
                    f"{results['correlation_preservation_reduced']:.3f}"
                )

    except Exception as e:
        if verbose:
            print(f"   [ERROR] Correlation analysis failed: {e}")
        results['correlation_preservation'] = 0
    
    # 5. MACHINE LEARNING UTILITY
    if target_column and target_column in real_data.columns:
        if verbose:
            print("\n[5] MACHINE LEARNING UTILITY")
            print("-" * 30)
        
        try:
            X_real = real_data[numeric_cols_no_target].fillna(0)
            y_real = real_data[target_column]
            X_synth = synthetic_data[numeric_cols_no_target].fillna(0)
            y_synth = synthetic_data[target_column] if target_column in synthetic_data.columns else None
            
            if y_synth is not None:
                # Train on real, test on synthetic
                rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_real.fit(X_real, y_real)
                synth_test_accuracy = rf_real.score(X_synth, y_synth)
                
                # Train on synthetic, test on real
                rf_synth = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_synth.fit(X_synth, y_synth)
                real_test_accuracy = rf_synth.score(X_real, y_real)
                
                ml_utility = (synth_test_accuracy + real_test_accuracy) / 2
                
                results.update({
                    'ml_utility': ml_utility,
                    'synth_test_accuracy': synth_test_accuracy,
                    'real_test_accuracy': real_test_accuracy
                })
                
                if verbose:
                    print(f"   [CHART] ML Utility (Cross-Accuracy): {ml_utility:.3f}")
                    print(f"   [CHART] Real->Synth Accuracy: {synth_test_accuracy:.3f}")
                    print(f"   [CHART] Synth->Real Accuracy: {real_test_accuracy:.3f}")
            
        except Exception as e:
            if verbose:
                print(f"   [ERROR] ML utility analysis failed: {e}")
    
    # 6. OVERALL QUALITY ASSESSMENT
    quality_scores = []
    if 'avg_statistical_similarity' in results:
        quality_scores.append(results['avg_statistical_similarity'])
    if 'avg_js_similarity' in results:
        quality_scores.append(results['avg_js_similarity'])
    if 'correlation_preservation' in results:
        quality_scores.append(results['correlation_preservation'])
    if 'overall_pca_similarity' in results:
        quality_scores.append(results['overall_pca_similarity'])
    if 'ml_utility' in results:
        quality_scores.append(results['ml_utility'])
    
    overall_quality = np.mean(quality_scores) if quality_scores else 0
    results['overall_quality_score'] = overall_quality
    
    # Quality assessment
    if overall_quality >= 0.8:
        quality_label = "EXCELLENT"
    elif overall_quality >= 0.6:
        quality_label = "GOOD"
    elif overall_quality >= 0.4:
        quality_label = "FAIR"
    else:
        quality_label = "POOR"
    
    results['quality_assessment'] = quality_label
    
    # Save comprehensive results summary
    if save_files and results_dir:
        summary_df = pd.DataFrame([{
            'Model': model_name,
            'Overall_Quality_Score': overall_quality,
            'Quality_Assessment': quality_label,
            'Statistical_Similarity': results.get('avg_statistical_similarity', 'N/A'),
            'Distribution_Similarity': results.get('avg_js_similarity', 'N/A'),
            'Correlation_Preservation': results.get('correlation_preservation', 'N/A'),
            'PCA_Similarity': results.get('overall_pca_similarity', 'N/A'),
            'ML_Utility': results.get('ml_utility', 'N/A')
        }])
        
        summary_file = results_dir / 'evaluation_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        results['files_generated'].append(str(summary_file))
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"[BEST] {model_name.upper()} OVERALL QUALITY SCORE: {overall_quality:.3f}")
        print(f"[INFO] Quality Assessment: {quality_label}")
        print("=" * 60)
        
        if save_files:
            print(f"\n[FOLDER] Generated {len(results['files_generated'])} output files:")
            for file_path in results['files_generated']:
                print(f"   - {Path(file_path).name}")
    
    return results
