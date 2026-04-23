"""
Batch evaluation functions for trained models.

Migrated from setup.py (Phase 4, Task 4.3) - streamlining setup.py
Unified evaluation function for both Section 3 and Section 5 trained models.
Now integrates SDAC tabular metrics for unified output.
"""

import os
import numpy as np
import pandas as pd

# Import required evaluation functions
from src.evaluation.quality import evaluate_synthetic_data_quality
from src.evaluation.trts import comprehensive_trts_analysis
from src.evaluation.sdac_metrics import compute_sdac_tabular_metrics
from src.visualization.section5 import create_trts_visualizations
from src.visualization.section4 import create_optuna_visualizations, create_all_models_optuna_summary
from src.utils.paths import get_results_path
from src.data.target_integrity import sanitize_synthetic_data, sanitize_numeric


def evaluate_trained_models(section_number, variable_pattern, scope=None, models_to_evaluate=None,
                           real_data=None, target_col=None,
                           protected_col=None, compute_mia=False):
    """
    Unified evaluation function for both Section 3 and Section 5 trained models.
    Replaces both evaluate_all_available_models and evaluate_section5_optimized_models
    to ensure 1:1 output correspondence and reduce code duplication.

    Now produces a unified sdac_evaluation_summary.csv with all SDAC metrics.

    Parameters:
    - section_number: Section number for file organization (3, 5, etc.)
    - variable_pattern: Pattern for variable names ('standard' or 'final')
      - 'standard': synthetic_data_ctgan, synthetic_data_ctabgan, etc. (Section 3)
      - 'final': synthetic_ctgan_final, synthetic_ctabgan_final, etc. (Section 5)
    - scope: globals() from notebook for variable access (required)
    - models_to_evaluate: List of specific models to evaluate (optional, evaluates all if None)
    - real_data: Real dataset (uses 'data' from scope if not provided)
    - target_col: Target column name (uses 'target_column' from scope if not provided)
    - protected_col: Protected attribute column for fairness metrics (None = skip fairness)
    - compute_mia: Whether to run MIA evaluation (expensive, default False)

    Returns:
    - Dictionary with comprehensive results for each evaluated model
    """

    if scope is None:
        print("[ERROR] ERROR: scope parameter required! Pass globals() from notebook")
        return {}

    # Get data and target from scope if not provided
    if real_data is None:
        real_data = scope.get('data')
        if real_data is None:
            print("[ERROR] ERROR: 'data' variable not found in scope")
            return {}

    if target_col is None:
        target_col = scope.get('target_column')
        if target_col is None:
            target_col = scope.get('TARGET_COLUMN')
        if target_col is None:
            print("[ERROR] ERROR: 'target_column' or 'TARGET_COLUMN' variable not found in scope")
            return {}

    dataset_id = scope.get('DATASET_IDENTIFIER', 'unknown-dataset')

    # Configure variable names based on pattern
    if variable_pattern == 'standard':
        # Section 3 pattern: synthetic_data_*
        model_checks = {
            'CTGAN': 'synthetic_data_ctgan',
            'CTABGAN': 'synthetic_data_ctabgan',
            'CTABGANPLUS': 'synthetic_data_ctabganplus',
            'GANerAid': 'synthetic_data_ganeraid',
            'CopulaGAN': 'synthetic_data_copulagan',
            'TVAE': 'synthetic_data_tvae',
            'MEDGAN': 'synthetic_data_medgan',
            'PATEGAN': 'synthetic_data_pategan',
            'TabDiffusion': 'synthetic_data_tabdiffusion',
            'GReaT': 'synthetic_data_great',
        }
    elif variable_pattern == 'final':
        # Section 5 pattern: synthetic_*_final
        model_checks = {
            'CTGAN': 'synthetic_ctgan_final',
            'CTABGAN': 'synthetic_ctabgan_final',
            'CTABGANPLUS': 'synthetic_ctabganplus_final',
            'GANerAid': 'synthetic_ganeraid_final',
            'CopulaGAN': 'synthetic_copulagan_final',
            'TVAE': 'synthetic_tvae_final',
            'MEDGAN': 'synthetic_medgan_final',
            'PATEGAN': 'synthetic_pategan_final',
            'TabDiffusion': 'synthetic_tabdiffusion_final',
            'GReaT': 'synthetic_great_final',
        }
    else:
        print(f"[ERROR] ERROR: Unknown variable_pattern '{variable_pattern}'. Use 'standard' or 'final'")
        return {}

    # Find available models in scope
    available_models = {}
    for model_name, var_name in model_checks.items():
        if var_name in scope and scope[var_name] is not None:
            # Filter by requested models if specified
            if models_to_evaluate is None or model_name in models_to_evaluate or model_name.lower() in [m.lower() for m in models_to_evaluate]:
                available_models[model_name] = scope[var_name]

    print(f"[SEARCH] UNIFIED BATCH EVALUATION - SECTION {section_number}")
    print("=" * 60)
    print(f"[INFO] Dataset: {dataset_id}")
    print(f"[INFO] Target column: {target_col}")
    print(f"[INFO] Protected column: {protected_col or 'None (fairness metrics skipped)'}")
    print(f"[INFO] MIA evaluation: {'ON' if compute_mia else 'OFF'}")
    print(f"[INFO] Variable pattern: {variable_pattern}")
    print(f"[INFO] Found {len(available_models)} trained models:")
    for model_name in available_models.keys():
        print(f"   [OK] {model_name}")

    if not available_models:
        available_vars = [var for var in model_checks.values() if var in scope]
        print("[ERROR] No synthetic datasets found!")
        print("   Train some models first before running batch evaluation")
        if available_vars:
            print(f"   Found variables: {available_vars}")
        return {}

    # Evaluate each available model using comprehensive evaluation
    evaluation_results = {}
    sdac_rows = []  # Collect SDAC rows for unified CSV

    for model_name, synthetic_data in available_models.items():
        print(f"\n{'='*20} EVALUATING {model_name} {'='*20}")

        try:
            # DEFENSIVE: Sanitize synthetic data before evaluation
            sanitized_synthetic = sanitize_synthetic_data(
                real_df=real_data,
                synth_df=synthetic_data,
                target_column=target_col,
                task_type="auto",
                verbose=False
            )

            # Use the comprehensive evaluation function for per-model plots
            results = evaluate_synthetic_data_quality(
                real_data=real_data,
                synthetic_data=sanitized_synthetic,
                model_name=model_name,
                target_column=target_col,
                section_number=section_number,
                dataset_identifier=dataset_id,
                save_files=True,
                display_plots=False,
                verbose=True
            )

            # Compute SDAC metrics for this model
            print(f"\n[SDAC] Computing SDAC metrics for {model_name}...")
            sdac_row = compute_sdac_tabular_metrics(
                real_df=real_data,
                synthetic_df=sanitized_synthetic,
                target_col=target_col,
                protected_col=protected_col,
                compute_mia=compute_mia,
                verbose=True
            )
            sdac_row['Model'] = model_name
            sdac_rows.append(sdac_row)

            results['sdac_metrics'] = sdac_row
            evaluation_results[model_name] = results
            print(f"[OK] {model_name} evaluation completed successfully!")

        except Exception as e:
            print(f"[ERROR] {model_name} evaluation failed: {e}")
            evaluation_results[model_name] = {'error': str(e)}

    # Create summary comparison
    print(f"\n{'='*25} EVALUATION SUMMARY {'='*25}")
    print(f"{'Model':<15} {'Quality Score':<15} {'Assessment':<12} {'Files':<8}")
    print("-" * 65)

    for model_name, results in evaluation_results.items():
        if 'error' not in results:
            quality_score = results.get('overall_quality_score', 0)
            assessment = results.get('quality_assessment', 'Unknown')
            file_count = len(results.get('files_generated', []))
            print(f"{model_name:<15} {quality_score:<15.3f} {assessment:<12} {file_count:<8}")
        else:
            print(f"{model_name:<15} {'ERROR':<15} {'FAILED':<12} {'0':<8}")

    # Save unified SDAC evaluation summary (replaces batch_evaluation_summary.csv)
    if sdac_rows:
        try:
            sdac_df = pd.DataFrame(sdac_rows)
            # Reorder so Model is first column
            cols = ['Model'] + [c for c in sdac_df.columns if c != 'Model']
            sdac_df = sdac_df[cols]

            summary_path = get_results_path(dataset_id, section_number)
            os.makedirs(summary_path, exist_ok=True)
            sdac_file = f"{summary_path}/sdac_evaluation_summary.csv"
            sdac_df.to_csv(sdac_file, index=False)
            print(f"\n[SDAC] SDAC evaluation summary saved to: {sdac_file}")

            # Print SDAC summary table
            print(f"\n{'='*25} SDAC METRICS SUMMARY {'='*25}")
            # Show key metrics per category
            key_cols = ['Model', 'Privacy_Score', 'Fidelity_JSD', 'Fidelity_Detection_AUC',
                       'Utility_ML_Efficacy', 'XAI_Feat_Imp_Corr']
            available_key_cols = [c for c in key_cols if c in sdac_df.columns]
            if available_key_cols:
                print(sdac_df[available_key_cols].to_string(index=False, float_format='%.4f'))

        except Exception as e:
            print(f"[WARNING] Could not save SDAC summary: {e}")

    # SECTION 4 OPTUNA VISUALIZATIONS
    # Automatically detect and visualize Optuna studies if present
    print(f"\n{'='*25} OPTUNA OPTIMIZATION ANALYSIS {'='*25}")

    # Map model names to study variable names
    study_var_names = {
        'CTGAN': 'ctgan_study',
        'CTABGAN': 'ctabgan_study',
        'CTABGANPLUS': 'ctabganplus_study',
        'GANerAid': 'ganeraid_study',
        'CopulaGAN': 'copulagan_study',
        'TVAE': 'tvae_study',
        'PATE-GAN': 'pategan_study',
        'MEDGAN': 'medgan_study'
    }

    detected_studies = {}
    for model_name, study_var in study_var_names.items():
        if study_var in scope and scope[study_var] is not None:
            detected_studies[model_name] = scope[study_var]

    if detected_studies:
        print(f"[INFO] Found {len(detected_studies)} Optuna studies: {list(detected_studies.keys())}")

        # Get Section 4 results path
        section4_results_dir = get_results_path(dataset_id, 4)
        os.makedirs(section4_results_dir, exist_ok=True)

        # Generate per-model Optuna visualizations
        optuna_files_generated = {}
        for model_name, study in detected_studies.items():
            print(f"\n[OPTUNA] Generating visualizations for {model_name}...")
            try:
                viz_paths = create_optuna_visualizations(
                    study=study,
                    model_name=model_name,
                    results_path=section4_results_dir,
                    verbose=True
                )
                if viz_paths:
                    optuna_files_generated[model_name] = viz_paths
                    # Add to evaluation results if model was evaluated
                    if model_name in evaluation_results:
                        if 'files_generated' not in evaluation_results[model_name]:
                            evaluation_results[model_name]['files_generated'] = []
                        evaluation_results[model_name]['files_generated'].extend(viz_paths.values())
            except Exception as e:
                print(f"[ERROR] Failed to generate Optuna visualizations for {model_name}: {e}")

        # Generate summary visualization comparing all models
        if len(detected_studies) >= 2:
            print(f"\n[OPTUNA] Generating summary comparison across all models...")
            try:
                summary_path = create_all_models_optuna_summary(
                    studies_dict=detected_studies,
                    results_path=section4_results_dir,
                    verbose=True
                )
                if summary_path:
                    print(f"[OK] Optuna summary saved to Section 4: {summary_path}")
                    # Add to all evaluated models
                    for model_name in evaluation_results:
                        if 'files_generated' not in evaluation_results[model_name]:
                            evaluation_results[model_name]['files_generated'] = []
                        if summary_path not in evaluation_results[model_name]['files_generated']:
                            evaluation_results[model_name]['files_generated'].append(summary_path)
            except Exception as e:
                print(f"[ERROR] Failed to generate Optuna summary: {e}")

        print(f"\n[OK] Optuna visualizations saved to: {section4_results_dir}")
    else:
        print("[INFO] No Optuna studies detected in notebook scope")
        print("      Studies are expected with names like: ctgan_study, ctabgan_study, etc.")

        # ADD COMPREHENSIVE TRTS ANALYSIS (SAME AS BOTH ORIGINAL FUNCTIONS)

    print(f"\n{'='*25} COMPREHENSIVE TRTS ANALYSIS {'='*25}")

    if len(available_models) >= 1:
        # Perform TRTS analysis for all models
        trts_results = {}

        for model_name, synthetic_data in available_models.items():
            print(f"\n[ANALYSIS] Running TRTS analysis for {model_name}...")

            try:
                # DEFENSIVE: Sanitize synthetic data before TRTS analysis
                sanitized_synthetic = sanitize_synthetic_data(
                    real_df=real_data,
                    synth_df=synthetic_data,
                    target_column=target_col,
                    task_type="auto",
                    verbose=False
                )

                trts_result = comprehensive_trts_analysis(
                    real_data=real_data,
                    synthetic_data=sanitized_synthetic,
                    target_column=target_col,
                    test_size=0.2,
                    random_state=42,
                    n_estimators=50 if section_number == 3 else 100,  # More thorough for optimized models
                    verbose=True,
                    store_predictions=True  # Phase 1: Enable for ROC/PR/Calibration curve generation
                )

                trts_results[model_name] = trts_result

                # Add TRTS results to evaluation results
                if model_name in evaluation_results:
                    evaluation_results[model_name]['trts_analysis'] = trts_result

            except Exception as e:
                print(f"[ERROR] TRTS analysis failed for {model_name}: {e}")
                trts_results[model_name] = {'error': str(e)}

        # Create TRTS visualizations
        if trts_results and any('error' not in result for result in trts_results.values()):
            try:
                results_dir = get_results_path(dataset_id, section_number)
                dataset_display_name = dataset_id.replace('-', ' ').title()
                suffix = " (Optimized Models)" if variable_pattern == 'final' else ""

                print(f"\n[CHART] Creating TRTS visualizations...")
                viz_results = create_trts_visualizations(
                    trts_results_dict=trts_results,
                    model_names=list(trts_results.keys()),
                    results_dir=results_dir,
                    dataset_name=f"{dataset_display_name}{suffix}",
                    save_files=True,
                    display_plots=False
                )

                if 'files_generated' in viz_results:
                    print(f"[OK] TRTS visualization files generated:")
                    for file_path in viz_results['files_generated']:
                        file_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
                        print(f"   [FOLDER] {file_name}")

                    # Add visualization files to results
                    for model_name in evaluation_results:
                        if 'files_generated' not in evaluation_results[model_name]:
                            evaluation_results[model_name]['files_generated'] = []
                        evaluation_results[model_name]['files_generated'].extend(viz_results['files_generated'])

                # Display TRTS summary
                if 'summary_stats' in viz_results:
                    stats = viz_results['summary_stats']
                    print(f"\n[STATS] TRTS Analysis Summary:")
                    print(f"   - Models analyzed: {stats.get('models_analyzed', 0)}")
                    print(f"   - Average combined score: {stats.get('avg_combined_score', 0):.4f}")
                    print(f"   - Best performing model: {stats.get('best_model', 'Unknown')}")
                    print(f"   - Total scenarios tested: {stats.get('total_scenarios_tested', 0)}")

                # Phase 3: Generate privacy analysis dashboard
                has_privacy_metrics = any(
                    'privacy' in model_results
                    for model_results in trts_results.values()
                )

                if has_privacy_metrics:
                    from src.visualization.section5 import create_privacy_dashboard

                    privacy_dash_result = create_privacy_dashboard(
                        trts_results_dict=trts_results,
                        model_names=list(trts_results.keys()),
                        results_dir=results_dir,
                        dataset_name=f"{dataset_display_name}{suffix}",
                        save_files=True,
                        display_plots=False,
                        verbose=True
                    )

                    if privacy_dash_result and 'files_generated' in privacy_dash_result:
                        for model_name in evaluation_results:
                            if 'files_generated' not in evaluation_results[model_name]:
                                evaluation_results[model_name]['files_generated'] = []
                            evaluation_results[model_name]['files_generated'].extend(privacy_dash_result['files_generated'])

                # Phase 2: Generate ROC/PR/Calibration curves (if predictions available)
                from src.visualization.section5 import (
                    create_trts_roc_curves,
                    create_trts_pr_curves,
                    create_trts_calibration_curves
                )

                roc_path = create_trts_roc_curves(
                    trts_results_dict=trts_results,
                    model_names=list(trts_results.keys()),
                    results_dir=results_dir,
                    dataset_name=f"{dataset_display_name}{suffix}",
                    save_files=True,
                    display_plots=False,
                    verbose=True
                )

                pr_path = create_trts_pr_curves(
                    trts_results_dict=trts_results,
                    model_names=list(trts_results.keys()),
                    results_dir=results_dir,
                    dataset_name=f"{dataset_display_name}{suffix}",
                    save_files=True,
                    display_plots=False,
                    verbose=True
                )

                calib_path = create_trts_calibration_curves(
                    trts_results_dict=trts_results,
                    model_names=list(trts_results.keys()),
                    results_dir=results_dir,
                    dataset_name=f"{dataset_display_name}{suffix}",
                    save_files=True,
                    display_plots=False,
                    verbose=True
                )

                # Add curve paths to evaluation results
                curve_paths = []
                if roc_path:
                    curve_paths.append(roc_path)
                if pr_path:
                    curve_paths.append(pr_path)
                if calib_path:
                    curve_paths.append(calib_path)

                if curve_paths:
                    for model_name in evaluation_results:
                        if 'files_generated' not in evaluation_results[model_name]:
                            evaluation_results[model_name]['files_generated'] = []
                        evaluation_results[model_name]['files_generated'].extend(curve_paths)

                # SDAC Visualizations (radar chart + heatmap)
                try:
                    from src.visualization.section5 import create_sdac_radar_chart, create_sdac_heatmap

                    if sdac_rows:
                        sdac_viz_df = pd.DataFrame(sdac_rows)
                        radar_path = create_sdac_radar_chart(
                            sdac_df=sdac_viz_df,
                            results_dir=results_dir,
                            dataset_name=f"{dataset_display_name}{suffix}",
                            save_files=True,
                            display_plots=False,
                            verbose=True
                        )
                        heatmap_path = create_sdac_heatmap(
                            sdac_df=sdac_viz_df,
                            results_dir=results_dir,
                            dataset_name=f"{dataset_display_name}{suffix}",
                            save_files=True,
                            display_plots=False,
                            verbose=True
                        )
                        for p in [radar_path, heatmap_path]:
                            if p:
                                for model_name in evaluation_results:
                                    if 'files_generated' not in evaluation_results[model_name]:
                                        evaluation_results[model_name]['files_generated'] = []
                                    evaluation_results[model_name]['files_generated'].append(p)
                except Exception as e:
                    print(f"[WARNING] SDAC visualizations failed: {e}")

            except Exception as e:
                print(f"[ERROR] TRTS visualization failed: {e}")

    else:
        print("[WARNING] Need at least 1 model for TRTS analysis")

    return evaluation_results


print("[OK] Batch evaluation functions loaded from src/evaluation/batch.py")
