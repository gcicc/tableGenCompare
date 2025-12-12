"""
Parameter management functions for Section 4 & 5 integration.

Migrated from setup.py (Phase 4, Task 4.3) - streamlining setup.py
Functions for saving and loading best hyperparameters from Optuna optimization.
"""

import os
import pandas as pd

# Import required utilities from src
from src.config import DATASET_IDENTIFIER
from src.utils.paths import extract_dataset_identifier, get_results_path


def save_best_parameters_to_csv(scope=None, section_number=4, dataset_identifier=None):
    """
    Save all best hyperparameters from Section 4 optimization to CSV format.

    Parameters:
    - scope: Notebook scope (globals()) to access study variables
    - section_number: Section number for file organization (default 4)
    - dataset_identifier: Dataset name for folder structure

    Returns:
    - Dictionary with save results and file path
    """

    if scope is None:
        scope = globals()

    # Auto-detect dataset identifier
    if dataset_identifier is None:
        if 'DATASET_IDENTIFIER' in scope and scope['DATASET_IDENTIFIER']:
            dataset_identifier = scope['DATASET_IDENTIFIER']
        elif DATASET_IDENTIFIER:
            dataset_identifier = DATASET_IDENTIFIER
        else:
            # Fallback extraction
            for var_name in ['data_file', 'DATA_FILE', 'current_data_file']:
                if var_name in scope and scope[var_name]:
                    dataset_identifier = extract_dataset_identifier(scope[var_name])
                    break
            if not dataset_identifier:
                dataset_identifier = 'unknown-dataset'

    # Get results directory
    results_dir = get_results_path(dataset_identifier, section_number)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[SAVE] SAVING BEST PARAMETERS FROM SECTION {section_number}")
    print("=" * 60)
    print(f"[FOLDER] Target directory: {results_dir}")

    # Model study variable mappings
    model_studies = {
        'CTGAN': 'ctgan_study',
        'CTAB-GAN': 'ctabgan_study',
        'CTAB-GAN+': 'ctabganplus_study',
        'GANerAid': 'ganeraid_study',
        'CopulaGAN': 'copulagan_study',
        'TVAE': 'tvae_study'
    }

    parameter_rows = []
    summary_rows = []

    for model_name, study_var in model_studies.items():
        print(f"\n[CHART] Processing {model_name} parameters...")

        try:
            if study_var in scope and scope[study_var] is not None:
                study = scope[study_var]

                if hasattr(study, 'best_trial') and study.best_trial:
                    best_trial = study.best_trial
                    best_params = best_trial.params
                    best_score = best_trial.value
                    trial_number = best_trial.number

                    print(f"[OK] Found {model_name}: {len(best_params)} parameters, score: {best_score:.4f}")

                    # Flatten parameters for CSV format
                    for param_name, param_value in best_params.items():
                        # Handle complex parameter types
                        param_type = type(param_value).__name__

                        # Convert tuples/lists to string representation
                        if isinstance(param_value, (tuple, list)):
                            # Also save individual components for tuple parameters
                            if isinstance(param_value, tuple) and len(param_value) == 2:
                                # Common case: betas=(0.5, 0.9) becomes betas_0=0.5, betas_1=0.9
                                parameter_rows.append({
                                    'model_name': model_name,
                                    'parameter_name': f'{param_name}_0',
                                    'parameter_value': param_value[0],
                                    'parameter_type': type(param_value[0]).__name__,
                                    'best_score': best_score,
                                    'trial_number': trial_number,
                                    'original_param': param_name,
                                    'is_component': True
                                })
                                parameter_rows.append({
                                    'model_name': model_name,
                                    'parameter_name': f'{param_name}_1',
                                    'parameter_value': param_value[1],
                                    'parameter_type': type(param_value[1]).__name__,
                                    'best_score': best_score,
                                    'trial_number': trial_number,
                                    'original_param': param_name,
                                    'is_component': True
                                })

                            # Always save full tuple/list as string
                            param_value_str = str(param_value)
                        else:
                            param_value_str = param_value

                        parameter_rows.append({
                            'model_name': model_name,
                            'parameter_name': param_name,
                            'parameter_value': param_value_str,
                            'parameter_type': param_type,
                            'best_score': best_score,
                            'trial_number': trial_number,
                            'original_param': param_name,
                            'is_component': False
                        })

                    # Add summary row
                    summary_rows.append({
                        'model_name': model_name,
                        'best_score': best_score,
                        'trial_number': trial_number,
                        'num_parameters': len(best_params),
                        'study_variable': study_var,
                        'parameters_saved': len(best_params)
                    })

                else:
                    print(f"[WARNING]  {model_name}: No best_trial found")

            else:
                print(f"[WARNING]  {model_name}: Study variable '{study_var}' not found")

        except Exception as e:
            print(f"[ERROR] {model_name}: Error processing - {str(e)}")

    # Save results to CSV files
    files_saved = []

    if parameter_rows:
        # Main parameters file
        params_df = pd.DataFrame(parameter_rows)
        params_file = f"{results_dir}/best_parameters.csv"
        params_df.to_csv(params_file, index=False)
        files_saved.append(params_file)
        print(f"\n[OK] Parameters saved: {params_file}")
        print(f"   - Total parameter entries: {len(parameter_rows)}")

        # Summary file
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = f"{results_dir}/best_parameters_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            files_saved.append(summary_file)
            print(f"[OK] Summary saved: {summary_file}")
            print(f"   - Models processed: {len(summary_rows)}")

    else:
        print("[ERROR] No parameters found to save!")
        return {
            'success': False,
            'message': 'No parameters found',
            'files_saved': []
        }

    print(f"\n[SAVE] Parameter saving completed!")
    print(f"[FOLDER] Files saved to: {results_dir}")

    return {
        'success': True,
        'files_saved': files_saved,
        'parameters_count': len(parameter_rows),
        'models_count': len(summary_rows),
        'results_dir': results_dir
    }


def load_best_parameters_from_csv(section_number=4, dataset_identifier=None, fallback_to_memory=True, scope=None):
    """
    Load best hyperparameters from CSV files with memory fallback.

    Parameters:
    - section_number: Section number for file location (default 4)
    - dataset_identifier: Dataset name for folder structure
    - fallback_to_memory: Use in-memory study variables if CSV not found
    - scope: Notebook scope (globals()) for memory fallback

    Returns:
    - Dictionary with parameters for each model: {'ctgan': {...}, 'ctabgan': {...}}
    """

    if scope is None:
        scope = globals()

    # Auto-detect dataset identifier
    if dataset_identifier is None:
        if 'DATASET_IDENTIFIER' in scope and scope['DATASET_IDENTIFIER']:
            dataset_identifier = scope['DATASET_IDENTIFIER']
        elif DATASET_IDENTIFIER:
            dataset_identifier = DATASET_IDENTIFIER
        else:
            # Fallback extraction
            for var_name in ['data_file', 'DATA_FILE', 'current_data_file']:
                if var_name in scope and scope[var_name]:
                    dataset_identifier = extract_dataset_identifier(scope[var_name])
                    break
            if not dataset_identifier:
                dataset_identifier = 'unknown-dataset'

    # Get results directory
    results_dir = get_results_path(dataset_identifier, section_number)
    params_file = f"{results_dir}/best_parameters.csv"

    print(f"[LOAD] LOADING BEST PARAMETERS FROM SECTION {section_number}")
    print("=" * 60)
    print(f"[FOLDER] Looking for: {params_file}")

    parameters = {}
    load_source = "unknown"

    # Try loading from CSV first
    if os.path.exists(params_file):
        try:
            print(f"[OK] Found parameter CSV file")
            params_df = pd.read_csv(params_file)

            # Reconstruct parameter dictionaries per model
            for model_name in params_df['model_name'].unique():
                model_params = {}
                model_data = params_df[params_df['model_name'] == model_name]

                # Group parameters, handling tuple reconstruction
                for _, row in model_data.iterrows():
                    param_name = row['parameter_name']
                    param_value = row['parameter_value']
                    param_type = row['parameter_type']
                    is_component = row.get('is_component', False)
                    original_param = row.get('original_param', param_name)

                    # Skip component entries - we'll reconstruct tuples from full entries
                    if is_component:
                        continue

                    # Type conversion
                    if param_type == 'int':
                        param_value = int(param_value)
                    elif param_type == 'float':
                        param_value = float(param_value)
                    elif param_type == 'bool':
                        param_value = str(param_value).lower() in ['true', '1', 'yes']
                    elif param_type == 'tuple':
                        # Reconstruct tuple from string representation
                        try:
                            param_value = eval(param_value)  # Safe for controlled parameter data
                        except:
                            param_value = str(param_value)
                    # str and other types use as-is

                    model_params[param_name] = param_value

                # Map model name to standard format
                model_key = model_name.lower().replace('-', '').replace('+', 'plus')
                parameters[model_key] = model_params

                print(f"[OK] Loaded {model_name}: {len(model_params)} parameters")

            load_source = "CSV file"

        except Exception as e:
            print(f"[ERROR] Error reading CSV file: {str(e)}")
            parameters = {}

    else:
        print(f"[WARNING]  Parameter CSV file not found")

    # Fallback to memory if CSV loading failed or not found
    if not parameters and fallback_to_memory:
        print(f"\n[PROCESS] Falling back to in-memory study variables...")

        model_studies = {
            'CTGAN': ('ctgan_study', 'ctgan'),
            'CTAB-GAN': ('ctabgan_study', 'ctabgan'),
            'CTAB-GAN+': ('ctabganplus_study', 'ctabganplus'),
            'GANerAid': ('ganeraid_study', 'ganeraid'),
            'CopulaGAN': ('copulagan_study', 'copulagan'),
            'TVAE': ('tvae_study', 'tvae')
        }

        memory_loaded = 0
        for model_name, (study_var, model_key) in model_studies.items():
            if study_var in scope and scope[study_var] is not None:
                study = scope[study_var]
                if hasattr(study, 'best_trial') and study.best_trial:
                    parameters[model_key] = study.best_trial.params
                    memory_loaded += 1
                    print(f"[OK] {model_name}: Loaded from memory ({len(study.best_trial.params)} params)")

        if memory_loaded > 0:
            load_source = "memory fallback"
        else:
            print(f"[ERROR] No parameters found in memory either")
            load_source = "none"

    print(f"\n[LOAD] Parameter loading completed!")
    print(f"[SEARCH] Source: {load_source}")
    print(f"[CHART] Models loaded: {len(parameters)}")
    for model_key, params in parameters.items():
        print(f"   - {model_key}: {len(params)} parameters")

    return {
        'parameters': parameters,
        'source': load_source,
        'models_count': len(parameters),
        'file_path': params_file if load_source == "CSV file" else None
    }


def get_model_parameters(model_name, section_number=4, dataset_identifier=None, scope=None):
    """
    Unified parameter retrieval for a specific model with CSV/memory fallback.

    Parameters:
    - model_name: Model name ('ctgan', 'ctabgan', etc.)
    - section_number: Section number for file location
    - dataset_identifier: Dataset name for folder structure
    - scope: Notebook scope for memory fallback

    Returns:
    - Dictionary with model parameters or None if not found
    """

    # Load all parameters
    param_data = load_best_parameters_from_csv(
        section_number=section_number,
        dataset_identifier=dataset_identifier,
        fallback_to_memory=True,
        scope=scope
    )

    # Normalize model name
    model_key = model_name.lower().replace('-', '').replace('+', 'plus')

    if model_key in param_data['parameters']:
        print(f"[OK] {model_name.upper()} parameters loaded from {param_data['source']}")
        return param_data['parameters'][model_key]
    else:
        print(f"[ERROR] {model_name.upper()} parameters not found")
        return None


def compare_parameters_sources(scope=None, section_number=4, dataset_identifier=None, verbose=True):
    """
    Compare parameters between CSV files and in-memory study variables.

    Parameters:
    - scope: Notebook scope (globals()) for memory access
    - section_number: Section number for CSV location
    - dataset_identifier: Dataset name for folder structure
    - verbose: Print detailed comparison results

    Returns:
    - Dictionary with comparison results
    """

    if scope is None:
        scope = globals()

    if verbose:
        print(f"[SEARCH] COMPARING PARAMETER SOURCES")
        print("=" * 50)

    # Load from CSV (without memory fallback)
    csv_data = load_best_parameters_from_csv(
        section_number=section_number,
        dataset_identifier=dataset_identifier,
        fallback_to_memory=False,
        scope=scope
    )

    # Load from memory directly
    memory_params = {}
    model_studies = {
        'ctgan': 'ctgan_study',
        'ctabgan': 'ctabgan_study',
        'ctabganplus': 'ctabganplus_study',
        'ganeraid': 'ganeraid_study',
        'copulagan': 'copulagan_study',
        'tvae': 'tvae_study'
    }

    for model_key, study_var in model_studies.items():
        if study_var in scope and scope[study_var] is not None:
            study = scope[study_var]
            if hasattr(study, 'best_trial') and study.best_trial:
                memory_params[model_key] = study.best_trial.params

    # Compare results
    comparison_results = {
        'csv_available': csv_data['source'] == "CSV file",
        'memory_available': len(memory_params) > 0,
        'models_in_csv': list(csv_data['parameters'].keys()) if csv_data['source'] == "CSV file" else [],
        'models_in_memory': list(memory_params.keys()),
        'matches': {},
        'differences': {}
    }

    if verbose:
        print(f"[FOLDER] CSV source: {csv_data['source']}")
        print(f"[MEMORY] Memory models: {len(memory_params)}")

    # Check for matches and differences
    all_models = set(csv_data['parameters'].keys()) | set(memory_params.keys())

    for model_key in all_models:
        csv_params = csv_data['parameters'].get(model_key, {})
        mem_params = memory_params.get(model_key, {})

        if csv_params and mem_params:
            # Compare parameters
            matches = {}
            differences = {}

            all_param_keys = set(csv_params.keys()) | set(mem_params.keys())
            for param_key in all_param_keys:
                csv_val = csv_params.get(param_key)
                mem_val = mem_params.get(param_key)

                if csv_val == mem_val:
                    matches[param_key] = csv_val
                else:
                    differences[param_key] = {'csv': csv_val, 'memory': mem_val}

            comparison_results['matches'][model_key] = matches
            comparison_results['differences'][model_key] = differences

            if verbose:
                match_pct = len(matches) / len(all_param_keys) * 100 if all_param_keys else 0
                print(f"   - {model_key.upper()}: {match_pct:.1f}% match ({len(matches)}/{len(all_param_keys)} params)")
                if differences and verbose:
                    print(f"     Differences: {list(differences.keys())}")

        elif csv_params:
            if verbose:
                print(f"   - {model_key.upper()}: CSV only ({len(csv_params)} params)")
        elif mem_params:
            if verbose:
                print(f"   - {model_key.upper()}: Memory only ({len(mem_params)} params)")

    return comparison_results


print("[OK] Parameter management functions loaded from src/utils/parameters.py")
