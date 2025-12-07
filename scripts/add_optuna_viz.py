"""
Script to add Optuna visualization calls to all notebooks after study.optimize() calls
"""

import json
import re
import sys
from pathlib import Path

# Model mapping: study_name -> model display name
MODEL_NAMES = {
    'ctgan': 'CTGAN',
    'ctabgan': 'CTABGAN',
    'ctabganplus': 'CTABGANPLUS',
    'ganeraid': 'GANERAID',
    'copulagan': 'COPULAGAN',
    'tvae': 'TVAE'
}

def create_viz_cell(study_var, model_name):
    """Create a visualization cell for a specific study"""
    code = f"""# Generate Optuna visualizations for {model_name}
from src.visualization.section4 import create_optuna_visualizations

create_optuna_visualizations(
    study={study_var}_study,
    model_name='{model_name}',
    results_path=results_path,
    verbose=True
)
"""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    }

def create_summary_cell():
    """Create the summary visualization cell"""
    code = """# Create Optuna summary comparing all models
from src.visualization.section4 import create_all_models_optuna_summary

# Collect all studies (only include those that completed successfully)
studies_dict = {}
if 'ctgan_study' in locals():
    studies_dict['CTGAN'] = ctgan_study
if 'ctabgan_study' in locals():
    studies_dict['CTABGAN'] = ctabgan_study
if 'ctabganplus_study' in locals():
    studies_dict['CTABGAN+'] = ctabganplus_study
if 'ganeraid_study' in locals():
    studies_dict['GANerAid'] = ganeraid_study
if 'copulagan_study' in locals():
    studies_dict['CopulaGAN'] = copulagan_study
if 'tvae_study' in locals():
    studies_dict['TVAE'] = tvae_study

if studies_dict:
    create_all_models_optuna_summary(
        studies_dict=studies_dict,
        results_path=results_path,
        verbose=True
    )
    print(f"\\n[OK] Optuna summary visualization created for {len(studies_dict)} models")
else:
    print("[WARNING] No Optuna studies available for summary visualization")
"""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    }

def process_notebook(notebook_path):
    """Add visualization cells to a notebook"""
    print(f"\n[NOTEBOOK] Processing: {notebook_path.name}")

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find cells with study.optimize and track insertion points
    insertions = []  # List of (index, study_name, model_name)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Look for study.optimize calls
            match = re.search(r'(\w+)_study\.optimize', source)
            if match:
                study_name = match.group(1).lower()
                if study_name in MODEL_NAMES:
                    model_name = MODEL_NAMES[study_name]
                    insertions.append((i, study_name, model_name))
                    print(f"  Found: {study_name}_study.optimize at cell {i}")

    if not insertions:
        print(f"  [WARNING] No study.optimize calls found - skipping")
        return False

    # Insert visualization cells (work backwards to preserve indices)
    cells_added = 0
    for idx, study_name, model_name in reversed(insertions):
        # Insert after the optimize cell
        insert_pos = idx + 1
        viz_cell = create_viz_cell(study_name, model_name)
        nb['cells'].insert(insert_pos, viz_cell)
        cells_added += 1
        print(f"  [OK] Added viz cell for {model_name} after cell {idx}")

    # Add summary cell at the end of the last optimization
    if insertions:
        # Find the position after the last optimization
        last_idx = insertions[-1][0]  # This is before we inserted cells
        # Account for all the cells we inserted
        summary_pos = last_idx + cells_added + 1
        summary_cell = create_summary_cell()
        nb['cells'].insert(summary_pos, summary_cell)
        print(f"  [OK] Added summary cell at position {summary_pos}")
        cells_added += 1

    # Write modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  [OK] Saved {notebook_path.name} ({cells_added} cells added)")
    return True

def main():
    # List of notebooks to process
    notebooks = [
        'SynthethicTableGenerator-Alzheimer.ipynb',
        'SynthethicTableGenerator-BreastCancer.ipynb',
        'SynthethicTableGenerator-Liver.ipynb',
        'SynthethicTableGenerator-Pakistani.ipynb',
        'STG-BreastCancerV2.ipynb',
        'STG-LiverV2.ipynb',
        'STG-PakistaniV2.ipynb'
    ]

    base_path = Path('.')

    print("=" * 60)
    print("Adding Optuna Visualizations to Notebooks (Task 4.1)")
    print("=" * 60)

    success_count = 0
    for nb_name in notebooks:
        nb_path = base_path / nb_name
        if not nb_path.exists():
            print(f"\n[WARNING] Notebook not found: {nb_name}")
            continue

        if process_notebook(nb_path):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"[COMPLETE] Successfully updated {success_count}/{len(notebooks)} notebooks")
    print("=" * 60)
    print("\n[NEXT STEPS]:")
    print("  1. Review changes in notebooks")
    print("  2. Test by running Section 4 in one notebook")
    print("  3. Commit changes if satisfied")

    return success_count == len(notebooks)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
