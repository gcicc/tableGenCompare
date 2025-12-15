"""
Remove redundant manual Optuna visualization cells from notebooks.

Now that Optuna visualizations are auto-generated during batch evaluation,
these manual chunks are no longer needed.
"""

import json
import os
import sys

def should_remove_cell(cell):
    """
    Determine if a cell contains redundant Optuna visualization code.

    Returns True if the cell should be removed.
    """
    if cell.get('cell_type') != 'code':
        return False

    # Get cell source as a single string
    source = cell.get('source', [])
    if isinstance(source, list):
        source = ''.join(source)

    # Check if this is a manual Optuna visualization cell
    has_optuna_comment = '# Generate Optuna visualizations for' in source
    has_create_call = 'create_optuna_visualizations(' in source

    return has_optuna_comment and has_create_call


def clean_notebook(notebook_path):
    """
    Remove redundant Optuna visualization cells from a notebook.

    Returns: (total_cells_before, total_cells_after, cells_removed)
    """
    print(f"\n[PROCESSING] {os.path.basename(notebook_path)}")

    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        cells_before = len(notebook['cells'])

        # Filter out redundant cells
        cells_to_keep = []
        cells_removed = []

        for i, cell in enumerate(notebook['cells']):
            if should_remove_cell(cell):
                # Extract model name from comment for reporting
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = ''.join(source)

                # Find model name
                model_name = "UNKNOWN"
                if '# Generate Optuna visualizations for' in source:
                    for line in source.split('\n'):
                        if '# Generate Optuna visualizations for' in line:
                            model_name = line.split('for')[-1].strip()
                            break

                cells_removed.append(model_name)
                print(f"   [REMOVE] Cell {i+1}: Optuna visualization for {model_name}")
            else:
                cells_to_keep.append(cell)

        # Update notebook
        notebook['cells'] = cells_to_keep
        cells_after = len(notebook['cells'])

        # Save cleaned notebook
        if cells_removed:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)

            print(f"   [OK] Removed {len(cells_removed)} cells: {', '.join(cells_removed)}")
            print(f"   [OK] Cells: {cells_before} -> {cells_after}")
        else:
            print(f"   [INFO] No redundant Optuna cells found")

        return cells_before, cells_after, cells_removed

    except Exception as e:
        print(f"   [ERROR] Failed to process notebook: {e}")
        return 0, 0, []


def main():
    """Clean all notebooks in the project."""
    print("=" * 80)
    print("REMOVING REDUNDANT OPTUNA VISUALIZATION CELLS")
    print("=" * 80)
    print("\nReason: Optuna visualizations are now auto-generated during batch evaluation")
    print("        (Section 4.2) and saved to Section 4 results automatically.\n")

    # Find all notebooks (exclude submodules)
    notebook_files = []
    for root, dirs, files in os.walk('.'):
        # Skip git, CTAB-GAN, and CTAB-GAN-Plus directories
        dirs[:] = [d for d in dirs if d not in ['.git', 'CTAB-GAN', 'CTAB-GAN-Plus', '__pycache__']]

        for file in files:
            if file.endswith('.ipynb') and not file.startswith('.'):
                notebook_files.append(os.path.join(root, file))

    print(f"Found {len(notebook_files)} notebooks to process:\n")

    # Process each notebook
    total_cells_removed = 0
    notebooks_modified = 0

    for notebook_path in sorted(notebook_files):
        cells_before, cells_after, cells_removed = clean_notebook(notebook_path)

        if cells_removed:
            notebooks_modified += 1
            total_cells_removed += len(cells_removed)

    # Summary
    print("\n" + "=" * 80)
    print("CLEANUP SUMMARY")
    print("=" * 80)
    print(f"Notebooks processed: {len(notebook_files)}")
    print(f"Notebooks modified: {notebooks_modified}")
    print(f"Total cells removed: {total_cells_removed}")

    if total_cells_removed > 0:
        print(f"\n[OK] Successfully removed all redundant Optuna visualization cells!")
        print(f"     Optuna visualizations will now be auto-generated during batch evaluation.")
    else:
        print(f"\n[INFO] No redundant cells found. Notebooks are already clean!")

    print("=" * 80)


if __name__ == '__main__':
    main()
