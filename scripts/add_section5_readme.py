"""
Script to add Section 5 README generation to all notebooks
"""

import json
import re
import sys
from pathlib import Path

def create_readme_cell():
    """Create the Section 5 README generation cell"""
    code = """# Generate Section 5 README documentation
from src.utils.documentation import create_section5_readme

create_section5_readme(
    results_path=results_path,
    dataset_id=DATASET_IDENTIFIER,
    timestamp=SESSION_TIMESTAMP
)
"""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    }

def process_notebook(notebook_path):
    """Add Section 5 README generation cell to a notebook"""
    print(f"\n[NOTEBOOK] Processing: {notebook_path.name}")

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Look for Section 5 and find a good insertion point
    # We want to add it at the end of Section 5, after all visualizations
    section5_found = False
    last_section5_cell = -1

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] in ['markdown', 'code']:
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

            # Check if this is Section 5
            if re.search(r'#.*Section\s*5', source, re.IGNORECASE) or \
               re.search(r'Final.*Comparison', source, re.IGNORECASE):
                section5_found = True
                print(f"  Found Section 5 marker at cell {i}")

            # If in Section 5, keep track of last cell
            if section5_found:
                # Stop if we hit Section 6 or end
                if re.search(r'#.*Section\s*6', source, re.IGNORECASE):
                    break
                last_section5_cell = i

            # Check if README generation already exists
            if 'create_section5_readme' in source:
                print(f"  [INFO] Section 5 README generation already exists at cell {i} - skipping")
                return False

    if not section5_found:
        print(f"  [WARNING] Section 5 not found - skipping")
        return False

    if last_section5_cell == -1:
        print(f"  [WARNING] Could not determine insertion point - skipping")
        return False

    # Insert README cell at the end of Section 5
    insert_pos = last_section5_cell + 1
    readme_cell = create_readme_cell()
    nb['cells'].insert(insert_pos, readme_cell)
    print(f"  [OK] Added Section 5 README cell at position {insert_pos}")

    # Write modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  [OK] Saved {notebook_path.name}")
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
    print("Adding Section 5 README Generation to Notebooks (Task 4.2)")
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
    print("  2. Test by running Section 5 in one notebook")
    print("  3. Commit changes if satisfied")

    return success_count == len(notebooks)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
