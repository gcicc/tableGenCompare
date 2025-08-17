"""
Dataset Configuration Templates for Clinical Synthetic Data Generation Framework

This file contains the validated configurations for all 4 tested datasets.
Copy and paste the appropriate configuration into the USER CONFIGURATION section
of the Clinical_Synthetic_Data_Generation_Framework.ipynb notebook.
"""

# ===================================================================
# CONFIGURATION 1: BREAST CANCER WISCONSIN (CURRENT DEFAULT)
# ===================================================================
"""
# =================== USER CONFIGURATION ===================
# 📝 CONFIGURE YOUR DATASET: Update these settings for your data
DATA_FILE = 'data/Breast_cancer_data.csv'  # Path to your CSV file
TARGET_COLUMN = 'diagnosis'                 # Name of your target/outcome column

# 🔧 OPTIONAL ADVANCED SETTINGS (Auto-detected if left empty)
CATEGORICAL_COLUMNS = []                    # List categorical columns or leave empty for auto-detection
MISSING_STRATEGY = 'mice'                   # Options: 'mice', 'drop', 'median', 'mode'
DATASET_NAME = 'Breast Cancer Wisconsin'   # Descriptive name for your dataset

# 🚨 IMPORTANT: Verify these settings match your dataset before running!
print(f"📊 Configuration Summary:")
print(f"   Dataset: {DATASET_NAME}")
print(f"   File: {DATA_FILE}")
print(f"   Target: {TARGET_COLUMN}")
print(f"   Missing Data Strategy: {MISSING_STRATEGY}")
# =========================================================

# ✅ VALIDATION RESULTS:
# - Dataset Shape: (569, 6)
# - Target Column: 'diagnosis' (binary: 0/1)
# - Missing Data: None
# - Column Types: 5 continuous features + 1 binary target
# - Target Balance: 0.59 (well-balanced)
"""

# ===================================================================
# CONFIGURATION 2: PAKISTANI DIABETES DATASET
# ===================================================================
"""
# =================== USER CONFIGURATION ===================
# 📝 CONFIGURE YOUR DATASET: Update these settings for your data
DATA_FILE = 'data/Pakistani_Diabetes_Dataset.csv'  # Path to your CSV file
TARGET_COLUMN = 'Outcome'                          # Name of your target/outcome column

# 🔧 OPTIONAL ADVANCED SETTINGS (Auto-detected if left empty)
CATEGORICAL_COLUMNS = ['Gender', 'Rgn']            # List categorical columns or leave empty for auto-detection
MISSING_STRATEGY = 'mice'                          # Options: 'mice', 'drop', 'median', 'mode'
DATASET_NAME = 'Pakistani Diabetes'               # Descriptive name for your dataset

# 🚨 IMPORTANT: Verify these settings match your dataset before running!
print(f"📊 Configuration Summary:")
print(f"   Dataset: {DATASET_NAME}")
print(f"   File: {DATA_FILE}")
print(f"   Target: {TARGET_COLUMN}")
print(f"   Missing Data Strategy: {MISSING_STRATEGY}")
# =========================================================

# ✅ VALIDATION RESULTS:
# - Dataset Shape: (912, 19)
# - Target Column: 'outcome' (after standardization, binary: 0/1)
# - Missing Data: None
# - Column Types: 11 continuous + 8 binary features
# - Target Balance: 0.88 (well-balanced)
# - Note: Gender and Rgn columns are detected as binary, but can be treated as categorical
"""

# ===================================================================
# CONFIGURATION 3: ALZHEIMER'S DISEASE DATASET
# ===================================================================
"""
# =================== USER CONFIGURATION ===================
# 📝 CONFIGURE YOUR DATASET: Update these settings for your data
DATA_FILE = 'data/alzheimers_disease_data.csv'                    # Path to your CSV file
TARGET_COLUMN = 'Diagnosis'                                       # Name of your target/outcome column

# 🔧 OPTIONAL ADVANCED SETTINGS (Auto-detected if left empty)
CATEGORICAL_COLUMNS = ['Gender', 'Ethnicity', 'EducationLevel']   # List categorical columns or leave empty for auto-detection
MISSING_STRATEGY = 'mice'                                         # Options: 'mice', 'drop', 'median', 'mode'
DATASET_NAME = 'Alzheimers Disease'                              # Descriptive name for your dataset

# 🚨 IMPORTANT: Verify these settings match your dataset before running!
print(f"📊 Configuration Summary:")
print(f"   Dataset: {DATASET_NAME}")
print(f"   File: {DATA_FILE}")
print(f"   Target: {TARGET_COLUMN}")
print(f"   Missing Data Strategy: {MISSING_STRATEGY}")
# =========================================================

# ✅ VALIDATION RESULTS:
# - Dataset Shape: (2149, 35)
# - Target Column: 'diagnosis' (after standardization, binary: 0/1)
# - Missing Data: None
# - Column Types: 16 continuous + 16 binary + 3 categorical features
# - Target Balance: 0.55 (moderately balanced)
# - Note: Large feature set with mixed data types
"""

# ===================================================================
# CONFIGURATION 4: LIVER DISEASE DATASET
# ===================================================================
"""
# =================== USER CONFIGURATION ===================
# 📝 CONFIGURE YOUR DATASET: Update these settings for your data
DATA_FILE = 'data/liver_train.csv'         # Path to your CSV file
TARGET_COLUMN = 'Result'                   # Name of your target/outcome column

# 🔧 OPTIONAL ADVANCED SETTINGS (Auto-detected if left empty)
CATEGORICAL_COLUMNS = ['Gender of the patient']  # List categorical columns or leave empty for auto-detection
MISSING_STRATEGY = 'mice'                  # Options: 'mice', 'drop', 'median', 'mode'
DATASET_NAME = 'Liver Disease'             # Descriptive name for your dataset

# 🚨 IMPORTANT: Verify these settings match your dataset before running!
print(f"📊 Configuration Summary:")
print(f"   Dataset: {DATASET_NAME}")
print(f"   File: {DATA_FILE}")
print(f"   Target: {TARGET_COLUMN}")
print(f"   Missing Data Strategy: {MISSING_STRATEGY}")
# =========================================================

# ✅ VALIDATION RESULTS:
# - Dataset Shape: (30691, 11)
# - Target Column: 'result' (after standardization, binary: 1/2)
# - Missing Data: 1.6% across 10 columns (manageable with MICE)
# - Column Types: 9 continuous + 1 categorical + 1 binary target
# - Target Balance: 0.40 (imbalanced but workable)
# - Note: Largest dataset, requires latin-1 encoding, has missing data
# - Special: Non-standard target values (1/2 instead of 0/1)
"""

# ===================================================================
# CONFIGURATION VALIDATION SUMMARY
# ===================================================================
"""
ALL 4 DATASETS SUCCESSFULLY TESTED WITH SECTIONS 1-3

✅ Framework Robustness Confirmed:
- Handles diverse dataset sizes (569 to 30,691 rows)
- Manages different column naming conventions
- Automatically standardizes column names
- Detects target columns correctly
- Handles missing data appropriately
- Categorizes column types accurately
- Works with various encodings (UTF-8, Latin-1)

🔧 Key Framework Features Validated:
1. Column name standardization (removes special chars, lowercase)
2. Target column auto-detection with fallback options
3. Column type analysis (continuous, categorical, binary)
4. Missing data assessment and strategy selection
5. Configuration validation with helpful error messages
6. Robust data loading with encoding detection

⚠️ Important Notes:
- Liver dataset requires latin-1 encoding (handled automatically)
- Some categorical columns may be detected as binary (manual override available)
- Framework handles both 0/1 and 1/2 target encodings
- MICE imputation recommended for datasets with missing data
"""

if __name__ == "__main__":
    print("Dataset Configuration Templates for Clinical Synthetic Data Generation Framework")
    print("="*80)
    print("Copy the appropriate configuration block into your notebook's USER CONFIGURATION section.")
    print("\nValidated Datasets:")
    print("1. Breast Cancer Wisconsin - 569 rows, 6 columns, no missing data")
    print("2. Pakistani Diabetes - 912 rows, 19 columns, no missing data") 
    print("3. Alzheimer's Disease - 2149 rows, 35 columns, no missing data")
    print("4. Liver Disease - 30691 rows, 11 columns, 1.6% missing data")
    print("\nAll datasets successfully pass sections 1-3 validation!")