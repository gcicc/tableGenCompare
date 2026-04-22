# Datasets

Four single-table healthcare benchmarks, selected to stress generators along
different schema shapes: small-wide-numeric (breast cancer), medium-mixed
(Alzheimer), small-mixed (diabetes), and large-tall-numeric (liver). All
classification targets; all kept on-repo and laptop-tractable.

| Dataset | File | Rows | Cols | Num | Cat | Target | Task | Driver notebook |
|---------|------|------|------|-----|-----|--------|------|-----------------|
| Breast Cancer (WBCD) | `data/Breast_cancer_data.csv` | 569 | 6 | 6 | 0 | `diagnosis` (binary) | classification | `STG-Driver-breast-cancer2.ipynb` |
| Alzheimer | `data/alzheimers_disease_data.csv` | 2,149 | 35 | 34 | 1 | `Diagnosis` (binary) | classification | `STG-Driver-Alzheimer2.ipynb` |
| Pakistani Diabetes | `data/Pakistani_Diabetes_Dataset.csv` | 912 | 19 | 19 | 0 | `Outcome` (binary) | classification | `STG-Driver-diabetes2.ipynb` |
| Indian Liver | `data/liver_train.csv` | 30,691 | 11 | 11 | 0 | `Result` (binary) | classification | `STG-Driver-liver-train2.ipynb` |

---

## 1. Breast Cancer (WBCD subset) — *primary POC*

**What it is.** The classical five-feature subset of the Wisconsin Breast
Cancer Diagnostic dataset. Each row is one fine-needle aspirate of a breast
mass.

**Scale.** 569 rows × 6 columns. Binary target, mild class imbalance
(~37% malignant).

**Schema.** All continuous: `mean_radius`, `mean_texture`, `mean_perimeter`,
`mean_area`, `mean_smoothness`. Target `diagnosis` (0 = benign, 1 = malignant).

**Why it's the primary POC.**
- Smallest and fastest to iterate on laptop hardware.
- Contains the textbook near-deterministic triple
  `mean_radius` / `mean_perimeter` / `mean_area` (pairwise |association| ≥ 0.985),
  which makes it the canonical validation case for the collinearity reducer
  (see [collinearity-reduction.md](collinearity-reduction.md)).
- Canonical UCI benchmark — any quality issue is immediately recognizable to
  reviewers from outside the project.

**License.** UCI ML Repository terms — public, attribution required.

**Source:** [UCI: Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

---

## 2. Alzheimer — *mixed-schema bench*

**What it is.** A simulated cohort of 2,149 patients with demographic,
lifestyle, cardiovascular, cognitive, and symptom fields plus a binary
Alzheimer's diagnosis. Good stand-in for a pharma-style clinical-trial row
schema without CDISC formatting.

**Scale.** 2,149 rows × 35 columns. Binary target; ~35% positive.

**Schema highlights.**
- **Demographic / lifestyle:** `Age`, `Gender`, `Ethnicity`, `EducationLevel`,
  `BMI`, `Smoking`, `AlcoholConsumption`, `PhysicalActivity`, `DietQuality`.
- **Clinical:** `SystolicBP`, `DiastolicBP`, `CholesterolTotal`,
  `CholesterolLDL`, `CholesterolHDL`, `CholesterolTriglycerides`.
- **Cognitive / functional:** `MMSE`, `FunctionalAssessment`, `ADL`.
- **Symptoms (binary):** `MemoryComplaints`, `BehavioralProblems`, `Confusion`,
  `Disorientation`, `PersonalityChanges`, `DifficultyCompletingTasks`,
  `Forgetfulness`.
- **Target:** `Diagnosis` (0 = no AD, 1 = AD).
- **Ignored:** `PatientID`, `DoctorInCharge` (constant `XXXConfid` placeholder).

**Why it's a good bench.**
- Richest feature mix (34 numeric / 1 categorical) of the four; exercises
  mixed-association scoring.
- Large enough for the scorecard to exhibit stable differences between
  generators.

**Source:** [Kaggle: Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

---

## 3. Pakistani Diabetes — *small-mixed bench*

**What it is.** 912 Pakistani adult patients screened for type-2 diabetes.
Originally tabular with short, partially-abbreviated column names preserved
from the source.

**Scale.** 912 rows × 19 columns. Binary `Outcome`; ~50% positive (fairly
balanced — the only balanced dataset of the four).

**Schema.**
- **Demographic:** `Age`, `Gender`, `Rgn` (region), `wt`, `BMI`, `wst` (waist).
- **Vitals:** `sys` (systolic BP), `dia` (diastolic BP).
- **Labs / clinical:** `A1c`, `B.S.R` (blood sugar random), `HDL`, `Dur`
  (duration), `his` (family history), `vision`, `Exr` (exercise), `dipsia`
  (polydipsia), `uria` (polyuria), `neph` (nephropathy).
- **Target:** `Outcome` (0 = non-diabetic, 1 = diabetic).

**Why it's a good bench.**
- Class-balanced — fairness and ML-efficacy axes don't get confounded by
  marginal imbalance.
- Small enough for smoke runs but enough columns to stress mixed-association
  metrics.
- Real-world noise: abbreviated / non-standard column names test the
  preprocessor's column-name standardization path.

**Source:** [Kaggle: Pakistani Diabetes Dataset](https://www.kaggle.com/datasets/mhrzn/pakistani-diabetes-dataset)

---

## 4. Indian Liver Patient — *scale bench*

**What it is.** 30,691 liver-function panel rows for classifying liver disease.
Largest dataset in the suite by an order of magnitude.

**Scale.** 30,691 rows × 11 columns. Binary `Result`; strongly imbalanced
toward positives.

**Schema.** All continuous except `gender_of_the_patient`:
- `Age of the patient`, `Total Bilirubin`, `Direct Bilirubin`,
  `Alkphos Alkaline Phosphotase`, `Sgpt Alamine Aminotransferase`,
  `Sgot Aspartate Aminotransferase`, `Total Protiens`, `ALB Albumin`,
  `A/G Ratio Albumin and Globulin Ratio`.
- Target `Result` (1 = liver disease).

**Why it's a good bench.**
- 30k rows stresses generator training time and scales the HPO budget
  meaningfully.
- Column names contain spaces — exercises the notebook's column-name
  standardization contract.
- Bilirubin / enzyme ratios give at least one moderately-correlated pair for
  the collinearity reducer to consider.

**Source:** UCI / Indian Liver Patient Dataset.

---

## Repo conventions

- Target column is binary integer (0/1) in every dataset. Notebook's
  `task_type: "classification"` is set explicitly in `NOTEBOOK_CONFIG`.
- Categorical columns are enumerated via `NOTEBOOK_CONFIG["categorical_columns"]`
  per driver; empty list means "nothing forced — auto-detect everything as
  continuous" (breast-cancer, diabetes, liver).
- Datasets are checked in. No external download step. Preprocessing happens
  entirely inside §2.2 (`load_and_preprocess_from_config`).
- Column names are standardized to `lower_snake_case` by the preprocessor, so
  downstream code references the cleaned names (e.g. `result`, not
  `Result`).

## Dataset-to-application matrix

| Dataset | Preferred use |
|---------|---------------|
| Breast Cancer | Primary quick-iteration benchmark; collinearity validation (perimeter/radius/area). |
| Alzheimer | Mixed-type scoring, fairness axis (if `Gender` / `Ethnicity` selected as `protected_col`). |
| Pakistani Diabetes | Class-balanced ML-efficacy comparison; fairness without imbalance confound. |
| Indian Liver | Scale / HPO-budget test; notebook runtime characterization.  |
