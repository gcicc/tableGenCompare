# Setting Up an AWS SageMaker Notebook Instance for CTAB-GAN / Deep Tabular GANs

These steps cover creating the notebook instance, cloning the repo + submodules, creating a Python 3.10 conda env, installing pinned requirements, and registering a Jupyter kernel.

---

## 1) Accessing AWS and SageMaker

1. Log in to AWS via Okta: **AWS TEC**
2. Select your environment: `tec-rnd-sqs-dev`
3. Open **Amazon SageMaker AI**
4. Go to **Notebook instances**

---

## 2) Create the Notebook Instance

1. Click **Create notebook instance**
2. Recommended settings:
   - **Instance type:** `ml.g4dn.xlarge`
   - **Volume:** 50 GB (or more if datasets/models are large)
   - **IAM role:** use an existing role that worked previously
3. Under **Git repositories**, add:
   - `https://github.com/gcicc/tableGenCompare.git`
   - **Branch:** choose your working branch (example: `AWS_Round2_envfix`)
4. Create and wait until status is **InService**
5. Click **Open JupyterLab**

---

# Project Setup Instructions

## First-time setup (per notebook instance)

### 1) Open a Terminal in JupyterLab

Do not start Jupyter from the terminal; use the SageMaker-managed JupyterLab UI.

### 2) Initialize conda for the terminal session (if needed)

```bash
source ~/anaconda3/bin/activate
conda init bash
exec bash
```

### 3) Create a conda environment (Python 3.10)
Python 3.10 avoids wheel/build issues and worked with the pinned stack.
``` bash
conda create -n tablegen python=3.10 -y
conda activate tablegen
python -m pip install -U pip setuptools wheel
```

### 4) Go to the repo and initialize submodules (CRITICAL)
CTAB-GAN / CTAB-GAN-Plus / GANerAid are tracked as submodules.

```bash
cd ~/tableGenCompare
git submodule update --init --recursive
```

### 5) Install dependencies (single source of truth: requirements.txt)

pip install -r requirements.txt

### 6) Register the kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name tablegen --display-name "Python (tablegen)"
```

### 7) In JupyterLab
Kernel → Change Kernel → Python (tablegen)

### Returning users (each session)
```bash
source ~/anaconda3/bin/activate
conda activate tablegen
cd ~/tableGenCompare
```

### Notes / gotchas (based on issues encountered)

* Do not use Python 3.11+ for this project on classic Notebook Instances unless you control the image/toolchain.
* Do not pin ipython==9.3.0 (invalid). Let Jupyter manage it, or pin ipython<9 only if needed.
* scikit-learn must be pinned to a CTAB-compatible version (we used scikit-learn==1.2.2).
* CTAB-GAN / CTAB-GAN-Plus / GANerAid must be fetched via git submodule update --init --recursive (not pip install).
* If you work on GPU models (GANerAid), instantiate the wrapper with CUDA in notebooks, e.g.: GANerAidModel(device="cuda")