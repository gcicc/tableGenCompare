# Setting Up an AWS SageMaker Notebook Instance for CTAB-GAN/Deep Tabular GANs

These steps walk you through accessing AWS, setting up a GPU-powered notebook, installing requirements, and cloning key repositories.

---

## 1. Accessing AWS and SageMaker

1. Log in to AWS via Okta: **AWS TEC**
2. Select your environment: `tec-rnd-sqs-dev`
3. Choose **Amazon SageMaker AI** from services menu
4. In the SageMaker Console, click **Notebook Instances**

---

## 2. Setup: Attach a Git Repository

5. Choose **Git repositories**, add:  https://github.com/gcicc/tableGenCompare.git

## 3. Create the Notebook Instance

6. **Create Notebook Instance** with:
 - **Instance type:** `ml.g4dn.xlarge` (GPU, suitable for deep learning)
 - **Volume:** 50 GB (customize as needed)
 - **Attach your Git repo**

---

## 4. Launching and Configuring Your Environment

7. Once instance is "InService," click **Open JupyterLab**

---

# Project Setup Instructions

## First-Time Setup

If you're setting up the environment and Jupyter kernel for the first time, follow these steps:

1. **Open a Terminal in JupyterLab**: Start by opening a terminal session in JupyterLab. By default, this will use the base environment unless another is activated.

2. **Switch to Bash**: Ensure you are using the bash shell:
   ```bash
   bash
   ```

3. **Create the Conda Environment**: Create the `clinical_synth` environment with Python 3.11. This step only needs to be done once.
   ```bash
   conda create -n clinical_synth python=3.11
   ```

4. **Initialize Conda for Your Shell**: Run this command to ensure Conda commands work in your shell. This step only needs to be done once per shell type.
   ```bash
   conda init bash
   ```
   Restart your terminal or source your shell configuration file to apply the changes:
   ```bash
   source ~/.bashrc
   ```

5. **Activate the Conda Environment**: Activate your new environment each time you start working on your project.
   ```bash
   conda activate clinical_synth
   ```

6. **Install Project Dependencies**: Install necessary packages from `requirements.txt`. This step ensures that all necessary packages are installed in your environment.
   ```bash
   pip install -r requirements.txt
   ```

7. **Fix Version Conflicts**: Especially for SageMaker compatibility, ensure specific package versions are installed.
   ```bash
   pip install numpy==1.26.4
   ```

8. **Clone Additional Repositories**: In the same terminal, clone the necessary repositories and install additional packages.
   ```bash
   git clone https://github.com/Team-TUD/CTAB-GAN.git
   git clone https://github.com/Team-TUD/CTAB-GAN-Plus.git
   git clone https://github.com/TeamGenerAid/GANerAid
   pip install ctgan
   pip install GANerAid
   pip install sdv
   ```

9. **Set Up Jupyter Kernel**: Make this environment available as a Jupyter kernel.
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name clinical_synth --display-name "Clinical Synthetic (py3.11)"
   ```

## Returning Users

If you've already set up the environment and kernel, follow these steps each time you return to work on your project:

1. **Open a Terminal in JupyterLab**: Start by opening a terminal session in JupyterLab.

2. **Activate the Conda Environment**: Activate your `clinical_synth` environment.
   ```bash
   conda activate clinical_synth
   ```

3. **Start Working on Your Project**: With the environment activated, you can now proceed with your project tasks.

## Additional Tips

- **Installing Python Packages**: If you need to install additional Python packages, do so from a notebook cell using `!pip install ...` to ensure they are installed in the correct environment:
  ```python
  !pip install <package-name>
  ```

- **Kernel Management**: When running a notebook, ensure you change the kernel to the appropriate environment (`clinical_synth`) if needed.

By following these instructions, you'll ensure that your environment is consistently set up for working on your project. Adjust the instructions as needed based on your specific workflow or project requirements.