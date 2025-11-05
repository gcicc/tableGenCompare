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

8. **Edit `requirements.txt`** in your project:
 - Comment out (add `#` before) these lines:
   ```txt
   # CTAB-GAN Models (installed from GitHub)
   #git+https://github.com/Team-TUD/CTAB-GAN.git
   #git+https://github.com/Team-TUD/CTAB-GAN-Plus.git
   ```

---

## 5. Install Project Dependencies

9. **Open a Terminal in JupyterLab** (important: will use the base environment unless otherwise activated)

10. (Optional but recommended!) **Switch to PyTorch environment for GPU/deep learning:**
 ```bash
 source activate pytorch_p310
 ```

11. **Install requirements:**
 ```bash
 pip install -r requirements.txt
 ```

12. **Fix version conflicts (especially for SageMaker compatibility):**
 ```bash
 pip install numpy==1.26.4
 ```

---

## 6. Clone Additional Repositories

13. In the same terminal:
 ```bash
 git clone https://github.com/Team-TUD/CTAB-GAN.git
 git clone https://github.com/Team-TUD/CTAB-GAN-Plus.git
 git clone https://github.com/TeamGenerAid/GANerAid
 ```

---

## 7. (Optional) Install Any Additional Requirements from Cloned Repos

- Check if either repo has a `requirements.txt` and install if needed:
 ```bash
 pip install -r CTAB-GAN/requirements.txt
 pip install -r CTAB-GAN-Plus/requirements.txt
 ```

---

## 8. Starting Your Notebook & Kernel

14. Return to **JupyterLab** > Launch a new notebook.
15. When prompted, select the **`conda_pytorch310`** kernel.

**Tip:**  
If you install Python packages, do so from a notebook cell using `!pip install ...` to ensure they go to the right environment:
```python
!pip install -r requirements.txt
!pip install numpy==1.26.4








Trying to fix CTAB-GAN
pip install scikit-learn==0.24.2

# When running notebook ensure you change kernel to cconda_pytorch_p310

# Issues with CTAB-GAN, CTAB-GAN-Plus, GANerAide
