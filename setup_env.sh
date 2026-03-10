#!/bin/bash
# ===========================================================================
# setup_env.sh — ONE-TIME environment setup on persistent EBS volume
#
# Run this ONCE after creating a new SageMaker notebook instance.
# The conda env is stored on /home/ec2-user/SageMaker (EBS) so it
# survives instance stop/start cycles.
#
# After running this, on-start.sh handles everything on subsequent boots.
# ===========================================================================
set -e

ENV_NAME="tablegen"
CONDA_DIR="/home/ec2-user/anaconda3"
# ---- KEY CHANGE: env lives on the persistent EBS volume ----
ENV_DIR="/home/ec2-user/SageMaker/.envs/${ENV_NAME}"
REPO_DIR="/home/ec2-user/SageMaker/tableGenCompare"

# ---------- 1. Create conda env on persistent volume ----------
if [ -d "$ENV_DIR" ]; then
    echo "Conda environment already exists at ${ENV_DIR}, skipping creation."
else
    echo "Creating conda environment at ${ENV_DIR} (persistent EBS volume)..."
    conda create --prefix "$ENV_DIR" python=3.10 -y
fi

# ---------- 2. Install dependencies ----------
echo "Installing dependencies from requirements.txt..."
# GANerAid's dep tab-gan-metrics pins dython==0.5.1, conflicting with dython>=0.7.12
# Install everything else first, then GANerAid + tab-gan-metrics with --no-deps
grep -v "GANerAid" "${REPO_DIR}/requirements.txt" > /tmp/_req_no_ganeraid.txt
conda run --prefix "$ENV_DIR" pip install --no-cache-dir -r /tmp/_req_no_ganeraid.txt
rm -f /tmp/_req_no_ganeraid.txt
conda run --prefix "$ENV_DIR" pip install --no-cache-dir --no-deps GANerAid==1.8
conda run --prefix "$ENV_DIR" pip install --no-cache-dir --no-deps tab-gan-metrics

# ---------- 3. Patch GANerAid library bugs ----------
echo "Patching GANerAid and tab-gan-metrics..."
GANERAID_DIR="${ENV_DIR}/lib/python3.10/site-packages/GANerAid"
TAB_DIR="${ENV_DIR}/lib/python3.10/site-packages/tab_gan_metrics"

# 3a. Patch tab-gan-metrics: dython >= 0.7.12 renamed compute_associations
python3 << 'PYEOF'
import pathlib, re
tab_dir = pathlib.Path("${TAB_DIR}")
for fname in ["helpers.py", "tab_gan_metrics.py"]:
    p = tab_dir / fname
    if not p.exists():
        continue
    txt = p.read_text()
    if "try:" in txt and "_compute_associations" in txt:
        continue  # already patched
    txt = txt.replace(
        "from dython.nominal import compute_associations,",
        "try:\n    from dython.nominal import compute_associations,\nexcept ImportError:\n    from dython.nominal import _compute_associations as compute_associations,\nif False:\n    from dython.nominal import"  # no-op line so trailing imports parse
    )
    p.write_text(txt)
    print(f"  Patched {fname}")
PYEOF

# 3b. Patch GANerAid device bug: library uses torch.cuda.is_available()
#     instead of respecting the device parameter, causing CPU/CUDA mismatch.
#     Copy fixed versions from repo.
cp "${REPO_DIR}/patches/ganeraid_model.py"  "${GANERAID_DIR}/model.py"       2>/dev/null || true
cp "${REPO_DIR}/patches/ganeraid_trainer.py" "${GANERAID_DIR}/gan_trainer.py" 2>/dev/null || true
cp "${REPO_DIR}/patches/ganeraid_utils.py"   "${GANERAID_DIR}/utils.py"       2>/dev/null || true

# Verify everything imports
conda run --prefix "$ENV_DIR" python -c "from GANerAid.ganeraid import GANerAid; print('[OK] GANerAid imports successfully')"

# ---------- 4. Initialize git submodules ----------
echo "Initializing git submodules (CTAB-GAN, CTAB-GAN-Plus, GANerAid)..."
cd "$REPO_DIR"
git submodule update --init --recursive

# ---------- 5. Register Jupyter kernel ----------
# (on-start.sh also does this, but do it here so it works immediately)
KERNEL_DIR="/home/ec2-user/.local/share/jupyter/kernels/${ENV_NAME}"
mkdir -p "$KERNEL_DIR"
cat > "${KERNEL_DIR}/kernel.json" <<KEOF
{
  "argv": [
    "bash",
    "-c",
    "source \"${CONDA_DIR}/bin/activate\" \"${ENV_DIR}\" && exec \"${ENV_DIR}/bin/python\" -m ipykernel_launcher -f '{connection_file}'"
  ],
  "display_name": "Python (tablegen)",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
KEOF

# ---------- 6. Add PATH for claude CLI ----------
if ! grep -q 'local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo "  Env location : ${ENV_DIR}  (persistent EBS)"
echo "  Kernel       : Python (tablegen)"
echo ""
echo "  Refresh the Jupyter page and select 'Python (tablegen)' kernel."
echo ""
echo "  NEXT STEP: Install the on-start lifecycle script so the"
echo "  kernel auto-registers on every boot. See on-start.sh."
echo "============================================================"
