#!/bin/bash
# ===========================================================================
# on-start.sh — Runs EVERY TIME the SageMaker notebook instance starts
#
# This script re-registers the Jupyter kernel (which lives on the ephemeral
# root volume) to point at the conda env on the persistent EBS volume.
# It also ensures git submodules are initialized.
#
# INSTALLATION (two options):
#
#   Option A — SageMaker Lifecycle Configuration (recommended):
#     1. Go to SageMaker Console > Notebook instances > Your instance
#     2. Edit > Lifecycle configuration > Create new
#     3. Paste the contents of this file into the "Start notebook" script
#     4. Save and restart the instance
#
#   Option B — Run manually each session:
#     bash ~/SageMaker/tableGenCompare/on-start.sh
#     (Still faster than reinstalling everything!)
# ===========================================================================
set -e

ENV_NAME="tablegen"
CONDA_DIR="/home/ec2-user/anaconda3"
ENV_DIR="/home/ec2-user/SageMaker/.envs/${ENV_NAME}"
REPO_DIR="/home/ec2-user/SageMaker/tableGenCompare"
KERNEL_DIR="/home/ec2-user/.local/share/jupyter/kernels/${ENV_NAME}"

# ---------- 1. Verify persistent env exists ----------
if [ ! -d "$ENV_DIR" ]; then
    echo "ERROR: Conda env not found at ${ENV_DIR}"
    echo "Run setup_env.sh first: bash ${REPO_DIR}/setup_env.sh"
    exit 1
fi

# ---------- 2. Re-register Jupyter kernel ----------
echo "Registering Jupyter kernel 'Python (tablegen)'..."
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

# ---------- 3. Ensure git submodules (GANerAid, CTAB-GAN, etc.) ----------
if [ -d "$REPO_DIR/.git" ] || [ -f "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git submodule update --init --recursive 2>/dev/null || true
fi

# ---------- 4. Ensure claude CLI is on PATH ----------
if ! grep -q 'local/bin' /home/ec2-user/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/ec2-user/.bashrc
fi

echo "✔ on-start complete. Kernel 'Python (tablegen)' is ready."
