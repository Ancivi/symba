#!/bin/bash
# Sets up the isolated environment for notebooks/brain_neuropixel_preprocessing.ipynb.
# See requirements-brain-preprocessing.txt for why this can't be a plain
# `pip install -r requirements.txt` — the pynwb/hdmf downgrade below is a
# deliberate override of allensdk's/ndx-events' own version constraints,
# which pip's resolver won't accept if asked to solve everything at once.
#
# Requires Python 3.11: allensdk's C-extension dependencies (e.g. `tables`)
# don't ship prebuilt wheels for Python 3.9/3.10 on macOS arm64 and fail to
# compile from source. Any Python 3.11 works (python.org, conda, pyenv, Homebrew);
# pass it explicitly with:  PYTHON=/path/to/python3.11 ./setup_brain_preprocessing_env.sh
#
# Windows / conda alternative (same steps, in an Anaconda prompt):
#   conda create -n brain_preprocessing_env python=3.11
#   conda activate brain_preprocessing_env
#   pip install allensdk==2.16.2 seaborn==0.13.2 ipykernel==7.3.0
#   pip install pynwb==2.3.3
#   pip install hdmf==3.5.4
#   pip install "scikit-learn<1.4"
#   pip install numpy==1.23.5 scipy==1.10.1
#   pip install h5py==3.16.0
set -euo pipefail

ENV_DIR="brain_preprocessing_env"

# Python 3.11 interpreter: $PYTHON if set, else python3.11 on the PATH, else Homebrew's
PYTHON311="${PYTHON:-}"
if [ -z "$PYTHON311" ]; then
    if command -v python3.11 >/dev/null 2>&1; then
        PYTHON311="$(command -v python3.11)"
    elif command -v brew >/dev/null 2>&1; then
        PYTHON311="$(brew --prefix python@3.11 2>/dev/null)/bin/python3.11"
    fi
fi

if [ -z "$PYTHON311" ] || ! "$PYTHON311" -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>/dev/null; then
    echo "Python 3.11 not found. Install it (python.org, conda, pyenv, or: brew install python@3.11)"
    echo "and run:  PYTHON=/path/to/python3.11 ./setup_brain_preprocessing_env.sh"
    exit 1
fi

"$PYTHON311" -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install --upgrade pip -q

# Step 1: install allensdk and the other direct dependencies normally —
# pip will resolve modern (too-new-for-old-NWB-files) pynwb/hdmf here.
"$ENV_DIR/bin/pip" install \
    allensdk==2.16.2 \
    seaborn==0.13.2 \
    ipykernel==7.3.0

# Step 2 & 3: deliberately downgrade pynwb/hdmf to read the Allen Institute's
# older-schema .nwb files correctly (see requirements-brain-preprocessing.txt
# for the full explanation). pip will warn about conflicts with ndx-events'
# and allensdk's own stated constraints — that's expected and does not
# actually break anything the notebook uses.
"$ENV_DIR/bin/pip" install "pynwb==2.3.3"
"$ENV_DIR/bin/pip" install "hdmf==3.5.4"

# Step 4: scikit-learn (needed transitively because `import symba` eagerly
# loads every submodule, including one that imports sklearn) pulls in a
# newer numpy than allensdk allows — re-pin both back down afterward.
"$ENV_DIR/bin/pip" install "scikit-learn<1.4"
"$ENV_DIR/bin/pip" install "numpy==1.23.5" "scipy==1.10.1"

# Step 5: pin h5py to the tested version.
"$ENV_DIR/bin/pip" install "h5py==3.16.0"

echo ""
echo "Done. Register the Jupyter kernel with:"
echo "  $ENV_DIR/bin/python3 -m ipykernel install --user --name brain_preprocessing_env --display-name \"Python 3.11 (brain preprocessing)\""
