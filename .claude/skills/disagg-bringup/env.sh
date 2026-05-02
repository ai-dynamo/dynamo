#!/bin/bash
# Source-able environment for dynamo + KVBM builds run via the Bash tool.
# The Bash tool spawns non-interactive bash which does NOT source ~/.bashrc,
# so we duplicate the CUDA-toolkit + venv exports here.
#
# Usage: prefix any cargo / maturin command with `. .sandbox/env.sh && ...`
# from the workspace root, OR `source .sandbox/env.sh` once per session.

if [ -d /usr/local/cuda/bin ] && [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda

# Force any silent nvcc-missing fallback in build scripts to error loudly.
# Without this, kvbm-kernels silently builds stubs and cudarc silently
# picks "fallback-latest" (CUDA 13.1) on this CUDA-13.0 host.
export KVBM_REQUIRE_CUDA=1

# Needed by `maturin develop`; the bash tool's env doesn't source the venv
# activator, so set VIRTUAL_ENV explicitly.
export VIRTUAL_ENV=/home/ryan/.venvs/dynamo-kvbm
