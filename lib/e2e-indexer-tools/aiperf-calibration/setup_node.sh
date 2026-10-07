#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Node-local setup for the AIPerf calibration track. Run inside the hold via srun.
set -euo pipefail
: "${SLURM_JOB_ID:?run inside the hold}"
R=${CALIB_ROOT:?set CALIB_ROOT to the shared calibration directory}
L=${CALIB_LOCAL:-/tmp/aiperf-calib-$SLURM_JOB_ID}
# Optional: a script that puts uv on PATH (and sets UV_CACHE_DIR).
[[ -n "${TOOLCHAIN_ENV:-}" ]] && source "$TOOLCHAIN_ENV"
mkdir -p "$R/logs"
mkdir -p "$L"
cd "$L"
if [[ ! -x venv/bin/aiperf ]]; then
  uv venv -q --python /usr/bin/python3.12 venv
  uv pip install -q --python venv/bin/python aiperf==0.13.0
fi
venv/bin/aiperf --version
venv/bin/python -c "import uvloop, orjson; print('uvloop', uvloop.__version__)"
export HF_HOME=$L/hf
venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-0.6B")
print("tokenizer snapshot", p)
PY
venv/bin/pip freeze 2>/dev/null > "$R/logs/venv-freeze.txt" || uv pip freeze --python venv/bin/python > "$R/logs/venv-freeze.txt"
echo setup-ok "$L"
