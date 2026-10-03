#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the AIPerf 0.13.0 environment for live payloads on the GPU cluster's shared filesystem,
# pinned to the workstation environment that produced the loadgen parity evidence. Run on the
# workstation; it touches only the cluster's login node (no allocation).
#
#   aiperf_env.sh FREEZE_FILE [ENV_DIR]
#
# The interpreter is a uv-managed CPython 3.12.3 under LR_LIVE_ROOT/env/uv-python, so the
# environment runs unchanged inside the vLLM image (Ubuntu 22.04, glibc 2.35), which mounts the
# shared filesystem. The aiperf wheel is fetched and checked against its pinned SHA-256 before install; every
# other package is installed at its frozen version, and the resulting freeze must equal the
# workstation freeze. An existing environment is verified, never rebuilt in place.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/deploy/common.sh"
lr_need LR_SSH_ALIAS LR_LIVE_ROOT LR_TOOLCHAIN_ENV
[[ $# -ge 1 ]] || die "usage: aiperf_env.sh FREEZE_FILE [ENV_DIR]"
freeze=$1
env_dir="${2:-$LR_LIVE_ROOT/env/aiperf-0.13.0}"
ssh_alias="$LR_SSH_ALIAS"
wheel_url="https://files.pythonhosted.org/packages/37/13/1f4a02dfe54158ae99b8ae2c0d3d593d2ed4d37b04ea983ee764c2719561/aiperf-0.13.0-py3-none-any.whl"
wheel_sha="a20100fd127f1d10ea45d4946b1658a186030e86770344c80dfd256a60192f34"
grep -qx 'aiperf==0.13.0' "$freeze" || die "$freeze does not pin aiperf==0.13.0"
ssh -o BatchMode=yes "$ssh_alias" "mkdir -p '$LR_LIVE_ROOT/env'"
scp -q "$freeze" "$ssh_alias:$env_dir.workstation-freeze.txt"
ssh -o BatchMode=yes "$ssh_alias" bash -s -- "$LR_LIVE_ROOT" "$env_dir" "$wheel_url" "$wheel_sha" \
  "$LR_TOOLCHAIN_ENV" <<'REMOTE'
set -euo pipefail
root=$1 env_dir=$2 wheel_url=$3 wheel_sha=$4 toolchain=$5
source "$toolchain" > /dev/null
export UV_CACHE_DIR="$root/cache/uv" UV_PYTHON_INSTALL_DIR="$root/env/uv-python"
wheel="$root/env/aiperf-0.13.0-py3-none-any.whl"
if [[ ! -s "$wheel" ]]; then
  curl -fsSL -o "$wheel.partial" "$wheel_url"
  mv "$wheel.partial" "$wheel"
fi
printf '%s  %s\n' "$wheel_sha" "$wheel" | sha256sum -c - > /dev/null
if [[ ! -x "$env_dir/bin/aiperf" ]]; then
  [[ ! -e "$env_dir" ]] || { echo "error: $env_dir exists without aiperf; use a new path" >&2; exit 1; }
  uv python install 3.12.3
  grep -vx 'aiperf==0.13.0' "$env_dir.workstation-freeze.txt" > "$env_dir.requirements.txt"
  # Virtual environments record their own path, so build in place under the final name.
  uv venv --quiet --python 3.12.3 "$env_dir"
  uv pip install --quiet --python "$env_dir/bin/python" -r "$env_dir.requirements.txt" "$wheel"
fi
uv pip freeze --python "$env_dir/bin/python" | sed 's|^aiperf @ .*|aiperf==0.13.0|' > "$env_dir.freeze.txt"
diff "$env_dir.workstation-freeze.txt" "$env_dir.freeze.txt"
"$env_dir/bin/python" --version
"$env_dir/bin/aiperf" --version 2>&1 | tail -1
echo "env_dir=$env_dir"
REMOTE
