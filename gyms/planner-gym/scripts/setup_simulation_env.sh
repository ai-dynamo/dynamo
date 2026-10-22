#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [DYNAMO_DIR [PYTHON]]

Build Dynamo's native replay runtime and install Planner Gym's simulation
requirements into gyms/planner-gym/.venv. Requires Linux and the Dynamo
source-build prerequisites (see the repository contribution guide).

DYNAMO_DIR defaults to this script's containing Dynamo checkout.
PYTHON defaults to python3.12; Python 3.11 and 3.12 are supported.
An existing .venv is preserved; move it aside before rerunning.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

gym_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
dynamo_dir="${1:-$gym_dir/../..}"
if [[ ! -d "$dynamo_dir" ]]; then
  echo "Dynamo source directory does not exist: $dynamo_dir" >&2
  exit 2
fi
dynamo_dir="$(cd "$dynamo_dir" && pwd -P)"
python_bin="${2:-python3.12}"
venv_dir="$gym_dir/.venv"

if [[ ! -f "$dynamo_dir/lib/bindings/python/Cargo.toml" ]]; then
  echo "Not a Dynamo source checkout: $dynamo_dir" >&2
  exit 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Offline simulation requires Linux: Dynamo's pinned AISimulate package supplies Linux wheels only." >&2
  echo "Trace generation, Match Config validation, and saved reports work without this runtime." >&2
  exit 2
fi

command -v "$python_bin" >/dev/null || {
  echo "Python executable not found: $python_bin" >&2
  exit 2
}
command -v rustup >/dev/null || {
  echo "rustup is required to select Dynamo's pinned Rust toolchain" >&2
  exit 2
}
command -v cargo >/dev/null || {
  echo "cargo is required to build Dynamo's native runtime" >&2
  exit 2
}
if [[ -e "$venv_dir" ]]; then
  echo "Refusing to reuse an existing environment: $venv_dir" >&2
  echo "Move it aside and rerun this script to get a clean setup." >&2
  exit 2
fi

"$python_bin" - <<'PY'
import sys

if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit(
        "Planner Gym offline simulation requires Python 3.11 or 3.12; "
        f"found {sys.version.split()[0]}"
    )
PY

"$python_bin" -m venv "$venv_dir"
# shellcheck disable=SC1091
source "$venv_dir/bin/activate"
python -m pip install --upgrade pip uv 'maturin[patchelf]'

pushd "$dynamo_dir" >/dev/null
maturin develop --release \
  -m lib/bindings/python/Cargo.toml \
  --features ais-forward-pass
popd >/dev/null

uv pip install -e "$dynamo_dir" -e "$gym_dir[sim]"

python - <<'PY'
import inspect

import autoscaling_arena.runners.sims
from dynamo import _core
from dynamo.replay import TelemetryOptions, run_trace_replay

low_level = inspect.signature(_core.run_mocker_trace_replay).parameters
public = inspect.signature(run_trace_replay).parameters

required_low_level = {
    "scaling_policy",
    "capture_telemetry",
    "telemetry_sample_interval_ms",
    "telemetry_callback",
    "telemetry_jsonl_path",
}
missing_low_level = required_low_level - low_level.keys()
if missing_low_level:
    raise SystemExit(
        "Dynamo's low-level replay binding is missing: "
        + ", ".join(sorted(missing_low_level))
    )

required_public = {"planner_config", "telemetry_options"}
missing_public = required_public - public.keys()
if missing_public:
    raise SystemExit(
        "Dynamo's public replay API is missing: "
        + ", ".join(sorted(missing_public))
    )

options = TelemetryOptions(
    sample_interval_ms=5_000,
    jsonl_path="telemetry.jsonl",
)
missing_options = {
    "sample_interval_ms",
    "capture_in_memory",
    "callback",
    "jsonl_path",
} - {name for name in dir(options) if not name.startswith("_")}
if missing_options:
    raise SystemExit(
        "Dynamo's TelemetryOptions is missing: "
        + ", ".join(sorted(missing_options))
    )
if options.jsonl_path != "telemetry.jsonl":
    raise SystemExit("Dynamo's TelemetryOptions did not retain jsonl_path")
PY

echo "Planner Gym simulation environment ready"
printf 'Activate it with: source %q\n' "$venv_dir/bin/activate"
