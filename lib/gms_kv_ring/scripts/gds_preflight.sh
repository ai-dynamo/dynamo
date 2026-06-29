#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# gds_preflight.sh — validate GMS NIXL GDS-direct path on a host
# before exercising the restore-on-hit production tests.
#
# Runs 4 gates in order; stops at the first failure with a clear
# diagnosis. Only if all gates pass does it run the env-gated
# real-GDS test that exercises promote_storage_to_hbm with
# `dest_offset != offset` — the path GMSKVCacheConnectorV1's
# restore-on-hit rides on.
#
# REQUIRED ENV:
#   GMS_KVR_NIXL_GDS_PATH  Writable directory on a cuFile-capable
#                          filesystem (NVMe-w/-cuFile, BeeGFS,
#                          Lustre, etc.). Will create + delete a
#                          handful of small files there during the
#                          test run.
#
# OPTIONAL ENV:
#   GMS_GDS_TEST_TIMEOUT_S Per-test timeout for pytest (default: 120).
#
# OUTPUT: structured PASS/FAIL per gate, final summary. Exit 0
# only if every gate passed AND the env-gated test passed.

set -eu

# ---------- helpers ----------

red()    { printf '\033[31m%s\033[0m' "$*"; }
green()  { printf '\033[32m%s\033[0m' "$*"; }
yellow() { printf '\033[33m%s\033[0m' "$*"; }
bold()   { printf '\033[1m%s\033[0m'  "$*"; }

gate() {
    local n="$1" ; shift
    local title="$1" ; shift
    printf '\n%s %s — %s\n' "$(bold "[gate $n]")" "$(bold "$title")" "$*"
}
pass() { printf '  %s %s\n' "$(green PASS)" "$*"; }
fail() { printf '  %s %s\n' "$(red FAIL)" "$*"; exit 1; }
info() { printf '  %s %s\n' "$(yellow info)" "$*"; }

# ---------- gate 0: required env vars ----------

gate 0 "environment" "resolve Python, repository root, and GDS path"

GMS_DYNAMO_ROOT="${GMS_DYNAMO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
GMS_PYTHON="${GMS_PYTHON:-${PYTHON:-python}}"
GMS_PYTHON="$(command -v "$GMS_PYTHON")" || fail "Python executable not found"
: "${GMS_KVR_NIXL_GDS_PATH:?Set GMS_KVR_NIXL_GDS_PATH to a cuFile-capable mount}"

[[ -d "$GMS_DYNAMO_ROOT/lib/gms_kv_ring" ]] || fail "no $GMS_DYNAMO_ROOT/lib/gms_kv_ring"
[[ -d "$GMS_KVR_NIXL_GDS_PATH" ]] || fail "$GMS_KVR_NIXL_GDS_PATH is not a directory"
[[ -w "$GMS_KVR_NIXL_GDS_PATH" ]] || fail "$GMS_KVR_NIXL_GDS_PATH is not writable by $(whoami)"

pass "python:      $GMS_PYTHON"
pass "dynamo root: $GMS_DYNAMO_ROOT"
pass "GDS path:    $GMS_KVR_NIXL_GDS_PATH (writable)"

# Use repository sources while preserving caller-provided loader paths.
# We do not re-export PATH-style libraries here because that is
# host-specific. Supply any extra loader paths via LD_LIBRARY_PATH
# before running this script.
export PYTHONPATH="$GMS_DYNAMO_ROOT/lib"
export GMS_KVR_NIXL_GDS_PATH

# ---------- gate 1: libcufile is loadable ----------

gate 1 "libcufile" "cuFile driver library must be loadable"

# Try unversioned soname first (system install), then versioned
# soname (.so.0 — what NVIDIA CUDA wheels ship). If only the
# versioned form resolves, print the LD_LIBRARY_PATH extension the
# user should set so subsequent gates find it.
"$GMS_PYTHON" - <<'PY' || fail "libcufile is not loadable. Add the directory containing libcufile.so to LD_LIBRARY_PATH and re-run."
import ctypes, sys

attempts = [
    "libcufile.so",      # unversioned (system install or dev pkg)
    "libcufile.so.0",    # soname'd, what CUDA wheels ship
]
last_err = None
loaded_via = None
for name in attempts:
    try:
        ctypes.CDLL(name)
        loaded_via = name
        break
    except OSError as e:
        last_err = e

if loaded_via is None:
    print("    dlopen failed for", attempts, ":", last_err, file=sys.stderr)
    print("    set LD_LIBRARY_PATH to the directory containing libcufile.so and re-run", file=sys.stderr)
    sys.exit(2)

print(f"    loaded {loaded_via} via dlopen")
PY
pass "libcufile resolves via dynamic loader"

# ---------- gate 2: NIXL has the GDS plugin ----------

gate 2 "NIXL plugins" "NIXL agent must enumerate the GDS plugin"

"$GMS_PYTHON" - <<'PY' || fail "NIXL is loadable but the 'GDS' plugin is NOT in available_plugins. Install nixl-cu12 with GDS support, or re-build NIXL with --enable-gds. The current host's NIXL is built without GDS."
import sys
try:
    from nixl._api import nixl_agent
except Exception as e:
    print("    cannot import nixl:", e, file=sys.stderr)
    sys.exit(2)
agent = nixl_agent("gms-preflight-probe")
plugins = list(agent.get_plugin_list())
print(f"    NIXL plugins available: {plugins}")
if "GDS" not in plugins and "GDS_MT" not in plugins:
    print("    GDS / GDS_MT plugin missing", file=sys.stderr)
    sys.exit(3)
PY
pass "NIXL GDS plugin available"

# ---------- gate 3: GMS NIXL backend constructs with the GDS plugin ----------

gate 3 "NixlBackend(plugin=GDS)" "construct NixlBackend on the candidate path"

"$GMS_PYTHON" - <<PY || fail "NixlBackend failed to construct with plugin=GDS on $GMS_KVR_NIXL_GDS_PATH. Likely causes: the mount is not cuFile-capable (try \`cufile_sample_001\` against it), filesystem is read-only at the path, or cuFile JSON config is missing."
import os, sys
from gms_kv_ring.daemon.backends_nixl import NixlBackend
base = os.environ["GMS_KVR_NIXL_GDS_PATH"]
try:
    nb = NixlBackend(base, plugin="GDS")
    print("    NixlBackend(plugin=GDS) constructed; name=", nb.name)
    nb.release_engine("__preflight__")
except Exception as e:
    print("    construction raised:", repr(e), file=sys.stderr)
    sys.exit(2)
PY
pass "NixlBackend constructs on the candidate path"

# ---------- gate 4: real-GDS round-trip + dest_offset test ----------

gate 4 "pytest" "exercise the GDS data plane end-to-end"

TIMEOUT="${GMS_GDS_TEST_TIMEOUT_S:-120}"

cd "$GMS_DYNAMO_ROOT/lib/gms_kv_ring"

info "running: pytest -v tests/test_backend_nixl.py::test_nixl_gds_round_trip_real_mount"
timeout "$TIMEOUT" "$GMS_PYTHON" -m pytest -v --tb=short \
    tests/test_backend_nixl.py::test_nixl_gds_round_trip_real_mount \
    || fail "single-block GDS round-trip failed — restore-on-hit cannot land on this host."
pass "single-block GDS demote+promote round-trip"

info "running: pytest -v tests/test_backend_nixl.py::test_nixl_gds_promote_to_hbm_with_separate_dest_offset"
timeout "$TIMEOUT" "$GMS_PYTHON" -m pytest -v --tb=short \
    tests/test_backend_nixl.py::test_nixl_gds_promote_to_hbm_with_separate_dest_offset \
    || fail "promote_storage_to_hbm with dest_offset != offset failed — restore-on-hit's cross-block-id remap is broken on this host."
pass "promote_storage_to_hbm dest_offset remap"

# ---------- summary ----------

printf '\n%s %s\n' "$(green ALL GATES PASSED)" "— GMS restore-on-hit validation succeeded."
