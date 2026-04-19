#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Layer B — launch a vllm+KVBM server using the same KvbmServerManager the
# integration tests use. Spec attributes (model id, attention backend,
# block size, batch_invariant) are sourced from the test module's
# parametrize list — this script never hardcodes per-model attributes.
# Prints KVBM_EXTERNAL_BASE_URL / KVBM_EXTERNAL_METRICS_PORT /
# KVBM_SPEC_ID exports for shell 3 (run_eval.sh). Ctrl-C to stop.
#
# Usage: run_server.sh <spec-id>
#
# spec-id matches KvbmServerSpec.id in the test module, e.g.:
#   v1-DeepSeek-R1-Distill-Llama-8B
#   v1-DeepSeek-V2-Lite          (gated; needs KVBM_ENABLE_MLA=1)
#   v1-Qwen3-0.6B                (only if KVBM_MODEL_ID=Qwen/Qwen3-0.6B)
#   v2-Qwen3-0.6B-intra          (v2, intra onboard mode)
#   v2-Qwen3-0.6B-inter          (v2, inter onboard mode)
#
# v1 spec ids: require NATS_SERVER + ETCD_ENDPOINTS in env (from run_deps_v1.sh).
# v2 spec ids: do NOT need NATS/etcd — leave NATS_SERVER/ETCD_ENDPOINTS unset.
#
# Env overrides honored:
#   KVBM_CPU_BLOCKS              (override spec.cpu_blocks)
#   KVBM_GPU_BLOCKS              (override spec.gpu_blocks)
#   KVBM_SERVER_START_TIMEOUT    (server bring-up timeout, default 600s)
#   KVBM_ENABLE_MLA              (required to launch any MLA spec)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <spec-id>" >&2
    echo "       (e.g. v1-DeepSeek-R1-Distill-Llama-8B, v2-Qwen3-0.6B-intra)" >&2
    exit 2
fi

SPEC_ID="$1"

if [[ "$SPEC_ID" != v1-* && "$SPEC_ID" != v2-* ]]; then
    echo "[server] unknown spec id prefix: $SPEC_ID (expected v1- or v2-)" >&2
    exit 2
fi

if [[ "$SPEC_ID" == v1-* ]]; then
    if [[ -z "${NATS_SERVER:-}" || -z "${ETCD_ENDPOINTS:-}" ]]; then
        echo "[server] v1 requires NATS_SERVER and ETCD_ENDPOINTS in env (run scripts/run_deps_v1.sh first)" >&2
        exit 4
    fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

START_TIMEOUT="${KVBM_SERVER_START_TIMEOUT:-600}"

export KVBM_SPEC_ID="$SPEC_ID"
export KVBM_SERVER_START_TIMEOUT="$START_TIMEOUT"

exec python - <<'PY'
import dataclasses
import os
import signal
import sys
from pathlib import Path

from tests.kvbm_integration.fixtures import KvbmServerManager
from tests.kvbm_integration.test_determinism_agg import _CACHE_RESET_SPECS

spec_id = os.environ["KVBM_SPEC_ID"]
specs_by_id = {s.id: s for s in _CACHE_RESET_SPECS}
if spec_id not in specs_by_id:
    sys.stderr.write(
        f"[server] unknown spec id: {spec_id!r}\n"
        f"[server] known: {sorted(specs_by_id)}\n"
    )
    sys.exit(6)

spec = specs_by_id[spec_id]

if spec.model_config.use_mla and os.environ.get("KVBM_ENABLE_MLA", "").lower() not in (
    "1",
    "true",
    "yes",
    "on",
):
    sys.stderr.write(
        f"[server] {spec_id} is an MLA spec and is gated; set KVBM_ENABLE_MLA=1 "
        f"to launch (see ACTIVE_PLAN.md phase 3)\n"
    )
    sys.exit(7)

cpu_override = os.environ.get("KVBM_CPU_BLOCKS")
gpu_override = os.environ.get("KVBM_GPU_BLOCKS")
overrides = {}
if cpu_override is not None:
    overrides["cpu_blocks"] = int(cpu_override)
if gpu_override is not None:
    overrides["gpu_blocks"] = int(gpu_override)
if overrides:
    spec = dataclasses.replace(spec, **overrides)

start_timeout = int(os.environ["KVBM_SERVER_START_TIMEOUT"])

mgr = KvbmServerManager(spec=spec, log_dir=Path("/tmp/kvbm-run-server-logs"))


def _stop(*_):
    print("[server] stopping ...", flush=True)
    try:
        mgr.stop_server()
    finally:
        sys.exit(0)


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

print(
    f"[server] starting spec={spec.id} "
    f"(cpu_blocks={spec.cpu_blocks}, gpu_blocks={spec.gpu_blocks}, "
    f"timeout={start_timeout}s) ...",
    flush=True,
)
ok = mgr.start_server(timeout=start_timeout)
if not ok:
    print("[server] failed to start", flush=True)
    sys.exit(5)

print("", flush=True)
print("=" * 64, flush=True)
print("[server] READY. Export these in shell 3 (run_eval.sh):", flush=True)
print(f"  export KVBM_EXTERNAL_BASE_URL={mgr.base_url}", flush=True)
print(f"  export KVBM_EXTERNAL_METRICS_PORT={mgr.metrics_port}", flush=True)
print(f"  export KVBM_SPEC_ID={spec.id}", flush=True)
print("=" * 64, flush=True)
print("[server] Ctrl-C to stop.", flush=True)
signal.pause()
PY
