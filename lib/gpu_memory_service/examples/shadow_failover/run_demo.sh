#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot orchestrator for the standalone (no Kubernetes operator) GMS
# shadow-engine failover demo on a single node, single GPU.
#
# It runs the whole flow end to end:
#   1. start etcd + nats              (start_infra.sh)
#   2. start the GMS server           (start_gms.sh)
#   3. start the Dynamo frontend
#   4. launch the PRIMARY  (RW writer, loads weights once -> publishes to GMS)
#   5. launch the SHADOW   (RO importer, imports resident weights, no reload)
#   6. pause the shadow    (POST /engine/control/sleep {"level":2})
#   7. send a completion to the frontend (proves the primary serves)
#   8. FAILOVER: SIGKILL the primary process group   (kill_primary.sh)
#   9. wake the shadow     (POST /engine/control/wake_up {})
#  10. verify takeover     (verify.sh)
#
# Always cleans up on exit (trap): frontend, both engine groups, GMS, infra.
#
# You can also run each step by hand using the individual scripts in this dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${GMS_SOCKET_DIR}" "${RUN_DIR}"

# --- cleanup: stray processes hold GPU memory; always reap everything --------
cleanup() {
  local code=$?
  echo "==> CLEANUP: tearing down all demo processes"

  # Engines (process groups led by setsid).
  for role in shadow primary; do
    local f="${RUN_DIR}/${role}.pgid"
    if [[ -f "${f}" ]]; then
      local pgid
      pgid="$(cat "${f}")"
      kill -KILL -"${pgid}" 2>/dev/null || true
    fi
  done

  # Frontend (its own process group).
  if [[ -f "${RUN_DIR}/frontend.pgid" ]]; then
    kill -KILL -"$(cat "${RUN_DIR}/frontend.pgid")" 2>/dev/null || true
  fi

  # GMS supervisor (process group => reaps every per-(device,tag) child).
  if [[ -f "${RUN_DIR}/gms.pgid" ]]; then
    kill -TERM -"$(cat "${RUN_DIR}/gms.pgid")" 2>/dev/null || true
    sleep 2
    kill -KILL -"$(cat "${RUN_DIR}/gms.pgid")" 2>/dev/null || true
  fi

  # Infra.
  for svc in nats etcd; do
    if [[ -f "${RUN_DIR}/${svc}.pid" ]]; then
      kill -TERM "$(cat "${RUN_DIR}/${svc}.pid")" 2>/dev/null || true
    fi
  done

  echo "==> CLEANUP done (exit ${code})"
}
trap cleanup EXIT INT TERM

echo "==> STEP 1: start infra (etcd + nats-server -js)"
bash "${SCRIPT_DIR}/start_infra.sh"

echo "==> STEP 2: start GMS server supervisor"
bash "${SCRIPT_DIR}/start_gms.sh"

echo "==> STEP 3: start Dynamo frontend on port ${FRONTEND_PORT}"
setsid env \
  ETCD_ENDPOINTS="${ETCD_ENDPOINTS}" \
  NATS_SERVER="${NATS_SERVER}" \
  DYN_LOG="${DYN_LOG}" \
  python -m dynamo.frontend --http-port "${FRONTEND_PORT}" \
  > "${RUN_DIR}/frontend.log" 2>&1 &
frontend_pid=$!
echo "${frontend_pid}" > "${RUN_DIR}/frontend.pid"
echo "${frontend_pid}" > "${RUN_DIR}/frontend.pgid"
wait_for_ready "${FRONTEND_PORT}" 120

echo "==> STEP 4: launch PRIMARY (RW writer; loads weights once -> GMS)"
bash "${SCRIPT_DIR}/run_engine.sh" primary

echo "==> STEP 5: launch SHADOW (RO importer; imports resident weights, no reload)"
bash "${SCRIPT_DIR}/run_engine.sh" shadow

echo "==> STEP 6: pause the SHADOW (POST /engine/control/sleep {\"level\":2})"
curl -sf -X POST "http://localhost:${SHADOW_SYSTEM_PORT}/engine/control/sleep" \
  -H 'Content-Type: application/json' \
  -d '{"level": 2}' \
  && echo "    shadow paused"

echo "==> STEP 7: send a completion to the frontend (primary should serve)"
RETRIES=10 RETRY_INTERVAL=2 bash "${SCRIPT_DIR}/verify.sh" \
  && echo "    primary serving confirmed"

echo "==> STEP 8: FAILOVER — SIGKILL the primary process group"
bash "${SCRIPT_DIR}/kill_primary.sh"

echo "==> STEP 9: wake the SHADOW (POST /engine/control/wake_up {})"
curl -sf -X POST "http://localhost:${SHADOW_SYSTEM_PORT}/engine/control/wake_up" \
  -H 'Content-Type: application/json' \
  -d '{}' \
  && echo "    shadow wake_up requested"

echo "==> STEP 10: verify takeover (frontend keeps serving via the shadow)"
bash "${SCRIPT_DIR}/verify.sh"

echo "==> DEMO COMPLETE: shadow took over without reloading model weights"
