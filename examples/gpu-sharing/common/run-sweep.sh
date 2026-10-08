#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Concurrency sweep: all N models are driven simultaneously at each point, one AIPerf
# process per model, payload fixed (ISL 2048 / OSL 256, stddev 0, seed 42). Only
# concurrency varies.
#   request-count = max(40, 10*c)   floor keeps low-c percentiles stable
#   warmup-count  = min(16, 2*c)
#
# Each AIPerf process targets one DGD frontend Service directly (no port-forwards:
# 8-16 long-lived port-forwards are flaky under load). ENDPOINT_MODE picks how:
#   clusterip  resolve each Service's ClusterIP with kubectl; run on a host that can
#              route to ClusterIPs, such as the node of a single-node cluster
#   dns        use the in-cluster DNS name <svc>.<namespace>.svc.cluster.local; run
#              inside the cluster (see aiperf-client-pod.yaml); kubectl not needed
#
# Results: ${RESULTS_DIR}/<run-label>/c<N>/worker-NN/ (raw AIPerf exports) and
#          ${RESULTS_DIR}/<run-label>/c<N>/worker-NN.log
#
# Every frontend ClusterIP is resolved before a point starts, so a missing Service
# stops the sweep before any AIPerf process launches. Each AIPerf exit status is
# checked; failing runs are listed by log path, and the script exits non-zero
# after the last point if any run failed.
#
# Usage: ./run-sweep.sh <run-label>
#
# Environment:
#   NUM_MODELS     number of DGDs to drive: 8 (baseline) or 16 (kai-hami,
#                  kai-gpu-fractions)                         (default: 16)
#   NAME_PREFIX    DGD name prefix; DGDs are <prefix>-01..NN  (default: qwen3-4b)
#   NAMESPACE      namespace the DGDs run in                  (default: default)
#   RESULTS_DIR    output root                                (default: ./results)
#   CONCURRENCIES  space-separated per-model concurrencies    (default: "1 2 4 8 16 32")
#   AIPERF         aiperf binary            (default: ${AIPERF_VENV:-$HOME/.aiperf-venv}/bin/aiperf)
#   HF_HOME        tokenizer cache for AIPerf (default: /opt/hf-cache, the model cache)
#   KUBECTL        kubectl command, e.g. "microk8s kubectl"   (default: kubectl)
#   ENDPOINT_MODE  clusterip or dns                           (default: clusterip)
#   CLUSTER_DOMAIN cluster DNS domain for dns mode            (default: cluster.local)
set -euo pipefail

RUN_LABEL="${1:?Usage: ./run-sweep.sh <run-label>}"
NUM_MODELS="${NUM_MODELS:-16}"
NAME_PREFIX="${NAME_PREFIX:-qwen3-4b}"
NAMESPACE="${NAMESPACE:-default}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"
AIPERF="${AIPERF:-${AIPERF_VENV:-${HOME}/.aiperf-venv}/bin/aiperf}"
export HF_HOME="${HF_HOME:-/opt/hf-cache}"
read -r -a KUBECTL_CMD <<< "${KUBECTL:-kubectl}"
ENDPOINT_MODE="${ENDPOINT_MODE:-clusterip}"
CLUSTER_DOMAIN="${CLUSTER_DOMAIN:-cluster.local}"
case "${ENDPOINT_MODE}" in
  clusterip|dns) ;;
  *) echo "ERROR: ENDPOINT_MODE must be clusterip or dns, got ${ENDPOINT_MODE}" >&2; exit 1 ;;
esac

BASE="${RESULTS_DIR}/${RUN_LABEL}"
mkdir -p "${BASE}"

FAILED_TOTAL=0

for c in ${CONCURRENCIES}; do
  req=$(( 10 * c )); [ "$req" -lt 40 ] && req=40
  warm=$(( 2 * c )); [ "$warm" -gt 16 ] && warm=16
  OUT="${BASE}/c${c}"
  mkdir -p "${OUT}"
  echo "=== point c=${c} req=${req} warmup=${warm} ==="

  # Resolve every frontend endpoint before launching any AIPerf process, so a
  # missing Service aborts the sweep without leaving orphaned load generators.
  workers=()
  ips=()
  for worker in $(seq -f '%02g' 1 "${NUM_MODELS}"); do
    svc="${NAME_PREFIX}-${worker}-frontend"
    if [ "${ENDPOINT_MODE}" = "dns" ]; then
      ip="${svc}.${NAMESPACE}.svc.${CLUSTER_DOMAIN}"
      if ! getent hosts "${ip}" >/dev/null; then
        echo "ERROR: cannot resolve ${ip}; is Service ${svc} deployed, and is this running in the cluster?" >&2
        exit 1
      fi
    elif ! ip=$("${KUBECTL_CMD[@]}" get svc "${svc}" -n "${NAMESPACE}" \
          -o jsonpath='{.spec.clusterIP}') || [ -z "${ip}" ]; then
      echo "ERROR: cannot resolve ClusterIP of Service ${svc} in namespace ${NAMESPACE}" >&2
      exit 1
    fi
    workers+=("${worker}")
    ips+=("${ip}")
  done

  pids=()
  for idx in "${!workers[@]}"; do
    worker="${workers[$idx]}"
    "${AIPERF}" profile \
      -m Qwen/Qwen3-4B \
      --url "http://${ips[$idx]}:8000" \
      --endpoint-type chat \
      --streaming \
      --isl 2048 \
      --isl-stddev 0 \
      --output-tokens-mean 256 \
      --output-tokens-stddev 0 \
      --concurrency "${c}" \
      --request-count "${req}" \
      --warmup-request-count "${warm}" \
      --tokenizer Qwen/Qwen3-4B \
      --tokenizer-revision 1cfa9a7208912126459214e8b04321603b3df60c \
      --random-seed 42 \
      --export-level raw \
      --artifact-dir "${OUT}/worker-${worker}" \
      >"${OUT}/worker-${worker}.log" 2>&1 &
    pids+=("$!")
  done

  # Wait on every AIPerf process individually so each exit status is checked.
  failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      failed=$(( failed + 1 ))
      echo "FAILED: ${OUT}/worker-${workers[$idx]}.log" >&2
    fi
  done
  FAILED_TOTAL=$(( FAILED_TOTAL + failed ))
  echo "=== point c=${c} done (${failed} of ${#pids[@]} AIPerf runs failed) ==="
done

if [ "${FAILED_TOTAL}" -gt 0 ]; then
  echo "Sweep ${RUN_LABEL} finished with ${FAILED_TOTAL} failed AIPerf run(s); see the FAILED lines above." >&2
  exit 1
fi
echo "Completed sweep: ${RUN_LABEL}"
