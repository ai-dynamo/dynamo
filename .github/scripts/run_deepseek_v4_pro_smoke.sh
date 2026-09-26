#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
readonly REPO_ROOT
readonly NAMESPACE="${NAMESPACE:-dynamo-nightly-dsv4}"
readonly ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/artifacts/deepseek-v4-pro-smoke}"
readonly DEPLOY_TIMEOUT="${DEPLOY_TIMEOUT:-100m}"
readonly SCHEDULING_TIMEOUT="${SCHEDULING_TIMEOUT:-10m}"
readonly PERF_TIMEOUT="${PERF_TIMEOUT:-30m}"
readonly PVC_TIMEOUT_SECONDS="${PVC_TIMEOUT_SECONDS:-300}"
readonly KEEP_RESOURCES="${KEEP_RESOURCES:-0}"

KUBECTL=(kubectl)
if [[ -n "${KUBE_CONTEXT:-}" ]]; then
    KUBECTL+=(--context "${KUBE_CONTEXT}")
fi

secret_manifest=""
deploy_manifest=""
smoke_manifest=""
mkdir -p "${ARTIFACT_DIR}"

collect_diagnostics() {
    "${KUBECTL[@]}" -n "${NAMESPACE}" get \
        pods,services,jobs,dynamographdeployments,computedomains \
        -o wide >"${ARTIFACT_DIR}/resources.txt" 2>&1 || true
    "${KUBECTL[@]}" -n "${NAMESPACE}" get events \
        --sort-by=.lastTimestamp >"${ARTIFACT_DIR}/events.txt" 2>&1 || true

    while IFS= read -r pod; do
        [[ -n "${pod}" ]] || continue
        "${KUBECTL[@]}" -n "${NAMESPACE}" describe pod "${pod}" \
            >"${ARTIFACT_DIR}/${pod}-describe.txt" 2>&1 || true
        "${KUBECTL[@]}" -n "${NAMESPACE}" logs "${pod}" --all-containers \
            --tail=500 >"${ARTIFACT_DIR}/${pod}.log" 2>&1 || true
    done < <("${KUBECTL[@]}" -n "${NAMESPACE}" get pods -o name 2>/dev/null | cut -d/ -f2)
}

cleanup() {
    local status="${1:-$?}"
    trap - EXIT INT TERM

    if (( status != 0 )); then
        collect_diagnostics
    fi

    if [[ "${KEEP_RESOURCES}" != "1" ]]; then
        [[ -z "${smoke_manifest}" ]] || \
            "${KUBECTL[@]}" -n "${NAMESPACE}" delete -f "${smoke_manifest}" \
                --ignore-not-found --wait=false --timeout=2m || true
        [[ -z "${deploy_manifest}" ]] || \
            "${KUBECTL[@]}" -n "${NAMESPACE}" delete -f "${deploy_manifest}" \
                --ignore-not-found --wait=false --timeout=2m || true
        "${KUBECTL[@]}" -n "${NAMESPACE}" delete secret hf-token-secret \
            --ignore-not-found --wait=false --timeout=2m || true
    else
        echo "Keeping benchmark resources in ${NAMESPACE} for inspection"
    fi

    [[ -z "${secret_manifest}" ]] || rm -f "${secret_manifest}"
    [[ -z "${deploy_manifest}" ]] || rm -f "${deploy_manifest}"
    [[ -z "${smoke_manifest}" ]] || rm -f "${smoke_manifest}"

    exit "${status}"
}
trap 'cleanup $?' EXIT
trap 'exit 130' INT TERM

for command in git kubectl; do
    command -v "${command}" >/dev/null || {
        echo "Required command not found: ${command}" >&2
        exit 2
    }
done

cd "${REPO_ROOT}"
secret_manifest="$(mktemp)"
deploy_manifest="$(mktemp)"
smoke_manifest="$(mktemp)"
chmod 600 "${secret_manifest}" "${deploy_manifest}" "${smoke_manifest}"

"${KUBECTL[@]}" kustomize .github/ci/deepseek-v4-pro/deploy \
    --load-restrictor=LoadRestrictionsNone >"${deploy_manifest}"
"${KUBECTL[@]}" kustomize .github/ci/deepseek-v4-pro/smoke \
    --load-restrictor=LoadRestrictionsNone >"${smoke_manifest}"

# The workflow owns these names and its concurrency group serializes runs.
"${KUBECTL[@]}" -n "${NAMESPACE}" delete -f "${smoke_manifest}" \
    --ignore-not-found --wait=true --timeout=5m
"${KUBECTL[@]}" -n "${NAMESPACE}" delete -f "${deploy_manifest}" \
    --ignore-not-found --wait=true --timeout=5m
"${KUBECTL[@]}" -n "${NAMESPACE}" delete secret hf-token-secret \
    --ignore-not-found --wait=true --timeout=2m

"${KUBECTL[@]}" -n "${NAMESPACE}" create secret generic hf-token-secret \
    --from-literal=HF_TOKEN="${HF_TOKEN:-cache-only-smoke}" \
    --dry-run=client -o yaml >"${secret_manifest}"
"${KUBECTL[@]}" apply -f "${secret_manifest}"

deadline=$((SECONDS + PVC_TIMEOUT_SECONDS))
until [[ "$("${KUBECTL[@]}" -n "${NAMESPACE}" get pvc shared-model-cache \
    -o jsonpath='{.status.phase}' 2>/dev/null || true)" == "Bound" ]]; do
    if (( SECONDS >= deadline )); then
        echo "shared-model-cache did not bind within ${PVC_TIMEOUT_SECONDS}s" >&2
        exit 1
    fi
    sleep 5
done

"${KUBECTL[@]}" -n "${NAMESPACE}" apply -f "${deploy_manifest}"

for _ in {1..60}; do
    worker_count="$("${KUBECTL[@]}" -n "${NAMESPACE}" get pods \
        -l 'nvidia.com/dynamo-component-type in (prefill,decode)' \
        -o name 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${worker_count}" == "4" ]] && break
    sleep 5
done
if [[ "${worker_count}" != "4" ]]; then
    echo "Dynamo operator did not create all four worker pods" >&2
    exit 1
fi
"${KUBECTL[@]}" -n "${NAMESPACE}" wait \
    --for=condition=PodScheduled pod \
    -l 'nvidia.com/dynamo-component-type in (prefill,decode)' \
    --timeout="${SCHEDULING_TIMEOUT}"
"${KUBECTL[@]}" -n "${NAMESPACE}" wait \
    --for=condition=Ready dynamographdeployment/dsv4-pro-disagg \
    --timeout="${DEPLOY_TIMEOUT}"

"${KUBECTL[@]}" -n "${NAMESPACE}" apply -f "${smoke_manifest}"
"${KUBECTL[@]}" -n "${NAMESPACE}" wait \
    --for=condition=Complete job/dsv4-pro-disagg-bench \
    --timeout="${PERF_TIMEOUT}"

"${KUBECTL[@]}" -n "${NAMESPACE}" logs job/dsv4-pro-disagg-bench \
    >"${ARTIFACT_DIR}/aiperf.log"
pod="$("${KUBECTL[@]}" -n "${NAMESPACE}" get pods \
    -l job-name=dsv4-pro-disagg-bench -o jsonpath='{.items[0].metadata.name}')"
"${KUBECTL[@]}" -n "${NAMESPACE}" cp \
    "${pod}:/tmp/aiperf/." "${ARTIFACT_DIR}/" || true

collect_diagnostics
echo "DeepSeek-V4-Pro AIPerf smoke test completed in ${NAMESPACE}"
