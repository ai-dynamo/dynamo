#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download Qwen/Qwen3-4B at the pinned revision into a hostPath model cache on the
# GPU node, using a one-shot Kubernetes Job that runs the same vllm-runtime image the
# workers use. Workers then mount the cache read-only and run fully offline
# (HF_HUB_OFFLINE=1).
#
# The snapshot lands under:
#   ${HF_CACHE_DIR}/hub/models--Qwen--Qwen3-4B/snapshots/${MODEL_REVISION}
# which is the default model path baked into every gen-dgds.sh in this directory.
#
# Two modes:
#   - NODE_NAME unset: run this on the (single) GPU node. It creates the hostPath
#     directory locally with sudo.
#   - NODE_NAME set: run this from any machine with cluster access. The Job is pinned
#     to that node, and a root init container creates the hostPath directory there,
#     so no shell or sudo on the node is needed.
#
# Environment:
#   HF_CACHE_DIR    host directory for the model cache       (default: /opt/hf-cache)
#   NODE_NAME       GPU node to place the cache on (kubernetes.io/hostname label);
#                   unset = local single-node mode            (default: unset)
#   NAMESPACE       namespace to run the download Job in     (default: default)
#   KUBECTL         kubectl command, e.g. "microk8s kubectl" (default: kubectl)
#   RUNTIME_IMAGE   image used for the download Job          (default: vllm-runtime:1.3.0)
#   HF_TOKEN        optional; Qwen/Qwen3-4B is public, so a token is not required
set -euo pipefail

HF_CACHE_DIR="${HF_CACHE_DIR:-/opt/hf-cache}"
NAMESPACE="${NAMESPACE:-default}"
read -r -a KUBECTL_CMD <<< "${KUBECTL:-kubectl}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0}"
MODEL_ID="Qwen/Qwen3-4B"
MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
JOB_NAME="download-qwen3-4b"
NODE_NAME="${NODE_NAME:-}"

# The vllm-runtime container runs as a non-root user. A root-owned cache directory
# makes the Job fail three times with PermissionError and report only
# BackoffLimitExceeded, so open the directory up first: locally with sudo, or, when
# NODE_NAME is set, with a root init container on that node.
if [ -z "${NODE_NAME}" ]; then
  sudo mkdir -p "${HF_CACHE_DIR}"
  sudo chmod 777 "${HF_CACHE_DIR}"
  NODE_SELECTOR=""
  INIT_CONTAINERS=""
  HOSTPATH_TYPE="Directory"
else
  NODE_SELECTOR="      nodeSelector:
        kubernetes.io/hostname: ${NODE_NAME}"
  INIT_CONTAINERS="      initContainers:
        - name: prepare-cache
          image: ${RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [\"chmod\", \"777\", \"${HF_CACHE_DIR}\"]
          securityContext:
            runAsUser: 0
          volumeMounts:
            - name: model-cache
              mountPath: ${HF_CACHE_DIR}"
  HOSTPATH_TYPE="DirectoryOrCreate"
fi

"${KUBECTL_CMD[@]}" -n "${NAMESPACE}" delete job "${JOB_NAME}" --ignore-not-found

# Pass HF_TOKEN through a Secret so the token is not stored in the Job or Pod
# spec. Without HF_TOKEN, remove any Secret left by an earlier run so a stale
# token is not used.
if [ -n "${HF_TOKEN:-}" ]; then
  "${KUBECTL_CMD[@]}" -n "${NAMESPACE}" create secret generic "${JOB_NAME}-hf-token" \
    --from-literal=token="${HF_TOKEN}" --dry-run=client -o yaml \
    | "${KUBECTL_CMD[@]}" apply -f -
else
  "${KUBECTL_CMD[@]}" -n "${NAMESPACE}" delete secret "${JOB_NAME}-hf-token" --ignore-not-found
fi

"${KUBECTL_CMD[@]}" -n "${NAMESPACE}" apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
${NODE_SELECTOR}
${INIT_CONTAINERS}
      containers:
        - name: download
          image: ${RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          command:
            - python3
            - -c
            - |
              from huggingface_hub import snapshot_download
              path = snapshot_download("${MODEL_ID}", revision="${MODEL_REVISION}")
              print(path)
          env:
            - name: HF_HOME
              value: ${HF_CACHE_DIR}
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: ${JOB_NAME}-hf-token
                  key: token
                  optional: true
          volumeMounts:
            - name: model-cache
              mountPath: ${HF_CACHE_DIR}
      volumes:
        - name: model-cache
          hostPath:
            path: ${HF_CACHE_DIR}
            type: ${HOSTPATH_TYPE}
EOF

# Poll for either terminal condition so a failed Job is reported as soon as it
# fails, instead of after the full timeout.
TIMEOUT_SECONDS=1800
elapsed=0
while :; do
  complete=$("${KUBECTL_CMD[@]}" -n "${NAMESPACE}" get "job/${JOB_NAME}" \
    -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')
  failed=$("${KUBECTL_CMD[@]}" -n "${NAMESPACE}" get "job/${JOB_NAME}" \
    -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}')
  if [ "${complete}" = "True" ]; then
    break
  fi
  if [ "${failed}" = "True" ] || [ "${elapsed}" -ge "${TIMEOUT_SECONDS}" ]; then
    if [ "${failed}" = "True" ]; then
      echo "ERROR: Job ${JOB_NAME} failed. Logs:" >&2
    else
      echo "ERROR: Job ${JOB_NAME} did not finish within ${TIMEOUT_SECONDS}s. Logs:" >&2
    fi
    "${KUBECTL_CMD[@]}" -n "${NAMESPACE}" logs "job/${JOB_NAME}" --all-containers --tail=50 >&2 || true
    "${KUBECTL_CMD[@]}" -n "${NAMESPACE}" describe "job/${JOB_NAME}" >&2 || true
    exit 1
  fi
  sleep 10
  elapsed=$(( elapsed + 10 ))
done
"${KUBECTL_CMD[@]}" -n "${NAMESPACE}" logs "job/${JOB_NAME}" | tail -1

SNAPSHOT="${HF_CACHE_DIR}/hub/models--Qwen--Qwen3-4B/snapshots/${MODEL_REVISION}"
# In single-node mode the cache is on this host, so list it.
[ -n "${NODE_NAME}" ] || ls -l "${SNAPSHOT}"
echo "Model snapshot: ${SNAPSHOT}${NODE_NAME:+ (on node ${NODE_NAME})}"
