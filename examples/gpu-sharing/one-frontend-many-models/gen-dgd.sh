#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate ONE DynamoGraphDeployment with one frontend and NUM_MODELS aggregated vLLM
# workers. Every worker serves the same weights under its own model name
# (qwen3-4b-01..NN), so the single frontend lists NUM_MODELS models and routes each
# request to the worker that serves the model it names.
#
# This is the one-frontend counterpart of ../kai-gpu-fractions/gen-dgds.sh, which
# renders one DGD and one frontend per model. The workers are the same kind of pod:
# a GPU fraction placed by KAI-Scheduler and capped by kai-gpu-fractioning (CUDA GPU
# fractions over MPS), with NO nvidia.com/gpu resource.
#
# Settings that matter:
#   - No DYN_NAMESPACE on the workers. The operator sets the frontend's
#     DYN_NAMESPACE_PREFIX to <k8s-namespace>-<dgd-name> and registers the workers
#     of the DGD in the Dynamo namespace <k8s-namespace>-<dgd-name>-<8 hex>, the
#     DGD's worker generation, which the frontend discovers. Observed with operator
#     and runtime 1.5.0 and 1.6.2.
#   - --served-model-name gives each worker its own model name,
#     <MODEL_NAME_PREFIX>-NN, which is also the name of its DGD component.
#   - The share of the GPU is set per container: gpu-compute.request (fraction of
#     the SMs) and gpu-memory.request/limit (MiB). With an explicit memory limit
#     vLLM sees the limit as the GPU's total memory, so --gpu-memory-utilization is
#     a fraction of the limit, not of the physical card.
#   - runtimeClassName: nvidia and the non-root securityContext are the same as in
#     ../kai-gpu-fractions/gen-dgds.sh, which explains why they are needed.
#
# Usage: ./gen-dgd.sh [model-snapshot-path] > dgd-2-models.yaml
#
# Environment:
#   NUM_MODELS              number of workers, one model each        (default: 2,
#                           which fills one A100 40GB with the other defaults)
#   DGD_NAME                name of the DGD                          (default: qwen3-4b-shared)
#   MODEL_NAME_PREFIX       served names are <prefix>-01..NN         (default: qwen3-4b)
#   GPU_COMPUTE             fraction of a GPU's SMs per worker       (default: 0.5)
#   GPU_MEMORY_MI           GPU memory cap per worker, MiB           (default: 19968)
#   GPU_MEMORY_UTILIZATION  vLLM --gpu-memory-utilization            (default: 0.80)
#   WORKER_CPU_REQUEST      CPU request per worker                   (default: 2)
#   WORKER_MEMORY_REQUEST   host memory request per worker           (default: 12Gi)
#   RUNTIME_IMAGE           image for the frontend and the workers   (default: vllm-runtime:1.6.2)
#   HF_CACHE_DIR            hostPath model cache on the node         (default: /opt/hf-cache)
#   NAMESPACE               namespace for the DGD                    (default: default)
#   NODE_NAME               pin the frontend and every worker to this node
#                           (kubernetes.io/hostname), the node that holds the hostPath
#                           cache; unset = no nodeSelector
set -euo pipefail

NUM_MODELS="${NUM_MODELS:-2}"
DGD_NAME="${DGD_NAME:-qwen3-4b-shared}"
MODEL_NAME_PREFIX="${MODEL_NAME_PREFIX:-qwen3-4b}"
GPU_COMPUTE="${GPU_COMPUTE:-0.5}"
GPU_MEMORY_MI="${GPU_MEMORY_MI:-19968}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
WORKER_CPU_REQUEST="${WORKER_CPU_REQUEST:-2}"
WORKER_MEMORY_REQUEST="${WORKER_MEMORY_REQUEST:-12Gi}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.2}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/opt/hf-cache}"
NAMESPACE="${NAMESPACE:-default}"
NODE_NAME="${NODE_NAME:-}"

# With NODE_NAME set, add a nodeSelector to each component's pod spec (the lines
# that are exactly "        spec:"); otherwise pass the manifest through unchanged.
add_node_selector() {
  if [ -z "${NODE_NAME}" ]; then
    cat
  else
    awk -v node="${NODE_NAME}" '{ print } $0 == "        spec:" {
      print "          nodeSelector:"
      print "            kubernetes.io/hostname: " node
    }'
  fi
}
MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_PATH="${1:-${HF_CACHE_DIR}/hub/models--Qwen--Qwen3-4B/snapshots/${MODEL_REVISION}}"

cat <<'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Rendered by gen-dgd.sh. To change this manifest, edit gen-dgd.sh and re-render.
EOF

cat <<EOF | add_node_selector
---
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: $DGD_NAME
  namespace: $NAMESPACE
  annotations:
    experiment.nvidia.com/runtime-image: "$RUNTIME_IMAGE"
spec:
  backendFramework: vllm
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      podTemplate:
        metadata:
          labels:
            experiment.nvidia.com/dgd: $DGD_NAME
            experiment.nvidia.com/role: frontend
        spec:
          containers:
            - name: main
              image: $RUNTIME_IMAGE
              imagePullPolicy: IfNotPresent
              command:
                - python3
                - -m
                - dynamo.frontend
              args:
                - --http-port
                - "8000"
EOF

for i in $(seq -f '%02g' 1 "${NUM_MODELS}"); do
NAME="$MODEL_NAME_PREFIX-$i"
cat <<EOF | add_node_selector
    - name: $NAME
      type: worker
      replicas: 1
      podTemplate:
        metadata:
          labels:
            experiment.nvidia.com/dgd: $DGD_NAME
            experiment.nvidia.com/role: worker
            experiment.nvidia.com/model: $NAME
            kai.scheduler/queue: default-queue
          annotations:
            nvidia.com/container.main.gpu-compute.request: "$GPU_COMPUTE"
            nvidia.com/container.main.gpu-memory.request: ${GPU_MEMORY_MI}Mi
            nvidia.com/container.main.gpu-memory.limit: ${GPU_MEMORY_MI}Mi
            nvidia.com/container.main.gpu-compute.mode: sm-sharing
        spec:
          schedulerName: kai-scheduler
          runtimeClassName: nvidia
          terminationGracePeriodSeconds: 180
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            runAsGroup: 1000
            seccompProfile:
              type: RuntimeDefault
          containers:
            - name: main
              image: $RUNTIME_IMAGE
              imagePullPolicy: IfNotPresent
              command:
                - python3
                - -m
                - dynamo.vllm
              args:
                - --model
                - $MODEL_PATH
                - --served-model-name
                - $NAME
                - --gpu-memory-utilization
                - "$GPU_MEMORY_UTILIZATION"
                - --max-model-len
                - "4096"
                - --max-num-seqs
                - "32"
              env:
                - name: HF_HOME
                  value: $HF_CACHE_DIR
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: TRANSFORMERS_OFFLINE
                  value: "1"
              resources:
                requests:
                  cpu: "$WORKER_CPU_REQUEST"
                  memory: $WORKER_MEMORY_REQUEST
              securityContext:
                allowPrivilegeEscalation: false
                capabilities:
                  drop:
                    - ALL
              volumeMounts:
                - name: model-cache
                  mountPath: $HF_CACHE_DIR
                  readOnly: true
                - name: shm
                  mountPath: /dev/shm
          volumes:
            - name: model-cache
              hostPath:
                path: $HF_CACHE_DIR
                type: Directory
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: 4Gi
EOF
done
