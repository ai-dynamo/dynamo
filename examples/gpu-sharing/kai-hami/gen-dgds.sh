#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate 16 DynamoGraphDeployments (qwen3-4b-01..16), packed 2-per-GPU on
# 8x A100 40GB via KAI-Scheduler + the HAMi resource isolator.
#
# Each DGD = 1 frontend + 1 aggregated vLLM worker (prefill and decode on the same
# worker). The worker requests gpu-fraction "0.5" from KAI and has NO nvidia.com/gpu
# resource: KAI's admission webhook rejects pods that carry both. On this stack KAI
# translates the 0.5 fraction into a ~20 GB HAMi memory cap
# (CUDA_DEVICE_MEMORY_LIMIT=20070m, GPU_PORTION=0.49). HAMi enforces memory only;
# compute is time-sliced by the driver. The gpu-compute.mode: sm-sharing annotation
# is inert on this stack (it belongs to the GPU fractions stack) and is kept only so
# the manifest matches the one that produced the published results.
#
# Settings that matter:
#   - --gpu-memory-utilization 0.85: under HAMi the pod sees only its ~20 GB cap, not
#     the physical 40 GB. 0.85 x ~20 GB ~= 17 GB.
#   - --max-num-seqs 32: high-concurrency points measure GPU contention rather than
#     vLLM queueing.
#   - DYN_DISCOVERY_BACKEND=etcd on both components and an empty
#     DYN_NAMESPACE_WORKER_SUFFIX on the worker: required with operator 1.4.2 +
#     vllm-runtime 1.3.0 (see ../common/install-dynamo-platform.sh).
#
# Usage: ./gen-dgds.sh [model-snapshot-path] > dgds-16x.yaml
#
# Environment:
#   HF_CACHE_DIR  hostPath model cache on the node  (default: /opt/hf-cache)
#   NAMESPACE     namespace for the DGDs             (default: default)
#   NUM_MODELS    number of DGDs                     (default: 16)
#   NODE_NAME     pin every frontend and worker to this node (kubernetes.io/hostname),
#                 the node that holds the hostPath cache; unset = no nodeSelector
set -euo pipefail

HF_CACHE_DIR="${HF_CACHE_DIR:-/opt/hf-cache}"
NAMESPACE="${NAMESPACE:-default}"
NUM_MODELS="${NUM_MODELS:-16}"
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
# Rendered by gen-dgds.sh. To change these manifests, edit gen-dgds.sh and re-render.
EOF

for i in $(seq -f '%02g' 1 "${NUM_MODELS}"); do
NAME="qwen3-4b-$i"
cat <<EOF | add_node_selector
---
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: $NAME
  namespace: $NAMESPACE
  annotations:
    experiment.nvidia.com/model-revision: "$MODEL_REVISION"
    experiment.nvidia.com/runtime-image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0"
spec:
  backendFramework: vllm
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      podTemplate:
        metadata:
          labels:
            experiment.nvidia.com/dgd: $NAME
            experiment.nvidia.com/role: frontend
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
              imagePullPolicy: IfNotPresent
              command:
                - python3
                - -m
                - dynamo.frontend
              args:
                - --http-port
                - "8000"
              env:
                - name: DYN_NAMESPACE
                  value: $NAMESPACE-$NAME
                - name: DYN_DISCOVERY_BACKEND
                  value: etcd
                - name: HF_HOME
                  value: $HF_CACHE_DIR
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: TRANSFORMERS_OFFLINE
                  value: "1"
              volumeMounts:
                - name: model-cache
                  mountPath: $HF_CACHE_DIR
                  readOnly: true
          volumes:
            - name: model-cache
              hostPath:
                path: $HF_CACHE_DIR
                type: Directory
    - name: worker
      type: worker
      replicas: 1
      podTemplate:
        metadata:
          labels:
            experiment.nvidia.com/dgd: $NAME
            experiment.nvidia.com/role: worker
            kai.scheduler/queue: default-queue
          annotations:
            gpu-fraction: "0.5"
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
              image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
              imagePullPolicy: IfNotPresent
              command:
                - python3
                - -m
                - dynamo.vllm
              args:
                - --model
                - $MODEL_PATH
                - --served-model-name
                - Qwen/Qwen3-4B
                - --gpu-memory-utilization
                - "0.85"
                - --max-model-len
                - "4096"
                - --max-num-seqs
                - "32"
              env:
                - name: DYN_NAMESPACE
                  value: $NAMESPACE-$NAME
                - name: DYN_DISCOVERY_BACKEND
                  value: etcd
                - name: DYN_NAMESPACE_WORKER_SUFFIX
                  value: ""
                - name: HF_HOME
                  value: $HF_CACHE_DIR
                - name: HF_HUB_OFFLINE
                  value: "1"
                - name: TRANSFORMERS_OFFLINE
                  value: "1"
              resources:
                requests:
                  cpu: "2"
                  memory: 12Gi
                limits:
                  cpu: "8"
                  memory: 32Gi
              securityContext:
                allowPrivilegeEscalation: false
                capabilities:
                  drop:
                    - ALL
              volumeMounts:
                - name: model-cache
                  mountPath: $HF_CACHE_DIR
                  readOnly: true
          volumes:
            - name: model-cache
              hostPath:
                path: $HF_CACHE_DIR
                type: Directory
EOF
done
