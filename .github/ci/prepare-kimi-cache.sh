#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pin the nightly consumers without changing the shared Hub refs/main.
set -euo pipefail

manifest=$1
export KIMI_MODEL_PATH=/model-cache/hub/models--nvidia--Kimi-K2.5-NVFP4/snapshots/2fd3a800dedd098b8327eb49e93ebc75f85da19f
export KIMI_EAGLE_PATH=/model-cache/hub/models--nvidia--Kimi-K2.5-Thinking-Eagle3/snapshots/b71566cc726ec39db7b93dce0f6939b16febb362
export KIMI_TOKENIZER_REVISION=${KIMI_MODEL_PATH##*/}
export CACHE_PREFLIGHT_CODE
CACHE_PREFLIGHT_CODE=$(cat "$(dirname "$0")/validate-kimi-cache.py")

for worker in ${WORKERS//,/ }; do
  export WORKER="$worker"
  yq -e 'select(.kind == "DynamoGraphDeployment") |
    .spec.services[strenv(WORKER)].extraPodSpec.mainContainer.args[0] == "--model-path"' "$manifest" >/dev/null
  yq -i 'with(select(.kind == "DynamoGraphDeployment");
    .spec.services[strenv(WORKER)].extraPodSpec.mainContainer.args[1] = strenv(KIMI_MODEL_PATH) |
    (.spec.services[strenv(WORKER)].extraPodSpec.mainContainer.volumeMounts[] |
      select(.name == "model-cache").readOnly) = true)' "$manifest"
done
yq -i 'with(select(.kind == "ConfigMap" and .metadata.name == strenv(DGD) + "-config");
    .data[] |= (from_yaml | .speculative_config.speculative_model_dir = strenv(KIMI_EAGLE_PATH) | to_yaml)) |
  with(select(.kind == "Job" and .metadata.name == strenv(BENCHMARK));
    .spec.template.spec.containers[0].env += [
      {"name": "TOKENIZER_REVISION", "value": strenv(KIMI_TOKENIZER_REVISION)},
      {"name": "HF_HUB_OFFLINE", "value": "1"},
      {"name": "HF_MODULES_CACHE", "value": "/tmp/hf_modules"}] |
    (.spec.template.spec.containers[0].volumeMounts[] | select(.name == "model-cache").readOnly) = true)' "$manifest"

# The worker image checks the same read-only volume before any GPUs are allocated.
export WORKER=${WORKERS%%,*}
preflight=$(mktemp)
trap 'rm -f "$preflight"' EXIT
yq 'select(.kind == "DynamoGraphDeployment")' "$manifest" |
  yq '. as $dgd |
  .spec.services[strenv(WORKER)].extraPodSpec as $pod |
  {"apiVersion": "batch/v1", "kind": "Job",
    "metadata": {"name": strenv(DGD) + "-cache-preflight", "namespace": .metadata.namespace, "labels": .metadata.labels},
    "spec": {"backoffLimit": 0, "activeDeadlineSeconds": 600,
      "template": {"metadata": {"labels": .metadata.labels, "annotations": $dgd.spec.services[strenv(WORKER)].extraPodMetadata.annotations},
        "spec": {"restartPolicy": "Never", "automountServiceAccountToken": false,
          "imagePullSecrets": $pod.imagePullSecrets, "nodeSelector": $pod.nodeSelector,
          "tolerations": $pod.tolerations, "affinity": {"nodeAffinity": $pod.affinity.nodeAffinity},
          "containers": [{"name": "validate", "image": $pod.mainContainer.image,
            "command": ["python3", "-u", "-c", strenv(CACHE_PREFLIGHT_CODE), strenv(KIMI_MODEL_PATH), strenv(KIMI_EAGLE_PATH)],
            "env": [{"name": "HF_HOME", "value": "/model-cache"},
              {"name": "HF_HUB_OFFLINE", "value": "1"},
              {"name": "HF_MODULES_CACHE", "value": "/tmp/hf_modules"}],
            "resources": {"requests": {"cpu": "250m", "memory": "512Mi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
            "securityContext": {"allowPrivilegeEscalation": false, "capabilities": {"drop": ["ALL"]}, "seccompProfile": {"type": "RuntimeDefault"}},
            "volumeMounts": [{"name": "model-cache", "mountPath": "/model-cache", "readOnly": true}]}],
          "volumes": [$pod.volumes[] | select(.name == "model-cache")]}}}}' > "$preflight"
printf '\n---\n' >> "$manifest"
cat "$preflight" >> "$manifest"
