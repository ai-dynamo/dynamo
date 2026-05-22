#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Idempotent teardown for the multi-DGD live test. Safe to re-run.
#
# Covers both reproduction paths from README.md:
#   Path A (production shape): real DGDs + Dynamo operator
#   Path B (stub-workers):     raw pods from 50-stub-workers.yaml
# Pre-2026-05-21 only handled Path A; the Path B teardown lines were
# added after the live run on dpp-dev-env AKS.

set -u   # NOT set -e — we want best-effort delete even if some objects are
         # already gone or in Terminating state.

NS=${NS:-kaim-dynamo-system}
TEST_NODE=${TEST_NODE:-}
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== Tearing down multi-DGD live test in namespace $NS ==="

# 1. Bystander pod (Path A only). Cleared first so GPU 0 conflict
#    clears before we tear down the rest — Power Agent gets a clean
#    final reconcile.
kubectl delete pod -n "$NS" power-conflict-bystander --ignore-not-found

# 2. Path B — stub worker pods. Same label selector as the test
#    fixtures, so this works even if the user lost the YAML.
kubectl delete pod -n "$NS" -l purpose=power-aware-multi-dgd-test \
    --ignore-not-found --wait=false

# 3. Path A — the 3 DGDs. The operator cascades the pod / DCD deletes.
#    No-op if you only ran Path B.
for d in vllm-dgd-a vllm-dgd-b vllm-dgd-c; do
    kubectl delete dynamographdeployment -n "$NS" "$d" \
        --ignore-not-found --wait=false 2>/dev/null || true
done

# 4. Path A — profiling ConfigMaps (operator does NOT cascade these —
#    they're user-managed).
for cm in dgd-a-planner-profile-data dgd-b-planner-profile-data dgd-c-planner-profile-data; do
    kubectl delete configmap -n "$NS" "$cm" --ignore-not-found
done

# 5. Wait for pods to actually go away before we tear down the agent
#    (otherwise SIGTERM-driven cap-restore races with our own delete).
echo "Waiting for worker pods to clear (up to 120s)..."
kubectl wait --for=delete pod -n "$NS" \
    -l purpose=power-aware-multi-dgd-test --timeout=120s || true

# 6. Power Agent helm release. SIGTERM handler restores default TGP on
#    all managed GPUs before exit (verified by test_shutdown.py).
helm uninstall -n "$NS" power-agent || true

# 7. Standalone DCGM hostengine DS + Service + ServiceAccount.
kubectl delete -f "$HERE/20-nvidia-dcgm-standalone-ds.yaml" \
    --ignore-not-found || true

# 8. Test-node label. ONLY clear if we know the node — otherwise leave it.
if [[ -n "$TEST_NODE" ]]; then
    kubectl label node "$TEST_NODE" power-test/node- --overwrite || true
fi

# 9. We deliberately DO NOT uninstall dynamo-platform — other tests in
#    this namespace may still depend on it. If you want a full nuke:
#        helm uninstall -n "$NS" dynamo-platform
#    But that's a destructive operation and should be a conscious choice.

# 10. Final cap restoration + verification. We spawn ONE pod that
#     unconditionally restores every GPU's cap to its NVML default —
#     this works around Finding #10 (DCGM-mode SIGTERM does NOT restore
#     caps; the test caps persist after helm uninstall). The pod needs
#     the Power Agent's exact securityContext (privileged + runAsUser:0)
#     because vllm-runtime's default non-root user gets
#     NVMLError_NoPermission on nvmlDeviceSetPowerManagementLimit even
#     under `privileged: true`. Idempotent: if caps are already at
#     default the set is a no-op.
#     Image: any image with pynvml available — we reuse the NVML-mode
#     Power Agent image since it's already in the cluster cache from
#     phase 2.
PA_IMAGE=${PA_IMAGE:-ttl.sh/dynamo-pa-kaim-4c3c606:24h}
if [[ -n "$TEST_NODE" ]]; then
    echo "=== Restoring + verifying caps on $TEST_NODE ==="
    cat <<EOF | kubectl apply -n "$NS" -f - 2>&1 | head -1
apiVersion: v1
kind: Pod
metadata:
  name: power-restore-final
  labels:
    purpose: power-aware-multi-dgd-test-teardown
spec:
  restartPolicy: Never
  hostPID: true
  nodeSelector:
    kubernetes.io/hostname: $TEST_NODE
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  runtimeClassName: nvidia
  containers:
    - name: r
      image: $PA_IMAGE
      imagePullPolicy: IfNotPresent
      env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"
      securityContext:
        privileged: true
        runAsUser: 0
      command: ["python", "-c"]
      args:
        - |
          import pynvml
          pynvml.nvmlInit()
          drift_before = []
          for i in range(pynvml.nvmlDeviceGetCount()):
              h = pynvml.nvmlDeviceGetHandleByIndex(i)
              cap = pynvml.nvmlDeviceGetPowerManagementLimit(h) // 1000
              dflt = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) // 1000
              if cap != dflt:
                  drift_before.append((i, cap, dflt))
              pynvml.nvmlDeviceSetPowerManagementLimit(h, dflt * 1000)
              after = pynvml.nvmlDeviceGetPowerManagementLimit(h) // 1000
              tag = "ok" if after == dflt else "FAILED"
              print(f"  gpu{i}: was={cap}W -> now={after}W (default {dflt}W) [{tag}]")
          if drift_before:
              print(f"NOTE: caps had drifted on {len(drift_before)}/8 GPUs before restore — Finding #10 in README")
          print("RESULT: all GPUs at default")
EOF
    kubectl wait pod -n "$NS" power-restore-final \
        --for=condition=Ready --timeout=120s || true
    sleep 3
    kubectl logs -n "$NS" power-restore-final || true
    kubectl delete pod -n "$NS" power-restore-final \
        --ignore-not-found --wait=false
fi

echo "=== Teardown complete ==="
