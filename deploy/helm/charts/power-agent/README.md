<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Power Agent

The Power Agent applies per-GPU power caps on each selected node. Its existing
static behavior remains the default. Phase 2 transactional enforcement is an
explicit, qualified-hardware canary feature.

## Transactional canary rollout

Do not enable transactional mode until the exact GPU SKU, driver, Power Agent
image, and selected actuator mode have passed qualification. The default NVML
actuator does not require a DCGM hostengine. The optional DCGM actuator must
also qualify its DCGM client and hostengine combination. `dcgm-exporter`
remains the cluster monitoring path in either mode; it is not the Power
Agent's cap-write transport. Use an otherwise idle canary node and record the
immutable image digests and hardware/software versions with the result.

The approved Phase 2 qualification matrix is explicit; do not substitute the
weaker row for the stronger one:

| Environment | Exact GPU product | Architecture | Required actuator evidence |
| --- | --- | --- | --- |
| Azure AKS | `NVIDIA-A100-SXM4-80GB` | `amd64` | NVML qualification |
| GB200 cluster | `NVIDIA-GB200` | `arm64` | NVML/DCGM parity |

Adding another environment, product, or architecture requires its own
qualification record. In particular, an NVML-only GB200 run does not replace
the required GB200 parity result.

1. Label a small canary node pool and keep all non-canary nodes excluded by the
   chart's `nodeSelector`.
2. Confirm every GPU is idle and at its factory default power limit.
3. Run the actuator fixture during an approved hardware window. The
   Kubernetes harness requests the supplied GPU count on one exact-product
   node, refuses busy/non-default devices, qualifies
   minimum/default/maximum plus both clamp directions, exercises
   source-mounted gate/CUDA ordering, saves immutable evidence, and deletes
   only its uniquely named Job and ConfigMap. For the default NVML path, use
   `--actuator-mode nvml`; no DCGM image or hostengine is needed:

   ```bash
   python3 deploy/power-agent/tests/run_k8s_hardware_qualification.py \
     --kubectl kubectl \
     --context <approved-context> \
     --namespace <approved-namespace> \
     --gpu-product <exact-nvidia.com/gpu.product-value> \
     --gpu-count <all-gpus-on-one-canary-node> \
     --architecture <amd64-or-arm64> \
     --actuator-mode nvml \
     --evidence-dir <local-evidence-directory>
   ```

   For the optional DCGM actuator, use parity mode. Before creating the Job,
   this mode verifies the live version reported by every ready Pod behind the
   DCGM Service (the hostengine namespace, Service, and container default to
   the GPU Operator's standard names):

   ```bash
   python3 deploy/power-agent/tests/run_k8s_hardware_qualification.py \
     --kubectl kubectl \
     --context <approved-context> \
     --namespace <approved-namespace> \
     --gpu-product <exact-nvidia.com/gpu.product-value> \
     --gpu-count <all-gpus-on-one-canary-node> \
     --architecture <amd64-or-arm64> \
     --actuator-mode nvml-dcgm-parity \
     --dcgm-image <hostengine-compatible-dcgm-image> \
     --hostengine-host <node-local-hostengine-service> \
     --expected-hostengine-version <approved-live-hostengine-version> \
     --evidence-dir <local-evidence-directory>
   ```

   The lower-level source-checkout command remains available inside an
   already prepared qualification container:

   ```bash
   python deploy/power-agent/tests/e2e_actuator_parity.py \
     --hostengine-host nvidia-dcgm.gpu-operator.svc.cluster.local \
     --hostengine-port 5555 \
     --skip-busy-gpus \
     --require-default-before-write
   ```

   The released Power Agent image intentionally omits `tests/`. To qualify an
   immutable release digest, first create a ConfigMap from the checkout's
   fixture, then run a one-shot Pod on an otherwise idle node that is not being
   controlled by another Power Agent:

   ```bash
   kubectl -n <namespace> create configmap power-agent-parity-fixture \
     --from-file=e2e_actuator_parity.py=deploy/power-agent/tests/e2e_actuator_parity.py \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

   ```yaml
   apiVersion: v1
   kind: Pod
   metadata:
     name: power-agent-actuator-qualification
     namespace: <namespace>
   spec:
     nodeName: <idle-canary-node>
     restartPolicy: Never
     runtimeClassName: nvidia
     hostPID: true
     containers:
       - name: qualify
         image: <power-agent-image>@sha256:<digest>
         command: ["python3", "/qualification/e2e_actuator_parity.py"]
         args:
           - --hostengine-host
           - nvidia-dcgm.gpu-operator.svc.cluster.local
           - --hostengine-port
           - "5555"
           - --skip-busy-gpus
           - --require-default-before-write
         env:
           - name: PYTHONPATH
             value: /app:/opt/dcgm/python
           - name: NVIDIA_VISIBLE_DEVICES
             value: all
           - name: NVIDIA_DRIVER_CAPABILITIES
             value: compute,utility
         securityContext:
           privileged: true
         volumeMounts:
           - name: fixture
             mountPath: /qualification
             readOnly: true
     volumes:
       - name: fixture
         configMap:
           name: power-agent-parity-fixture
   ```

   This mounts only the qualification script; `PYTHONPATH` makes it import the
   actuator from `/app` while retaining the image's vendored DCGM bindings in
   `/opt/dcgm/python`. Save the Pod logs, then delete the Pod and ConfigMap
   after confirming every GPU is back at its entry cap.

   The fixture must report matching UUID-keyed NVML and DCGM results for every
   accepted write, restore every tested GPU to its entry/default cap, and pass
   the final parity check. A read-only discovery pass is available with
   `--read-only`, but it does not qualify cap writes.
4. Verify each transactional worker runtime image contains the
   `dynamo-power-gate` executable. A missing executable must prevent the
   original backend command from running; it is a release blocker, not a
   condition to bypass.
5. Enable transactional access only on the Power Agent canary release:

   ```yaml
   agent:
     transactional:
       enabled: true
   nodeSelector:
     dynamo.nvidia.com/power-phase-2-canary: "true"
   ```

   This adds Pod `patch` permission and a read-only kubelet PodResources socket
   mount. Namespace-restricted mode limits Pod access to the release namespace.
6. Enable the paired Planner capability in the platform release:

   ```yaml
   dynamo-operator:
     planner:
       powerAwareness:
         enabled: true
   ```

7. Enroll one DGD. Before widening the canary, require its
   `DynamoGraphPowerBudget` `status.phase` to reach `Idle`, rather than
   `Unqualified`, and independently read every accepted Agent report's GPU
   UUID and cap with `nvidia-smi`. If a scale request remains pending, inspect
   the DGDSA `status.pendingReason` for `UnqualifiedHardware`.

Treat an unknown SKU, any assigned live minimum/maximum pair different from the
catalog-qualified range, failed or stale readback, DCGM reconnect ambiguity,
missing gate, or backend start without a valid report as a failed
qualification. Do not mock or waive these outcomes.

## Qualification record

Keep one record for each exact hardware and software pool. Record the GPU
product label, GPU count, driver, actuator mode, Power Agent image digest,
minimum/default/maximum limits, requested test cap, and observed post-clamp
cap. For DCGM mode, also record the DCGM client and hostengine versions. A tag
without its resolved digest is not sufficient.

Complete one row for every runtime image advertised for transactional use.
Do not enable a backend whose row lacks immutable image evidence.

| Backend | Image digest | Gate on `PATH` | Missing-gate negative test | Report precedes backend/CUDA initialization | Checkpoint rejected | Result |
| --- | --- | --- | --- | --- | --- | --- |
| vLLM | Required | Required | Required | Required | Required | PASS required |
| SGLang | Required | Required | Required | Required | Required | PASS required |
| TensorRT-LLM | Required | Required | Required | Required | Required | PASS required |

For the missing-gate negative test, launch the rendered gate command against an
image known not to contain `dynamo-power-gate`; the original command must write
a marker if reached. Qualification passes only when the container fails and the
marker is absent. Use
`deploy/power-agent/tests/e2e_gate_entrypoint.py` only for source-mounted gate
qualification; it does not prove that a released runtime image contains the
installed console script.

Exercise every lifecycle row on the approved canary and attach timestamps,
Pod UIDs, GPU UUIDs, Agent reports, independent `nvidia-smi` reads, and relevant
events or logs. Restore all tested GPUs to their entry cap before ending the
window.

| Scenario | Required observation |
| --- | --- |
| NVML and DCGM writes | Each path reports a UUID-bound successful write/readback that independently matches the target; an all-skipped or matching-failure run fails. |
| Safe-default conflict | Backend marker stays absent and DGPB charges `U_c`, even when safe-default write/readback succeeds. |
| Agent restart | Transactional cap remains live, durable ownership reloads, and fresh evidence is republished. |
| DaemonSet rollout | Each unavailable Agent retains transactional caps; replacement reports match independent reads before the DGPB returns to `Idle`. |
| DCGM reconnect | Ambiguous or failed identity closes the fence; recovery republishes exact UUID-bound evidence. |
| Device re-enumeration | UUID identity survives index changes and no cap is accepted for the wrong GPU. |
| Same-name Pod replacement | Old UID evidence is rejected; the new backend waits for a new UID-bound allocation report. |
| External cap change | Agent detects and repairs the change; accepted report and independent read agree afterward. |
| Unknown eligible SKU | DGPB enters `Unqualified`, scale-up is blocked, and ordinary reconciliation clears it only after operator-owned qualification data is corrected. |
| Assigned live-range drift | DGPB charges `max(U_c, reportedLiveMaximum)` and remains `Unqualified` until the exact catalog range is observed. |
| Static regression | Transactional flags remain false, Phase 1 RBAC/mounts/rendering are unchanged, and shutdown restores static caps. |

A run is qualified only when all applicable rows are PASS. Record a missing
runtime image, unavailable DCGM combination, insufficient cluster permission,
or absent operator qualification source as BLOCKED with the exact artifact or
authority needed; do not replace it with a mock result.

## Rollback

Stop new transactional enrollment and scale enrolled workloads to a safe
state. The enrollment annotation is immutable: replace the DGD without that
annotation, or delete it, rather than trying to patch it away. Wait for its
worker Pods to disappear and verify the Agent has restored every released GPU
UUID to its factory default. Then set both feature flags to `false` and remove
the canary label. Leaving `agent.transactional.enabled=false` preserves the
Phase 1 RBAC and does not mount the PodResources socket.
