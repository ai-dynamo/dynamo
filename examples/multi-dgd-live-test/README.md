# Multi-DGD live test artifacts

Concrete YAMLs + helm values that realize §8 of
`docs/design-docs/multi-tenant-test.md` on the AKS `dpp-dev-env` cluster.
This scaffold pairs with the live test at
`components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py`
— the `prefill_engine_gpu_power_limit` / `decode_engine_gpu_power_limit`
knobs the manifests rely on are introduced by PR #9683
(`pr1b/planner-infra`).

## Two reproduction paths

Pick the one that matches the state of your environment:

| | **Path A — production shape** | **Path B — stub-workers (used 2026-05-21)** |
|---|---|---|
| Worker source | 3 real DGDs (`30/31/32-vllm-dgd-*.yaml`) | 5 raw pods (`50-stub-workers.yaml`) |
| Requires Dynamo operator? | Yes | No |
| Requires #9683 planner image? | Yes | No |
| Annotations emitted by | Power-aware planner | Hardcoded in pod YAML |
| Validates | Planner-side **and** agent-side | Agent-side only (NVML #9682, DCGM #9790) |
| Defer-until | #9683 lands on main + operator deployed | n/a — runs today |

Path A is the "design-doc" reproduction. Path B is what we actually ran
on 2026-05-21 (see [§ Live-run record](#2026-05-21-live-run-record-on-dpp-dev-env-aks)
below) and is the most ergonomic way to smoke-test the Power Agent on a
cluster that doesn't yet have the planner-infra wiring.

## Topology

Single 8-GPU A100 node (`$TEST_NODE`, labelled `power-test/node=kaim`):

| DGD | Planner type | Shape | GPUs | Cap (W) |
|---|---|---|---|---|
| `vllm-dgd-a` | DisaggPlanner | prefill TP=1 + decode TP=1 | 0, 1 | prefill=350, decode=325 |
| `vllm-dgd-b` | DisaggPlanner | prefill TP=2 + decode TP=2 | 2,3 / 4,5 | prefill=375, decode=325 |
| `vllm-dgd-c` | AggPlanner | aggregated TP=2 (decode-only) | 6, 7 | 300 |

A100 SXM4 max TDP = 400 W. Safe default = 280 W.

## Apply order

The test exercises **two separate actuator code paths** — NVML and DCGM.
Each path needs a clean Power Agent install configured for that actuator,
followed by its own pytest invocation. Running the suite once with one
Power Agent install can only ever prove one actuator's write path — the
other actuator's tests would either skip (DCGM tests gated on
`DCGM_HOSTENGINE_AVAILABLE=1`) or, worse, silently observe driver state
that was written by the *other* actuator and report it as a pass.

This README therefore lays out **Phase 1 (shared setup)** followed by
**Phase 2 (NVML pass)** and **Phase 3 (DCGM pass)**. Phase 3 is optional
on clusters where you don't want to deploy a DCGM hostengine.

```bash
export KUBECONFIG=$HOME/.kube/dynamo-kubeconfig
export TEST_NODE=aks-a100a-36888584-vmss000002   # or whichever idle 8-GPU node
export NS=kaim-dynamo-system
export TEST_FILE=components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py
```

### Phase 1 — Shared setup (run once)

```bash
# 1.1 namespace + label test node
kubectl apply -f 00-namespace.yaml
kubectl label node $TEST_NODE power-test/node=kaim --overwrite

# 1.2 Dynamo operator (§8.3.A in the design doc).
#     Chart name is `dynamo-platform` (see deploy/helm/charts/platform/Chart.yaml).
#     The release name we use is also `dynamo-platform` for consistency with
#     existing installs on this cluster.
helm upgrade --install dynamo-platform deploy/helm/charts/platform \
    --namespace $NS --create-namespace \
    --set dynamo-operator.controllerManager.manager.image.tag=1.2.0 \
    --set dynamo-operator.namespaceRestriction.enabled=true \
    --set dynamo-operator.namespaceRestriction.targetNamespace=$NS
kubectl wait deploy -n $NS \
    dynamo-platform-dynamo-operator-controller-manager \
    --for=condition=Available --timeout=180s

# 1.3 The 3 DGDs (§8.4) — same workload in both passes.
kubectl apply -f 30-vllm-dgd-A.yaml -n $NS
kubectl apply -f 31-vllm-dgd-B.yaml -n $NS
kubectl apply -f 32-vllm-dgd-C.yaml -n $NS
# Workers reach Ready (a few minutes — Qwen3-0.6B is tiny but the image is multi-GB)
for d in vllm-dgd-a vllm-dgd-b vllm-dgd-c; do
    kubectl wait pod -n $NS -l nvidia.com/dynamo-graph-deployment-name=$d \
        --for=condition=Ready --timeout=900s
done
```

### Phase 2 — NVML actuator pass

```bash
# 2.1 Install Power Agent in NVML mode.
#     If a DCGM-mode install exists from a prior phase 3, uninstall it first
#     so we get a clean cap-restore on SIGTERM before reinstalling. We MUST
#     wait for the old DS's pod to fully terminate before installing the new
#     DS — `helm uninstall` schedules deletion but the old pod takes up to
#     `terminationGracePeriodSeconds: 30` (from the chart) to SIGTERM-restore
#     caps and exit. If a new DS is installed during that window, both the
#     old (Terminating) and new (Pending/Running) pods coexist on the node
#     and the test fixture's `len(pods.items) == 1` assertion at
#     test_multi_dgd_live.py:186 fails.
if helm status power-agent -n $NS >/dev/null 2>&1; then
    helm uninstall power-agent -n $NS
fi
kubectl wait pod -n $NS \
    -l app.kubernetes.io/component=power-agent \
    --field-selector spec.nodeName=$TEST_NODE \
    --for=delete --timeout=90s || true   # tolerates "no pods to wait for"
helm upgrade --install power-agent deploy/helm/charts/power-agent \
    --namespace $NS -f 10-power-agent-values-nvml.yaml
kubectl rollout status ds -n $NS power-agent --timeout=180s
# Belt-and-suspenders: confirm the new pod is the ONLY power-agent pod on
# the test node before letting the suite read pods.items[0].
test "$(kubectl get pod -n $NS -l app.kubernetes.io/component=power-agent \
    --field-selector spec.nodeName=$TEST_NODE \
    --no-headers 2>/dev/null | wc -l)" = "1" \
    || { echo "Expected exactly 1 power-agent pod on $TEST_NODE"; exit 1; }

# 2.2 Run the live test. Do NOT set DCGM_HOSTENGINE_AVAILABLE — the
#     DCGM-specific tests will skip by design (we'll run them in phase 3).
#     On Path A, the two planner-side tests run. On Path B, they skip
#     automatically because there are no planner pods/logs to inspect.
kubectl exec -n $NS planner-dev -- env \
    RUN_MULTI_DGD_LIVE=1 \
    TEST_NODE=$TEST_NODE \
    python3.10 -m pytest /workspace/repo/$TEST_FILE \
        -v --tb=short \
        -k 'not test_dcgm_'
# Expected on Path A: TestPerDgdAnnotation (3), TestPowerAgentReconcile (2),
# TestCapApplicationProof::test_nvml_per_gpu_cap_matches_topology (1) PASS;
# TestMultiPodConflict SKIP; DCGM tests deselected by -k.
# Expected on Path B: 4 PASS (annotation presence, PowerAgent reconcile x2,
# NVML driver proof), 3 SKIP (2 Path-A-only planner tests + conflict class);
# DCGM tests deselected by -k.
```

### Phase 3 — DCGM actuator pass (optional)

```bash
# 3.1 Deploy standalone DCGM hostengine DS (§8.3.D.2 in the design doc).
#     Required because the cluster's GPU Operator runs with
#     dcgm.enabled=false (no operator-managed nvidia-dcgm hostengine).
kubectl apply -f 20-nvidia-dcgm-standalone-ds.yaml -n $NS
kubectl rollout status ds -n $NS nvidia-dcgm-standalone --timeout=180s

# 3.2 Reinstall Power Agent in DCGM mode. SIGTERM on the NVML-mode pods
#     restores GPU caps to default; the DCGM-mode pods then write the new
#     caps via dcgmConfigSet on their first reconcile.
#     Same wait-for-deletion pattern as phase 2.1 — without it, two
#     power-agent pods can coexist on the node during the helm cutover
#     and the test fixture's len(pods.items) == 1 fails.
helm uninstall power-agent -n $NS
kubectl wait pod -n $NS \
    -l app.kubernetes.io/component=power-agent \
    --field-selector spec.nodeName=$TEST_NODE \
    --for=delete --timeout=90s || true
helm upgrade --install power-agent deploy/helm/charts/power-agent \
    --namespace $NS -f 11-power-agent-values-dcgm.yaml
kubectl rollout status ds -n $NS power-agent --timeout=180s
test "$(kubectl get pod -n $NS -l app.kubernetes.io/component=power-agent \
    --field-selector spec.nodeName=$TEST_NODE \
    --no-headers 2>/dev/null | wc -l)" = "1" \
    || { echo "Expected exactly 1 power-agent pod on $TEST_NODE"; exit 1; }

# 3.3 Re-run the same test, this time WITH DCGM_HOSTENGINE_AVAILABLE=1.
#     The NVML cap-application proof STILL runs and STILL passes (NVML's
#     read of the cap reflects the DCGM-set value), but it no longer
#     uniquely proves the NVML actuator's write path — phase 2 did that.
#     What phase 3 newly proves is the DCGM write path + target-config
#     registration.
kubectl exec -n $NS planner-dev -- env \
    RUN_MULTI_DGD_LIVE=1 \
    TEST_NODE=$TEST_NODE \
    DCGM_HOSTENGINE_AVAILABLE=1 \
    python3.10 -m pytest /workspace/repo/$TEST_FILE \
        -v --tb=short
# Expected on a freshly-rebuilt suite:
#   PASS — TestPowerAgentReconcile (2 tests, actuator-blind counters/metrics)
#   PASS — TestCapApplicationProof::test_nvml_per_gpu_cap_matches_topology
#   PASS — TestCapApplicationProof::test_dcgm_per_gpu_cap_matches_topology
#          (rewritten 2026-05-21 to use per-GPU-group iteration; see
#          Finding #9 + the test's docstring)
#   PASS — TestCapApplicationProof::test_dcgm_target_config_registered
#          (same per-GPU-group iteration shape, reads Target instead of Current)
#   PASS — TestPerDgdAnnotation (3 tests on Path A)
#   SKIP — TestMultiPodConflict (Finding #7: AKS CDI rejects
#          NVIDIA_VISIBLE_DEVICES=<uuid>; class-level skip with reason)
```

If you only have time for one pass, run Phase 2 (NVML) — it covers the
common-path actuator. The conflict scenario is currently skipped on
AKS-CDI clusters (Finding #7); rerunning Phase 2 with that test
re-enabled is the planned follow-up once the resource-claim rewrite lands.

## What this complements

PR #9683 already ships
`components/src/dynamo/planner/tests/integration/test_actuation_knobs_live.py`
which validates per-DGD annotation round-trip, label selectors, and
`_apply_power_annotations` orchestration **for a single DGD**. This
artifact adds the cross-DGD invariants that single-DGD tests can't reach:

- A2 — three planners patch disjoint pod sets
- B2 / B4 — agent applies 8 caps with zero counter increments
- B5 — multi-pod conflict + safe-default behaviour
- §8.5.6 / §8.6.2 — actual NVML / DCGM cap-application proven by
  reading the driver state, not via mocks

## Path B reproduction — stub-workers (run on a cluster WITHOUT #9683 / operator)

Skip Phase 1 step 1.2 (operator) and step 1.3 (DGDs) from above. Replace
with the stub-workers manifest, which carries the same labels +
annotations the planner would emit but as plain pod manifests.

The test suite **auto-detects** which path you're on by looking for
planner pods at import time
(`test_multi_dgd_live.py::_detect_path_a_planner_pods`):

- **Path-A-only tests** (skipped on Path B with explicit reason):
  - `TestPerDgdAnnotation::test_aggregated_dgd_c_pod_was_patched_via_decode_branch`
    (would pass *vacuously* on Path B because stub-workers carry the
    same labels statically; assertion's value rests on AggPlanner having
    emitted the annotation, so we skip rather than false-pass).
  - `TestPerDgdAnnotation::test_no_cross_dgd_annotation_contamination`
    (scrapes planner stdout for `Annotated pod ... with ...` log lines;
    no planner pods = no logs to scrape; would hard-fail without the skip).
- **Path-A-or-B tests** (run + pass on both):
  - `TestPerDgdAnnotation::test_all_5_worker_pods_carry_expected_annotation`
  - `TestPowerAgentReconcile` (2 tests)
  - `TestCapApplicationProof` (NVML; DCGM gated additionally on
    `DCGM_HOSTENGINE_AVAILABLE=1`)
- **Class-level-skipped tests** (skipped on every path until rewritten):
  - `TestMultiPodConflict` (Finding #7 — AKS CDI mode)

So on Path B the suite reports e.g. `4 passed, 3 skipped` with skip
reasons that tie back to specific findings — no need for a `-k` filter.

### Path-B prerequisites

You need two non-mainline images, both built on a Linux host with NGC
read access (`viking-prod-216` on the NVIDIA internal network worked
for us). Once built, push to **ttl.sh** — it requires no auth and the
test cluster's existing `nvcr-imagepullsecret` can't read
`nvcr.io/nvidia/cloud-native/dcgm`. ttl.sh images self-expire in ≤24h
so the namespace stays clean after the test.

```bash
# On viking-prod-216 (or any NGC-authenticated Linux host):
cd /tmp/kaim && git clone <dynamo-repo-url> dynamo   # or use an existing checkout
cd /tmp/kaim/dynamo

# ----- Image 1: Power Agent with DCGM 4.5.1 bindings -----
# Why not the Dockerfile default? The pinned 4.2.3-2 tag does NOT exist
# on NGC (see Finding #1); 4.5.x moved bindings to a new path (Finding #2);
# the pydcgm bindings in the NGC image are missing logger.py (Finding #6);
# and actuator.py uses c_dcgmDeviceConfig_v2 which only exists in 4.x
# (Finding #5). The block below works around all four:
mkdir -p /tmp/kaim/pa-dcgm45-build
sed 's|/usr/local/dcgm/bindings/python3|/usr/share/datacenter-gpu-manager-4/bindings/python3|g' \
    components/power_agent/Dockerfile > /tmp/kaim/pa-dcgm45-build/Dockerfile
cp components/power_agent/power_agent.py components/power_agent/actuator.py \
    /tmp/kaim/pa-dcgm45-build/

# logger.py shim — the NGC DCGM image's pydcgm imports `logger` but the
# upstream logger.py is in a separate source tree that isn't packaged.
cat > /tmp/kaim/pa-dcgm45-build/logger.py <<'PYEOF'
import logging as _logging
_LOG = _logging.getLogger("dcgm.bindings")
debug = _LOG.debug
info = _LOG.info
warning = _LOG.warning
error = _LOG.error
critical = _LOG.critical
log = _LOG.log
PYEOF

# Insert the COPY for logger.py after the vendored-lib COPY
awk '/COPY --from=dcgm-vendor \/vendor\/dcgm-lib    \/opt\/dcgm\/lib/ {
        print
        print "COPY logger.py /opt/dcgm/python/logger.py"
        next
      } { print }' /tmp/kaim/pa-dcgm45-build/Dockerfile > /tmp/Dockerfile.new \
    && mv /tmp/Dockerfile.new /tmp/kaim/pa-dcgm45-build/Dockerfile

cd /tmp/kaim/pa-dcgm45-build
docker buildx build --load \
    -t ttl.sh/dynamo-pa-kaim-dcgm45-v2:24h \
    --build-arg DCGM_IMAGE=nvcr.io/nvidia/cloud-native/dcgm:4.5.1-1-ubuntu22.04 \
    -f Dockerfile .
docker push ttl.sh/dynamo-pa-kaim-dcgm45-v2:24h

# ----- Image 2: NVML-only Power Agent build (DCGM 3.3.9 bindings) -----
# Used by Phase 2 (NVML pass) when you don't need to exercise the DCGM
# code path. Cheaper to build because we skip the path patch.
docker buildx build --load \
    -t ttl.sh/dynamo-pa-kaim-4c3c606:24h \
    --build-arg DCGM_IMAGE=nvcr.io/nvidia/cloud-native/dcgm:3.3.9-1-ubuntu22.04 \
    -f components/power_agent/Dockerfile components/power_agent
docker push ttl.sh/dynamo-pa-kaim-4c3c606:24h

# ----- Image 3: DCGM 4.5.1 hostengine -----
# The cluster's nvcr-imagepullsecret 403s on cloud-native/dcgm
# (Finding — see below). Retag through ttl.sh.
docker pull nvcr.io/nvidia/cloud-native/dcgm:4.5.1-1-ubuntu22.04
docker tag  nvcr.io/nvidia/cloud-native/dcgm:4.5.1-1-ubuntu22.04 \
            ttl.sh/dynamo-dcgm45-kaim:24h
docker push ttl.sh/dynamo-dcgm45-kaim:24h
```

If you bump any source file between successive builds you MUST also bump
the ttl.sh tag (e.g. `:24h-v2`) — kubelet's `IfNotPresent` policy on the
helm chart skips the re-pull when the tag matches an already-cached
image.

### Path-B execution

```bash
export KUBECONFIG=$HOME/.kube/dynamo-kubeconfig
export TEST_NODE=aks-a100b-22138447-vmss000000   # any idle 8-GPU node
export NS=kaim-dynamo-system
export TEST_FILE=components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py
export FIX_DIR=examples/multi-dgd-live-test    # this README's directory

# B.1 — namespace + node label
kubectl apply -f $FIX_DIR/00-namespace.yaml
kubectl label node $TEST_NODE power-test/node=kaim --overwrite

# B.2 — Power Agent in NVML mode (skip operator + DGDs)
helm.exe upgrade --install power-agent deploy/helm/charts/power-agent \
    --namespace $NS \
    -f $FIX_DIR/10-power-agent-values-nvml.yaml \
    --wait --timeout 5m
kubectl rollout status ds/power-agent -n $NS --timeout=180s

# B.3 — stub workers (5 pods, 8 GPUs total). Annotations are hardcoded
#       to mimic what the power-aware planner would have emitted.
kubectl apply -f $FIX_DIR/50-stub-workers.yaml
kubectl wait pod -n $NS -l purpose=power-aware-multi-dgd-test \
    --for=condition=Ready --timeout=5m

# B.4 — run the NVML pass. Path-A-only tests skip automatically
#       (auto-detection picks up the absence of planner pods); DCGM
#       tests are deselected here and run in B.7.
RUN_MULTI_DGD_LIVE=1 TEST_NODE=$TEST_NODE \
python3.10 -m pytest $TEST_FILE -v --tb=short -k 'not test_dcgm_'
# Expected: 4 PASSED, 3 SKIPPED. DCGM tests deselected by -k.
#   passed:  test_all_5_worker_pods_carry_expected_annotation,
#            test_applied_limit_watts_metric_matches_topology,
#            test_happy_path_zero_counters,
#            test_nvml_per_gpu_cap_matches_topology
#   skipped: test_aggregated_dgd_c_pod_was_patched_via_decode_branch (Path-A only),
#            test_no_cross_dgd_annotation_contamination (Path-A only),
#            TestMultiPodConflict class (Finding #7).

# --- DCGM pass --------------------------------------------------------
# B.5 — standalone DCGM 4.5.1 hostengine DS (image was retagged to
#       ttl.sh in the build step above).
kubectl apply -f $FIX_DIR/20-nvidia-dcgm-standalone-ds.yaml
kubectl rollout status ds/nvidia-dcgm-standalone -n $NS --timeout=180s

# B.6 — clean cut-over to DCGM-mode Power Agent
helm.exe uninstall power-agent -n $NS
kubectl wait pod -n $NS -l app.kubernetes.io/name=power-agent \
    --for=delete --timeout=90s || true
helm.exe install power-agent deploy/helm/charts/power-agent \
    --namespace $NS \
    -f $FIX_DIR/11-power-agent-values-dcgm.yaml \
    --wait --timeout 5m

# B.7 — re-run with DCGM mode enabled (no -k filter needed — auto-detect)
RUN_MULTI_DGD_LIVE=1 TEST_NODE=$TEST_NODE DCGM_HOSTENGINE_AVAILABLE=1 \
python3.10 -m pytest $TEST_FILE -v --tb=short
# Expected on the rewritten suite (post-2026-05-21 fixes):
#   6 PASSED, 3 SKIPPED.
#   newly passing (vs B.4): test_dcgm_per_gpu_cap_matches_topology,
#                           test_dcgm_target_config_registered.
#   still skipped:          2 Path-A tests + TestMultiPodConflict class.
```

## 2026-05-21 live-run record on dpp-dev-env AKS

Executed end-to-end on `aks-a100b-22138447-vmss000000` (NVIDIA A100-SXM4-80GB
×8, driver 590.48.01, CUDA 13.1, Ubuntu 22.04). Captured for future
reference and as PR-feedback context.

### Results

| Phase | Test class / target | Outcome |
|---|---|---|
| 1 NVML | `TestPerDgdAnnotation::test_all_5_worker_pods_carry_expected_annotation` | **PASS** |
| 1 NVML | `TestPowerAgentReconcile::test_applied_limit_watts_metric_matches_topology` | **PASS** |
| 1 NVML | `TestPowerAgentReconcile::test_happy_path_zero_counters` | **PASS** |
| 1 NVML | `TestCapApplicationProof::test_nvml_per_gpu_cap_matches_topology` | **PASS** |
| 1 NVML | `TestMultiPodConflict::test_conflict_triggers_safe_default_on_gpu0_only` | **BLOCKED** by AKS CDI (Finding #7) — *now class-level `@pytest.mark.skip` with that reason; no longer false-fails on a re-run* |
| 2 DCGM | Same 4 NVML tests re-run with PA in DCGM mode | **PASS** (4/4) |
| 2 DCGM | `TestCapApplicationProof::test_dcgm_per_gpu_cap_matches_topology` | **SKIP** on live run (`dcgmi --json -g all` parse error — Finding #9). *Rewritten 2026-05-21 to iterate per-GPU groups; expected to PASS on next live run.* **Update 2026-05-21T20:11Z** — next live Path-B run (same date, post-rewrite) SKIPed again on a NEW defect: `kubernetes-client>=35.0.0` mangles JSON pod-exec stdout into a Python-dict-repr string before `json.loads` sees it (Finding #11). Manual per-group readback shows the DCGM contract holds (`Current == Target`, multiset matches topology). |
| 2 DCGM | `TestCapApplicationProof::test_dcgm_target_config_registered` | **SKIP** on live run (same `dcgmi --json` issue). *Same rewrite; reads `Target` instead of `Current` from the same per-GPU-group payload.* **Update 2026-05-21T20:11Z** — same Finding #11 mangling causes the same skip; `Target` field present and matches NVML on manual readback. |
| 3 Planner | `TestPerDgdAnnotation::test_aggregated_dgd_c_pod_was_patched_via_decode_branch`, `TestPerDgdAnnotation::test_no_cross_dgd_annotation_contamination` | **DEFERRED** (no operator + no live planner) — *now auto-`skipif(not _PATH_A_AVAILABLE)`; suite reports SKIP cleanly on Path B instead of either failing or being filtered out by `-k`* |

Manual verification of the 2 SKIPPED DCGM tests (per-GPU-group query):

```text
gpu0 (dcgmGroup=2): current=325W, target=325W
gpu1 (dcgmGroup=3): current=325W, target=325W
gpu2 (dcgmGroup=4): current=350W, target=350W
gpu3 (dcgmGroup=5): current=325W, target=325W
gpu4 (dcgmGroup=6): current=375W, target=375W
gpu5 (dcgmGroup=7): current=375W, target=375W
gpu6 (dcgmGroup=8): current=300W, target=300W
gpu7 (dcgmGroup=9): current=300W, target=300W
```

`Current == Target` for all 8 GPUs proves both `dcgmConfigSet` AND
`dcgmConfigEnforce` wrote successfully to the driver — the substantive
contract the 2 skipped tests were trying to assert. The skip is a
test-code shape issue, not a real DCGM actuator failure.

### PR feedback findings discovered during the run

Each is captured as an inline comment in the file noted; the index here
is just for cross-referencing.

1. **Power Agent Dockerfile pins a non-existent DCGM tag.**
   `nvcr.io/nvidia/cloud-native/dcgm:4.2.3-2-ubuntu22.04` returns 404
   on NGC (verified via `docker manifest inspect`). The chart's image
   build will fail without an explicit `--build-arg DCGM_IMAGE=...`
   override. File: `components/power_agent/Dockerfile:35`.
2. **DCGM 4.5+ moved bindings path.** From
   `/usr/local/dcgm/bindings/python3/` to
   `/usr/share/datacenter-gpu-manager-4/bindings/python3/`. The
   Dockerfile's `COPY` still targets the old path, so building with
   any 4.5+ tag fails silently (copies 0 files). File same as #1, line 47.
3. **Chart `rbac.namespaceRestricted=true` mismatched with agent CLI.**
   The chart creates a namespace-scoped `Role`, but the daemonset
   template never passes `--namespace={{ .Release.Namespace }}` to the
   agent — so the agent falls back to `list_pod_for_all_namespaces`
   (cluster-scope) and gets a `403 Forbidden`. Files:
   `deploy/helm/charts/power-agent/templates/daemonset.yaml`,
   `power_agent.py:798` (the unused `--namespace` flag).
4. **Chart `limits.memory: 128Mi` too tight for live testing.**
   The agent's steady-state RSS is ~80 MiB, but a `kubectl exec --
   python -c "from kubernetes import ..."` probe alongside it
   OOM-killed the agent twice on 2026-05-21. Live tests need at least
   256 MiB. File: `deploy/helm/charts/power-agent/values.yaml`.
5. **`actuator.py` calls a DCGM 4.x-only struct.**
   `dcgm_structs.c_dcgmDeviceConfig_v2` was added in DCGM 4.0. With
   DCGM 3.x bindings the agent crashes at the first reconcile with
   `AttributeError`. The Dockerfile's pin (per Finding #1) doesn't
   exist, and the only DCGM 4.x tag we found (4.5.1) needs the path
   fix from Finding #2. File: `components/power_agent/actuator.py:807`.
6. **NGC DCGM image ships pydcgm with a missing `logger.py`.**
   `DcgmGroup.py:20` does `import logger` then calls `logger.debug(...)`,
   but `logger.py` lives in a separate
   `datacenter-gpu-manager-4/testing/python3/` source tree that isn't
   packaged into the NGC runtime image. Workaround: drop a 10-line
   stdlib-`logging` shim into `/opt/dcgm/python/logger.py` (see the
   Path-B build script above).
7. **`TestMultiPodConflict` needs a CDI-mode-safe bystander pattern.**
   The test pins the bystander to a specific GPU UUID via
   `NVIDIA_VISIBLE_DEVICES=<uuid>`. AKS clusters in CDI device mode
   reject both index and UUID forms with
   `unresolvable CDI devices management.nvidia.com/gpu=*`. Fix: use
   `nvidia.com/gpu: 1` device-plugin claim + discover the assigned
   GPU UUID from `nvmlDeviceGetComputeRunningProcesses` inside the
   agent pod, then drive the conflict assertions off that UUID.
   File: `components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py:849-862`.
8. **`_EXPECTED_CAP_PER_GPU` hard-codes per-index mapping.**
   The K8s device plugin assigns GPU indices nondeterministically — on
   our run the TP=2 pods landed on GPUs 0,1 and 4,5,6,7 instead of the
   2,3 and 4,5 and 6,7 indices the table assumed. We patched the tests
   to compare via `collections.Counter(...values())` (multiset
   equivalence). Both per-index and per-multiset values are printed on
   assertion failure for diagnostics. Files: same test file, the four
   `assert ... == _EXPECTED_CAP_PER_GPU` sites.
9. **DCGM 4.x dcgmi UX changes break the test's readback path.**
   `dcgmi config --get -g all -j` prints usage and exits 1.
   `dcgmi config --get-target` was removed entirely. The right shape
   is to iterate the per-GPU groups the Power Agent creates
   (`dynamo-power-agent-gpu-N`, dynamic groupIds 2..9 on a fresh
   hostengine) using `dcgmi config --get -g <groupId> -j`. The 2
   `test_dcgm_*` tests should be updated to do this lookup rather
   than expect a single all-GPUs JSON payload. File:
   `components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py:660-661, 735`.
10. **DCGM-mode SIGTERM handler does not restore caps to default.**
    After `helm uninstall power-agent` in DCGM mode, all 8 GPUs were
    observed to retain the test caps (325/325/350/325/375/375/300/300 W
    on A100-80GB w/ 400 W default) rather than being restored. The
    NVML-mode SIGTERM path (`test_shutdown.py`) is unit-tested, but the
    DCGM-mode equivalent either:
    (a) doesn't call `dcgmConfigSet(default)` on shutdown,
    (b) calls it but the prior `dcgmConfigEnforce` re-applies the cap
    via the stored target before the agent exits, or
    (c) calls it but the hostengine has already disconnected.
    Workaround during teardown: spawn a one-shot privileged pod from
    the same Power Agent image and call
    `pynvml.nvmlDeviceSetPowerManagementLimit(h, default)` directly
    (the `99-teardown.sh` `power-restore-verify` step in this directory
    catches the drift; an analogous `power-restore-do` step would fix
    it). Suggested fix is to add a regression test mirroring
    `test_shutdown.py` for the DCGM actuator. File:
    `components/power_agent/actuator.py` (DCGM restore path).
11. **`kubernetes-client>=35.0.0` mangles JSON pod-exec stdout into a
    Python-dict-repr string.** Surfaced on the 2026-05-21T20:11Z re-run
    of Path B (the first live exercise of the 2026-05-21 DCGM-test
    rewrite). `kubernetes.stream.stream(..., _preload_content=True)`
    sniffs JSON-shaped stdout, parses it client-side, then returns
    `str(dict)`, e.g. `printf '%s' '{"a":1}'` comes back as the literal
    8-char string `{'a': 1}` (single quotes). `json.loads(raw)` then
    raises `JSONDecodeError`, and the rewritten `test_dcgm_*` cases
    silently skip on the "did not return JSON" branch they intended as
    a *defensive* skip for genuinely-unparseable hostengine versions.
    Manual readback via the same `dcgmi config --get -g <id> -j` shows
    `Current == Target` for all 8 GPUs (multiset
    `{375×2, 325×3, 300×2, 350×1}` matches topology) — i.e. the DCGM
    actuator's write contract IS satisfied; only the test plumbing is
    broken. There is also a **second** shape issue stacked underneath:
    DCGM 4.5.1 `dcgmi config --get -g <id> -j` returns the per-field
    payload as `{"body":{"Power Limit":{"children":{"Current":{"value":"375"}}}}}`,
    NOT the `{"body":[{"Power Limit (W)":{"Current":"375"}}, ...]}`
    shape the rewrite assumed (the field is `"Power Limit"`, not
    `"Power Limit (W)"`, and `body` is a dict of fields, not a list of
    GPU entries).

    **Status: fixed in `test_multi_dgd_live.py` (commit landing with
    PR #9683 alongside this README).** The fix combines (b) — `_exec_in_pod`
    now passes `_preload_content=False` and accumulates raw stdout via
    `WSClient.read_stdout()` — with a new `_extract_power_limit_from_dcgmi_body`
    helper that walks BOTH the DCGM 4.0–4.4 (list-of-fields, `"Power Limit (W)"`,
    flat `Current/Target`) shape AND the DCGM 4.5+ (dict-of-fields,
    `"Power Limit"`, nested `children.{Current,Target}.value`) shape
    uniformly. Verification: feeding the helper synthetic payloads for
    both shapes returns the expected `(current, target)` watts; the next
    live re-run on `aks-a100b-22138447-vmss000000` is expected to flip
    `test_dcgm_per_gpu_cap_matches_topology` and
    `test_dcgm_target_config_registered` from SKIPPED to PASSED.

### Live-run patches kept in tree

These are the patches we applied during the run that should be
preserved (already saved):

- `10-power-agent-values-nvml.yaml` — switched to `ttl.sh/dynamo-pa-kaim-4c3c606:24h`
  image, cluster-scoped RBAC (workaround for Finding #3), bumped
  memory limit (Finding #4). Inline comments explain each override.
- `11-power-agent-values-dcgm.yaml` — switched to
  `ttl.sh/dynamo-pa-kaim-dcgm45-v2:24h`, same RBAC + memory tweaks.
- `20-nvidia-dcgm-standalone-ds.yaml` — DCGM 4.5.1 image via ttl.sh
  (Findings #1 + #2 + #5 + #6), corrected `nv-hostengine` CLI flags
  (`-n -b 0.0.0.0 -p 5555` — the design-doc's `--bind-address`/`--port`
  long forms don't exist in 3.x or 4.x).
- `50-stub-workers.yaml` — new artifact (Path B). 5 pods, hardcoded
  annotations, `nvidia.com/gpu: N` resource claims (not
  `NVIDIA_VISIBLE_DEVICES` — see Finding #7 for why).
- `components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py` —
  `_scrape_power_agent_metrics` rewritten to use `python -c urllib`
  (the agent image has no `curl`); `_read_nvml_caps_via_pod` emits
  `key=value` lines rather than `json.dumps` (the kubernetes
  Python-stream exec mangles the JSON quoting under some conditions);
  all 4 per-index cap assertions switched to multiset comparison
  (Finding #8); module-level `k8s_config.load_incluster_config()`
  now falls back to `load_kube_config()` so the suite can be driven
  from a laptop with `$KUBECONFIG`; `TestMultiPodConflict` blast-radius
  assertion compares against an observed pre-conflict baseline rather
  than the hard-coded topology table (also Finding #8).

## Recovery / teardown

```bash
TEST_NODE=aks-a100b-22138447-vmss000000 \
    NS=kaim-dynamo-system \
    bash 99-teardown.sh
# Cleans up:
#   - bystander pod, stub workers, 3 DGDs, profile CMs
#   - power-agent helm release (NVML or DCGM)
#   - nvidia-dcgm-standalone DS + SA + Svc
#   - power-test/node label on $TEST_NODE
# Then:
#   - actively restores all GPU caps to their NVML default
#     (works around Finding #10 — DCGM-mode SIGTERM doesn't restore)
#   - prints per-GPU drift+after watts so you can audit the result
```

After teardown, one piece of host-side state remains on the test node
under `/var/lib/dynamo-power-agent-kaim-test/` (configured by
`10-power-agent-values-nvml.yaml:64`). This contains the agent's
`managed_gpus.json` and is read by orphan-recovery on the next agent
startup — its presence is **benign** (orphan recovery restores managed
GPUs to default if no annotated pod claims them), but if you want a
truly pristine node, SSH to the node and `rm -rf
/var/lib/dynamo-power-agent-kaim-test/`. AKS does not give us SSH
directly, so on AKS this requires either a privileged DaemonSet or
`kubectl debug node/...` — we leave the cleanup to the user since it's
non-functional.

## Known caveats

- **Qwen3-0.6B is overkill-tiny for a power test** — the model barely
  loads the GPU. We're testing *power-cap enforcement*, not the
  effectiveness of the cap. A larger model would produce more
  observable wattage draw but isn't needed for the contract assertions.
- **NVML/DCGM cap values are observational, not enforced as a *minimum* TDP.**
  GPU may run well below the cap during the test. The assertion is
  "`nvmlDeviceGetPowerManagementLimit` returns the expected value,"
  which proves the WRITE landed; observability of the cap's *effect* on
  power draw is out of scope.
- **`nv-hostengine` standalone DS shares /var/run/nvidia-mps with the
  driver** — coexists fine with the GPU Operator's `nvidia-dcgm-exporter`
  (verified empirically; both use the embedded host engine pattern but
  the standalone DS adds its own TCP listener on 5555 in the
  `kaim-dynamo-system` namespace, no clash).
