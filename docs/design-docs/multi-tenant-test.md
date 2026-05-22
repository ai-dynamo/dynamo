# Multi-Tenant Node Test Plan — PRs #9682, #9683, #9790

**Status:** v0.8 (follow-up hardening pass — see Changelog at end)
**Scope:** Phase 1 only (NVML/DCGM cap actuation + per-pod annotation contract).
Phase 2 (`_apply_power_budget`) and Phase 3+ (AIC closed loop) are out of scope.

**Tier coverage:** §1–§7 cover Tier 1/2 (HW-agnostic, multi-framework). §8 specializes
Tier 3 to the AKS `dpp-dev-env` cluster — VLLM-only on A100 SXM4 (see §8.1
ground truth). The Tier 1/2 multi-framework topology in §2 is **not** what
the live test exercises; the live test uses the §8.2 VLLM-only A100 topology.

---

## 1. Why this gap matters

The three PRs intersect at one contract — `dynamo.nvidia.com/gpu-power-limit` on a
worker Pod. PRs #9682 / #9790 only see the agent side (one DaemonSet per node, sees
everything on the node). PR #9683 only sees the planner side (one planner per DGD,
sees only its own pods). The seam between them — *"N planners ↔ 1 agent ↔ M
frameworks on one node"* — is never exercised by any current test:

| Surface | Today's coverage | What multi-framework breaks |
|---|---|---|
| Planner `_apply_power_annotations` (#9683, `core/base.py:746-794`) | One DGD, one component type; existing tests only verify the selector string in isolation | Three planners simultaneously PATCHing disjoint pod sets in one cluster |
| Power Agent `_reconcile_gpu` + UID dedup (#9790 commit `ba3b5803a9`) | Dedup tested with hand-rolled 2-PID-one-pod cases | Mixed TP / aggregated topology where TRTLLM tp=2 and SGLang tp=2 colocate on one node |
| Power Agent multi-pod policy (#9682 `tests/test_multi_pod_policy.py`) | Inputs are hand-rolled `[("uid-1","480"), ("uid-2","350")]` tuples | Well-formed mixed topology should leave every counter at **zero** — currently no test asserts this as a positive property |
| DCGM identity map (#9790 `actuator.py:606-667`) | 2-GPU mismatch case in isolation | 8 distinct UUIDs across 5 pods and 3 frameworks all need stable mapping |
| Planner Prometheus gauges (#9683 `monitoring/planner_metrics.py`) | One planner per Prom port — production-correct | (No bug — see §3 A-note: each planner pod has its own Prom port; this is a fixture concern only) |

The multi-pod-per-GPU counter is the only place "more than one DGD on this node" is
currently named in code, and it's tested as a **misconfiguration**. The
"stays-at-zero in the supported mixed topology" invariant is the inversion we need.

---

## 2. Canonical topology under test

```
Node:  umb-b200-N  (8 × B200, single nvidia-dcgm hostengine, single power-agent pod)

  GPU 0  ──┐
  GPU 1    ├─ DGD-V (vLLM disagg)       prefill pod V-P    cap=450W   1 GPU
                                        decode  pod V-D    cap=425W   1 GPU

  GPU 2  ──┐
  GPU 3    ├─ DGD-T (TRTLLM disagg)     prefill pod T-P    cap=480W   2 GPUs (tp=2)
  GPU 4    ├
  GPU 5    ┘                            decode  pod T-D    cap=450W   2 GPUs (tp=2)

  GPU 6  ──┐ DGD-S (SGLang aggregated)
  GPU 7    ┘                            aggr    pod S-A    cap=500W   2 GPUs (tp=2)
```

Invariants we want to assert:

1. **5 pods, 8 GPUs, exactly 1 pod-uid per GPU.** No multi-pod warnings or conflicts.
2. **5 pod annotations** (carrying 4 distinct values — V-P and T-D both at 450W)
   patched by **3 independent planner reconcile loops**, each scoped by its own
   label selector.
3. **8 `apply_cap(gpu_idx, watts)` calls per agent tick**, with `(gpu_idx, watts)`
   matching the topology table (4 distinct watt values across 8 GPUs).
4. **N PIDs per GPU may be ≥ 1** (TP-rank main + helper processes); dedup-by-UID
   collapses them to one pod per GPU.
5. **No cross-DGD annotation contamination** — *enforced by correct selector
   construction in `get_component_pods` and by namespace-scoped RBAC on the
   planner's ServiceAccount, NOT by the K8s API.* `patch_pod_annotation(pod_name,
   key, value)` will happily PATCH any pod in the planner's namespace; the test
   must pin the selector string and verify `patch_namespaced_pod` is only ever
   called with pod-names returned by that selector.

---

## 3. Coverage matrix

Letters: A planner-only, B agent-only, C cross-component, D observability,
E failure modes. The Tier column gives the cheapest test that catches it.

### A. Planner side — per-DGD scoping (#9683)

| ID | Assertion | Tier |
|---|---|---|
| A1 | `KubernetesConnector.get_component_pods(PREFILL)` for DGD-V constructs the exact selector `nvidia.com/dynamo-graph-deployment-name=<dgd-V>,nvidia.com/dynamo-component=<service-key>` where `<service-key>` is the operator's service-name (e.g. `VllmPrefillWorker`). Test pins **both halves** of the selector string. | 1 (unit) |
| A2 | Three planner instances (DGD-V, DGD-T, DGD-S) running against a shared `MockKubernetesAPI` produce three disjoint sets of `patch_pod_annotation(pod_name, ...)` calls, each carrying that planner's own `prefill_engine_gpu_power_limit` / `decode_engine_gpu_power_limit`. **No pod_name ever leaks between planners.** | 1 (unit) |
| A3 | `AggPlanner` has `require_prefill=False, require_decode=True` (`core/adapters.py:99-103`). `_apply_power_annotations` therefore **enters the DECODE branch only** and patches the aggregated worker pod (S-A) via `SubComponentType.DECODE` with `decode_engine_gpu_power_limit`. Test asserts: (a) `get_component_pods(PREFILL)` is **not** called for AggPlanner; (b) `get_component_pods(DECODE)` is called once and returns S-A; (c) the PATCH body carries 500. | 1 (unit) |
| A4 | If an external mutator overwrites a pod's annotation with a wrong value — including a value that happens to be another DGD's `prefill_engine_gpu_power_limit` (e.g., DGD-V's V-P gets stamped with 480W = DGD-T's value) — the next planner tick re-PATCHes it back to V-P's own 450W. Verifies no convergence across DGDs. | 1 (unit) |

**A-note (was A5).** In production, three planners run as three separate K8s
pods, each with its own `prometheus_port` — different processes, different
Prom endpoints, no collision possible.

The "three planners in one Python process" case only arises in tests.
**Distinct ports alone do NOT solve it.** `PlannerPrometheusMetrics.__init__()`
(see `monitoring/planner_metrics.py`) constructs `Gauge(...)` / `Counter(...)`
calls with no `registry=` argument — they register on the process-wide
`prometheus_client.REGISTRY`. Instantiating two `PlannerPrometheusMetrics`
objects in one process raises `Duplicated timeseries in CollectorRegistry`
regardless of port assignment.

**Fixture path that works today:** use bare planner instances
(`object.__new__(NativePlannerBase)`) with `planner.prometheus_metrics =
MagicMock()` — the existing pattern in `test_actuation_knobs.py`. The mock
absorbs `.set(...)` / `.inc(...)` calls without touching the real registry.
This is what §5.2's planner-side stub does.

**Fixture path that needs product change:** if a future test wants real
metric *values* (not just call shapes), `PlannerPrometheusMetrics.__init__`
would need a `registry=None` parameter defaulting to the global registry — a
small backwards-compatible product fix landed alongside the test. Out of
scope for this plan.

### B. Power Agent side — node-wide reconcile (#9682, #9790)

| ID | Assertion | Tier |
|---|---|---|
| B1 | `_list_pods_on_node()` returns all 5 pods (with 3 different `nvidia.com/dynamo-graph-deployment-name` label values); `_build_uid_to_annotation` yields 5 entries with the topology-table values. | 2 (pseudo-integration) |
| B2 | After one `reconcile_once()`: 8 cap-write calls fire with `(gpu_idx, cap_w)` matching the topology table — gpu 0→450, gpu 1→425, gpu 2,3→480, gpu 4,5→450, gpu 6,7→500. (Note: 4 distinct watt values across 8 calls — value 450 appears 3× on gpus 0, 4, 5.) **Per-branch surface:** on #9790 the assertion is `actuator.apply_cap(gpu_idx, cap_w)`; on #9682 (no actuator abstraction — `pr1a/power-agent` has only `power_agent.py`, no `actuator.py`) the assertion is `pynvml.nvmlDeviceSetPowerManagementLimit(handle, cap_w * 1000)` (or a patched module-level `_apply_cap`). The actuator Protocol arrives in #9790. | 2 |
| B3 | When TRTLLM prefill (tp=2) is mocked to have **2 PIDs on each of GPU 2 and GPU 3** (rank main + helper), `multi_pod_gpu_total{disposition=agree}` stays at zero. This is the regression test for the UID-dedup added by `ba3b5803a9` on **#9790**. **Cannot be ported back to #9682** — that branch's `_reconcile_gpu` appends one entry per PID and the test would spuriously fire the multi-pod-agree warning. | 2 |
| B4 | `multi_pod_gpu_total{disposition="agree"}`, `{disposition="conflict"}`, `safe_default_applied_total`, and `apply_failures_total` all stay at zero across the full reconcile in the well-formed topology. (Today these are implicitly true but never asserted as positive invariants.) | 2 |
| B5 | Deliberate misconfig: schedule a hypothetical DGD-T helper pod onto GPU 0 (where DGD-V prefill lives) with a different annotation value. **`multi_pod_gpu_total{disposition="conflict"}` increments to exactly 1, `safe_default_applied_total` increments to exactly 1, GPU 0's cap-write call carries `safe_default_watts`. GPUs 1–7 are still reconciled this tick (the loop iterates every GPU), but each receives its correct topology cap and contributes zero increments to the conflict / safe-default counters.** Proves blast-radius containment — the misconfig affects only GPU 0's *value*, not the other GPUs' cap writes. | 2 |
| B6 | (DCGM, #9790) After `_ensure_identity_map()` runs, `actuator._dcgm_uuid_by_idx` is an 8-entry list whose ordering matches DCGM's gpuId enumeration, and `actuator._nvml_index_by_uuid` is an 8-entry dict whose values match the (deliberately different) NVML enumeration order. Behavioural assertion: each `list_running_pids(k)` for `k ∈ 0..7` reads from `pynvml.nvmlDeviceGetHandleByIndex(_nvml_index_by_uuid[_dcgm_uuid_by_idx[k]])` — i.e., routes to the physical UUID DCGM index `k` belongs to, not to NVML index `k`. (Note: `_ensure_identity_map` mutates state and returns `None` — assert via the maps + via `list_running_pids` behavior, not via a return value.) | 2 |
| B7 | (DCGM, #9790) `dcgmConfigSet` raises `DCGM_ST_CONNECTION_NOT_VALID` on 2 of 8 GPUs mid-reconcile (hostengine flap). `_with_reconnect` catches it once, rebuilds the handle (clears `_groups`, `_dcgm_uuid_by_idx`, `_nvml_index_by_uuid`), and retries. Final state: 8 caps live, **`apply_failures_total == 0`, `dcgm_enforce_failures_total == 0`** — a successful reconnect leaves both counters at zero. | 2 |
| B8 | (DCGM, #9790; companion to B7) `dcgmConfigSet` succeeds on all 8 GPUs, but `dcgmConfigEnforce` raises on 2 of them. Set-OK-Enforce-fail path: caps are still live and tracked in managed state; **`apply_failures_total == 0`, `dcgm_enforce_failures_total == 2`**. Only the auto-reapply-after-reset target-config registration is missing on those two GPUs. | 2 |

### C. Cross-component (planner ↔ agent ↔ DCGM)

| ID | Assertion | Tier |
|---|---|---|
| C1 | Planner-V flips an annotation from 450W→460W. Within one agent reconcile tick (15 s), `apply_cap(0, 460)` fires. Annotations from DGD-T and DGD-S are unchanged. | 3 (live) |
| C2 | SIGTERM on the agent restores all 8 GPUs to default TDP through `_active_actuator.restore_default(gpu_idx)`, regardless of which DGD owned them. (Existing `test_shutdown.py` covers a single-GPU case; we extend to the 8-GPU mixed-framework case.) | 1 (unit, mocked actuator) |
| C3 | Orphan recovery on agent restart: `managed_gpus.json` contains 8 UUIDs from the prior session. **Per-UUID contract** (`_restore_orphaned_gpus_on_startup`): for each GPU index, the agent reads `actuator.get_uuid(gpu_idx)`, checks UUID-in-`_previously_managed`, then checks `actuator.list_running_pids(gpu_idx) == []`. If both gates pass, restore default. Test scenarios: (a) all 5 pods still running → 8 GPUs have non-empty PID lists → zero restorations. (b) DGD-T entirely down (T-P, T-D gone), V + S still running → GPUs 2-5 idle and restored, GPUs 0, 1, 6, 7 left alone. (c) all 5 pods gone → all 8 GPUs restored. UUID-keying never trips on framework affinity. *(Asserting per-pod count is ambiguous — "1 pod per framework" picks any 3 from {V-P, V-D, T-P, T-D, S-A} with one-per-DGD and covers exactly 1+2+2 = 5 GPUs, not 6.)* | 2 |
| C4 | Three planners run for 5 reconcile ticks while traffic varies; total `apply_cap` calls = `5 × 8 = 40` (steady state — no cross-DGD chatter triggering extra patches). | 3 (live) |

### D. Observability invariants

| ID | Assertion | Tier |
|---|---|---|
| D1 | After reconcile: `dynamo_power_agent_applied_limit_watts{gpu=0..7}` samples match the topology table per GPU index (4 distinct watt values across 8 samples). | 2 |
| D2 | `dynamo_planner_power_projected_watts` from each of the 3 planner instances equals that DGD's own `N_p × cap_p × gpu_per_p + N_d × cap_d × gpu_per_d` — three independent gauges on three separate Prom ports (one per planner pod). | 2 |
| D3 | In the well-formed reconcile: `multi_pod_gpu_total`, `safe_default_applied_total`, `apply_failures_total`, `dcgm_enforce_failures_total`, `cap_clamped_total` are all unchanged from their baseline. (We currently have no "happy-path zero counters" assertion anywhere in the suite.) | 2 |

### E. Failure modes specific to this topology

| ID | Assertion | Tier |
|---|---|---|
| E1 | Planner-T crashes (Python exception in its `on_tick`). The agent's next reconcile still applies V's and S's caps. DGD-T's pods keep whatever cap they already have. | 3 |
| E2 | Pod V-P restarts (new UID, same name, same node, GPU 0 reassigned). Within one tick the new UID's annotation drives `apply_cap(0, 450)`. **Pod T-P's cap stays at 480 W throughout V-P's restart window** — every tick still produces `apply_cap(2, 480)` and `apply_cap(3, 480)` (the reconcile loop iterates all GPUs every tick), but the value never drifts to safe-default or to V-P's cap. The right invariant is "T-P's cap *value* is stable," not "T-P's GPUs see zero NVML calls" (they see one per tick by design). | 3 |
| E3 | `nvidia-dcgm` hostengine restart while reconcile is in flight: `_with_reconnect` recovers; 8 caps end up live; no GPU briefly goes to safe-default. | 3 (live, opt-in) |

---

## 4. Tier definitions and where each test lives

| Tier | Substrate | Cost | New file |
|---|---|---|---|
| **1 — unit** | In-process Python; mock `kubernetes.client.CoreV1Api` + `MockActuator`; no GPU; no real K8s | Sub-second per test; runs in default `pytest` | `components/power_agent/tests/test_multi_dgd_topology.py` (on #9790; happy-path subset back-portable to #9682) and `components/src/dynamo/planner/tests/unit/test_multi_dgd_annotations.py` (on #9683) |
| **2 — pseudo-integration** | Single-process; `FakeNode` fixture (8 GPU handles, 5 pods, 7+ PIDs); patch `pynvml`, `pydcgm`, `kubernetes.client` together; drive `PowerAgent.reconcile_once()` (agent side) and `NativePlannerBase._apply_power_annotations()` (planner side — the targeted async method for the TGP knob; there is no public `_run_one_tick` callable in #9683 — the per-tick body lives inside `async def run()`'s while-loop) | A few seconds per test; runs in default `pytest` | `components/power_agent/tests/test_node_reconcile_e2e.py` (scaled-up mirror of `test_reconcile_wiring.py`); `components/power_agent/tests/test_actuator_eight_gpu_parity.py` |
| **3 — live (opt-in)** | Real 8-GPU node (`umb-b200-*`), real `nvidia-dcgm`, real kubelet; gated on `RUN_MULTI_DGD_TESTS=1` | Minutes per scenario; only runs when an operator schedules it on a B200 node | `components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py`; `components/power_agent/tests/e2e_actuator_parity.py` extended with the 8-GPU + 3-framework scenario |

**Fixture: `FakeNode` (Tier 2).** Single Python module that constructs:

- 8 fake NVML handles with distinct UUIDs and configurable enumeration order.
- A fake DCGM hostengine that returns matching UUIDs but in a **different**
  gpuId order (forces `_ensure_identity_map` to actually do work).
- 5 fake `V1Pod` objects with realistic annotations + 3 different
  `nvidia.com/dynamo-graph-deployment-name` label values + matching
  `nvidia.com/dynamo-component=<service-key>` labels + 3 different namespaces.
- A fake `/proc/{pid}/cgroup` (via `mock_open`) per PID, including TP-rank
  helper PIDs that should be deduped by UID.
- A fake K8s API that filters by `field_selector=spec.nodeName=...` and
  `label_selector=<exact string>`.
- (For multi-planner tests) a `make_bare_planner(dgd_name)` helper that
  constructs planner instances via `object.__new__(NativePlannerBase)` and
  binds `planner.prometheus_metrics = MagicMock()`. **Do NOT instantiate
  real `PlannerPrometheusMetrics` per planner** — its `__init__` registers
  on the process-wide `prometheus_client.REGISTRY` and the second instance
  raises `Duplicated timeseries in CollectorRegistry`. Distinct
  `prometheus_port` values do not solve this; the registry is process-global.

With that fixture, B1–B8, C2, C3, D1, D3 are 20–40 line tests each.

---

## 5. Concrete Tier-1 test skeletons

These are the four that plug the immediate coverage holes. They land on
**different branches** — see §7.

### 5.1 Power-agent side — `components/power_agent/tests/test_multi_dgd_topology.py`

Targets **#9790** (`pr1b/power-agent-dcgm-actuator`) because the dedup test
requires commit `ba3b5803a9`. The happy-path-only subset is back-portable to
**#9682** (`pr1a/power-agent`).

```python
class TestMultiFrameworkReconcileHappyPath:
    """8-GPU node hosting vLLM disagg + TRTLLM disagg + SGLang aggr.

    Asserts:
      - 8 apply_cap calls matching the topology table (4 distinct watt values)
      - multi_pod_gpu_total == 0 for both 'agree' and 'conflict' dispositions
      - safe_default_applied_total == 0
      - apply_failures_total == 0
    """
    def test_eight_cap_calls_match_topology_table(self): ...

    # Requires UID-dedup in _reconcile_gpu — back-port-blocked from #9682.
    def test_tp_rank_helper_pids_deduped_by_uid(self):
        """TRTLLM tp=2 prefill has 2 PIDs on each of GPU 2 + GPU 3
        (rank main + helper). Each GPU still resolves to one pod-uid;
        multi_pod_gpu_total{disposition=agree} stays at 0."""
        ...
```

### 5.2 Planner side — `components/src/dynamo/planner/tests/unit/test_multi_dgd_annotations.py`

Targets **#9683** (`pr1b/planner-infra`).

```python
class TestThreePlannersOneCluster:
    async def test_label_selectors_isolate_each_dgd(self, mock_kube_api):
        """Three planners (DGD-V, DGD-T, DGD-S) with different graph-deployment-name
        values produce three disjoint patch_pod_annotation call sets. The exact
        selector string for each DGD is pinned:

            nvidia.com/dynamo-graph-deployment-name=<dgd>,
            nvidia.com/dynamo-component=<service-key>

        Asserts no pod-name from another DGD ever appears in this planner's
        patch_namespaced_pod calls — even when the mock returns colliding pod
        names, the planner must scope by selector at the get_component_pods
        layer."""
        ...

    async def test_aggregated_dgd_patches_decode_branch_only(self, mock_kube_api):
        """SGLang AggPlanner has require_prefill=False, require_decode=True.
        _apply_power_annotations must:
          - NOT call get_component_pods(PREFILL)
          - call get_component_pods(DECODE) once
          - PATCH the returned aggregated worker pod with
            decode_engine_gpu_power_limit (= 500W in the canonical topology)."""
        ...
```

Both files are ~150–200 LOC. The power-agent file reuses the existing
`_FakeMetrics` from `test_multi_pod_policy.py`; the planner file reuses the
existing `mock_kube_api` fixture from `test_actuation_knobs.py`.

---

## 6. Explicitly out of scope (keeps the plan Phase 1)

- **AIC closed-loop interaction** — Phase 3 / PR #9684+. Mixed topology only
  matters for AIC once `c_power_p` / `c_power_d` start updating per-component.
- **MIG / MPS shared-GPU multi-DGD** — design doc §3.3 declares this v2
  (last-writer-wins on annotation race). B5 covers only the *detection counter*
  for this misconfig, not recovery.
- **Budget enforcement (`_apply_power_budget`)** — Phase 2 / PR #9684. Cap
  values in this plan come from static config; verifying feasibility against
  `total_gpu_power_limit` is Phase 2's job.
- **Cross-node coordination** — every DGD in scope is single-node by
  construction. Multi-node DGDs (e.g., one TRTLLM disagg with prefill on node A,
  decode on node B) are orthogonal.

---

## 7. Recommended landing sequence

1. **Follow-up on `pr1a/power-agent` (#9682):** land the **happy-path subset**
   of `test_multi_dgd_topology.py` — just `test_eight_cap_calls_match_topology_table`
   (re-targeted at the #9682 surface) and the zero-counters assertion. **Do
   NOT include the dedup test** — the dedup fix isn't on this branch.
   **#9682 surface re-target:** #9682 has no `actuator.py`, so the assertion
   shape changes from `actuator.apply_cap.call_args_list` to either
   `pynvml.nvmlDeviceSetPowerManagementLimit.call_args_list` (with `pynvml`
   patched at the `power_agent` module level) or `power_agent._apply_cap`
   (patched as a module function). The PIDs-per-GPU map and the topology
   table stay the same; only the actuation-call sink differs. ~80 LOC.
   Concrete answer to the reviewer's question that doesn't depend on #9790
   landing first.

2. **Follow-up on `pr1b/planner-infra` (#9683):** land
   `test_multi_dgd_annotations.py` (A1 + A2 + A3 + A4). ~200 LOC, no new
   product code expected — but the test exercise itself will surface whether
   `_apply_power_annotations` is robust against the cross-DGD contamination
   scenario in A4.

3. **Follow-up on `pr1b/power-agent-dcgm-actuator` (#9790):** land the **full**
   `test_multi_dgd_topology.py` (incl. dedup test from §5.1) plus the
   DCGM-specific B6 / B7 / B8 cases as either pure unit (mocked `pydcgm`) or
   as additions to `e2e_actuator_parity.py` (Tier 2). Counters semantics for
   B7 vs B8 must be split into two distinct tests — `_with_reconnect`
   recovery leaves both counters at zero; only Enforce-fail-after-Set ticks
   `dcgm_enforce_failures_total`.

4. **Cluster validation playbook** in `docs/components/planner/dpp-dev-env.md`:
   add a "Multi-DGD smoke" section documenting the manual `umb-b200-N`
   procedure for Tier-3 (C1, C4, E1–E3). We don't land Tier-3 into CI — it's
   a release-gate playbook.

---

## 8. Tier 3 live test on AKS `dpp-dev-env` — concrete realization

This section turns Tier 3 from "manual playbook in a doc" into an executable
test backed by real K8s objects on the AKS `dpp-dev-env` cluster.

> **2026-05-21 live-run record:** Phases 2 (NVML) + 3 (DCGM, agent-half)
> were executed end-to-end on `aks-a100b-22138447-vmss000000`. Results
> + 10 PR findings discovered during the run + the exact patches kept
> in tree to enable re-pro are written up in
> [`examples/multi-dgd-live-test/README.md`](../../examples/multi-dgd-live-test/README.md)
> under "2026-05-21 live-run record on dpp-dev-env AKS". Subsequent
> reviewers updating this design doc should keep that README's
> `Findings #1..N` numbering stable when adding new ones.

### 8.1 Cluster ground truth (verified May 21, 2026)

Diagnostic queries (`kubectl get nodes -o wide`, `kubectl get ds -A`,
`kubectl get dynamographdeployments -A`, `kubectl get pods -A -o
jsonpath={...}.containers[*].image`) returned the following facts. Every
decision in §8.2–§8.7 traces back to one of these.

| Fact | Verified value | Source of constraint |
|---|---|---|
| 8-GPU nodes | 3× A100 SXM4 (`aks-a100a-vmss000002`, `…000003`, `aks-a100b-vmss000000`) | Pins per-GPU max TDP to **400 W** (§8.2 cap math) |
| Power Agent DaemonSet | **Not present** on the cluster | Setup runbook §8.3.B must `helm install` `deploy/helm/charts/power-agent` |
| `nvidia-dcgm` (hostengine) | **Not present.** Only `nvidia-dcgm-exporter` (embedded DCGM) is deployed. | DCGM live pass §8.6 must deploy a standalone hostengine DS OR have a cluster admin flip GPU Operator `dcgm.enabled=true` |
| Dynamo CRDs | All present (`dynamographdeployments.nvidia.com`, etc.) | DGD objects from §8.4 can be applied without CRD install |
| Operator in `kaim-dynamo-system` | **Not present** | Setup runbook §8.3.A must install `kubernetes-operator` chart in that namespace |
| Existing TRTLLM / SGLang DGDs | **None anywhere on the cluster** | TRTLLM + SGLang runtime images unverified — live test scopes to **VLLM-only** (Tier 1/2 retains the multi-framework topology in §2) |
| Pull secret in `kaim-dynamo-system` | `nvcr-imagepullsecret` covers `nvcr.io` only | DGDs in §8.4 pin to `nvcr.io/nvidia/ai-dynamo/vllm-runtime` tags — no `nvstaging` or `dynamoci.azurecr.io` |
| Operator tag with recent pull-success on cluster | `nvcr.io/nvidia/ai-dynamo/kubernetes-operator:1.2.0` (running in `ycha-snapshot-poc`) | Setup runbook pins this exact tag |
| Power Agent helm chart present in repo | `deploy/helm/charts/power-agent/{Chart.yaml,values.yaml,templates/*}` | NVML default actuator; `--set agent.actuator=dcgm` flips to DCGM; A100 SXM4 safe-default = **280 W** (chart values.yaml comment) |

### 8.2 Live topology — VLLM-only on a single 8-GPU A100 node

| DGD | Planner type | Worker pods on node | GPUs | Cap (W) | Annotation path exercised |
|---|---|---|---|---|---|
| `vllm-dgd-A` (disagg TP=1) | `DisaggPlanner` | prefill (1 GPU) + decode (1 GPU) | 0, 1 | prefill=350, decode=325 | `_apply_power_annotations` PREFILL **and** DECODE branches; `prefill_engine_gpu_power_limit` + `decode_engine_gpu_power_limit` |
| `vllm-dgd-B` (disagg TP=2) | `DisaggPlanner` | prefill (TP=2, 1 pod → 2 GPUs) + decode (TP=2, 1 pod → 2 GPUs) | 2, 3 (prefill); 4, 5 (decode) | prefill=375, decode=325 | Multi-GPU-per-pod path — one annotation on a pod with 2 PIDs (`actuator.list_running_pids` returns ≥2) |
| `vllm-dgd-C` (agg TP=2) | `AggPlanner` | decode-only (TP=2, 1 pod → 2 GPUs) | 6, 7 | 300 (decode-side knob) | A3 case live: `AggPlanner` uses `decode_engine_gpu_power_limit` only; PREFILL branch must not fire |

**A100 SXM4 cap math** (max TDP 400 W):

| Role | Cap | % of TDP | Notes |
|---|---|---|---|
| Safe default (Power Agent fallback) | 280 W | 70 % | Chart-recommended A100 SXM (80 GB) value |
| DGD-C decode-only (lighter work) | 300 W | 75 % | |
| DGD-A decode, DGD-B decode | 325 W | 81.25 % | Same value on 3 GPUs (gpu 1, 4, 5) |
| DGD-A prefill | 350 W | 87.5 % | |
| DGD-B prefill | 375 W | 93.75 % | Heaviest live cap |

Distinct annotation values across 8 GPUs = **4** (300, 325, 350, 375) — same
structural property as the Tier 1/2 B200 topology (§2), so the planner-side
and agent-side assertions about "4 distinct watt values" port across.

**Why VLLM-only is sufficient for the *planner-side* invariants this PR ships.**
The cross-DGD label-selector isolation (#A1, #A2), the agg-decode-branch-only
path (#A3), and the external-overwrite recovery (#A4) are all properties of
`_apply_power_annotations` in `core/base.py`, which is **framework-blind**
(it operates on `nvidia.com/dynamo-component=<service-key>` labels, not on
runtime image identity). Whether the worker container is a VLLM or TRTLLM
binary changes nothing about the planner's label-selector semantics. The
*agent-side* multi-GPU-per-pod path (#B3, TP-rank PID dedup) is exercised by
DGD-B's TP=2 pods — `vllm-runtime` with TP=2 produces ≥2 host PIDs per pod
just as TRTLLM TP=2 does.

The **only** invariant that VLLM-only cannot exercise live is "three
*different* frameworks coexisting on one node." That's a packaging concern
(can three runtime images schedule on the same kubelet, with three sets of
container-toolkit hooks?), not a Phase-1 contract concern. Tier 2 covers it
with `FakeNode`.

### 8.3 Setup runbook (one-time, ~10 min)

All steps run from a workstation with `kubectl` already pointing at
`dpp-dev-env` (`$KUBECONFIG = ~/.kube/dynamo-kubeconfig`).

**A. Install Dynamo operator in `kaim-dynamo-system`.**

The cluster has the CRDs installed (verified §8.1) but the operator
controller-manager isn't running in this namespace, so DGD objects there
won't reconcile until we install one.

```bash
# Pin the same tag already proven pullable on this cluster
helm upgrade --install dynamo-platform \
    deploy/helm/charts/platform \
    --namespace kaim-dynamo-system \
    --create-namespace \
    --set dynamo-operator.controllerManager.manager.image.tag=1.2.0 \
    --set dynamo-operator.namespaceRestriction.enabled=true \
    --set dynamo-operator.namespaceRestriction.targetNamespace=kaim-dynamo-system

# Wait for controller-manager
kubectl wait deploy -n kaim-dynamo-system \
    dynamo-platform-dynamo-operator-controller-manager \
    --for=condition=Available --timeout=180s
```

**B. Install Power Agent DaemonSet (NVML mode first).**

```bash
# Pull-secret already exists in namespace (verified §8.1): nvcr-imagepullsecret
helm upgrade --install power-agent \
    deploy/helm/charts/power-agent \
    --namespace kaim-dynamo-system \
    --set image.repository=nvcr.io/nvidia/dynamo/power-agent \
    --set image.tag=v1.2.0 \
    --set "imagePullSecrets[0].name=nvcr-imagepullsecret" \
    --set agent.actuator=nvml \
    --set agent.safeDefaultWatts=280 \
    --set agent.prometheusPort=9100 \
    --set rbac.namespaceRestricted=true

kubectl rollout status ds -n kaim-dynamo-system power-agent --timeout=180s
```

Verification: `kubectl get ds -n kaim-dynamo-system power-agent` shows
DESIRED == CURRENT == READY == (# of A100 nodes the GPU operator labels
with `nvidia.com/gpu.present=true`).

**C. Pin to a single test node.** The live test exercises one 8-GPU node at
a time. Pick an idle one and label it; the DGDs in §8.4 use `nodeSelector:
power-test/node: kaim` to land all 5 worker pods on the same kubelet
(DGD-A prefill + decode, DGD-B prefill + decode, DGD-C decode = 5 pods on 8 GPUs).

```bash
# Pick whichever A100 node currently has zero existing dynamo workloads
TEST_NODE=aks-a100a-36888584-vmss000002
kubectl label node $TEST_NODE power-test/node=kaim --overwrite
```

**D. (Optional, only for §8.6 DCGM pass) Deploy a standalone hostengine.**

The cluster's GPU Operator runs with `dcgm.enabled=false` (verified §8.1).
The chart's default `agent.dcgm.host = nvidia-dcgm.gpu-operator.svc...`
won't resolve. Two options, pick one:

- **D.1 (cluster-admin coordination):** flip `dcgm.enabled=true` in the
  GPU Operator's `ClusterPolicy`. This adds `nvidia-dcgm` DS to
  `gpu-operator` namespace, exposes hostengine on 5555. Cluster-wide impact;
  needs cluster admin.
- **D.2 (namespace-local DS, self-service):** apply
  `examples/multi-dgd-live-test/20-nvidia-dcgm-standalone-ds.yaml` (see
  §8.4) which creates a privileged DS in `kaim-dynamo-system` named
  `nvidia-dcgm-standalone` running `nv-hostengine -b 0.0.0.0 -p 5555` +
  a headless Service exposed at
  `nvidia-dcgm-standalone.kaim-dynamo-system.svc.cluster.local`. Then
  re-install Power Agent with
  `--set agent.actuator=dcgm
   --set agent.dcgm.host=nvidia-dcgm-standalone.kaim-dynamo-system.svc.cluster.local
   --set agent.dcgm.enforce=true`. Restricted to our namespace; no
  cluster-wide effect.

We choose **D.2** for the test design (no cross-team coordination on the
PR's critical path), and document D.1 as the production-equivalent.

### 8.4 Artifacts: where the YAMLs live

DGD specs + setup manifests live in `examples/multi-dgd-live-test/`,
introduced by PR #9683 alongside the live test that consumes them
(`components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py`):

```
examples/multi-dgd-live-test/
├── README.md                          # apply order + recovery
├── 00-namespace.yaml                  # kaim-dynamo-system + pull-secret reference
├── 10-power-agent-values-nvml.yaml    # helm values override for §8.3.B
├── 11-power-agent-values-dcgm.yaml    # helm values override for §8.3.D.2 retest
├── 20-nvidia-dcgm-standalone-ds.yaml  # §8.3.D.2 alternative to cluster-wide flip
├── 30-vllm-dgd-A.yaml                 # disagg TP=1 + cap annotations
├── 31-vllm-dgd-B.yaml                 # disagg TP=2
├── 32-vllm-dgd-C.yaml                 # agg TP=2 (decode-only)
└── 99-teardown.sh                     # idempotent cleanup
```

Each DGD pins:

- **(operator-injected, not in our YAML)** Worker pods generated by the
  Dynamo operator carry `nvidia.com/dynamo-graph-deployment-name=<dgd-name>`
  and `nvidia.com/dynamo-component-type=worker` /
  `nvidia.com/dynamo-sub-component-type=<prefill|decode>` labels — the
  planner uses these to construct the #A1 selector. We rely on the operator
  to stamp them; the DGD CR's own `metadata.labels` does not need to carry
  the `graph-deployment-name` label (existence in pod labels is the
  product contract).
- `spec.services.<WorkerKind>.subComponentType`: `prefill` / `decode` —
  drives the operator's `sub-component-type` pod label (#A1, #A3).
- `spec.services.<WorkerKind>.resources.limits.gpu`: 1 (TP=1) or 2 (TP=2).
- `spec.services.<WorkerKind>.extraPodSpec.nodeSelector`:
  `power-test/node: kaim` — pins all worker pods to a single node.
- `spec.services.Planner.extraPodSpec.mainContainer.args[--config]`
  includes:
  - `"mode": "disagg"` for A and B; `"mode": "agg"` for C.
    **Required** — PlannerConfig.mode has no implicit "agg" path
    (core/__main__.py:40-51 raises ValueError on missing/unknown mode).
  - `"enable_power_awareness": true`
  - `"prefill_engine_gpu_power_limit"` / `"decode_engine_gpu_power_limit"`
    per §8.2 cap table

### 8.5 NVML actuator live pass — `test_multi_dgd_live.py` (phase 2, NVML mode)

**Pre-conditions:** §8.3.A + §8.3.B + §8.3.C done. `power-agent` DS
healthy with `agent.actuator=nvml`. 5 worker pods Ready, all on
`$TEST_NODE` (A prefill + A decode + B prefill + B decode + C decode).

**Assertions** (in order, each with a kubectl-equivalent shown to make
hand-debugging easy):

1. **Planner-side annotations land** (A1/A2 live). For each of A/B/C, the
   planner pod logs `Annotated pod <pod-name> with
   dynamo.nvidia.com/gpu-power-limit=<W>` on every worker pod it owns
   (exact format from `core/base.py::_apply_power_annotations`:
   `logger.info("Annotated pod %s with %s=%s", ...)`), and
   `kubectl get pod -n kaim-dynamo-system <worker> -o
   jsonpath='{.metadata.annotations.dynamo\.nvidia\.com/gpu-power-limit}'`
   returns the §8.2 value. Pod count = 5 (A-prefill, A-decode, B-prefill,
   B-decode, C-decode). Distinct annotation values across the 5 pods = 4
   (325 appears on both A-decode and B-decode).

   *Why this beats Tier 1/2:* the planner's selector is wired to the
   operator's real label injection here, not a mock. If anything in
   `core/base.py` or `connectors/kubernetes.py` drifts from what the
   operator stamps, this catches it.

2. **AggPlanner decode-only path live** (A3). On planner-C, `kubectl
   logs deploy/<planner-C> | grep PREFILL` returns no `get_component_pods`
   calls (logged at DEBUG); only DECODE.

3. **No cross-DGD annotation contamination** (A2 live). For each DGD's
   planner pod, the set of pod-names it ever PATCHes (read from pod logs)
   is exactly the set returned by its own `get_component_pods(...)` —
   never any pod from another DGD. Asserts the §2 invariant #5 (RBAC +
   correct selectors, not the K8s API).

4. **Power Agent applies all 8 caps** (B2 live, NVML surface). Read
   Prometheus from `power-agent` pod on `$TEST_NODE`:

   ```bash
   POD=$(kubectl get pod -n kaim-dynamo-system -l app.kubernetes.io/component=power-agent \
         --field-selector spec.nodeName=$TEST_NODE -o name)
   kubectl exec -n kaim-dynamo-system $POD -- curl -s localhost:9100/metrics \
       | grep dynamo_power_agent_applied_limit_watts
   ```

   Expect 8 samples with `(gpu=0..7, watts=350,325,375,375,325,325,300,300)`.

5. **Zero-counter invariant** (B4, D3 live). Same metrics endpoint:
   `multi_pod_gpu_total{disposition="agree|conflict"}`,
   `safe_default_applied_total`, `apply_failures_total`,
   `dcgm_enforce_failures_total`, `cap_clamped_total` all equal their
   baseline samples taken **before** the DGDs were applied. Any non-zero
   delta is a regression.

6. **NVML actually changed the cap on the GPU** (cross-component live —
   the one assertion that mocked tests fundamentally cannot make). Exec
   into the power-agent pod and read NVML directly:

   ```bash
   kubectl exec -n kaim-dynamo-system $POD -- python -c \
     "import pynvml; pynvml.nvmlInit(); \
      print([(i, pynvml.nvmlDeviceGetPowerManagementLimit(pynvml.nvmlDeviceGetHandleByIndex(i))/1000) for i in range(8)])"
   ```

   Expected output equals the §8.2 cap column (in mW → W). This closes
   the loop end-to-end: planner annotation → agent reconcile → NVML
   `nvmlDeviceSetPowerManagementLimit` → driver applied. **The Tier 1/2
   tests cannot make this assertion** — both mock NVML at the pynvml
   layer.

7. **Conflict scenario (multi-pod policy live, B5) — deferred on AKS CDI.**
   The intended live assertion is to add a second CUDA process on GPU 0
   with a conflicting `dynamo.nvidia.com/gpu-power-limit=200` annotation,
   then verify the agent switches only that GPU to the safe default.
   The original bystander-pod shape used `NVIDIA_VISIBLE_DEVICES=GPU-<uuid>`,
   but dpp-dev-env AKS runs the NVIDIA runtime in CDI device mode and
   rejects that form before the pod is admitted. The test class is
   therefore skipped today with an explicit Finding #7 reason. Re-enable
   it only after rewriting the bystander to request `nvidia.com/gpu: 1`
   through the device plugin and discover the assigned UUID/PID at runtime.

   ```bash
   pytest ... -k TestMultiPodConflict
   # SKIPPED today:
   # Finding #7: AKS CDI device mode rejects NVIDIA_VISIBLE_DEVICES=<uuid>.
   ```

   Once the CDI-safe rewrite lands, the expected invariant is: within 30 s
   (two agent reconcile ticks), `multi_pod_gpu_total{disposition="conflict"}`
   delta `>= 1`, `safe_default_applied_total` delta `>= 1`, GPU 0's
   `applied_limit_watts == 280` (the safe default), and GPUs 1-7 unchanged.

   *Why `>= 1`, not `== 1`:* both counters are cumulative `_total`
   counters that tick once per reconcile detection. The test waits at
   least 2 reconcile intervals after the bystander's CUDA context
   appears (so the agent gets a chance to observe + react), and each
   tick that sees the still-running conflict increments the counter
   again. The robust invariant is "delta ≥ 1" not "exactly 1" —
   asserting `== 1` would be flaky on a 15s reconcile interval where
   the test naturally waits 30+ s.

   Cleanup deletes the bystander pod; next tick restores GPU 0 to its
   topology cap.

### 8.6 DCGM actuator live pass — `test_multi_dgd_live.py` (phase 3, DCGM mode)

**Pre-conditions:** all of §8.5 plus §8.3.D.2 done (standalone hostengine
DS healthy). Power Agent re-installed with
`--set agent.actuator=dcgm --set agent.dcgm.host=...kaim-dynamo-system...
--set agent.dcgm.enforce=true`.

**Assertions** that differ from §8.5:

1. Reuses the active §8.5 assertions with the same Path-A/Path-B gates:
   planner-log/AggPlanner assertions require Path A; agent-side metrics
   and NVML-readback assertions run on Path A or Path B; the conflict
   class remains skipped until Finding #7 is fixed.
2. **Assertion 6 surface changes to DCGM** (B6/B8 live):

   ```bash
   # On the same node, query DCGM hostengine directly, NOT NVML.
   # DCGM 4.x does not accept `-g all -j` for config --get, so enumerate
   # the Power-Agent-created per-GPU groups and query each one.
   kubectl exec -n kaim-dynamo-system <hostengine-pod> -- \
       dcgmi group --list -j
   kubectl exec -n kaim-dynamo-system <hostengine-pod> -- \
       dcgmi config --get -g <dynamo-power-agent-gpu-N group-id> -j
   ```

   Expected per-GPU `Power Limit (W)` matches §8.2. *Why this is a
   different test from §8.5.6:* DCGM's `dcgmConfigSet` writes go through
   a different driver path than NVML's `nvmlDeviceSetPowerManagementLimit`
   — the live observation must come from DCGM's view of the cap, not
   NVML's. (NVML and DCGM are *expected* to agree, and they will; but
   the test exists because PR #9790 changes which library issues the
   write.)
3. **`dcgmConfigEnforce` registers target config** (B8 live, success
   path). After 1 reconcile cycle with `agent.dcgm.enforce=true`:
   `dcgm_enforce_failures_total == 0` AND each per-GPU group's
   `dcgmi config --get -g <id> -j` payload has `Power Limit (W).Target`
   matching the topology cap. DCGM 4.x removed the old
   `dcgmi config --get-target` path, so the test reads `Target` from
   the same per-group `--get` payload it uses for `Current`. The
   auto-reapply contract is testable by `dcgmi diag` triggering a GPU
   reset and asserting the cap returns within DCGM's reset window —
   defer to E3 (release-gate playbook only).

### 8.7 Execution plan — two passes against one test file

The test file contains four test classes covering both actuator paths
in one module:

| Class | Tests | Phase 2 (NVML) | Phase 3 (DCGM) |
|---|---|---|---|
| `TestPerDgdAnnotation` | 3 — annotation convergence, AggPlanner decode-only, planner-log isolation | Path A: runs all 3; Path B: annotation-presence runs, 2 planner-only tests skip | same |
| `TestPowerAgentReconcile` | 2 — 8-cap topology match, happy-path zero counters | runs | runs |
| `TestCapApplicationProof::test_nvml_per_gpu_cap_matches_topology` | 1 | runs (uniquely proves the NVML write path) | runs (no longer unique — NVML view here reflects what DCGM wrote) |
| `TestCapApplicationProof::test_dcgm_per_gpu_cap_matches_topology` | 1 | **skipped** via `DCGM_HOSTENGINE_AVAILABLE` gate | runs (uniquely proves the DCGM write path) |
| `TestCapApplicationProof::test_dcgm_target_config_registered` | 1 | **skipped** via same gate | runs (proves enforce=true registers target config) |
| `TestMultiPodConflict` | 1 — bystander conflict + recovery | **skipped** until CDI-safe bystander rewrite lands (Finding #7) | **skipped** for same reason |

**Gating mechanics:**
- Module-level: `RUN_MULTI_DGD_LIVE=1` opts the file in. Without it,
  `pytest.skip(allow_module_level=True)` skips the entire module so the
  test file is inert in normal CI.
- `TEST_NODE`: required to pin the test to one 8-GPU node. The fixture
  asserts exactly one Power Agent pod on that node.
- Path detection: tests that strictly need live planner pods use
  `@pytest.mark.skipif(not _PATH_A_AVAILABLE, ...)`. On Path B
  (stub-workers), annotation-presence still runs but the AggPlanner
  branch proof and planner-log isolation tests skip with explicit
  reasons.
- `DCGM_HOSTENGINE_AVAILABLE=1`: gates **only** the DCGM-specific tests
  via `@pytest.mark.skipif`. Phase 2 leaves the env var unset (DCGM
  tests skip), phase 3 sets it (DCGM tests run) and the README
  explicitly reinstalls the Power Agent in DCGM mode before phase 3.
- `TestMultiPodConflict` is class-skipped until the bystander is
  rewritten for AKS CDI mode (Finding #7). It is not counted as an
  expected pass in either actuator phase today.

CI wiring: the runner sets `DCGM_HOSTENGINE_AVAILABLE` based on whether
it deployed the §8.3.D.2 standalone hostengine DS. A "smoke" runner
that only does phase 2 leaves the var unset; a "full" runner that does
both phases sets the var only on the second pytest invocation.

The live test file lives at
`components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py`,
shipped by PR #9683 alongside this design doc.

### 8.8 Relationship to the existing single-DGD live suite

PR #9683 already ships
`components/src/dynamo/planner/tests/integration/test_actuation_knobs_live.py`
(~567 LOC) which runs inside the `planner-dev` pod and covers:

- `TestTgpAnnotationRoundTrip` — single TGP annotation lands on one real pod
- `TestPodDiscovery` — label selectors resolve real pods (one DGD)
- `TestApplyPowerAnnotationsLive` — `_apply_power_annotations` orchestration on one DGD
- `TestPostBusyThresholdLive`, `TestScalingAdvisoryMode`, `TestScalingRealMutation`

**These all run against exactly one DGD.** The `_DGD_NAME` env var pins a
single DGD; the pod-discovery selector reads `_DGD_NAME` once. The
multi-DGD live test in §8.5 / §8.6 is the natural extension: it leaves the
single-DGD tests alone (they keep validating per-DGD mechanics) and adds
the cross-DGD invariants (A2 isolation, B2 / B4 agent-side aggregation,
NVML and DCGM end-to-end actuation). B5 conflict coverage is designed
here but deferred on AKS until the CDI-safe bystander rewrite lands.
The two suites should
co-exist on `pr1b/planner-infra` once §8 lands — the single-DGD tests run
on every PR, the multi-DGD test runs only when `RUN_MULTI_DGD_LIVE=1`
because it requires 8 GPUs and the §8.3 setup.

### 8.9 What §8 does NOT cover

- **H100 / B200 SKU coverage.** Only A100 is available on this cluster.
  When H100/B200 nodes return to the fleet, the same test file should
  parametrize over `hw_sku` and re-run with the SKU-keyed cap table.
- **TRTLLM / SGLang multi-framework live coverage.** Stays in Tier 2
  (FakeNode). Will become testable once those runtime images are proven
  pullable in `kaim-dynamo-system` (need an nvstaging-scoped pull-secret,
  or a public mirror of the runtime image).
- **MIG.** Out of scope per §6 (Phase 2+).

---

## Changelog

### v0.8 (this revision)

Three hardening fixes from the follow-up review after the 2026-05-21
live run.

1. **Path B expectation mismatch.** The test file now skips the two
   planner-only assertions on stub-worker Path B, but the generic runbook
   still read as if all `TestPerDgdAnnotation` cases and the conflict test
   passed in the NVML phase. Aligned the README expectations with the
   actual skip behavior and clarified the test-file docstring's Path A vs
   Path B preconditions.
2. **DCGM 4.x CLI contract.** §8.6 still described the stale
   `dcgmi config --get -g all` / `dcgmi config --get-target -g all`
   forms. Updated the operative text to match the rewritten test: enumerate
   `dynamo-power-agent-gpu-N` groups with `dcgmi group --list -j`, query
   each group with `dcgmi config --get -g <id> -j`, and read `Current` /
   `Target` from that per-group payload.
3. **Conflict coverage status.** §8.5 and §8.7 still advertised B5 as a
   runnable live test even though AKS CDI rejects the current bystander
   admission path. Reframed B5 as designed-but-deferred, documented the
   class-level skip, and removed it from the expected-pass rows until the
   resource-claim + runtime-discovery rewrite lands.

### v0.7 (prior revision)

Four corrections from the fifth review pass. All findings verified
against the codebase before fix.

1. **M1 — `helm uninstall` immediately followed by `helm install`
   created a race that broke the live-test fixture.** The chart's
   `terminationGracePeriodSeconds: 30` (verified in
   `deploy/helm/charts/power-agent/templates/daemonset.yaml:41`) gives
   the old pod up to 30 s for its SIGTERM handler to restore caps.
   `helm uninstall` schedules deletion but returns immediately; a
   subsequent `helm install` provisions a new DS whose pod can be
   scheduled before the old one finishes terminating. Both pods
   coexist on the test node for a brief window and the test fixture's
   `assert len(pods.items) == 1` at `test_multi_dgd_live.py:186` would
   fail. Fixed by adding two safeguards in both phase 2.1 and phase 3.2
   of the README: (a) `kubectl wait pod ... --for=delete --timeout=90s`
   after uninstall, scoped by node + chart component label, with
   `|| true` to tolerate "no pods to wait for" (first phase 2 install);
   (b) a post-install belt-and-suspenders assertion via `kubectl get
   pod ... --no-headers | wc -l` that fails the script if more than
   one power-agent pod is on the test node.
2. **M2 — bystander PID precondition (`count >= 2`) was not a reliable
   proof the bystander had landed.** vLLM workers can legitimately
   contribute multiple compute PIDs (rank-main + helper subprocesses,
   plus KV cache async workers depending on the runtime version), so
   "GPU 0 has ≥ 2 compute PIDs" is satisfied by the pre-existing
   `vllm-dgd-a-prefill` alone. Replaced with a baseline-vs-delta diff:
   `TestMultiPodConflict` now captures `_gpu0_compute_pids()` **before**
   creating the bystander pod, then polls until at least one NEW PID
   (not in the baseline) appears in
   `nvmlDeviceGetComputeRunningProcesses(handle_0)`. Failure message
   prints both baseline and current PID sets to make root-causing
   easier. This proves *some* new compute context started on GPU 0
   after the bystander pod was created — the only weakness left is
   the "rank-restart at the wrong moment" race, which is steady-state
   stable in practice.
3. **L1 — §8.5/§8.6 headers and §8.7 referenced nonexistent test
   names.** Headers said
   `test_multi_dgd_live.py::test_nvml_actuator_pass` /
   `::test_dcgm_actuator_pass`; the actual test file uses
   `TestCapApplicationProof::test_nvml_per_gpu_cap_matches_topology`
   and `::test_dcgm_per_gpu_cap_matches_topology`. §8.7 also claimed
   "each gated by `pytest.mark.skipif(os.getenv("DCGM_HOSTENGINE_
   AVAILABLE") != "1")`" — only the DCGM-specific tests are
   env-gated, not the NVML ones. Rewrote §8.7 with an accurate
   per-class × per-phase coverage table and an explicit "Gating
   mechanics" section that documents the three env vars
   (`RUN_MULTI_DGD_LIVE`, `TEST_NODE`, `DCGM_HOSTENGINE_AVAILABLE`)
   and which tests each affects. Headers in §8.5 / §8.6 retitled to
   "(phase 2, NVML mode)" / "(phase 3, DCGM mode)" tying directly
   to the README workflow.
4. **L2 — conflict counters `== 1` vs `>= 1`.** §8.5.7 said
   `multi_pod_gpu_total{disposition="conflict"} == 1` and
   `safe_default_applied_total == 1`. The test asserts deltas `>= 1`
   (`test_multi_dgd_live.py:868`) because the agent ticks both
   counters on every reconcile that still sees the conflict, and the
   test waits ≥ 2 reconcile intervals. `== 1` would be flaky on a
   15 s reconcile interval where the natural wait is 30 s+. Doc
   text aligned with the test and now explicitly explains the
   reconcile semantics that make `>= 1` the correct invariant.

### v0.6 (prior revision)

Six corrections from the fourth review pass. All findings verified against
the codebase before fix; severity H/M/L preserved from the reviewer's
classification.

1. **H1 — runbook never actually exercised both actuator write paths.**
   The previous flow installed Power Agent in NVML mode, then *optionally*
   reinstalled it in DCGM mode before a **single** pytest invocation. That
   invocation didn't set `DCGM_HOSTENGINE_AVAILABLE=1`, so DCGM tests
   skipped silently — and if you did set the env var, the NVML
   cap-application proof (§8.5.6) would be reading NVML's view of a
   *DCGM-written* cap, which is not a proof of the NVML actuator's write
   path. Fixed by restructuring `examples/multi-dgd-live-test/README.md`
   into Phase 1 (shared setup), Phase 2 (NVML pass), and Phase 3 (DCGM
   pass), each with its own `helm uninstall` + `helm install` of Power
   Agent and its own pytest invocation with explicit env vars. Phase 2
   uses `-k 'not test_dcgm_'` to deselect DCGM cases; Phase 3 sets
   `DCGM_HOSTENGINE_AVAILABLE=1` and runs the full suite. Test docstring
   updated to match.
2. **M1 — `-m 'multi_dgd_live'` in the test docstring would deselect the
   suite.** `multi_dgd_live` is intentionally not registered in
   `pyproject.toml:243-305`; v0.5 removed it from `pytestmark` (review
   finding H2 of that revision) but the docstring example still showed
   `-m 'multi_dgd_live'`, which under `--strict-markers` either errors or
   selects nothing. Replaced with the two-phase recipe (above), which
   uses `-k` patterns and env-var gating instead of marker selection.
3. **M2 — standalone DCGM hostengine pinned to DCGM 3.x while the Power
   Agent vendors DCGM 4.x.** `components/power_agent/Dockerfile:35` sets
   `DCGM_IMAGE=nvcr.io/nvidia/cloud-native/dcgm:4.2.3-2-ubuntu22.04` and
   the comment at lines 31-34 explicitly notes that the hostengine and
   the bindings must share a major version (DCGM only guarantees
   intra-major back-compat). The standalone DS was pinned to
   `3.3.9-1-ubuntu22.04` — a major-version skew that would crash
   `dcgmConnect()` or silently corrupt configs. Bumped to
   `4.5.1-1-ubuntu22.04` (the only 4.x ubuntu22.04 tag actually
   resolvable on NGC — see PR #9790 finding #1) in
   `examples/multi-dgd-live-test/20-nvidia-dcgm-standalone-ds.yaml` with
   an inline comment linking back to the Dockerfile.
4. **L1 — stale "4 worker pods" / "Pod count = 4" in operative text.**
   §8.5 assertion 1 said "Pod count = 4" while listing all 5 pods
   inline; the `all_worker_pods` fixture docstring said "Return all 4
   worker pods" while the body's assert checks `== 5`. Both corrected to
   5 (A prefill + A decode + B prefill + B decode + C decode = 5 pods
   on 8 GPUs).
5. **L2 — doc described the wrong planner log format.** §8.5 assertion 1
   said the planner logs `Patched annotation
   dynamo.nvidia.com/gpu-power-limit=<W>`, but the actual log line
   emitted by `core/base.py::_apply_power_annotations` is
   `logger.info("Annotated pod %s with %s=%s", ...)` — i.e., `Annotated
   pod <pod-name> with dynamo.nvidia.com/gpu-power-limit=<W>`. The
   v0.5 fix to the M3 test regex used the right format
   (`Annotated pod`), but the doc text was never updated to match.
   Fixed.
6. **L3 — unused `Tuple` import.** v0.5 rewrote the cross-DGD
   contamination test (M3) into a log-scraping form that no longer uses
   `Tuple`. The import was left in `from typing import Dict, List, Set,
   Tuple`. Removed. Repo lints (flake8/ruff F401) now pass.

### v0.5 (prior revision)

Eight corrections from the third review pass against the codebase on
`pr1a/power-agent`, `pr1b/planner-infra`, and `pr1b/power-agent-dcgm-actuator`.
Findings categorized H/M/L by severity, all verified before fix.

1. **H1 — `mode` was missing from §8.4 DGD planner configs.**
   `PlannerConfig.mode: Literal["disagg", "prefill", "decode", "agg"]`
   (`config/planner_config.py:67`), dispatched by
   `core/__main__.py:40-51`. Without an explicit `mode`, vllm-dgd-c
   would have instantiated `DisaggPlanner`, demanded a `VllmPrefillWorker`
   that doesn't exist in the agg topology, and crashed at startup.
   §8.4 now requires `"mode": "disagg"` (A, B) / `"mode": "agg"` (C);
   YAML 30-/31-/32-vllm-dgd-{A,B,C}.yaml updated with comments tracing
   the requirement back to `__main__.py`.
2. **H2 — `multi_dgd_live` and `power_agent` were unregistered pytest
   markers.** `pyproject.toml:164` enables `--strict-markers`; the
   markers list (`pyproject.toml:243-305`) registers `planner`,
   `gpu_8`, `integration`, `pre_merge` but NOT `multi_dgd_live` or
   `power_agent`. With `RUN_MULTI_DGD_LIVE=1` the module-level
   `pytest.skip` wouldn't fire, `pytestmark` would be evaluated, and
   collection would fail. `test_multi_dgd_live.py` now uses only
   registered markers; a TODO comment instructs the eventual
   `components/src/dynamo/planner/tests/integration/` landing to add
   the two markers to `pyproject.toml`.
3. **H3 — bystander pod created no CUDA compute context.** The Power
   Agent reads PIDs via `nvmlDeviceGetComputeRunningProcesses`
   (verified at `actuator.py:199` and `actuator.py:604`), which only
   returns CUDA compute PIDs. The previous bystander ran
   `nvidia-smi --query-gpu=power.draw` in a loop — nvidia-smi
   queries don't create a CUDA context, so the multi-pod conflict
   detection in `_reconcile_gpu` would never see the bystander.
   `40-conflict-bystander-pod.yaml` and the inline manifest in
   `test_multi_dgd_live.py` now run `vllm-runtime` (cached on the
   test node, has torch) with `torch.cuda.init() + active kernel loop`
   that keeps the PID on the compute-running list. The test also adds
   a PID-visibility precondition: polls
   `nvmlDeviceGetComputeRunningProcesses(handle_0)` from inside the
   power-agent pod until the bystander's CUDA context shows up,
   *before* asserting the conflict counters tick.
4. **M1 — wrong helm chart path.** §8.3.A and the README ran
   `helm install ... deploy/helm/charts/dynamo-platform`. The actual
   path is `deploy/helm/charts/platform/` (chart's `metadata.name` is
   `dynamo-platform` — only the release-name convention coincides).
   Doc and README corrected; README now also explains the distinction
   between chart path, chart metadata name, and release name.
5. **M2 — wrong DCGM standalone DS file path and service hostname in
   §8.3.D.2.** The doc referenced `nvidia-dcgm-standalone-ds.yaml`
   (omitting the `20-` prefix) and host
   `nvidia-dcgm.kaim-dynamo-system.svc.cluster.local`. The actual file
   is `20-nvidia-dcgm-standalone-ds.yaml`, the Service is
   `nvidia-dcgm-standalone`, and the fully-qualified DNS is
   `nvidia-dcgm-standalone.kaim-dynamo-system.svc.cluster.local`.
   The `11-power-agent-values-dcgm.yaml` had the correct hostname
   all along (`agent.dcgm.host`), so the doc was the lone offender.
   Fixed.
6. **M3 — cross-DGD contamination assertion was strictly weaker than
   the doc claim.** §8.5.3 said "no pod-name from another DGD ever
   appears in this planner's `patch_namespaced_pod` calls"
   (planner-attribution claim). The test only checked whether
   "exclusive" watt values (350, 375, 300) appeared on the wrong
   pod, explicitly skipping the shared 325W value. Scenario "A-planner
   patched B-decode with 325" was undetectable. `test_no_cross_dgd_
   annotation_contamination` rewritten to scrape each planner pod's
   logs (`logger.info("Annotated pod %s with %s=%s", ...)` is emitted
   by `_apply_power_annotations` per `core/base.py` on
   `pr1b/planner-infra`), regex-match the per-patch log lines, and
   assert every patched pod-name belongs to that planner's DGD via
   the operator-stamped `nvidia.com/dynamo-graph-deployment-name`
   label. This matches the doc's stated invariant.
7. **M4 — `test_dcgm_target_config_registered` didn't verify
   registration.** §8.6.3 promised "After 1 reconcile cycle with
   `agent.dcgm.enforce=true`: `dcgm_enforce_failures_total == 0` AND
   `dcgmi config --get-target -g all` shows the same per-GPU watts as
   `--get`." The test asserted only the counter. Now it additionally
   `exec`s `dcgmi config --get-target -g all --json` inside the
   standalone hostengine pod and compares the parsed per-GPU watts
   against `_EXPECTED_CAP_PER_GPU`. Fallback path: if `dcgmi --json`
   isn't supported on the deployed hostengine version, the test
   skips with explicit reproduction steps.
8. **L1 — assorted stale/inconsistent doc details.**
   (a) "4 worker pods" in §8.3.C and §8.5 pre-conditions → corrected
       to 5 (A prefill + A decode + B prefill + B decode + C decode).
   (b) `conflict-bystander-pod.yaml` reference in §8.5.7 → now
       includes the `40-` prefix matching the actual file.
   (c) `--set image.tag=v1.1.0` in §8.3.B → bumped to `v1.2.0` to
       match `10-/11-power-agent-values-*.yaml`.
   (d) "Each DGD pins `metadata.labels.\"nvidia.com/
       dynamo-graph-deployment-name\"`" in §8.4 → rewritten. The
       graph-deployment-name label is operator-stamped on *worker
       pods*, not on the DGD CR's own metadata. The §A1 selector
       reads pod labels, so the DGD CR's labels are not the contract
       surface — the operator's label-injection behavior is.

### v0.4 (prior revision)

Tier 3 specialization for the AKS `dpp-dev-env` cluster. Added §8 with:

1. **Cluster ground truth table (§8.1).** Recorded the verified state of
   the live cluster as of 2026-05-21: 3× 8-GPU A100 nodes; no Power Agent
   DS deployed; no `nvidia-dcgm` hostengine (only `dcgm-exporter`); no
   TRTLLM / SGLang DGDs anywhere; `kaim-dynamo-system` has no operator yet;
   pull-secret reaches `nvcr.io` only. Each downstream §8.x decision
   traces back to a row in this table.
2. **A100 cap re-baseline (§8.2).** Re-baselined cap math from B200 max
   500 W (§2 Tier 1/2 numbers) to A100 SXM4 max 400 W: 300 / 325 / 350 /
   375 W across 4 distinct values, same structural shape as §2. Safe
   default 280 W = 70 % TDP per the chart's `values.yaml` recommendation.
3. **Topology specialization to VLLM-only (§8.2).** Replaced the
   multi-framework topology with three VLLM DGDs of varying shapes
   (disagg TP=1, disagg TP=2, agg TP=2) on a single 8-GPU node. Documented
   the reasoning: the only thing VLLM-only **cannot** exercise live is
   "three different runtime images coexisting" — a packaging concern, not
   a Phase-1 contract concern. Tier 2 retains the full multi-framework
   topology via FakeNode (§4).
4. **Setup runbook (§8.3).** Concrete `helm upgrade --install` commands
   for the Dynamo operator and the Power Agent chart, plus the two
   DCGM-hostengine deployment options (cluster-admin flip vs.
   namespace-local standalone DS). Picked the namespace-local option as
   the default to keep the test off the cluster's critical path.
5. **Six live assertions per actuator (§8.5, §8.6).** Each assertion has
   a `kubectl` snippet to make it hand-runnable, and each calls out *why*
   it cannot be made in Tier 1/2 — the NVML cap-application proof
   (§8.5.6) and the DCGM cap-application proof (§8.6.2) are the two
   assertions that are fundamentally beyond mocked tests.
6. **Multi-pod conflict scenario, live (§8.5.7).** Designed an ad-hoc
   "bystander" pod approach for triggering the B5 multi-pod conflict on
   real hardware, since A100 without MIG doesn't share GPU 0 across
   normal Pods.
7. **Relationship to existing single-DGD live suite (§8.8).** Verified
   `test_actuation_knobs_live.py` on `pr1b/planner-infra` covers only
   single-DGD mechanics. §8 is additive (cross-DGD invariants), not a
   replacement.

### v0.3 (prior revision)

Six corrections from second review pass against the codebase on
`pr1a/power-agent`, `pr1b/planner-infra`, and `pr1b/power-agent-dcgm-actuator`:

1. **E2 "zero NVML calls" wrong.** `reconcile_once` iterates every GPU every
   tick and calls `actuator.list_running_pids` + (when PIDs present)
   `actuator.apply_cap` on each. T-P's GPUs 2, 3 absolutely see NVML calls
   during V-P's restart window — one per tick by design. The correct
   invariant is "T-P's cap *value* stays at 480 W," not "zero calls." E2
   rewritten.
2. **B2 / §7 #9682 back-port surface wrong.** `pr1a/power-agent` has only
   `power_agent.py` (verified via `git ls-tree pr1a/power-agent
   components/power_agent/`) — no `actuator.py`, no actuator Protocol. The
   #9682 cap-write path is module-level `_apply_cap(handle, gpu_idx, w,
   metrics)` → `pynvml.nvmlDeviceSetPowerManagementLimit(handle, w*1000)`.
   B2 split to call out both surfaces; §7's #9682 follow-up describes
   patching `pynvml` or `_apply_cap` instead of `actuator.apply_cap`.
3. **B5 "GPUs 1-7 untouched" misleading.** The full reconcile still iterates
   them — they receive their *correct* topology caps, just not safe-default.
   The actual product invariant is that the conflict / safe-default
   *counters* do not increment for those GPUs, and their cap *values* match
   the table. B5 rewritten. (My actual `test_conflict_on_gpu0_does_not_disturb_other_gpus`
   already asserts the right thing — only the doc wording was off.)
4. **`_run_one_tick()` does not exist on #9683.** The per-tick body lives
   inside `async def run()`'s while-loop; the targeted async methods are
   `_apply_effects`, `_apply_power_annotations`, `_apply_scaling_targets`.
   §4 Tier-2 description re-targeted to `_apply_power_annotations()` as the
   right granularity for the planner-side TGP knob.
5. **Distinct Prom ports do NOT solve same-process planner fixtures.**
   `PlannerPrometheusMetrics.__init__()` (`monitoring/planner_metrics.py`)
   constructs `Gauge(...)` / `Counter(...)` without a `registry=` argument.
   They land on the process-wide `prometheus_client.REGISTRY` — a second
   instantiation raises `Duplicated timeseries in CollectorRegistry`
   regardless of port. A-note + §4 fixture re-written to use bare planners
   with `planner.prometheus_metrics = MagicMock()`. Real per-planner
   metrics would need a `registry=` parameter in product code (out of
   scope for this plan).
6. **C3 "3 pods cover 6 GPUs" wrong arithmetic + wrong granularity.** Any
   "one pod per framework" subset covers 1+2+2 = 5 GPUs, never 6.
   `_restore_orphaned_gpus_on_startup` operates per-UUID anyway: for each
   GPU index, check UUID-in-`_previously_managed` AND
   `list_running_pids == []`. C3 rewritten with three concrete per-UUID
   scenarios; per-pod-count framing dropped.

### v0.2

Seven corrections from first review pass against the codebase on `pr1a/power-agent`,
`pr1b/planner-infra`, and `pr1b/power-agent-dcgm-actuator`:

1. **Dedup test attribution.** The UID-dedup in `_reconcile_gpu` is on
   `pr1b/power-agent-dcgm-actuator` commit `ba3b5803a9`, NOT on
   `pr1a/power-agent`. The dedup test (B3, §5.1) lands on #9790, not #9682;
   §7 landing sequence updated.
2. **AggPlanner behavior.** `AggPlanner` has `require_prefill=False,
   require_decode=True` (`core/adapters.py:99-103`), so
   `_apply_power_annotations` patches the aggregated worker via the DECODE
   branch — does NOT short-circuit. A3 rewritten; §5.2 skeleton renamed to
   `test_aggregated_dgd_patches_decode_branch_only`.
3. **Label name.** The canonical label is
   `nvidia.com/dynamo-graph-deployment-name` (with `-name` suffix), and per-component
   pods carry `nvidia.com/dynamo-component=<service-key>` —
   `pr1b/planner-infra` `connectors/kubernetes.py`. A1, §4 fixture, §5.2 fixed.
4. **K8s API does NOT enforce DGD isolation.** `patch_pod_annotation(pod_name,
   key, value)` patches by name in the planner's namespace — any pod name will
   work. Invariant #5 (§2) rewritten; A1/A2 sharpened to pin the selector
   string and require RBAC scope.
5. **"Distinct" miscount.** Cap value 450W appears on 3 GPUs (V-P, T-D × 2)
   and 480W appears on 2 GPUs (T-P × 2) and 500W on 2 GPUs (S-A × 2). Total
   distinct watt values = 4, not 8. Invariants, B2, D1, §5 docstrings fixed.
6. **DCGM counter semantics.** `dcgm_enforce_failures_total` increments only
   when `dcgmConfigEnforce()` fails after a successful `dcgmConfigSet()`
   (`actuator.py:836-875`). A successful `_with_reconnect` retry on a
   `DCGM_ST_CONNECTION_NOT_VALID` raised during Set leaves BOTH counters at
   zero. B7 split into B7 (reconnect-on-Set, zero counters) and B8
   (Enforce-fails-after-Set, only `dcgm_enforce_failures_total` ticks).
7. **`_ensure_identity_map` return shape.** Returns `None`; mutates
   `_dcgm_uuid_by_idx` (list) and `_nvml_index_by_uuid` (dict)
   (`actuator.py:606-667`). B6 rewritten to assert state mutation + behavior
   via `list_running_pids`, not a return value.

**Demotion:** original A5 ("three planners writing to shared Prom registry")
demoted to A-note. Each planner pod has its own `prometheus_port` in
production — the concern is purely a test-fixture artifact, solvable by
distinct ports per planner. Not a product test.

### v0.1

Initial draft.
