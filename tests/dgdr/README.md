# DGDR v1beta1 End-to-End Test Suite

This directory contains the end-to-end test suite for **DynamoGraphDeploymentRequest
(DGDR) v1beta1** — the high-level, SLA-driven Kubernetes API for deploying
inference models with Dynamo.

## What's tested

| Test group | Marker(s) | GPU req | Mocker OK? | What it covers |
|---|---|---|---|---|
| `TestDGDRValidation` | `gpu_0`, `pre_merge` | None | ✅ | Webhook validation: rejected/accepted specs, value enforcement, storage version, shortname |
| `TestDGDRVersionConversion` | `gpu_0`, `pre_merge` | None | ✅ | v1alpha1 → v1beta1 conversion webhook |
| `TestDGDRMinimalDeployment` | `gpu_1`, `pre_merge`, `e2e` | 1+ | ⚠️ see note | Full Pending → Profiling → Ready → Deploying → Deployed lifecycle |
| `TestDGDRBackendSelection` | `gpu_1`, `nightly`, `e2e` | 1+ | ⚠️ vllm+trtllm only | vllm and trtllm pass; sglang **skipped** (no AIC silicon data for sglang on the mocker GPU SKU) |
| `TestDGDRSearchStrategies` | `gpu_1`/`gpu_8`, `e2e` | 1 or 8 | ⚠️ rapid only | `rapid` uses AIC and works; `thorough` requires real GPU sweeps |
| `TestDGDRSLATargets` | `gpu_1`, `nightly`, `e2e` | 1+ | ✅ | ttft+itl, e2eLatency, optimizationType (latency/throughput) |
| `TestDGDRWorkloadPickingModes` | `gpu_1`, `nightly`, `e2e` | 1+ | ✅ | requestRate, concurrency, isl/osl |
| `TestDGDRFeatures` | `gpu_1`, `nightly`, `e2e` | 1+ | ⚠️ see note | planner (rapid/none sweep), mocker |
| `TestDGDRModelCache` | `gpu_1`, `nightly`, `e2e` | 1+ | ✅ | PVC-backed model cache, cache propagated to DGD |
| `TestDGDRHardwareOverride` | `gpu_1`, `pre_merge`, `e2e` | ✅ | ✅ | Manual gpuSku/numGpusPerNode/totalGpus/vramMb |
| `TestDGDRAutoApply` | `gpu_1`, `pre_merge`, `e2e` | 1+ | ⚠️ see note | autoApply=true **skipped** in mocker (operator race); autoApply=false keeps Ready |
| `TestDGDROverrides` | `gpu_1`, `nightly`, `e2e` | 1+ | ✅ | Profiling job tolerations; DGD metadata label merging **xfail** (operator gap) |
| `TestDGDRStatusAndConditions` | `gpu_1`, `pre_merge`, `e2e` | 1+ | ⚠️ see note | All conditions set correctly, sub-phases tracked, Pareto configs; all-conditions **xfail** in mocker; pareto **skipped** in mocker |
| `TestDGDRImmutability` | mixed | 0–1 | ⚠️ see note | Spec rejected in Profiling/Deployed, metadata always allowed |
| `TestDGDRCleanup` | `gpu_1`, `pre_merge`, `e2e` | 1+ | ⚠️ see note | Job deleted with DGDR; DGD preserved; ConfigMap cleanup **xfail** (operator gap); DGD-persistence test **skipped** in mocker |
| `TestDGDRMoEModels` | `gpu_8`, `nightly`, `e2e` | 8 | ❌ | DeepSeek-R1 MoE on SGLang — requires real 8-GPU node |

## Prerequisites

1. A running Kubernetes cluster with GPU nodes (or see [GPU-free mode](#gpu-free-mocker-mode) below)
2. The Dynamo operator installed (including CRDs and webhooks)
3. `kubectl` configured and pointing at the cluster
4. Python 3.10+ with `pytest` and `pyyaml` installed:
   ```bash
   pip install pytest pyyaml
   # or, from the repo root:
   pip install -e ".[test]"
   ```

## One-time cluster setup

Before running any tests, ensure the following are in place in your cluster.
These are required even for GPU-free (mocker) mode.

### 1. Install the Dynamo operator

```bash
cd deploy/operator
helm install dynamo-operator helm/dynamo-operator -n dynamo-system --create-namespace
```

### 2. Deploy NATS

Mocker workers (and real workers) connect to NATS for inter-component messaging.
The operator expects NATS at `nats://dynamo-operator-nats.dynamo-system.svc.cluster.local:4222`.

```bash
helm repo add nats https://nats-io.github.io/k8s/helm/charts/
helm repo update
helm install dynamo-operator-nats nats/nats -n dynamo-system --create-namespace
```

### 3. Create the HuggingFace token secret

The profiling job reads the HF token from a secret named `hf-token-secret` using the
key `HF_TOKEN` (not `HUGGING_FACE_HUB_TOKEN`).

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your-hf-token> \
  -n default
# If running in a non-default namespace, adjust -n accordingly
```

> **Important:** The key must be `HF_TOKEN`.  The secret name must be `hf-token-secret`.
> Using a different key name will cause the profiling job to fail silently.

## Running the tests

There are two main ways to run the suite depending on whether you have GPU hardware.

---

### GPU-free (mocker mode) — recommended for local development and CI

No GPU nodes required.  Uses AIC simulation for profiling and mock inference workers
for deployment.  Covers all `gpu_0` and `gpu_1` tests (~45 tests); `gpu_8` tests are
excluded because they require a real 8-GPU node even in mocker mode.

```bash
python3 -m pytest tests/dgdr/ -m "gpu_0 or gpu_1" -v \
  --dgdr-namespace=default \
  --dgdr-image=<your-image>
```

Expect: 37 passed, 6 skipped (2 model-cache PVC; sglang backend; pareto in mocker; DGD-persistence in mocker; auto-apply-true in mocker), 4 xfail (DGD label merging; all-conditions requires Deployed; dry-run immutability requires Deployed; ConfigMap cleanup on deletion).
`test_backend[sglang]` is one of the 6 skips (no AIC silicon data for sglang in mocker mode).

---

### Full suite with real GPUs — for production/nightly validation

Requires a Kubernetes cluster with GPU nodes.  Set `--dgdr-no-mocker` to disable mocker
injection and run against real hardware.  `gpu_8` tests additionally require an 8-GPU node.

```bash
# gpu_0 + gpu_1 tests on real GPUs (single-GPU node sufficient)
python3 -m pytest tests/dgdr/ -m "gpu_0 or gpu_1" -v \
  --dgdr-namespace=dynamo-test \
  --dgdr-image=<your-image> \
  --dgdr-no-mocker \
  --dgdr-profiling-timeout=3600 \
  --dgdr-deploy-timeout=1800

# Full nightly suite including 8-GPU tests
python3 -m pytest tests/dgdr/ -v \
  --dgdr-namespace=dynamo-test \
  --dgdr-image=<your-image> \
  --dgdr-no-mocker \
  --dgdr-pvc-name=model-cache \
  --dgdr-profiling-timeout=14400 \
  --dgdr-deploy-timeout=3600
```

Expect (gpu_0 + gpu_1, with `--dgdr-pvc-name`): **~43 passed, 0 skipped, 2 xfail** (DGD label-merging operator gap; ConfigMap cleanup operator gap).
Without `--dgdr-pvc-name`: 2 additional skips for the model-cache tests.

> **Note:** Two xfails are **permanent operator gaps** that persist in both mocker and GPU mode:
> - `test_dgd_override_injects_custom_labels` — the operator does not yet merge `spec.overrides.dgd.metadata.labels` onto the created DGD.
> - `test_deletion_removes_output_configmap` — the operator's `FinalizeResource` is a no-op and does not delete the output ConfigMap on DGDR deletion.
> All other mocker-mode xfails/skips disappear in GPU mode and are expected to pass.

---

### Other useful invocations

```bash
# Validation + conversion tests only (no cluster setup required beyond CRDs)
python3 -m pytest tests/dgdr/ -m "gpu_0" -v \
  --dgdr-namespace=default \
  --dgdr-image=<your-image>

# Pre-merge gate (GPU-free)
python3 -m pytest tests/dgdr/ -m "pre_merge" -v \
  --dgdr-namespace=default \
  --dgdr-image=<your-image>

# Single test class
python3 -m pytest tests/dgdr/test_dgdr_v1beta1.py::TestDGDRAutoApply -v \
  --dgdr-namespace=default \
  --dgdr-image=<your-image>
```

## CLI options

| Option | Default | Description |
|---|---|---|
| `--dgdr-namespace` | _(required)_ | Kubernetes namespace for test resources |
| `--dgdr-image` | _(required)_ | Container image for profiling and inference workers |
| `--dgdr-model` | `Qwen/Qwen3-0.6B` | HuggingFace model ID used by most tests |
| `--dgdr-backend` | `vllm` | Default backend for DGDR tests |
| `--dgdr-pvc-name` | _(empty)_ | PVC name holding pre-downloaded model weights (PVC tests are skipped if unset) |
| `--dgdr-profiling-timeout` | `3600` | Seconds to wait for profiling to complete |
| `--dgdr-deploy-timeout` | `600` | Seconds to wait for DGD to reach Deployed phase |
| `--dgdr-no-mocker` | `false` | Disable mocker mode (require real GPU nodes) |

## DGDR v1beta1 feature coverage matrix

The following spec fields are exercised by at least one test:

| Field | Tests that exercise it |
|---|---|
| `spec.model` | All tests |
| `spec.backend` (auto/vllm/sglang/trtllm) | `TestDGDRBackendSelection`, `TestDGDRValidation` |
| `spec.image` | All tests |
| `spec.searchStrategy` (rapid/thorough) | `TestDGDRSearchStrategies` |
| `spec.sla.ttft` + `spec.sla.itl` | `TestDGDRSLATargets::test_sla_ttft_and_itl` |
| `spec.sla.e2eLatency` | `TestDGDRSLATargets::test_sla_e2e_latency` |
| `spec.sla.optimizationType` | `TestDGDRSLATargets::test_sla_optimization_type_*` |
| `spec.workload.isl` + `spec.workload.osl` | `TestDGDRWorkloadPickingModes` |
| `spec.workload.requestRate` | `TestDGDRWorkloadPickingModes::test_request_rate_picking` |
| `spec.workload.concurrency` | `TestDGDRWorkloadPickingModes::test_concurrency_picking` |
| `spec.features.planner` (opaque config) | `TestDGDRFeatures::test_planner_enabled_*` |
| `spec.features.mocker.enabled` | `TestDGDRFeatures::test_mocker_enabled` |
| `spec.modelCache.pvcName` | `TestDGDRModelCache` |
| `spec.hardware.gpuSku` | `TestDGDRHardwareOverride::test_hardware_manual_override` |
| `spec.hardware.numGpusPerNode` | `TestDGDRHardwareOverride` |
| `spec.hardware.totalGpus` / `spec.hardware.vramMb` | `TestDGDRHardwareOverride::test_hardware_total_gpus_and_vram` |
| `spec.autoApply` | `TestDGDRAutoApply` |
| `spec.overrides.profilingJob` | `TestDGDROverrides::test_profiling_job_toleration_override` |
| `spec.overrides.dgd` | `TestDGDROverrides::test_dgd_override_injects_custom_labels` |
| `status.phase` | All lifecycle tests |
| `status.profilingPhase` | `TestDGDRStatusAndConditions::test_profiling_sub_phase_tracked` |
| `status.profilingJobName` | `TestDGDRStatusAndConditions::test_profiling_job_name_populated` |
| `status.dgdName` | `TestDGDRAutoApply`, `TestDGDRMinimalDeployment` |
| `status.profilingResults.selectedConfig` | Multiple |
| `status.profilingResults.pareto` | `TestDGDRStatusAndConditions::test_pareto_configs_in_profiling_results` |
| `status.deploymentInfo` | `TestDGDRMinimalDeployment` |
| `status.conditions` (all types) | `TestDGDRStatusAndConditions` |
| `status.observedGeneration` | `TestDGDRStatusAndConditions::test_observed_generation_tracks_spec` |

## GPU-free mode (default)

By default, the test suite runs the full DGDR lifecycle **without any GPU nodes**
by combining two simulation features:

| Feature | How it's enabled | Which phase it affects |
|---|---|---|
| **AIC (AI Configurator)** | `searchStrategy: rapid` (the default) | **Profiling** — profiler runs CPU-only simulation instead of online GPU sweep |
| **Mocker** | Enabled by default (disable with `--dgdr-no-mocker`) | **Deployment** — DGD uses mock inference workers (no GPU resources requested) |

**How it works:**

- `searchStrategy: rapid` is the default for v1beta1 DGDRs.  The profiler automatically
  uses AI Configurator (AIC) simulation when rapid is set — no additional config needed.
- Mocker mode is **enabled by default**.  The `dgdr_factory` fixture automatically injects
  `spec.features.mocker.enabled: true` and a default `spec.hardware` config into every DGDR.
- AIC profiling creates a Kubernetes Job that runs CPU-only (job prefix: `profile-aic-`).
  The profiling pod does not request GPU resources.
- Mocker deployment selects the profiler's `mocker_config_with_planner.yaml` output
  instead of the real deployment config, resulting in DGD pods that don't request GPUs.
- Pass `--dgdr-no-mocker` to disable mocker mode and run against real GPU hardware.

> **Note:** Some test assertions (e.g., status.deploymentInfo.gpuCount, pareto configs)
> may produce different values under mocker than under real GPU profiling.
> The tests are written to validate structure and phase transitions, not exact
> profiling output values, so they work correctly in both modes.

> **Note:** `searchStrategy: thorough` requires online (GPU) profiling even with mocker,
> since thorough performs real benchmark measurements.  Use rapid for GPU-free testing.

> **Note:** `TestDGDRFeatures::test_planner_enabled_with_rapid_sweep` runs with
> `auto_apply=False` in mocker mode (same root cause as the note below — the operator
> pre-sets `Status.DGDName` from the profiling output and then immediately fires
> `handleDGDDeleted` when the DGD cannot be found).  In mocker mode the test only
> validates that spec generation succeeds (waits for `PHASE_READY` and checks `dgdName`
> + `selectedConfig`).  Full deployment with rapid sweeping is verified outside mocker
> mode.  `test_planner_enabled_no_pre_deployment_sweep` and `test_mocker_enabled` are
> likewise restricted to `PHASE_READY` in mocker mode.

> **Note:** `auto_apply=True` consistently hits `handleDGDDeleted` in mocker mode. The
> operator's `generateDGDSpec` pre-populates `Status.DGDName` from the profiling output
> (e.g. `mocker-disagg`) _before_ the DGD is actually created.  When `handleDeployingPhase`
> then runs it checks `DGDName != ""` and immediately tries to GET that DGD; since it does
> not exist yet it fires `handleDGDDeleted` and the DGDR transitions to Failed.
> All tests that would enter the Deploying phase in mocker mode therefore use
> `auto_apply=False`/`PHASE_READY` instead (minimal lifecycle, backend selection,
> mocker feature, planner-no-sweep, planner-rapid-sweep, DGD label override).
> Tests whose sole purpose is to verify `auto_apply=True` DGD creation are skipped in
> mocker mode (`test_auto_apply_true_creates_dgd_automatically`,
> `test_deletion_does_not_remove_created_dgd`).
> Non-mocker mode (real GPU cluster) is unaffected.

> **Note:** `TestDGDRImmutability::test_spec_immutable_in_deployed_via_dry_run` is **xfail**
> in mocker mode.  The test relies on the session `deployed_dgdr` fixture which, in mocker
> mode, stops at `PHASE_READY` instead of `PHASE_DEPLOYED`.  The webhook's
> `ValidateUpdate` immutability enforcement only activates when the DGDR is in `Deployed`
> phase, so the server-dry-run mutation is accepted rather than rejected.

> **Note:** `gpu_8` tests cannot be run with mocker and require a real 8-GPU node.
> `TestDGDRSearchStrategies::test_thorough_strategy_completes` uses `searchStrategy: thorough`
> which performs real GPU benchmark sweeps.  `TestDGDRMoEModels` (DeepSeek-R1) requires 8 GPUs
> for the real inference workload.  Exclude them from GPU-free runs with `-m "gpu_0 or gpu_1"`.

### AIC silicon data availability

AIC operates in **silicon mode**: it looks up pre-recorded per-op performance data
files shipped inside the `aiconfigurator` Python package.  These files are organised
by `{gpu_sku}/{backend}/{backend_version}/`.  The mocker fixture injects
`gpuSku: a100_sxm` into every DGDR — but the package only ships vllm data for that SKU:

| Backend | a100_sxm data? | Mocker result |
|---|---|---|
| `vllm` | ✅ present | Profiling succeeds |
| `trtllm` | ✅ present | Profiling succeeds |
| `sglang` | ❌ missing | Test **skipped** automatically (no `sglang/0.5.8` perf data for `a100_sxm`) |

To test sglang/trtllm, run against a real GPU cluster (`--dgdr-no-mocker`) where AIC
can use a GPU SKU for which those data files are present.

## Cleanup

Tests clean up their own DGDRs via the `dgdr_factory` fixture.  If a test is
interrupted, resources can be cleaned up manually:

```bash
# Delete all DGDRs created by the test suite (they are labelled automatically)
kubectl delete dgdr -n default -l "test.dynamo/managed=true"

# If you used a custom namespace:
kubectl delete dgdr -n <namespace> -l "test.dynamo/managed=true"
```

## Architecture notes

- All tests interact with the cluster **exclusively via `kubectl`** subprocess calls,
  consistent with the rest of the Dynamo test suite.
- The `dgdr_factory` fixture ensures DGDR cleanup via `yield` regardless of test
  outcome.
- Tests that require an optional PVC (`--dgdr-pvc-name`) skip automatically when the
  option is not provided.
- Timeout values are configurable to accommodate clusters with varying profiling speeds.
