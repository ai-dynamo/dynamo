# Branch Context: `failover/m6-operator`

## Purpose

This branch implements **single-node GPU failover** for vLLM inference engines on Kubernetes. Two vLLM engine instances share the same physical GPUs within a single pod via Dynamic Resource Allocation (DRA). One engine is active; the other is a hot standby. When the active engine dies, the standby acquires a file lock, wakes up, allocates KV cache (~170ms), and registers with discovery — achieving sub-second failover with no extra nodes and no cold model load.

## Branch History (failover-specific commits only)

```
452dc1597  Add GPU failover POC decisions document
bc3fc1099  feat: add flock-based failover lock for engine leader election
70077195d  feat: deterministic weight loading via ENGINE_ID env var
6b1e5cbaf  feat: add GMS failure handling in remap path (Diff 3)
91c632d88  feat: lock-driven shadow engine failover with health probes (Diff 4+5)
88fb83c0c  fix: missed adding it to setup.py
bd442b1a6  feat: add failover CRD, pod transformation, and webhook validation (Diff 6)
78d5ff8e7  fix: align failover operator with validated K8s pod spec
b05c99c7f  fix: use ServiceName for RCT naming consistency in DCD path
e8aeb6849  fix: add vllm-agg-failover.yaml
```

The branch is rebased on top of a recent `main` snapshot (commit `3121cae20 fix: refreshed branch`). The large diff stat (746 files) is dominated by upstream main changes; the failover-specific changes touch ~30 files.

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│           Failover Worker Pod               │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │  GMS Weight Server  (init sidecar)    │  │
│  │  Loads model weights once into VRAM   │  │
│  └──────────────────┬────────────────────┘  │
│                     │ shared weights        │
│           ┌─────────┴─────────┐             │
│           ▼                   ▼             │
│  ┌─────────────────┐ ┌─────────────────┐   │
│  │   Engine-0      │ │   Engine-1      │   │
│  │   ACTIVE  ✅    │ │   SHADOW  💤    │   │
│  │                 │ │                 │   │
│  │  • Holds lock   │ │  • Blocked on   │   │
│  │  • KV cache up  │ │    lock         │   │
│  │  • Serving reqs │ │  • No KV cache  │   │
│  └────────┬────────┘ └────────┬────────┘   │
│           │    🔒 failover.lock │           │
│           └─────────┬─────────┘             │
│                     │                       │
│  ┌──────────────────┴────────────────────┐  │
│  │  ResourceClaimTemplate (DRA)          │  │
│  │  All 3 containers share same GPUs     │  │
│  └──────────────────┬────────────────────┘  │
└─────────────────────┼───────────────────────┘
                      │
               ┌──────┴──────┐
               │ GPU 0  GPU 1│
               └─────────────┘
```

### Three Pillars

1. **Dynamic Resource Allocation (DRA)**: Both engines + GMS share GPUs via a `ResourceClaimTemplate` with `gpu.nvidia.com` device class, replacing traditional `nvidia.com/gpu` resource limits.

2. **GPU Memory Service (GMS)**: An init sidecar (`restartPolicy: Always`) that loads model weights once into VRAM. Both engines read weights via VA-stable shared memory through Unix Domain Sockets (`/shared/gms_<GPU-UUID>.sock`). `TMPDIR=/shared` makes both GMS and engines agree on socket location.

3. **Flock-based Leader Election**: A POSIX advisory lock on `/shared/failover.lock` (an `emptyDir` volume). Both engines start in `shadow` mode — they load weights via GMS, skip KV cache, then sleep and race for the lock. The winner wakes, allocates KV cache, and registers with etcd discovery. When it dies, the lock is released, and the shadow takes over.

---

## Components Modified

### Python (engine/GMS side) — THIS IS WHERE THE NEXT AGENT WILL WORK

| File | What it does |
|---|---|
| `lib/gpu_memory_service/failover_lock/` | **New package**: `FlockFailoverLock` — POSIX advisory file lock with `acquire()` (blocking flock) and `release()` |
| `lib/gpu_memory_service/failover_lock/flock/lock.py` | Core lock implementation using `fcntl.flock(LOCK_EX)` |
| `lib/gpu_memory_service/failover_lock/interface.py` | Abstract `FailoverLock` interface |
| `lib/gpu_memory_service/setup.py` | Added `failover_lock` and `failover_lock.flock` packages |
| `lib/gpu_memory_service/integrations/vllm/patches.py` | **New**: Monkey-patches vLLM's `determine_num_available_blocks` to return 0 when `SHADOW_SKIP_KV_CACHE=1` (shadow mode skips KV cache at init); provides `allocate_kv_cache_on_wake()` for deferred allocation |
| `lib/gpu_memory_service/integrations/vllm/worker.py` | Modified to support `ENGINE_ID` env var for deterministic weight loading order (ENGINE_ID=0 gets RW lock, others get RO) |
| `lib/gpu_memory_service/integrations/vllm/model_loader.py` | GMS failure handling improvements in remap path |
| `components/src/dynamo/vllm/main.py` | **Core failover orchestration**: when `gms_mode == "shadow"`, the engine loads weights → sleeps → sets health status to ready → acquires failover lock → wakes (allocates KV cache) → registers with discovery |
| `components/src/dynamo/vllm/args.py` | Validation: `--gms-mode shadow` requires `--load-format gms` |
| `components/src/dynamo/vllm/backend_args.py` | Added `--gms-mode` argument (choices: `normal`, `shadow`; env var: `DYN_VLLM_GMS_MODE`) |

### Go (Kubernetes operator)

| File | What it does |
|---|---|
| `deploy/operator/api/v1alpha1/common.go` | Added `FailoverSpec` struct (`enabled bool`) |
| `deploy/operator/api/v1alpha1/dynamocomponentdeployment_types.go` | Added `Failover *FailoverSpec` field to `DynamoComponentDeploymentSharedSpec` |
| `deploy/operator/internal/dynamo/failover.go` | **Core**: `buildFailoverPod()` transforms a single-container pod spec into a failover pod with GMS sidecar, 2 engine containers, DRA claims, volumes, tolerations. Also `GenerateFailoverResourceClaimTemplate()` |
| `deploy/operator/internal/dynamo/failover_test.go` | 25 unit tests covering all failover functions |
| `deploy/operator/internal/dynamo/graph.go` | Calls `buildFailoverPod()` when `isFailoverEnabled()`, passes `controllerConfig.EtcdAddress` for fail-fast check. Uses `Spec.ServiceName` for consistent naming |
| `deploy/operator/internal/controller/dynamographdeployment_controller.go` | Grove path: `SyncResource` for `ResourceClaimTemplate` |
| `deploy/operator/internal/controller/dynamocomponentdeployment_controller.go` | DCD path: `SyncResource` for `ResourceClaimTemplate` |
| `deploy/operator/internal/dynamo/shared.go` | Webhook validation: prevents `failover.enabled` on non-worker component types |
| `deploy/helm/charts/platform/components/operator/crds/` | Updated CRDs with `FailoverSpec` schema |

### Kubernetes Manifests & Test Assets

| File | What it does |
|---|---|
| `failover-pod-test.yaml` | **Source of truth**: Manually validated pod spec with 2 engines + GMS + DRA (used as reference for operator alignment) |
| `failover-infra.yaml` | etcd deployment for manual testing |
| `failover-frontend.yaml` | Frontend pod for manual testing |
| `examples/backends/vllm/deploy/vllm-agg-failover.yaml` | DGD example for operator-driven failover |
| `test_lock_driven_failover.sh` | **Primary smoke test**: Full failover lifecycle (5 phases, TP=2) |
| `test_shadow_failover.sh` | Failover test variant (single GPU, HTTP wake) |
| `test_shadow_sleep_wake.sh` | Shadow sleep/wake + inference test (single GPU) |
| `tests/fault_tolerance/gpu_memory_service/test_failover_lock.py` | Unit tests for `FlockFailoverLock` (no GPU needed) |

---

## Key Environment Variables Injected by Operator

For each engine container (engine-0, engine-1), the operator injects:

| Env Var | engine-0 | engine-1 | Source |
|---|---|---|---|
| `ENGINE_ID` | `0` | `1` | `failover.go` |
| `TMPDIR` | `/shared` | `/shared` | `failover.go` |
| `FAILOVER_LOCK_PATH` | `/shared/failover.lock` | `/shared/failover.lock` | `failover.go` |
| `DYN_SYSTEM_PORT` | `9090` | `9091` | `failover.go` |
| `DYN_VLLM_GMS_MODE` | `shadow` | `shadow` | `failover.go` |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `5600` | `5601` | `failover.go` |
| `DYN_VLLM_KV_EVENT_PORT` | `20080` | `20081` | `failover.go` |
| `DYN_DISCOVERY_BACKEND` | `etcd` | `etcd` | `failover.go` (overrides global) |
| `ETCD_ENDPOINTS` | auto | auto | `graph.go` (standard injection) |
| `DYN_NAMESPACE` | auto | auto | `component_common.go` |

---

## Validation Procedures

### Test Script Inventory

| Script | GPUs Needed | TP Size | What it Tests |
|---|---|---|---|
| `test_lock_driven_failover.sh` | 2 | 2 | Full failover: deterministic loading, flock race, health probes, inference, kill+failover, post-failover inference. **This is the primary validation script.** |
| `test_shadow_failover.sh` | 1 | 1 | Primary→shadow failover with HTTP wake (kill primary, wake shadow, test inference) |
| `test_shadow_sleep_wake.sh` | 1 | 1 | Shadow KV skip, wake allocation, inference. Simpler version for isolated shadow testing |
| `tests/fault_tolerance/gpu_memory_service/test_failover_lock.py` | 0 | - | Pure Python flock tests (asyncio + multiprocess). Run with `pytest` |

### Step-by-Step: Setting Up a Worktree for Python Development & Testing

This is the workflow the next agent should follow when making Python changes.

**1. Create a git worktree (to isolate from the main checkout):**

```bash
cd /home/mabdulwahhab/repos/dynamo-8
git worktree add ../dynamo-8-failover-wt failover/m6-operator
cd ../dynamo-8-failover-wt
```

**2. Create and activate virtualenv:**

```bash
uv venv dynamo
source dynamo/bin/activate
uv pip install pip maturin
```

**3. Build the Rust bindings:**

```bash
cd lib/bindings/python
maturin develop --uv
cd ../../..
```

**4. Install GPU Memory Service (includes failover_lock package):**

```bash
uv pip install -e lib/gpu_memory_service
```

**5. Install the dynamo wheel:**

```bash
uv pip install -e .
```

**6. Set up .env with your HF token:**

```bash
echo 'export HF_TOKEN=<your_token>' > .env
```

**7. Run the primary smoke test:**

```bash
./test_lock_driven_failover.sh Qwen/Qwen3-0.6B
```

### What the Primary Test (`test_lock_driven_failover.sh`) Validates

The test runs 5 phases and produces PASS/FAIL assertions:

- **Phase 0: GMS Startup** — Starts `gpu_memory_service --device 0` and `--device 1`. Waits for "waiting for connections".
- **Phase 1: Deterministic Weight Loading (D5)** — Starts Engine B (ENGINE_ID=1, RO) first — it must block. Then starts Engine A (ENGINE_ID=0, RW). Engine A commits weights, Engine B unblocks.
- **Phase 2: Lock-Driven Wake (D4)** — Both engines reach STANDBY (sleeping, waiting for flock). Exactly one acquires the lock and wakes.
- **Phase 3: Health Probes (D2)** — Winner returns 200 (ACTIVE), loser returns 200 (STANDBY).
- **Phase 4: Discovery & Inference** — Frontend starts, discovers winner, inference works. Loser never registered.
- **Phase 5: Failover (D4+D7)** — Kill winner → loser acquires lock → wakes → registers → inference works again. Reports failover timing breakdown.

**Important test script details:**
- Uses `VENV_NAME` env var (default: `dynamo`) to find the virtualenv
- `SCRIPT_DIR` determines where to find `.env` — must be in the worktree root
- Logs go to `/tmp/failover_tp2_test_$$` (PID-isolated)
- Traps EXIT for cleanup
- Staggered ports: Engine A uses 8100/5600/20080, Engine B uses 8101/5601/20081

### Rebuilding After Python Changes

| What changed | Rebuild command |
|---|---|
| `lib/gpu_memory_service/` (GMS, failover_lock, patches) | `uv pip install -e lib/gpu_memory_service` |
| `components/` or top-level Python (vllm main, args) | `uv pip install -e .` |
| `lib/bindings/python/` (Rust bindings) | `cd lib/bindings/python && maturin develop --uv` |

After rebuilding, re-run the relevant test script.

### Running the Lock Unit Tests (no GPU needed)

```bash
source dynamo/bin/activate
pytest tests/fault_tolerance/gpu_memory_service/test_failover_lock.py -v
```

### Operator Unit Tests (Go)

```bash
cd deploy/operator
go test ./internal/dynamo/ -v -run TestBuild
go test ./internal/dynamo/ -v -run TestGenerate
go test ./internal/dynamo/ -v -run TestFailover
```

Or run all operator tests:

```bash
cd deploy/operator
make test
```

---

## Kubernetes E2E Validation (Operator Path)

This section documents how to validate the operator changes end-to-end. The Python agent likely won't need this, but it's here for completeness.

**Cluster requirements:**
- Kubernetes >= 1.31 with DRA support
- NVIDIA GPU DRA driver installed (`gpu.nvidia.com` DeviceClass)
- 2+ GPUs on a single node

**Platform setup:**

```bash
cd deploy/helm/charts && helm dep build ./platform/

helm install dynamo-platform ./platform/ \
  --namespace failover-test --create-namespace \
  --set global.etcd.install=true \
  --set dynamo-operator.discoveryBackend=etcd \
  --set dynamo-operator.namespaceRestriction.enabled=true \
  --set dynamo-operator.gpuDiscovery.enabled=false \
  --set dynamo-operator.controllerManager.manager.image.repository=<registry>/dynamo \
  --set dynamo-operator.controllerManager.manager.image.tag=<tag>
```

**Deploy the DGD:**

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<token> -n failover-test

kubectl apply -f examples/backends/vllm/deploy/vllm-agg-failover.yaml -n failover-test
```

**Validation workflow:**

```bash
# Verify 3/3 containers running (gms-weights + engine-0 + engine-1)
kubectl get pods -n failover-test -l nvidia.com/dynamo-component=worker

WORKER=$(kubectl get pods -n failover-test -l nvidia.com/dynamo-component=worker \
  -o jsonpath='{.items[0].metadata.name}')

kubectl port-forward svc/vllm-agg-failover-frontend 8000:8000 -n failover-test &

# Test completions
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":32}'

# Kill active engine, verify failover
kubectl exec $WORKER -n failover-test -c engine-0 -- kill 1
sleep 3
kubectl logs $WORKER -n failover-test -c engine-1 --tail=15 | grep -E "lock|wake|register"

# Test completions again
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":32}'
```

---

## Critical Design Decisions to Be Aware Of

1. **`--load-format gms` is a required user-supplied CLI arg**: The Python validation in `args.py` enforces that `--gms-mode shadow` requires `--load-format gms`. This is NOT injected automatically — users must specify it in their DGD args or on the CLI.

2. **`DYN_VLLM_GMS_MODE=shadow` is injected by the operator**: This env var maps to the `--gms-mode` argparse parameter. The Python code reads it from the env var, not the CLI.

3. **Etcd discovery is mandatory for failover**: The operator forces `DYN_DISCOVERY_BACKEND=etcd` on failover engines and validates that etcd is configured (fail-fast if no `etcdAddress`). Kubernetes-based discovery doesn't support the re-registration pattern needed for failover.

4. **Port staggering avoids conflicts**: Two engines in the same pod need different ports for system, NIXL side channel, and KV event communication.

5. **GMS sidecar is an init container with `restartPolicy: Always`**: This Kubernetes 1.28+ feature makes it a sidecar — it starts before main containers and keeps running alongside them.

6. **`TMPDIR=/shared` is the convention for socket directory**: Both GMS and engine containers mount the same `emptyDir` volume at `/shared`. GMS writes `gms_<UUID>.sock` files there. Engines discover them via `TMPDIR`.

---

## Known Issues

### Engine-1 First-Boot OOM (Unresolved)

On the first deployment, engine-1 may OOM during CUDA graph compilation if engine-0 is simultaneously compiling. This is a transient race condition during peak GPU memory usage. A pod restart resolves it. Root cause: both engines compile CUDA graphs concurrently, and the peak memory during compilation exceeds steady-state usage. **Needs investigation** — may require staggered startup or a memory-aware sequencing mechanism in `patches.py` or `main.py`.

---

## Quick Reference: File Locations

```
# Python failover code
lib/gpu_memory_service/failover_lock/          # Lock interface + flock impl
lib/gpu_memory_service/integrations/vllm/      # patches.py, worker.py, model_loader.py
components/src/dynamo/vllm/main.py             # Shadow mode orchestration
components/src/dynamo/vllm/args.py             # Arg validation
components/src/dynamo/vllm/backend_args.py     # --gms-mode arg definition

# Go operator code
deploy/operator/internal/dynamo/failover.go    # Pod transformation
deploy/operator/internal/dynamo/failover_test.go
deploy/operator/internal/dynamo/graph.go       # Entry point for pod spec gen

# Test scripts (run from repo root)
test_lock_driven_failover.sh                   # Primary: 2-GPU, TP=2, full failover
test_shadow_failover.sh                        # 1-GPU, TP=1, HTTP wake failover
test_shadow_sleep_wake.sh                      # 1-GPU, TP=1, shadow sleep/wake

# Unit tests
tests/fault_tolerance/gpu_memory_service/test_failover_lock.py  # No GPU needed

# Kubernetes manifests
failover-pod-test.yaml                         # Source-of-truth pod spec
examples/backends/vllm/deploy/vllm-agg-failover.yaml  # DGD for operator testing
```
