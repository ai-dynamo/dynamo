# Planner Dev Environment Setup

Development environment for the Dynamo planner component, using a dev pod on a shared Azure AKS cluster. No local Docker or WSL required. This is the single reference for everything from first-time cluster access through day-to-day planner development.

---

## Prerequisites

- `kubectl` (v1.24+) on your Windows machine
- Azure CLI (`az`) logged in (`az login`)
- `kubelogin` installed (for AKS auth)
- A namespace on the dev cluster (e.g., `<your-name>-dynamo-system`)
- A HuggingFace token (for model downloads — [token guide](https://huggingface.co/docs/hub/en/security-tokens))

---

## Cluster Access

```powershell
# Get credentials (one-time)
az aks get-credentials --resource-group <rg> --name <your-aks-cluster> --file "$HOME\.kube\dynamo-kubeconfig"

# Switch to azurecli auth — avoids device code prompts on every kubectl call.
# Must be run once after get-credentials; uses your existing az login session.
kubelogin convert-kubeconfig -l azurecli --kubeconfig "$HOME\.kube\dynamo-kubeconfig"

# Set for your terminal session (use a separate file to avoid stomping ~/.kube/config)
$env:KUBECONFIG="$HOME\.kube\dynamo-kubeconfig"

# Verify
kubectl get nodes
```

---

## Platform Status (Read Before Deploying)

**The Dynamo platform is already installed cluster-wide in `dynamo-system`.** Do not run `helm install` in your own namespace — CRDs and ClusterRoles are cluster-scoped and will conflict. The operator watches all namespaces, so just create your own namespace and deploy DGDs there.

**DGDR (auto-profiling) does not work yet.** The image `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.0.2` referenced in the quickstart docs does not exist on nvcr.io or the cluster ACR. Skip DGDR and deploy a **DGD directly** using the template in the One-Time Setup section below.

**Node disk pressure.** Some CPU nodes run low on ephemeral storage. If a pod is evicted with "low on resource: ephemeral-storage", force-delete the failed pod and the operator will reschedule it to a fresh node.

---

## Working Image Tags

Last verified **2026-05-10** (full from-scratch repro: **589 tests passing**, 0 failures, 4 documented skips):

| Component | Image |
|-----------|-------|
| Worker (vLLM) | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1` |
| Frontend | `dynamoci.azurecr.io/ai-dynamo/dynamo:e637f35dc03b916d95cabfac7ebdc6fab56d078a-dynamo-frontend` |
| Operator | `dynamoci.azurecr.io/ai-dynamo/dynamo:operator-05-02-26-main-882375b` |

To find a newer frontend tag:

```powershell
az acr repository show-tags --name dynamoci --repository ai-dynamo/dynamo `
  --top 30 --orderby time_desc --output tsv | Select-String "frontend"
```

**Why use the frontend image for the dev pod:** The planner is pure Python. It only needs `dynamo._core` (Rust bindings, pre-built in the frontend image), Python deps, and cluster services. The frontend image is ~2 GB vs ~20 GB+ for the full vLLM runtime, avoiding disk pressure on CPU nodes.

---

## One-Time Setup

### 1. Create Your Namespace

```powershell
kubectl create namespace <your-namespace>
```

### 2. Create the HuggingFace Token Secret

```powershell
$env:HF_TOKEN="<your-hf-token>"
kubectl create secret generic hf-token-secret `
  --from-literal=HF_TOKEN="$env:HF_TOKEN" `
  -n <your-namespace>
```

### 3. Create Planner ServiceAccount + RBAC

The planner needs permissions to read/patch DynamoGraphDeployments. The operator installs a ClusterRole for this; you only need a ServiceAccount and RoleBinding in your namespace.

```powershell
kubectl create sa planner-dev-sa -n <your-namespace>

kubectl create rolebinding planner-dev-binding `
  --clusterrole=dynamo-platform-dynamo-operator-planner `
  --serviceaccount=<your-namespace>:planner-dev-sa `
  -n <your-namespace>
```

### 3b. Apply the dev RBAC patch (operator chart ≤ 1.2.0 only)

The shipped `dynamo-platform-dynamo-operator-planner` ClusterRole in operator
charts ≤ 1.2.0 does **not** grant `pods` permissions, so the planner cannot
list worker pods or patch TGP annotations. The live actuation tests
(`test_actuation_knobs_live.py`) and any power-aware deployment will fail
without this patch:

```powershell
kubectl apply -f deploy/planner-pod-rbac-dev.yaml -n <your-namespace>
```

Operator chart 1.2.1+ ships these rules built in
(`deploy/helm/charts/platform/components/operator/templates/planner.yaml`);
delete this Role/RoleBinding once the cluster is upgraded.

### 4. Deploy a DGD (your test workload)

A working DGD spec already lives at `qwen3-quickstart-dgd.yaml` in the repo
root (Qwen3-0.6B, aggregated mode, single GPU). It uses the verified frontend
SHA from the Working Image Tags table. Apply it as-is:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-quickstart
spec:
  services:
    Frontend:
      envFromSecret: hf-token-secret
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: dynamoci.azurecr.io/ai-dynamo/dynamo:<sha>-dynamo-frontend
    VllmWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
        requests:
          custom:
            ephemeral-storage: "5Gi"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1
          workingDir: /workspace/examples/backends/vllm
          command: ["python3", "-m", "dynamo.vllm"]
          args: ["--model", "Qwen/Qwen3-0.6B"]
```

```powershell
kubectl apply -f qwen3-quickstart-dgd.yaml -n <your-namespace>
kubectl get dgd -n <your-namespace> -w   # wait for READY: True (~3-5 min on a fresh node)
```

> **Avoid `qwen3-quickstart.yaml`** if you see one in older branches — it's a
> stale `DynamoGraphDeploymentRequest` (DGDR) referencing
> `nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.0.1`, which does not exist on
> nvcr.io or the cluster ACR. DGDR auto-profiling is broken until that image
> ships.

### 5. Deploy the Dev Pod

The dev pod uses the lightweight frontend image (has `dynamo._core` Rust bindings, Python 3.12, git, pytest). Update the env vars in `dev-pod.yaml` to match your namespace and DGD name:

| Env Var | What to set it to |
|---------|-------------------|
| `DYN_NAMESPACE` | `<your-namespace>-<dgd-name>` (matches DGD namespace prefix) |
| `DYN_NAMESPACE_PREFIX` | Same as `DYN_NAMESPACE` |
| `DYN_PARENT_DGD_K8S_NAME` | Your DGD name (e.g., `qwen3-quickstart`) |
| `DYN_PARENT_DGD_K8S_NAMESPACE` | Your K8s namespace |

```powershell
kubectl apply -f dev-pod.yaml -n <your-namespace>
kubectl wait --for=condition=Ready pod/planner-dev -n <your-namespace> --timeout=60s
```

### 6. Bootstrap the Dev Pod

After the pod starts, clone the repo and install planner dependencies. **All
of these packages are required** to land at the green test pass below —
`pytest-asyncio` in particular is needed for `tests/unit/test_state_machine.py`
async cases (without it, you'll see 2 collection-time failures unrelated to the
planner code):

```powershell
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  cd /workspace &&
  git clone --depth 1 https://github.com/ai-dynamo/dynamo.git repo &&
  uv pip install --python /opt/dynamo/venv/bin/python3 \
    pytest-benchmark pytest-asyncio \
    pmdarima==2.1.1 prometheus-api-client==0.6.0 \
    filterpy==1.4.5 plotly prophet==1.2.1 \
    scikit-learn scipy
"
```

### 7. Push your local working tree (for uncommitted changes)

The dev pod cloned from `main`. To test uncommitted local changes, push them
into the pod. **`kubectl cp` of a directory is unreliable on Windows
PowerShell** (binary stream issues; the copy silently no-ops). Use a tar
roundtrip via a regular file:

```powershell
# Tar the planner module + power_agent + any other touched files
tar -czf planner_push.tar.gz `
  components/src/dynamo/planner `
  components/power_agent `
  components/src/dynamo/profiler/utils/aic_dataframe.py

# Use a relative path so kubectl doesn't interpret the C: drive letter as a remote spec
kubectl cp planner_push.tar.gz <your-namespace>/planner-dev:/tmp/planner_push.tar.gz

# Extract in-pod
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  cd /workspace/repo && tar -xzf /tmp/planner_push.tar.gz && echo OK
"

Remove-Item planner_push.tar.gz
```

---

## Quick Deploy Checklist

For a fresh setup, run these steps in order:

1. `$env:KUBECONFIG="$HOME\.kube\dynamo-kubeconfig"` — set kubeconfig (every terminal session)
2. `kubectl create namespace <your-namespace>`
3. `kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN="<token>" -n <your-namespace>`
4. `kubectl apply -f qwen3-quickstart.yaml -n <your-namespace>`
5. `kubectl get dgd -n <your-namespace>` — wait for `READY: True`
6. `kubectl port-forward svc/qwen3-quickstart-frontend 8000:8000 -n <your-namespace>`

---

## Send a Request

Once the DGD is ready and you have a port-forward running:

```powershell
# Auto-detect the frontend service and port-forward
$svc = kubectl get svc -n <your-namespace> -o name | Select-String "frontend" | Select-Object -First 1
kubectl port-forward $svc 8000:8000 -n <your-namespace>
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is NVIDIA Dynamo?"}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

---

## Day-to-Day Development Workflow

### Edit-Push-Test Cycle

Edit planner code locally in Cursor/VS Code, push changed files to the pod, run tests:

```powershell
# Set kubeconfig (once per terminal session)
$env:KUBECONFIG="$HOME\.kube\dynamo-kubeconfig"

# Push a changed file
kubectl cp components/src/dynamo/planner/<path-to-file> `
  <your-namespace>/planner-dev:/workspace/repo/components/src/dynamo/planner/<path-to-file>

# Push entire planner directory (after larger changes)
kubectl cp components/src/dynamo/planner `
  <your-namespace>/planner-dev:/workspace/repo/components/src/dynamo/planner
```

### Run Unit Tests (Phase 1)

No GPU, no cluster services, no running DGD required. Tests exercise the planner state machine, config validation, Prometheus parsers, K8s connector, etc.

```powershell
# Run all planner unit tests (~10 s)
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/unit/ -v --tb=short
"
```

**Expected (verified 2026-05-10):** `465 passed in ~10 s`.

```powershell
# Run a specific test file
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/unit/test_state_machine.py -v --tb=short
"
```

### Run Power Agent Unit Tests

The Power Agent is a separate component under `components/power_agent/`. Run
its unit tests in-place (no `PYTHONPATH` needed, the `pyproject.toml` is
co-located):

```powershell
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  cd /workspace/repo/components/power_agent &&
  python3 -m pytest tests/ -v --tb=short
"
```

**Expected:** `43 passed in <1 s`.

### Run AIC Power Sim Tests (no cluster services needed)

Pure-simulation tests: mocked AIC responses, no GPU, no live DGD. These
exercise the EMA feedback loop, budget constraints, and the Phase 4 synthetic
multi-power-level path.

```powershell
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/integration/test_aic_power_e2e_sim.py \
    -v --tb=short -m 'pre_merge or integration'
"
```

**Expected:** `15 passed in ~5 s`.

### Run AIC Power Optimizer Integration Tests (no cluster services needed)

Round-trip tests for the AIC optimizer with a mocked AIC estimator. Covers
optimize() happy path, failure handling, EMA gating, and throughput regression
detection.

```powershell
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/integration/test_aic_power_optimizer.py \
    -v --tb=short
"
```

**Expected:** `34 passed in ~5 s`.

### Run Live Cluster Integration Tests (DGD must be Running)

These tests probe the **real** Prometheus, real DCGM, real router `/metrics`,
real K8s API, and real frontend admission endpoint of your running DGD. They
require:

- A `READY=True` DGD in your namespace (the `qwen3-quickstart-dgd.yaml` from
  step 4)
- The dev RBAC patch from step 3b applied (otherwise pod-listing fails)
- Env vars `DYN_PARENT_DGD_K8S_NAME` / `DYN_PARENT_DGD_K8S_NAMESPACE` set on
  the dev pod (already in `dev-pod.yaml`)

```powershell
# Live metric paths (Prometheus, router /metrics, DCGM, MDC reads)
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/integration/test_metric_paths_live.py \
    -v --tb=short
"

# Live actuation knobs (TGP annotations, pod discovery, admission push, advisory mode)
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/integration/test_actuation_knobs_live.py \
    -v --tb=short
"

# Disruptive scaling test (mutates DGD spec — opt in)
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  export RUN_DISRUPTIVE_TESTS=1 &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/integration/test_actuation_knobs_live.py::TestScalingRealMutation \
    -v --tb=short
"
```

**Expected (verified 2026-05-10 against the `qwen3-quickstart` DGD in a single dev namespace):**

| Suite | Pass | Skip | Notes on skips |
|---|---|---|---|
| `test_metric_paths_live.py` | 22 | 3 | (1) frontend metric series empty (no traffic); (2,3) LocalRouter `/metrics` not exposed in non-KV router mode (open-question #14 in the design doc) |
| `test_actuation_knobs_live.py` | 10 | 1 | Scaling test gated by `RUN_DISRUPTIVE_TESTS=1`; passes when enabled |

### Full Test-Suite Sweep (the "tests pass from scratch" bar)

Runs every unit + integration test in one shot, excluding the pre-existing
`test_virtual_connector.py` which has unrelated env requirements:

```powershell
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/ -q --tb=line \
    --ignore=components/src/dynamo/planner/tests/integration/test_virtual_connector.py
"
```

**Expected (verified 2026-05-10):** `546 passed, 4 skipped in ~10 s` (all 4
skips are documented above; 0 failures). With the power agent suite this
totals **589 passing tests** for the full power planner patchset.

### Sanity-check the chat completion path

Quick in-cluster smoke test that the deployment serves requests
(no port-forward needed):

```powershell
kubectl exec -n <your-namespace> planner-dev -- python3 -c @'
import urllib.request, json
body = json.dumps({
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Reply with a single word: hi"}],
    "max_tokens": 10,
}).encode()
req = urllib.request.Request(
    "http://qwen3-quickstart-frontend.<your-namespace>.svc.cluster.local:8000/v1/chat/completions",
    data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    print("Response:", json.load(r)["choices"][0]["message"]["content"][:80])
'@
```

### Run Planner Against Live Cluster (Phase 2)

Runs the planner process inside the dev pod, connected to the cluster's K8s API, NATS, and Prometheus. Use `advisory=True` to log scaling decisions without executing them.

```powershell
# Copy and run the launch script
kubectl cp test_planner_launch.py <your-namespace>/planner-dev:/workspace/test_planner_launch.py

# Run in background (planner is long-lived)
kubectl exec -n <your-namespace> planner-dev -- bash -c "
  python3 /workspace/test_planner_launch.py > /tmp/planner_out.log 2>&1 &
  PID=\$!; echo Started PID=\$PID;
  sleep 20;
  cat /tmp/planner_out.log
"
```

Set `advisory=False` in `test_planner_launch.py` when you want the planner to actually enforce scaling decisions on the DGD workers.

---

## Architecture Notes

### How scaling decisions are enforced

```
PlannerStateMachine (core/state_machine.py)
  -> computes PlannerEffects.scale_to
  -> NativePlannerBase._apply_scale() (core/base.py)
     -> skipped if advisory=True
     -> KubernetesConnector.set_component_replicas()
        -> KubernetesAPI.update_service_replicas()
           -> patches DynamoGraphDeploymentScalingAdapter via K8s API
              -> operator watches and scales worker pods
```

### Cluster services used

| Service | Address |
|---------|---------|
| NATS | `nats://dynamo-platform-nats.dynamo-system.svc.cluster.local:4222` |
| Prometheus | `http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090` |
| K8s API | In-cluster (via ServiceAccount token) |

### Key env vars for the DistributedRuntime

| Env Var | Purpose | Example |
|---------|---------|---------|
| `DYN_DISCOVERY_BACKEND` | Discovery mode | `kubernetes` |
| `DYN_NAMESPACE` | Dynamo namespace (DGD-scoped) | `<ns>-<dgd-name>` |
| `DYN_PARENT_DGD_K8S_NAME` | DGD name | `qwen3-quickstart` |
| `DYN_PARENT_DGD_K8S_NAMESPACE` | K8s namespace | `<your-namespace>` |
| `NATS_SERVER` | NATS connection URL | `nats://...` |
| `POD_NAME` / `POD_UID` | Pod identity (downward API) | auto-injected |

---

## Recovering the Dev Pod

The dev pod uses `restartPolicy: Never` and `sleep infinity`, so it persists until deleted. If it dies or you need a fresh start:

```powershell
kubectl delete pod planner-dev -n <your-namespace> --grace-period=0 --force
kubectl apply -f dev-pod.yaml -n <your-namespace>
# Then re-run the bootstrap step (clone + pip install)
```

---

## Files

| File | Purpose |
|------|---------|
| `dev-pod.yaml` | Dev pod spec with SA, downward API, env vars |
| `qwen3-quickstart-dgd.yaml` | Minimal DGD (Qwen3-0.6B, 1× GPU, agg mode) used by all repro steps |
| `deploy/planner-pod-rbac-dev.yaml` | Dev RBAC patch — `pods` permissions on `planner-dev-sa` (operator chart ≤ 1.2.0 gap) |
| `test_planner_launch.py` | Script to launch planner in advisory mode against the live cluster |
| `test_k8s_access.py` | Quick sanity check for K8s API, Prometheus, and NATS connectivity |
| `examples/deployments/powerplanner/PIPECLEAN.md` | Step-by-step power planner e2e validation runbook |
| `examples/deployments/powerplanner/disagg-power-aware.yaml` | Power-aware disaggregated DGD example (Phase 1+2) |
| `examples/deployments/powerplanner/verify_poweraware.bash` | Phase-2/3 deployment smoke test (Power Agent DS + planner metrics + AIC sanity) |

---

## From-Scratch Repro Script (the canonical "tests pass" recipe)

This is the exact sequence verified on **2026-05-10** that produced
`589 passing, 4 skipped, 0 failures` on a clean namespace. Run it any time you
want to convince yourself the dev environment is healthy.

```powershell
$env:KUBECONFIG="$HOME\.kube\dynamo-kubeconfig"
$ns = "<your-namespace>"

# 1. Cluster-side workload teardown (keep namespace + hf-token-secret)
kubectl delete dgd qwen3-quickstart -n $ns --ignore-not-found
kubectl delete pod planner-dev      -n $ns --grace-period=10 --ignore-not-found
kubectl delete -f deploy/planner-pod-rbac-dev.yaml -n $ns --ignore-not-found
kubectl delete rolebinding planner-dev-binding -n $ns --ignore-not-found
kubectl delete sa planner-dev-sa               -n $ns --ignore-not-found

# 2. Recreate SA + RBAC
kubectl create sa planner-dev-sa -n $ns
kubectl create rolebinding planner-dev-binding `
  --clusterrole=dynamo-platform-dynamo-operator-planner `
  --serviceaccount="${ns}:planner-dev-sa" -n $ns
kubectl apply -f deploy/planner-pod-rbac-dev.yaml -n $ns

# 3. Apply workload + dev pod
kubectl apply -f qwen3-quickstart-dgd.yaml -n $ns
kubectl apply -f dev-pod.yaml              -n $ns

# 4. Wait for DGD ready (3-5 min)
kubectl wait --for=jsonpath='{.status.state}'=successful dgd/qwen3-quickstart -n $ns --timeout=600s

# 5. Bootstrap the dev pod
kubectl exec -n $ns planner-dev -- bash -c "
  cd /workspace &&
  git clone --depth 1 https://github.com/ai-dynamo/dynamo.git repo &&
  uv pip install --python /opt/dynamo/venv/bin/python3 \
    pytest-benchmark pytest-asyncio \
    pmdarima==2.1.1 prometheus-api-client==0.6.0 \
    filterpy==1.4.5 plotly prophet==1.2.1 \
    scikit-learn scipy
"

# 6. Push your local working tree (uncommitted patchset)
tar -czf planner_push.tar.gz `
  components/src/dynamo/planner `
  components/power_agent `
  components/src/dynamo/profiler/utils/aic_dataframe.py
kubectl cp planner_push.tar.gz "${ns}/planner-dev:/tmp/planner_push.tar.gz"
kubectl exec -n $ns planner-dev -- bash -c "cd /workspace/repo && tar -xzf /tmp/planner_push.tar.gz"
Remove-Item planner_push.tar.gz

# 7. Run the full sweep
kubectl exec -n $ns planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/ -q --tb=line \
    --ignore=components/src/dynamo/planner/tests/integration/test_virtual_connector.py
"
kubectl exec -n $ns planner-dev -- bash -c "
  cd /workspace/repo/components/power_agent &&
  python3 -m pytest tests/ -q
"

# 8. Disruptive scaling test (mutates DGD spec briefly, scales back down)
kubectl exec -n $ns planner-dev -- bash -c "
  export PYTHONPATH=/workspace/repo/components/src:/workspace/repo &&
  export RUN_DISRUPTIVE_TESTS=1 &&
  cd /workspace/repo &&
  python3 -m pytest components/src/dynamo/planner/tests/integration/test_actuation_knobs_live.py::TestScalingRealMutation -q
"
```

**Pass criteria:**

| Step | Expected (cold) | Expected (after sanity chat) |
|------|----------|----------|
| 7a (planner sweep) | `546 passed, 4 skipped in ~10 s` | `547 passed, 3 skipped in ~10 s` |
| 7b (power agent) | `43 passed in <1 s` | `43 passed in <1 s` |
| 8 (disruptive scaling) | `1 passed in ~6 s` | `1 passed in ~6 s` |
| **Total** | **590 passing, 4 skipped, 0 failures** | **591 passing, 3 skipped, 0 failures** |

The 1-test difference is `test_frontend_metric_series_exists`, which skips on a
cold deploy and passes after any traffic. Sending the sanity-check chat
completion above lights it up.
