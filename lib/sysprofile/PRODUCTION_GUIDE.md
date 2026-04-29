# sysprofile Production Integration Guide

## Architecture Overview

```
Dynamo Cluster (dynamo-system namespace)
+--------------------------------------------------+
|  frontend-0    router-0    engine-prefill-0  ...  |
|    |              |              |                 |
|    +-- writes .pftrace.gz to shared PVC ----------+
|                                                    |
|  /data/sysprofile/<run_id>/                        |
|    frontend.pftrace.gz                             |
|    router.pftrace.gz                               |
|    engine-prefill-0.pftrace.gz                     |
|    engine-prefill-1.pftrace.gz                     |
|    engine-decode-0.pftrace.gz                      |
+--------------------------------------------------+
                      |
            sysprofile-analyze Job
                      |
            merge_result.json, gpu_util.json,
            kernels.json, comm.json, report.html
```

Each Dynamo component writes its own `.pftrace.gz` trace file to a shared
volume. A post-capture k8s Job runs the merge + analyze + report pipeline.
All output files persist on the PVC after the Job completes.

---

## DynamoGraphDeployment (DGD) Integration

This is the recommended way to enable sysprofile on a real Dynamo deployment.
The DGD CRD supports `envs`, `pvcs`, and `volumeMounts` natively — no need
to patch raw Deployments.

### Complete DGD Manifest with sysprofile

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: llm-serving-profiled
  namespace: dynamo-system
spec:
  backendFramework: vllm

  # ── Top-level PVC reference (shared by all services) ──
  pvcs:
    - name: profiling-output-gemma4     # existing PVC in dynamo-system
    # To create a fresh PVC instead:
    # - name: profiling-output-mymodel
    #   create: true
    #   storageClass: nfs-csi
    #   size: 10Gi
    #   volumeAccessMode: ReadWriteMany

  # ── Global env vars (applied to ALL services) ──
  envs:
    - name: DYN_SYSPROFILE_ENABLE
      value: "1"
    - name: DYN_SYSPROFILE_DIR
      value: "/data/sysprofile"
    - name: DYN_SYSPROFILE_RUN_ID
      value: "bench-001"
    - name: DYN_SYSPROFILE_SAMPLING
      value: "1.0"                       # 100% for benchmarks, 0.01-0.10 for production
    - name: DYN_SYSPROFILE_BACKENDS
      value: "vllm"

  services:
    # ── Frontend ──
    frontend:
      envFromSecret: hf-token-secret
      componentType: frontend
      replicas: 1
      volumeMounts:
        - name: profiling-output-gemma4
          mountPoint: /data/sysprofile
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest

    # ── Router ──
    router:
      componentType: main
      replicas: 1
      volumeMounts:
        - name: profiling-output-gemma4
          mountPoint: /data/sysprofile
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest

    # ── Prefill Engine ──
    engine-prefill:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: prefill
      replicas: 2
      resources:
        limits:
          gpu: "1"
      volumeMounts:
        - name: profiling-output-gemma4
          mountPoint: /data/sysprofile
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest
          command: ["python3", "-m", "dynamo.vllm"]
          args: ["--model", "google/gemma-3-4b-it", "--tensor-parallel-size", "1"]

    # ── Decode Engine ──
    engine-decode:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: decode
      replicas: 1
      resources:
        limits:
          gpu: "1"
      volumeMounts:
        - name: profiling-output-gemma4
          mountPoint: /data/sysprofile
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest
          command: ["python3", "-m", "dynamo.vllm"]
          args: ["--model", "google/gemma-3-4b-it", "--tensor-parallel-size", "1"]
```

### How DGD Fields Map to sysprofile

| DGD Field | Where | Purpose |
|-----------|-------|---------|
| `spec.pvcs[].name` | Top-level | References existing PVC or creates new one. Must be `nfs-csi` (RWX). |
| `spec.pvcs[].create` | Top-level | Set `true` to have the operator create the PVC. |
| `spec.envs[]` | Top-level | Global env vars injected into **all** service pods. |
| `services.<name>.envs[]` | Per-service | Service-specific env overrides (take precedence over global). |
| `services.<name>.volumeMounts[]` | Per-service | Mounts a PVC defined in `spec.pvcs`. Every service needs this. |
| `services.<name>.envFromSecret` | Per-service | References a Secret for HF tokens etc. |

### Enabling/Disabling per Service

Use per-service `envs` to override the global setting. For example, to
profile only engines and skip the frontend:

```yaml
spec:
  envs:
    - name: DYN_SYSPROFILE_ENABLE
      value: "1"

  services:
    frontend:
      envs:
        - name: DYN_SYSPROFILE_ENABLE
          value: "0"                      # disable on frontend
      # ... rest of frontend config
```

### Quick Enable: Adding sysprofile to an Existing DGD

If you already have a DGD running, add these three things:

1. **PVC reference** in `spec.pvcs`:
   ```yaml
   pvcs:
     - name: profiling-output-gemma4
   ```

2. **Global env vars** in `spec.envs`:
   ```yaml
   envs:
     - name: DYN_SYSPROFILE_ENABLE
       value: "1"
     - name: DYN_SYSPROFILE_RUN_ID
       value: "bench-001"
   ```

3. **Volume mount** on every service:
   ```yaml
   services:
     frontend:
       volumeMounts:
         - name: profiling-output-gemma4
           mountPoint: /data/sysprofile
     router:
       volumeMounts:
         - name: profiling-output-gemma4
           mountPoint: /data/sysprofile
     # ... same for all engine services
   ```

Then `kubectl apply` the updated DGD. The operator will rolling-restart all
pods with the new env vars and volume mounts.

### Using a Secret for sysprofile Config

For frequently changing values (run ID, sampling rate), you can use a Secret
instead of hardcoding in the DGD:

```bash
kubectl create secret generic sysprofile-config -n dynamo-system \
  --from-literal=DYN_SYSPROFILE_ENABLE=1 \
  --from-literal=DYN_SYSPROFILE_RUN_ID=bench-$(date +%s) \
  --from-literal=DYN_SYSPROFILE_SAMPLING=1.0 \
  --from-literal=DYN_SYSPROFILE_BACKENDS=vllm
```

Then reference it in the DGD (applies to individual services):

```yaml
services:
  engine-prefill:
    envFromSecret: sysprofile-config
    volumeMounts:
      - name: profiling-output-gemma4
        mountPoint: /data/sysprofile
```

To start a new capture, update the secret and restart:

```bash
kubectl create secret generic sysprofile-config -n dynamo-system \
  --from-literal=DYN_SYSPROFILE_ENABLE=1 \
  --from-literal=DYN_SYSPROFILE_RUN_ID=bench-002 \
  --from-literal=DYN_SYSPROFILE_SAMPLING=1.0 \
  --from-literal=DYN_SYSPROFILE_BACKENDS=vllm \
  --dry-run=client -o yaml | kubectl apply -f -

# Trigger restart by changing the restart ID
kubectl patch dgd llm-serving-profiled -n dynamo-system --type merge \
  -p '{"spec":{"restart":{"id":"bench-002"}}}'
```

---

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_SYSPROFILE_ENABLE` | `0` | Master switch. Accepts `1`, `true`, `yes`, `on`. When off, all hooks are no-ops with zero overhead. |
| `DYN_SYSPROFILE_DIR` | `/data/sysprofile` | Base directory for trace output. Must be a shared volume (PVC). |
| `DYN_SYSPROFILE_SAMPLING` | `0.10` | Fraction of requests to trace (0.0 to 1.0). Clamped to [0, 1]. Deterministic per trace ID (same request always sampled or not). Use `1.0` for benchmarks, `0.01` for production. |
| `DYN_SYSPROFILE_RUN_ID` | auto UUID | Groups all traces from one capture session. **All pods must share the same value.** |
| `DYN_SYSPROFILE_BACKENDS` | `vllm` | Comma-separated list of engine backends to profile: `vllm`, `sglang`, `trtllm`. |
| `DYN_SYSPROFILE_FLUSH_TIMEOUT_S` | `900` | Max seconds to wait for trace flush on graceful shutdown. |

### Sampling Recommendations

| Scenario | `DYN_SYSPROFILE_SAMPLING` | Notes |
|----------|--------------------------|-------|
| Short benchmark (<5min) | `1.0` | Capture everything |
| Long benchmark (>30min) | `0.10` | 10% keeps trace files manageable |
| Production canary | `0.01` | 1% — minimal overhead |
| Production disabled | `0` or `DYN_SYSPROFILE_ENABLE=0` | Zero overhead, hooks compile to no-ops |

---

## Shared PVC for Trace Data

Use the existing `profiling-output-gemma4` PVC (5Gi, `nfs-csi`, RWX) in
`dynamo-system`. This is already provisioned and bound.

Existing PVCs in `dynamo-system`:
| Name | Storage class | Size | Purpose |
|------|---------------|------|---------|
| `dynamo-platform-nats-js-dynamo-platform-nats-0` | `nfs-csi` | 10Gi | NATS JetStream |
| `model-cache` | `ibm-spectrum-scale` | 80Gi | Model weights |
| `profiling-output-gemma4` | `nfs-csi` | 5Gi | **Profiling output (use this)** |

If you need a fresh PVC for a different model/run, create one with `nfs-csi`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: profiling-output-<model>
  namespace: dynamo-system
spec:
  accessModes: [ReadWriteMany]
  storageClassName: nfs-csi
  resources:
    requests:
      storage: 10Gi
```

Or use `spec.pvcs[].create: true` in the DGD (see above).

---

## Build the Profiler Container Image

```dockerfile
# Dockerfile.sysprofile
FROM rust:1.82-bookworm AS rust-builder
WORKDIR /build
COPY . .
RUN cargo build --release -p dynamo-sysprofile

FROM python:3.12-slim-bookworm
WORKDIR /app

# Rust binaries
COPY --from=rust-builder /build/target/release/dynamo-sysprofile-merge /usr/local/bin/
COPY --from=rust-builder /build/target/release/dynamo-sysprofile-demo /usr/local/bin/

# Python profiler
COPY lib/profiler/ /app/lib/profiler/
ENV PYTHONPATH=/app/lib/profiler

ENTRYPOINT ["python3", "-m", "dynamo_profiler"]
```

Build and push:

```bash
docker build -f Dockerfile.sysprofile -t your-registry/dynamo-sysprofile:latest .
docker push your-registry/dynamo-sysprofile:latest
```

---

## Deploy the Report Viewer (one-time)

A persistent Deployment + NodePort serves the latest report from the PVC.
Deploy once — it auto-serves whichever report was last generated.

```bash
kubectl apply -f deploy/sysprofile/viewer-deployment.yaml -n dynamo-system
```

This creates:
- **Deployment** `sysprofile-viewer` — Python HTTP server mounting the profiling PVC (read-only)
- **NodePort Service** on port **30090** — accessible at `http://<any-node-ip>:30090/report.html`

The viewer automatically finds the most recent `report.html` on the PVC.
No restart needed after new reports are generated.

---

## Run the Analyzer Job

After your benchmark completes and all components have flushed their traces:

```bash
# Delete previous job if exists
kubectl delete job sysprofile-analyze -n dynamo-system --ignore-not-found

# Run analyzer
kubectl apply -f deploy/sysprofile/analyze-job.yaml -n dynamo-system

# Watch progress
kubectl logs -f -n dynamo-system job/sysprofile-analyze
```

The Job merges traces, runs all analyzers, and writes the report to the PVC.
Once it completes, the viewer at **`http://<node-ip>:30090/report.html`**
immediately shows the updated report.

### Output files (persist on PVC)

| File | Contents |
|------|----------|
| `merge_result.json` | Per-request latency breakdown, critical path, component stats |
| `gpu_util.json` | Per-GPU busy/idle ratio, stage attribution, idle gap analysis |
| `kernels.json` | Top CUDA kernels by time, TP imbalance (CV%) |
| `comm.json` | NCCL/NiXL bandwidth, NATS propagation latency |
| `report.html` | Self-contained HTML report with interactive Plotly charts |

The Job YAML is at `deploy/sysprofile/analyze-job.yaml`. Edit the `RUN_DIR`
and `--title` values to match your benchmark run.

---

## Quick Reference: CLI Commands

```bash
# Full pipeline (merge + analyze + report + serve)
python3 -m dynamo_profiler profile /data/sysprofile/<run_id> [--port 8080] [--open] [--title "..."]

# With demo traces (for testing the pipeline)
python3 -m dynamo_profiler profile /tmp/demo --demo --open

# Individual steps
dynamo-sysprofile-merge /data/sysprofile/<run_id>
python3 -m dynamo_profiler analyze /data/sysprofile/<run_id>
python3 -m dynamo_profiler report /data/sysprofile/<run_id> --title "..."

# Convert nsys exports to pftrace (for nsys-based workflows)
python3 -m dynamo_profiler nsys-convert \
  --sqlite /path/to/nsys_export.sqlite \
  --component engine-prefill-0 \
  --host node-0 \
  --output /data/sysprofile/<run_id>/engine-prefill-0.pftrace.gz
```

---

## NATS Control Plane (programmatic capture)

Components subscribe to NATS subjects for remote capture control.
Your cluster already has NATS running (`dynamo-platform-nats-js-dynamo-platform-nats-0` PVC).

```
dynamo.sysprofile.start   ->  StartRequest { run_id, duration_s, sampling, ... }
dynamo.sysprofile.stop    ->  StopRequest  { run_id }
dynamo.sysprofile.status  ->  StatusReply  { state, component, host, ... }
```

To trigger a capture from another pod or the operator:

```bash
# Publish start (all components begin capturing)
nats pub dynamo.sysprofile.start \
  '{"run_id":"bench-002","duration_s":300,"sampling":1.0,"backends":["vllm"],"output_dir":"/data/sysprofile","cupti":false,"nsys":false}'

# Check status
nats req dynamo.sysprofile.status '' --replies=10

# Stop early
nats pub dynamo.sysprofile.stop '{"run_id":"bench-002"}'
```

---

## Sizing Guide

| Metric | Estimate |
|--------|----------|
| Trace size per component | ~1MB per 1000 traced requests |
| Merge time | ~2s for 5 components, 10K requests |
| Analyze time | ~5s for 5 trace files |
| Report generation | <1s |
| PVC recommendation | 5-10Gi per model profile run |
| Memory for analyzer Job | 512Mi request, 2Gi limit |
| CPU for analyzer Job | 500m request, 2 limit |

---

## Troubleshooting

**Empty trace files**: Check `DYN_SYSPROFILE_ENABLE=1` is set and pods have
write access to the PVC. Check pod logs for `sysprofile enabled`.

**Missing components in merge**: Each component must share the same `RUN_ID`
and write to the same directory on the shared PVC.

**GPU/kernel sections empty**: Ensure the engine backends have nsys or CUPTI
integration enabled. For vLLM, the adapter captures GPU kernel events via
torch profiler hooks. Without backend integration, only orchestration-plane
events (frontend/router/transport) are captured.

**NATS pairs show 0**: All components must write to the same shared PVC
directory. The analyzer correlates pub/sub events by `msg_id` across files.

**High overhead**: Reduce `DYN_SYSPROFILE_SAMPLING` to `0.01` (1%) for
production. At `0.0` or with `DYN_SYSPROFILE_ENABLE=0`, overhead is zero
(all hooks compile to no-ops).

---

## End-to-End Workflow

```bash
# 1. Deploy viewer (one-time)
kubectl apply -f deploy/sysprofile/viewer-deployment.yaml -n dynamo-system

# 2. Apply DGD with sysprofile enabled (see DGD manifest above)
kubectl apply -f my-dgd-with-profiling.yaml -n dynamo-system

# 3. Wait for pods to come up
kubectl get pods -n dynamo-system -l app.kubernetes.io/managed-by=dynamo-operator -w

# 4. Run your benchmark / load test
# ... (e.g. genai-perf, locust, custom benchmark)

# 5. Wait for traces to flush (pods write on shutdown or after flush timeout)
# Scale down to flush: kubectl scale dgd llm-serving-profiled -n dynamo-system --replicas=0

# 6. Verify trace files exist
kubectl exec -n dynamo-system deploy/sysprofile-viewer -- ls -la /data/sysprofile/bench-001/

# 7. Run analyzer Job
kubectl delete job sysprofile-analyze -n dynamo-system --ignore-not-found
kubectl apply -f deploy/sysprofile/analyze-job.yaml -n dynamo-system
kubectl logs -f -n dynamo-system job/sysprofile-analyze

# 8. View report
echo "Report: http://<any-node-ip>:30090/report.html"

# 9. (Optional) Disable profiling for next run
kubectl patch dgd llm-serving-profiled -n dynamo-system --type merge \
  -p '{"spec":{"envs":[{"name":"DYN_SYSPROFILE_ENABLE","value":"0"}]}}'
```
