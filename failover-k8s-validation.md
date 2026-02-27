# Failover K8s Validation

Iterative validation of the failover pod spec on `dynamo-exp` cluster (DRA enabled, A100 nodes).

## Operator Fixes Needed

Fixes discovered during validation that need to be applied to `failover.go` before the operator-level test.

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `failover.go` | GMS CLI is `--device <int>` + `--socket-path <path>`, not `--socket-dir` / `--devices` | Change `buildGMSSidecar` to use a bash wrapper: launch N subprocesses with `--device $dev --socket-path /shared/gms_${dev}.sock`, use `wait -n` to exit if any child dies |
| 2 | `failover.go` | `resource.k8s.io/v1beta1` → cluster uses GA `resource.k8s.io/v1`; `ResourceClaimTemplate` schema uses `exactly.deviceClassName` + `exactly.allocationMode: ExactCount` | Update DRA API version and claim template schema |
| 3 | `failover.go` | GMS and engine discover sockets via `get_socket_path(device)` using GPU UUID + `tempfile.gettempdir()` — containers don't share `/tmp` | Set `TMPDIR=/shared` on GMS sidecar and engine containers; let both use default UUID-based socket resolution. Startup probe checks `ls /shared/gms_*.sock | wc -l >= N` |
| 4 | `failover.go` | Missing GPU toleration `nvidia.com/gpu: NoSchedule` | Add toleration to generated pod spec |
| 5 | `main.py` | Top-level `from gpu_memory_service.failover_lock.flock import FlockFailoverLock` crashes non-shadow engines if module not installed | Move import inside `if config.gms_mode == "shadow"` branch (lazy import) — **FIXED in 88fb83c** |
| 6 | `setup.py` | `gpu_memory_service.failover_lock` package not listed in explicit `packages` / `package_dir` | Add `failover_lock` and `failover_lock.flock` entries — **FIXED in 88fb83c** |
| 7 | `failover.go` | Failover engines don't need KV events / prefix caching. Default `DYN_EVENT_PLANE=nats` + auto-enabled KV events triggers NATS connection | Inject `--no-enable-prefix-caching` into failover engine args (or set `DYN_EVENT_PLANE=zmq`) |

## Epoch Results

### Epoch 1 — DRA + single-device GMS (PASSED)

**Criteria:**
1. `ResourceClaimTemplate` provisions 1 GPU via DRA
2. GMS sidecar init container creates `/shared/gms_0.sock`
3. Startup probe gates main container correctly
4. Both containers see the GPU via shared DRA claim

**Result:** All passed. GPU visible as `NVIDIA A100-SXM4-80GB`, socket created in ~1s, validator started after sidecar ready.

**Pod spec:** `failover-pod-test.yaml` (epoch 1 revision)

### Epoch 2 — 2-device GMS via wrapper container

**Criteria:**
1. `ResourceClaimTemplate` provisions 2 GPUs via DRA
2. Single GMS container launches 2 GMS subprocesses (device 0, device 1)
3. Any subprocess death kills the container (triggers pod restart)
4. Startup probe gates on BOTH sockets being ready
5. Validator container sees both GPUs

**Result:** All passed. 2x A100-SXM4-80GB visible, both GMS processes initialized, both sockets created, validator gated correctly. Bash wrapper with `wait -n` is clean and simple.

**Pod spec:** `failover-pod-test.yaml` (epoch 2 revision)

### Epoch 3 — Single engine with GMS weights + etcd discovery (PASSED)

**Criteria:**
1. GMS sidecar starts both device 0 and 1
2. Engine loads weights via GMS (`--load-format gms`)
3. Engine registers with etcd-based discovery
4. Engine startup/liveness/readiness probes all pass
5. Endpoints registered (`generate`, `clear_kv_blocks`)

**Result:** All passed. 2/2 Running, 0 restarts. Weights loaded via GMS, etcd shows registered instances under `v1/instances/default_failover-test/backend/generate/`. Health probes return `{"status":"ready"}`. TCP request plane bound on dynamic port.

**Key discoveries:**
- `DYN_EVENT_PLANE` defaults to `nats`, and KV events auto-enable when prefix caching is on (vLLM default). This triggers a NATS connection. Fix: `--no-enable-prefix-caching` disables KV events entirely → no NATS needed.
- `TMPDIR=/shared` on both GMS and engine makes UUID-based socket discovery work across containers.
- `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` and `DYN_SYSTEM_ENABLED` are deprecated but harmless.

**Pod spec:** `failover-pod-test.yaml` (epoch 3 revision)
**Image:** `failover-m6-88fb83c-vllm-runtime`
