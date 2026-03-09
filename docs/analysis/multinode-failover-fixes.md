# Multinode Failover: Fixes and Findings

Tracking changes needed to make single-node failover work in multinode TP setups.

## Fix 1: worker_cls not propagated to headless workers

**File:** `components/src/dynamo/vllm/main.py` — `run_dynamo_headless()`

**Problem:** `run_dynamo_headless()` calls vLLM's `run_headless()` directly, which
creates a `MultiprocExecutor` that spawns worker processes. These workers use vLLM's
default model loader, not the GMS loader. The `worker_cls` override (set to
`gpu_memory_service.integrations.vllm.worker.GMSWorker` when `--load-format gms`) is
applied in `setup_vllm_engine()` which headless mode bypasses entirely.

**Fix:** Set `config.engine_args.worker_cls` in `run_dynamo_headless()` before calling
`run_headless()`, matching the same logic in `setup_vllm_engine()`.

**Scope:** Affects GMS and ModelExpress load formats on headless workers.

## Fix 2: SHADOW_SKIP_KV_CACHE not propagated to headless workers

**File:** `components/src/dynamo/vllm/main.py` line 437

**Problem:** `os.environ["SHADOW_SKIP_KV_CACHE"] = "1"` is set in the leader process
only. In single-node TP, child worker processes inherit this env var (forked from same
parent). In multinode, the headless worker is a separate independently-started process
that does not inherit the leader's environment.

This causes a mismatch during `collective_rpc("initialize_from_config")`:
- Leader's local workers (rank 0): `_shadow_init_phase=True` -> skip KV cache -> `{}`
- Headless worker (rank 1): `_shadow_init_phase=False` -> allocate KV cache normally

**Workaround:** Pass `SHADOW_SKIP_KV_CACHE=1` in the headless worker's environment
when launching it (test scripts, operator pod spec).

**Future improvement:** Consider propagating shadow mode through `run_dynamo_headless()`
or the engine args namespace, rather than relying on env var inheritance. The operator
would need to inject `SHADOW_SKIP_KV_CACHE=1` into headless worker containers when
the service is in shadow/failover mode.

## Fix 3: --enforce-eager incompatible with shadow mode

Not a bug — by design. Shadow mode requires PIECEWISE CUDA graph mode because attention
ops are stubbed during warm-up (no KV cache at init). `--enforce-eager` forces CUDA
graph mode to NONE, which attempts real attention ops that need KV cache tensors.

Multinode tests must not use `--enforce-eager` with `--gms-mode shadow`.

## Fix 4: GMS socket path ignores CUDA_VISIBLE_DEVICES

**File:** `lib/gpu_memory_service/common/utils.py` — `get_socket_path()`

**Problem:** `get_socket_path()` uses NVML (`pynvml`) to resolve GPU UUIDs. NVML always
enumerates physical devices regardless of `CUDA_VISIBLE_DEVICES`. So when a process
runs with `CUDA_VISIBLE_DEVICES=1`, calling `get_socket_path(0)` (local device 0)
resolves to physical GPU 0's UUID instead of physical GPU 1's UUID.

This breaks multinode-on-same-machine testing: the headless worker (local_rank=0)
connects to the wrong GMS server.

**Fix:** Added `_resolve_physical_device()` that parses `CUDA_VISIBLE_DEVICES` and
remaps the CUDA runtime device index to the physical NVML device index before UUID
lookup. When `CUDA_VISIBLE_DEVICES` is not set, the device index passes through
unchanged (backward compatible).

**Impact for real K8s multinode:** Also beneficial — the NVIDIA device plugin sets
`CUDA_VISIBLE_DEVICES` in containers, so GMS socket resolution now correctly follows
the device mapping the container runtime provides.

## Architecture Notes

### GMS per-GPU in multinode
Each GMS server is bound to exactly one GPU via UUID-based socket path. For multinode
TP=2 across 2 GPUs, need 2 GMS servers (one per GPU). Clients auto-discover via
`get_socket_path(device)` which uses CUDA device enumeration (respects
CUDA_VISIBLE_DEVICES remapping).

### collective_rpc propagation
vLLM's `collective_rpc("sleep")` and `collective_rpc("wake_up")` propagate to headless
workers via `create_mq_broadcaster` for `nnodes > 1`. Verified working — both ranks
sleep/wake atomically.

## Fix 5: CUDA graph mode escalation causes assertion in multinode shadow

**File:** `vllm/v1/worker/gpu_model_runner.py` (installed vLLM)

**Problem:** vLLM resolves cudagraph_mode based on attention backend capabilities.
When the backend supports full decode graphs, vLLM escalates PIECEWISE to
FULL_AND_PIECEWISE (lines 5485-5490). This creates two capture rounds:

1. PIECEWISE (works — attention ops excluded from graph, no KV cache needed)
2. FULL (fails — calls `_build_attention_metadata` which asserts `slot_mappings is not None`)

The shadow patch `_get_slot_mappings` returns `(None, None)` when KV caches are empty.
FULL mode calls `_build_attention_metadata` with this None value → assertion crash.

**Why single-node passes:** In single-process TP=2, both workers are child processes
of the same `MultiprocExecutor`. The leader's shadow mode forces PIECEWISE on the
`compilation_config` BEFORE the engine creates workers. All workers inherit PIECEWISE
and the backend resolution doesn't escalate (it reads the already-set mode).

In multinode MP, the headless worker parses its own config independently. The
`run_dynamo_headless` override sets PIECEWISE in `compilation_config`, but vLLM's
backend resolution at `initialize_attn_backend` (called later during worker init)
escalates it to FULL_AND_PIECEWISE based on hardware support. This happens AFTER
our config override.

**Verified via debug prints:**
```
Leader (Worker_TP0): cudagraph_mode=PIECEWISE     → 51 PIECEWISE captures
Worker (Worker_TP1): cudagraph_mode=FULL_AND_PIECEWISE → 51 PIECEWISE + 35 FULL captures
```

The FULL captures on the worker trigger the slot_mappings assertion. The leader
(only PIECEWISE) finishes its captures and moves on, causing NCCL timeout when
the worker tries FULL captures alone.

**Fix:** Added a clamp in `gpu_model_runner.py` right before `initialize_cudagraph_keys`
that checks `SHADOW_SKIP_KV_CACHE=1` and empty `kv_caches`, clamping any mode
above PIECEWISE back to PIECEWISE. This prevents the dispatcher from generating
FULL capture descriptors when KV caches don't exist.

```python
if os.environ.get("SHADOW_SKIP_KV_CACHE") == "1" and not self.kv_caches:
    if cudagraph_mode not in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE):
        cudagraph_mode = CUDAGraphMode.PIECEWISE
```

**Note:** This is currently applied as a hotpatch to the installed vLLM. For
production, this should be either:
1. Moved to a proper GMS patch in `patches.py` that monkey-patches the resolution
2. Upstreamed to vLLM as a shadow/sleep mode consideration
3. Made part of the engine config propagation from leader to headless workers

### Shadow mode config propagation to headless workers
The leader's `setup_vllm_engine()` overrides several settings for shadow mode:
- `SHADOW_SKIP_KV_CACHE=1` (env var, Fix 2)
- `cudagraph_mode → PIECEWISE` (compilation_config override)
- `worker_cls → GMSWorker` (Fix 1)

None of these propagate to headless workers automatically. The headless worker parses
its own CLI args independently. If the headless worker receives different compilation
config (e.g., `cudagraph_mode=none` while leader uses PIECEWISE), the two ranks
desynchronize during `compile_or_warm_up_model` — one does CUDA graph capture, the
other does model warmup, and they deadlock on mismatched NCCL collectives.

**Rule:** Headless workers must receive the same `--compilation-config` as the leader,
or no explicit config (let both use defaults). Shadow-specific overrides should be
propagated through `run_dynamo_headless()` or applied at the headless worker level.

### Failover lock scope
The flock-based lock is a local POSIX advisory lock on a shared file. For multinode,
the lock file must be accessible to all leader processes (shared filesystem or same
machine). The lock only coordinates leaders — headless workers follow their leader's
collective_rpc commands.
