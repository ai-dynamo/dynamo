## disagg-bench — KVBM Disaggregated Serving Benchmark

AIPerf latency/throughput benchmark for the KVBM 1P+1D disaggregated vLLM stack.
Sibling to disagg-smoke (R1+R2 functional validation); this skill measures steady-state
throughput and TTFT/ITL latency under a realistic ISL distribution.

### Scripts

| Script | Purpose |
|---|---|
| `lognormal-bench.sh` | Launch 1P+1D, wait for health, run AIPerf |
| `generate-lognormal-dataset.py` | Generate lognormal ISL JSONL for aiperf custom_file |

### Workload shape

The default ISL of 2048 tokens is the p50 of a lognormal(μ=7.0, σ=1.5) distribution
clamped to [256, 8192] — the same shape as SOLBench (mean/median ratio ~9.3x). Full
distribution can be generated with `generate-lognormal-dataset.py`.

### Usage (inside Pyxis container on dlcluster)

```bash
# Build KVBM first (from inside srun Pyxis session)
source .claude/skills/disagg-bringup/env.sh
source $KVBM_VENV/bin/activate
cd lib/bindings/kvbm && maturin develop --uv && cd -
cargo build -p kvbm-hub --bin kvbm_hub
cp $CARGO_TARGET_DIR/debug/deps/libkvbm_kernels.so lib/bindings/kvbm/python/kvbm/

# Download model once
export HF_HOME=/tmp/hf_cache
hf download Qwen/Qwen3-8B --local-dir $HF_HOME/qwen3-8b

# Run benchmark
export KVBM_BENCH_MODEL_DIR=$HF_HOME/qwen3-8b
bash .claude/skills/disagg-bench/lognormal-bench.sh
```

### Key config facts (hard-learned)

- `kv_role` must be `"kv_both"` for **both** prefill and decode — role distinction
  is in `kv_connector_extra_config.leader.disagg.role` (`"prefill"` / `"decode"`),
  not in `kv_role`. Using `"kv_prefill"` raises a pydantic validation error.
- `kv_connector_module_path` is **required** — without it vLLM can't find
  `DynamoConnector` by class name alone. Value: `"kvbm.v2.vllm.schedulers.connector"`.
- `aiperf profile` (not `benchmark`) is the correct subcommand.
- `--model-names` is **required** by aiperf — omitting it errors.
- `--num-prompts` controls request count (not `--num-requests`).
- `--osl` without `--extra-inputs ignore_eos:true` is advisory only (model may stop early).

### Cache configuration: Pyxis vs Docker

This benchmark uses `cache: {host: {cache_size_gb: N}}` which relies on UCX/POSIX NIXL plugins.
These plugins **are available in Pyxis/srun containers** (injected from the host at runtime) but
**are NOT available in plain Docker** on dlcluster. If you run this outside Pyxis:

- Use `cache: {device: {}}` instead (CUDA IPC, no UCX required)
- Remove explicit `nixl.backends` from the worker config
- Set `NIXL_PLUGIN_DIR` and `LD_LIBRARY_PATH` to the wheel's NIXL (see `launch-kvbm-docker.sh`)

### dlcluster-specific traps

On dlcluster H100x8 nodes (`viking-prod-*`, `4u8g-gen-*`):
- Use **Pyxis srun** (`srun --jobid=... --container-name=...`), NOT Docker directly
- Docker mode has NFS root_squash issues that prevent build artifacts from landing
- The Pyxis container shares `/tmp` with the host node, so cargo target persists
- HF cache goes to `/tmp/hf_cache`; model download is ~1min for Qwen3-8B (14GB)
