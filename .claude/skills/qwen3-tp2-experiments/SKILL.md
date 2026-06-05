---
name: qwen3-tp2-experiments
description: Qwen3-32B KVBM experiments on a 4xGB200 box — NUMA-aligned TP=2 instances for remote-search and conditional-disagg smokes, plus an asymmetric prefill TP=1 + decode TP=2 disagg run. Block size 64, 128k context, G1/GPU prefix caching off.
---

## What this is

A heavier-weight variant of the existing KVBM smoke skills (`remote-search-smoke`, `disagg-smoke`) tuned for:

- **Model**: `Qwen/Qwen3-32B` (download at `/scratch/ryan/Qwen3-32B`).
- **Block size 64**, **max-model-len 131072** (128k).
- **TP=2** instances NUMA-aligned: GPUs `0,1`→NUMA 0; GPUs `2,3`→NUMA 1.
- G1/GPU prefix caching **off** in vLLM on every instance.
- Asymmetric run: prefill TP=1, decode TP=2.

## Scenarios

| # | Script | Hub features | Block layout | Port 8000 | Port 8001 |
|---|---|---|---|---|---|
| 1 | `scenario-remote-search-uniform-tp2.sh` | `indexer,p2p` | operational | TP=2 GPUs 0,1 NUMA 0 (instance A) | TP=2 GPUs 2,3 NUMA 1 (instance B) |
| 2 | `scenario-disagg-uniform-tp2.sh`        | `disagg`      | operational | TP=2 GPUs 0,1 NUMA 0 (role=prefill) | TP=2 GPUs 2,3 NUMA 1 (role=decode) |
| 3 | `scenario-disagg-asymmetric.sh`         | `disagg`      | **universal** | TP=1 GPU 2 NUMA 1 (role=prefill)    | TP=2 GPUs 0,1 NUMA 0 (role=decode) |

### Why scenario 3 forces Universal layout

The hub's P2P layout-compat check (`lib/kvbm-protocols/src/control/layout_compat.rs:153-176`) enforces `baseline.tp_size == candidate.tp_size` in Operational mode. With mismatched TP between prefill and decode, the second registrant is rejected with `operational tp_size differs (baseline=N, candidate=M)` and EngineCore aborts with the (chain-stripped) anyhow context `disagg P2P foundation wiring failed`. Universal mode compares canonical (whole-model) shape only, so asymmetric peers can coexist — G1 stays its native vLLM Operational HND, G2 (the cross-instance transfer plane) becomes Universal. The asymmetric orchestrator exports `KVBM_BLOCK_LAYOUT=universal` automatically.

Each orchestrator does: teardown stale → build hub+ctl → start hub → launch the two vLLM instances → poll `/v1/models` → reset cache tiers → run the matching smoke.

## Smokes invoked

- Scenario 1 → `remote-search-smoke.sh`: R→A; verify A's blocks land in hub index; same R→B; assert B drives a remote search → opens session on A → RDMA-pulls blocks → onboards → decodes. Greedy output of B must match golden output of A.
- Scenarios 2 & 3 → `two-request-smoke.sh`: R1 cold; reset prefill G2; R2 with longer prompt; assert prefill matched ≥3 blocks worth on R2 and the CD GNMT path fired.

## Prerequisites

- A Python venv with both `vllm` AND `kvbm` importable. The container's `/opt/dynamo/venv` works (vllm via system site-packages, kvbm editable from this repo). Override with `KVBM_VENV=...` if needed.
- `numactl` available (used to pin each vLLM to its NUMA node).
- 4×GB200 (or equivalent 4-GPU 2-NUMA box) — verified at runtime via `nvidia-smi topo -m`.
- Model weights at `/scratch/ryan/Qwen3-32B` (or set `KVBM_MODEL`).

## Run

```bash
bash .claude/skills/qwen3-tp2-experiments/build-deps.sh
bash .claude/skills/qwen3-tp2-experiments/scenario-remote-search-uniform-tp2.sh
bash .claude/skills/qwen3-tp2-experiments/scenario-disagg-uniform-tp2.sh
bash .claude/skills/qwen3-tp2-experiments/scenario-disagg-asymmetric.sh
```

Logs land in `/tmp/kvbm-experiments/<ts>-<label>/{hub,instance_a,instance_b,prefill,decode}.log`.

## Honored env vars (override any with `KEY=val bash scenario-...`)

| Var | Default | Notes |
|---|---|---|
| `KVBM_VENV` | `/opt/dynamo/venv` | Must have `import vllm` and `import kvbm` both work. |
| `KVBM_MODEL` | `/scratch/ryan/Qwen3-32B` | Path or HF id. |
| `KVBM_HUB_BLOCK_SIZE` | `64` | Hub primary; rendered into vllm `--block-size`. |
| `KVBM_HUB_MAX_SEQ_LEN` | `131072` | Must be multiple of block size. |
| `KVBM_HUB_G2_MEMORY_GIB` | `16` | Advisory G2 size; 128k contexts want headroom. |
| `KVBM_TP2_GPU_MEMUTIL` | `0.85` | TP=2 instances. |
| `KVBM_TP1_GPU_MEMUTIL` | `0.85` | Asymmetric prefill. |
| `KVBM_MAX_NUM_SEQS` | `4` | Conservative; raise after smoke passes. |
| `KVBM_BLOCK_LAYOUT` | `operational` | `operational` or `universal`. |
| `KVBM_VLLM_READY_TIMEOUT` | `900` | 32B+TP=2 NCCL+weight load is slow. |

## Files

```
SKILL.md                                # this file
env.sh                                  # shared defaults
numa-lib.sh                             # GPU↔NUMA map + verify_numa_topology
build-deps.sh                           # cargo build --bin kvbm_hub --bin kvbmctl
start-hub-remote-search.sh              # hub --features indexer,p2p
start-hub-disagg.sh                     # hub --features disagg (+ CD prefill dispatcher)
launch-instance.sh                      # NUMA-pinned vLLM launcher (TP=1 or TP=2; --role optional)
remote-search-smoke.sh                  # mirrors remote-search-smoke patterns; bigger prompt; bs=64
two-request-smoke.sh                    # mirrors disagg-smoke/two-request-smoke; bigger prompts; bs=64
scenario-remote-search-uniform-tp2.sh
scenario-disagg-uniform-tp2.sh
scenario-disagg-asymmetric.sh
```
