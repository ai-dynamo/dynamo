# Resume: P2P Smoke

This file is intentionally portable. Keep machine names, usernames, scratch
paths, SSH details, and auth notes out of checked-in skill material.

## Current Shape

The P2P smoke starts:

- one `kvbm_hub`
- instance A on port 8000
- instance B on port 8002

The request flow is:

1. Send streaming chat request R1 to instance A to warm G2.
2. Poll the hub indexer for A's published `hash_u128` block values.
3. Use hub control-plane calls for `open_session`, `pull_from_session`, and
   `close_session`.
4. Assert the hub index lists instance B as an owner for the pulled block.
5. Send streaming chat request R2 to instance B through the chat streaming
   endpoint.
6. Render the trace artifact from the experiment directory.
7. Write `trace-gate.env` and only print PASS when the trace is non-empty and
   useful.

Runtime Python is resolved in this order: explicit `PYTHON_BIN`, optional
`KVBM_VENV`, runtime image `/opt/dynamo/venv`, then `python3` on `PATH`. Do not
assume a worktree-local sandbox or virtualenv exists.

## Hardware Profiles

Do not bake hardware choices into the workflow. The smoke sources
`../disagg-bringup/hardware-profiles.sh`.

| Profile | Use | Notes |
|---|---|---|
| `h100-a100` | Experiment F and cross-node validation | DeepSeek 8B, two GPUs |
| `spark-gb10` | Local smoke/debug only | Qwen 0.6B, shared GPU |
| `custom` | Other allocations | Caller must provide model, cache, GMU, and placement |

Run examples:

```bash
P2P_HARDWARE_PROFILE=h100-a100 bash .claude/skills/p2p-smoke/p2p-smoke.sh
P2P_HARDWARE_PROFILE=spark-gb10 bash .claude/skills/p2p-smoke/p2p-smoke.sh
```

For custom hardware, set:

```bash
P2P_HARDWARE_PROFILE=custom
P2P_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
P2P_GMU=0.70
P2P_CACHE_GB=16
P2P_MAX_MODEL_LEN=2048
P2P_A_CUDA_VISIBLE_DEVICES=0
P2P_B_CUDA_VISIBLE_DEVICES=1
```

## Do Not Regress

- Keep endpoint type `chat` and streaming enabled for experiment runs.
- Keep trace generation and `trace-gate.env` as part of the run. A rendered
  but empty trace is not a useful trace.
- Keep profile defaults in `hardware-profiles.sh`, not inline in
  `p2p-smoke.sh` or `launch-instance.sh`.
- Keep broad stale-process cleanup enabled by default. Set
  `P2P_CLEANUP_STALE_PROCESSES=0` only in shared sessions where machine-level
  process or socket cleanup could affect another allocation.
- Keep `P2P_ENFORCE_EAGER=1` for functional smoke runs unless the task is
  specifically to debug vLLM compiled startup.
- Treat `nvidia-smi` as cleanup telemetry only, not as a validation gate.
- If TTFT is more than 1 second slower than the same-node NIXL baseline,
  preserve the run artifacts and inspect logs before rerunning.
