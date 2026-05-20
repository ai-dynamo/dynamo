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
2. Scrape A's audit log for `offload_register_complete` block hashes.
3. Use hub control-plane calls for `open_session`, `pull_from_session`, and
   `close_session`.
4. Send streaming chat request R2 to instance B and assert a G2 hit.
5. Render the trace artifact from the experiment directory.

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
- Keep trace generation as part of the run.
- Keep profile defaults in `hardware-profiles.sh`, not inline in
  `p2p-smoke.sh` or `launch-instance.sh`.
- If TTFT is more than 1 second slower than the same-node NIXL baseline,
  preserve the run artifacts and inspect logs before rerunning.
