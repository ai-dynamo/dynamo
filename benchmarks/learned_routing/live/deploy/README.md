<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Live deployment recipe (learned-routing campaign, Amendment A13)

This directory deploys the router that offline replay emulates onto real H100s, so live AIPerf runs
can check whether the simulated ranking of routing policies holds. One 8 x H100 SXM node runs a
Dynamo frontend with an embedded aggregated KV router plus 4 vLLM 0.24.0 workers serving
Qwen/Qwen3-32B at TP2 (N = 4). An optional two-node variant runs N = 8. The scripts target a Slurm
GPU cluster with Pyxis/Enroot containers and a shared filesystem; every site-specific value comes
from `site.env` (copy `site.env.example`).

Status, 2026-10-03: the scripts are written, unit-tested and preflighted on the workstation against
mocker workers. The recipe then ran on GPUs on 2026-10-03: two smoke jobs on one 8 x H100 node.
`notes/learned-routing/campaign/facts/compute_summary.md` lists GPU-cluster jobs only in snapshots
that mirror the GPU lane's job record. The first GPU jobs found four bugs, now fixed: the job env
file was not exported to srun steps; the wheel and venv parent directories were missing; bindgen
with the PyPI libclang had no `stdbool.h`; and stopping a frontend killed only a wrapper subshell,
so the serve-phase frontend could not bind and readiness was answered by the check-phase frontend.

The smoke job was submitted as (`L` is the cluster-side live root `LR_LIVE_ROOT`, `C` the staged
commit):

```bash
bash benchmarks/learned_routing/live/deploy/submit.sh --plan-id <plan_id> --commit $C \
  --policies "default_defaults" --payload $L/src/${C:0:12}/benchmarks/learned_routing/live/smoke_payload.sh \
  --payload-timeout 4200 --purpose "A13 GPU smoke" --env LR_SMOKE_INPUTS=$L/smoke/s1/inputs \
  --env LR_AIPERF_ENV=$L/env/aiperf-0.13.0 --env LR_RUN_TIMEOUT_S=2400
```

## Contents

- [Files](#files)
- [What "same router" means here](#what-same-router-means-here)
- [Engine fidelity map](#engine-fidelity-map)
- [Known live-versus-replay differences](#known-live-versus-replay-differences)
- [Runbook](#runbook)
- [Health checks](#health-checks)
- [Payload contract (for the AIPerf runner)](#payload-contract-for-the-aiperf-runner)
- [Build path](#build-path)
- [Weights and image](#weights-and-image)
- [Two-node variant (N = 8)](#two-node-variant-n--8)
- [Site settings](#site-settings)
- [Layout on the cluster](#layout-on-the-cluster)
- [GPU etiquette](#gpu-etiquette)
- [Local validation](#local-validation)
- [What the first GPU job must confirm](#what-the-first-gpu-job-must-confirm)

## Files

| File | Runs where | Purpose |
|---|---|---|
| `plan.py` | workstation, worktree `.venv` | Freezes a plan from `CR/config/engine.json` and harness policy specs; proves flag parity (below). |
| `specs/smoke.json` | input to `plan.py` | `default@defaults`, `round_robin`, `learned-choice@theta0` (the contract's parity anchor). |
| `stage_source.sh` | workstation, then cluster login | Thin git bundle of the commit, remote checkout, engine shim patch, plan upload, node-side `site.env`. |
| `fetch_etcd.sh` | workstation, then cluster login | Two-node only: pinned, checksummed etcd binary on the shared filesystem. |
| `submit.sh` / `finish.sh` | workstation | Lock, `sbatch`, `facts/live.json` record; end state and lock release. |
| `hold_lock.sh` | workstation | The cooperative GPU hold lock (`$LR_HOLD_LOCK_DIR/ACTIVE`, default `~/.lr-gpu-hold`). |
| `live_facts.py` | workstation | Appends and updates job records in `CR/facts/live.json`. |
| `job_node.sh` | Slurm batch script (head node) | The whole job: prep, staging, build, workers, per-policy checks and payload, teardown. |
| `common.sh` | sourced everywhere | Site settings (`site.env`), paths, topology, ports, CPU masks, container and runtime environment. |
| `site.env.example` | template | Every site-specific setting, as placeholders. |
| `node_prep.sh` | each node, host | Hardware check (8 x H100 80GB HBM3), node-local root, image copy, topology record. |
| `stage_weights.sh`, `weights.py` | each node, host | Parallel copy of the pinned checkpoint, full hash check, YaRN `config.json`. |
| `build_env.sh`, `verify_env.py` | head node host / container | Wheel build; venv over the image; import and call-site checks. |
| `node_workers.sh`, `worker.sh` | each node, container | The 4 vLLM workers of a node (`python -m dynamo.vllm`). |
| `frontend.sh` | head node, container | `python -m dynamo.frontend` for one planned policy. |
| `health_check.py` | head node host | `wait`, `smoke`, `reset`, `snapshot` (stdlib only). |
| `patches/0001-vllm-0.24-engine-compat.patch` | applied by `stage_source.sh` | Live-only shim so this commit's `dynamo.vllm` runs on vLLM 0.24 (below). |
| `qwen3-32b-9216db57.manifest.json` | input | Sizes and hashes of every file at the pinned revision, from the Hugging Face API. |
| `tests/test_deploy.py`, `tests/local_mocker_smoke.sh` | workstation | Unit tests; CPU preflight against mocker workers. |

## What "same router" means here

Offline replay builds `KvRouterConfig(router_policy_config=<yaml>, **spec.router_config)` with
`router_mode="kv_router"`, or uses round-robin. The live frontend must reach the same router:

- **Router mode.** Aggregated KV router inside the frontend (`--router-mode kv`), or
  `--router-mode round-robin` for the round-robin baseline. Workers never set `--router-mode`, so
  they inherit the frontend's config.
- **Policy YAML.** `plan.py` writes, per policy, the exact bytes `lr-eval` hands replay for
  replicate `k` (`PolicySpec.replay_yaml_text(seed = k + 1)`, A1). For `default@defaults` at `k = 2`
  the file is byte-identical to the harness's own `CR/runs/policies/replay/8f6f4849...yaml`.
  `frontend.sh` refuses a YAML whose sha256 differs from the plan.
- **Knobs.** Each spec's `router_config` sidecar knob (for example `router_temperature`,
  `overlap_score_credit`, `router_queue_threshold`) becomes the frontend flag whose argparse `dest`
  is that knob. `plan.py` then parses the flags with the frontend's own `FrontendArgGroup` (with
  `DYN_*` removed from the environment) and requires `config.kv_router_kwargs()` to equal replay's
  kwargs (the `KvRouterConfig` signature defaults, overridden by `router_config`, plus the YAML path)
  key for key. Any difference fails the plan. Today the defaults agree on all 38 fields.
- **No live-only routing features.** The plan also requires: no session affinity
  (`--router-session-affinity-ttl-secs` unset; replay never sets an affinity target), no busy-worker
  rejection thresholds, no request migration (`migration_limit` 0), and router AIS prefill model
  `none`. The launch environment drops every inherited `DYN_ROUTER_*`, `DYN_ACTIVE_*`,
  `DYN_KV_*`, `DYN_SESSION_*` and similar variable, so the plan is the only source of router config.
- **KV events on.** Every worker publishes vLLM KV events (`--kv-events-config` with
  `enable_kv_cache_events: true`, one ZMQ port per worker) to the frontend's indexer over the ZMQ
  event plane, so the router sees real device-tier overlaps. The router block size is 16, matching
  the engine.
- **Policy identity.** The frontend resolves and validates the policy at startup: a malformed
  `learned-choice` YAML makes it exit with `invalid parameters for worker-selection policy type
  "learned-choice": theta has 2 entries; feature_set v1 needs 8` (checked locally). The health check
  reads the frontend's `--dump-config-to` file and requires the YAML path's sha256 and every
  planned kwarg to match.
- **Fresh router state per run.** Each policy's measured run starts a fresh frontend (fresh
  indexer, policy RNG at its seed, empty session maps) against workers whose prefix caches were just
  flushed and proven cold, mirroring replay's empty start.

## Engine fidelity map

`plan.py` maps every `mock_engine_args` key; an unknown key, or a value the live engine cannot
reproduce, fails the plan.

| `engine.json` | Live vLLM worker | Note |
|---|---|---|
| `model`, `tp`, `ais_perf_config` | `--served-model-name Qwen/Qwen3-32B --tensor-parallel-size 2` | Image vLLM 0.24.0 checked in `verify_env.py`; 8 x `H100 80GB HBM3` checked in `node_prep.sh`. |
| `dp_size` 1 | `--data-parallel-size 1` | |
| `block_size` 16 | `--block-size 16` | Also the router block size. |
| `max_model_len` 131072 | `--max-model-len 131072` | YaRN `{rope_type: yarn, factor: 4.0, original_max_position_embeddings: 32768}` written into the node-local `config.json` (the Qwen model card's file route). |
| `max_num_seqs` 1024 | `--max-num-seqs 1024` | Equals vLLM 0.24's API-server default on 80 GB GPUs (verified in the 0.24.0 source). |
| `max_num_batched_tokens` 8192 | `--max-num-batched-tokens 8192` | Same. |
| `enable_chunked_prefill` true | `--enable-chunked-prefill` | |
| `enable_prefix_caching` true | `--enable-prefix-caching` | Required (KV events depend on it). |
| `num_gpu_blocks` 18863 | `--num-gpu-blocks-override 18864` | vLLM reserves one null block, so 18,864 total = 18,863 usable, the `MockEngineArgs.num_gpu_blocks` convention. vLLM 0.24 logs `Overriding num_gpu_blocks=<profiled> with num_gpu_blocks_override=18864`; the first job must see profiled >= 18864. |
| `engine_type` vllm, `worker_type` aggregated | `python -m dynamo.vllm`, no disaggregation flags | |
| `speedup_ratio`, `decode_speedup_ratio` 1.0 | n/a | Simulator time scale only; any other value fails the plan. |
| (live only) | `--dtype bfloat16 --kv-cache-dtype auto --gpu-memory-utilization 0.90 --generation-config vllm` | bf16 weights and KV, as AIS assumes; the override pins capacity, 0.90 leaves headroom; no sampling defaults from the model's generation config. |

## Known live-versus-replay differences

These are intrinsic or deliberate, and each is small or unavoidable. Hypotheses about their effect
are labeled as such.

1. **Worker capacity seen by policies.** The router reads `total_kv_blocks` = 18,864 live versus
   18,863 in replay (vLLM counts the null block). `learned-choice` feature 4 (`kv_load_frac`)
   differs by a relative 5.3e-5. Usable capacity is identical.
2. **Router state staleness.** Replay applies KV events and load updates instantly; live events
   arrive over ZMQ after the engine step. A5's offline lag knob (`router_state_lag_ms`) brackets
   this; the live runs measure it.
3. **Prefill completion signal.** Replay marks prefill done at the mocker's first-token event; the
   live router marks it when the first token reaches the frontend.
4. **Policy RNG streams.** Same seed (`k + 1`) and a fresh frontend per run, but live decisions
   happen in a different order and timing, so draws diverge after the first difference. Live runs
   validate relative results only and never feed selection (A13.2).
5. **Engine-side `session_id`.** This commit's `dynamo.vllm` passes `session_id` to
   `AsyncLLM.generate`, which vLLM 0.24 lacks. The shim drops it. Hypothesis: no effect on 0.24,
   whose engine has no session feature; the router-side session context is untouched.
6. **Binding features.** The live wheel uses default features (`custom-policy`) without
   `ais-forward-pass`. The router and policy code are the same source trees as the tuning build
   (`lib/`, `components/` and `Cargo.lock` tree hashes equal those of the tuning build
   `6955b0ee`); `plan.py` refuses a commit whose trees differ.
7. **Engine behavior AIS approximates.** Real CUDA graphs, vLLM 0.24's default asynchronous
   scheduling, real preemption and real kernel timings. This is what the live lane exists to test.
8. **Request lengths are forced.** Requests carry exact token-ID prompts and
   `ignore_eos`, `min_tokens = max_tokens = OSL`, so ISL and OSL match the trace exactly and generated
   text never feeds a later prompt. The health check verifies prompt and completion token counts.

## Runbook

All workstation commands run from the worktree root with its `.venv`. Nothing allocates a GPU
before step 5.

1. **Plan.** Policy specs are harness spec files (finalists later, `specs/smoke.json` now). The plan
   directory is write-once.

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python benchmarks/learned_routing/live/deploy/plan.py \
     --engine $CR/config/engine.json \
     --spec benchmarks/learned_routing/live/deploy/specs/smoke.json \
     --replicate 0 --out $CR/runs/live/plans/<name>
   ```

   It prints the `plan_id` and writes `PLAN.json`, `engine_plan.json`, `engine.json` and
   `policies/<slug>/{policy.yaml, policy_plan.json}`.
2. **Commit** this directory in the worktree (`git commit -s -- <paths>`). `stage_source.sh` ships a
   commit, not the working tree.
3. **Stage** the commit and plan on the cluster (login node only):

   ```bash
   bash benchmarks/learned_routing/live/deploy/stage_source.sh <plan_dir> [<commit>]
   ```

   It prints `LR_COMMIT`, the remote `plan_dir`, the source manifest and the local scratch path to
   list for cleanup.
4. **Two-node only:** `bash .../fetch_etcd.sh`.
5. **Submit.** Check the scheduler first, then submit. `--payload` is a cluster path to the AIPerf
   runner (omit it for a recipe-only smoke that runs every check and exits).

   ```bash
   bash .../submit.sh --plan-id <id> --commit <sha> --test-only
   bash .../submit.sh --plan-id <id> --commit <sha> --policies "default_defaults" \
     --payload <cluster path> --purpose "GPU smoke"
   ```

   The default partition is `LR_PARTITION` with `--time 01:55:00`; pick a partition and time
   limit that fit your cluster's policy. The lock caps any job at 2 h 45 min.
6. **Monitor** read-only from the login node: `squeue -j <id>`, and the run directory
   `.../live/runs/<id>-<tag>/` (`job.log`, `logs/`, `policies/<slug>/check/*.json`).
7. **Finish:** `bash .../finish.sh <id> <tag>` records the Slurm end state in `facts/live.json`
   and releases the lock. Add `--cancel` only to stop that exact job early.
8. **Ledger:** list the run directory, node-local paths and any new shared-filesystem directories
   for cleanup. Nothing in this recipe deletes anything; the job's node-local root
   (`$LR_NODE_BASE/[<LR_NODE_TAG>-]lr-<job>`) is left for the node's normal scratch policy.

`job_node.sh` stages, per job:

| Stage | What | Typical cost (estimate, unmeasured) |
|---|---|---|
| 1 | `node_prep.sh` on every node; container created from the node-local image copy | image copy ~18 GB from the shared filesystem |
| 2 | weights to node-local disk on every node, in parallel with the wheel build on the head host | 65.5 GB copy + hash; release Rust build (the Oct 1 SGLang-image build took 315 s on 128 CPUs) |
| 3 | venv + `verify_env.py` in the container | seconds when reused |
| 4 | etcd (two-node), then one `node_workers.sh` step per node | model load + CUDA graph capture |
| 5 | per policy: `check` frontend -> `wait` -> `smoke` -> `reset`; then `serve` frontend -> `wait` -> snapshot -> payload -> snapshot | |
| 6 | teardown, `runtime-manifest.json` | |

The wheel and venv are reused by later jobs at the same commit, so stage 2 shrinks to the weights.
Set `LR_STOP_ON_FAILURE=0` (via `--env`) to continue to the next policy after a failed check.

## Health checks

`health_check.py` writes one JSON report per command under `policies/<slug>/check/` and exits
non-zero on any failed check, which stops the job before a measured run.

| Check | Evidence |
|---|---|
| `ready` | Every worker's system server reports `generate` ready; the frontend's `/health` lists exactly N `generate` instances in the job's namespace; `/v1/models` serves `Qwen/Qwen3-32B`. The first wait allows 40 min for model load and fails fast if a worker or frontend step exits. |
| `instance_map` | Frontend instance IDs mapped to workers through the hex `worker_id` label on each worker's `generate` metrics; one pinned request per instance (`x-dynamo-worker-instance-id`) must come back with that `nvext.worker_id`. |
| `exact_isl_forced_osl` | One pinned token-ID request per worker: `usage.prompt_tokens` equals the ISL and `usage.completion_tokens` equals the forced OSL. |
| `kv_events_reach_router` | The frontend's `dynamo_kvrouter_kv_cache_events_applied{event_type="stored"}` rises after a 2,048-token request (n/a for round-robin, which has no indexer). |
| `kv_event_source_consistent` | `router_kv_event_source_mismatch_workers` is 0 or absent. |
| `prefix_reuse` | The same prefix plus 128 tokens, unpinned: under `default@defaults` it must land on the first request's worker with `cached_tokens` >= 2,048 - 16 (overlap routing works end to end). Other policies record the outcome without asserting it. |
| `policy_evidence` | The config dump's `router_mode`, the YAML path's sha256 and every planned kwarg match the plan. |
| `cold_reset` | Per worker: a sentinel prompt, its repeat (positive control, `cached_tokens` >= 512 - 16), `POST /engine/flush_cache`, the repeat again (`cached_tokens` 0, and no `vllm:prefix_cache_hits` increase); then a final flush of every worker. vLLM's flush returns "ok" even when it could not flush, so the after-flush probe is the real evidence. |
| `snapshot` | Raw `/metrics` of the frontend and every worker before and after each payload (preemptions, prefix hits, router counters). |

Probe prompts are seeded random token IDs in [1000, 150000) with a per-invocation nonce, so they
never collide with workload prefixes or earlier probes. They are diagnostics, not benchmark numbers.

## Payload contract (for the AIPerf runner)

`job_node.sh` runs `LR_PAYLOAD` once per policy, after the cold reset and a fresh frontend, inside
the container on the head node, pinned to `LR_CLIENT_CPUS` (`42-55,154-167`, NUMA 0, disjoint from
the frontend's `28-41,140-153` and the workers'), under `timeout LR_PAYLOAD_TIMEOUT_S`.

Environment provided: `LR_ENDPOINT` (`http://<head>:18000`), `LR_MODEL`, `LR_POLICY_SLUG`,
`LR_POLICY_DIR`, `LR_PAYLOAD_DIR` (write outputs here), `LR_NUM_WORKERS`, plus every `LR_*`
from the job env file (`submit.sh --env LR_X=...`). The payload brings its own AIPerf environment
(for example a venv on the shared filesystem); it must not install into the serving venv.

Requests that keep replay semantics through this deployment:

- `POST /v1/completions`, `model` = `Qwen/Qwen3-32B`, `prompt` = the token-ID list (exact ISL; the
  frontend does not re-tokenize integer prompts), `max_tokens` = `min_tokens` = OSL,
  `ignore_eos: true`, `stream: true` with `stream_options.include_usage`. Optionally
  `nvext: {"extra_fields": ["worker_id"]}` returns the routed worker in the final chunk, which
  gives per-request placement to compare with replay's `routing_history`.
- Sessions: send `x-dynamo-session-id: <replay session_id>` exactly for the requests that carry a
  session in replay (multi-turn Mooncake, synthetic sessions, AgentX), and no session header for
  single-turn Mooncake and FAST25 rows, which replay runs without session metadata. Replay builds
  `SessionContext::new(session_id, None, None, None)`; the header alone gives the policies the same
  `session_id` (`learned-choice` feature 6 and `sticky-session` read only that). Do not send
  `x-dynamo-parent-session-id`. With affinity disabled the header never pins a worker.
- ISL + OSL must stay within 131,072 (replay truncates, vLLM rejects); the campaign traces fit
  (max 125,878 / 126,527 / 124,597).
- Do not send `x-dynamo-worker-instance-id` or other routing headers in measured runs.

## Build path

The campaign commit's Rust core (router, `learned-choice`, `sticky-session`) must be the code that
routes live, and vLLM must stay 0.24.0 (operator-fixed; AIS timing is calibrated to it). Two facts
shape the build:

- **Image and glibc.** The image is `vllm/vllm-openai:v0.24.0`
  (`sha256:f9de5cd9...`, Ubuntu 22.04, glibc 2.35, torch 2.11.0+cu130, CUDA 13 with forward
  compatibility on a 535-series driver). The GPU hosts were also Ubuntu 22.04 / glibc 2.35 with
  libclang 11, and the binding is abi3 (cp310), so `build_env.sh wheel` builds on the head node's
  host with the shared toolchain `LR_TOOLCHAIN_ENV` (Rust 1.96.1, protoc, uv). `verify_env.py` then imports the
  binding inside the image and checks its hash against the wheel. If the host lacks libclang or
  CMake, the script takes them from PyPI wheels in a throwaway build venv (no root, no apt).
- **vLLM API drift.** Dynamo's vLLM pin moved from 0.24.0 to 0.30.0 between July and September.
  Static checks of this commit's `components/src/dynamo/vllm` against the vLLM 0.24.0 source
  (commit `ee0da84a`) found two hard breaks on the request path: `vllm.exceptions.VLLMClientError`
  (absent in 0.24) and `session_id=` passed to `AsyncLLM.generate` at three call sites (absent in
  0.24). Every other `vllm` import is guarded by version fallbacks or unused by text serving.
  `patches/0001-vllm-0.24-engine-compat.patch` adapts a fix already validated on vLLM 0.24
  ("gate version-specific request interfaces"): it falls back
  to the concrete 0.24 error classes and passes `session_id` only when `generate` accepts it. It
  touches only `components/src/dynamo/vllm/{errors,handlers}.py`, applies cleanly to this commit,
  and is applied by `stage_source.sh` to the cluster checkout as a recorded uncommitted diff (the
  source manifest stores the patch and diff sha256). Code changes stay under `live/`, per the
  campaign rule. `verify_env.py` re-checks every `engine_client.generate` call site against the
  installed `AsyncLLM.generate` signature before any model loads.

The venv (`--system-site-packages` over the image's python3) gets the wheel, ai-dynamo's runtime
dependencies except `aisimulate` and `transformers`, and ai-dynamo editable from the patched
checkout. The worktree's `.venv` and its tuning bindings are never touched.

## Weights and image

- **Checkpoint.** `Qwen/Qwen3-32B` at revision `9216db5781bf21249d130ec9da846c4624c16137`, already
  staged on the shared filesystem (`LR_MODEL_MASTER`, read-only here). `stage_weights.sh` checks
  sizes against the manifest, copies the 25 files with 8 parallel streams to
  `<node root>/models/Qwen3-32B`, hashes every byte (LFS sha256 for weights and `tokenizer.json`,
  git blob sha1 for small files), then writes the YaRN config and keeps the verified original as
  `config.json.orig`. The weights are node-local so 4 workers do not load 65.5 GB concurrently from
  the shared filesystem.
- **Fallback.** If the shared copy fails the size check and `LR_ALLOW_HF_DOWNLOAD=1`, the node
  downloads the pinned revision with `hf download --max-workers 16` straight to local disk. Qwen3-32B
  is public; a token is optional. If one is wanted, put it in a mode-0600 file on the cluster and
  pass its path as `LR_HF_TOKEN_FILE`. The script reads it into that one command's environment and never prints it.
  The serving processes run with the token unset and `HF_HUB_OFFLINE=1`.
- **Image.** `node_prep.sh` copies the existing SquashFS on the shared filesystem
  (`LR_IMAGE_SQSH`, 18,085,605,376 bytes, imported from the digest above) to node-local disk before Pyxis starts it, per the field note that starting multi-GB
  images from shared storage is slow.

## Two-node variant (N = 8)

`submit.sh --nodes 2` allocates two whole nodes; `job_node.sh` then:

- starts etcd on the head node (`fetch_etcd.sh` stages v3.5.21, sha256-pinned) and switches
  discovery to `etcd` (`ETCD_ENDPOINTS=http://<head ip>:2379`). File discovery relies on inotify,
  which does not see another node's writes, so it cannot span nodes;
- stages weights and creates the container on both nodes, and runs 4 workers per node (local GPU
  pairs and ports identical on each node);
- advertises the request, response and event planes on each node's own IPv4
  (`DYN_TCP_RPC_HOST`, `DYN_TCP_RESPONSE_STREAM_HOST`, `DYN_EVENT_PLANE_HOST`), and runs the
  frontend and checks on the head node with `--expect 8`.

The two-node path is scripted but untested; run it only after the single-node smoke passes. The
campaign held one owned request at a time, so check a two-node job against your cluster's per-user
node limit.

## Site settings

`common.sh` reads `site.env` next to it (or the file named by `LR_SITE_ENV`); variables already in
the environment win. Nothing site-specific is built in: `submit.sh`, `stage_source.sh`,
`finish.sh`, `fetch_etcd.sh`, `../aiperf_env.sh` and `job_node.sh` each check the settings they
need. `site.env.example` lists them:

| Setting | Used by |
|---|---|
| `LR_CR` | workstation: the campaign root (`CR/facts/live.json`) |
| `LR_SSH_ALIAS`, `LR_ACCOUNT`, `LR_PARTITION` | workstation: the cluster's login node, Slurm account and default partition |
| `LR_BASE_REPO`, `LR_BUNDLE_BASE` | `stage_source.sh`: the cluster-side clone that already has the bundle base |
| `LR_SHARED_ROOT`, `LR_LIVE_ROOT` | the shared-filesystem root mounted into containers, and the live lane's durable root |
| `LR_PRIOR_RUNTIME` or `LR_IMAGE_SQSH` + `LR_MODEL_MASTER` | the staged vLLM image and Qwen3-32B checkpoint |
| `LR_TOOLCHAIN_ENV` | the wheel build's toolchain (rustc, protoc, uv) |
| `LR_NODE_BASE`, `LR_NODE_TAG`, `LR_HOLD_LOCK_DIR`, `LR_CLUSTER` | optional: node-local scratch, its directory prefix, the hold-lock directory, the job-record label |

`stage_source.sh` writes the node-side subset (`LR_NODE_SITE_KEYS`) to `site.env` in the staged
deploy directory on the cluster, so `job_node.sh` and every srun step read the same values.

## Layout on the cluster

| Path | Contents |
|---|---|
| `$LR_LIVE_ROOT/bundles/` | Git bundles (`lr-<sha12>.bundle`). |
| `.../live/src/<sha12>/` and `<sha12>.source-manifest.json` | Checkout + engine shim; commit, tree hashes, patch and diff sha256. |
| `.../live/wheels/<sha12>/` | `ai_dynamo_runtime-*.whl`, `build-manifest.json`. |
| `.../live/venvs/<sha12>-vllm0.24.0/` | Serving venv and `venv-manifest.json`. |
| `.../live/plans/<plan_id>/` | Uploaded plans. |
| `.../live/jobs/<tag>.env` | Per-job settings read by `job_node.sh`. |
| `.../live/runs/<job>-<tag>/` | `job.log`, `scontrol-job.txt`, `nodes/`, `weights/`, `build/`, `logs/`, `policies/<slug>/{check,serve,metrics,payload}`, `runtime-manifest.json`; Slurm output in `runs/slurm-<job>-<tag>.out`. |
| `.../live/cache/` | uv cache and vLLM / TorchInductor compile caches, shared across jobs. |
| `.../live/tools/` | etcd (two-node). |
| `$LR_NODE_BASE/[<LR_NODE_TAG>-]lr-<job>/` (each node) | Image copy, weights, Enroot data, Cargo target, discovery store, HOME. |

## GPU etiquette

- One owned GPU request at a time, whole nodes only (`--exclusive`, 8 GPUs charged per node).
- `submit.sh` takes the cooperative lock `$LR_HOLD_LOCK_DIR/ACTIVE` (shared with any other sessions
  that hold nodes on the same cluster; hard expiry = time limit + 15 min, at most 3 h) and refuses
  when it is held. The lock is renamed to `released-<tag>-<UTC>` on release, never deleted.
  Amendment A13 requires it.
- Every job is recorded with its cancel command (`ssh $LR_SSH_ALIAS scancel <id>`) in
  `CR/facts/live.json`.
- The job ends after its last policy, so the allocation is never held idle; there is no hold-open
  mode. `finish.sh` closes the record and the lock.
- Nothing is deleted locally or remotely; keep a list of remote scratch for cleanup.

## Local validation

Done on the workstation, 2026-10-03 (no GPU):

- `tests/test_deploy.py`: 16 tests pass with the worktree `.venv`. They cover the engine flag map and
  its refusals (unknown key, simulator scale, prefix caching off, capacity mismatch, non-H100),
  byte equality of each planned YAML with `PolicySpec.replay_yaml_text` at seed `k + 1`, frontend
  kwargs parity for float, bool and nullable knobs, weight copy, hash and YaRN patching (including
  corruption detection and git blob sha1 against `git hash-object`), the lock's exclusivity and
  rename-only release, `live.json` merging, and Prometheus parsing.
- `plan.py` on the frozen `engine.json` with `default`, `round_robin` and three test specs: all
  parity checks pass; the planned `default@defaults` YAML for `k = 2` is byte-identical to the
  harness's replay file.
- `tests/local_mocker_smoke.sh`: the real `frontend.sh` and `health_check.py` against two mocker
  workers (file discovery, TCP planes, ZMQ events) for `default@defaults`, `learned-choice@theta0`,
  `sticky-session` (hard), a default-cost spec with three router knobs, and round-robin. `wait` and
  `smoke` pass for all five: exact ISL and forced OSL, KV events reaching the router, prefix
  extension routed to the same worker with 2,048 cached tokens, and policy evidence with no kwarg
  differences.
- The frontend exits at startup on an invalid `learned-choice` YAML (negative control above).
- The compat patch applies cleanly (`git apply --check`); the static import and keyword checks of
  the patched `dynamo.vllm` against vLLM 0.24.0's source report nothing left on the request path.

On the cluster (login node, read-only apart from staging): `stage_source.sh` staged the smoke
commit and plan (the first attempt failed in checkout on LFS test media; fixed by skipping LFS
smudge and making the step resumable); the shared checkpoint passed the manifest size check for all
25 files; the image SquashFS has the recorded size; the shared toolchain reports rustc 1.96.1, uv
0.12.9 and protoc 29.3; `health_check.py` and `weights.py` run under the host's Python 3.10; and
`submit.sh --test-only` predicted an immediate start. No job was submitted before the smoke.

Not validated locally: anything that needs vLLM or a GPU (`stage_weights.sh` on real paths,
`build_env.sh`, `verify_env.py`, `worker.sh`, the cold reset, `job_node.sh` as a whole), and the
two-node path.

## What the first GPU job must confirm

In order, before trusting any measured run:

1. `node_prep`: 8 x H100 80GB HBM3; `topo.txt` puts GPUs 0-3 on NUMA 0 and 4-7 on NUMA 1 (the CPU
   masks assume it).
2. `build/wheel-build-manifest.json` and `env-verify.json`: wheel built, binding hash matches,
   vLLM 0.24.0, no call-site problems.
3. Each `worker-<i>.log`: `Overriding num_gpu_blocks=<profiled> with num_gpu_blocks_override=18864`
   with profiled >= 18,864 (else lower the override and re-run replay with the same capacity, or
   raise `--gpu-memory-utilization`); `max_model_len` 131072 accepted with YaRN; KV event publisher
   subscribed on its port.
4. `check/wait.json`, `smoke.json`, `reset.json` all `ok` for `default_defaults`.
5. Then the A13 smoke itself: one Mooncake cell under `default@defaults` through the AIPerf payload,
   compared against offline replay.
