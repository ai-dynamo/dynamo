<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo.power: GPU power control with NVIDIA EPD

`dynamo.power` runs an [EPD](https://gitlab-master.nvidia.com/capa/nvidia-epd)
power policy (NUPD redistribution, UBPC calibration, or both) over a Dynamo
deployment. EPD decides one GPU power cap per *GPU group* (the GPUs on one
node that belong to one power pool) from a throughput signal, the figure of
merit (FoM); this package senses, feeds and applies those decisions. The
package name is not "EPD" because in Dynamo that means Encode-Prefill-Decode.

A GPU index, everywhere in this package, is the host NVML index, which must
equal the `/dev/nvidiaN` minor number. That is what srt-slurm's `gpu_indices`
and `Endpoint.gpus_on_node` mean, what the node_agent's
`NVIDIA_VISIBLE_DEVICES` carries and what the applier hands to
`nvidia-smi -i`; the applier refuses to start on a host where the two
numberings disagree.

## The two commands

### node_agent: one per GPU group

```bash
python -m dynamo.power.node_agent --pool-name shared --rank 0 \
    --nodelist hecate0001,hecate0002 --log-dir /logs/power_control/shared \
    --gpu-indices 0,1,2,3 --mode nupd --budget-w-per-gpu 1500 --fom tokens_per_gpu \
    --interval-s 30 --perf-threshold 0.03 --ubpc-buffer 1.1 --warmup-policy adaptive \
    --metrics-url frontend=http://frontend:8000/metrics \
    --metrics-url worker=http://hecate0001:9090/metrics,worker=http://hecate0002:9090/metrics
```

Every rank starts EPD's pull driver with an explicit `RuntimeConfig` and
`JobContext` (rank, roster, one process per node) and the `file` actuator, so
EPD's per-node monitor writes this group's NVML energy snapshot every
interval to `<log-dir>/epd/nvml/node_energy/node_energy_<node>.json`. The
node started with `--rank 0` runs the controller and the FoM provider
(EPD gates on the rank it is given, not on the roster order; a mismatch
between the two only earns a warning) and publishes
`<log-dir>/epd/actuation/power_decision.json`
(`dynamo.power.decision.decision_file_path(log_dir)`), the exact file EPD's
`file` actuator writes: `{version, sequence, timestamp,
per_node_gpu_power_limit_w: {node: watts}}`.

At start the agent checks that EPD will sense exactly the group's GPUs (NVML
devices `0..n-1` must be the group's indices by minor number; run the
container with `NVIDIA_VISIBLE_DEVICES=<indices>` when the group is not a
prefix of the node), auto-detects the max TGP unless `--max-tgp-w` is given,
refuses `--mode nupd` with a budget that does not bind (`budget_w_per_gpu >=
max TGP`), warns when `--mode ubpc` is given a budget that *does* bind (NUPD
then redistributes too, exactly as in `both`), and probes the FoM (every
selected `--metrics-url` must answer; a `python:` provider must load and
return a number). `SIGTERM` stops EPD cleanly, which writes EPD's
`run_summary.json`.

`--min-tgp-w` is refused (exit 2). EPD's cap law
(`nupd._calculate_power_limits`) has no lower floor: `MIN_TGP` is read only
by the history-based ensemble, which this agent keeps off
(`NUPD_SKIP_HISTORY=1`), so a floor passed there would be silently ignored.
The floor the recipe's `min_tgp_w` asks for is enforced by the applier
(`--min-w-per-gpu`, below).

`--metrics-url` takes repeated or comma-separated URLs, each optionally
tagged `frontend=<url>` (the frontend's HTTP port) or `worker=<url>` (a pool
worker's system-metrics port). The tags are what a `prom:` FoM's `scope`
selects; untagged URLs are scraped only by `scope` `all`. The launcher
must pass one `worker=` URL per worker of the pool: a `pool_workers` FoM
read from a subset of the workers would make EPD react to how the router
spreads requests rather than to power. The agent refuses a scope that
selects no URL.

`--fom` forms:

| form | meaning |
|------|---------|
| `output_tokens` | rate of `dynamo_frontend_output_tokens_total` [tokens/s] |
| `tokens_per_gpu` | the same rate divided by the pool's GPU count [tokens/s/GPU] |
| `per_user_speed_p25` | 1 / p75 of `dynamo_frontend_inter_token_latency_seconds` over the interval [tokens/s per stream] |
| `tokens_per_joule` | output tokens / GPU energy from EPD's snapshots [tokens/J] |
| `ctx_tokens` | rate of new context tokens on the pool's workers: `trtllm_prompt_tokens_total − trtllm_prompt_cached_tokens_total` (vLLM / SGLang: `vllm:` / `sglang:prompt_tokens_total`, which have no cached-prompt counter) [tokens/s] |
| `gen_tokens` | rate of `trtllm_generation_tokens_total` (vLLM / SGLang: `vllm:` / `sglang:generation_tokens_total`) [tokens/s] |
| `inv_ttft_p50` | 1 / p50 of `dynamo_frontend_time_to_first_token_seconds` over the interval [1/s] |
| `prom:<metric>[:<k>=<v>,...][:scope=<frontend\|pool_workers>]:<counter\|gauge>:<sum\|avg>` | any series on the selected endpoints; counters are differenced into a rate, gauges taken as-is; without `scope=` every endpoint is scraped (a label literally named `scope` cannot be used as a filter) |
| `python:<module>:<Class>` | an object with `get_perf() -> float`; it is constructed without arguments and may read `DYNAMO_POWER_METRICS_URLS` (all), `DYNAMO_POWER_FRONTEND_METRICS_URLS`, `DYNAMO_POWER_WORKER_METRICS_URLS`, `DYNAMO_POWER_POOL_GPU_COUNT` and `LOG_DIR` |

The mapping form of the recipe accepts the same scope:
`fom: {metric: trtllm_generation_tokens_total, kind: counter, agg: sum, scope: pool_workers}`;
`FomSpec.to_cli()` renders it back into the grammar above.

Every provider follows EPD's signal contract: higher is better, a
per-interval rate over the whole pool, and exactly `0.0` for "no signal"
(before the first sample, when an endpoint is unreachable, after a counter
reset, when nothing was served, or when a gauge's aggregate is negative; a
single negative sample that leaves the sum or mean at or above zero only
lowers it). Each reading is appended to `<log-dir>/fom.csv`. The engine
presets (`ctx_tokens`, `gen_tokens`) read the workers' own `/metrics` on
`DYN_SYSTEM_PORT` (the URLs tagged `worker=`, or every URL when none is
tagged). TensorRT-LLM workers publish the `trtllm_*` counters only when
started with `--publish-metrics` (`--publish-events-and-metrics` before it),
which also turns on the engine's `enable_iter_perf_stats` and
`return_perf_metrics`; both counters are credited when a request finishes, so
give every arm of a comparison the same flags (iteration stats cost
throughput).

### applier: one per node

```bash
python -m dynamo.power.applier --decision-file /logs/power_control/shared/epd/actuation/power_decision.json \
    --node hecate0001 --gpu-indices 0,1,2,3 --mode sudo-smi --budget-w-per-gpu 1500 \
    --stale-after-s 90 --poll-s 5 --restore-default-on-exit \
    --trace-csv /logs/power_control/shared/applier_hecate0001.csv  # [--min-w-per-gpu 900]
```

The applier polls the decision file and applies this node's entry to each
GPU index of the group through one backend:

* `sudo-smi`: `sudo -n nvidia-smi -pl <W> -i <N>` per GPU (hecate; `-n`
  makes a missing sudoers rule fail at once instead of hanging on a password
  prompt, so the launcher's final reset should use the same form);
* `dcgmi`: one DCGM group of the GPUs that share a cap plus
  `dcgmi config -g <id> --set -P <W>` (AWS CMH). Groups are named
  `dynamo-power-<pid>-<gpus>` and deleted on exit; at start the applier
  lists the hostengine's groups and deletes `dynamo-power-*` groups whose
  pid is gone (an applier that was SIGKILLed or OOM-killed leaks its
  groups), and the launcher's final per-node reset should do the same;
* `nvml`: `nvmlDeviceSetPowerManagementLimit` (root only; the Kubernetes
  sidecar form).

At start the applier verifies its GPU identities: every `--gpu-indices`
entry must be an existing NVML index whose minor number equals it (exit 2
otherwise, with the mismatch in the log), because the node_agent identifies
the group by minor numbers and the applier by NVML indices.

Every apply is verified through read-only NVML and written to the trace CSV
(`ts, gpu, requested, setpoint, enforced, status`; watts, unix seconds;
status `ok` / `not_landed` / `clamped` / `unverified`, EPD's own verdicts).
`requested` is the watts EPD asked for, `setpoint` what landed after the
floor and NVML's settable range were applied. A `not_landed` whose setpoint
missed is retried on the next poll; a `not_landed` whose setpoint landed but
whose enforced limit is higher (EPD's "diverged" case) and an `unverified`
GPU are reported once and not rewritten every poll, since a rewrite cannot
change them.

With `--min-w-per-gpu <W>` (the recipe's `min_tgp_w`) a requested cap below
the floor is raised to it before writing and reported as `clamped`; the floor
must not exceed the uniform budget, and it is not applied to the hardware
default restored on exit.

When the decision file is older than `--stale-after-s` (a sequence that
keeps advancing counts as alive even when clocks disagree) or disappears
after it existed, every GPU of the group is set to the uniform budget
(`--budget-w-per-gpu`), never to the hardware default: a provisioned budget
is a promise to the facility. EPD rewrites the file once per controller
cycle, so `stale_after_s` must be at least `2 x interval_s` or the applier
would flap between the budget and EPD's caps once per cycle; the launcher
renders it from `PowerControlConfig.stale_after_s`
(`stale_after_s_for(interval_s)` = `max(90, 3 x interval_s)`, about three
cycles as the design asks). With `--restore-default-on-exit` the hardware
default limit (`nvmlDeviceGetPowerManagementDefaultLimit`) is restored on
exit; DCGM caps persist as hostengine target config, so use it there.

## EPD is installed separately

EPD's license is still being settled, so it is not a dependency of the
`ai-dynamo` wheel and nothing in this package imports it except
`dynamo.power.node_agent` (at module top, unconditionally). The launcher
bind-mounts a checkout of `nvidia-epd/src/epd` into the node_agent container
at `<site-packages>/epd`; that container needs Python >= 3.11 (nvidia-epd
declares `requires-python >= 3.11`) and `nvidia-ml-py`.

The applier runs in the host interpreter of every compute node, which needs
Python >= 3.9 (module-level `collections.abc.Callable[...]` subscriptions)
and `nvidia-ml-py` installed; the `ai-dynamo` wheel itself is not required
there. Only `dynamo.power.applier` and `dynamo.power.node_agent` import
`pynvml`: `dynamo.power.config`, `dynamo.power.decision`,
`dynamo.power.applier_core` (the applier's state machine),
`dynamo.power.node_agent_cli` and `dynamo.power.fom` import only the
standard library, so a launcher can validate a recipe and the unit tests
can run without GPUs, EPD or NVML.

## How srt-slurm wires it

The recipe block (`dynamo.power.config.PowerControlConfig.from_dict` parses
and validates it; `fom` is required and has no default):

```yaml
power_control:
  enabled: true
  mode: nupd                 # nupd | ubpc | both
  pools: shared              # shared | per-role | list of {name, roles|nodes, budget_w_per_gpu, fom, ubpc_buffer, min_tgp_w}
  budget_w_per_gpu: 1500
  fom: tokens_per_gpu
  interval_s: 30             # the applier's --stale-after-s is rendered as max(90, 3 x interval_s)
  perf_threshold: 0.03
  ubpc_buffer: {prefill: 2.0, decode: 1.1}
  min_tgp_w: null            # per-GPU floor, enforced by the applier (--min-w-per-gpu), never by EPD; <= budget
  warmup_policy: adaptive
  applier: sudo-smi          # sudo-smi | dcgmi
  epd_source: /lustre/.../nvidia-epd/src/epd
  dynamo_power_source: /lustre/.../dynamo/components/src/dynamo/power   # optional dev overlay
  critical: true
```

`mode: ubpc` and `mode: both` configure EPD identically (UBPC on, and NUPD
redistributes whenever the budget binds); they differ only by the user
choosing a budget at or above the max TGP for `ubpc`, where NUPD has nothing
to move. Pick `nupd` for a binding budget without calibration.

After the workers are healthy and before the benchmark starts, the stage
builds `WorkerPlacement(role, node, gpu_indices)` entries from
`Endpoint.gpus_on_node` (or `gpu_indices`) and calls
`build_pool_layouts(config, placements)`, which merges workers of one pool
that share a node (the TEP6 4+2 / 2+4 packing) and refuses a node split
between two pools. For every `PoolLayout` it creates one `LOG_DIR`
(`power_control/<pool>/`), launches one node_agent per GPU group with the
worker image, the two bind mounts and `--overlap` (rank = position in
`layout.nodes`, `layout.controller_node` is rank 0), passes the frontend
`/metrics` URL as `--metrics-url frontend=<url>` and every pool worker's
system-metrics URL as `--metrics-url worker=<url>`, and launches one applier
per node without a container with `--decision-file decision_file_path(log_dir)`,
`--stale-after-s config.stale_after_s` and, when the pool has a `min_tgp_w`,
`--min-w-per-gpu <pool.min_tgp_w>` (never `--min-tgp-w` on the node_agent,
which refuses it). At teardown it stops the node_agents first, then the
appliers (which restore defaults), then runs a final per-node reset
(`sudo -n nvidia-smi -pl <default> -i <N>` per GPU, or with `dcgmi` a
`dcgmi group -l` sweep that deletes leftover `dynamo-power-*` groups).
`NodeAgentSettings` and `epd_environment` show which EPD environment
variables the agent sets (`LOG_DIR`, `EPD_NODELIST`, `EPD_NUM_NODES`,
`GPUS_PER_NODE`, `MAX_TGP`, `AVERAGE_GPU_POWER_LIMIT`, `MIN_TGP` pinned to
`0`, `NUPD_SKIP_UBPC`, `EPD_PERF_THRESHOLD`, `UBPC_BUFFER`,
`EPD_WARMUP_POLICY`, `EPD_ACTUATOR=file`, `SLURMD_NODENAME`).

Caveats worth knowing: NUPD keeps the *sum of per-node caps* equal to
`budget x node count`, so a pool whose last node holds a partial group (TEP6
on 8 nodes) is capped per node as designed but its total is not guaranteed
to equal `budget x GPUs in use` (it lands above or below depending on the
nodes' relative draw); `tokens_per_gpu` divides by
`len(nodelist) x gpus_per_node`, a constant that does not change EPD's
relative-change decisions; UBPC has no load-aware release (a cap set during
quiet traffic stays until the FoM drops enough to trigger a revert), so it
is a user choice, not the default; the applier compares the controller's
timestamp with the local clock, so keep the nodes NTP-synchronised.

## Tests

```bash
python -m pytest tests/power -m unit -q
```

The tests need no GPU, no EPD, no NVML and no network: they inject scrape
results, clocks, a fake actuator and a fake NVML reader, and exercise the
node_agent's command line and the applier's state machine through the
modules that import neither `epd` nor `pynvml`.
