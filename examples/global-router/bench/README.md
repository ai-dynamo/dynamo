<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Pool-selection benchmark on a Mocker fleet

`bench.py` compares Global Ingress pool-selection policies (`--router-policy`)
on CPU Mocker pools spread over several EKS clusters. Each region gets a CPU
node group, a fast and a slow pool, and its own Global Ingress and Global
Discovery Service. A load-generator Job in every region replays the AgentX
Claude Code traces (aiperf public dataset
`semianalysis_cc_traces_weka_with_subagents_060526`) against its local
ingress, all starting at the same time.

Everything is created through CloudFormation, reusing each cluster's existing
CDK kubectl provider (`../rr-backend/cluster.py`). The bench namespace and
nodes are separate from any serving deployment; the node group is tainted so
only bench pods land on it.

### TCP settings

Every run uses the TCP settings a tuned deployment has. Without them, a
cross-region hop costs 2–4 round trips instead of 1:

- **Nagle off on every HTTP server in the request path** (pool frontends and
  Global Ingress). The images here predate the frontend's own `TCP_NODELAY`
  (#16728), so `nodelay.c` is preloaded and sets it on each accepted
  connection. Without it, the first streamed token waits for the client to
  acknowledge the response headers.
- **`net.ipv4.tcp_slow_start_after_idle=0` on every sender** (Global Ingress,
  pool frontends, load generator), so an idle connection keeps its send window
  for the next prompt. Each pod has its own network namespace and kubelet
  allows no unsafe sysctls, so a privileged init container sets it. `probe`
  runs on the node itself, so it sets the node's value for its run and
  restores it afterwards.

`tcp-check` connects to every frontend and ingress pod from inside its network
namespace. It confirms the accepted connection has `TCP_NODELAY` and the pod's
sysctl is 0, and exits non-zero otherwise. Run it before measuring.

### Configure

Build the shim once with any x86-64 glibc toolchain:

```bash
cc -shared -fPIC -Os -s -o libnodelay.so nodelay.c
```

Write an operator-local JSON file outside Git with:

- `namespace`, `model`, `workerReplicas`, `frontendCpu`, `routerReplicas`, `routerCpu`.
- `nodelayShim`: the path of the built `libnodelay.so`.
- `regions`: `region`, `site` (the Karmada member name), `short`,
  `clusterName`, `kubectlProviderServiceToken`, `kubectlProviderHandlerArn`,
  `gpuNodegroup` (the node group whose role, subnet and security groups the
  CPU nodes copy), and digest-pinned `plannerImage`, `relayImage`, `controlImage`.
- `pools`: `stem` and `mockerArgs` for each pool, for example
  `["--speedup-ratio", "0.4", "--max-num-seqs", "16"]`.
- `nodegroup`: `instanceType`, `count`, `diskGiB`.
- `load`: `dataset`, `datasetRepo`, `entries`, `warmupRequests`,
  `profilingS`, `windowOffsetS`, `measureS`, `graceS`, `aiperfWorkers`,
  `recordProcessors`, `cpu`, `memory`. aiperf rebuilds the AgentX traces
  before profiling, which takes minutes and varies by region, so every region
  measures the same absolute window: `windowOffsetS` after the shared start,
  for `measureS`. `profilingS` must keep profiling running past that window
  in every region.

### Run

```bash
B="python3 bench.py --config bench.local.json --evidence evidence"
$B nodegroup up --region us-east-2          # repeat per region; wait for the stack
$B backend --region us-east-2
$B control --region us-east-2 --policy round-robin
$B hub > hub.json                            # apply to the Karmada API
$B pods --region us-east-2
$B tcp-check                                 # every region; must pass

START=$(( $(date +%s) + 600 ))               # time for every Job to install aiperf
$B loadgen up --region us-east-2 --label rr-768 --concurrency 192 --start-at $START
# ...after aiperf finishes:
$B collect --region us-east-2 --label rr-768
$B table --label rr-768
$B loadgen down --region us-east-2 --label rr-768
```

Switch policies by re-running `control` with the other `--policy`; only the
ingress Deployment rolls. `ping` and `probe --out FILE` measure the round
trip and the first-token cost of each cross-region path, with the bench idle.
`collect` reads the Job's `BENCH_SUMMARY` line and a per-container CPU sample
from the bench nodes over SSM. Delete the `global-router-bench-*` stacks and
the hub objects when finished.

### Session-aware policies and a daytime load cycle

`--policy rendezvous` and `--policy affinity` keep each session on one pool and
prefer the region a request entered. They read each pool's load from its Pool
Relay, so the pools must run it:

- `relay: "pool"` runs `python -m dynamo.pool_relay` from the planner image,
  one WAN port per pool, and makes the frontends publish serving load. `hub`
  then exports only that port. The relay drops a pool's KV filter if it
  overflows, so `backend` refuses mocker KV sizes that could: keep
  `workerReplicas` × `--num-gpu-blocks-override` under 75% of 1,048,576.
- `routerMode: "kv"` routes inside each pool by KV overlap, so a session that
  stays on a pool reuses its cache.
- Each region's `utcOffset` places it on the daily curve.

With `load.schedule: "daytime"`, a load-generator Job replays one compressed
day instead of a fixed concurrency:

- `dayS` (one day), `peakQps`, `nightFraction`, `peakHour` (local time),
  `startHourUtc`, `maxInFlight` (a safety cap), `graceS`, and `cacheBust` (an
  aiperf cache-bust target, for example `system_suffix`). Every region's rate
  follows its own local time, so regions take turns being busy.
- Requests carry `X-Dynamo-Session-ID`. Before the shared start, a one-request
  run builds aiperf's dataset cache, so every region's day starts within
  seconds of the others; `start_skew_s` in the summary records the offset.
- The whole day is measured. `collect` adds a `BENCH_HOURLY` breakdown per
  simulated hour (requests sent and completed, TTFT, and the ingress's routing
  counts), and `table` adds the share of requests kept in their region and the
  share of a session's later turns that went to the same pool.

`schedule: "ramp"` with `rampFrom` and `rampTo` raises every region's rate in
a straight line instead, to find capacity. `backend --restart TOKEN` rolls
every pool Deployment so a run starts with empty caches. `router-stats`
prints each ingress's routing counts, pool loads, and round trips.
