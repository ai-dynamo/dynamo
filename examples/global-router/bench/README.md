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

### Configure

Write an operator-local JSON file outside Git with:

- `namespace`, `model`, `workerReplicas`, `frontendCpu`, `routerReplicas`, `routerCpu`.
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

START=$(( $(date +%s) + 600 ))               # time for every Job to install aiperf
$B loadgen up --region us-east-2 --label rr-768 --concurrency 192 --start-at $START
# ...after aiperf finishes:
$B collect --region us-east-2 --label rr-768
$B table --label rr-768
$B loadgen down --region us-east-2 --label rr-768
```

Switch policies by re-running `control` with the other `--policy`; only the
ingress Deployment rolls. `collect` reads the Job's `BENCH_SUMMARY` line and a
per-container CPU sample from the bench nodes over SSM. Delete the
`global-router-bench-*` stacks and the hub objects when finished.
