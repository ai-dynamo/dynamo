<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend vs Gateway+EPP A/B benchmark

This harness compares the two ways Dynamo can sit in front of its workers. It runs on GPU-free
mocker workers, for both aggregated and disaggregated serving.

| Side | Request path | Who picks the worker |
|---|---|---|
| **A: `frontend`** | client → Dynamo Frontend → worker | KV router inside the Frontend (`DYN_ROUTER_MODE=kv`) |
| **B: `gateway`** | client → Gateway (agentgateway) → EPP → Frontend sidecar (direct mode) in the chosen pod → worker | KV router inside the EPP, Dynamo runtime mode |

It reports TTFT, inter-token latency (ITL), end-to-end latency, output-token throughput and request
throughput for each side, plus B's delta against A with a confidence interval and a verdict.

```text
A  aiperf ──HTTP──▶ Frontend (tokenize + KV route) ──TCP──▶ mocker
B  aiperf ──HTTP──▶ Gateway ──ext_proc──▶ EPP (tokenize + KV route)
                      └──HTTP──▶ sidecar Frontend (tokenize, direct) ──TCP──▶ mocker
```

## Why mockers, and why a cluster anyway

The mocker (`python3 -m dynamo.mocker`) registers with Dynamo discovery, publishes KV events, and
simulates prefill and decode timing on the CPU. A real GPU backend saturates long before the routing
tier does, so it would hide the cost being measured. The mocker keeps the backend cheap and
deterministic, which makes routing overhead and routing quality visible.

Side B still needs Kubernetes. A Dynamo-mode EPP will not start without a Kubernetes API server: it
watches worker pods to map each worker ID, a hash of the pod name, to the pod's IP. The Gateway
itself is a Gateway API object, too. Mockers remove the GPUs, not the cluster. Every component here
runs on CPU, so a local [minikube](https://minikube.sigs.k8s.io/) or [kind](https://kind.sigs.k8s.io/)
cluster is enough to run the whole comparison.

## Prerequisites

- `kubectl` and `helm` on the machine that drives the benchmark.
- `python3` 3.10 or later. The harness uses only the standard library, and aiperf runs inside
  the cluster.
- A Kubernetes cluster. Pods must be able to pull from `nvcr.io` and download the tokenizer from
  Hugging Face. About 16 CPUs covers the defaults. GPUs are not used.

## Quick start on a local cluster

```bash
minikube start -p gateway-ab --driver=docker --cpus=16 --memory=48g

# Dynamo platform 1.5.1, Gateway API + Inference Extension CRDs, agentgateway, and a Gateway.
KUBE_CONTEXT=gateway-ab bash benchmarks/gateway_ab/setup_cluster.sh

# Deploy, measure, report, and tear down both topologies (about 20 minutes with defaults on a 16-CPU minikube).
python3 -m benchmarks.gateway_ab --context gateway-ab all
```

Run all `python3 -m benchmarks.gateway_ab` commands from the repository root. Results land in
`gateway_ab_results/<UTC timestamp>/`, and the report also prints to stdout.

## On an existing cluster

If the cluster already runs a Dynamo operator, skip the platform install:

```bash
KUBE_CONTEXT=my-cluster SKIP_PLATFORM=1 BENCH_NAMESPACE=my-ns \
  bash benchmarks/gateway_ab/setup_cluster.sh

python3 -m benchmarks.gateway_ab --context my-cluster --namespace my-ns preflight
python3 -m benchmarks.gateway_ab --context my-cluster --namespace my-ns all
```

`preflight` checks that the namespace, the three CRDs (`DynamoGraphDeployment`, `InferencePool`,
`HTTPRoute`), and a Programmed Gateway are all present, and that any secrets you name exist.

The operator must be 1.5.0 or later to run the native Rust EPP. Pass `--hf-secret` with a Secret
holding `HF_TOKEN` for a gated model or to avoid Hugging Face rate limits, and
`--image-pull-secret` for a private registry.

## Disaggregated serving needs an EPP that recognizes mocker prefill workers

EPP builds up to and including 1.5.1 recognize a prefill worker only by the legacy
`ModelType::Prefill` marker bit on its model card. vLLM still emits that bit for compatibility,
but the mocker registers its prefill workers with `worker_type: prefill` alone. Against mocker
workers, those EPP builds therefore never activate prefill routing. Every disaggregated request
through the Gateway then fails at the decode sidecar with `Prefill worker ID required in Direct
routing mode`. The aggregated topology and the whole `frontend` side are unaffected, because the
Frontend resolves the worker role from `worker_type`.

The EPP fix ships with this harness: `deploy/inference-gateway/ext-proc/src/epp.rs` now resolves
the role through `ModelDeploymentCard::effective_worker_type()`. Until a release includes it, run
the disaggregated comparison with a frontend image whose `/epp` binary is built from a branch that
has the fix. The Python Frontend in that image is unchanged, so both sides still run the same
Frontend code. One way to build it, on a host whose glibc matches the image (Ubuntu 24.04 for
1.5.x):

```bash
cargo build --release -p dynamo-ext-proc
mkdir -p /tmp/epp-image && cp target/release/dynamo-ext-proc /tmp/epp-image/epp
cat > /tmp/epp-image/Dockerfile <<'EOF'
FROM nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.5.1
COPY --chmod=755 epp /epp
EOF
docker build -t dynamo-frontend:1.5.1-epp-fix /tmp/epp-image
minikube -p gateway-ab image load dynamo-frontend:1.5.1-epp-fix   # or push to your registry

python3 -m benchmarks.gateway_ab --context gateway-ab \
  --frontend-image dynamo-frontend:1.5.1-epp-fix all
```

When the EPP misses the prefill workers, the disaggregated topology fails its smoke test with that
diagnosis before any measurement runs. `all` still measures and reports the other topologies, then
exits `1`.

## Step by step

`all` chains these commands. Run them separately to keep deployments up between iterations:

```bash
B="python3 -m benchmarks.gateway_ab --context gateway-ab"
$B render                       # write the manifests to gateway_ab_manifests/ for review
$B deploy                       # loadgen pod + both sides of every topology, smoke-tested
$B --results out/run1 run       # measure; re-running the same command resumes
$B --results out/run1 report    # recompute summary.json and report.md
$B teardown                     # delete everything labeled app.kubernetes.io/part-of=gateway-ab
```

`all` deploys one topology at a time and tears each down before the next, so agg and disagg never
compete for CPU. With `--keep`, it leaves everything running.

`deploy` and `all` always recreate a topology's deployments rather than updating them in place.
Every run then starts with cold caches and router state. Recreating also avoids an EPP failure
seen during in-place updates: when one update rolls both the EPP and its workers, the new EPP can
bind to the outgoing workers' generation and keep returning `503 no endpoints available` after
they drain.

Re-running `run` with the same `--results` resumes it. Only valid measurements count as done.
Raising `--rounds` adds rounds. Any other workload or deployment change is refused, because it
would mix incomparable rounds into one comparison.

## What makes the comparison fair

- **Identical workers.** One function builds both DynamoGraphDeployments. Mocker arguments, worker
  counts, images, and resource requests match, and B adds only the direct-mode sidecar and the EPP.
  A unit test enforces this.
- **Same router, same image.** The Frontend and the EPP both run the Dynamo KV router from the same
  `dynamo-frontend` image, with equal CPU requests. The EPP's `DYN_KV_CACHE_BLOCK_SIZE` is set to the
  mocker's `--block-size`. A mismatch would change prefix hashes and degrade B's routing without any
  error.
- **Same client.** One loadgen pod runs aiperf against both sides with the same arguments. Only the
  URL differs, and a unit test enforces that too. Both sides receive the routing headers. A ignores
  them.
- **Same requests.** Each round uses one seed for both sides, so A and B see the same prompts.
  Prompts share prefixes (`--prefix-length`, `--num-prefixes`), which gives the KV router something
  to route on. With unique random prompts, a KV router reduces to a load balancer and most of what
  the EPP does goes unmeasured.
- **Drift cancellation.** Within a round, the side that runs first alternates by load point, and it
  flips each round (ABBA). Slow drift such as cache warmth, noisy neighbours, or node temperature
  cannot land on only one side.
- **Primed before measuring.** Before round 0, each side gets one unrecorded run at the highest
  load point. Its seed is disjoint from the measured rounds, so it warms connections and code
  paths but not the prefixes those rounds use. Without it, the first measured point carries a
  cold-start cost, which lands on whichever side runs first. Pass `--no-prime` to skip it.
- **No CPU limits.** Containers set CPU requests but no limits, so CFS throttling cannot add latency
  spikes that would swamp the difference being measured.
- **Valid runs only.** A run counts only if every request succeeded and none went missing. Failed or
  partial runs appear under "Excluded runs" and are left out of the statistics. A failed measurement
  is retried once, then recorded as missing.

## Choosing the load

The harness offers two load modes:

- **Open loop (`--request-rate`, the default: `6,14` req/s).** Requests arrive as a Poisson process
  at a fixed average rate, regardless of how fast the server answers. This is the mode for TTFT.
  The default rates sit near 30% and 70% of what the default mocker deployments sustain (about
  19 req/s aggregated and 22 req/s disaggregated). The mocker's capacity comes from its simulated
  engine, not the host, so these rates hold on any cluster that is not CPU-starved. Rescale them
  when you change worker counts or `--speedup-ratio`.
- **Closed loop (`--concurrency`).** A fixed number of clients each send the next request as soon
  as the previous one finishes. This mode is self-limiting and cannot overload the deployment, so
  use it to find where throughput saturates after changing the deployment shape. Do not use it to
  compare TTFT. Closed-loop clients phase-lock with the engine's batching. On a 2-worker
  aggregated mocker deployment at c=8, TTFT split into separate modes near 47, 78, and 95 ms,
  according to how many prefills were batched together. The median jumped between modes from one
  round to the next, and the TTFT p50 interval spanned about 70 percentage points. At 6 and
  14 req/s on the same deployment, the interval was about 2 points wide.

To rescale after a deployment change, find the saturated rate first, then measure open-loop at
about 30% and 70% of it:

```bash
python3 -m benchmarks.gateway_ab --context gateway-ab --concurrency 32 --rounds 1 all  # saturated req/s
python3 -m benchmarks.gateway_ab --context gateway-ab --request-rate 5,12 all
```

Keep the rates below saturation. Past capacity, queues grow without bound for the whole run, and
latency measures the run length rather than the routing path.

## Reading the report

`report.md` has one table per topology. Each row gives A, B, Δ (B relative to A), a 95% confidence
interval, and a verdict. Negative latency Δ means B is faster. Latency rows cover the mean, p50,
p90, and p99. When the distribution is multi-modal, trust the mean more than the median.

Latency intervals come from a two-level bootstrap. It resamples rounds first, then requests within
each chosen round. Run-to-run variance is usually larger than request-to-request variance, and a
per-request bootstrap would hide it. That is why the default is three rounds and why one round
cannot produce a trustworthy interval. Throughput is a single number per run, so its interval is the
range of all pairwise round deltas.

Verdicts read the interval against `--min-effect` (default 2%):

| Verdict | Interval |
|---|---|
| `B better` / `B worse` | Entirely beyond ±min-effect, on one side |
| `B slightly better` / `B slightly worse` | Excludes zero but reaches inside ±min-effect. The difference is real but may be negligible. |
| `same` | Includes zero and lies within ±min-effect |
| `inconclusive` | Includes zero but reaches past ±min-effect. The data cannot separate a negligible difference from a large one. Add rounds or use a quieter cluster. |
| `n/a (1 round)` | Fewer than two valid rounds on a side. With no between-round variance, the interval is too narrow to support a verdict. |

The per-round coefficient of variation printed under each table is the A/A noise floor for that
cluster. Treat any Δ smaller than it with suspicion.

`summary.json` holds the same data in machine-readable form, along with `meta.json`: images,
aiperf version, repository commit, node list, and the full configuration.

## Running it regularly

Use a dedicated namespace on a cluster whose node pool stays the same from run to run. Then
schedule the `all` command, for example from cron on a host with cluster credentials:

```bash
# Nightly: fail if B's TTFT is worse than A's by more than 10% with confidence.
python3 -m benchmarks.gateway_ab --context perf-cluster --namespace gateway-ab \
  --results "/srv/gateway-ab/$(date -u +%Y%m%d)" \
  --max-ttft-regression 10 all
```

`--max-ttft-regression` makes the command exit `3` when the lower bound of the TTFT mean, p50, or
p90 interval exceeds that percentage at any load point, so noise alone cannot trip it. p99 over a
few rounds is too noisy to gate on.

| Exit | Meaning |
|---|---|
| `0` | Every point compared, and no regression gate tripped |
| `1` | Incomplete: a point has no comparison, a run was excluded as invalid, or a step failed. The report still covers everything that was measured. |
| `2` | Preflight failed |
| `3` | The TTFT regression gate tripped |

Keep every run directory under one parent and tabulate B's TTFT Δ across them. This needs no
cluster access:

```bash
python3 -m benchmarks.gateway_ab trend --history /srv/gateway-ab            # TTFT p50
python3 -m benchmarks.gateway_ab trend --history /srv/gateway-ab --quantile p90
```

The threshold gate catches a sudden jump. The trend table catches a slow drift that stays under the
threshold in every individual run.

To test a build, pass its images. The EPP binary ships inside the frontend image, so one flag swaps
it on both sides, and the comparison stays at a single commit:

```bash
python3 -m benchmarks.gateway_ab --context perf-cluster \
  --frontend-image registry.example.com/dynamo-frontend:pr-1234 \
  --worker-image registry.example.com/dynamo-planner:pr-1234 all
```

To compare two builds, run each and compare their reports. Run them back to back on the same nodes.

## Knobs

| Flag | Default | Notes |
|---|---|---|
| `--topologies` | `agg,disagg` | |
| `--request-rate` | `6,14` | Open-loop Poisson load points in requests/s, comma-separated. One measurement per value per round. |
| `--concurrency` | — | Closed-loop load points instead, for example `8,32`. Mutually exclusive with `--request-rate`. |
| `--rounds` | `3` | Use at least 3. Intervals need more than one round. |
| `--isl` / `--osl` | `1024` / `128` | Synthetic input and output tokens per request. |
| `--prefix-length` / `--num-prefixes` | `512` / `8` | Shared-prefix structure. Set `--prefix-length 0` for unique prompts. |
| `--requests` / `--warmup-requests` | about 40 s / 5 s of arrivals (min 200 / 10); closed loop `max(200, 10×c)` / `max(10, 2×c)` | Per point. |
| `--agg-workers` | `2` | |
| `--prefill-workers` / `--decode-workers` | `2` / `2` | |
| `--block-size` | `16` | Applied to the mocker and the EPP. |
| `--speedup-ratio` | `1.0` | Mocker time compression. Raise it to stress the routing tier rather than the simulated engine. |
| `--mocker-arg` | — | Extra mocker flag, repeatable, for example `--mocker-arg=--num-gpu-blocks-override --mocker-arg=4096`. |
| `--router-cpu` / `--worker-cpu` / `--sidecar-cpu` / `--loadgen-cpu` | `2` / `1` / `1` / `4` | CPU requests. |
| `--dynamo-version` | `1.5.1` | Sets the default `dynamo-frontend` and `dynamo-planner` image tags. |
| `--min-effect` | `2.0` | Percent below which a Δ is reported as `same`. |

## Caveats

- **Mocker timings are synthetic.** Compare A against B from the same run, never a mocker number
  against a GPU number. The mocker does not model GPU contention between prefill and decode, so
  treat absolute headroom as an upper bound.
- **B has more CPU in total.** B tokenizes in the EPP and again in each sidecar, while A tokenizes
  once in the Frontend. B also has a sidecar per worker pod and a gateway proxy. This difference is
  part of the architecture, and the benchmark measures it rather than hiding it. Keep it in mind
  when reading throughput.
- **The Gateway is shared.** All B-side pools use the Gateway in `--gateway-namespace`, and routes
  select a pool by the `X-Dynamo-AB-Target` header. Other traffic through that Gateway adds noise.
- **Local clusters are for wiring and trends.** On minikube or kind, the load generator, the
  gateway, and every pod share the host's cores. Use a real cluster for numbers you publish.
- **EPP log volume.** In aggregated mode, the EPP polls discovery for prefill workers once a
  second, and the discovery client logs every poll at `INFO` level. This costs little, but it shows
  in the EPP logs.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Preflight: CRD or Gateway missing | Run `setup_cluster.sh`. |
| Smoke to `gateway` fails with 404 | The HTTPRoute is not attached. Check `kubectl get httproute -n <ns> -o yaml` for `Accepted`, and confirm the Gateway's listener allows routes from your namespace. |
| Smoke to `gateway` fails with 500 or 503 | The EPP is not ready or found no workers. Check `kubectl logs -l nvidia.com/dynamo-component-type=epp`. |
| DGD never becomes Ready | Usually an image pull or Hugging Face download failure. Check `kubectl describe pod` and the worker logs, and pass `--hf-secret` when rate-limited. |
| Excluded runs with errors | Look in the loadgen pod: `kubectl exec <prefix>-loadgen -- cat /results/<topology>/<side>/<load>/r<R>.log` (`<load>` is `c8`, `rps14`, and so on). |

## Files

| File | Purpose |
|---|---|
| `setup_cluster.sh` | Installs the Dynamo platform and the Gateway API Inference Extension stack. |
| `cli.py` | Commands and orchestration. |
| `manifests.py` | DynamoGraphDeployment, HTTPRoute, and loadgen manifests for both sides. |
| `loadgen.py` | aiperf command and in-pod result extraction. |
| `stats.py` | Percentiles, hierarchical bootstrap, verdicts. Standard library only. |
| `report.py` | `summary.json` and `report.md`. |
