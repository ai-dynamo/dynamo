<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend vs Gateway+EPP A/B benchmark

This harness compares the two ways Dynamo can sit in front of its workers. It runs on GPU-free
mocker workers, for both aggregated and disaggregated serving.

| Side | Request path | Who picks the worker |
|---|---|---|
| **A: `frontend`** | client → Gateway (agentgateway) → Dynamo Frontend → worker | KV router inside the Frontend (`DYN_ROUTER_MODE=kv`) |
| **B: `epp`** | client → Gateway (agentgateway) → EPP → Frontend sidecar (direct mode) in the chosen pod → worker | KV router inside the EPP, Dynamo runtime mode |

Both sides sit behind the same Gateway, so what separates them is the EPP, not the gateway hop. A
production Frontend usually sits behind an ingress or gateway too. With `--with-direct`, a third arm
measures what the gateway itself costs; see
[Measuring the gateway's cost](#measuring-the-gateways-cost).

It reports TTFT, inter-token latency (ITL), end-to-end latency, output-token throughput and request
throughput for each side, plus B's delta against A with a confidence interval and a verdict.

```text
A  aiperf ──HTTP──▶ Gateway ──HTTP──▶ Frontend (tokenize + KV route) ──TCP──▶ mocker
B  aiperf ──HTTP──▶ Gateway ──ext_proc──▶ EPP (tokenize + KV route)
                      └──HTTP──▶ sidecar Frontend (direct, EPP's tokens) ──TCP──▶ mocker
```

## Why mockers, and where it runs

The mocker (`python3 -m dynamo.mocker`) registers with Dynamo discovery, publishes KV events, and
simulates prefill and decode timing on the CPU. A real GPU backend saturates long before the routing
tier does, so it would hide the cost being measured. The mocker keeps the backend cheap and
deterministic, which makes routing overhead and routing quality visible.

There are two backends. They share the workload, the measurement loop, the statistics, and the
report:

| | `--backend kubernetes` (default) | `--backend local` |
|---|---|---|
| Needs | A cluster (minikube or kind is enough), `kubectl`, `helm` | Docker only |
| Gateway | agentgateway, one HTTPRoute per side, selected by the `X-Dynamo-AB-Target` header | Envoy, one route per side, selected by the same header |
| Side A | DGD: Frontend + mocker workers, HTTPRoute to the Frontend's Service | Containers: Frontend + mocker workers, Envoy route with `ext_proc` off |
| Side B | DGD with an EPP, InferencePool, a direct-mode sidecar per worker pod | The EPP, one direct-mode Frontend, mocker workers |
| Worker id → address | The EPP's Kubernetes pod reflector | `DYN_EPP_STATIC_ENDPOINT` |
| Setup per topology | Minutes | Seconds |
| Use it for | Numbers you publish; anything that touches the operator, the Gateway, or pod discovery | Fast iteration on EPP and Frontend code, and frequent regression runs |

Kubernetes is needed for B only because a Dynamo-mode EPP maps each picked worker ID, a hash of the
pod name, to the pod's address through a pod reflector. The local backend replaces that one step.
See [Running without Kubernetes](#running-without-kubernetes).

## Prerequisites

- `kubectl` and `helm` on the machine that drives the benchmark.
- `python3` 3.10 or later. The harness uses only the standard library, and aiperf runs inside
  the cluster.
- A Kubernetes cluster. Pods must be able to pull from `nvcr.io` and download the tokenizer from
  Hugging Face. GPUs are not used. At the default CPU requests, `all` needs about 20 allocatable
  CPUs while it measures the disaggregated topology (A 6, B 10, the loadgen 4). `deploy` keeps
  both topologies up and needs about 30, and `--with-direct` adds 4 for agg and 6 for disagg.

## Quick start on a local cluster

```bash
minikube start -p gateway-ab --driver=docker --cpus=20 --memory=48g

# Dynamo platform 1.5.1, Gateway API + Inference Extension CRDs, agentgateway, and a Gateway.
KUBE_CONTEXT=gateway-ab bash benchmarks/gateway_ab/setup_cluster.sh

# Deploy, measure, report, and tear down both topologies (about 20 minutes with defaults).
python3 -m benchmarks.gateway_ab --context gateway-ab all
```

Run all `python3 -m benchmarks.gateway_ab` commands from the repository root. Results land in
`benchmarks/results/gateway_ab/<UTC timestamp>/`, which the repository ignores, and the report
also prints to stdout.

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
Credentials limited to the namespace skip the CRD check, and need `--gateway-url` when they cannot
read the Gateway.

The operator must be 1.5.0 or later to run the native Rust EPP. Pass `--hf-secret` with a Secret
holding `HF_TOKEN` for a gated model or to avoid Hugging Face rate limits, and
`--image-pull-secret` for a private registry.

## Disaggregated serving needs an EPP that recognizes mocker prefill workers

EPP builds up to and including 1.5.1 recognize a prefill worker only by the legacy
`ModelType::Prefill` marker bit on its model card. vLLM still emits that bit for compatibility,
but the mocker registers its prefill workers with `worker_type: prefill` alone. Against mocker
workers, those EPP builds therefore never activate prefill routing. Every disaggregated request
through the EPP then fails at the decode sidecar with `Prefill worker ID required in Direct
routing mode`. The aggregated topology and the whole `frontend` side are unaffected, because the
Frontend resolves the worker role from `worker_type`.

The EPP fix ships with this harness: `deploy/inference-gateway/ext-proc/src/epp.rs` now resolves
the role through `ModelDeploymentCard::effective_worker_type()`. Until a release includes it, run
the disaggregated comparison with a frontend image whose `/epp` binary is built from a branch that
has the fix. Only `/epp` changes, so the HTTP and tokenizer code on both sides stays the same. The
KV router does not: B's comes from the branch build and A's from the 1.5.1 Frontend, so any router
change between the two lands in the deltas. For an architecture-only comparison, build the whole
frontend image from one commit. One way to swap only `/epp`, on a host whose glibc matches the
image (Ubuntu 24.04 for 1.5.x):

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

## Running without Kubernetes

`--backend local` runs both sides in Docker on one host, with no cluster and no host ports. Each
topology gets its own Docker network and its own file-discovery volume:

```text
A  aiperf ─▶ Envoy ─▶ Frontend (KV router) ─▶ mocker workers
B  aiperf ─▶ Envoy ─ext_proc─▶ EPP (Dynamo runtime mode, KV router)
               └─▶ direct-mode Frontend ─▶ the mocker worker the EPP picked
```

- **Envoy** stands in for the shared Gateway, with one route per side selected by the
  `X-Dynamo-AB-Target` header. B's route is configured the way Gateway API Inference Extension
  gateways program it: full-duplex streamed `ext_proc` in both directions, and an
  original-destination cluster that forwards to the address the EPP returns in
  `x-gateway-destination-endpoint`. A's route turns `ext_proc` off and forwards to A's Frontend.
- **The EPP** runs its production code: tokenization, KV-aware selection from the workers' KV
  events, prefill routing, load bookkeeping from the response stream, and the `ext_proc` protocol.
- **The one substitution:** the EPP runs with `DYN_EPP_STATIC_ENDPOINT` set to the direct-mode
  Frontend, in place of its Kubernetes pod reflector. That Frontend forwards each request to the
  worker and prefill worker named in the EPP's routing headers.

That knob is new, so it needs an EPP built from this branch. Build it on a host whose glibc is no
newer than the image's (Ubuntu 24.04 for 1.5.x), and pass the binary:

```bash
cargo build --release -p dynamo-ext-proc
python3 -m benchmarks.gateway_ab --backend local \
  --epp-binary target/release/dynamo-ext-proc all
```

The binary is mounted into the `dynamo-frontend` image, so everything else stays on the released
images. The run pulls `dynamo-frontend`, `dynamo-planner`, and `envoyproxy/envoy` on first use, and
caches the tokenizer in the `gwab-hf` Docker volume.

> [!WARNING]
> With `--epp-binary`, B's KV router is the binary's while A's is the one in `--frontend-image`.
> Unless that image is built from the same commit, the deltas include every router change between
> the two builds, not only the cost of the routing path. Preflight warns about this. Pass a
> frontend image built from the binary's commit to measure the architecture alone.

Containers, networks, and volumes are named and labeled after `--name-prefix`, so runs with
different prefixes can share a Docker host. Two runs with the same prefix tear each other down.

Compared with the Kubernetes backend:

- **Not exercised:** the pod reflector, the operator, InferencePool, HTTPRoute, and agentgateway.
  Envoy and agentgateway can differ in how they pool connections and acknowledge data, so a cost
  seen behind one gateway may not appear behind the other.
- **One Frontend instead of a sidecar per pod:** a single direct-mode Frontend serves every B worker.
- **Shared CPU:** the load generator and every container share the host's CPUs. Trust the deltas,
  and measure on a cluster for published absolute numbers.
- **Isolation:** each container gets a 1M open-file limit, matching Kubernetes defaults rather than
  Docker's 1024, which the EPP exceeds in disaggregated mode.

## Step by step

`all` chains these commands. Run them separately to keep deployments up between iterations:

```bash
B="python3 -m benchmarks.gateway_ab --context gateway-ab"
$B render                       # write the manifests to benchmarks/results/gateway_ab/manifests/
$B deploy                       # loadgen pod + every arm of every topology, smoke-tested
$B --results out/run1 run       # measure; re-running the same command resumes
$B --results out/run1 report    # recompute summary.json and report.md
$B teardown                     # delete this prefix's objects (app.kubernetes.io/instance=<prefix>)
```

`all` deploys one topology at a time and tears each down before the next, so agg and disagg never
compete for CPU. With `--keep`, it leaves everything running.

`deploy` stamps each deployment with a fingerprint of the flags that shape it: images, worker
counts, mocker arguments, `--with-direct`, and, locally, the EPP binary's content. `run` refuses to measure a
deployment whose fingerprint differs from its own flags, so results are never labeled with flags
other than the ones deployed. Give each step the same flags, as `$B` does above.

`deploy` and `all` always recreate a topology's deployments rather than updating them in place.
Every run then starts with cold caches and router state. Recreating also avoids an EPP failure
seen during in-place updates: when one update rolls both the EPP and its workers, the new EPP can
bind to the outgoing workers' generation and keep returning `503 no endpoints available` after
they drain.

Re-running `run` with the same `--results` resumes it. A round counts as done only when every arm
measured it validly; otherwise every arm measures it again, so no pair mixes two sessions. Raising
`--rounds` adds rounds. Any other change is refused, because it would mix incomparable rounds into
one comparison: the workload, `--no-prime`, the deployment settings, the EPP binary's content, an
image id or the Docker host (local backend), the cluster context, namespace, or `--gateway-url`,
and the harness version. A directory that holds no measurement yet takes the new settings.
Kubernetes image tags are compared by name, so do not re-push a tag mid-run.

## What makes the comparison fair

- **Same gateway.** Both sides go through the same Gateway. Their routes differ only in the backend
  and in the value of the `X-Dynamo-AB-Target` header that selects them, so the gateway's own cost,
  which a Frontend behind an ingress pays as well, cancels out of the comparison. Each route matches
  the exact chat path and both headers the requests carry, and the target value includes the
  namespace, so neither a broader route nor another run's route on a shared Gateway can take the
  traffic. A unit test checks the routes.
- **Identical workers.** One function builds every DynamoGraphDeployment. Mocker arguments, worker
  counts, images, and resource requests match, and B adds only the direct-mode sidecar and the EPP.
  A unit test enforces this.
- **Same router, when built from one commit.** On Kubernetes the Frontend and the EPP both run the
  Dynamo KV router from the same `dynamo-frontend` image, with equal CPU requests. Both read the KV
  block size from the workers' model cards, so their prefix hashes agree. An EPP taken from another
  build (`--epp-binary`, or an image with a swapped `/epp`) runs that build's router, and the
  deltas then include the difference between the two builds.
- **Same client.** One loadgen pod runs aiperf against both sides with the same arguments and the
  same URL, the Gateway's. Only the target header's value differs, and a unit test enforces that
  too.
- **Same requests.** Each round and load point has its own seed, the same for both sides, so A and
  B see the same prompts. Seeds differ across load points because aiperf draws its prompts in order
  from the seed; with a shared seed the second load point would replay the first one's prompts and
  find them cached. Prompts share prefixes (`--prefix-length`, `--num-prefixes`), which gives the KV
  router something to route on. With unique random prompts, a KV router reduces to a load balancer
  and most of what the EPP does goes unmeasured.
- **Drift cancellation.** Within a round, the side that runs first alternates by load point, and it
  flips each round (ABBA). Slow drift such as cache warmth, noisy neighbours, or node temperature
  cannot land on only one side.
- **Primed before measuring.** Before round 0, each arm gets one unrecorded run at the highest
  load point. Its seed is far from every measured one, so it warms connections and code paths
  but not the prefixes those rounds use. Without it, the first measured point carries a
  cold-start cost, which lands on whichever side runs first. Pass `--no-prime` to skip it.
- **No CPU limits.** Containers set CPU requests but no limits, so CFS throttling cannot add latency
  spikes that would swamp the difference being measured.
- **Valid runs only.** A run counts only if every request succeeded and none went missing. Failed or
  partial runs appear under "Excluded runs" and are left out of the statistics. A failed measurement
  is not retried: it has already sent the round's prompts, and a replay would find them cached.
  Re-running with the same `--results` measures the round again on every arm.

## Measuring the gateway's cost

`--with-direct` adds a third arm, `direct`: a second copy of A's deployment that aiperf reaches
without the Gateway. The report then adds a "Gateway cost" table that compares A through the
Gateway with this copy. Its Δ is what the gateway hop costs a Frontend deployment.

- **Why a copy.** Every arm replays the same prompts in a round. Against A's own workers, whichever
  arm ran second would find those prompts in A's prefix caches and look faster.
- **Order.** A runs between the other two arms, and the order reverses at every point, so A always
  runs right next to both arms it is compared with.
- **Cost.** One more deployment per topology, which adds about half again as many runs and, on
  Kubernetes, A's CPU requests once more.

Earlier versions of this harness compared B with A reached directly, which charged the gateway's
cost to the EPP. That comparison is also how the harness found a Frontend bug: the Frontend's HTTP
server left Nagle's algorithm on, and behind agentgateway about a quarter of disaggregated first
tokens were held for about 40 ms
([#16859](https://github.com/ai-dynamo/dynamo/issues/16859)). With both sides behind the gateway,
both would show a problem like that, and the A/B table would not. The gateway-cost table would.

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

The round is the unit of measurement. Each round reduces to one value per side, aiperf's own
statistic for that run, such as its TTFT p90 or its throughput. A and B are the geometric means of
those values over the rounds
measured validly on both sides, and Δ is B against A as those columns show it. The interval is a
paired t-interval over the per-round ratios B/A: both sides replay the same seed in a round, back
to back, so pairing cancels the round-to-round variation they share (the seed's arrival pattern,
drift on the node), and the t distribution keeps the interval at 95% with as few as three rounds.
Requests within a round are not independent replications, and treating them as such, or
bootstrapping over three rounds, gives intervals that are too narrow. More rounds narrow the
interval; three gives verdicts on clear differences, and five or more resolves `same` more often.

Under open-loop load below saturation, each side delivers the rate it is offered, so throughput
rows carry `n/a (open loop)` instead of a verdict. Use the closed-loop mode to compare capacity.

Verdicts read the interval against `--min-effect` (default 2%):

| Verdict | Interval |
|---|---|
| `B better` / `B worse` | Entirely beyond ±min-effect, on one side |
| `B slightly better` / `B slightly worse` | Excludes zero but reaches inside ±min-effect. The difference is real but may be negligible. |
| `same` | Includes zero and lies within ±min-effect |
| `inconclusive` | Includes zero but reaches past ±min-effect. The data cannot separate a negligible difference from a large one. Add rounds or use a quieter cluster. |
| `n/a (too few rounds)` | Fewer than two rounds measured validly on both sides. One pair has no spread to estimate an interval from. |
| `n/a (open loop)` | Throughput rows under open-loop load, where each side delivers the rate it is offered |

With `--with-direct`, a "Gateway cost" section follows, with the same columns for the direct arm
against A through the gateway. Its verdicts read `gateway worse`, `gateway better`, and so on. The
regression gate and `trend` cover only B against A.

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
p90 interval exceeds that percentage at any load point, so a delta within the run-to-run noise does
not trip it. p99 over a few rounds is too noisy to gate on. A gated statistic without a verdict,
for example with one round, fails the run with exit `1` rather than passing unchecked.

| Exit | Meaning |
|---|---|
| `0` | Every point compared, and no regression gate tripped |
| `1` | Incomplete: a point has no comparison, a run was excluded as invalid, the regression gate could not be evaluated, or a step failed. The report still covers everything that was measured. |
| `2` | Nothing ran: preflight failed, the arguments were invalid, or `--results` holds runs measured with other settings |
| `3` | The TTFT regression gate tripped |
| `143` | Stopped by SIGTERM, after tearing down what it deployed |

Keep every run directory under one parent and tabulate B's TTFT Δ across them. This needs no
cluster access:

```bash
python3 -m benchmarks.gateway_ab trend --history /srv/gateway-ab            # TTFT p50
python3 -m benchmarks.gateway_ab trend --history /srv/gateway-ab --quantile p90
```

The threshold gate catches a sudden jump. The trend table catches a slow drift that stays under the
threshold in every individual run. A run measured with another backend, workload, or worker shape
than the newest one is marked `†`, since its deltas are not comparable with the rest of the column.

To test a build, pass its images. The EPP binary ships inside the frontend image, so one flag swaps
it on both sides, and the comparison stays at a single commit as long as that image is built from
one:

```bash
python3 -m benchmarks.gateway_ab --context perf-cluster \
  --frontend-image registry.example.com/dynamo-frontend:pr-1234 \
  --worker-image registry.example.com/dynamo-planner:pr-1234 all
```

To compare two builds, run each and compare their reports. Run them back to back on the same nodes.

## Knobs

| Flag | Default | Notes |
|---|---|---|
| `--backend` | `kubernetes` | `local` runs everything in Docker on this host. |
| `--epp-binary` | — | Local backend: an EPP binary to mount in place of the image's `/epp`. See the warning in [Running without Kubernetes](#running-without-kubernetes). |
| `--envoy-image` | `envoyproxy/envoy:v1.35.3` | Local backend: the gateway. |
| `--topologies` | `agg,disagg` | |
| `--request-rate` | `6,14` | Open-loop Poisson load points in requests/s, comma-separated. One measurement per value per round. |
| `--concurrency` | — | Closed-loop load points instead, for example `8,32`. Mutually exclusive with `--request-rate`. |
| `--rounds` | `3` | Use at least 3. Intervals need two rounds, and narrow as rounds are added. |
| `--isl` / `--osl` | `1024` / `128` | Synthetic input and output tokens per request. The input count includes the shared prefix. |
| `--prefix-length` / `--num-prefixes` | `512` / `8` | Shared-prefix structure. Set `--prefix-length 0` for unique prompts. |
| `--requests` / `--warmup-requests` | about 40 s / 5 s of arrivals (min 200 / 10); closed loop `max(200, 10×c)` / `max(10, 2×c)` | Per point. |
| `--agg-workers` | `2` | |
| `--prefill-workers` / `--decode-workers` | `2` / `2` | |
| `--block-size` | `16` | The mocker's KV block size. Both routers read it from the model cards. |
| `--speedup-ratio` | `1.0` | Mocker time compression. Raise it to stress the routing tier rather than the simulated engine. |
| `--mocker-arg` | — | Extra mocker flag, repeatable, for example `--mocker-arg=--num-gpu-blocks-override --mocker-arg=4096`. |
| `--router-cpu` / `--worker-cpu` / `--sidecar-cpu` / `--loadgen-cpu` | `2` / `1` / `1` / `4` | Kubernetes backend: CPU requests. |
| `--name-prefix` | `ab` | Names this run's deployments, containers, and labels. Give concurrent runs distinct prefixes. |
| `--with-direct` | off | Also deploy a copy of A and measure it without the gateway. See [Measuring the gateway's cost](#measuring-the-gateways-cost). |
| `--dynamo-version` | `1.5.1` | Sets the default `dynamo-frontend` and `dynamo-planner` image tags. |
| `--min-effect` | `2.0` | Percent below which a Δ is reported as `same`. |

## Caveats

- **Mocker timings are synthetic.** Compare A against B from the same run, never a mocker number
  against a GPU number. The mocker does not model GPU contention between prefill and decode, so
  treat absolute headroom as an upper bound.
- **B has more CPU in total.** B tokenizes in the EPP, and each sidecar renders the chat template
  again before reusing the EPP's token ids from `nvext.token_data`, while A does both once in the
  Frontend. B also has a sidecar per worker pod. This difference is part of the architecture, and
  the benchmark measures it rather than hiding it. Keep it in mind when reading throughput.
- **The Gateway is shared.** Both sides use the Gateway in `--gateway-namespace`, and routes select a
  side by the `X-Dynamo-AB-Target` header. Other traffic through that Gateway adds noise to both.
- **Local clusters are for wiring and trends.** On minikube or kind, the load generator, the
  gateway, and every pod share the host's cores. Use a real cluster for numbers you publish.
- **EPP log volume.** In aggregated mode, the EPP polls discovery for prefill workers once a
  second, and the discovery client logs every poll at `INFO` level. This costs little, but it shows
  in the EPP logs.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Preflight: CRD or Gateway missing | Run `setup_cluster.sh`. |
| Smoke to `frontend` or `epp` fails with 404 | The HTTPRoute is not attached. Check `kubectl get httproute -n <ns> -o yaml` for `Accepted`, and confirm the Gateway's listener allows routes from your namespace. |
| Smoke to `epp` fails with 500 or 503 | The EPP is not ready or found no workers. Check `kubectl logs -l nvidia.com/dynamo-component-type=epp`. |
| DGD never becomes Ready | Usually an image pull or Hugging Face download failure. Check `kubectl describe pod` and the worker logs, and pass `--hf-secret` when rate-limited. |
| Local: an EPP container exits with `Too many open files` | File discovery ran out of inotify instances, which are capped per user across the host. Raise them with `sudo sysctl fs.inotify.max_user_instances=1024`. Preflight warns when the limit is below 512. |
| Local: once the workers register, the EPP exits with `Failed to infer configuration` | The EPP lacks `DYN_EPP_STATIC_ENDPOINT` and looked for a Kubernetes API server. An EPP that has it logs `DYN_EPP_STATIC_ENDPOINT is set` at startup. Pass `--epp-binary` with a build from this branch. |
| `run`: deployed with other settings than these flags | Pass the flags `deploy` used, or run `deploy` again with the new ones. |
| Excluded runs with errors | Look in the loadgen pod: `kubectl exec <prefix>-loadgen -- cat /results/<topology>/<side>/<load>/r<R>.log` (`<load>` is `c8`, `rps14`, and so on). |

## Files

| File | Purpose |
|---|---|
| `setup_cluster.sh` | Installs the Dynamo platform and the Gateway API Inference Extension stack. |
| `cli.py` | Commands and orchestration. |
| `k8s_backend.py` | Kubernetes backend: DGDs, HTTPRoutes, loadgen pod, rollout checks. |
| `local_backend.py` | Docker backend: containers, Envoy config, EPP with a static endpoint. |
| `manifests.py` | DynamoGraphDeployment, HTTPRoute, and loadgen manifests for every arm. |
| `loadgen.py` | aiperf command, detached runs, and aiperf's per-run summary. |
| `stats.py` | Paired t-intervals and verdicts. Standard library only. |
| `report.py` | `summary.json` and `report.md`. |
