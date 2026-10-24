<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Gateway A/B: Frontend vs EPP on 2048 mocker workers

This benchmark compares the two ways a Kubernetes Gateway can route requests to Dynamo workers. Each
arm runs 2048 GPU-free mocker workers. It uses only tools the repository already has:

- The Dynamo operator deploys each arm from a DynamoGraphDeployment (DGD).
- The mocker (`python3 -m dynamo.mocker`) stands in for the GPU workers.
- [aiperf](https://github.com/ai-dynamo/aiperf) replays recorded coding-agent sessions through each
  arm, measures them, scrapes the components' Prometheus metrics, and plots A against B.

| Arm | Request path | Manifest |
|---|---|---|
| **A** `a-frontend` | client → Gateway → Frontend (KV router) → mocker worker | [`a-frontend.yaml`](a-frontend.yaml) |
| **B** `b-epp` | client → Gateway → EPP (KV router) → Frontend sidecar in the chosen worker's pod (direct mode) → mocker worker | [`b-epp.yaml`](b-epp.yaml) |

Both arms serve `Qwen/Qwen3-0.6B`, aggregated, with the block size of the
[Qwen recipe](../../../recipes/qwen3-0.6b/vllm/agg/gaie/deploy.yaml). Arm A is
[`agg_router.yaml`](../../../examples/backends/vllm/deploy/agg_router.yaml) and arm B is
[`gaie/agg.yaml`](../../../examples/backends/vllm/deploy/gaie/agg.yaml), with each vLLM worker
replaced by one mocker worker. Both arms sit behind the same Gateway, so the comparison measures the
routing path rather than the gateway hop.

## What a run produces

`run.sh all` writes one directory per run. For each arm it holds the DGD, the exact aiperf
commands, aiperf's artifacts, and the Prometheus metrics of every component:

```text
benchmarks/results/gateway_epp_ab/<UTC time>/
├── a-frontend/
│   ├── deploy.env                every setting and the repository commit when the arm was deployed
│   ├── bench.env                 the same when it was measured, with the aiperf image it used
│   ├── manifest.json             the DGD and HTTPRoute that run.sh applied
│   ├── deployed/                 the DGD with its status, pods, image digests, nodes, operator image
│   ├── inputs/                   the aiperf Job, its parameters, the list of metrics endpoints
│   ├── sessions512_r0/           one aiperf run: live sessions and round
│   │   ├── command.txt           the exact aiperf command
│   │   ├── console.log           aiperf's console output, with its summary table
│   │   ├── profile_export_aiperf.json, .csv     TTFT, ITL, latency and throughput statistics
│   │   ├── profile_export.jsonl                 one record per request
│   │   └── server_metrics_export.json, .csv     Prometheus metrics scraped during the run
│   ├── sessions2048_r0/ ...
│   ├── prometheus/               /metrics of every component, before the first run and after each run
│   ├── runs.tsv                  aiperf's exit status for each run
│   └── aiperf-job.log
└── b-epp/                        the same for arm B
```

## Prerequisites

1. **A Dynamo operator, 1.5.0 or later.** The native Rust EPP needs it.
2. **Gateway API, the Inference Extension CRDs, agentgateway v1.2.x, and a Gateway named
   `inference-gateway`.** The repository's installer sets them up, with agentgateway v1.2.1:

   ```bash
   NAMESPACE=<namespace> bash deploy/inference-gateway/scripts/install_gaie_crd_agentgateway.sh
   ```

3. **A ReadWriteMany PVC named `model-cache` that holds `Qwen/Qwen3-0.6B`**, in the benchmark
   namespace. Use the Qwen recipe's manifests. Set `storageClassName` in `model-cache.yaml` to a
   ReadWriteMany class first. `model-download.yaml` reads `HF_TOKEN` from a Secret named
   `hf-token-secret`.

   ```bash
   kubectl apply -n <namespace> -f recipes/qwen3-0.6b/model-cache/model-cache.yaml
   kubectl apply -n <namespace> -f recipes/qwen3-0.6b/model-cache/model-download.yaml
   kubectl wait -n <namespace> job/model-download --for=condition=Complete --timeout=30m
   ```

   Every worker and router pod reads the tokenizer and model config from this cache, with
   `HF_HUB_OFFLINE=1`. Without the cache, thousands of pods would call Hugging Face at once and hit
   its rate limits.
4. **Capacity for one arm at a time.** `run.sh all` deploys arm A, measures it and deletes it before
   it deploys arm B. See [Running 2048 workers](#running-2048-workers) for the details.

   | | Arm A | Arm B |
   |---|---|---|
   | Worker pods | 2048 | 2048 |
   | Containers per worker pod | mocker | mocker and Frontend sidecar |
   | Requests of all worker pods | 205 CPU, 1 TiB | 410 CPU, 2 TiB |
   | Router pod | Frontend: 8 CPU, 16 GiB | EPP: 8 CPU, 16 GiB |
   | aiperf pod | 16 CPU, 16 GiB | 16 CPU, 16 GiB |
   | Nodes, at the default 110 pods per node | at least 19 for the workers | at least 19 for the workers |

5. **Hugging Face access for the aiperf pod.** It downloads the session corpus, 1.85 GB, and the
   tokenizer into a scratch volume, so it must reach huggingface.co, and its node needs about 4 GB
   of free disk.
6. **On your workstation:** `kubectl` and `python3`. `run.sh plot` also needs aiperf.

## Run it

Run these from the repository root:

```bash
export KUBE_CONTEXT=<context> NAMESPACE=<namespace>
B=benchmarks/incluster/gateway_epp_ab/run.sh

# 1. Check the wiring: 2 workers per arm, 2 sessions replayed for 60 seconds. This takes about
#    10 minutes.
WORKERS=2 SESSIONS=2 DURATION_S=60 TRACES=2 ROUNDS=1 $B all

# 2. The benchmark: 2048 workers per arm, one arm at a time, on images with the latency
#    fixes that the 1.5.1 defaults lack (see "Comparing the arms").
ROUNDS=1 FRONTEND_IMAGE=<frontend image> WORKER_IMAGE=<planner image> $B all

# 3. Plot arm B against arm A with aiperf.
$B plot benchmarks/results/gateway_epp_ab/<UTC time>
```

`all` deploys, measures and tears down arm A, then arm B. It keeps one arm deployed at a time: it
tears down the other arm first if an earlier run left it up. `KEEP=1` leaves the last arm deployed.
To keep an arm deployed between iterations, run the steps one at a time with the same `RESULTS`
directory:

```bash
export RESULTS=benchmarks/results/gateway_epp_ab/my-run
$B deploy a    # apply the DGD and the HTTPRoute, wait until every worker is ready
$B bench a     # run the aiperf Job, then copy its results to $RESULTS/a-frontend
$B teardown a  # delete the DGD, the HTTPRoute and the Job
```

`deploy` scales the workers up `RAMP_STEP` at a time and waits for each step to be ready. An arm
that is already deployed scales from its current size. `deploy` warns when the other arm is also
deployed. `bench` starts the aiperf Job, waits for it, copies its results out of the pod, and
records what was deployed. The pod keeps the results until a copy succeeds, so if `bench` stops
before that, `$B collect a` copies them.

## The workload and the aiperf command

aiperf's
[InferenceX AgentX MVP scenario](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark)
replays recorded Claude Code sessions: many turns, subagents, think time between turns, and
contexts of tens to hundreds of thousands of tokens. Each turn reuses its session's earlier
context, so a turn that the router sends away from the worker holding that context loses its cache.
The benchmark therefore tests how well each arm keeps a session on its worker, and its long prompts
load the tokenizer of both routers.

Each run is one aiperf command. Arms A and B run the same commands. Only the value of the
`X-Dynamo-AB-Arm` header differs, which selects the arm's HTTPRoute on the shared Gateway. This is
the first run of arm A with the default settings:

```bash
env AIPERF_HTTP_CONNECTION_LIMIT=8192 \
  AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 \
  aiperf profile --model Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B \
  --url http://inference-gateway.agentgateway-system.svc.cluster.local:80 \
  --endpoint-type chat --streaming \
  --header X-Gateway-Model-Name:Qwen/Qwen3-0.6B --header X-Dynamo-AB-Arm:<namespace>/qwen-a \
  --scenario inferencex-agentx-mvp --burst-phase-starts \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --max-context-length 128000 --concurrency 512 --benchmark-duration 900 \
  --extra-inputs ignore_eos:true --use-server-token-count \
  --random-seed 42 --no-gpu-telemetry \
  --server-metrics <the /metrics URLs of the router and 4 worker pods> \
  --server-metrics-formats json csv \
  --artifact-dir /results/a-frontend/sessions512_r0 --ui none
```

Every run writes its own command to `command.txt`. The defaults work like this:

- **The scenario sets the rules.** It replays each session's recorded think time and caps idle gaps
  at 10 seconds. It tags the first turn of each session, so a replayed session never finds an
  earlier play's cache. It warms each session up by replaying one turn with its full history, and it
  rejects `--request-rate`: the load is a number of live sessions, `--concurrency`.
- **A warmup in one burst.** By default aiperf spreads the warmup over the sessions' recorded arrival
  times, which took 107 minutes at 2048 sessions. `bench.sh` passes `--burst-phase-starts`, which
  the scenario allows: every warmup request goes out at once, and so does each session's first
  measured turn. `BURST_PHASE_STARTS=0` keeps the recorded spread.
- **Load points.** At 2048 workers, the default `SESSIONS` of 512, 2048 and 4096 are about a quarter
  of a session, one, and two per worker. At 4096, two long sessions can fill a worker's KV cache, so
  the router has to trade cache hits against load. Subagents add requests beyond `--concurrency`, so
  `bench.sh` raises aiperf's connection limit to twice the largest point.
- **Closed loop.** Each session waits for its previous turn, so a slower arm sends fewer requests.
  Compare throughput along with latency.
- **Duration.** Each run lasts `DURATION_S`, 900 seconds by default, the scenario's minimum. The
  default sweep, 3 points in 3 rounds, measures for about 2.3 hours per arm, so start with
  `ROUNDS=1`. A shorter run adds `--unsafe-override`, and aiperf then marks it
  `submission_valid: false`, a field this benchmark does not use.
- **The same sessions for both arms.** A round and point use the same seed on both arms, so A and B
  replay the same sessions from the same turns.
- **Context cap.** `MAX_CONTEXT`, 128,000 tokens as in the tutorial, drops every session whose
  longest request, prompt plus output, exceeds it. That keeps 78 of the corpus's 393 sessions, the
  shortest fifth: half of them peak above 226,000 tokens. The mockers set no length limit, and each
  holds 16,384 KV blocks of 16 tokens, 262,144 tokens, so two sessions at the cap fit in one worker.
  A cap of 262,144 keeps 220 sessions, but then one session can fill a worker's KV cache, so raise
  `--num-gpu-blocks-override` in both manifests as well.
- **The corpus.** `DATASET` names a date-pinned corpus, so every run replays the same sessions. Each
  arm's aiperf pod downloads it from Hugging Face. `TRACES` keeps only the first sessions that fit
  the cap; use it for smoke runs only.
- **Routing.** Both arms route each turn by KV overlap. `bench.sh` does not send aiperf's
  `X-Dynamo-Session-ID` header, so session affinity plays no part.
- **Server token counts.** The mocker emits random token ids, which do not survive re-tokenizing,
  so aiperf counts tokens from the responses' `usage`.

The results compare the two arms with each other; they are not comparable with published AgentX
results, which run real models.

## Prometheus metrics

The bundle has two layers. Both come from the components' own `/metrics` endpoints:

| Component | Endpoint | Arm |
|---|---|---|
| Gateway proxy (agentgateway) | port 15020, or the pod's `prometheus.io/port` annotation | A and B |
| Frontend | container port `http` (8000) | A |
| EPP | container port `metrics` (9090) | B |
| Mocker worker | container port `system` (9090) | A and B |
| Frontend sidecar | container port `http` (8000) | B |

1. **During each run**, aiperf's `--server-metrics` scrapes the router (A's Frontend or B's EPP) and
   the first `SAMPLE_WORKERS` worker pods (4 by default; on arm B, both the mocker and the sidecar).
   aiperf writes per-endpoint statistics over the measured window to `server_metrics_export.json`
   and `.csv` in the run's directory. Add `jsonl` to `SERVER_METRICS_FORMATS` to keep the raw time
   series as well.
2. **Before the first run and after each run**, `bench.sh` fetches `/metrics` from every endpoint in
   the table: every worker, every sidecar, the router and the Gateway. Each snapshot is one archive,
   `prometheus/<NNN>_<run>.tar.gz`, with one file per endpoint under its role (`decode/`,
   `frontend/`, `epp/`, `gateway/`). The difference between two consecutive snapshots is what one
   run did. For example, the counters show how evenly the router spread the requests over 2048
   workers.

aiperf does not scrape every worker during a run. Each endpoint costs the client CPU at every
scrape interval, so scraping 2048 or 4096 endpoints would load the client in proportion to the
worker count and change what it measures.

The Gateway appears only in the snapshots. aiperf 0.12 and 0.13 reject agentgateway's whole
endpoint, because it exposes one metric of the OpenMetrics `info` type.

The EPP's own `/metrics` exposes only `dynamo_epp_cached_tokens`. On arm B, the request path is
visible mostly in the sidecars, the workers and the Gateway, while arm A's Frontend exposes the
router's metrics as well.

If the cluster runs Prometheus with the platform chart's PodMonitors, Prometheus scrapes the same
components. The timestamps in `aiperf-job.log` give the window of each run.

## Comparing the arms

- `$B plot <results-dir>` runs `aiperf plot` over both arms, with arm A as the baseline and arm B as
  the treatment. It needs the aiperf version in `<arm>/aiperf-version.txt` on your workstation.
- Each run's `profile_export_aiperf.json` holds aiperf's statistics. For example, TTFT by run:

  ```bash
  for f in benchmarks/results/gateway_epp_ab/<UTC time>/*/sessions*/profile_export_aiperf.json; do
    echo "$f $(jq -c '.time_to_first_token | {avg, p50, p90, p99}' "$f")"
  done
  ```

- `usage_prompt_cache_read_tokens` in the same file counts the prompt tokens served from a worker's
  prefix cache. Against `input_sequence_length`, it shows how well each arm keeps sessions on their
  workers.

Two things keep the comparison fair, and the tests in `tests/` check both of them:

- **The same workers.** The worker components of the two DGDs are identical, apart from arm B's
  sidecar.
- **The same router code and settings.** A's Frontend and B's EPP run the same image, with the same
  resources and environment. The only difference is `DYN_ROUTER_MODE=kv`, which the EPP does not
  need. Both read the KV block size from the workers' model cards. The sidecar runs the same image
  as A's Frontend. If you tune the router, set the same variables on both.

The client is the same too: one Job manifest, the same aiperf flags and seeds, and the same Gateway
URL.

Keep these limits in mind when reading the results:

- **The arms run one after the other.** Drift on the cluster between the two arms is not cancelled.
  Run `ORDER=b,a $B all` as well, to see whether the order matters.
- **Arm B uses more CPU in total.** It runs a Frontend sidecar in every worker pod. That cost is
  part of the architecture, so the benchmark includes it.
- **Mocker timings are synthetic.** Compare A with B, never a mocker number with a GPU number.
- **The 1.5.1 default images lack two latency fixes.** In 1.5, the Frontend's HTTP server and the
  EPP's TLS ext_proc listener leave Nagle's algorithm on
  ([#16859](https://github.com/ai-dynamo/dynamo/issues/16859)). Behind agentgateway, a reply can
  then wait about 40 ms for the gateway's delayed ACK. Arm A's requests cross one such hop and arm
  B's cross two, so these images skew the difference between the arms as well as the absolute TTFT.
  [#16728](https://github.com/ai-dynamo/dynamo/pull/16728) fixes the Frontend and
  [#16862](https://github.com/ai-dynamo/dynamo/pull/16862) the EPP. Releases up to 1.6.0 lack the
  EPP fix. Until a release has both, use NGC's nightly images from `20261022-5df849a` on, which are
  public and built for amd64 and arm64: `nvcr.io/nvidia/ai-dynamo/dynamo-frontend-nightly:<tag>`
  for `FRONTEND_IMAGE` and `nvcr.io/nvidia/ai-dynamo/dynamo-planner-nightly:<tag>` for
  `WORKER_IMAGE`, with the same tag for both. Their tags carry no version, so also set
  `RUNTIME_VERSION`: 1.6.0 for `20261022-5df849a`. `run.sh` warns when `FRONTEND_IMAGE` is a 1.5
  image.
- **Arm B needs agentgateway v1.2.x.** Before v1.2.0, agentgateway opens a new TLS connection to
  the EPP for every request and keeps it idle for 90 to 180 seconds
  ([agentgateway#1528](https://github.com/agentgateway/agentgateway/pull/1528)). The EPP accepts at
  most 1024 connections, so from about 10 requests per second arm B's requests time out at the
  Gateway and fail with HTTP 500. From v1.3.0, agentgateway sends an ext_proc protocol configuration
  that the EPP rejects, so every arm B request fails with HTTP 500. Arm A is not affected. The
  installer installs v1.2.1, and `run.sh` warns when the Gateway runs a release before v1.2.0.

## Running 2048 workers

- **One mocker per pod.** The EPP maps the worker it picks to a pod through the hash of the pod's
  name. Several mockers in one pod (`--num-workers`) would share that identity. Arm A uses the same
  layout, so the workers of the two arms match.
- **Pods per node.** Kubernetes allows 110 pods per node by default, so 2048 worker pods need at
  least 19 nodes. Give the router pod and the aiperf pod room on nodes of their own if you can. The
  aiperf Job prefers a node without the router.
- **Discovery traffic.** Every Dynamo process (each mocker, each sidecar, the Frontend or the EPP)
  watches every worker's discovery record in the namespace. The traffic at startup therefore grows
  with the square of the number of pods. `deploy` adds `RAMP_STEP` workers (512) at a time. Lower it
  if the API server throttles.
- **Changing a deployed arm.** `deploy` ramps only the worker count. A change to a deployed arm's
  images, resources or mocker settings makes the operator replace every worker at once, without the
  ramp. Tear the arm down before you deploy it with a changed spec.
- **Slow registrations.** A worker writes its discovery record through the API server and holds its
  registration until the write returns. With Dynamo 1.5.1 on minikube, one of 32 workers started
  together waited about 290 seconds for one write, then registered and became ready. `deploy` names
  the workers that are still not ready when no worker becomes ready for 5 minutes.
- **Images.** The frontend image is about 8 GB and the planner image about 2 GB. Each node pulls
  them once, so pre-pull them or use a nearby registry.
- **The client.** At thousands of sessions, check that the aiperf pod is not short of CPU or memory.
  A starved client reports its own delays as server latency. Replaying 2 sessions on minikube, it
  used up to 6.1 GB of memory and 3.5 GB of scratch space.
- **The Gateway.** One agentgateway replica serves both arms. If it saturates, scale it for both.
- **Resource requests.** The mocker and the sidecar each request 100m CPU, with no CPU limit, and
  512 MiB of memory. With 64 workers on minikube, a mocker used 200–230 MiB (peak 280 MiB), a
  sidecar 245–300 MiB (peak 306 MiB) and the EPP 200 MiB. Each process also keeps every worker's
  discovery record, so expect more at 2048 workers. Measure your own with `kubectl top pods`, and
  edit both manifests.

## Settings

`run.sh` reads these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `KUBE_CONTEXT` | current context | kubectl context |
| `NAMESPACE` | `default` | namespace of the DGDs, the model cache and the aiperf Job |
| `WORKERS` | `2048` | mocker workers per arm |
| `RAMP_STEP` | `512` | workers added at a time; `0` adds them all at once |
| `DYNAMO_VERSION` | `1.5.1` | tag of the default images |
| `FRONTEND_IMAGE` | `nvcr.io/nvidia/ai-dynamo/dynamo-frontend:$DYNAMO_VERSION` | Frontend, EPP and sidecar; 1.5 images lack the latency fixes |
| `WORKER_IMAGE` | `nvcr.io/nvidia/ai-dynamo/dynamo-planner:$DYNAMO_VERSION` | mocker workers |
| `RUNTIME_VERSION` | none | Dynamo version of both images, for tags without one, such as nightly tags; the operator then requires it |
| `AIPERF_IMAGE` | the image of the arm's Frontend or EPP | the aiperf Job; the frontend image ships aiperf. Pin it to keep one aiperf version across the builds you compare |
| `MODEL_CACHE_PVC` | `model-cache` | PVC with the model files |
| `IMAGE_PULL_SECRET` | none | pull secret for every pod |
| `GATEWAY_NAME`, `GATEWAY_NAMESPACE` | `inference-gateway`, `agentgateway-system` | the Gateway both arms attach to |
| `ORDER` | `a,b` | the order of the arms in `all` |
| `KEEP` | `0` | `1` leaves the last arm deployed after `all` |
| `RESULTS` | `benchmarks/results/gateway_epp_ab/<UTC time>` | results directory |
| `SESSIONS` | `512,2048,4096` | live sessions, one run each |
| `DURATION_S` | `900` | seconds per run; below 900 adds `--unsafe-override` |
| `ROUNDS` | `3` | repetitions of the whole sweep |
| `DATASET` | `semianalysis_cc_traces_weka_062126` | aiperf's name for the session corpus |
| `MAX_CONTEXT` | `128000` | drops every session with a longer request |
| `TRACES` | none | smoke runs: replay only the first N sessions that fit |
| `COOLDOWN_S` | `10` | pause after each run |
| `SEED` | `42` | seed of the first run |
| `SAMPLE_WORKERS` | `4` | worker pods aiperf scrapes during each run |
| `SERVER_METRICS_FORMATS` | `json csv` | aiperf's server-metrics formats; add `jsonl` for time series |
| `BURST_PHASE_STARTS` | `1` | `1` sends each run's warmup, and each session's first measured turn, at once; `0` spreads them over the recorded arrival times |
| `READY_TIMEOUT` | `3600` | seconds to wait for each ramp step, the DGD, or the aiperf pod |
| `SETTLE_SECONDS` | `60` | wait after the last worker is ready, so the router discovers it |
| `SMOKE_TIMEOUT_S` | `600` | how long `bench.sh` retries its first request through the Gateway |
| `BENCH_TIMEOUT` | `21600` | seconds to wait for the aiperf Job |
| `AIPERF_PYTHON` | found from the `aiperf` command | `plot` only: a Python that can import aiperf |

`run.sh` checks the workload settings before it touches the cluster, and exits with status 2 when
one cannot work, such as a `SESSIONS` entry that is not a positive integer.

To change the model, the block size or the mocker settings, edit both manifests the same way. The
tests fail if their workers differ.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `deploy` waits for workers that never become ready | Usually an image pull, a full node, or a missing model cache. Check `kubectl describe pod` on a pending worker. |
| `bench` reports that the aiperf pod is not running yet | `run.sh` prints the scheduler's or the container's reason. A common one is `Insufficient cpu`: the pod requests 16 CPUs, which must fit on a node next to the workers. |
| The smoke request fails with HTTP 404 | The HTTPRoute is not attached, or another route takes the request. Check `kubectl get httproute qwen-a -o yaml` for `Accepted`. |
| The smoke request fails with HTTP 500 or 503 on arm B | The EPP has not found the workers yet. Check `kubectl logs -l nvidia.com/dynamo-component-type=epp`. |
| Arm B fails with HTTP 500 under load, and arm A does not | The Gateway runs agentgateway before v1.2.0, which uses up the EPP's connections; see [Comparing the arms](#comparing-the-arms). Upgrade it with the installer. |
| Every arm B request fails with HTTP 500 `ext_proc failed: no more response messages` | The Gateway runs agentgateway v1.3.0 or later. Its log shows `ext_proc response stream failed` with the EPP's protocol configuration error. Install v1.2.1 with the installer. |
| A run's `console.log` shows errors or timeouts | The load point is past saturation, or the client ran short of CPU. Lower `SESSIONS`, or give the aiperf pod a node of its own. |
| A run fails to load `semianalysis_cc_traces_weka_062126` | The aiperf pod cannot reach huggingface.co, or its node has no room for the 1.85 GB corpus. |
| A snapshot reports fewer endpoints than expected | Some pods restarted or were not running. The archive holds an `.err` file for each failed endpoint. |
| `run.sh` stops before collecting | The Job's pod keeps the results until a copy succeeds: `RESULTS=<dir> $B collect <arm>`. After that copy the pod exits and its results are gone. |

## Files

| File | Purpose |
|---|---|
| `a-frontend.yaml`, `b-epp.yaml` | The DGD and HTTPRoute of each arm |
| `aiperf-job.yaml` | The aiperf Job, the same for both arms |
| `bench.sh` | Runs in the aiperf Job: smoke request, metrics snapshots, one aiperf run per number of sessions and round |
| `run.sh` | Runs on your workstation: deploy, bench, collect, teardown, plot |
| `render.py` | Applies a run's settings to the manifests, and lists the metrics endpoints of a deployed arm |
| `tests/` | Checks that the arms differ only in their routing path |
