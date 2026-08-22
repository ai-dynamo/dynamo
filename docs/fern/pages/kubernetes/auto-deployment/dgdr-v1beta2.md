---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Search Deployment Configurations with DGDR v1beta2
subtitle: Run a replay-backed deployment search, inspect its candidates, and promote a candidate to a DynamoGraphDeployment
---

`DynamoGraphDeploymentRequest` (DGDR) `v1beta2` uses [AI Simulate's
Sweeper](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/overview.md)
to evaluate deployment configurations with Replay. A request can optimize one metric or search for
a Pareto front. During the search, Dynamo publishes a bounded set of
`DynamoGraphDeploymentCandidate` (DGDC) resources. Each candidate represents one evaluated point.
Its spec contains the flat `DynamoGraphDeployment` (DGD) fields and the resolved non-deployment
parameters that distinguish the point.

Each accepted change to the DGDR spec creates an immutable `DynamoGraphDeploymentRun`. Its spec is
a complete copy of the request that started it. The run owns the Sweeper Job, its DGDCs, progress,
and report references. This lets one DGDR retain a history of independent searches without keeping
completed Jobs alive.

> [!WARNING]
> **Experimental.** The `v1beta2` API and its unstructured search parameters can change in a future
> release.

Unlike the `v1beta1` profiler, Sweeper searches commonly run for hours and can return multiple
useful configurations. DGDR reports progress after each search round and refreshes its visible
candidates as better configurations are found.

This guide describes the Kubernetes workflow and the stable DGDR fields. Use the Sweeper
documentation as the reference for traffic, optimization goals, and the unstructured search
parameters supported by the installed Sweeper version.

## Create a Search

The following request searches for a vLLM deployment for MiniMax-M2.5 on up to 32 B200 GPUs. It
replays a Mooncake trace and optimizes goodput per GPU while enforcing Time To First Token (TTFT)
and Inter-Token Latency (ITL) limits.

```yaml
apiVersion: nvidia.com/v1beta2
kind: DynamoGraphDeploymentRequest
metadata:
  name: minimax-planner-search
  namespace: inference
spec:
  modelRef:
    name: MiniMaxAI/MiniMax-M2.5
    revision: f710177d938eff80b684d42c5aa84b382612f21f
    remoteCode: AlwaysTrust # AlwaysTrust | TrustCacheAndRevision | Never
    cache:
      pvc:
        name: model-cache
        modelPath: minimax-m2.5
        mountPath: /opt/model-cache

  backends: [vllm]
  image: nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.3.0

  hardware:
    gpu:
      sku: b200_sxm
      budget: 32

  workload:
    trace:
      format: mooncake
      source:
        pvc:
          claimName: workload-traces
          path: mooncake/trace.jsonl
      replay:
        arrivalSpeedupRatio: 1.0

  objective:
    mode: optimize
    metric: goodputPerGpu
    sla:
      ttftMs: 2000
      itlMs: 50

  search:
    budget:
      maxRounds: 80
      candidatesPerRound: 8
      parallelEvaluations: 8
      maxEvaluationDuration: 120s
    parameters: # unstructured
      search_space:
        deployment_mode: [agg]
        min_gpu_budget: 8
        parallel_configs:
          - tp: 4
            attention_dp: 1
            moe_tp: 4
            moe_ep: 1
            replicas: 2
        agg_max_num_batched_tokens: [16384]
        agg_max_num_seqs: [512]
      workload:
        shared_prefix_ratio: 0.25
        num_prefix_groups: 4
        turns_per_session: 1
        inter_turn_delay_ms: 0
      adapters:
        dynamo.router:
          search_space:
            mode: [kv_router]
            overlap_score_credit: [1.0]
            prefill_load_scale: [4.0]
            temperature: [0.0]
        dynamo.planner:
          search_space:
            scaling_policy: [load_180_5]
            load_sensitivity: [default]
            fpm_sampling: [default]

  recommendation:
    maxCandidates: 5

  overrides:
    profilingJob:
      backoffLimit: 0
    dgd:
      apiVersion: nvidia.com/v1beta1
      kind: DynamoGraphDeployment
```

Apply the request:

```bash
kubectl apply -f minimax-planner-search.yaml
kubectl get dgdr minimax-planner-search -n inference -w
```

`modelRef.revision` pins the model contents. For models that execute repository-provided Python
code, set `remoteCode` explicitly:

| Value                   | Behavior                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `Never`                 | Reject models that require remote code                                                     |
| `TrustCacheAndRevision` | Allow remote code only when both a pinned revision and the configured model cache are used |
| `AlwaysTrust`           | Allow the backend to execute remote model code                                             |

Use `AlwaysTrust` only for a repository and revision that you trust.

## Choose a Workload

Set exactly one of `workload.static` or `workload.trace`.

Use a static workload when a fixed traffic shape represents the expected load:

```yaml
spec:
  workload:
    static:
      isl: 1024
      osl: 1024
      concurrency: 32
      # requestRate: 10 # alternative to concurrency
```

Use a trace workload to preserve request timestamps and sequence lengths from recorded traffic:

```yaml
spec:
  workload:
    trace:
      format: mooncake
      source:
        pvc:
          claimName: workload-traces
          path: mooncake/trace.jsonl
      replay:
        arrivalSpeedupRatio: 1.0
```

`arrivalSpeedupRatio` maps directly to the Replay traffic-rate control. A value of `1.0` preserves
the recorded arrival rate. [Sweeper Traffic](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/traffic.md)
defines the supported workload shapes, trace fields, and open-loop versus closed-loop behavior.

To create a Mooncake-format trace for testing, follow [Generate a Synthetic Trace in the Sweeper
examples](https://github.com/ai-dynamo/dynamo/blob/main/aisimulate/examples/sweeper/README.md#generate-a-synthetic-trace),
then copy the generated JSONL file to the PVC referenced by `workload.trace.source`.

## Choose an Objective

For a scalar search, set `mode: optimize` and one metric:

```yaml
spec:
  objective:
    mode: optimize
    metric: goodputPerGpu
    sla:
      ttftMs: 2000
      itlMs: 50
```

Common Sweeper metrics include throughput, end-to-end latency, goodput, and their per-GPU variants.
See [Sweeper Optimization Goals](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/optimization-goals.md)
for the supported targets, directions, and SLA semantics.

For a multi-objective search, set `mode: pareto` and list the metrics:

```yaml
spec:
  objective:
    mode: pareto
    metrics:
      - goodput
      - goodputPerGpu
      - e2eLatency
    sla:
      e2eMs: 2000
```

Set either `sla.ttftMs` with `sla.itlMs`, or `sla.e2eMs`. Do not combine both latency
forms.

Sweeper computes the complete non-dominated front. DGDR publishes no more than
`recommendation.maxCandidates` candidates from that front as Kubernetes resources.

## Configure the Search

`search.budget` controls how long and how broadly Sweeper evaluates configurations:

| Field                   | Meaning                                                   |
| ----------------------- | --------------------------------------------------------- |
| `maxRounds`             | Maximum optimizer rounds per resolved deployment branch   |
| `candidatesPerRound`    | Target successful unique candidates in one branch round   |
| `parallelEvaluations`   | Replay worker-process fan-out                             |
| `maxEvaluationDuration` | Timeout for one candidate evaluation                      |

`search.parameters` is an unstructured object passed to the search implementation. The Kubernetes
API preserves these values but does not validate their nested schema. Use [Sweeper
Configuration](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/configuration.md)
as the parameter reference for the installed Sweeper version. For Dynamo-specific Planner and
Router parameters, see [Dynamo Sweeper Integration](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md).

The unstructured boundary is intentional. Search adapters and experimental knobs evolve faster than
the Kubernetes API. Stable concepts such as the workload, objective, hardware budget, and search
budget use typed fields; implementation-specific search spaces remain under `parameters`.

## Monitor Progress

Large searches can take several hours. Inspect `status.progress` instead of treating the profiling
Job as a short, opaque operation:

```bash
RUN=$(kubectl get dgdr minimax-planner-search -n inference \
  -o jsonpath='{.status.activeRunRef.name}')

kubectl get dynamographdeploymentrun "$RUN" -n inference \
  -o jsonpath='{.status.progress}'
```

The DGDR points to the active run:

```yaml
status:
  activeRunRef:
    name: minimax-planner-search-run-4
```

While the search is active, the run status resembles:

```yaml
status:
  phase: Running
  jobRef:
    name: minimax-planner-search-run-4
  conditions:
    - type: Completed
      status: "False"
      reason: EvaluatingCandidates
  progress:
    branches:
      - name: agg
        rounds:
          completed: 23
          total: 80
    evaluations:
      scheduled: 52
      running: 8
      feasible: 41
      infeasible: 1
      failed: 2
      unsupported: 0
      cacheHits: 3
    candidates:
      paretoFront: 0
      published: 2
  candidateRefs:
    - name: minimax-planner-search-g4-4b5c3f31
    - name: minimax-planner-search-g4-9a7d681c
  provenance:
    sweeperVersion: 0.2.0
    replayVersion: 1.3.0
  startTime: "2026-07-16T13:20:00Z"
  lastProgressTime: "2026-07-16T13:28:42Z"
```

`Completed=True` marks a terminal run. Inspect its `reason` to distinguish success, failure, and
cancellation.

DGDR refreshes the visible candidate set after completed search rounds. While `Completed=False`, a
candidate is provisional and can be replaced when Sweeper finds a better result. DGDR publishes at
most `recommendation.maxCandidates` resources even when Sweeper evaluates hundreds of internal
configurations.

## Open the Search UI

The Dynamo Search UI runs as a cluster service, independently of individual Sweeper Jobs. Port
forward the service to inspect active and completed runs in a local browser:

```bash
kubectl port-forward service/dynamo-search-ui \
  -n dynamo-system 8080:8080
```

Open the local address:

```text
http://localhost:8080
```

Select the `inference` namespace, `minimax-planner-search`, and its active run. The first UI version
shows:

- completed rounds and evaluations;
- feasible, infeasible, failed, and cached evaluations;
- evaluated configurations across the resolved search-space dimensions;
- the current scalar leaders or Pareto front;
- published DGDCs and their ranks; and
- an optional Sweeper log panel.

![Mock Dynamo Search UI showing run progress, a parallel-coordinates search-space plot, the Pareto front, ranked candidates, and the Sweeper log](../../../assets/img/dgdr-search-ui-mock.svg)

The UI does not run in the Sweeper Job. A Job-local server would stop when the search process exits,
and a long-running sidecar would prevent the Job from completing. The cluster UI reads run status
and DGDCs from Kubernetes and loads detailed evaluation points from the run's report artifacts.

For an active run, the optional log panel streams the Sweeper Pod log. After the Pod is deleted, the
log remains available only when log retention is enabled for the run. Progress, candidates, and the
search report remain available independently of the Job.

## Inspect Candidates

DGDC resources carry the owning run UID, DGDR generation, and input hash as labels. List the
candidates for the active run:

```bash
RUN=$(kubectl get dgdr minimax-planner-search -n inference \
  -o jsonpath='{.status.activeRunRef.name}')
RUN_UID=$(kubectl get dynamographdeploymentrun "$RUN" -n inference \
  -o jsonpath='{.metadata.uid}')

kubectl get dgdc -n inference -l "nvidia.com/dgdr-run-uid=${RUN_UID}"
```

Inspect the selected candidate:

```bash
CANDIDATE=$(kubectl get dynamographdeploymentrun "$RUN" -n inference \
  -o jsonpath='{.status.candidateRefs[0].name}')

kubectl get dgdc "$CANDIDATE" -n inference -o yaml
```

A candidate spec contains the DGD fields flatly plus immutable, unstructured `parameters`. The
parameters identify resolved evaluation inputs that are not part of the deployable DGD. The status
reports simulation results rather than deployment health:

```yaml
apiVersion: nvidia.com/v1beta2
kind: DynamoGraphDeploymentCandidate
metadata:
  name: minimax-planner-search-g4-4b5c3f31
  ownerReferences:
    - apiVersion: nvidia.com/v1beta2
      kind: DynamoGraphDeploymentRun
      name: minimax-planner-search-run-4
  labels:
    nvidia.com/dgdr-run-uid: ddc22b7a-6557-4a3e-a1d7-a32da8849694
    nvidia.com/dgdr-generation: "4"
    nvidia.com/dgdr-input-hash: 7d3a9c18e24f9468b307c21f03c4a662
    nvidia.com/dgdc-point-hash: 4b5c3f31ef237481
    nvidia.com/dgd-spec-hash: 8f2a0ab82d09d50d
spec: # flat DGD fields plus DGDC-only parameters
  backendFramework: vllm
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.3.0
              env:
                - name: DYN_ROUTER_MODE
                  value: kv
                - name: DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT
                  value: "1.0"
                - name: DYN_ROUTER_PREFILL_LOAD_SCALE
                  value: "4.0"
                - name: DYN_ROUTER_TEMPERATURE
                  value: "0.0"
    - name: Planner
      type: planner
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.3.0
              command: [python3, -m, dynamo.planner]
              args:
                - --config
                - '{"environment":"kubernetes","backend":"vllm","optimization_target":"sla","enable_throughput_scaling":false,"enable_load_scaling":true,"load_adjustment_interval_seconds":180,"load_scaling_down_sensitivity":80,"load_min_observations":5}'
    - name: VllmWorker
      type: worker
      replicas: 2
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0
              command: [python3, -m, dynamo.vllm]
              args:
                - --model
                - MiniMaxAI/MiniMax-M2.5
                - --tensor-parallel-size
                - "4"
                - --trust-remote-code
              resources:
                limits:
                  nvidia.com/gpu: "4"
  parameters: # unstructured, resolved values rather than search ranges
    workload:
      kv_load_ratio: 0.5
status: # candidate-specific, not DGD deployment status
  rank: 1
  conditions:
    - type: Evaluated
      status: "True"
      reason: CandidateMaterialized
  experimental: # unstructured
    score: 87.7
    used_gpus: 8
    backend_version: "0.11.0"
    metrics:
      output_throughput_tok_s: 1200.0
      mean_ttft_ms: 1840.0
      mean_tpot_ms: 42.0
      goodput_output_throughput_tok_s: 727.9
      gpu_hours: 0.83
      duration_ms: 360000.0
```

The `experimental` status object is unstructured. Treat its metrics and diagnostics as specific to
the Sweeper version that produced the candidate.

The `nvidia.com/dgdc-point-hash` label covers the materialized DGD fields and `parameters`. The
`nvidia.com/dgd-spec-hash` label covers only the deployable DGD fields. Multiple evaluated points
can therefore share a DGD-spec hash when Sweeper evaluates the same deployment under different
resolved workload parameters.

## Create a DGD from a Candidate

Create a DGD after the owning DGDRRun reports `Completed=True`.

The Search UI shows the selected DGDC as syntax-highlighted YAML. Enter a name, review the
materialized spec, and select **Create DGD**. The UI copies the flat DGD fields, omits the
DGDC-only `parameters`, and creates an independent DGD. It does not modify or delete an existing
deployment.

The deployment action does not modify an Ingress, LoadBalancer, or HTTPRoute and does not shift
traffic between two DGDs. After creating a new DGD, wait until it is ready and update external
routing separately.

![Mock dialog showing syntax-highlighted DGDC YAML, a new DGD name and namespace, unchanged traffic routing, and a Create DGD button](../../../assets/img/dgdr-candidate-create-dgd-ui-mock.svg)

```bash
kubectl get dgdc minimax-planner-search-g4-4b5c3f31 -n inference -o json \
  | jq '{
      apiVersion: "nvidia.com/v1beta1",
      kind: "DynamoGraphDeployment",
      metadata: {name: "minimax-production-g4", namespace: "inference"},
      spec: (.spec | del(.parameters))
    }' \
  | kubectl create -f -
```

The new DGD starts its own deployment lifecycle. DGDC `status.conditions` only confirms that Replay
evaluated the configuration; it does not indicate that the configuration has been deployed.

## Start a New Search

Changing a search input creates a new immutable run for the new DGDR generation. The run contains a
complete copy of the accepted DGDR spec, so later edits to the request do not change an active or
completed run. Use `status.activeRunRef` to find the run currently being evaluated. Run and
candidate labels identify the generation and input hash that produced each result.

Changes that affect the model, hardware, workload, objective, search budget, or unstructured
parameters require new evaluations. Metadata-only changes do not change the search input.

For recurring trace-based searches, write every completed trace to a new location and update the
trace source. An object-storage URI identifies a trace directly:

```yaml
spec:
  workload:
    trace:
      source:
        uri: s3://dynamo-traces/qwen/2026-08-12.jsonl
```

For a trace stored on a PersistentVolumeClaim (PVC), update the path within the claim:

```yaml
spec:
  workload:
    trace:
      source:
        pvc:
          claimName: workload-traces
          path: qwen/2026-08-12.jsonl
```

DGDR treats the contents at a trace location as immutable. Updating the URI or PVC path starts the
new run; it does not require a separate trigger or content digest.

To repeat a search without changing its inputs, change `spec.rerun.reason`:

```yaml
spec:
  rerun:
    reason: "Repeat after Sweeper upgrade on 2026-08-12"
```

Every new value requests one run and is copied into that run's DGDR snapshot. Reapplying the same
value does not request another run. If a trace is replaced at the same location, change
`spec.rerun.reason` because Kubernetes cannot observe the content change.

To capture traffic from an active DGD and repeat this workflow on a schedule, see [Continuous
Profiling](continuous-profiling.md). That higher-level workflow treats the active DGD as the
baseline and creates independent runs from successive trace windows.

## Customize Generated DGD Specs

Use `spec.overrides.dgd` to merge common settings into every generated candidate. The override uses
the `nvidia.com/v1beta1` DGD schema:

```yaml
spec:
  overrides:
    dgd:
      apiVersion: nvidia.com/v1beta1
      kind: DynamoGraphDeployment
      spec:
        components:
          - name: Frontend
            podTemplate:
              spec:
                nodeSelector:
                  workload.nvidia.com/class: inference
```

Use `spec.overrides.profilingJob` for Kubernetes Job settings such as tolerations, node selectors,
and garbage-collection TTL. The MVP requires `backoffLimit: 0` because it does not resume a failed
Sweeper process.

## Work with v1beta1 Clients

`v1beta1` and `v1beta2` represent different workflows. Conversion preserves the complete source
payload instead of selecting one candidate or discarding search state:

- A native `v1beta2` request stores its fields directly under `spec`. A `v1beta1` client sees that
  payload under `spec.v1beta2`.
- A native `v1beta1` request keeps its profiler fields directly under `spec`. A `v1beta2` client sees
  that payload under `spec.v1beta1`.
- The native schema and the versioned compatibility payload are mutually exclusive.

Use the same API version for reads and writes when updating a request. The versioned compatibility
payload is for lossless round trips; it does not translate a single `v1beta1` result into a
multi-candidate `v1beta2` search.

## Clean Up

Delete the request and its owned runs and candidates:

```bash
kubectl delete dgdr minimax-planner-search -n inference
```

Promoted DGDs have an independent lifecycle and remain after the DGDR is deleted:

```bash
kubectl delete dgd minimax-production-g4 -n inference
```

## Related Documentation

- [Auto Deployment Overview](overview.mdx)
- [Auto Deploy with DGDR v1beta1](auto-deploy-with-dgdr.md)
- [Continuous Profiling](continuous-profiling.md)
- [DynamoGraphDeployment Reference](../../reference/kubernetes-api/dynamo-graph-deployment.mdx)
- [AI Simulate's Sweeper Overview](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/overview.md)
- [Sweeper Configuration](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/configuration.md)
- [Sweeper Traffic](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/traffic.md)
- [Sweeper Optimization Goals](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/optimization-goals.md)
- [Dynamo Sweeper Integration](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md)
