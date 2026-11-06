---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Search Deployment Configurations with DGDR v1beta2
subtitle: Run an AISimulate-backed deployment search, inspect immutable runs and candidates, and create a DynamoGraphDeployment from a result
---

`DynamoGraphDeploymentRequest` (DGDR) `v1beta2` uses AISimulate to evaluate deployment
configurations without creating one Kubernetes resource per optimizer trial. A mutable Request
creates immutable `DynamoGraphDeploymentRun` resources. Each Run publishes a bounded set of
immutable `DynamoGraphDeploymentCandidate` (DGDC) resources.

Each Candidate represents one evaluated point. Its spec contains the unpruned
`DynamoGraphDeployment` (DGD) spec. Its status contains the resolved evaluation `parameters`,
evaluation `metrics`, and a `Valid` condition. Choose a valid result and copy its spec into an
independent DGD.

> [!WARNING]
> **Experimental.** DGDR v1beta2 is not feature complete and may change incompatibly in future
> releases. This guide covers static and recorded-trace workloads, scalar optimization, DGD
> overrides, and manual Candidate-to-DGD creation.

## Prerequisites

Install the operator and matching DGDR v1beta2 CRDs as described in the
[installation guide](../installation/install-dynamo.md). The API is available without an API
feature gate. Installations with `dynamo-operator.upgradeCRD=false` require an administrator to
install the matching CRDs before upgrading the operator.

Planner live retune has its own disabled-by-default
`dynamo-operator.dgdrV1Beta2.plannerLiveRetune` Helm opt-in and requires Planner configuration.
It is not required for a manually submitted search. See [Planner Live Retune](planner-live-retune.md).

The operator selects the planner image configured by its Helm release. Requests and Runs cannot
select a different AISimulate image or version.

## Create a Search

The following Request pins the hardware topology, so admission does not need cluster discovery:

```yaml
apiVersion: nvidia.com/v1beta2
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen-search
  namespace: inference
spec:
  model:
    name: Qwen/Qwen3-32B
  hardware:
    gpu:
      sku: h200_sxm
      budget: 8
    numGpusPerNode: 8
```

Admission supplies the omitted stable defaults. The accepted Request is equivalent to this shape:

```yaml
spec:
  model:
    name: Qwen/Qwen3-32B
  backends: [vllm, sglang, trtllm]
  hardware:
    gpu:
      sku: h200_sxm
      budget: 8
    numGpusPerNode: 8
  workload:
    static:
      inputTokens: 1024
      outputTokens: 128
      load:
        concurrency: 10
  objective:
    mode: Optimize
    metric: ThroughputPerGPU
    sla:
      ttftMs: 2000
      itlMs: 30
  recommendation:
    maxCandidates: 1
```

Apply the Request and wait for admission and the Request controller to accept it:

```bash
kubectl apply -f qwen-search.yaml
kubectl get dgdr qwen-search -n inference -w
```

The Request reports the active immutable Run:

```yaml
status:
  observedGeneration: 1
  activeRunRef:
    name: qwen-search-1
  conditions:
    - type: Accepted
      status: "True"
      reason: RunCreated
```

The exact generated name can differ when the Request name needs truncation.

## Adjust the Search

Set `spec.backends` to choose from `vllm`, `sglang`, and `trtllm`; omission searches all three.
Explicit hardware values pin the SKU, GPU budget, and node width. Omit hardware to use cluster
discovery. Explicit empty hardware and static-workload objects opt out of their typed defaults;
use the [DGDR API reference](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx)
when supplying those inputs through AISimulate overrides.

For static traffic, choose exactly one of `requestsPerSecond`, `concurrency`, or
`kvCapacityFraction` under `spec.workload.static.load`. The following fragment uses an open-loop
arrival rate and keeps up to five published Candidates:

```yaml
spec:
  workload:
    static:
      inputTokens: 2048
      outputTokens: 256
      load:
        requestsPerSecond: 10
  recommendation:
    maxCandidates: 5
```

`maxCandidates` defaults to `1`, accepts 1 through 100, and bounds published Candidates rather than
internal optimizer trials. The supported objective mode is `Optimize` with one scalar `metric`;
`Pareto` and `objective.metrics` are not admitted. `Goodput` and `GoodputPerGPU` require at least one
SLA bound. See the [scalar objective
reference](../../reference/kubernetes-api/full-api-reference.mdx#objectivespec) for metrics and SLA
combinations.

## Replay a Recorded Trace

To evaluate recorded traffic instead of synthetic load, use a completed Dynamo request trace on a
PersistentVolumeClaim (PVC) in the Request namespace:

```yaml
apiVersion: nvidia.com/v1beta2
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen-trace-search
  namespace: inference
spec:
  model:
    name: Qwen/Qwen3-32B
  hardware:
    gpu:
      sku: h200_sxm
      budget: 8
    numGpusPerNode: 8
  workload:
    trace:
      format: Dynamo
      source:
        persistentVolumeClaim:
          claimName: workload-traces
          subPath: qwen/window.jsonl.gz
      replay:
        arrivalSpeedupRatio: 1.0
```

`Dynamo` is the only typed trace format, and a PVC is the only typed trace source. The file must
be complete, readable by the Run Job, and immutable before the Request is created. JSON Lines and
gzip-compressed JSON Lines are supported. The operator mounts only that file, read-only, into the
AISimulate container. `1.0` preserves recorded arrival times; a larger speedup replays arrivals faster.

`workload.static` and `workload.trace` are mutually exclusive. When changing an existing Request
from static traffic to a trace, set `workload.static: null` in the same update. No static traffic
default is added to a trace workload. See the [trace source and claim lifecycle
reference](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx#replay-a-recorded-trace-v1beta2)
for file permissions, PVC access modes, deletion, and input-override behavior.

## Override Native AISimulate Input

Use `spec.aisim.inputOverrides` for supported AISimulate input that is not part of the stable
typed API:

```yaml
spec:
  aisim:
    inputOverrides:
      traffic:
        stop:
          requests: 1000
      optimizer:
        algorithm: random
        max_trials: 50
        parallelism: 4
        candidate_timeout_seconds: 120
```

The object uses the upstream AISimulate field names, including `snake_case`. The operator applies
it as an RFC 7386 JSON Merge Patch over the typed lowering. Override values win; JSON `null` deletes
a lowered field.

The effective input is assembled in this order:

```text
operator underlay
  + typed Request lowering
  + spec.aisim.inputOverrides
  + protected operator-owned dgdr_run values
```

Admission rejects overrides of `dgdr_run.name`, `max_candidates`, `snapshot_dir`, `runtime_image`,
`runtime_version_override`, `derive_runtime_images`, `num_gpus_per_node`, and `namespace`. These
values define the Run identity, trust boundary, and transport.

Request and Run creation validate the merged input against the bundled AISimulate input schema.
Request spec changes are validated again; unchanged specs remain metadata-updatable after a bundled
schema change. Adapter-owned configuration and Python-only constraints are checked by AISimulate
at execution time. The operator validates DGD overrides independently of AISimulate input. Use the
[AISimulate Sweeper configuration](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/configuration.md)
that matches the operator release.

## Customize the Candidate DGD

Add one partial `nvidia.com/v1beta1` DGD under `spec.overrides.dgd` to customize every selected
Candidate independently. For example, add a scheduling constraint to the generated Frontend:

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

The publisher applies the override atomically using DGD Server-Side Apply merge rules. Components
must already exist; named containers can be added. Atomic lists, including container `command` and
`args`, replace the generated list. A failed merge preserves the original generated spec. A
successful merge is checked with strict server-side dry-run; its defaulted response is discarded.
The Candidate's `Valid` condition records the result.

Overrides can change deployment configuration, but AISimulate does not reevaluate it. Metrics and
parameters continue to describe the original evaluation. For supported merge directives, metadata
restrictions, and invalid-result diagnostics, see [v1beta2 DGD
Overrides](../../reference/kubernetes-api/dynamo-graph-deployment-request.mdx#v1beta2-dgd-overrides).

## Monitor the Run

Resolve the active Run from the Request:

```bash
RUN=$(kubectl get dgdr qwen-search -n inference \
  -o jsonpath='{.status.activeRunRef.name}')

kubectl get dgdrrun "$RUN" -n inference -o yaml
```

The Run spec is an immutable, fully defaulted copy of the accepted Request spec. The Run controller
lowers that snapshot exactly once, writes the effective input to an immutable Run-owned ConfigMap,
and reports the same public input at `status.aisim.input`.

While execution is active, an abridged status resembles:

```yaml
status:
  phase: Running
  jobRef:
    name: qwen-search-1
  aisim:
    input:
      engine:
        model: Qwen/Qwen3-32B
        backend:
          choices: [vllm, sglang, trtllm]
      optimizer:
        max_trials: 320
  progress:
    candidates:
      published: 1
  candidateRefs:
    - name: qwen-search-1-1176365e-evaluated-point-0123456789ab
  conditions:
    - type: Completed
      status: "False"
      reason: JobRunning
```

The publisher updates ordered `candidateRefs`, `progress.candidates.published`, the publication
condition, and coarse progress time. Per-branch rounds and detailed evaluation counters are reserved
in the API but are not currently populated from the snapshot; their absence is not a failed search.

Treat `status.aisim.input` as the authoritative record of what the Job received. For a native
static workload, lowering also sets `traffic.stop.requests` to 1000. It independently defaults
`optimizer.max_trials` to 320. An input override can replace or delete either value.

`Completed=True` marks a terminal Run, including failures. `phase: Succeeded` with reason
`SearchSucceeded` requires a successful search and at least one `Valid=True` Candidate. Inspect
`phase`, conditions, and diagnostic messages before deploying a result.

The publisher owns the Run's `Published` condition. `Published=True` means the current projection
is published, including invalid Candidates; it does not mean that every Candidate is valid.
Retryable preparation errors, including an HTTP 403 from DGD dry-run admission, report
`Published=False` and keep retrying after permissions are repaired. The Run controller owns
`Completed` and the terminal outcome.

The Job has two containers. The planner image runs `aisimulate recommend` and writes atomic
`dgdr_run` snapshots to a shared `emptyDir`. The publisher in the operator image reads those
snapshots, updates Run progress, and reconciles Candidates. Only the publisher receives projected
Kubernetes credentials.

## Inspect Candidates

Use the Run's ordered `candidateRefs`; list order is the scalar ranking, best first:

```bash
CANDIDATE=$(kubectl get dgdrrun "$RUN" -n inference \
  -o jsonpath='{.status.candidateRefs[0].name}')

kubectl get dgdc "$CANDIDATE" -n inference -o yaml
```

An abridged Candidate resembles:

```yaml
apiVersion: nvidia.com/v1beta2
kind: DynamoGraphDeploymentCandidate
metadata:
  name: qwen-search-1-1176365e-evaluated-point-0123456789ab
  labels:
    nvidia.com/dgdr-run-uid: 59aeff3d-5d41-4981-85a2-822edeb7976f
    nvidia.com/dgdr-candidate-id: evaluated-point-0123456789ab
spec:
  backendFramework: vllm
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/dynamo-runtime:latest
    - name: VllmWorker
      type: worker
      replicas: 1
      podTemplate:
        spec:
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest
              resources:
                limits:
                  nvidia.com/gpu: "8"
status:
  parameters:
    backend: vllm
  metrics:
    score: 87.7
    usedGpus: 8
  conditions:
    - type: Valid
      status: "True"
      reason: ValidationSucceeded
      lastTransitionTime: "2026-11-06T10:00:00Z"
      message: Strict DGD dry-run succeeded
```

The Candidate spec is immutable and preserves the DGD before dry-run defaulting or webhook
mutation. Evaluation `parameters` distinguish points with the same deployable DGD. Parameters and
metrics describe the original evaluation even when DGD overrides change the stored spec.

Check for `Valid=True`; a missing condition is not success. `Valid` reports strict API admission
validity, not image availability, GPU scheduling, deployment readiness, or the performance of a
modified deployment. Invalid Candidates remain visible and count toward `maxCandidates`.

## Create a DGD from a Candidate

Wait for the owning Run to become terminal, then inspect its outcome and select a Candidate with
`Valid=True`:

```bash
kubectl wait --for=condition=Completed "dgdrrun/$RUN" -n inference --timeout=30m
kubectl get dgdrrun "$RUN" -n inference
CANDIDATE=$(kubectl get dgdrrun "$RUN" -n inference \
  -o jsonpath='{.status.candidateRefs[0].name}')
kubectl get dgdc "$CANDIDATE" -n inference -o yaml
```

Export the Candidate spec into an independent DGD. The guard refuses an invalid or unchecked
Candidate; evaluation parameters are already outside the spec:

```bash
kubectl get dgdc "$CANDIDATE" -n inference -o json \
  | jq -e '
      if any(.status.conditions[]?; .type == "Valid" and .status == "True") then
        {
          apiVersion: "nvidia.com/v1beta1",
          kind: "DynamoGraphDeployment",
          metadata: {name: "qwen-production", namespace: "inference"},
          spec: .spec
        }
      else error("Candidate must have Valid=True") end
    ' > qwen-production.yaml
```

Review the generated manifest, select any labels and annotations intended for the new DGD, and
create it:

```bash
kubectl create -f qwen-production.yaml
```

The DGD has no owner reference to the Request or Run. Deleting the search history does not delete a
DGD created from a Candidate.

## Start a New Search

Change any native Request spec field to create a new generation. If the previous Run is active, the
Request controller marks it for supersession. The Run controller deletes only the execution Job,
waits for it to disappear, marks the immutable Run terminal with reason `Superseded`, and lets the
newest Request generation start. Rapid successive edits can coalesce intermediate generations.

The Request retains the five newest terminal Runs in `status.recentRunRefs`; older owned Runs and
their Candidates are garbage-collected. Metadata-only Request changes do not create a Run.

The current API has no unchanged-input `spec.rerun` trigger. To execute another search, change an
execution input, such as a seed in `inputOverrides`, and apply it as a new generation.

## Work with v1beta1 Clients

`v1beta1` and `v1beta2` represent different workflows. Conversion preserves the complete native
payload instead of attempting a semantic conversion:

- A native v1beta2 Request stores its fields directly under `spec`. A v1beta1 client sees that
  payload under `spec.v1beta2`.
- A native v1beta1 Request keeps its profiler fields directly under `spec`. A v1beta2 client sees
  that payload under `spec.v1beta1`.
- Native fields and the foreign-version carrier are mutually exclusive. The carrier is read-only
  through the foreign API version.

Use the native API version to modify a Request. Run and Candidate resources have no v1beta1 form.

## Clean Up

Delete the Request and its owned Runs and Candidates:

```bash
kubectl delete dgdr qwen-search -n inference
```

Independently created DGDs remain until you delete them:

```bash
kubectl delete dgd qwen-production -n inference
```

## Related Documentation

- [Auto Deployment Overview](overview.mdx)
- [Auto Deploy with DGDR v1beta1](auto-deploy-with-dgdr.md)
- [DynamoGraphDeployment Reference](../../reference/kubernetes-api/dynamo-graph-deployment.mdx)
- [Full Kubernetes API Reference](../../reference/kubernetes-api/full-api-reference.mdx#v1beta2-dynamographdeploymentrequest)
- [AISimulate Sweeper Overview](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/overview.md)
- [AISimulate Sweeper Configuration](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/configuration.md)
- [AISimulate Traffic](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/traffic.md)
- [AISimulate Optimization Goals](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/optimization-goals.md)
- [Dynamo AISimulate Integration](../../developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/dynamo-integration.md)
