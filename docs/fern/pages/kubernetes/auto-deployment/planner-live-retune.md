---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Live Retune
subtitle: Let the Planner tune the KV router and its own scaling parameters on a running DynamoGraphDeployment from its recorded traffic, through DGDR v1beta2 search runs.
---

> [!CAUTION]
> **Experimental, development clusters only.** Live retune builds on DGDR v1beta2, which is
> available without an API feature gate. Retune remains disabled by default and writes
> router parameters through the frontend admin API, which has no authentication. Do not enable it
> on a cluster that serves production traffic.

Live retune is the Kubernetes form of [Periodic Retune](../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md#periodic-retune-experimental).
Every `retune.interval_seconds`, the Planner cuts the window of request traces that just ended,
creates DynamoGraphDeploymentRequests (DGDR) v1beta2 that replay the window for a grid of KV
router and Planner parameters, scores the candidates the Runs report on `retune.objective`, and
keeps the current parameters unless a candidate beats them by more than
`retune.min_improvement_percentage`. Outside advisory mode it then sets the winning router
parameters on every frontend router with `PATCH /router_policy` and applies the winning Planner
parameters to itself.

Live retune searches the three router knobs a DGDR Request can express,
`overlap_score_credit`, `prefill_load_scale` and `temperature`, in one of two setups:

- **Fixed replicas.** The Planner runs in retune-only mode (`enable_load_scaling` and
  `enable_throughput_scaling` both `false`) and never scales. Each candidate replays with the
  replica counts of the DGD spec.
- **Scaling.** The Planner scales on load, throughput or both, and `retune.planner` may also
  search its runtime parameters, the scaling policy included (see
  [Search domains](../../reference/components/planner-configuration.mdx#search-domains)). Each
  candidate replays the Planner's own configuration with the candidate's fields, starting from
  the replica counts that were ready at the window start, as
  [replay retune](../../developer-guide/knowledge-base/modular-components/planner/planner-design.md#candidates-and-the-sweep)
  does. This needs an operator whose configured planner image includes this release: the Runs'
  AISimulate Planner adapter must read the Requests' `planner.retune_sweep` section.

## How it works

1. **Record.** Each frontend writes a `request_end` record per request to its own directory on a
   shared ReadWriteMany claim (`<segments_dir>/<dgd>/<pod>/`).
2. **Read the history** (scaling only). At `L_k`, before any Request is created, the Planner
   reads the records of the `history_seconds` before the window and buckets them at each
   throughput interval a candidate scales at. With throughput scaling it runs the load-predictor
   pre-sweep on them and prepares every predictor a decision could swap in. This takes another
   `cut_budget_seconds` at most.
3. **Cut a window.** At `L_k = window end + grace_seconds`, the Planner merges the records of the
   window into one canonical trace file,
   `<windows_dir>/<dgd>/<window_end_ms>-<selection>.jsonl.gz`, in a subprocess bounded by
   `cut_budget_seconds`. `<selection>` is a digest of the window's start, end and cutoff
   (`window end + grace_seconds`), so a Planner restarted with another `interval_seconds` or
   `grace_seconds` cuts its own file and creates its own Requests instead of reusing those of a
   window with the same end.
4. **Search.** The Planner splits the grid into at most `run.max_shards` Cartesian shards of at
   most 100 points and creates one DGDR v1beta2 Request per shard. With scaling, the Planner
   points are one more shard dimension. Each Request replays the window file on the pinned
   `retune.dgdr.engine`, with the replica counts of the DGD spec (fixed replicas) or those ready
   at the window start (scaling). With scaling, it also carries the Planner's scaling
   configuration, the shard's Planner points and the history warm-up in
   `spec.aisim.inputOverrides.planner.retune_sweep`. One more single-point Request replays a grid
   point again as a common-random-numbers probe: its score must match the grid's within a
   relative error of 1e-9.
5. **Decide.** The Planner reads each Run's Candidates, scores them on `retune.objective`, and
   decides with the same rules as replay retune. A decision is persisted in a Lease before it
   reaches any router.
6. **Apply.** The Planner sets all eight router policy parameters of every frontend router to
   the decided values, and keeps them there: a manual change to them is reverted. Decided Planner
   parameters take effect in the Planner itself once the decision is persisted, through its
   [runtime reconfiguration](../../reference/components/planner-configuration.mdx#system-port-reconfiguration-api),
   with a new load predictor warmed on the window's history when the decision needs one. They
   share the control API's one-update-at-a-time rule: the decision's Planner parameters are
   dropped when another update is still preparing, or activates while their predictor is warmed.
   A Planner that restarts runs its configured values again: retune does not restore applied
   Planner parameters, and logs the ones it lost.

Windows are aligned to `retune.dgdr.phase_seconds + k × interval_seconds` in epoch time, so a
restarted Planner continues the same windows. Leadership and state live in two Leases owned by
the DGD, `<dgd>-planner-retune` and `<dgd>-planner-retune-state`. The first window after retune is
first enabled is skipped, since its trace is incomplete.

Every DGDR trace Run replays with common random numbers: ordinal request ids and a fixed router
seed. A Run's score therefore does not depend on the other candidates or the Run's parallelism.
This applies to all DGDR trace Runs, not only those the Planner creates.

## Prerequisites

- **Operator with DGDR v1beta2 traces.** Use an operator that accepts `spec.workload.trace`.
  The platform chart grants the Planner ServiceAccount the required [RBAC](#rbac).
- **Run Job scheduling.** The namespace needs a LimitRange for the Run Jobs, and the operator
  must discover GPUs, through DCGM or GPU discovery with labeled nodes. See
  [Auto Deploy with DGDR](auto-deploy-with-dgdr.md#prerequisites).
- **A public model.** The Run Job has no Hugging Face credentials and no model mount, so
  `retune.dgdr.model` must be a public Hugging Face repository id that AISimulate can resolve.
- **A ReadWriteMany claim** in the DGD namespace, dedicated to traces and writable by the user
  the frontend and Planner containers run as. The operator does not provision the claim. The
  Planner writes each window file with mode `0644`, so the Run Job, which reads it as UID 1000
  with supplemental group 65532, can read it whatever user the Planner runs as.
- **Frontend admin API support.** The frontend must expose `/router_policy` for observing and
  updating router parameters. See [Deployment rules](#deployment-rules).
- **Operator-owned Retune wiring on the Planner component.** Enable the component's `retune`
  field and name the trace claim as shown below. This configures the frontend admin listener;
  restrict access to its unauthenticated port before enabling the feature.
- **One Planner replica.**
- **An acknowledged Retune target.** The operator publishes
  `status.components.<planner>.retune.configHash` with a global `status.observedGeneration` matching
  `metadata.generation`. Retune does not consume role Pod templates, commands, arguments, or
  environment lists from spec or status.

## Configure the frontend

Enable `retune` on the component whose type is `planner`, and reference an existing writable trace
claim. The operator selects the one `frontend` component by type, independently of its name. The
following DGD spec excerpt omits images, workers, and the ordinary Planner configuration mount:

```yaml
spec:
  components:
  - name: gateway
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          command: [python3]
          args: [-m, dynamo.frontend]
  - name: optimizer
    type: planner
    retune:
      enabled: true
      traceStorage:
        claimName: request-traces
      router:                       # Startup values, not search domains; optional.
        overlapScoreCredit: 1.0      # Default: 1.0.
        prefillLoadScale: 1.0        # Default: 1.0.
        temperature: 0.0            # Default: 0.0.
```

This normal component API is not under `experimental`. It requests deployment wiring, not a
search strategy or replica-scaling mode. Exactly one frontend and at most one enabled Retune
Planner are supported. An enabled standalone DCD cannot provide this graph-wide wiring.

The optional `router` fields must be finite, non-negative numbers. The operator supplies them
through existing frontend environment variables, enables KV routing and replica synchronization,
and passes the same values to the Planner. The remaining router settings use ordinary SDK
defaults, not a version-pinned profile. These are startup values; the Planner's `retune.router`
configuration below defines the search space instead.

Use otherwise default router settings on the frontend and workers. Do not override these values
through command-line arguments or router policy files. Normal frontend precedence still applies;
this wiring does not validate the full effective runtime configuration. The Planner checks live
policy parameters, but that does not prove agreement on other router settings.

The operator mounts the claim at `/trace` in the frontend and Planner and owns these frontend
bindings while enabled:

| Binding | Operator-owned value |
| --- | --- |
| `DYN_ROUTER_MODE` | `kv` |
| `DYN_ROUTER_REPLICA_SYNC` | `true` |
| `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT` | `retune.router.overlapScoreCredit` |
| `DYN_ROUTER_PREFILL_LOAD_SCALE` | `retune.router.prefillLoadScale` |
| `DYN_ROUTER_TEMPERATURE` | `retune.router.temperature` |
| `DYN_POD_NAME` | The Pod name from the Downward API |
| `DYN_REQUEST_TRACE` | `1` |
| `DYN_REQUEST_TRACE_RECORDS` | `request_end` |
| `DYN_REQUEST_TRACE_SINKS` | `file` |
| `DYN_REQUEST_TRACE_FILE_FORMAT` | `jsonl_gz` |
| `DYN_REQUEST_TRACE_FILE_PATH` | `/trace/segments/<dgd>/$(DYN_POD_NAME)/req` |
| `DYN_ENABLE_FRONTEND_ADMIN_API` | `1` |
| `DYN_FRONTEND_ADMIN_HOST` | The Pod IP from the Downward API |
| `DYN_FRONTEND_ADMIN_PORT` | `8002` |

These feature-owned bindings replace same-named authored values; this does not reserve all
operator environment names. Existing direct read-write mounts of the same claim at `/trace`
can be reused. Conflicting paths, another claim, read-only mounts, and subpaths fail rendering.
The trace writer and Planner create their required subdirectories; the claim must be writable.

The operator supplies a versioned JSON launch contract through `DYN_PLANNER_RETUNE_CAPABILITY`
in the Planner. It contains the selected frontend identity, trace layout, admin port, router
startup values, and an opaque `configHash`. The unchanged frontend consumes its existing
environment variables, not this contract. Do not set or override these operator-owned bindings.

After the target workloads acknowledge lowering, the operator publishes only the binding hash:

```yaml
status:
  observedGeneration: 7             # Applied target generation, not Pod readiness.
  components:
    optimizer:
      retune:
        configHash: "<operator-generated>"
```

Retune waits until its immutable launch hash matches the current target hash. Generic role
status and a frontend runtime hash acknowledgment are not required. During a storage rollout,
an old Planner still mounted to PVC A must not create replay
requests referring to PVC B. Cleanup uses the historical binding checkpointed with the previous
router process identities. Replica-only edits do not change this hash or roll the Planner.

The admin listener is unauthenticated: admit only the Planner to port `8002` before enabling the
feature ([Security](#security) has a NetworkPolicy). No NetworkPolicy is created automatically.
- Records are dropped without a count for `prompt_embeds`, multimodal, `n > 1` and `best_of > 1`
  requests, and when the writer lags or a batch write fails.
- The trace holds unkeyed xxh3 hash chains of the prompts. Anyone who can create pods in the
  namespace can mount the claim and read them.

See [Agent Tracing](../../use-cases/agents/agent-tracing.md) for the `request_end` record.

## Configure the Planner

The operator supplies the shared claim mount and immutable semantic launch contract. Pass a Planner
configuration like the one below with `--config`; search and scaling remain runtime configuration:

```yaml
environment: kubernetes
backend: vllm
mode: agg
advisory: true                     # start advisory; see Roll out
enable_load_scaling: false         # fixed replicas: both flags false
enable_throughput_scaling: false
retune:
  interval_seconds: 21600          # T: window length and decision cadence
  grace_seconds: 120               # covers p99 request time + 1 s flush + claim visibility
  timeout_seconds: 20400           # required; at most T - 600; includes the window cut
  decision_latency_seconds: 0
  objective: GoodputPerGPU
  sla_ttft_ms: 500
  sla_itl_ms: 30
  router: default                  # 3 x 9 x 4 = 108 points
  executor: dgdr
  dgdr:
    phase_seconds: 0
    model: Qwen/Qwen3-32B
    sku: h200_sxm
    engine:
      mode: aggregated
      backend: vllm
      context_length: 32768
      workers:
        aggregated:
          parallelism:
            preset:
            - {tensor: 2, pipeline: 1, attention_data: 1, moe_tensor: 1, moe_expert: 1}
          scheduler: {max_batched_tokens: 8192, max_sequences: 256}
        prefill: null
        decode: null
    cut_budget_seconds: 900
    job_startup_seconds: 300
    run:
      parallelism: 8
      candidate_timeout_seconds: 1800
      max_shards: 3
    keep_requests: 5
    max_window_rows: 5000000
    promote_last_advisory: false
```

`retune.dgdr` fields:

| Field | Default | Meaning |
|---|---|---|
| `claim_name` | launch contract | Deprecated compatibility setting; if explicitly supplied, must match the bound trace claim |
| `mount_path` | launch contract | Deprecated compatibility setting; if explicitly supplied, must match the bound mount |
| `segments_dir`, `windows_dir` | launch contract | Deprecated compatibility settings; explicitly supplied directories must match the bound layout |
| `phase_seconds` | `0` | Window phase in epoch seconds |
| `model` | required | Hugging Face repository id the workers serve; the Requests' `spec.model.name` |
| `sku` | required | The Requests' `spec.hardware.gpu.sku` |
| `engine` | required | The pinned AISimulate engine section; see [Engine section](#engine-section) |
| `cut_budget_seconds` | `900` | Deadline of the window cut subprocess |
| `job_startup_seconds` | `300` | Run Job start-up time reserved when deciding whether a shard can still finish |
| `run.parallelism` | `8` | AISimulate `optimizer.parallelism` of each Run; Job resources may lower it |
| `run.candidate_timeout_seconds` | `1800` | AISimulate `optimizer.candidate_timeout_seconds` of each Run |
| `run.max_shards` | `3` | Most grid shards (Requests) per window; the probe and off-grid points do not count |
| `keep_requests` | `5` | Windows whose finished Requests are kept for audit |
| `max_window_rows` | `5000000` | A larger window fails with `window_too_large` |
| `promote_last_advisory` | `false` | On switching to active, apply the last advisory recommendation if it is at most one interval old and the configuration is unchanged |
| `lease_duration_seconds` | `60` | Leadership Lease duration, renewed every third of it or sooner; at least `37`, so two renewals of up to 11 s each fit before the Planner stops leading, 15 s before the Lease expires |
| `actuator_interval_seconds` | `15` | Period of the router and clean-up rounds |

For the other `retune` fields and the limits that `executor: dgdr` adds to them, see the
[Retune field reference](../../reference/components/planner-configuration.mdx#retune-experimental).

To let the Planner scale and retune its own parameters, enable the scaling modes and add a
`planner` domain. A `router` domain is still required, since every Request sets the three router
knobs; pin them with scalars to the frontend's startup values to retune the Planner alone:

```yaml
optimization_target: sla
ttft_ms: 500
itl_ms: 30
enable_load_scaling: true
enable_throughput_scaling: true
retune:
  # interval_seconds, timeout_seconds, executor and dgdr as above
  router:
    overlap_score_credit: 1.0      # pinned to the frontend's startup values
    prefill_load_scale: 1.0
    temperature: 0.0
  planner:
    scaling_policy: {preset: [hybrid_180_5, hybrid_600_5]}
    load_sensitivity: {preset: false}
    load_scaling_down_sensitivity: {choices: [70, 90]}
```

Live retune takes its scoring SLA when it starts: `retune.sla_ttft_ms` and `retune.sla_itl_ms`,
or the Planner's `ttft_ms` and `itl_ms` where they are unset. An update of `ttft_ms` or `itl_ms`
through the
[runtime reconfiguration API](../../reference/components/planner-configuration.mdx#system-port-reconfiguration-api)
that would change that SLA fails with HTTP `422`. To change the SLA, restart the Planner with the
new values; to keep `ttft_ms` and `itl_ms` changeable at runtime, set both `retune` SLA fields.

### Engine section

`retune.dgdr.engine` becomes `spec.aisim.inputOverrides.engine` of every Request, so every
value in it must be a scalar. A `{choices: ...}` or `{range: ...}` domain, or an omitted knob that
AISimulate searches by default, would make the Run search engine settings as well.

- `mode` is `aggregated` for `mode: agg` and `disaggregated` for `mode: disagg`.
- Each worker role of the mode (`aggregated`, or `prefill` and `decode`) sets
  `parallelism.preset` to a list of exactly one entry with `tensor`, `pipeline`, `attention_data`,
  `moe_tensor` and `moe_expert`, and sets `scheduler.max_batched_tokens` and
  `scheduler.max_sequences`.
- The preset entry may also set `prefill_context`, but only to `1`, and must not set
  `decode_context`: the pinned AISimulate recommendation does not run context parallelism.
- Do not set `replicas` anywhere: the Planner fills them in at each launch, from the DGD spec with
  fixed replicas or from the ready counts at the window start with scaling, and sets
  `spec.hardware.gpu.budget` to their GPU total.
- Write roles the mode does not use as `null`, or leave them out.
- Set the model and GPU with `retune.dgdr.model` and `retune.dgdr.sku`, not in the engine.
- Leave `backend_version` out to let the pinned AISimulate pick its version for
  `retune.dgdr.sku`. If you set it, it must be a version AISimulate accepts for that GPU;
  otherwise every Run fails with `UnlistedQueryVersionError`.

### Time budget

Each shard must finish before the window's deadline, `L_k + timeout_seconds`. A shard of `n`
points needs

```text
need(n) = cut_budget_seconds + job_startup_seconds
          + (1 + ceil((n - 1) / run.parallelism)) x run.candidate_timeout_seconds
```

AISimulate evaluates the first candidate alone, hence the `1 +`. With scaling, a shard also needs
another `cut_budget_seconds` for the history read and the pre-sweep. The Planner rejects a
configuration whose largest grid shard needs more than `timeout_seconds`, because the baseline
can fall in any shard and a window without a baseline score fails. With the defaults, the 108-point
grid splits into 3 shards of 36 points and needs 900 + 300 + 6 × 1800 = 12000 seconds. Measure
replay time and memory on your trace before relying on these defaults.

## Deployment rules

The Planner checks three kinds of rules. None of them stops a running Planner except a
configuration error at startup.

**Configuration**, checked when the Planner starts:

| Rule | Requirement |
|---|---|
| P1 | `environment: kubernetes`; `decision_latency_seconds` is `0`, or `measured` with `decision_latency_scale: 1`; `timeout_seconds` is set and at most `interval_seconds - 600` |
| P2 | `retune.router` is set; to retune only the Planner, pin its three knobs with scalars |
| P3 | `retune.router` writes only `overlap_score_credit`, `prefill_load_scale`, `temperature`, `policy: kv_router` and `prefill_load_model: {type: none}` |
| P5 | `retune.dgdr.engine` is all scalars, as in [Engine section](#engine-section) |
| P13 | The largest grid shard has at most 100 points and fits the [time budget](#time-budget). With scaling, the Planner points count as one more shard dimension, and the history before a window holds at most 1024 buckets of the shortest throughput interval |

The Planner reads its immutable semantic launch contract, not component Pod templates. It uses
the existing SDK defaults with the supplied router values. This does not validate arbitrary
frontend or worker overrides, ConfigMap or Secret values, or complete trace/admin runtime
configuration. Keep the supported default configuration described above.

**Deployment**, checked at startup, when the DGD changes and before every router round. While one
fails, the Planner launches no windows, resets the routers to their startup parameters, and
retries:

| Rule | Requirement |
|---|---|
| P4 | The supported versioned launch contract resolves the three startup router values using existing SDK defaults |
| P7 | The DGD serves one model; the frontend is not an EPP or a `frontendSidecar` |
| P8 | Effective router configuration satisfies frontend replica-synchronization requirements |
| P9 | No `scalingAdapter` on worker components; one Planner replica |
| P10 | The immutable launch hash matches current `status.components.<planner>.retune.configHash`; explicitly configured legacy trace-storage values agree with the launch contract |
| P14 | Every router's startup policy parameters match the values resolved from the semantic launch contract |

**Health**, checked every round. While one fails, windows do not launch; nothing is reset:

| Rule | Requirement |
|---|---|
| P11 | The API server accepts a dry-run Request with `spec.workload.trace` |
| P12 | At least `max(1, replicas)` frontends are Ready, each with a router per worker type, and `/router_policy` answers on each one's admin listener |

The Planner only warns when AISimulate does not list the model among its default models (P6).

## Roll out

1. Prepare a writable trace claim and restrict frontend admin access, then enable the Planner
   component's `retune` setting with that claim.
2. Start with `advisory: true`. In advisory mode the Planner records the decision it would make
   and changes no router value it has not set itself. Watch `dynamo_planner_retune_ready` and the
   decision history:

   ```bash
   kubectl get lease <dgd>-planner-retune-state -n <namespace> \
     -o jsonpath='{.metadata.annotations.nvidia\.com/planner-retune-history}'
   ```

3. Set `advisory: false`. With `retune.dgdr.promote_last_advisory: true`, the Planner applies the
   last advisory recommendation right away if it is at most one interval old and the
   configuration has not changed since.

## Disable

1. Set `advisory: true` and wait until `dynamo_planner_retune_router_converged` is 1 and
   `dynamo_planner_retune_rollback_pending` is 0. The routers are then back at their startup
   parameters. The mode stored in the state alone does not show that.
2. Remove `retune` from the Planner configuration.
3. Keep the Planner component's operator-owned `retune` wiring enabled while cleanup completes,
   so the old frontend admin endpoint and trace storage remain reachable.
4. Delete the Planner's Requests and both Leases. Select the Requests by the DGD's
   `metadata.uid`: the `nvidia.com/planner-retune-dgd` label holds a hash instead of the name
   when the name is longer than 63 characters.

   ```bash
   kubectl delete dynamographdeploymentrequests.nvidia.com -n <namespace> \
     -l nvidia.com/planner-retune-dgd-uid=$(kubectl get dgd <dgd> -n <namespace> -o jsonpath='{.metadata.uid}')
   kubectl delete lease -n <namespace> <dgd>-planner-retune <dgd>-planner-retune-state
   ```

   A Planner started without `retune` also cleans up automatically. It looks for the Leases
   in the background after startup (once more after a
   transient API error), resets any router the removed retune changed, then deletes the
   Requests and both Leases.

5. Set the Planner component's `retune.enabled` to `false`, or remove that component field. This
   removes the operator-owned wiring on the next normal frontend/Planner rollout. Do not remove
   the wiring before confirming router rollback.
6. Delete `<segments_dir>/<dgd>` and `<windows_dir>/<dgd>` from the claim.

Before upgrading the operator or its CRDs, switch to advisory or disable retune, then delete the
retune Requests.

The Planner checkpoints prior wiring and router process identities in Retune Lease state format
v2, and can read v1 state. Before upgrading an active deployment from v1 state, finish rollback
and cleanup with the old Planner, including removal of its Retune Leases. A legacy state that
may still own router policies has no historical binding: the new Planner retains it and reports
that cleanup is blocked rather than guessing that the current frontend is the old one.

Before downgrading to an older Planner that cannot read v2, complete
the rollback and cleanup above with the newer Planner, including removal of its Retune Leases.

## Security

- **Frontend admin API.** The Planner calls `/router_policy` on each frontend's admin listener,
  which has no authentication. The same listener serves the busy-threshold API
  (`/busy_threshold` and `/router_policy/busy_threshold`), and a threshold set there makes the
  frontend reject requests. [Configure the frontend](#configure-the-frontend) binds the listener
  to the Pod's IP, so anything that reaches a frontend Pod on port `8002`
  can change its router policies and its busy thresholds. Admit only the Planner to that
  port, for example with the NetworkPolicy below, and do not expose it through a Service or an
  Ingress. While a frontend's admin listener does not answer `/router_policy`, the Planner
  launches no windows (P12) and logs why.
- **Ownership.** Outside advisory mode the Planner owns the eight router policy parameters and
  reverts manual changes.
- **Trace data.** See the privacy note in [Configure the frontend](#configure-the-frontend).

For a DGD named `my-dgd`, this NetworkPolicy admits only its Planner pods to port 8002 of its
frontend pods and leaves every other port of the frontends as open as with no policy: the other
TCP ports, and every UDP port, which the QUIC response plane (`DYN_RESPONSE_PLANE=quic`) uses to
receive worker responses. The component contract owns admin port `8002`. The policy
takes effect only where the cluster's network plugin enforces NetworkPolicy, and needs Kubernetes
1.25 or later for `endPort`:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: my-dgd-frontend-admin
spec:
  podSelector:
    matchLabels:
      nvidia.com/dynamo-graph-deployment-name: my-dgd
      nvidia.com/dynamo-component-type: frontend
  policyTypes: [Ingress]
  ingress:
  # Every port but the admin port, from anywhere: only when no other policy
  # selects these pods (see below).
  - ports:
    - {protocol: TCP, port: 1, endPort: 8001}
    - {protocol: TCP, port: 8003, endPort: 65535}
    - {protocol: UDP}
  # The admin port, from the Planner only.
  - from:
    - podSelector:
        matchLabels:
          nvidia.com/dynamo-graph-deployment-name: my-dgd
          nvidia.com/dynamo-component-type: planner
    ports:
    - {protocol: TCP, port: 8002}
```

Keep the first rule only when no other NetworkPolicy selects the frontend pods. Kubernetes admits a
connection that any policy selecting the pod allows, so next to a stricter policy the first rule
would open every other port to every source. If a policy already restricts the frontends, leave
the first rule out, and check that none of that policy's rules admits the admin port: a rule
without `ports` admits every port, so list the frontend's other ports in it.

### RBAC

The platform chart includes these rules in the Planner Role for namespace-restricted installs
and the Planner ClusterRole for cluster-wide installs.

| API group | Resources | Verbs |
|---|---|---|
| `nvidia.com` | `dynamographdeploymentrequests` | create, get, list, delete |
| `nvidia.com` | `dynamographdeploymentruns` | get |
| `nvidia.com` | `dynamographdeploymentcandidates` | list |
| `coordination.k8s.io` | `leases` | get, create, update, delete |

The chart has one Planner role for the whole install, so Planners without `retune` get these
rules too. The Planner also lists pods with its existing permission. These rules are namespace-wide: a
Role or a RoleBinding of the ClusterRole has no `resourceNames`, because the Planner creates its
Leases and Requests under per-DGD names. As a result, any workload that runs as the Planner
ServiceAccount can:

- create DGDR Requests, which start Run Jobs, in its namespace;
- read, update or delete any Lease in its namespace, not only the two retune Leases. In a
  namespace-restricted install, the Planner Role lives in the release namespace, which by default
  also holds the operator's leader-election Lease, so such a workload can disrupt operator leader
  election.

In a cluster-wide install, the operator binds the Planner ClusterRole into each DGD namespace with
a RoleBinding. Kubernetes requires the operator to hold every permission it binds. If you
manage operator RBAC separately, include the rules above so the operator can reconcile the
Planner RoleBinding in new DGD namespaces.

## Metrics

The Planner exports these on `metric_reporting_prometheus_port`:

| Metric | Type | Meaning |
|---|---|---|
| `dynamo_planner_retune_decisions_total{outcome}` | counter | Decisions, counted after the state write persisted them |
| `dynamo_planner_retune_failures_total{reason}` | counter | Failed windows by reason, such as `launch_too_late`, `replicas_changed`, `window_cut_failed`, `trace_corrupt`, `crn_mismatch` or `k8s_error` |
| `dynamo_planner_retune_skipped_total{reason}` | counter | Due windows held back: `not_leader`, `not_ready` or `already_decided` |
| `dynamo_planner_retune_leader`, `_ready`, `_initialized`, `_rollback_pending` | gauge | Supervisor state, 0 or 1 |
| `dynamo_planner_retune_controller_generation` | gauge | Generation of the retune controller; it grows when a DGD change rebuilds it |
| `dynamo_planner_retune_baseline_score`, `_best_score` | gauge | Scores of the latest decision; NaN when not scored |
| `dynamo_planner_retune_sweep_seconds`, `_missing_candidates` | gauge | Duration of the latest sweep and its candidates without a score |
| `dynamo_planner_retune_router_converged` | gauge | 1 when every frontend router holds the target parameters |
| `dynamo_planner_retune_router_patch_errors_total{code}` | counter | Failed router updates by HTTP status or error kind |
| `dynamo_planner_retune_trace_sink_missing{pod}` | gauge | 1 for a Ready frontend without its own trace directory |
| `dynamo_planner_retune_trace_dropped_members_total` | counter | Corrupt or truncated trace gzip members skipped by window cuts |

## Limitations

| Not supported | Reason |
|---|---|
| Retune with scaling on an operator whose planner image predates this release | The Run's AISimulate Planner adapter rejects `planner.retune_sweep`, so every Run fails |
| Restoring applied Planner parameters after a Planner restart | They live in the Planner process, and a load predictor needs a history warm-up; the next window starts from the configured values |
| Load-predictor history beyond 1024 buckets of the shortest throughput interval | The warm-up travels inside every Request and Candidate |
| Frontends with non-default routing fields beyond the three knobs, or `kv_event_lag_ms` other than 0 | The DGDR router section expresses only the three knobs, and the Run does not pass a lag |
| Workers that set their own router mode (`--router-mode` or `DYN_ROUTER_MODE`) | A worker's advertised router configuration replaces the frontend's for its workers, and the Run replays only the frontend's |
| Missing/stale target roles or a target container without a declared command | The Planner waits for current operator facts; it does not infer an image entrypoint or reconstruct defaults from the authored spec |
| Worker arguments that put an unknown status value where an option could stand, including an unquoted `$VAR` of one in a worker shell script or one that `eval` or a shell given `-c` parses again (P4 fails) | The Planner cannot read the value, so it cannot rule out a router mode the worker would advertise. No injected name or raw source kind makes an unknown value automatically safe |
| Router flags that a worker produces through shell command substitution, backticks, script variables, positional parameters, literal text that `eval` or a nested shell parses again, or a script file, and unknown values in text that a shell or `xargs` reads from a heredoc, a here-string or a pipe, or that `su -c` runs (P4 does not see them) | The Planner reads the declared target command and does not run its script |
| Raising `overlap_score_credit` on a frontend that starts it at 0: an omitted credit then searches only 0, and a written positive value fails P4 | A live router rejects a positive credit after starting at 0 |
| Private, fine-tuned or local-path models | The Run Job has no Hugging Face credentials or model mount |
| Several frontends without router replica sync | AISimulate simulates one router |
| EPP or Gateway API Inference Extension frontends, `frontendSidecar`, TLS frontends | No `/router_policy` there, different addressing, or certificates |
| Run Job resources, deadline and tolerations | Fixed by the DGDR Run Job |
| Time-based segment rotation and a segment count limit | The frontend trace writer rotates by size only |
| A separate or authenticated admin listener | Not available in the frontend yet |
