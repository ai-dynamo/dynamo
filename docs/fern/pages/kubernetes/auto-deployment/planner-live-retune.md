---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Live Retune
subtitle: Let the Planner tune the KV router of a running DynamoGraphDeployment from its own recorded traffic, through DGDR v1beta2 search runs.
---

> [!CAUTION]
> **Experimental, development clusters only.** Live retune builds on DGDR v1beta2, which the
> operator serves only with the experimental `dgdrV1Beta2.enabled` Helm value, and it writes
> router parameters through the frontend admin API, which has no authentication. Do not enable it
> on a cluster that serves production traffic.

Live retune is the Kubernetes form of [Periodic Retune](../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md#periodic-retune-experimental).
Every `retune.interval_seconds`, the Planner cuts the window of request traces that just ended,
creates DynamoGraphDeploymentRequests (DGDR) v1beta2 that replay the window for a grid of KV
router parameters, scores the candidates the Runs report on `retune.objective`, and keeps the
current parameters unless a candidate beats them by more than `retune.min_improvement_percentage`.
Outside advisory mode it then sets the winner on every frontend router with
`PATCH /router_policy`.

Live retune searches the three router knobs a DGDR Request can express:
`overlap_score_credit`, `prefill_load_scale` and `temperature`. Replicas stay fixed: the Planner
runs in retune-only mode and never scales.

## How it works

1. **Record.** Each frontend writes a `request_end` record per request to its own directory on a
   shared ReadWriteMany claim (`<segments_dir>/<dgd>/<pod>/`).
2. **Cut a window.** At `L_k = window end + grace_seconds`, the Planner merges the records of the
   window into one canonical trace file,
   `<windows_dir>/<dgd>/<window_end_ms>-<selection>.jsonl.gz`, in a subprocess bounded by
   `cut_budget_seconds`. `<selection>` is a digest of the window's start, end and cutoff
   (`window end + grace_seconds`), so a Planner restarted with another `interval_seconds` or
   `grace_seconds` cuts its own file and creates its own Requests instead of reusing those of a
   window with the same end.
3. **Search.** The Planner splits the router grid into at most `run.max_shards` Cartesian shards of
   at most 100 points and creates one DGDR v1beta2 Request per shard. Each Request replays the
   window file on the pinned `retune.dgdr.engine` with the replica counts of the DGD spec. One more
   single-point Request replays a grid point again as a common-random-numbers probe: its score
   must match the grid's within a relative error of 1e-9.
4. **Decide.** The Planner reads each Run's Candidates, scores them on `retune.objective`, and
   decides with the same rules as replay retune. A decision is persisted in a Lease before it
   reaches any router.
5. **Apply.** The Planner sets all eight router policy parameters of every frontend router to
   the decided values, and keeps them there: a manual change to them is reverted.

Windows are aligned to `retune.dgdr.phase_seconds + k × interval_seconds` in epoch time, so a
restarted Planner continues the same windows. Leadership and state live in two Leases owned by
the DGD, `<dgd>-planner-retune` and `<dgd>-planner-retune-state`. The first window after retune is
first enabled is skipped, since its trace is incomplete.

Every DGDR trace Run replays with common random numbers: ordinal request ids and a fixed router
seed. A Run's score therefore does not depend on the other candidates or the Run's parallelism.
This applies to all DGDR trace Runs, not only those the Planner creates.

## Prerequisites

- **Operator with DGDR v1beta2 traces.** Install the platform chart with
  `dynamo-operator.dgdrV1Beta2.enabled=true` and an operator that accepts `spec.workload.trace`.
  Also set `dynamo-operator.dgdrV1Beta2.plannerLiveRetune=true` to grant the Planner
  ServiceAccount the access it needs (see [RBAC](#rbac)).
  In a cluster-wide install, the operator must also hold the DGDR v1beta2 Run and Candidate
  permissions of [ai-dynamo/dynamo#17308](https://github.com/ai-dynamo/dynamo/pull/17308); see
  [RBAC](#rbac).
- **Run Job scheduling.** The namespace needs a LimitRange for the Run Jobs, and the operator
  must discover GPUs, through DCGM or GPU discovery with labeled nodes. See
  [Auto Deploy with DGDR](auto-deploy-with-dgdr.md#prerequisites).
- **A public model.** The Run Job has no Hugging Face credentials and no model mount, so
  `retune.dgdr.model` must be a public Hugging Face repository id that AISimulate can resolve.
- **A ReadWriteMany claim** in the DGD namespace, dedicated to traces. Create `segments/` and
  `windows/` at its root, writable by the user the frontend and Planner containers run as. The
  Planner writes each window file with mode `0644`, so the Run Job, which reads it as UID 1000
  with supplemental group 65532, can read it whatever user the Planner runs as.
- **A KV-router frontend with default routing fields**, and workers that do not set their own
  router mode or pass a value the Planner cannot read where an option could stand. See
  [Deployment rules](#deployment-rules).
- **The frontend admin listener, reachable from the Planner.** The frontend serves
  `/router_policy` only on its admin listener, which is off by default and binds `127.0.0.1`.
  See [Configure the frontend](#configure-the-frontend).
- **One Planner replica.**

## Configure the frontend

Add the trace environment, the admin listener and the claim to the frontend container of the
DGD. The planner checks these values against `retune.dgdr` at startup and whenever the DGD
changes.

```yaml
spec:
  components:
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          command: [python3]
          args: [-m, dynamo.frontend, --router-mode, kv]
          env:
          - name: DYN_POD_NAME
            valueFrom: {fieldRef: {fieldPath: metadata.name}}
          - {name: DYN_REQUEST_TRACE, value: "1"}
          - {name: DYN_REQUEST_TRACE_RECORDS, value: request_end}
          - {name: DYN_REQUEST_TRACE_SINKS, value: file}
          - {name: DYN_REQUEST_TRACE_FILE_FORMAT, value: jsonl_gz}
          - name: DYN_REQUEST_TRACE_FILE_PATH
            value: /trace/segments/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/req
          - {name: DYN_REQUEST_TRACE_FILE_ROLL_BYTES, value: "67108864"}
          - {name: DYN_REQUEST_TRACE_CAPACITY, value: "8192"}
          - {name: DYN_ENABLE_FRONTEND_ADMIN_API, value: "1"}
          - {name: DYN_FRONTEND_ADMIN_HOST, value: "0.0.0.0"}
          volumeMounts:
          - {name: request-traces, mountPath: /trace}
        volumes:
        - name: request-traces
          persistentVolumeClaim: {claimName: request-traces}
```

- `DYN_REQUEST_TRACE_FILE_PATH` is a file prefix directly under
  `<mount_path>/<segments_dir>/$(DYN_PARENT_DGD_K8S_NAME)/$(DYN_POD_NAME)/`, such as `req`: the
  part after `$(DYN_POD_NAME)/` must not contain `/`. `<mount_path>` is where the frontend mounts
  `retune.dgdr.claim_name`. The operator injects `DYN_PARENT_DGD_K8S_NAME`.
- `DYN_ENABLE_FRONTEND_ADMIN_API` and `DYN_FRONTEND_ADMIN_HOST` start the frontend admin
  listener on the pod's addresses, on port 8002 unless `DYN_FRONTEND_ADMIN_PORT` sets another.
  The Planner reads the port from each frontend pod and sets the router policies there; the
  inference port does not serve `/router_policy`. The listener has no authentication: admit only
  the Planner to its port before you enable it ([Security](#security) has a NetworkPolicy).
  - The host is `0.0.0.0` (IPv4), `::` (IPv6 and dual-stack pods), or the pod IP through
    `valueFrom: {fieldRef: {fieldPath: status.podIP}}`.
  - The other values must be literal, without `$(VAR)` references: the Planner reads them from
    the pod spec, where Kubernetes has not expanded them. Set `DYN_FRONTEND_ADMIN_PORT` in `env`
    when an `envFrom` source could set it; otherwise the Planner assumes port 8002 and warns.
- Records are dropped without a count for `prompt_embeds`, multimodal, `n > 1` and `best_of > 1`
  requests, and when the writer lags or a batch write fails.
- The trace holds unkeyed xxh3 hash chains of the prompts. Anyone who can create pods in the
  namespace can mount the claim and read them.

See [Agent Tracing](../../use-cases/agents/agent-tracing.md) for the `request_end` record.

## Configure the Planner

Mount the same claim in the Planner container at `retune.dgdr.mount_path`, and pass a
configuration like the one below with `--config`:

```yaml
environment: kubernetes
backend: vllm
mode: agg
advisory: true                     # start advisory; see Roll out
enable_load_scaling: false         # retune-only: both flags false
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
    claim_name: request-traces
    mount_path: /trace
    segments_dir: segments
    windows_dir: windows
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
| `claim_name` | required | ReadWriteMany claim, in the DGD namespace, that holds the traces |
| `mount_path` | `/trace` | Where the Planner container mounts the claim; absolute |
| `segments_dir`, `windows_dir` | `segments`, `windows` | Claim-relative directories of the frontend segments and the window files; neither may contain the other |
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
- Do not set `replicas` anywhere: the Planner fills them in from the DGD spec at each launch and
  sets `spec.hardware.gpu.budget` to their GPU total.
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

AISimulate evaluates the first candidate alone, hence the `1 +`. The Planner rejects a
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
| P2 | `enable_load_scaling` and `enable_throughput_scaling` are both `false`; `retune.planner` is not set |
| P3 | `retune.router` writes only `overlap_score_credit`, `prefill_load_scale`, `temperature`, `policy: kv_router` and `prefill_load_model: {type: none}` |
| P5 | `retune.dgdr.engine` is all scalars, as in [Engine section](#engine-section) |
| P13 | The largest grid shard has at most 100 points and fits the [time budget](#time-budget) |

**Deployment**, checked at startup, when the DGD changes and before every router round. While one
fails, the Planner launches no windows, resets the routers to their startup parameters, and
retries:

| Rule | Requirement |
|---|---|
| P4 | The frontend runs `--router-mode kv` with every routing field the Request cannot express at its default, `kv_event_lag_ms` unset or 0, no `DYN_ROUTER_WORKER_SELECTION_POLICY`, a `none` prefill load model, router env given as plain values (no `valueFrom`), and a command that is not wrapped in a shell. The Planner cannot read `envFrom` sources: it assumes they set no router env, only warns about a source whose `prefix` could produce a `DYN_` name, and treats a `$(VAR)` that only an `envFrom` source could define as unreadable. Keep router env out of `envFrom`: P14 catches only a changed policy parameter. No worker advertises its own router configuration: worker components set neither `--router-mode` nor `DYN_ROUTER_MODE`, directly or through the graph-level `spec.env`, because a worker's advertised configuration replaces the frontend's for its workers. The Planner reads a worker's command as the operator runs it: an omitted `command` is `/bin/sh -c` with the first of `args` as the script, and a shell started through `env`, `tini --` or `dumb-init --` is a script too. For a worker whose launches are in role pod templates (`roles[].podTemplate`, no component `podTemplate`), it reads each role's main container and env. A worker with a native `runtime` sidecar is not checked: its main runs the engine, and the sidecar advertises no router configuration. A value the Planner cannot read and that could hold an option (a ConfigMap, Secret or annotation `valueFrom`, or a `$(VAR)` that the container's `env` leaves to an `envFrom` source) must not stand where a worker option could: attach it to its option (`--option=$(VAR)`) or pass it as the separate value right after `--model`, `--model-path`, `--served-model-name`, `--model-name` or `--speculative-config`. A frontend argument follows the same rule, with `--tool-call-parser`, `--reasoning-parser` and `--chat-template` as the options whose separate value it may be. In a worker shell script any such `$(VAR)` fails, since Kubernetes substitutes it before the shell parses the script; use the shell's own `$VAR` instead. A `$VAR` or `${VAR}` of such a value fails when it is unquoted, stands alone as an argument of the worker or of `env`, `export`, `source` or a shell (also by path and behind `exec`, `nohup`, `command` or `builtin`), or sits in text that `eval` or a shell given `-c` parses again; quote it outside the worker's arguments and attach it to its option among them (`--option="$VAR"`). A redirection target, such as `> "$LOG"`, is fine. The Planner does not follow the script's conditions, so the script's own assignment to a ConfigMap, Secret or annotation variable does not clear it. With an `envFrom` source that has no `prefix`, every name the container's `env` leaves undefined counts as such a value, image env such as `PATH` and `LD_LIBRARY_PATH` included, so quote those references too (`export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"`). Names the operator sets on the container do not count: `DYN_NAMESPACE`, `DYN_COMPONENT`, `DYN_PARENT_DGD_K8S_*` and `POD_*`, a worker's `DYN_SYSTEM_*`, `DYN_HEALTH_CHECK_ENABLED`, `NIXL_TELEMETRY_*` and `DYN_FORWARDPASS_METRIC_PORT`, and a multinode component's `DYNAMO_RANK` and `DYNAMO_LEADER_ADDRESS`. Other `fieldRef` values, such as the pod name, may stand anywhere except inside an option's name (`--x-$(POD_NAME)`) or as the value of a frontend router flag. The Planner does not evaluate command substitution, backticks, the scripts a worker runs, text that a shell or `xargs` reads from a heredoc, a here-string or a pipe, or a script that `su -c` runs, so keep router flags and such values out of them |
| P7 | The DGD serves one model; the frontend is not an EPP, not a `frontendSidecar`, and has no TLS |
| P8 | One frontend replica, or `--router-replica-sync` |
| P9 | No `scalingAdapter` on worker components; one Planner replica |
| P10 | The frontend trace environment, claim mount and admin listener match [Configure the frontend](#configure-the-frontend): the trace file prefix is directly under `$(DYN_POD_NAME)/`, and the admin API is on, bound to `0.0.0.0`, `::` or the pod IP, with literal values and a port from 1 to 65535 |
| P14 | Every router's startup parameters equal the ones the Planner parsed from the frontend |

**Health**, checked every round. While one fails, windows do not launch; nothing is reset:

| Rule | Requirement |
|---|---|
| P11 | The API server accepts a dry-run Request with `spec.workload.trace` |
| P12 | At least `max(1, replicas)` frontends are Ready, each with a router per worker type, and `/router_policy` answers on each one's admin listener |

The Planner only warns when AISimulate does not list the model among its default models (P6).

## Roll out

1. Prepare the claim and its directories, then add the frontend trace configuration and the
   Planner mount.
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
3. Remove the trace environment from the frontend.
4. Delete the Planner's Requests and both Leases. Select the Requests by the DGD's
   `metadata.uid`: the `nvidia.com/planner-retune-dgd` label holds a hash instead of the name
   when the name is longer than 63 characters.

   ```bash
   kubectl delete dynamographdeploymentrequests.nvidia.com -n <namespace> \
     -l nvidia.com/planner-retune-dgd-uid=$(kubectl get dgd <dgd> -n <namespace> -o jsonpath='{.metadata.uid}')
   kubectl delete lease -n <namespace> <dgd>-planner-retune <dgd>-planner-retune-state
   ```

   While the chart still grants the live retune RBAC, a Planner without `retune` does this on
   its own: it looks for the Leases in the background after it starts (once more after a
   transient API error), resets any router the removed retune changed, then deletes the
   Requests and both Leases.

5. Delete `<segments_dir>/<dgd>` and `<windows_dir>/<dgd>` from the claim.

Before upgrading the operator or its CRDs, switch to advisory or disable retune, then delete the
retune Requests.

## Security

- **Frontend admin API.** The Planner calls `/router_policy` on each frontend's admin listener,
  which has no authentication. The same listener serves the busy-threshold API
  (`/busy_threshold` and `/router_policy/busy_threshold`), and a threshold set there makes the
  frontend reject requests. [Configure the frontend](#configure-the-frontend) binds the listener
  to the pod's addresses, so anything that reaches a frontend pod on the admin port (8002 by
  default) can change its router policies and its busy thresholds. Admit only the Planner to that
  port, for example with the NetworkPolicy below, and do not expose it through a Service or an
  Ingress. While a frontend's admin listener does not answer `/router_policy`, the Planner
  launches no windows (P12) and logs why.
- **Ownership.** Outside advisory mode the Planner owns the eight router policy parameters and
  reverts manual changes.
- **Trace data.** See the privacy note in [Configure the frontend](#configure-the-frontend).

For a DGD named `my-dgd`, this NetworkPolicy admits only its Planner pods to port 8002 of its
frontend pods and leaves every other port of the frontends as open as with no policy: the other
TCP ports, and every UDP port, which the QUIC response plane (`DYN_RESPONSE_PLANE=quic`) uses to
receive worker responses. With another `DYN_FRONTEND_ADMIN_PORT`, move the three port numbers. It
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

With `dynamo-operator.dgdrV1Beta2.enabled=true` and
`dynamo-operator.dgdrV1Beta2.plannerLiveRetune=true`, the platform chart adds these rules to the
Planner Role (namespace-restricted installs) and ClusterRole (cluster-wide installs). Both values
default to `false`; `dgdrV1Beta2.enabled` alone grants the Planner nothing new.

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
a RoleBinding, and Kubernetes lets it bind only permissions it holds itself. Set
`dynamo-operator.dgdrV1Beta2.plannerLiveRetune=true` there only with an operator whose own role
grants `get` on `dynamographdeploymentruns` and `list` on `dynamographdeploymentcandidates`,
which the DGDR v1beta2 operator change in
[ai-dynamo/dynamo#17308](https://github.com/ai-dynamo/dynamo/pull/17308) adds.
Without it, the API server rejects the Planner RoleBinding in every namespace that does not have
one yet, and the DGDs there fail to reconcile their Planner RBAC, with or without live retune.

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
| Retune with load or throughput scaling, or of Planner parameters | A DGDR Run has no base Planner configuration, warm-up or predictor pre-sweep |
| Frontends with non-default routing fields beyond the three knobs, or `kv_event_lag_ms` other than 0 | The DGDR router section expresses only the three knobs, and the Run does not pass a lag |
| Workers that set their own router mode (`--router-mode` or `DYN_ROUTER_MODE`) | A worker's advertised router configuration replaces the frontend's for its workers, and the Run replays only the frontend's |
| Worker arguments that pass a ConfigMap, Secret, annotation or `envFrom` value where an option could stand, including an unquoted `$VAR` of one in a worker shell script or one that `eval` or a shell given `-c` parses again (P4 fails) | The Planner cannot read the value, so it cannot rule out a router mode the worker would advertise |
| Router flags that a worker produces through shell command substitution, backticks, script variables, positional parameters, literal text that `eval` or a nested shell parses again, or a script file, and ConfigMap, Secret, annotation or `envFrom` values in text that a shell or `xargs` reads from a heredoc, a here-string or a pipe, or that `su -c` runs (P4 does not see them) | The Planner reads the worker's spec and does not run its script |
| Raising `overlap_score_credit` on a frontend that starts it at 0: an omitted credit then searches only 0, and a written positive value fails P4 | A live router rejects a positive credit after starting at 0 |
| Private, fine-tuned or local-path models | The Run Job has no Hugging Face credentials or model mount |
| Several frontends without router replica sync | AISimulate simulates one router |
| EPP or Gateway API Inference Extension frontends, `frontendSidecar`, TLS frontends | No `/router_policy` there, different addressing, or certificates |
| Run Job resources, deadline and tolerations | Fixed by the DGDR Run Job |
| Time-based segment rotation and a segment count limit | The frontend trace writer rotates by size only |
| A separate or authenticated admin listener | Not available in the frontend yet |
