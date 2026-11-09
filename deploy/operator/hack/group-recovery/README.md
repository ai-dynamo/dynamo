<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native LWS CPU reproducer (Part of #17951)

Characterization aid for native LeaderWorkerSet group replacement. A native
CPU cluster run on 2026-11-06 is recorded below. These files do not install a
controller, start the Dynamo operator, or decide that recovery succeeded.

The fixture is only `recovery-target` and `recovery-control`
(`leaderworkerset.x-k8s.io/v1`, size 2, `RecreateGroupOnPodRestart`). It is
not a DynamoGraphDeployment, DynamoComponentDeployment, DisaggregatedSet, or
Grove workload, and it cannot validate operator or cache restart.

## Validation record

The recorded 2026-11-06 experiment applies to public commit
`8d25e776bd511c9173e86f5632da42c6de978678`. Cloud source was the exact
archive, with no Git HEAD or commit object; the independently rebuilt Git
tree was `39e8cb2e380af2339a3ec8398ec1f11d7d5deb4b`, and all three
source-file hashes matched the frozen revision. This later docs edit does not
change or retest the fixture or collector. Details:
<https://github.com/ai-dynamo/dynamo/pull/18296>.

One ECS CPU host: 8 vCPU, 32 GiB RAM, 200 GiB ESSD PL0, Ubuntu 24.04 x86_64.
kind v0.32.0, 1 control-plane and 2 logical workers, Kubernetes v1.36.1,
Docker 29.1.3, a real LeaderWorkerSet v0.10.0 controller, matching CRDs, and
active webhooks. No Dynamo operator or GPU ran.

Five native cases. Every accepted case recovered with full target and control
membership and Ready, with a minimum observed healthy age over 15 seconds.
Confirmed failed incarnations / leader replacements:

| Case | Failed incarnations / leader replacements |
| --- | --- |
| Persistent leader | 7/7 |
| Persistent follower | 15/15 |
| Transient leader | 1/1 |
| Transient follower, valid repeat | 1/1 |
| Healthy follower, exact-UID deletion | 0/1 |

The committed collector produced 878 complete snapshots. Separate task-only
Pod watch windows had 664 + 65 = 729 records, plus selected StatefulSet owner
metadata. Those observers are not shipped here. The collector alone does
periodic GETs. Root UID and generation, actual role revision, and original
healthy-control UIDs stayed stable. Strict controller owner UID chains were
checked. Observed API Pod `restartCount` never exceeded 1 during accepted
windows despite replacements.

The first transient-follower attempt, without complete valid coverage, remains
inconclusive. The repeat was accepted separately. Transient actions used a
Ready=False cue before repair, not a claim of subsecond client exit or patch
order. About 0.2 seconds client-receive failure windows prove API visibility,
not Dynamo informer, cache, or reconcile capture. Healthy administrative
deletion also caused exit 137 on deleting Pods. UID changes or arbitrary
nonzero exits alone are not sufficient failed-recovery evidence.

This is manual native-provider characterization. It is not production
diagnosis acceptance or CI integration. DynamoComponentDeployment,
DynamoGraphDeployment, DisaggregatedSet, and Grove diagnosis, durable reset
and restart, isolation around 10,000 unrelated Pods, and inference remain
unimplemented and unvalidated by this experiment.

## Production LWS-DCD observations

The Dynamo operator separately observes Pods owned by a
DynamoComponentDeployment-backed LeaderWorkerSet during reconcile. That path
reads the manager cache (`readOrigin=manager_cache`) and does not use this
native fixture, DisaggregatedSet, or Grove.

At debug verbosity the operator emits one structured line per observation:

| Message | `verdict` |
| --- | --- |
| `Observed LeaderWorkerSet Pod failure evidence` | `CurrentFailure` |
| `Observed LeaderWorkerSet Pod without current failure evidence` | `NoCurrentFailureEvidence` |
| `LeaderWorkerSet Pod observation incomplete` | `Unknown` |

`CurrentFailure` requires a non-deleting Pod with Ready=False, a current
non-zero `main` exit, and verified creation / FinishedAt / Ready last
transition times, controller-owner chain, and template revision. Missing
creation, FinishedAt, Ready transition, or identity evidence is `Unknown`
with a fixed `reason`. FinishedAt and Ready last-transition times more than
five seconds ahead of reconcile time are also `Unknown`. When the cache
Get/List itself fails, the operator logs
`LeaderWorkerSet Pod observation unavailable` and leaves Available / Ready
unchanged.

Identity fields on the success path include `namespace`, `dcdName`,
`dcdUID`, `dcdGeneration`, `lwsName`, `lwsUID`, `lwsGeneration`,
`groupIndex`, `revision`, `leaderUID`, `podName`, `podUID`,
`podResourceVersion`, `container`, `exitCode`, `finishedAt`, `ready`,
`readyLastTransitionTime`, `reason`, and `readOrigin`. Repeated lines are
the current level fact for that reconcile, not a recovery counter or
durable history. The operator currently enables debug by default; the
manager binary accepts `--zap-log-level=debug` and `--zap-encoder=json`.

Production CPU results for this observation path remain pending. Do not
treat the native counts above as operator validation.

## Prerequisites

Use a real LeaderWorkerSet v0.10.0 controller, its validating and mutating
webhooks, and the matching v0.10 CRD. CRD-only envtest and ordinary operator
clustertest mocks do not run that controller. Installation stays outside this
directory. Record the controller image, served and storage CRD versions, and
the webhook names for `leaderworkerset.x-k8s.io` before interpreting a run.
The v0.10 spec has no restart-backoff or group-restart budget field. Do not
change those. Stop if the installed controller or CRD is not this v0.10 pair.

Each Pod uses `busybox:1.36.1` and reads `/gates/<role>` every loop. The mount
has no `subPath`. `healthy` stays up and writes `/tmp/local-ready`. `fail`
exits 23. Any other file text exits 25. A missing file exits 24. The probe
checks that local file and the gate. That is CPU process readiness, not
Dynamo inference, rank or collective readiness, or GPU recovery.

The collector's ConfigMap fields are API strings. They are not proof that the
mounted file has changed. Exit code and Ready are separate observations.
Kubelet sync can lag the API. A fixed sleep is not evidence.

Leader Pods use `worker-index=0`. Followers use `worker-index=1`.

## Setup

Run from the repository root. Use a new namespace.

```bash
CONTEXT="${CONTEXT:?set CONTEXT}"
NAMESPACE="${NAMESPACE:?set NAMESPACE}"
CONTROLLER_NAMESPACE="${CONTROLLER_NAMESPACE:?set CONTROLLER_NAMESPACE}"
CONTROLLER_DEPLOYMENT="${CONTROLLER_DEPLOYMENT:?set CONTROLLER_DEPLOYMENT}"
OUTPUT="${OUTPUT:?set OUTPUT to a new directory outside the git worktree}"
FIXTURE="deploy/operator/hack/group-recovery/native-lws.yaml"
```

`OUTPUT` must not exist. Record the controller, then create the namespace
only when the get reports NotFound:

```bash
kubectl --context "$CONTEXT" --namespace "$CONTROLLER_NAMESPACE" \
  --request-timeout=30s get deployment "$CONTROLLER_DEPLOYMENT" \
  -o jsonpath='{range .spec.template.spec.containers[*]}{.name}{" "}{.image}{"\n"}{end}'
kubectl --context "$CONTEXT" --request-timeout=30s \
  get crd leaderworkersets.leaderworkerset.x-k8s.io \
  -o jsonpath='{.metadata.name}{"\n"}{range .spec.versions[*]}{.name}{" served="}{.served}{" storage="}{.storage}{"\n"}{end}'
kubectl --context "$CONTEXT" --request-timeout=30s get namespace "$NAMESPACE"
kubectl --context "$CONTEXT" --request-timeout=30s create namespace "$NAMESPACE"
kubectl --context "$CONTEXT" --namespace "$NAMESPACE" --request-timeout=30s \
  apply -f "$FIXTURE"
```

Bounded cues, not proof of membership:

```bash
kubectl --context "$CONTEXT" --namespace "$NAMESPACE" --request-timeout=30s \
  wait --for=condition=Available --timeout=180s \
  leaderworkerset.leaderworkerset.x-k8s.io/recovery-target
kubectl --context "$CONTEXT" --namespace "$NAMESPACE" --request-timeout=30s \
  wait --for=condition=Available --timeout=180s \
  leaderworkerset.leaderworkerset.x-k8s.io/recovery-control
```

A label wait can return when the selector matches a subset, or when it matches
nothing. It does not prove that both Pods exist.

## Two terminals

Terminal A, for both arms. Do not background it or clean it up by PID.

```bash
python3 deploy/operator/hack/group-recovery/collect.py \
  --context "$CONTEXT" \
  --namespace "$NAMESPACE" \
  --output "$OUTPUT" \
  --duration 900 \
  --interval 5 \
  --command-timeout 15
```

Each GET uses `min(--command-timeout, time still inside --duration)`. The
process does not reserve time for the rest of the sample, so a 300-second
duration is not cut short by a six-request budget. Requests stop at the
deadline. One owned kubectl group adds at most 2 seconds of wait, plus local
writes, and that wait is not itself a completed collection. Exit 0 means the
duration ended after at least one observation. It is not validation.

Terminal B. Before **each** arm, read the latest complete snapshot and record:

- exactly two `recovery-target` Pods and exactly two `recovery-control` Pods
- Ready status True for all four
- each LeaderWorkerSet UID and generation
- every Pod UID, and that group's nonempty Pod
  `leaderworkerset.sigs.k8s.io/template-revision-hash` for each role

That record is the arm's healthy baseline. Repeat it after the leader arm
recovers, before the follower arm.

Leave the other target key and both control gates at `healthy`. Do not edit a
LeaderWorkerSet spec or Pod template. Run the leader pair for the leader arm.
After that arm has recovered, run the follower pair instead. The patch and
wait commands use those variables and do not set them again.

```bash
GATE_KEY=leader
WORKER_INDEX=0
```

```bash
GATE_KEY=follower
WORKER_INDEX=1
```

```bash
kubectl --context "$CONTEXT" --namespace "$NAMESPACE" --request-timeout=30s \
  patch configmap recovery-target-gates --type merge \
  -p "{\"data\":{\"${GATE_KEY}\":\"fail\"}}"
kubectl --context "$CONTEXT" --namespace "$NAMESPACE" --request-timeout=30s \
  wait --for=condition=Ready=False --timeout=180s \
  pod -l "leaderworkerset.sigs.k8s.io/name=recovery-target,leaderworkerset.sigs.k8s.io/worker-index=${WORKER_INDEX}"
```

The wait is a bounded cue. A timeout, an empty match, or a successful wait is
not the failure evidence. Leave `fail` in place long enough for samples to see
replacements, then restore the same key:

```bash
kubectl --context "$CONTEXT" --namespace "$NAMESPACE" --request-timeout=30s \
  patch configmap recovery-target-gates --type merge \
  -p "{\"data\":{\"${GATE_KEY}\":\"healthy\"}}"
kubectl --context "$CONTEXT" --namespace "$NAMESPACE" --request-timeout=30s \
  wait --for=condition=Ready --timeout=180s \
  pod -l 'leaderworkerset.sigs.k8s.io/name=recovery-target'
```

While the arm's gate is `fail`, repeated group-replacement history requires
all of the following in the snapshots. This history is not a production
recovery counter:

- `config_maps` shows that target gate as `fail` (the other target gate and
  both control gates stay `healthy`)
- exit code 23 on that role when a container state was captured
- samples in that window do not show the target group staying Ready
- target LeaderWorkerSet UID and generation stay at the baseline
- each target role's Pod `leaderworkerset.sigs.k8s.io/template-revision-hash`
  is nonempty and stays equal to that role's baseline value; a missing label
  leaves revision stability unproven and the arm inconclusive
- the target **leader** Pod UID (`worker-index` 0) moves baseline A to B and
  then to C: two replacement transitions. Two distinct UIDs prove one
  replacement, not two

Sampling does not prove a Pod stayed unready between polls. The API gate, the
container exit, and process Ready can disagree until the mount updates. Exit
24 or 25 is a different fault from `fail`.

After the same gate returns to `healthy`, a recovered arm shows both target
Pods present and Ready, the same target LWS UID and generation, the same
nonempty Pod `template-revision-hash` for each target and control role, and both control
Pods still at their baseline UIDs and Ready. Patch success, a Ready=False wait,
one replacement, an empty Pod list, a fixed sleep, or collector exit 0 is not
that result. If the snapshots do not show it, the arm is inconclusive.

The collector accepts only the exact API strings `healthy` and `fail`. Any
other gate stops the run with `invalid_gate` and does not store the text.

## Collector contract

Stdlib and `kubectl` argv only. Every call passes `--context`, `--namespace`,
and `--request-timeout`. One sample, in order, GETs `recovery-target`, its
Pods (`leaderworkerset.sigs.k8s.io/name=recovery-target`), `recovery-target-gates`,
then the same three for `recovery-control`. Pod lists are label-scoped. The
script does not list the namespace, watch, write, or read Secrets, environment,
or other configuration. Each kubectl is started in its own POSIX session, so
this collector is for macOS or Linux. On timeout or Ctrl-C, only that process
group is signalled. Ctrl-C keeps partial files, writes `result: interrupted`,
and exits 130. It does not write `collection.json`.

| Flag | Default | Bound |
| --- | --- | --- |
| `--context`, `--namespace`, `--output` | required | new output directory; DNS-1123 namespace |
| `--duration` | 300 | 1 to 1800 seconds, overall monotonic budget |
| `--interval` | 5 | 1 to 60 seconds, and no greater than the duration |
| `--command-timeout` | 15 | 1 to 60 seconds; clipped to the time still left |

| Exit | Meaning |
| --- | --- |
| 0 | at least one snapshot, then the duration ended |
| 1 | query, object, gate, JSON, timeout, cleanup, or file failure; no observations is also 1 |
| 2 | invalid arguments |
| 130 | interrupted |

`snapshot-NNNN.json` has `sample_started_at_utc` and `sample_finished_at_utc`
because the GETs are sequential. `complete` is false when the deadline stopped
the sample early; already projected objects stay in that file. `elapsed_seconds`
is monotonic time from process start. Omitted keys were absent. A returned
numeric 0 stays 0.

LeaderWorkerSet fields: identity (`name`, `namespace`, `uid`, `resourceVersion`,
plus `generation`, `creationTimestamp`, and `deletionTimestamp` when present),
owner `apiVersion`/`kind`/`name`/`uid`, and these labels when present:
`name`, `group-index`, `group-key`, `worker-index`, `template-revision-hash`
under `leaderworkerset.sigs.k8s.io`. Spec paths when present: `replicas`,
`size`, `restartPolicy`, `startupPolicy`, rollout `type`, `partition`,
`maxUnavailable`, `maxSurge`, subgroup type and size, `subdomainPolicy`.
Status when present: `observedGeneration`, replica counts, and condition
`type`, `status`, `reason`, `lastTransitionTime`, `observedGeneration`.

Pod fields: the same identity and labels; `nodeName` from **spec**; `phase`;
conditions `Ready`, `PodScheduled`, `ContainersReady`, and `Initialized` when
present; container and init-container `name`, `image`, `imageID`, `ready`,
`restartCount`, and current/last `phase` with `reason`, `exitCode`,
`startedAt`, and `finishedAt` when that phase has them. An empty state object
is omitted. Exit code 0 is kept when the API sent 0. Pod JSON may be `kind: List`
or `PodList`; each item must be a Pod in this namespace with the fixture name
label.

ConfigMap fields: `name`, `namespace`, `uid`, `resourceVersion`, `leader`,
`follower`.

No annotations, Pod spec body, commands, args, environment, or
`last-applied-configuration`. v0.10 has no durable repeated-recovery annotation,
so none is stored. The script does not count UIDs or print a pass.

`collection.json` is written only after a completed window. `started_at_utc`
is the clock at collection start. `elapsed_seconds` is actual monotonic time
and can pass the requested duration by the final local write. `reason` is
`duration_elapsed`. `incomplete_sample_index` is the last snapshot when the
deadline cut it, otherwise null.

`error.json` has `result` `failed` or `interrupted`, the same start time,
`error.category`, `error.operation`, and `error.returncode` when kubectl
returned one. Categories include `kubectl_failed`, `kubectl_not_found`,
`command_timeout`, `invalid_json`, `invalid_response`, `invalid_gate`,
`no_observations`, `cleanup_failed`, and `file_error`. Stdout and stderr are
not saved. A failed reap is `cleanup_failed` / `reap/kubectl` and does not
become `duration_elapsed`. On Ctrl-C, that same reap failure is
`cleanup_error` beside `result: interrupted`, and the exit stays 130.

Stopping one kubectl sends SIGKILL to the process group from that
`start_new_session` Popen, closes the collector's pipe ends, and waits at most
2 seconds. That wait, plus local writes, is the allowance beyond the collection
duration. A process that leaves the session is outside this owned group. No
other process group is signalled. Timeout and Ctrl-C cleanup have not been
executed.

Keep `$OUTPUT` outside the git worktree. It has names, UIDs, nodes, conditions,
exits, image IDs, and the two gate words.

## Cleanup

If a delete exceeds its timeout, record what remains and stop.

```bash
kubectl --context "$CONTEXT" --namespace "$NAMESPACE" --request-timeout=60s \
  delete -f "$FIXTURE" --wait=true --timeout=180s
kubectl --context "$CONTEXT" --request-timeout=60s \
  delete namespace "$NAMESPACE" --wait=true --timeout=180s
```

## Deferred

Not this fixture: DynamoComponentDeployment, DisaggregatedSet, and Grove paths;
durable recovery accounting and reset semantics; manager and cache
reconstruction; performance around 10,000 Pods. No operator restart was
validated. The native LeaderWorkerSets are not owned by the Dynamo operator.
