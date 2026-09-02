<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Planner autonomous batch E2E

- Date: 2026-08-28
- Result: passed
- Canonical run: `20260828T213549Z-planner-native-1e3ff8`
- Batch: `batch_f962abe4-dbf2-420e-b3c5-582ed2bccd9a`
- Namespace: dedicated POC namespace (deployment-specific)
- Planner image: local image built from this branch (registry details omitted)

## Result

The POC works end to end through the normal Planner tick. Starting with the
authoritative worker scaling adapter at zero and no schedulable serving path, a
new durable Gateway job caused Planner to publish a zero-RPS safety lease and
scale the owned DGDSA from zero to one. Grove/KAI then scheduled the frontend
and worker. Planner opened a 5-RPS lease only after both Kubernetes and Dynamo
reported the worker ready, llm-d Async drained all 100 requests, and a later
native tick returned the batch floor and authoritative Redis lease to zero.

No standalone control-loop process participated. The experiment driver made no
Kubernetes writes from evidence time `T0=2026-08-28T21:35:31Z` through
`T1=2026-08-28T21:40:16Z`.

## Control path

1. The lifecycle-owned batch collector reads durable Gateway job state on each
   native Planner tick. Frontend traffic and Async feedback are optional
   observations; an unavailable source remains missing and cannot erase a
   durable active job.
2. The pure batch policy calculates a replica floor and leased maximum batch
   admission rate. Missing capacity or readiness closes admission.
3. Planner applies the renewable Redis lease before replica effects. A lease
   write failure skips scaling, and a missing, malformed, mismatched, or expired
   lease is fail-closed in llm-d Async.
4. The batch floor is merged into Planner's normal final scale projection. It
   is not a plugin side channel or a second controller.
5. The Kubernetes connector reads the authoritative DGDSA. A zero-to-positive
   recovery on an unready DGD is allowed only after validating DGD generation,
   state, declared scaling-adapter ownership, exact references and UID, labels,
   deletion state, and the DGDSA resource version used as the optimistic scale
   precondition.

## Starting condition

One explicit DGDSA `1 -> 0` patch at `2026-08-28T21:28:07Z` established the
repeatable pre-run condition. It occurred more than seven minutes before T0 and
is visible in `dgdsa.before.json`; it was setup, not an in-run control action.

At T0:

- DGDSA desired/status replicas were `0/0`, generation `6`, resource version
  `807643579`;
- no worker pod existed;
- DGD generation and observed generation were both `17`;
- the DGD was Pending and its frontend pod was `SchedulingGated` because the
  Grove/KAI PodGroup still required the absent worker member;
- the Redis lease was fresh with cap `0`;
- the Planner pod was Ready with zero restarts.

This starting state exercised the intended partial-observation behavior: direct
frontend traffic telemetry was connection-refused, but the durable Gateway job
remained visible and actionable.

## Timeline

| UTC | Evidence |
|---|---|
| 21:36:14.724 | Gateway reported the new 100-request batch in validation. |
| 21:36:15.637 | Planner decided `replica_floor=1`, `max_admission_rps=0`. |
| 21:36:15.695 | Planner logged `Updating ... VllmDecodeWorker from 0 ... to 1`. |
| 21:36:15.707 | Planner logged the DGDSA scale to one; the watch observed generation `6 -> 7`, spec/status `0 -> 1`. |
| 21:38:11 | Observer first recorded DGD Ready, one ready worker replica, and one Ready worker pod; the lease remained zero. |
| 21:38:16.891 | Planner first published `replica_floor=1`, `max_admission_rps=5`. |
| 21:38:17.330 | Async's sampled dispatch counter first increased, after the positive lease. |
| 21:38:41.265 | Gateway reached `completed=100/100`, `failed=0`. |
| 21:38:42.162 | The next Planner tick published `replica_floor=0`, `max_admission_rps=0`. |

## Quantitative evidence

- Gateway terminal result: 100 total, 100 completed, zero failed.
- Retrieved result validation: 100 output lines, 100 unique custom IDs, zero
  error lines.
- Gateway progress duration: 146.539409 seconds.
- Average completion rate: 0.682410 RPS, including about two minutes of
  scale-up and model initialization.
- Peak poll-interval completion rate: 6.727552 RPS.
- Async dispatched-request counter: `400 -> 500`.
- Async successful-request counter: `400 -> 500`.
- Terminal Async backlog, in-flight requests, and in-process queue depth: all
  zero.
- Terminal Redis lease: API `llm-d.ai/v1alpha1`, pool `dynamo-batch`, cap `0`,
  with 57,644 ms TTL at capture.
- Compiled data-quality issues: zero.

The machine verifier passed all 15 assertions, including zero-to-one watch
evidence, ordered Planner decisions, closed admission until readiness, dispatch
after the positive lease, exact counter deltas, terminal result validation, and
a fresh terminal zero lease.

## Validation

- 478 relevant Planner/configuration/contract/runtime/connector/proto/harness
  tests passed using the real project dependencies.
- Affected llm-d Async unit packages and the build-tagged integration suite
  passed; targeted `go vet` passed.
- Planner lint, formatting, and Python compilation passed.
- Helm lint/render passed for restricted and cluster-wide Planner RBAC modes.
- The six-resource deployment passed Kubernetes server-side dry-run.
- The deployed Planner container passed an in-image import and policy smoke
  test before rollout.

## Scope and limitations

- The final worker remains at one replica. A batch floor is a lower bound, not
  a scale-down command; without a separate load-planner downscale opinion, this
  run proves autonomous `0 -> 1`, not autonomous `1 -> 0`. The important
  terminal control invariant is floor `0` and admission cap `0`.
- Async's exported drain-limit gauge records its last gate evaluation and can
  remain at `5` after the queue becomes idle. The renewable Redis lease is the
  authoritative state and was independently captured at zero. Consumers of the
  gauge must combine it with lease expiry.
- This run is batch-only, so the compiled report correctly marks online
  observations missing. A concurrent online-traffic treatment remains the next
  performance experiment.
- Grove retained a stale Planner pod template during rollout; the exact stale
  pod was deleted before T0 so the controller recreated it with the r2 image.
  This is a rollout issue, not part of the runtime control path.
- Planner logs, the continuous DGDSA watch, managed-field timestamps, and the
  absence of experiment-driver writes establish correlated attribution. The
  cluster's generic managed-field manager name is not an identity-grade audit
  record; Kubernetes audit logs would be required for that stronger claim.

## Artifacts and reproduction

- [Raw canonical run](../results/raw/20260828T213549Z-planner-native-1e3ff8/metadata.md)
- [Machine assertions](../results/raw/20260828T213549Z-planner-native-1e3ff8/autonomous-scale-evidence/assertions.json)
- [Continuous state stream](../results/raw/20260828T213549Z-planner-native-1e3ff8/autonomous-scale-evidence/state.jsonl)
- [Planner evidence-window log](../results/raw/20260828T213549Z-planner-native-1e3ff8/autonomous-scale-evidence/planner.log)
- [DGDSA watch](../results/raw/20260828T213549Z-planner-native-1e3ff8/autonomous-scale-evidence/dgdsa.watch.jsonstream)
- [Compiled summary](../results/compiled/20260828T213549Z-planner-native-1e3ff8-summary/README.md)

Re-run the machine checks from the experiment root:

```bash
python3 workloads/verify_native_planner_e2e.py \
  --run-dir results/raw/20260828T213549Z-planner-native-1e3ff8 \
  --evidence-dir \
    results/raw/20260828T213549Z-planner-native-1e3ff8/autonomous-scale-evidence
```
