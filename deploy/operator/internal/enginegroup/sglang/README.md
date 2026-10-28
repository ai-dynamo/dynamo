<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang Engine Group Elastic EP

One opted-in DGD component creates one Engine Group and one Grove PCSG world
containing a member PodClique. Scaling the child resizes that clique. Grove owns
Pod creation and placement; SGLang owns bootstrap, membership and internal admission.

## Native contract

The runtime image must include [SGLang #39834](https://github.com/sgl-project/sglang/pull/39834)
and the Dynamo worker integration in this branch. Released growth APIs without
runtime, operation and joining-allocation correlation are insufficient.

Healthy suffix retirement additionally requires [SGLang #33111](https://github.com/sgl-project/sglang/pull/33111)
composed with the same native runtime/operation correlation. The engine must expose
`retiree_lifecycle: external` and `safe_to_terminate_ranks`. An uncorrelated shrink
API is insufficient, even when its reported EP size reaches the requested count.

Every Pod runs `python3 -m dynamo.sglang` from one template using native
`--elastic-ep-join-mode auto`. Standard operator rendering supplies infrastructure
environment, discovery, service accounts, shared memory, security and image-pull
configuration. The DGD declares native Elastic EP arguments; the Engine Group
path adds no wrapper or command/argument rewriting. Portable topology aliases
provide the allocation index and rendezvous address:

```yaml
env:
- name: DYNAMO_ALLOCATION_UID
  valueFrom: {fieldRef: {fieldPath: metadata.uid}}
args:
- --elastic-ep-join-mode=auto
- --elastic-ep-allocation-index=$(DYNAMO_RANK)
- --elastic-ep-allocation-id=$(DYNAMO_ALLOCATION_UID)
- --dist-init-addr=$(DYNAMO_LEADER_ADDRESS):29500
# Model, initial/max EP size and other engine flags are also explicit.
```

SGLang derives all rank/process geometry and forms the complete initial world
without synthetic growth. There is no Dynamo bootstrap launcher or command payload.
`DYNAMO_RANK` is the allocation index, not every native member's DP rank. For a
four-member allocation, index 2 identifies the allocation hosting members 8–11.
`DYNAMO_LEADER_ADDRESS` resolves to allocation zero in this world. Grove-specific
names stay inside the renderer. These aliases precede user environment entries
and cannot be overridden, including through parent `spec.env`.
Parent `spec.env` entries use the standard component-over-parent merge.
The operator retains ownership, discovery and revision labels.
Engine Group main containers do not inherit generated inference probes; explicit
caller probes are preserved and should measure allocation/process health rather than require an inference
endpoint on every Pod: only the primary exposes that endpoint, and a temporary
inference pause should not accidentally trigger gang termination.

The fixed TP1/attention-CP1 profile supports one or several native members per Pod.
Visible GPUs determine the native allocation width, which status must confirm
against the resolved Kubernetes profile. Initial and maximum EP sizes must divide
into complete allocations. For example, two four-GPU Pods form EP8; one more
allocation grows it to EP12. Larger absolute targets converge one allocation at a time.
The initial physical EP size must exceed one for attention DP. A single four-GPU
allocation forming EP4 is valid; a single one-GPU allocation forming EP1 is not.

The primary exposes one stable Dynamo inference endpoint. SGLang's internal
round-robin scheduler selects committed ranks; a fixed startup-only router DP
range must not prevent appended ranks from receiving requests. Local KV/FPM rank
ranges remain allocation-specific. Rank-aware external routing and elastic KV
capacity advertisement are not established by this integration.

## Correlation and safety

`MembershipAdapter` sends the operation ID, expected runtime instance and exact joining
Pod UID for growth. Retirement carries no joining allocation. The adapter observes
before replay and retries only the identical request. A native
terminal success, matching operation/target/allocation IDs, and the effective size
are required to commit the result. A transport error or unsuccessful response with
pending work is not a definitive rejection.

Growth uses the native tokenizer's existing bounded scheduler acknowledgment and
queued execution, not a second asynchronous submission layer. Each request also
binds the original opaque topology token. Same-operation replay is recognized
before checking that old token as a new-request precondition, including when
submission is unknown. Acceptance rotates the token, preventing a delayed request
from becoming new work after a safe successor replaces its retained result.
Shrink requires its own native acceptance and completion qualification; growth
behavior alone does not establish that contract.

A native operation visible before its cached journal is inconclusive, not absent.
On an unexplained token/operation mismatch, a compare-and-swap check distinguishes
stale cached state from confirmed drift. It preserves the original token and
request; confirmed drift requires recovery even when the member count is unchanged.

The Kubernetes journal retains Dynamo control revision/digest correlation and the
topology-generation projection. Native engine status remains the transaction authority.
A changed runtime instance, allocation width or frozen Pod/process binding fails closed.
Each native process has its own Pod-UID-and-rank incarnation under the
non-restarting profile; a shared Pod UID is not a shared process identity.
This binding relies on SGLang not independently restarting ranks and on
`restartPolicy: Never`; container restarts require group recovery.
Current native runtime health is checked independently of retained operation results.
Unsupported recovery blocks reconciliation without rewriting a committed resize;
missing or unknown runtime health also fails closed.

`TrafficProjection` represents engine-internal admission, not independently observed
discovery registration. The verifier checks a real frontend completion; its proof
does not grant registration.

## Healthy shrink

Declare `--elastic-ep-retiree-lifecycle=external` and the
`dynamo.nvidia.com/retained` Pod readiness gate in the shared template. The engine
parks retirees after graceful drain instead of exiting the shared process tree.
The controller initializes the retention gate during capacity reconciliation,
without changing launch arguments or granting inference admission.

Each operation retires one complete highest-rank allocation. Native commit and
terminal drain for **every** retiring native member must precede Pod release.
Survivor serving verification is required before the transition completes.
Drain proof binds to the native operation, runtime,
and frozen process incarnations and survives controller restart and physical release.
An EP-size change alone is never drain proof.

The shared Grove capacity adapter marks only that retired Pod NotReady, scales
its member clique, and checks the released Pod UID. This is single-victim
readiness steering, not a Grove authorized-release API. Other unavailable Pods
or a different victim fail closed.

The complete initial world remains the lower bound. Native automatic bootstrap
cannot yet refill occupied rank slots: growth after shrink is rejected before
creating replacement capacity. Failure recovery, packed partial survival, slot
reuse, multi-Pod allocations, and whole-world retirement remain unsupported.

## Validation

CPU tests cover declarative geometry, native request/result forwarding, one- and
four-member allocations, ambiguous submission, exact replay after restart, native
runtime changes and real API-server Scale/Pod identity behavior. Retirement tests
prove that one safe member cannot release a multi-member Pod and retain exact
all-member drain evidence across restart.

GPU validation requires a newly built runtime image containing #39834. The
[`growth-test.yaml`](growth-test.yaml) manual fixture uses `${IMAGE}` for that image
and requires three B200 GPUs, working NIXL/Mooncake networking and the indicated
model PVC. Render the image variable before applying it in an isolated vcluster.
The [`packed-test.yaml`](packed-test.yaml) fixture uses two GPUs/members per Pod
and tests EP2 -> EP4 -> EP2 with four B200 GPUs. It requires both native PRs and
the correlated retirement contract above. The network-selection environment
is specific to the manual test image, not operator-injected launch behavior.
These CPU tests do not establish GPU serving, idle growth or failure recovery.

Packed GPU growth was validated separately with two GPUs/members per Pod:
one DGD completed EP2 -> EP4 with 24/24 recorded requests successful, preserving
the original Pod UID, native runtime, PodClique and shared template with no
container restarts. The image contained native #39834; this bounded functional
run does not establish an availability/performance SLO or failure recovery.

The full EP2 -> EP4 -> EP2 cycle passed with 32/32 recorded requests successful.
Both retiring members had correlated terminal drain evidence before release
of their exact Pod UID. This run used a local native composition of #39834 and
#33111 with shrink-correlation and warm-up integration fixes, not the two
unmodified PR heads. Those native changes remain an upstream prerequisite.
