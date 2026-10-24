<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang Engine Group growth

One opted-in DGD component creates one Engine Group and one Grove PCSG world
containing a member PodClique. Scaling the child grows that clique. Grove owns
Pod creation and placement; SGLang owns bootstrap, membership and internal admission.

## Native contract

The runtime image must include [SGLang #39834](https://github.com/sgl-project/sglang/pull/39834)
and the Dynamo worker integration in this branch. Released growth APIs without
runtime, operation and joining-allocation correlation are insufficient.

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

`GrowthAdapter` sends the operation ID, expected runtime instance and exact joining
Pod UID. It observes before replay and retries only the identical request. A native
terminal success, matching operation/target/allocation IDs, and the effective size
are required to commit the result. A transport error or unsuccessful response with
pending work is not a definitive rejection.

The Kubernetes journal retains Dynamo control revision/digest correlation and the
topology-generation projection. Native engine status remains the transaction authority.
A changed runtime instance, allocation width or frozen Pod/process binding fails closed.
Container restarts are prohibited by this profile.
Current native runtime health is checked independently of retained operation results.
Unsupported recovery blocks reconciliation without rewriting a committed resize;
missing or unknown runtime health also fails closed.

`TrafficProjection` represents engine-internal admission, not independently observed
discovery registration or drain. The verifier checks a real frontend completion;
its proof does not grant registration. Healthy growth does not establish shrink,
failure recovery, slot reuse, multi-Pod allocations or whole-world retirement.

## Validation

CPU tests cover declarative geometry, native request/result forwarding, one- and
four-member allocations, ambiguous submission, exact replay after restart, native
runtime changes and real API-server Scale/Pod identity behavior.

GPU validation requires a newly built runtime image containing #39834. The
[`growth-test.yaml`](growth-test.yaml) manual fixture uses `${IMAGE}` for that image
and requires three B200 GPUs, working NIXL/Mooncake networking and the indicated
model PVC. Render the image variable before applying it in an isolated vcluster.
These CPU tests do not establish GPU serving, idle growth or failure recovery.
