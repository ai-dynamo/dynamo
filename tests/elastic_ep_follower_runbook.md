<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runbook — building and testing the on-demand elastic EP follower

This runbook hands off work in progress on operator support for on-demand elastic expert-parallel
followers. The design lives in
[`docs/design-docs/operator-elastic-ep-follower.md`](../docs/design-docs/operator-elastic-ep-follower.md);
which holds the full reasoning and is worth reading first. This document covers only what has been
done, what remains, and how to finish it: build two ARM64 images from the branches below, deploy
them to a GB200 cluster, and confirm the two code changes behave as intended against a real vLLM
engine.

## Table of contents

- [What the feature does](#what-the-feature-does)
- [What has already been established](#what-has-already-been-established)
- [The two changes under test](#the-two-changes-under-test)
- [Branch state](#branch-state)
- [Environment and prerequisites](#environment-and-prerequisites)
- [Part A — build and push the images](#part-a--build-and-push-the-images)
- [Part B — install a scoped test operator](#part-b--install-a-scoped-test-operator)
- [Part C — run the test](#part-c--run-the-test)
- [Pass criteria](#pass-criteria)
- [Known blockers](#known-blockers)
- [Leftovers to clean up](#leftovers-to-clean-up)

## What the feature does

A leader pod serves the model alone. When load rises, a follower pod is created, joins the leader's
Ray cluster, and lends its GPUs to the expert-parallel group. When load falls the reverse happens
and the GPUs go back to the cluster. The point is that nothing is reserved while the follower is
absent, which rules out the warm-standby topology used by the existing bare vLLM test, where an
idle worker pod holds GPUs from the start.

Attaching is two ordered steps, and most of the difficulty lives in the gap between them.
Kubernetes first makes the follower pod exist and the pod joins Ray, which makes its GPUs *visible*
but still idle. Only a subsequent `scale_elastic_ep` call on the leader makes vLLM create ranks on
them. Detaching runs the same two steps in reverse: shrink the engine first, then remove the pod so
its GPUs are released.

```mermaid
flowchart LR
    A["Leader serving alone<br/>follower at 0 replicas"] -->|"load rises:<br/>scale adapter to 1"| B["Follower pod created<br/>joins Ray"]
    B -->|"GPUs visible but idle"| C["POST /scale_elastic_ep<br/>larger dp"]
    C --> D["Follower GPUs serving"]
    D -->|"load falls:<br/>POST /scale_elastic_ep<br/>smaller dp"| E["Follower ranks released<br/>GPUs idle, pod still holds them"]
    E -->|"scale adapter to 0"| A
    B -.->|"no capacity:<br/>pod stays Pending"| A

    style A fill:#cfe8fc,stroke:#5b9bd5,color:#1f3b57
    style B fill:#fff4cc,stroke:#d6b656,color:#5c4813
    style C fill:#ede7f6,stroke:#9575cd,color:#3b2c57
    style D fill:#d8f3dc,stroke:#74c69d,color:#1b4332
    style E fill:#ffe5ec,stroke:#e08fa4,color:#5c2333
```

## What has already been established

Two cluster experiments settled the shape of the design; neither needs repeating, but their
conclusions explain why the test below looks the way it does.

The first asked whether a Grove `PodClique` can rest at zero replicas. It cannot be declared that
way: Grove's validating webhook rejects `minAvailable: 0`, its defaulting webhook rewrites
`replicas: 0` to `1`, and the clique list is immutable after creation so a follower clique cannot
be added later. Scaling an existing clique to zero afterwards does work and is durable.

The second experiment was more decisive. Deploying the same leader-plus-parked-follower
DynamoGraphDeployment on both pathways showed that **on Grove a component at zero replicas leaves
the leader `SchedulingGated` and the whole deployment `pending`**; scaling the follower to one
released the leader immediately, proving the zero-replica member was the cause. This reproduces
[grove#676](https://github.com/ai-dynamo/grove/issues/676). On the non-Grove pathway the same spec
worked out of the box: the leader ran immediately, the follower Deployment sat at `0/0`, and a full
`0 → 1 → 0` cycle left the leader's pod untouched.

Consequently **v1 runs on the non-Grove pathway**, selected per deployment with the annotation
`nvidia.com/enable-grove: "false"`. Grove remains the eventual target, pending upstream support for
zero-replica gang members.

The scaling mechanism needs no new code. Setting `scalingAdapter: {}` on a component makes the
operator create a `DynamoGraphDeploymentScalingAdapter` carrying a real `/scale` subresource that
targets that one component, drivable with plain `kubectl scale`.

## The two changes under test

Everything above was verified with busybox stand-ins, so **neither code change has yet run against
a real vLLM engine**. That is the entire purpose of this runbook.

The operator change in `deploy/operator/internal/dynamo/backend_vllm.go` makes a *single-pod*
elastic EP component start a Ray head. Previously only a multi-node leader received the Ray launch
wiring, so a solo leader came up with no Ray cluster for followers to join later. A single-pod
component is expanded as `RoleMain`, never `RoleLeader`, so the leader arm now matches both roles.

The engine change in `components/src/dynamo/vllm/handlers.py` replaces a hardcoded
`new_data_parallel_size < 2` rejection with the constraint vLLM actually enforces, namely that
`tensor_parallel_size * data_parallel_size` must exceed one. The old floor made `dp=1` impossible
even when tensor parallelism alone already provides several expert-parallel ranks, which is exactly
the state a fully drained follower leaves behind.

## Branch state

Three branches, each independent and cut from `main` at `36e8f4a4c0`. They touch disjoint files, so
they can be built, reviewed, and merged separately. Rebasing onto current `main` is advisable, as
`main` has moved on since they were cut.

| Branch | Contents |
|---|---|
| `tzulingk/elastic-ep-p0-grove-spike` | Design document and the Phase 0 Grove spike manifest with its recorded results. Documentation only. |
| `tzulingk/elastic-ep-scale-floor` | The `handlers.py` scale-floor fix plus `components/src/dynamo/vllm/tests/test_vllm_elastic_ep_handlers.py`. |
| `tzulingk/elastic-ep-p2-ray-head` | The `backend_vllm.go` Ray-head change plus cases added to `backend_vllm_test.go`. |

Unit tests on both code branches pass locally. The Go tests were confirmed to fail without the
production change and pass with it.

## Environment and prerequisites

Testing has been done on the Teleport context
`nv-prd-dgxc.teleport.sh-dynamo-aws-dev-01`, namespace `tzulingk-ft-tests`. Despite the name, this
cluster's GPU nodes are GB200: **arm64, four GPUs each**, all sixteen sharing one NVLink clique, so
MNNVL is available between any pair. Grove `v0.1.0-alpha.11`, kai-scheduler, and the Dynamo operator
are all installed cluster-wide. The namespace already has a large RWX `shared-model-cache` PVC and
an `nvcr-imagepullsecret`.

All images must be **arm64**. An Apple Silicon machine builds them natively.

Before starting, confirm the Teleport session is live (`tsh status`), Docker is running, and the
namespace has an `hf-token-secret` — it did not exist at handoff time and gated models need it.
Follow the credential steps in the
[image push guide](https://gitlab.com/tzulingk/wideep-ft/-/blob/main/fault_tolerance_tests/image/pull_push_image.md);
in short, export `NGC_API_KEY`, then:

```bash
export NAMESPACE=tzulingk-ft-tests
echo $NGC_API_KEY | docker login nvcr.io --username '$oauthtoken' --password-stdin
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<your-hf-token> -n $NAMESPACE
```

## Part A — build and push the images

Two images are needed and they differ enormously in cost.

**The operator image** is a Go binary and builds in minutes. From `deploy/operator` on the
`tzulingk/elastic-ep-p2-ray-head` branch:

```bash
export REGISTRY=nvcr.io/nvidian/dynamo-dev
export TAG=elastic-ep-follower
make docker-build-operator REGISTRY=$REGISTRY/ TAG=$TAG
make docker-push-operator  REGISTRY=$REGISTRY/ TAG=$TAG
```

Check the resulting image is `linux/arm64` before pushing; the Makefile does not pass `--platform`,
so it inherits the host architecture, which is correct on Apple Silicon and wrong on an x86 builder.

**The worker image** is the awkward one. A full `make docker-build-vllm` compiles CUDA and vLLM for
arm64 and takes hours, so prefer deriving from an existing arm64 runtime and overlaying only the
changed Python. Pick a recent internal arm64 vLLM runtime — the newest publicly pullable one is
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0`, but an internal tag closer to `main` is strongly
preferred, because overlaying a `main`-era `handlers.py` onto an older package may not fit. Verify
the guard being replaced actually exists in the base image before trusting the result:

```bash
docker run --rm --entrypoint sh <base-image> -c \
  'grep -n "new_data_parallel_size" $(python3 -c "import dynamo.vllm.handlers as h; print(h.__file__)")' | head
```

Then build the overlay from the `tzulingk/elastic-ep-scale-floor` branch, at the repository root:

```dockerfile
# Dockerfile.elastic-ep-overlay
ARG BASE
FROM ${BASE}
COPY components/src/dynamo/vllm/handlers.py \
     /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/handlers.py
```

The destination path must match what the `docker run` check above printed; it varies between image
builds. Then:

```bash
docker build --platform linux/arm64 --build-arg BASE=<base-image> \
  -t $REGISTRY/tzulingk-dynamo-vllm:$TAG -f Dockerfile.elastic-ep-overlay .
docker push $REGISTRY/tzulingk-dynamo-vllm:$TAG
```

If the overlay proves incompatible with the base, fall back to the full build rather than shipping a
half-patched image.

## Part B — install a scoped test operator

Do **not** upgrade the cluster-wide operator in `dynamo-system`; other people's deployments depend
on it. Install a second operator watching only `tzulingk-ft-tests`, using the operator image from
Part A, and confirm it is namespace-scoped before creating any DynamoGraphDeployment — two operators
reconciling the same object will fight.

Verify the scoped operator is running and the cluster-wide one is untouched, then continue.

## Part C — run the test

Write a DynamoGraphDeployment modelled on
[`tests/fault_tolerance/deploy/templates/vllm/moe_elastic_ep_demo.yaml`](fault_tolerance/deploy/templates/vllm/moe_elastic_ep_demo.yaml),
which already carries the correct vLLM flags and environment for elastic EP, but reshaped as
follows. Annotate the deployment with `nvidia.com/enable-grove: "false"`. Give it a leader component
at one replica requesting four GPUs, and a follower component at `replicas: 0` with
`scalingAdapter: {}` and an identical pod template. Set tensor-parallel size to four so one pod
holds exactly one data-parallel rank, keep `--data-parallel-backend ray`, `--enable-elastic-ep`,
`--enable-eplb` and `VLLM_ALL2ALL_BACKEND=allgather_reducescatter`, and point `HF_HOME` at the
`shared-model-cache` PVC. Use the worker image from Part A for both components.

Work through the sequence in the diagram above, checking each stage before moving on.

Start by confirming the **solo leader really starts a Ray head** — this is the operator change.
Inspect the rendered leader container's command: it should begin with `ray start --head` followed by
a readiness poll and then the vLLM launch. Without the fix the Ray head is absent entirely, which is
the failure this change exists to prevent.

```bash
kubectl get deploy -n $NAMESPACE <dgd>-leader \
  -o jsonpath='{.spec.template.spec.containers[0].args}' | head -c 400
kubectl exec -n $NAMESPACE deploy/<dgd>-leader -- ray status
```

Next confirm the **parked follower costs nothing**: the follower Deployment reports `0/0`, no pod
exists, and the leader serves inference normally.

Then **attach the follower** with `kubectl scale dgdsa -n $NAMESPACE <dgd>-follower --replicas=1`.
A pod should appear and, after waiting for the leader's `/health`, join Ray — `ray status` on the
leader should now show two active nodes. Expect this to take a while on first start; the existing
scripts allow generous timeouts because Python bytecode compilation can delay `ray start`
substantially. Confirm here that the follower's GPUs are **visible but idle** in `nvidia-smi`, which
is the gap the reconciler will eventually close automatically.

Now **grow the engine** by posting to the leader's scale endpoint, exactly as
[`run_bare_multinode_elastic_ep_scale_test.sh`](fault_tolerance/deploy/templates/vllm/run_bare_multinode_elastic_ep_scale_test.sh)
does, noting that under Dynamo the path is prefixed rather than bare. The follower's GPUs should now
show memory in use, and inference should still succeed.

Finally **run the sequence in reverse**: shrink the data-parallel size, confirm the follower's ranks
are gone and inference still works from the leader alone, then scale the adapter back to zero and
confirm the pod is deleted and its GPUs return to the cluster. Because vLLM fills the leader first
and spills over in order, shrinking removes exactly the follower's ranks, so the pod is safe to
delete once the engine is back to leader-only width.

One extra check exercises the second code change. With tensor-parallel size four, request
`new_data_parallel_size: 1`. The old hardcoded floor rejected this outright; it should now be
accepted, because four expert-parallel ranks remain. Requesting a size below one should still be
rejected.

## Pass criteria

The changes are correct if the solo leader starts a Ray head and serves on its own, the follower
consumes nothing while parked, an attached follower joins Ray and its GPUs are put to work by a
subsequent scale call, the reverse sequence returns those GPUs to the cluster, and the leader pod is
never restarted at any point — check its UID before and after, not just that a pod is running.
`dp=1` at tensor-parallel size four must be accepted.

If the leader's pod restarts during any transition, or the deployment leaves a ready state, stop and
record it: that would contradict what the busybox experiments showed and would matter more than any
other result here.

## Known blockers

**Capacity is the immediate one.** At handoff, `dynamo-aws-dev-01` had four free GPUs fragmented
across three nodes with no node wholly free, and `dynamo-aws-dev-gb200` had none at all. The shape
above needs two whole nodes, one for the leader and one for the follower. Either wait for capacity,
find another GB200 cluster, or fall back to a reduced shape with tensor-parallel size one and fewer
GPUs per pod, which still exercises the mechanism but no longer matches the one-pod-per-node premise
and cannot exercise the `dp=1` case.

Two smaller ones are worth knowing. No verification has been done that a non-Grove pod can join a
ComputeDomain, so if MNNVL turns out to be needed across the leader and follower, that is untested
ground. And the internal registries could not be reached from the handoff machine, so Part A's push
step is unproven end to end.

## Leftovers to clean up

Several throwaway objects from the experiments are still running in `tzulingk-ft-tests` and should
be removed. They are all busybox and consume no GPUs, but they clutter the namespace and one of them
still holds a follower pod at one replica.

```bash
kubectl delete pcs ep0-spike-c ep0-addclq -n tzulingk-ft-tests
kubectl delete dgd ep-cmp-nogrove ep-cmp-grove -n tzulingk-ft-tests
```

Deleting the two DynamoGraphDeployments removes their scaling adapters and Deployments as well.
