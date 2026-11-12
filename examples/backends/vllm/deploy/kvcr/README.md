<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCR deployment examples

These examples deploy two aggregated vLLM workers on separate GPU nodes and
use KVCR as a secondary KV-cache tier over RDMA. Both variants use one
KVCR-capable Dynamo vLLM runtime image for every process.

## Prerequisites

- Two Kubernetes nodes with one NVIDIA GPU each and the allocatable RDMA
  resource shares described below.
- The Dynamo Kubernetes Platform and `nvidia.com/v1beta1` DGD API.
- The Dynamo operator's Grove workload provider and the Grove PodCliqueSet
  API. The example uses the stable PodClique replica index to place one state
  agent for the two engines.
- An operator-configured etcd endpoint for the process-local variant.
- An `hf-token-secret` in the target namespace.
- A runtime image containing mutually compatible Dynamo, vLLM, KVCR, NIXL,
  and UCX builds.
- Linux 6.5 or a kernel with equivalent `SO_PEERPIDFD` support for the
  memory-service variant.
- `envsubst` and `kubectl` on the deployment host.

The manifests default to the `rdma/shared_ib` extended resource. The
process-local variant requests one share per GPU node; the memory-service
variant requests two shares per node because both `main` and `kvcr-services`
use RDMA. Set
`DYNAMO_RDMA_RESOURCE=rdma/ib` or another cluster-specific resource name when
required. `UCX_TLS=rc_x,cuda` prevents TCP fallback. Set
`DYNAMO_UCX_NET_DEVICES` to the GPU-local HCA and port exposed by the selected
RDMA resource, such as `mlx5_0:1`; do not copy that example device name without
checking the target nodes.

The image tag must contain a Dynamo semantic release version. Pin the same
image by digest for every component, for example
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0@sha256:REPLACE_ME`. Until KVCR is
available in a Dynamo release image, build that image from matching Dynamo,
KVCR, and vLLM revisions. Use [KVCR tag
`v0.1.0`](https://github.com/ai-dynamo/kvcr/tree/v0.1.0). Until
[vLLM PR 53624](https://github.com/vllm-project/vllm/pull/53624) is merged, pin
vLLM to the PR's immutable head SHA; at the time of this update it is
[`7015687c`](https://github.com/vllm-project/vllm/commit/7015687c390442fae384b78fc62e6621b3bf6e56).
After the PR merges, pin the commit that lands on vLLM `main`. Record the exact
Dynamo, KVCR, and vLLM commit IDs with the image digest used for every qualified
deployment.

`DYNAMO_KVCR_COMPATIBILITY_DIGEST` is an opaque layout version shared by the
engine and memory service. Change it whenever model, dtype, block layout, or
the KVCR integration changes.

## Deploy a variant

```bash
export DYNAMO_VLLM_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0@sha256:REPLACE_ME
export DYNAMO_UCX_NET_DEVICES=REPLACE_WITH_GPU_LOCAL_HCA:1
export DYNAMO_KVCR_COMPATIBILITY_DIGEST=qwen3-0.6b-example-v1
export NAMESPACE=REPLACE_ME

# In-process KVCR with process-local host memory.
KVCR_MEMORY_SERVICE_ENABLED=false ./deploy.sh

# KVCR memory service with a pool and Guard that survive an engine restart.
KVCR_MEMORY_SERVICE_ENABLED=true ./deploy.sh
```

Run `./deploy.sh --render-only` to inspect the selected DGD. Both manifests
use required Pod anti-affinity, so the second worker remains Pending unless a
second eligible node is available. After apply, the script also verifies that
the operator materialized Grove as the DGD's workload provider. Each engine's
`KVCR_CACHE_SLOT` is populated from its stable Grove replica index in this
example. The slot identifies a logical DP slot in the cache pool, not a Pod or
process. Its assignment must remain unchanged when an engine is restarted or
replaced and must be unique among DP ranks sharing the pool. The startup command
validates the slot against `KVCR_CACHE_SLOT_COUNT` before constructing the
cache-owner ID, so a replacement Pod reclaims the same state-agent slot.

## Understand the process lifecycle

In `agg.yaml`, the state agent and `dynamo.vllm` run under one supervisor in
worker 0's `main` container. Both workers use etcd discovery because Kubernetes
container discovery permits one metadata writer per actual container. If the
state agent or its local vLLM process exits, the supervisor terminates worker
0's container and Kubernetes restarts both processes. The operator supplies
the standard HTTP probes for the `main` container. Worker 1 runs only vLLM.
KVCR uses process-local host memory, so this variant does not preserve its KV
pool across a restart. Because live state-agent host reselection is not yet
supported, restart both workers to restore state tracking after worker 0
restarts.

In `agg-memory-service.yaml`, `dynamo.vllm` runs in `main`. A regular
`kvcr-services` sidecar runs the KVCR memory service and state agent. Both
containers mount `/run/kvcr` and the memory-backed `/dev/shm/kvcr` volume. A
vLLM container restart leaves the service, Guard, and pool running. The
sidecar's probes check the KVCR Unix socket. The state agent runs in worker 0's
sidecar, and that sidecar's probe also checks state-agent health. Kubernetes
container discovery gives `main` and the real `kvcr-services` sidecar separate
metadata writers. The explicit sidecar supplies its own downward-API Pod UID;
the operator injects that identity only into its generated `main` container.
Kubernetes currently restarts a failed `kvcr-services` container independently.
In this MVP, any such restart invalidates the Guard-recovery guarantee and is
deployment-fatal: a replacement service cannot adopt the original Guard and
pool while vLLM remains alive. An external deployment controller must replace
the entire affected worker Pod instead of restarting only the sidecar. If
worker 0's sidecar restarts, recreate the worker group or DGD because its state
agent's routing state was also lost.

The state agent carries routing and residency information; it does not move
KV payloads. KVCR uses NIXL and UCX for the remote payload transfer.

## Verify Guard recovery

Use the memory-service variant and the opt-in live-cluster test. The test adds
a fault gate to its temporary manifest; the deployment example itself contains
no test-only startup controls.

```bash
export DYNAMO_UCX_NET_DEVICES=REPLACE_WITH_GPU_LOCAL_HCA:1
export DYNAMO_KVCR_COMPATIBILITY_DIGEST=qwen3-0.6b-example-v1
python3 -m pytest tests/deploy/test_kvcr_guard.py \
  -m framework_with_kvcr \
  --image="$DYNAMO_VLLM_IMAGE" \
  --namespace="$NAMESPACE" --skip-service-restart -v -s
```

The namespace must be empty of an earlier deployment with the same name. The
test requires read-only `hostPath` access to InfiniBand counters and captures
every worker container's current and previous logs at the before-failure,
failed, remote-delivery, and recovered phases under `DYN_TEST_OUTPUT_PATH` (or
the standard `test_output` directory). It verifies that the workers are on
separate hosts, one source engine restarts while its KVCR sidecar remains up,
the Guard serves the preserved cache to the other engine, the response matches,
the KVCR transfer metrics increase, and the selected active HCA carries the
transfer.
