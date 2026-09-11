<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
- `envsubst` and `kubectl` on the deployment host.

The manifests default to the `rdma/shared_ib` extended resource. The
process-local variant requests one share per GPU node; the memory-service
variant requests two shares per node because both `main` and `kvcr-services`
use RDMA. Set
`DYNAMO_RDMA_RESOURCE=rdma/ib` or another cluster-specific resource name when
required. `UCX_TLS=rc_x,cuda` prevents TCP fallback. `UCX_PROTO_INFO=y` prints
the selected UCX transport without enabling debug logs. Set
`DYNAMO_UCX_NET_DEVICES` to the GPU-local HCA and port exposed by the selected
RDMA resource, such as `mlx5_0:1`; do not copy that example device name without
checking the target nodes.

The image tag must contain a Dynamo semantic release version. Pin the same
image by digest for every component, for example
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0@sha256:REPLACE_ME`. Until KVCR is
available in a Dynamo release image, build that image from matching Dynamo,
KVCR, and vLLM revisions; the vLLM integration is based on
[vLLM PR 53624](https://github.com/vllm-project/vllm/pull/53624) on the
`mv-kvcc/kvcc_repo` branch.

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
cache-owner ID uses its stable Grove replica index, so a replacement Pod
reclaims the same state-agent slot.

## Understand the process lifecycle

In `agg.yaml`, the state agent and `dynamo.vllm` run under one supervisor in
worker 0's `main` container. Both workers use etcd discovery because Kubernetes
container discovery permits one metadata writer per actual container. If the
state agent, its local vLLM process, or either health endpoint fails, Kubernetes
restarts worker 0's container and both processes. Worker 1 runs only vLLM.
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

The state agent carries routing and residency information; it does not move
KV payloads. KVCR uses NIXL and UCX for the remote payload transfer.

## Verify Guard recovery

Use the memory-service variant for this workflow:

Use separate shells to forward each worker's metrics port to a distinct local
port, for example `kubectl port-forward pod/$POD 19090:9090`.

1. Record `vllm:kvcr_transfer_blocks_total{operation="local_fill"}` and
   `vllm:kv_offload_tiering_write_bytes_total{tier="1:kvcr"}` on both workers,
   then send a long-prefix request with temperature zero. The source Pod is the
   one whose local-fill blocks and tier-write bytes increase. The other Pod's
   values must remain unchanged.
2. Terminate the `VLLM::EngineCore` process in that source Pod's `main`
   container:

   ```bash
   export SOURCE_POD=REPLACE_WITH_SOURCE_POD
   kubectl exec "$SOURCE_POD" -c kvcr-services -- \
     touch /run/kvcr/hold-engine-start
   kubectl exec "$SOURCE_POD" -c main -- \
     pkill -9 -f '[V]LLM::EngineCore'
   ```

   The marker keeps the replacement `main` container from starting vLLM; it
   does not stop `kvcr-services`.
3. Confirm `main` has one additional restart while `kvcr-services` remains
   Ready with an unchanged restart count, and confirm the marker exists in the
   replacement `main` container:

   ```bash
   kubectl exec "$SOURCE_POD" -c main -- \
     test -e /run/kvcr/hold-engine-start
   ```

4. Confirm the source sidecar logs contain `KVCR_EVENT guard_promoted`:

   ```bash
   kubectl logs "$SOURCE_POD" -c kvcr-services | \
     grep 'KVCR_EVENT guard_promoted'
   ```

5. Send the same prefix while the source engine is unavailable.
6. Confirm the surviving target's port 9090 metrics increased for:
   - `vllm:kvcr_transfer_blocks_total{operation="remote_deliver"}`
   - `vllm:kv_offload_tiering_read_bytes_total{tier="1:kvcr"}`
   - `vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}`
7. Compare the post-failure response with the pre-failure baseline for content
   correctness.
8. Confirm UCX protocol output names `rc_mlx5` or another configured RDMA
   transport on both workers.

Remove the fault-injection marker to restore the source engine:

```bash
kubectl exec "$SOURCE_POD" -c kvcr-services -- \
  rm -f /run/kvcr/hold-engine-start
```

The standard Kubernetes and Prometheus interfaces expose every required MVP
signal, so these examples do not add a separate Guard-checking utility.

The same workflow is automated by the opt-in live-cluster test:

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
the standard `test_output` directory).
