<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## KVCR deployment examples

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
- `envsubst`, `kubectl`, and Python 3 on the deployment host.

The manifests default to the `rdma/shared_ib` extended resource. The
process-local variant requests one share per GPU node; the memory-service
variant requests two shares per node because both `main` and `kvcr-services`
use RDMA. Set
`DYNAMO_RDMA_RESOURCE=rdma/ib` or another cluster-specific resource name when
required. `UCX_TLS=rc_x,cuda` prevents TCP payload fallback. Set
`DYNAMO_UCX_NET_DEVICES=all` for per-container single-HCA allocations, or pin
an HCA and port exposed in every engine and sidecar, such as `mlx5_0:1`.

The image tag must contain a Dynamo semantic release version. Pin the same
image by digest for every component, for example
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0@sha256:REPLACE_ME`. Until KVCR is
available in a Dynamo release image, build that image from matching Dynamo,
KVCR, and vLLM revisions. The memory-service variant requires KVCR's
`--health-port`, `--health-host`, `--health-dependency-url`, and
`--startup-timeout-seconds` flags, which are not all present in `v0.1.1`.
vLLM's KVCR tier must accept `kvcr_service_socket_path` and forward
`startup_timeout_seconds` to `KVCRConfig`. Record the three commit IDs and
image digest used for each deployment.

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

The memory-service variant requires the KVCR HTTP-health implementation in both
containers. To test it without rebuilding the existing compatible runtime image,
build a pure-Python wheel from the KVCR feature tree and export
`KVCR_WHEEL=/absolute/path/to/kvcr-0.1.1-py3-none-any.whl` before deployment.
The script mounts it through `PYTHONPATH` in both containers; it does not install
packages, update vLLM, or change image environment settings. Rebuild the image
for native dependency or vLLM changes. Without `KVCR_WHEEL`, the installed KVCR
must support the flags above. For pinned registration, ensure
`UCX_REG_NONBLOCK_MEM_TYPES` and `UCX_IB_ODP_PREFETCH` are unset in both containers.

`deploy.sh` applies an immutable, content-addressed ConfigMap containing
`services.sh`, Dynamo's shared `launch_utils.sh`, and the optional wheel before
applying the DGD. The bundle is limited to 1 MiB. It is retained after deployment
deletion because other deployments may share it; delete an old bundle only after
no Pods or deployment templates reference it. Do not replace a live bundle or
roll an incompatible KVCR wheel under engines that still map the old pools.

Run `./deploy.sh --render-only` to inspect the selected DGD. For the memory-service
variant, `./deploy.sh --render-configmap` separately renders its prerequisite
ConfigMap; apply that first when deploying rendered files manually. Both manifests
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
vLLM container restart leaves the service, Guard, and pool running.
`services.sh` starts KVCR and, on worker 0, the state agent. It uses Dynamo's
`wait_any_exit` and process-group cleanup helpers: any child exit ends the
wrapper and terminates/reaps its companion; SIGTERM does the same. Kubernetes'
Pod termination grace period (default: 30 seconds) is the final bound for stuck
children, not a timeout implemented by the shell helper. For a planned shutdown,
drain and stop engines before their service; the wrapper does not impose
cross-container shutdown ordering.

Memory-service settings accept positive decimal integers without leading zeros:

| Variable | Default | Controls |
|---|---:|---|
| `KVCR_POOL_SIZE_GIB` | 2 | Guard pool capacity per worker, in GiB |
| `KVCR_STARTUP_TIMEOUT_SECONDS` | 120 | HTTP `/ready` wait before launching vLLM |
| `KVCR_GUARD_STARTUP_TIMEOUT_SECONDS` | 30 | Guard preparation and activation, each with this budget |
| `KVCR_ENGINE_STARTUP_TIMEOUT_SECONDS` | 30 | Engine-side NIXL initialization |
| `KVCR_TERMINATION_GRACE_PERIOD_SECONDS` | 30 | Pod termination grace for drain and cleanup |

Timeouts are in seconds and independent. HTTP readiness is not inference readiness.
Shared memory is pool size + 1 GiB; the sidecar requests pool size + 1%
(rounded up to GiB), with a limit 1 GiB above that. Engine CPU offload stays 2 GiB.
For a 1 TiB pool, reserve 1035 GiB per sidecar (1036 GiB limit), for example:

```bash
KVCR_MEMORY_SERVICE_ENABLED=true KVCR_POOL_SIZE_GIB=1024 \
  KVCR_STARTUP_TIMEOUT_SECONDS=600 KVCR_GUARD_STARTUP_TIMEOUT_SECONDS=600 \
  KVCR_ENGINE_STARTUP_TIMEOUT_SECONDS=600 KVCR_TERMINATION_GRACE_PERIOD_SECONDS=180 \
  ./deploy.sh
```

The sidecar's HTTP readiness probe checks that the Unix accept
loop and Guard actor loops progress, plus worker 0's state-agent `/live` endpoint.
It does not require an attached engine or mutate a pool. No liveness/startup
restart probe is set on the sidecar: a failed probe must not restart it beneath
a live engine. Readiness does not itself fence engines or remove Dynamo routing
state. Kubernetes
container discovery gives `main` and the real `kvcr-services` sidecar separate
metadata writers. The explicit sidecar supplies its own downward-API Pod UID;
the operator injects that identity only into its generated `main` container.
Kubernetes currently restarts a failed `kvcr-services` container independently.
In this MVP, any such restart invalidates the Guard-recovery guarantee and is
deployment-fatal: a replacement service cannot adopt the original Guard and
pool while vLLM remains alive. An external deployment controller must replace
the entire affected worker Pod instead of restarting only the sidecar. If
worker 0's sidecar restarts, recreate the worker group or DGD because its state
agent's routing state was also lost. State-agent loss alone does not destroy KV
payloads, but current engines bind a discovered agent instance and do not
automatically reselect its replacement. Group recreation is the conservative
way to restore those bindings; this change does not add a recovery controller.

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
transfer. With `DYNAMO_UCX_NET_DEVICES=all`, each measured container must expose
exactly one active RDMA port; its counters are read independently.
