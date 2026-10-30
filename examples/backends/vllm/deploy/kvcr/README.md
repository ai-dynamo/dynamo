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
KVCR, and vLLM revisions. This example targets KVCR **0.2.0** at
[`7ce17c358b867bd89f40a4a42bee9c76839506d0`](https://github.com/ai-dynamo/kvcr/commit/7ce17c358b867bd89f40a4a42bee9c76839506d0)
and vLLM's 0.2.0 adapter at
[`fed7de909f894fbee0c45130376aed2e2c9bf0d3`](https://github.com/vllm-project/vllm/commit/fed7de909f894fbee0c45130376aed2e2c9bf0d3).
The vLLM adapter also needs the `startup_timeout_seconds` forwarding change:
accept that optional tier setting and pass it to `KVCRConfig`. It is not in the
vLLM commit above. KVCR requires NIXL 1.5.0 and the runtime's UCX, CUDA, and
`libatomic` libraries. Record the Dynamo revision, patched vLLM revision, KVCR
revision, and image digest for each deployment; these source pins alone do not
establish that an existing image contains a compatible stack.

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

To test KVCR changes without rebuilding an otherwise compatible runtime image,
build a pure-Python wheel from the pinned KVCR tree and export
`KVCR_WHEEL=/absolute/path/to/kvcr-0.2.0-py3-none-any.whl` before deployment.
The script mounts it through `PYTHONPATH` in both containers; it does not install
packages, update vLLM, or change image environment settings. Rebuild the image
for native dependency or vLLM changes. Without `KVCR_WHEEL`, the installed KVCR
must be compatible with this example. A wheel overlay does not upgrade an old
vLLM adapter to the 0.2.0 API. For pinned registration, ensure
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
`kvcr-services` sidecar runs the KVCR memory service (`kvcr.service.server`)
and, on worker 0 only, the state agent. Both containers mount `/run/kvcr` for
the Unix socket. Only the sidecar mounts the memory-backed `/dev/shm/kvcr`
volume. Guard passes the pool and control-listener file descriptors over the
socket using `SCM_RIGHTS`; the engine maps the same backing pages without
opening a pool path. No shared pool mount, `hostPath`, or permission-setting
init container is needed in the engine. Both containers must use the same UID
to access the service socket. A vLLM container restart leaves the service,
Guard, and pool running.

Both engines report residency to worker 0's state agent. Each Guard can serve
its preserved pool after its engine exits. The FD handoff
does not let a replacement service adopt a live engine's pool; the coupled
failure policy below still applies.

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

## Guard HugePages

KVCR 0.2.0 tries native HugeTLB backing first, then transparent huge pages
(THP), then base pages. HugeTLB allocation and THP advice are best-effort;
check Guard logs for fallback warnings rather than assuming huge pages were
obtained. See the [KVCR memory-service guide](https://github.com/ai-dynamo/kvcr/blob/7ce17c358b867bd89f40a4a42bee9c76839506d0/docs/dev-guide.md).

### Native hugetlbfs

Have the cluster administrator reserve huge pages on eligible nodes. Add a
`HugePages-2Mi` emptyDir mounted **only in `kvcr-services`**, with matching
`hugepages-2Mi` requests and limits. Kubernetes creates the mount; no `hostPath`,
privileged mount command, or permission-setting init container is needed.
See [Kubernetes HugePages configuration](https://kubernetes.io/docs/tasks/manage-hugepages/scheduling-hugepages/).

Merge these additions into the worker's `podTemplate.spec` in
`agg-memory-service.yaml`; retain its existing containers, resources, and volumes:

```yaml
containers:
- name: kvcr-services
  resources:
    requests:
      hugepages-2Mi: 3Gi
    limits:
      hugepages-2Mi: 3Gi
  volumeMounts:
  - name: kvcr-hugepages
    mountPath: /hugepages
volumes:
- name: kvcr-hugepages
  emptyDir:
    medium: HugePages-2Mi
```

This illustrates a 2 GiB pool with headroom: budget for the additional 100 MiB
journal per Guard, page-size rounding, and recovery snapshots. Adjust the
reservation for larger pools. For 1 GiB pages, change both resource keys to
`hugepages-1Gi` and the volume medium to `HugePages-1Gi`.

The mount path is arbitrary. KVCR discovers writable hugetlbfs mounts from
`/proc/self/mounts`, tries larger page sizes first, and creates its `kvcr`
subdirectory there; no mount-path CLI option is needed. For non-root containers,
use a suitable Pod `fsGroup` so Guard can write the volume, and retain the same
UID in Guard and vLLM for socket access. The engine needs neither this mount nor
a hugepage reservation for the already-backed pool received by FD.

Keep the existing `/dev/shm/kvcr` volume, `--pool-dir`, and ordinary memory
budget for runtime fallback. A Pod whose hugepage request cannot be scheduled
stays Pending; KVCR's fallback only applies after it starts. To skip native
HugeTLB, add `--disable-hugepages` to the service command in `services.sh`.
That flag does not disable THP.

### Transparent HugePages

No extra Kubernetes volume or `hugepages-*` resource is needed: retain the
default memory-backed emptyDir at `/dev/shm/kvcr`. On Linux with Python 3.12+,
KVCR attempts aligned mappings with `MADV_HUGEPAGE`; Guard also attempts
`MADV_COLLAPSE` on supported kernels (Linux 6.1+). THP remains part of ordinary
memory accounting, not the pre-reserved HugeTLB resource.

THP availability depends on the host kernel and its tmpfs/shmem policy. Have
the cluster administrator configure those at node level; do not change host
sysfs settings from the deployment. Advice or collapse can fail and leave base
pages in use. See the [Linux THP guide](https://www.kernel.org/doc/html/v6.7/admin-guide/mm/transhuge.html).

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
