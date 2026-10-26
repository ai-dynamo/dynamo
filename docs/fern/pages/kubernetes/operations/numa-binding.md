---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: NUMA Binding
subtitle: Place worker CPUs, host memory, and KV-transfer NICs near their GPUs
---

Non-uniform memory access (NUMA) binding keeps NVIDIA Dynamo workers' CPU execution and host memory
close to their GPUs. It applies to H100, H200, B200, and B300 servers as well as Grace-based platforms.
Bind the frontend separately.

For the DGD examples, edit an existing [DynamoGraphDeployment (DGD)](../model-deployment/deploy-with-dgd.md).
Keep its images, model arguments, discovery, networking, resources, and probes. Merge the shown
arguments and environment variables into the existing `main` container; these are partial manifests.
For local processes, use the [CLI NUMA Binding guide](../../cli/operations/numa-binding.md).

## Separate Allocation From Binding

NUMA placement happens at three stages:

| Stage | Responsibility | Limit |
|---|---|---|
| Scheduler or resource allocator | Choose a host and, where supported, devices meeting workload constraints. | Node totals can hide NUMA fragmentation. |
| Kubelet resource managers | Admit the Pod and allocate CPUs, memory, and devices on the node. | Alignment depends on node policy and providers' topology information. |
| Application launcher and engine | Bind frontend and GPU worker processes within their allocations. | Binding cannot use CPUs, memory nodes, or devices outside the allocation. |

Current DGD/Grove constraints do not provide general intra-node NUMA placement for prefill/decode
groups. The API's `numa` domain name reserves the concept; `packDomain: numa` does not implement
intra-node device allocation. See [Dynamo issue #10171](https://github.com/ai-dynamo/dynamo/issues/10171)
and [Grove's NUMA placement work](https://github.com/ai-dynamo/grove/issues/644).

The kubelet settings below are an administrator-managed workaround for individual workers. They
cannot select workload-specific NUMA IDs, spread prefill replicas across PCIe roots, or colocate
prefill/decode groups. The launcher still owns CPU affinity and host-memory binding after allocation.

Inspect both GPU-to-GPU and GPU-to-NIC paths when diagnosing KV transfers. Devices on the same NUMA
node can still use different PCIe roots.

## Make NUMA Resources Available

Allocate resources before binding processes. Host topology in sysfs can include CPUs and memory
nodes excluded by the container's cgroup. `UCX_NET_DEVICES` likewise selects exposed devices;
it cannot allocate an RDMA device or create a network interface.

### Configure Node Resource Managers

Use a dedicated node pool sized for the workers. Ask its administrator for the NUMA IDs, socket
mapping, CPUs and memory available after reservations, GPU UUIDs and PCI addresses, and RDMA NIC
ports. GPUs may attach to only some NUMA domains. Keep socket IDs, NUMA IDs, and GPU indices separate.

This **kubelet configuration** suits a worker container that fits in one NUMA domain. It combines
exclusive CPU allocation, host-memory placement, and topology admission, using `container` scope because
the main container owns the engine's critical resources:

```yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cpuManagerPolicy: static
memoryManagerPolicy: Static
topologyManagerPolicy: single-numa-node
topologyManagerScope: container
kubeReserved:
  cpu: "1"
  memory: 1Gi
systemReserved:
  cpu: "1"
  memory: 1Gi
evictionHard:
  memory.available: 1Gi
  nodefs.available: "10%"
  nodefs.inodesFree: "5%"
  imagefs.available: "15%"
reservedMemory:
  - numaNode: 0
    limits:
      memory: 2Gi
  - numaNode: 1
    limits:
      memory: 1Gi
```

Apply this through the node pool's kubelet configuration mechanism. It is not a DGD or a resource
for `kubectl apply`. Keep the site's other kubelet settings and size reservations for the host.
For CPU NUMA nodes `0` and `1`, the example reserves `3Gi`, equal to
`kubeReserved.memory + systemReserved.memory + evictionHard["memory.available"]`. Recalculate this
sum after changing reservations or thresholds. CPU Manager's `static` policy also needs a nonzero
CPU reservation. Follow the administrator's drain, kubelet-state migration, and restart procedure
for policy changes; a Pod edit cannot change node policy.

`single-numa-node` admits a container only when its participating resources fit in one NUMA domain.
GPU and NIC plugins must report NUMA information to participate in alignment. Use
`topologyManagerScope: pod` when coupled containers need a common allocation. It counts the Pod's
effective combined demand, including init containers and sidecars, so it can reject a Pod whose
main container fits alone. Neither scope selects a numeric node such as `1`.

CPU Manager's `static` policy assigns logical CPUs. For physical-core isolation, enable
`cpuManagerPolicyOptions.full-pcpus-only: "true"` on a supported version. Exclusive CPU requests must
then be multiples of the host's simultaneous multithreading (SMT) width. Odd requests fail on a
two-thread-per-core host. Audit containers receiving exclusive CPUs to avoid `SMTAlignmentError`.

For tensor parallel (TP) groups spanning NUMA domains, use a multi-domain policy such as `restricted`
and keep per-rank engine binding. `restricted` requires preferred alignment but accepts multiple domains;
`best-effort` also allows unaligned placement. `single-numa-node` can reject Pods despite sufficient
aggregate CPU, memory, and GPUs.

CPU Manager can pack CPUs into fewer domains than the GPU group uses. Where supported, its
`distribute-cpus-across-numa` option spreads CPU assignments. Check that the resulting cgroup includes
CPU and memory capacity in every required domain before binding ranks.

See Kubernetes [CPU Manager](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/),
[Memory Manager](https://kubernetes.io/docs/tasks/administer-cluster/memory-manager/),
[Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/), and
[device-plugin topology integration](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/#device-plugin-integration-with-the-topology-manager).

### Account for Scheduling and NUMA Fragmentation

The default scheduler checks node totals before kubelet checks NUMA alignment. A domain with eight
free CPUs and no GPU, plus another with two free CPUs and a GPU, cannot satisfy a four-CPU/one-GPU
`single-numa-node` request. Host memory must also be available in the selected CPU/GPU domain.

Kubelet rejects that Pod with `TopologyAffinityError`. The workload controller must create a
replacement; the failed Pod is not rescheduled. Replacements can fail for the same reason. Reduce
requests only if the model's parallelism and workload allow it; otherwise choose a multi-domain
policy or another node pool. See [kubelet admission behavior](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/#single-numa-node-policy).

Administrators can reduce these failures with a scheduler profile using `NodeResourceTopologyMatch`,
its `NodeResourceTopology` CRD, topology exporter, and resource cache. This is an out-of-tree plugin
from Kubernetes SIG Scheduling's `scheduler-plugins` project; the default Kubernetes scheduler
does not include it.

If your platform has installed a scheduler with this plugin, set `podTemplate.spec.schedulerName`
to that scheduler profile's configured `profiles[].schedulerName`. There is no built-in scheduler
named `numa-aware-scheduler`. With Grove or another gang scheduler, check compatibility with its
backend before changing the profile.
Stale topology data or caches can still cause admission failures. This profile cannot guarantee
specific NUMA IDs or coordinated prefill/decode placement; kubelet decides admission. See the
[scheduler plugin configuration and cache](https://github.com/kubernetes-sigs/scheduler-plugins/blob/master/pkg/noderesourcetopology/README.md).

For more on worker sizing, CPU isolation, and admission failures, see
[Keeping GPU Workloads NUMA-Local in Kubernetes](https://ronaknathani.com/blog/2026/05/keeping-gpu-workloads-numa-local-in-kubernetes/).

### Request CPUs, Memory, and Devices in the DGD

Exclusive CPUs under `static` require a whole-number CPU request in a `Guaranteed` Pod. Set equal,
positive CPU and memory requests and limits on **every application and init container**, including
injected sidecars and model loaders. Check the rendered Pod's QoS class after injection and admission.

This worker requests eight logical CPUs, `64Gi` of host memory, one GPU, and one shared slot from a
cluster-defined RDMA pool named `rdma/numa0`:

```yaml
spec:
  components:
    - name: decode
      podTemplate:
        spec:
          containers:
            - name: main
              resources:
                requests:
                  cpu: "8"
                  memory: 64Gi
                  nvidia.com/gpu: 1
                  rdma/numa0: 1
                limits:
                  cpu: "8"
                  memory: 64Gi
                  nvidia.com/gpu: 1
                  rdma/numa0: 1
```

Size requests for the workload and local capacity. Replace `rdma/numa0` with the plugin's advertised
resource name. Frontends need CPU and memory requests; include device resources only if their
network attachment needs them. Fractional CPU requests can be `Guaranteed` but receive shared CPUs
under the `static` policy.

GPU access needs the NVIDIA device plugin or a supported resource driver, plus container-runtime
integration. RDMA access needs its plugin or driver and a device allocation. Check node resources:

```bash
kubectl get node "$NODE" -o jsonpath='{.status.allocatable}'
kubectl describe node "$NODE"
```

Set `NODE` to the intended GPU node. Allocatable resources show node-wide capacity, not free capacity
per NUMA domain. Check the admitted allocation before choosing domain IDs. `CUDA_VISIBLE_DEVICES`
filters devices within an allocation; it cannot change allocated devices or enlarge cgroup masks.

### Expose the Intended RDMA NICs and Network Interfaces

Follow [RDMA Setup](../installation/rdma-setup/overview.md) for node drivers, GPU RDMA support, and
device plugins. An extended-resource request exposes allocated verbs devices such as
`/dev/infiniband/uverbs*`. The matching network interface and IP may still be outside the Pod's
network namespace.

A shared-device resource unit can be a slot in an HCA pool. It need not represent an exclusive NIC.
Split pools spanning NUMA domains into domain-local resources. For example, in the RDMA Shared Device
Plugin's `config.json`:

```json
{
  "configList": [
    {
      "resourcePrefix": "rdma",
      "resourceName": "numa0",
      "rdmaHcaMax": 63,
      "selectors": {"ifNames": ["ib0", "ib1"]}
    },
    {
      "resourcePrefix": "rdma",
      "resourceName": "numa1",
      "rdmaHcaMax": 63,
      "selectors": {"ifNames": ["ib2", "ib3"]}
    }
  ]
}
```

Replace `ib*` with netdev names from `ibdev2netdev` after checking sysfs locality. UCX uses different
names: `mlx5_*:port`. `rdmaHcaMax` counts shared slots, not physical HCAs. Load the config through the
plugin's mounted ConfigMap or Network Operator's `NicClusterPolicy.spec.rdmaSharedDevicePlugin.config`.
Keep each pool name mapped to the same NUMA domain across eligible nodes.

Request `rdma/numa0` or `rdma/numa1`, then select devices from that pool in `UCX_NET_DEVICES`.
Shared plugin version `1.5.3` supplies no NUMA hints. Splitting pools controls NIC availability;
aligning GPUs, CPUs, and memory still requires topology-aware allocation and verification. See the
[RDMA Shared Device Plugin configuration](https://github.com/Mellanox/k8s-rdma-shared-dev-plugin/tree/v1.5.3#rdma-shared-device-plugin-configurations)
and its [allocation implementation](https://github.com/Mellanox/k8s-rdma-shared-dev-plugin/blob/v1.5.3/pkg/resources/server.go).

Expose network interfaces through one of the platform's supported arrangements:

- **Host networking:** add these Pod settings for deployments using host NICs and addresses. Keep
  the RDMA device request and avoid host-port conflicts between replicas on the same node.

  ```yaml
  podTemplate:
    spec:
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
  ```

- **Secondary network:** use the network operator and Multus/SR-IOV attachment to allocate a NIC
  or virtual function with an interface and address in the Pod. Request its associated resource.
  Check the network-status annotation and interface names inside the container.

Verify both verbs-device access and the TCP interface before setting `UCX_NET_DEVICES`. Host
networking provides interfaces; device allocation provides device access. For examples, see the
[NVIDIA Network Operator deployment guide](https://docs.nvidia.com/networking/display/kubernetes25100/deployment-guide-kubernetes.html).

## Inspect the Pod Topology

Worker images need `numactl`, `libnuma`, GPU topology access, and readable NUMA sysfs entries on Linux.
Set `POD` to a running worker Pod and inspect its topology:

```bash
kubectl exec "$POD" -c main -- nvidia-smi topo -m
kubectl exec "$POD" -c main -- numactl --hardware
kubectl exec "$POD" -c main -- lscpu --extended=CPU,NODE,SOCKET,CORE
kubectl exec "$POD" -c main -- awk '/Cpus_allowed_list|Mems_allowed_list/' /proc/self/status
```

Use the CPU Affinity and NUMA Affinity columns to find each GPU's nearby CPUs and host memory.
Check every node from inside the container, where CPU sets, memory nodes, and visible GPUs can
differ from the host. GPU NUMA ID describes a different property and may be unavailable. See the
[NVIDIA topology reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html#topology).

> [!IMPORTANT]
> `nvidia.com/gpu` requests do not guarantee CPU, host-memory, and GPU alignment. Use the platform's
> topology and device-allocation policies, then check the Pod. A `nodeSelector` selects only the host.

Binding lists follow visible-device order. The four-GPU examples assume GPUs 0–1 on NUMA node 0 and
GPUs 2–3 on node 1. Replace the lists from your topology; GPU model and count do not determine it.

## Choose a Host-Memory Policy

| Policy | Behavior |
|---|---|
| `--cpunodebind=N` | Use allowed CPUs in node `N`, preserving memory policy. |
| `--cpunodebind=N --membind=N` | Bind CPUs and new allocations governed by the policy to node `N`. Allocation can fail when local memory runs out, even with free memory elsewhere. |
| `--cpunodebind=N --preferred=N` | Bind CPUs to node `N`; prefer its memory, with fallback to other allowed nodes. |

Use `--preferred` when local memory capacity varies. These policies leave existing and shared pages
in place and do not control ordinary CUDA device allocations. For hardware-coherent systems, see NVIDIA's
[NUMA and coherent memory guidance](https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms/).

Test the policy inside the Pod before changing its launch command:

```bash
kubectl exec "$POD" -c main -- numactl --cpunodebind=0 --preferred=0 true
kubectl exec "$POD" -c main -- numactl --cpunodebind=0 --membind=0 true
```

Choose a domain in the container's allowed CPU and memory sets. If a probe fails, check the error and
runtime policy. Request `SYS_NICE` only where required and permitted; it cannot expand the allocation.
Check effective memory placement after startup, since backend fallback can leave only CPU binding.

### Allow NUMA Policy and Memory Registration Operations

For `Operation not permitted`, check capabilities and seccomp. Some runtime profiles allow NUMA
memory-policy calls with `SYS_NICE`. Add it to `main` only if required and allowed by cluster policy:

```yaml
securityContext:
  capabilities:
    add:
      - SYS_NICE
```

A custom seccomp profile must allow `get_mempolicy`, `set_mempolicy`, and `mbind`; capabilities
cannot override explicit denials. Use an approved profile and repeat the probe. Privileged mode or
host-device mounts cannot supply missing resource allocations. See the
[runtime seccomp documentation](https://docs.docker.com/engine/security/seccomp/#significant-syscalls-blocked-by-the-default-profile)
for blocked NUMA calls, and check the profile your cluster actually uses.

RDMA registration also needs sufficient locked memory. Check
`kubectl exec "$POD" -c main -- sh -c 'ulimit -l'`. Set memlock through the runtime or add `IPC_LOCK`
where required; `SYS_NICE` does not cover it. See the
[RDMA resource and memlock setup](../installation/rdma-setup/infiniband-on-azure.mdx#pod-resource-requests)
for an example; requirements vary by fabric and runtime.

> [!NOTE]
> `Static` Memory Manager with `single-numa-node` can limit `cpuset.mems` to one domain.
> `--preferred=N` falls back only within the **allowed** memory nodes. A single-domain allocation
> therefore has no remote-domain fallback.

## Configure vLLM Workers

Dynamo forwards vLLM arguments and uses the `spawn` method needed by NUMA wrappers. Append these
entries to each prefill or decode component:

```yaml
spec:
  components:
    - name: decode
      podTemplate:
        spec:
          containers:
            - name: main
              args:
                - --numa-bind
                - --numa-bind-nodes
                - "0"
                - "0"
                - "1"
                - "1"
```

`--numa-bind` defaults to disabled. Omit `--numa-bind-nodes` for automatic discovery. To assign CPU
ranges, add `--numa-bind-cpus` with one list per visible GPU, such as `0-3 4-7 48-51 52-55`. Both
lists must follow visible GPU order and fit the allocation.

vLLM wraps GPU worker subprocesses in `numactl` before Python starts and binds EngineCore to its
shard's domains. Frontend and coordinator placement is separate. The normal policy is `--membind`;
failed probes can leave CPU-only or no binding. Existing CPU affinity can block automatic discovery,
so check the allocation and supply explicit nodes if needed. See [upstream vLLM NUMA configuration](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/utils/numa_utils.py).

> [!WARNING]
> For ranks spanning NUMA domains, use engine binding. A single-node `numactl` wrapper around the
> entire launch can prevent per-rank placement. Use such a wrapper only for a single-domain launch.

## Configure SGLang Workers

SGLang 0.5.21 and 0.5.22 enable automatic NUMA selection and the V2 subprocess wrapper by default.
Set them explicitly if a recipe overrides them. Add these entries to each worker component:

```yaml
spec:
  components:
    - name: decode
      podTemplate:
        spec:
          containers:
            - name: main
              env:
                - name: SGLANG_AUTO_NUMA_BIND
                  value: "1"
                - name: SGLANG_NUMA_BIND_V2
                  value: "1"
                - name: SGLANG_SET_CPU_AFFINITY
                  value: "0"
              args:
                - --numa-node
                - "0"
                - "0"
                - "1"
                - "1"
```

Omit `--numa-node` for automatic selection; explicit mappings take precedence over
`SGLANG_AUTO_NUMA_BIND`. Before each GPU scheduler starts, V2 requests `--membind` and binds CPUs to
the intersection of the target domain and allowed affinity. Failed probes can fall back to
`--preferred`, CPU-only, or no binding.

`SGLANG_NUMA_BIND_V2=0` selects legacy binding: CPU affinity and preferred memory applied inside
Python. To disable automatic NUMA selection, set `SGLANG_AUTO_NUMA_BIND=0` and omit `--numa-node`.
Keep the separate CPU-only `SGLANG_SET_CPU_AFFINITY` disabled with this policy. Worker NUMA binding
does not need HiCache. See [SGLang defaults](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/environ.py#L1565)
and [NUMA implementation](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/utils/numa_utils.py).

## Configure TensorRT-LLM Workers

For TP1 or GPUs sharing one CPU NUMA domain, wrap the engine command in `numactl`. This example
requires the worker's GPU, allocated CPUs, and allowed host memory to be in **NUMA node 0**. Verify
that placement first; a GPU request alone is insufficient.

```yaml
spec:
  backendFramework: trtllm
  components:
    - name: decode
      podTemplate:
        spec:
          containers:
            - name: main
              env:
                - name: TLLM_NUMA_AWARE_WORKER_AFFINITY
                  value: "0"
              command:
                - numactl
                - --cpunodebind=0
                - --preferred=0
                - python3
                - -m
                - dynamo.trtllm
```

Keep model, parallelism, and disaggregation flags in `args`. Use `--membind=0` for strict memory
binding or omit the memory option for CPU-only binding. `TLLM_NUMA_AWARE_WORKER_AFFINITY=0` preserves
the launch CPU mask during worker initialization. A value of `1` can replace it with the GPU's native
CPU-affinity mask; memory policy is independent. Check your TensorRT-LLM version against the
[native affinity implementation](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/tensorrt_llm/llmapi/utils.py#L713).

For multiple domains, wrap each rank's launch separately. Resolve its physical GPU from the
launcher's rank-to-device mapping and `CUDA_VISIBLE_DEVICES`. Normalize the PCI address to Linux
format and read its NUMA node from sysfs. Check that the domain has CPUs and is allowed by the Pod.
Launch the rank with `numactl --cpunodebind=N` and `--membind=N` or `--preferred=N`, preserving the
mask with `TLLM_NUMA_AWARE_WORKER_AFFINITY=0`. Check mappings on every host, including colocated
groups. One wrapper around the Dynamo parent cannot set separate rank policies.

## Bind the Frontend

> [!NOTE]
> A platform that selects its host address from a primary NIC on NUMA node `0` chooses a node-0-local
> address. Check the node and Pod topology. Dynamo's default `--http-host 0.0.0.0` listens on all
> local IPv4 addresses; it makes no NUMA choice. For placement on node `1`, select a node-1-local
> listen address explicitly.

Choose a domain in the frontend Pod's CPU and memory allocation, then set `--http-host` to an address
on a local NIC. Set `FRONTEND_POD` to the frontend Pod name and list its assigned addresses:

```bash
kubectl exec "$FRONTEND_POD" -c main -- ip -br address
```

Map the interface to its physical NIC and NUMA domain. Bonds, VLANs, and virtual interfaces may lack
a direct PCI mapping; a missing mapping or node `-1` means unknown locality. Choose a reachable IPv4
or IPv6 address without `/prefix`. Interface names and address order vary by network attachment.

For a frontend on node `1` with local NIC address `192.0.2.20`, wrap its command as follows.
Append the listen arguments to its existing frontend flags:

```yaml
spec:
  components:
    - name: Frontend
      podTemplate:
        spec:
          containers:
            - name: main
              command:
                - numactl
                - --cpunodebind=1
                - --preferred=1
                - python3
                - -m
                - dynamo.frontend
              args:
                - --http-host
                - "192.0.2.20"
```

Replace the IP with one assigned inside the frontend Pod. Expose host NIC addresses through host
networking or a secondary network first. Check client routing, Service routing, and HTTP probes:
probes to the primary Pod IP can fail when the frontend listens only on a secondary IP.

`DYN_HTTP_HOST` also sets the listen IP. Configure CPU and memory affinity, outbound routing, and
NIC interrupts separately. See
[Frontend HTTP and Networking](../../reference/components/frontend-configuration.mdx#http-and-networking).

Use `--membind=1` for strict local memory, with enough capacity reserved. Use `--physcpubind` to
select specific CPUs.
The frontend warns if allowed CPUs span domains but leaves affinity unchanged. Backend NUMA flags
do not bind the frontend Pod.

## Align KV-Transfer NICs

For key-value (KV) cache transfer over Remote Direct Memory Access (RDMA), choose network interface
cards (NICs) near the worker's GPUs and CPUs. Check `nvidia-smi topo -m` and
`/sys/class/infiniband/<device>/device/numa_node` inside the Pod, then allocate devices through the
cluster's networking policy.

For a Unified Communication X (UCX) KV-transfer path, add these entries to a worker in domain 0.
Replace the device names with its allocation and include a reachable Ethernet interface if TCP is needed:

```yaml
env:
  - name: UCX_NET_DEVICES
    value: mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1,eth0
  - name: UCX_TLS
    value: rc,cuda_ipc,cuda_copy,sm,self,tcp
```

In a rank-aware wrapper, choose the list from the resolved `numa_node` before launching the worker:

```bash
case "${numa_node}" in
    0)
        export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1,eth0
        ;;
    1)
        export UCX_NET_DEVICES=mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_11:1,eth0
        ;;
    *)
        echo "No NIC mapping for NUMA node ${numa_node}" >&2
        exit 1
        ;;
esac
```

`mlx5_*:1` selects RDMA ports; `eth0` keeps TCP available in both domains. Allocate and expose every
listed device. Wrapper exports affect only its worker and children, leaving other running ranks unchanged.

`UCX_NET_DEVICES` limits devices. Optional `UCX_TLS` selects reliable connection (RC), CUDA IPC and
copy, shared memory, loopback, and TCP here; check support in your UCX build and KV path. Explicit
lists need `tcp` for TCP interfaces. Keep it for control, active-message (AM), or fallback paths that
use TCP; AM can also use RDMA. TCP remains eligible for payloads, so check that KV data uses RC/RDMA.
See UCX's [transport selection](https://github.com/openucx/ucx/blob/v1.21.0/docs/source/faq.md#which-transports-does-ucx-use)
and [GPU RDMA guidance](https://github.com/openucx/ucx/blob/v1.21.0/docs/source/faq.md#does-ucx-support-zero-copy-for-gpu-memory-over-rdma).
Backend or connector device lists can override the environment; for example,
[NIXL passes explicit device lists into UCX](https://github.com/ai-dynamo/nixl/blob/v1.4.1/src/plugins/ucx/ucx_utils.cpp#L417).
For multi-domain Pods, select devices per rank or use domain-local worker groups. One Pod-wide NIC
list can force ranks across sockets. These settings affect only UCX paths, leaving frontend HTTP,
Dynamo request transport, and other KV backends unchanged.

## Verify Effective Placement

After rollout, check the node, QoS class, rendered requests, and devices:

```bash
kubectl get pod "$POD" -o jsonpath='{.spec.nodeName}{" "}{.status.qosClass}{"\n"}'
kubectl get pod "$POD" -o yaml
kubectl describe pod "$POD"
kubectl exec "$POD" -c main -- nvidia-smi -L
kubectl exec "$POD" -c main -- ls -l /dev/infiniband
kubectl exec "$POD" -c main -- ibv_devinfo -l
kubectl exec "$POD" -c main -- ibdev2netdev
kubectl exec "$POD" -c main -- ip -br address
kubectl exec "$POD" -c main -- ucx_info -d
kubectl exec "$POD" -c main -- sh -c 'ulimit -l'
```

Check for `Guaranteed` QoS and CPU/memory resources on every container. Match the Pod's HCA names
to `UCX_NET_DEVICES` and test transfers with the backend before benchmarking.

Inspect GPU scheduler, engine, and frontend processes:

```bash
kubectl exec "$POD" -c main -- ps -e -o pid,comm,args
kubectl exec "$POD" -c main -- awk '/Cpus_allowed_list|Mems_allowed_list/' /proc/1/status
kubectl exec "$POD" -c main -- cat /proc/1/numa_maps
kubectl exec "$POD" -c main -- numastat -p 1
kubectl logs "$POD" -c main
```

Replace PID `1` with worker PIDs from `ps`; PID 1 may be a launcher. Check thread affinity for worker
overrides. `Mems_allowed_list` shows allowed nodes; `numa_maps` shows policy and page placement, and
`numastat` shows distribution. Look for skipped binding or fallback in logs. For diagnostics, set
`UCX_LOG_LEVEL=info` and check the HCAs, TCP interface, and KV-payload transport. Compare frontend
latency, KV transfers, throughput, and host-memory pressure with the same
[AIPerf workload](benchmarking-with-aiperf.mdx) before and after the change.

## Diagnose Missing Resources

| Symptom | Check or change |
|---|---|
| Pod is pending with `Insufficient rdma/...` or GPU resources | Check plugin health, the exact advertised resource name, and node allocatable capacity. Request the intended resource in the DGD. |
| `TopologyAffinityError` after scheduling | Check resource availability together within each domain, scheduler topology freshness, and kubelet policy. The failed Pod is not rescheduled in place; a replacement can repeat the failure. |
| `SMTAlignmentError` | With `full-pcpus-only`, make each exclusive CPU request a multiple of the host's SMT width, including applicable init containers and sidecars. |
| Target CPUs or memory nodes are absent from the allowed sets | Check CPU/Memory Manager configuration, actual QoS, and the assigned cgroup masks. Reallocate the Pod; `numactl` cannot enlarge these sets. |
| `numactl` or `set_mempolicy` reports `Operation not permitted` | Check the effective seccomp profile and capabilities separately from the resource allocation. |
| `/dev/infiniband` or the expected HCA is missing | Check RDMA resource allocation and runtime device injection. Verify the configured pool contains the intended NICs. |
| UCX cannot find a device in `UCX_NET_DEVICES` | Use the names visible inside the Pod and check connector device-list overrides. Confirm the TCP netdev and RDMA ports are both available. |
| `ibv_reg_mr` fails to register memory | Check memlock, registration capacity, node drivers, and GPUDirect RDMA configuration. |
| The frontend reports `Cannot assign requested address` | Check that its listen IP is assigned inside its network namespace and that the intended host or secondary network is attached. |
| Devices are present but transfers cross NUMA domains | Compare actual GPU, CPU, memory, and NIC allocation. Shared-pool resource names alone do not align them. |

## Track Workload-Level Grove and DRA Support

Grove NUMA placement needs controller and scheduler support alongside Kubernetes Dynamic Resource
Allocation (DRA) features. Follow
[Grove issue #644](https://github.com/ai-dynamo/grove/issues/644) and
[the group-placement proposal](https://github.com/ai-dynamo/grove/pull/851) for workload-level
constraints, and check what the installed Dynamo/Grove versions implement.

The relevant Kubernetes features are:

| Feature | Role |
|---|---|
| [KEP-6072: Standard NUMA-node attribute](https://github.com/kubernetes/enhancements/issues/6072) | Standardizes `resource.kubernetes.io/numaNode` for devices; stable in Kubernetes `1.37`. |
| [KEP-6080: Derived attributes](https://github.com/kubernetes/enhancements/issues/6080) | Derives common attributes for claim constraints when driver schemas differ; an opt-in Alpha feature in `1.37`. |
| `resource.kubernetes.io/pcieRoot` | Describes PCIe root locality, separately from NUMA memory locality. |

For the Kubernetes `1.37` changes, see the
[DRA release summary](https://kubernetes.io/blog/2026/09/03/kubernetes-v1-37-dra-updates/).
Drivers must publish locality attributes, ResourceClaims must express the constraints, and the
controller and scheduler must support them. Derived attributes need `DRADerivedAttributes` enabled
on the API server and scheduler. Drivers publishing compatible standard attributes do not need
derivation. Missing locality attributes mean unknown topology, not node `0`. See the
[standard device attributes](https://kubernetes.io/docs/reference/node/dra-standard-device-attributes/)
and [derived-attribute requirements](https://kubernetes.io/docs/concepts/resource-management/dynamic-resource-allocation/dra-api/#derived-attributes).

Grove's [per-Pod DRA proposal](https://github.com/ai-dynamo/grove/pull/850) covers GPU co-allocation within
one Pod. Multi-Pod prefill/decode placement remains separate work. Matching GPU/NIC attributes also
leaves ordinary CPU and memory requests to their resource managers.

Check the platform's Kubernetes, DRA-driver, Grove, scheduler, and Dynamo versions, then validate the
allocation. Apply frontend and engine binding to the CPUs and memory domains actually assigned.
