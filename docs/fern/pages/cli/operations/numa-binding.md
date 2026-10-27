---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: NUMA Binding
subtitle: Place frontend threads, GPU workers, host memory, and KV-transfer NICs near each other
---

Non-uniform memory access (NUMA) binding keeps NVIDIA Dynamo's CPU work and host memory close to
the GPUs they serve. It's useful on multi-socket H100, H200, B200, and B300 servers, as well as
Grace-based systems. The right NUMA node depends on the server's CPU and PCIe layout.

Add these settings to a working deployment, keeping its model, discovery, and KV-transfer options.

## Inspect the Topology

Install `numactl` and `libnuma`, then check the Linux host's topology from the container or
scheduler allocation where the workers will run:

```bash
numactl --hardware
lscpu --extended=CPU,NODE,SOCKET,CORE
nvidia-smi topo -m
```

Note the CPUs and memory node closest to each GPU. Check the devices visible through
`CUDA_VISIBLE_DEVICES` and the CPU sets assigned by the container runtime or scheduler. GPU IDs
aren't necessarily ordered by socket, and a socket can contain several NUMA nodes.

To check one GPU, use NVIDIA's
[topology queries](https://docs.nvidia.com/deploy/nvidia-smi/index.html#topology):

```bash
nvidia-smi topo -C -i 0
nvidia-smi topo -M -i 0
```

The first command reports the nearest CPU NUMA node; the second reports the nearest memory node.
On hardware-coherent platforms, GPU high-bandwidth memory (HBM) can appear as a NUMA node without
CPUs. Use a node with CPUs for `--cpunodebind`. For the distinction between host-memory binding
and GPU HBM management, see NVIDIA's
[hardware-coherent memory guide](https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms/).

## Choose a CPU and Memory Policy

`numactl` sets CPU and memory policies before starting a process. Children inherit those policies,
although an engine or launcher can change them during startup.

| Option | Effect |
|---|---|
| `--cpunodebind=N` | Restrict CPU execution to cores belonging to NUMA node `N`. |
| `--physcpubind=CPU_LIST` | Restrict execution to specific Linux CPU IDs. |
| `--membind=N` | Allocate host memory only on node `N`; allocations can fail when it runs out of memory. |
| `--preferred=N` | Prefer host memory on node `N`, with fallback to other permitted nodes. |

Use `--preferred` if you'd rather use remote memory than fail an allocation. Use `--membind` when
allocations must stay local and the node has enough memory. Neither option moves all existing
pages or shared buffers to the selected node.
See the [numactl manual](https://man7.org/linux/man-pages/man8/numactl.8.html).

> [!IMPORTANT]
> Apply binding to the process or rank that owns each GPU. Wrapping a whole tensor parallel (TP)
> group in a single-node `numactl` policy can put every rank on the same socket, even when its GPUs
> span several domains. Use the engine's per-worker binding or a rank-aware launcher for those groups.

## Bind the Frontend

Bind the frontend separately from the backend workers. Choose a NUMA node with enough CPU and
memory for tokenization, routing, and response handling. For example, to use node `0`:

```bash
numactl --cpunodebind=0 --preferred=0 \
  python3 -m dynamo.frontend
```

Use `--membind=0` for strict local memory. To give the frontend only part of the node's CPUs,
replace `--cpunodebind=0` with `--physcpubind` and the CPU IDs you chose from the topology output.

At startup, Dynamo warns if the frontend's allowed CPUs span several NUMA nodes. You'll still
need to apply binding yourself. See the
[frontend affinity check](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/frontend/cpu_affinity.py).

### Select a NUMA-Local Listen Address

> [!NOTE]
> If your platform picks the host address from a primary NIC on NUMA node `0`, that address is
> local to node `0`. Check your server's layout before relying on this. Dynamo's default is
> `--http-host 0.0.0.0`, which listens on all local IPv4 addresses. To use a node-1-local address,
> set it explicitly.

Choose a NIC on the same NUMA node as the frontend and pass its IP address to `--http-host`.
Interface names vary, so run this in the frontend's network namespace to find directly attached
NICs and their IPv4 addresses on node `1`:

```bash
FRONTEND_NUMA_NODE=1
for frontend_nic_path in /sys/class/net/*; do
    frontend_numa_path="${frontend_nic_path}/device/numa_node"
    if [[ -r "${frontend_numa_path}" ]] &&
       [[ "$(<"${frontend_numa_path}")" == "${FRONTEND_NUMA_NODE}" ]]; then
        ip -o -4 addr show dev "${frontend_nic_path##*/}" scope global
    fi
done
```

Pick a reachable IP from the output and set `FRONTEND_IP` without the `/prefix`. Replace the
example address below with that IP:

```bash
FRONTEND_IP=192.0.2.20
numactl --cpunodebind="${FRONTEND_NUMA_NODE}" --preferred="${FRONTEND_NUMA_NODE}" \
  python3 -m dynamo.frontend --http-host "${FRONTEND_IP}"
```

Bonds, VLANs, and virtual interfaces may have no direct PCI NUMA mapping. A missing `numa_node`
file or a value of `-1` means the locality is unknown. Use `ip -br address` to list their addresses,
trace the underlying physical NIC, and choose the logical interface's IP. If several addresses
match, choose one reachable by your clients. For IPv6, use `ip -o -6 addr` and pass the IPv6
address to `--http-host`.

You can also set the listen IP with `DYN_HTTP_HOST`. This controls the HTTP listener; `numactl`
controls CPU and memory affinity. Outbound routing and NIC interrupt placement are OS settings.
See
[Frontend HTTP and Networking](../../reference/components/frontend-configuration.mdx#http-and-networking).

## Bind vLLM Workers

Dynamo accepts vLLM's native engine arguments. Add `--numa-bind` to discover each GPU's local
NUMA node and bind its execution subprocesses:

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 4 \
  --numa-bind
```

If GPUs `0` and `1` are on NUMA node `0`, and GPUs `2` and `3` are on node `1`, you can give
vLLM that mapping explicitly:

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 4 \
  --numa-bind \
  --numa-bind-nodes 0 0 1 1
```

Use `--numa-bind-cpus` to assign CPU sets, with one list per visible GPU. For example,
`--numa-bind-cpus 0-3 4-7 48-51 52-55` works if those CPUs belong to the matching NUMA nodes
and are available in your allocation.

vLLM normally uses `--cpunodebind=N --membind=N`, switching to `--physcpubind` for explicit CPU
lists. EngineCore binds to the NUMA domains used by its GPU shard. Each multiprocessing worker
binds to its GPU's domain. The frontend and data-parallel coordinator need separate placement.

Binding is off by default. Dynamo already uses the required `spawn` multiprocessing method.
If existing CPU restrictions prevent automatic discovery, pass the node mapping explicitly.
vLLM probes the requested policy and may fall back to CPU-only binding or no binding if it fails.
Check the logs and process placement after startup.

These options are available in vLLM `0.31.0` and `0.32.0`. See the
[upstream NUMA guide](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/configuration/optimization.md#numa-binding-for-multi-socket-gpu-nodes)
and [binding implementation](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/utils/numa_utils.py).
Those versions have no engine option for preferred-memory placement; `--preferred` belongs to
the `numactl` launcher.

## Bind SGLang Workers

Dynamo accepts SGLang's native `ServerArgs`. SGLang `0.5.21` and `0.5.22` enable automatic NUMA
discovery and the subprocess wrapper by default. To set them explicitly:

```bash
SGLANG_AUTO_NUMA_BIND=1 \
SGLANG_NUMA_BIND_V2=1 \
SGLANG_SET_CPU_AFFINITY=0 \
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-8B \
  --tp-size 4
```

For the same four-GPU mapping, add `--numa-node 0 0 1 1`, or set `numa_node: [0, 0, 1, 1]` in a
native SGLang config file. SGLang indexes the list by the scheduler's local GPU ID, so check
device visibility and `--base-gpu-id` when building it.

The V2 wrapper applies `--cpunodebind=N --membind=N` before the scheduler's Python process starts.
If only some CPUs on that node are allowed, it uses that subset with `--physcpubind`. A failed
memory-policy probe can fall back to `--preferred`, then CPU-only binding. Automatic discovery
may skip binding if NUMA support or memory-policy permissions are missing.

The switches have different jobs:

| Control | Behavior |
|---|---|
| `SGLANG_NUMA_BIND_V2=1` | Apply binding before the scheduler interpreter starts. |
| `SGLANG_NUMA_BIND_V2=0` | Select the legacy in-process CPU-affinity and preferred-memory path. |
| `SGLANG_AUTO_NUMA_BIND=0` | Disable automatic discovery; an explicit `--numa-node` still takes effect. |
| `SGLANG_SET_CPU_AFFINITY=1` | Apply a separate CPU partitioning scheme without setting memory policy. |

Leave `SGLANG_SET_CPU_AFFINITY=0` when using NUMA binding so the two CPU policies don't overlap.
To turn off automatic binding, set `SGLANG_AUTO_NUMA_BIND=0` and omit the node mapping.
Scheduler NUMA binding works without HiCache.

> [!NOTE]
> `SGLANG_CRASH_ON_NUMA_BIND_FAILURE=1` still allows successful fallback to preferred-memory or
> CPU-only binding. Check the resulting memory policy if you need strict placement.

See SGLang's [environment defaults](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/environ.py#L1565),
[binding implementation](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/utils/numa_utils.py),
and [legacy scheduler path](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler.py#L6021).

## Bind TensorRT-LLM Workers

Wrap the worker with `numactl` and set `TLLM_NUMA_AWARE_WORKER_AFFINITY=0` so TensorRT-LLM keeps
the CPU mask it inherits. This example assumes GPU `0` is local to CPU NUMA node `0` and the
model fits on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
TLLM_NUMA_AWARE_WORKER_AFFINITY=0 \
numactl --cpunodebind=0 --preferred=0 \
  python3 -m dynamo.trtllm \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 1
```

Use `--membind=0` for strict local memory, or omit the memory option for CPU binding alone.
You can use one wrapper for several GPUs if they all belong to the chosen CPU NUMA domain.

TensorRT-LLM's own CPU-affinity setting behaves as follows:

| `TLLM_NUMA_AWARE_WORKER_AFFINITY` | CPU-affinity behavior |
|---|---|
| Unset | Attempts to reset an inherited restricted mask to all CPUs; with an unrestricted mask, selects GPU-local NVML affinity. |
| `0` | Preserve inherited CPU affinity. |
| `1` | Select GPU-local NVML CPU affinity, even if the inherited mask is restricted. |

This applies to TensorRT-LLM `1.3.0rc29` and `1.4.0rc0`; see
[`configure_cpu_affinity`](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/tensorrt_llm/llmapi/utils.py#L713).
For native CPU affinity alone, set `TLLM_NUMA_AWARE_WORKER_AFFINITY=1` and omit the `numactl`
wrapper. Set host-memory policy separately.

### Bind Each TensorRT-LLM Rank

For TP or pipeline parallelism across NUMA domains, apply binding as each GPU worker rank starts:

1. Find the rank's physical GPU from the launcher mapping and `CUDA_VISIBLE_DEVICES`. Local rank
   IDs can restart at zero for each worker group sharing a host.
2. Query the GPU's PCI bus ID and normalize it to Linux PCI address format, for example
   `00000000:19:00.0` to `0000:19:00.0`, before reading
   `/sys/bus/pci/devices/<PCI_BUS_ID>/numa_node`. Check that the node has CPUs and is included in
   the rank's allowed CPU and memory sets.
3. Set `TLLM_NUMA_AWARE_WORKER_AFFINITY=0` and start the rank under
   `numactl --cpunodebind=N`, adding `--membind=N` for strict memory placement or `--preferred=N`
   for local memory with fallback.
4. Check that the engine ranks inherit the policy and environment settings.

Use a launcher that supports per-rank placement, and check the mapping on each host. A wrapper
around the Dynamo parent gives all its children the same initial policy.

## Select Local NICs for RDMA KV Transfer

For disaggregated KV transfer over remote direct memory access (RDMA), choose network interface
cards (NICs) close to each worker's GPU and NUMA domain. Set the NIC choice separately from CPU
and memory binding. The GPU-to-NIC PCIe path also affects GPUDirect RDMA performance; see the
[UCX GPU RDMA guidance](https://github.com/openucx/ucx/blob/v1.21.0/docs/source/faq.md#does-ucx-support-zero-copy-for-gpu-memory-over-rdma).

Check the devices and their locality:

```bash
nvidia-smi topo -m
ibdev2netdev
cat /sys/class/infiniband/mlx5_0/device/numa_node
cat /sys/class/infiniband/mlx5_0/device/local_cpulist
ucx_info -d
```

For a KV-transfer backend using Unified Communication X (UCX), select local RDMA host channel
adapter (HCA) ports before UCX initializes. Include a reachable Ethernet interface if the
deployment needs TCP. With `numa_node` already set for the worker, for example:

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

Replace the names with devices from your server. `mlx5_*:1` selects RDMA devices and ports;
`eth0` keeps a TCP interface available in both domains. All listed devices must be visible to
the worker, and peers must be able to reach the TCP interface.

To use reliable-connected (RC) RDMA with CUDA IPC and copy, host shared memory, loopback, and TCP:

```bash
export UCX_TLS=rc,cuda_ipc,cuda_copy,sm,self,tcp
```

TCP needs both the `tcp` transport and an interface in `UCX_NET_DEVICES`. Some deployments use
it for control, active-message (AM), or fallback traffic. AM can also run over RDMA. Since this
configuration also permits payload traffic over TCP, check that KV transfers use RC/RDMA as intended.

`rc` includes the required bootstrap transports; `cuda_ipc` and `cuda_copy` enable CUDA IPC and
copy paths, `sm` enables host shared memory, and `self` enables same-process communication.
RC works with compatible InfiniBand or RDMA over Converged Ethernet (RoCE) fabrics. Leave
`UCX_TLS` unset if you want UCX to choose transports automatically. The fabric must already be
configured for RDMA. See the
[UCX device and transport controls](https://github.com/openucx/ucx/blob/v1.21.0/docs/source/faq.md#which-transports-does-ucx-use).

Set NIC selection on both prefill and decode workers. Every rank inherits a container-wide
`UCX_NET_DEVICES` value. For a group spanning domains, set it per rank, use separate groups within
each domain, or give UCX a list such as `mlx5_0:1,mlx5_4:1`. Forcing all ranks through one NIC
can lose locality and multi-rail bandwidth.

A NIXL UCX backend `device_list` can override `UCX_NET_DEVICES`, so check the connector's config.
`NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` only configure NCCL. For a libfabric or other non-UCX
KV-transfer backend, use that backend's controls. See the
[NIXL device override](https://github.com/ai-dynamo/nixl/blob/v1.4.1/src/plugins/ucx/ucx_utils.cpp#L417)
and [TensorRT-LLM KV-transfer backends](../../developer-guide/additional-resources/tensorrt-llm-details/kv-cache-transfer.md).

## Verify Effective Placement

First test the policy in the container or scheduler task that launches the worker. For node `0`:

```bash
numactl --cpunodebind=0 --membind=0 true
```

The selected domain must be in the allocation's allowed CPU and memory sets. If the probe fails,
check its error: container security policies may block NUMA memory-policy calls. Some runtimes
need `SYS_NICE` or a seccomp profile allowing those calls. Adjust the permissions allowed by your
deployment policy and rerun the probe before relying on strict binding.

After startup, use `ps` to find the frontend and each GPU worker or scheduler. Set `WORKER_PID`
to a PID from the listing, then run the remaining checks for that process:

```bash
ps -eo pid,ppid,comm,args
taskset -pc "${WORKER_PID}"
grep -E 'Cpus_allowed_list|Mems_allowed_list' "/proc/${WORKER_PID}/status"
head -n 20 "/proc/${WORKER_PID}/numa_maps"
numastat -p "${WORKER_PID}"
```

`Mems_allowed_list` shows which memory nodes the process may use. Check `numa_maps` for its actual
policies and page placement under load, after the engine has finished initializing. For UCX,
temporarily set `UCX_LOG_LEVEL=info` and check the HCA, TCP interface, and KV-payload transport.

Compare latency, throughput, memory placement, and KV-transfer performance with the same workload
before and after binding. If you use Dynamo KV Block Manager (KVBM), check its
[`DYN_KVBM_ENABLE_NUMA` host-cache setting](../../reference/components/kvbm-configuration.mdx#mla-and-nccl)
as well; it controls host-cache allocation separately from process placement.
