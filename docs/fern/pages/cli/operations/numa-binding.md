---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: NUMA Binding
subtitle: Place frontend threads, GPU workers, host memory, and KV-transfer NICs near each other
---

Non-uniform memory access (NUMA) binding keeps NVIDIA Dynamo's CPU work and host memory close to
the GPUs they serve. Pay particular attention to placement on **GB200 and GB300 systems**. In
NVL72 compute trays, each Grace CPU connects to two GPUs over cache-coherent NVLink-C2C.
Threads and host buffers on the other Grace's NUMA node are remote to those GPUs. When GPU
high-bandwidth memory (HBM) is exposed to the operating system, a process without explicit NUMA
placement can also have host pages allocated in HBM. Binding is also useful on multi-socket
H100, H200, B200, and B300 servers. The right NUMA node depends on the server's CPU, memory,
GPU, and NIC topology.

Add these settings to a working deployment, keeping its model, discovery, and KV-transfer options.

## Where NUMA Binding Can Help

Keeping CPU threads, host buffers, and NICs close to their GPUs can improve several serving paths:

- **Time to First Token (TTFT):** Local CPU work, cached KV reloads, and prefill-to-decode
  handoffs can reduce the time spent waiting before the first token.
- **Inter-Token Latency (ITL):** Better locality can reduce gaps between decode steps when
  host-side work or KV movement delays token generation.
- **KV-cache offloading and onboarding:** Placing host-cache buffers near the GPU can improve
  bandwidth and latency when offloading KV blocks from GPU to host memory and onboarding them
  back to the GPU for reuse. For storage-backed caches that use host staging buffers, their
  placement matters too.
- **KV transfer:** Pairing each GPU with a nearby NIC and local host buffers, when used, can
  reduce cross-domain traffic during prefill/decode handoffs and remote-cache transfers.

Measure these paths with your own workload; the benefit depends on whether CPU work,
host-memory access, or transfers limit performance. See
[NVIDIA's Grace NUMA guidance](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html#unified-memory-considerations-on-grace-hopper-blackwell-systems)
and the [KV-cache offloading guide](../kv-cache-offloading/overview.mdx).

## Check the Tools in Your Image

Run the examples in a Linux environment with `numactl` and `libnuma` installed. The frontend
needs these too if you use the CPU-and-memory wrapper. The current CUDA image definitions differ:

| Dynamo image | Includes `numactl` |
|---|---|
| `vllm-runtime` | Yes |
| `tensorrtllm-runtime` | Yes |
| `sglang-runtime` | No; includes `libnuma` for SGLang's legacy binding path |
| `dynamo-frontend` | No |

Check your actual image with `command -v numactl`; packages can change between tags and platforms.
For a derived Ubuntu image, install the binding and diagnostic tools during the image build:

```bash
apt-get update
apt-get install -y --no-install-recommends numactl libnuma1 iproute2 util-linux procps
rm -rf /var/lib/apt/lists/*
```

`ip` comes from `iproute2`, `lscpu` and `taskset` from `util-linux`, `ps` from `procps`, and
`numastat` from `numactl`. Minimal runtime images may omit them. UCX libraries alone do not
provide `ucx_info`; use that diagnostic only if your UCX installation includes it.

## Inspect the Topology

Check the Linux host's topology from the container or scheduler allocation where the workers
will run:

```bash
numactl --hardware
lscpu --extended=CPU,NODE,SOCKET,CORE
nvidia-smi topo -m
```

Note the CPUs and memory node closest to each GPU. Check the devices visible through
`CUDA_VISIBLE_DEVICES` and the CPU sets assigned by the container runtime or scheduler. GPU IDs
aren't necessarily ordered by socket, and a socket can contain several NUMA nodes.

`nvidia-smi -i` accepts a physical NVML index, GPU UUID, or PCI bus ID, not a position in
`CUDA_VISIBLE_DEVICES`. List the devices first, then match their UUIDs or PCI addresses to the
engine's CUDA-visible order before building per-GPU NUMA mappings:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv
```

To check one GPU, use the Topology section of the
[nvidia-smi reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
Here `-i 0` refers to the physical NVML index:

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
| `--membind=N` | Allocate host memory only on node `N`; exhaustion can cause allocation failure or an out-of-memory (OOM) kill. |
| `--preferred=N` | Prefer host memory on node `N`, with fallback to other permitted nodes. |

Use `--preferred` if you'd rather use remote memory than risk allocation failure or an OOM kill
when local memory runs out. Use `--membind` when allocations must stay local and the node has
enough memory. Neither option moves all existing pages or shared buffers to the selected node.
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
For CPU affinity alone, `taskset -c CPU_LIST python3 -m dynamo.frontend` works without
`numactl`, but sets no memory policy. Replace `CPU_LIST` with your allowed CPU IDs.

At startup, Dynamo warns if the frontend's allowed CPUs span several NUMA nodes. You'll still
need to apply binding yourself. See the
[frontend affinity check](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/frontend/cpu_affinity.py).

### Optionally Select a NUMA-Local Listen Address

Dynamo defaults to `--http-host 0.0.0.0`, listening on all local IPv4 addresses, including
localhost. You can keep that default with the node-0 CPU binding above. Check which NUMA node
the NIC carrying client traffic is on; Dynamo doesn't select an address by NUMA node.

To move both the frontend and its incoming HTTP traffic to node `1`, choose a reachable IP on
a NIC local to that node and pass it to `--http-host`. Interface names vary, so run this in the
frontend's network namespace to find directly attached NICs and their IPv4 addresses:

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

With a specific listen IP, `http://localhost:8000` no longer reaches the frontend. Point clients,
benchmarks, and HTTP health checks at `http://${FRONTEND_IP}:8000` instead.

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

Dynamo accepts vLLM's native engine arguments. `--numa-bind` binds EngineCore and multiprocessing
GPU workers. Automatic topology discovery requires at least two NUMA nodes, unrestricted CPU
affinity, and permission to query memory policy with `get_mempolicy`:

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 4 \
  --numa-bind
```

For a restricted CPU allocation, including Slurm tasks or Docker `--cpuset-cpus`, pass the mapping
explicitly. If GPUs `0` and `1` are on NUMA node `0`, and GPUs `2` and `3` are on node `1`:

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 4 \
  --numa-bind \
  --numa-bind-nodes 0 0 1 1
```

Use `--numa-bind-cpus` to assign multiprocessing workers CPU sets, with one list per visible GPU.
For example, `--numa-bind-cpus 0-3 4-7 48-51 52-55` works if those CPUs belong to the matching
NUMA nodes and are available in your allocation. EngineCore ignores these per-worker lists.
For a single-GPU worker using the default `uni` executor, the model runs inside EngineCore without
worker subprocesses, so this option does not narrow its CPU affinity.

Normally, each multiprocessing worker uses `--cpunodebind=N --membind=N`, or `--physcpubind` for
an explicit CPU list. EngineCore binds to the domains used by its GPU shard. On detected
Priority Core Turbo (PCT) Xeons, such as the 6776P in DGX B300, vLLM instead uses `--physcpubind`
to select priority cores for workers and EngineCore when no CPU lists are supplied. Those cores
must fit the allowed CPU set; rejected CPU binding can leave the process without NUMA binding.

The `dynamo.vllm` parent hosting AsyncLLM, the separate Dynamo frontend, and the data-parallel
coordinator are not bound by `--numa-bind`. If you bind the parent with `numactl` or `taskset`, also
pass `--numa-bind-nodes`: its restricted affinity disables automatic discovery. Keep the parent
allocation wide enough for every GPU rank's target CPUs and memory nodes.

Binding is off by default. Dynamo already uses the required `spawn` multiprocessing method.
When discovery fails and no explicit nodes are supplied, `--numa-bind` aborts startup with
"could not detect the GPU-to-NUMA topology automatically". Only after node IDs are known does
vLLM probe the binding policy and possibly fall back to CPU-only binding or no binding.
Check the logs and process placement after startup.

These examples target vLLM `0.32.0`. See the
[upstream NUMA guide](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/configuration/optimization.md#numa-binding-for-multi-socket-gpu-nodes)
and [binding implementation](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/utils/numa_utils.py).
That version has no engine option for preferred-memory placement; `--preferred` belongs to
the `numactl` launcher.

## Bind SGLang Workers

Dynamo accepts SGLang's native `ServerArgs`. SGLang `0.5.21` and `0.5.22` enable automatic NUMA
discovery and the V2 subprocess wrapper by default, but V2 needs the `numactl` executable in the
worker image. The pinned CUDA runtime Dockerfiles install `libnuma` without `numactl`; verify
`command -v numactl` inside the container and add the package to a derived image before enabling V2.
Without it, automatic V2 binding is skipped with only a DEBUG-level message. By default, explicit
node lists warn and skip binding after failed probes. Set `SGLANG_NUMA_BIND_V2=0` to
use the legacy `libnuma` path if `numactl` is unavailable.

With `numactl` installed, set V2 explicitly:

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
| `SGLANG_NUMA_BIND_V2=1` | Apply binding through `numactl` before the scheduler interpreter starts, when the executable is available. |
| `SGLANG_NUMA_BIND_V2=0` | Select the legacy in-process CPU-affinity and preferred-memory path. |
| `SGLANG_AUTO_NUMA_BIND=0` | Disable automatic discovery; an explicit `--numa-node` still takes effect. |
| `SGLANG_SET_CPU_AFFINITY=1` | Apply a separate CPU partitioning scheme without setting memory policy. |

Leave `SGLANG_SET_CPU_AFFINITY=0` when using NUMA binding so the two CPU policies don't overlap.
To turn off automatic binding, set `SGLANG_AUTO_NUMA_BIND=0` and omit the node mapping.

> [!NOTE]
> `SGLANG_CRASH_ON_NUMA_BIND_FAILURE=1` does not fire when automatic discovery skips binding.
> With V2, supply an explicit `--numa-node` mapping to force a binding attempt. The flag still
> permits successful fallback to preferred-memory or CPU-only binding. Check `Mems_allowed_list`
> and `numa_maps` after startup; the flag does not guarantee strict memory placement.

See SGLang's [environment defaults](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/environ.py#L1565),
[binding implementation](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/utils/numa_utils.py),
and [legacy scheduler path](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler.py#L6021).

## Bind TensorRT-LLM Workers

Wrap the worker with `numactl` and set `TLLM_NUMA_AWARE_WORKER_AFFINITY=0` so TensorRT-LLM keeps
the CPU mask it inherits. Also disable Open MPI rebinding: otherwise MPI initialization or
worker spawning can replace that mask before TensorRT-LLM sees it. Dynamo's TensorRT-LLM runtime
image sets Open MPI's binding policy to `none`; the explicit settings below also cover other
environments. This example assumes GPU `0` is local to CPU NUMA node `0` and the model fits on
one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
TLLM_NUMA_AWARE_WORKER_AFFINITY=0 \
OMPI_MCA_hwloc_base_binding_policy=none \
OMPI_MCA_rmaps_base_inherit=1 \
numactl --cpunodebind=0 --preferred=0 \
  python3 -m dynamo.trtllm \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 1
```

Use `--membind=0` for strict local memory. Omitting the memory option gives CPU-only binding
only if libnuma reports NUMA available. If blocked `get_mempolicy` calls make `--cpunodebind`
fail, replace the `numactl` prefix with `taskset -c CPU_LIST`, using inspected, allowed CPU IDs
and keeping the TensorRT-LLM and Open MPI environment settings above. This sets no memory policy.
You can use one wrapper for several GPUs if they all belong to the chosen CPU NUMA domain.
Check the spawned workers' CPU masks after MPI initialization. See TensorRT-LLM's
[Open MPI affinity settings](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/docs/source/deployment-guide/configuring-cpu-affinity.md#openmpi).

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

The default `python3 -m dynamo.trtllm --tensor-parallel-size N` launch constructs TensorRT-LLM
in the Dynamo process. TensorRT-LLM then spawns its MPI workers internally, with a shared launch
environment and inherited memory policy. That path has no hook for a separate `numactl` command
per rank. Use native CPU affinity with `TLLM_NUMA_AWARE_WORKER_AFFINITY=1` for per-worker CPU
placement, or choose the external launch below when you need per-rank memory policies too.

For external MPI ranks, put the NUMA wrapper **before** `trtllm-llmapi-launch`. That launcher starts
the engine workers on the externally launched ranks and runs the Dynamo application from rank 0.
Calling `mpirun` directly on `python3 -m dynamo.trtllm` does not select this launch mode. See the
[TensorRT-LLM launcher](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/tensorrt_llm/llmapi/trtllm-llmapi-launch)
and [MPI session selection](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/tensorrt_llm/llmapi/llm.py#L459).

The following example is for one host with four allocated GPUs. Set `CUDA_VISIBLE_DEVICES` to
those four **GPU UUIDs**, in the same order for every rank; UUIDs avoid ambiguity between CUDA
and NVML device ordinals. Save this wrapper as `trtllm-numa-rank.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${OMPI_COMM_WORLD_RANK:?Run this wrapper under mpirun}"
: "${CUDA_VISIBLE_DEVICES:?Set the allocated GPU UUID list}"
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#devices[@]}" -eq 4 ]] || {
    echo "This example requires four allocated GPU UUIDs" >&2
    exit 2
}

gpu_index=$((OMPI_COMM_WORLD_RANK % ${#devices[@]}))
gpu="${devices[gpu_index]}"
pci_bus_id="$(nvidia-smi -i "${gpu}" --query-gpu=pci.bus_id --format=csv,noheader)"
pci_bus_id="${pci_bus_id,,}"
printf -v pci_bus_id '%04x:%s' "0x${pci_bus_id%%:*}" "${pci_bus_id#*:}"
numa_node="$(cat "/sys/bus/pci/devices/${pci_bus_id}/numa_node")"
[[ "${numa_node}" =~ ^[0-9]+$ ]] &&
    [[ -n "$(cat "/sys/devices/system/node/node${numa_node}/cpulist")" ]] || {
        echo "GPU ${gpu} has no usable CPU NUMA node" >&2
        exit 2
    }

export TLLM_NUMA_AWARE_WORKER_AFFINITY=0
exec numactl --cpunodebind="${numa_node}" --preferred="${numa_node}" "$@"
```

TensorRT-LLM's [worker device selection](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/tensorrt_llm/executor/base_worker.py#L149)
uses `global_rank % visible_device_count`, so the wrapper uses the **global** MPI rank, not a
local-rank-to-socket table. It normalizes both the PCI domain width and hex case:
`00000000:9D:00.0` becomes `0000:9d:00.0`. Unknown or CPU-less NUMA nodes stop the launch;
use an explicit, inspected CPU-node mapping on systems whose PCI affinity is insufficient.

With the four-GPU UUID mask exported, launch four ranks:

```bash
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_inherit=1

mpirun -n 4 --bind-to none \
  -x CUDA_VISIBLE_DEVICES \
  -x OMPI_MCA_hwloc_base_binding_policy \
  -x OMPI_MCA_rmaps_base_inherit \
  bash ./trtllm-numa-rank.sh \
  trtllm-llmapi-launch \
  python3 -m dynamo.trtllm \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 4
```

Replace `--preferred` in the wrapper with `--membind` for strict per-rank host memory. Verify that
the selected CPU and memory nodes are permitted by the allocation and that every rank keeps its
policy after initialization. For multiple hosts, validate the global-rank modulo mapping on each
host, along with its visible-device list; rank ordering must assign each local GPU correctly.
An internally spawned TP group cannot obtain distinct rank memory policies from a wrapper around
the Dynamo parent alone.

## Select Local NICs for RDMA KV Transfer

For disaggregated KV transfer over remote direct memory access (RDMA), choose network interface
cards (NICs) close to each worker's GPU and NUMA domain. The GPU-to-NIC PCIe path also affects
GPUDirect RDMA performance; see the
[UCX GPU RDMA guidance](https://github.com/openucx/ucx/blob/v1.21.0/docs/source/faq.md#does-ucx-support-zero-copy-for-gpu-memory-over-rdma).

SGLang `0.5.22` defaults to Mooncake for KV transfer. Choose its NICs with
`--disaggregation-ib-device mlx5_0`, a shared comma-separated list, or a per-GPU JSON map such as
`--disaggregation-ib-device '{"0":"mlx5_0","1":"mlx5_6"}'`. Replace the example mapping and
cover every GPU used by the worker. These are SGLang GPU IDs, not NUMA node IDs. Unified
Communication X (UCX) settings affect SGLang KV transfer only when
`--disaggregation-transfer-backend nixl` selects NIXL with its UCX backend; they do not control
Mooncake. See the
[SGLang transfer options](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/arg_groups/fields/disagg.py#L65-L84).

UCX already selects devices using bandwidth, latency, PCI bandwidth,
and NUMA locality. Keep a working recipe's automatic or injected device selection. Restrict
`UCX_NET_DEVICES` when transfer logs show the wrong device or measurements justify a narrower
set; a static list can otherwise hide usable devices or break backend initialization. For
examples of deployment-specific requirements, see the
[Solar Open2 networking notes](../../recipes/model-recipes/solar-open2-250b.mdx#configuration-notes) and
[DeepSeek-V4.1 Flash device allocation](../../recipes/model-recipes/deepseek-v4-1-flash.mdx).

Check GPU/NIC locality and map RDMA devices to their Linux network interfaces through sysfs:

```bash
nvidia-smi topo -m
for hca in /sys/class/infiniband/*; do
    [[ -d "${hca}" ]] || continue
    printf '%s: ' "${hca##*/}"
    ls "${hca}/device/net/"
done
cat /sys/class/infiniband/mlx5_0/device/numa_node
cat /sys/class/infiniband/mlx5_0/device/local_cpulist
```

If the UCX tools are installed, `ucx_info -d` lists the devices and transports that UCX can use.
Enable `UCX_LOG_LEVEL=info` for a diagnostic run to check the devices actually selected. Confirm
which ports belong to the compute fabric with the cluster administrator; NUMA proximity alone
does not distinguish compute, storage, and management networks.

For a worker confined to one domain, a rank-aware launcher can restrict RDMA host channel adapter
(HCA) ports before UCX initializes. The following is an illustrative compute-NIC split, not a
mapping for every H100, H200, B200, or B300 server. Use it only if the inspected topology and
fabric roles match. With `numa_node` already resolved for the worker:

```bash
case "${numa_node}" in
    0)
        export UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,eth0
        ;;
    1)
        export UCX_NET_DEVICES=mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,eth0
        ;;
    *)
        echo "No NIC mapping for NUMA node ${numa_node}" >&2
        exit 1
        ;;
esac
```

Replace every name with a device available to that worker, excluding storage and management HCAs.
`mlx5_*:1` selects RDMA devices and ports; `eth0` illustrates a reachable TCP interface shared by
both lists. Do not overwrite a device list supplied by the recipe or device allocator.

Listing four HCAs makes them eligible; it does not create four-rail transfers. UCX `1.21` allows
up to two rendezvous rails by default, controlled by `UCX_MAX_RNDV_RAILS`. NIXL's UCX backend uses RMA
operations, whose rail limit is `UCX_MAX_RMA_RAILS`; NIXL `1.5.0` sets that limit to two unless
overridden. Tune the relevant limit for the actual transfer path and measure the result. See
[UCX multi-rail selection](https://github.com/openucx/ucx/blob/v1.21.0/docs/source/faq.md#multi-rail)
and [NIXL's UCX defaults](https://github.com/ai-dynamo/nixl/blob/v1.5.0/src/plugins/ucx/ucx_utils.cpp#L484).

To use reliable-connected (RC) RDMA with CUDA IPC and copy, host shared memory, loopback, and TCP:

```bash
export UCX_TLS=rc,cuda_ipc,cuda_copy,sm,self,tcp
```

TCP needs both the `tcp` transport and an interface in `UCX_NET_DEVICES`. Some deployments use
it for control, active-message (AM), or fallback traffic. AM can also run over RDMA. Since this
configuration also permits payload traffic over TCP, check that KV transfers use RC/RDMA as intended.
Preserve recipes that deliberately omit `tcp` to fail when RDMA is unavailable instead of falling
back; the transport list above is an optional choice for deployments that need TCP.

`rc` includes the required bootstrap transports; `cuda_ipc` and `cuda_copy` enable CUDA IPC and
copy paths, `sm` enables host shared memory, and `self` enables same-process communication.
RC works with compatible InfiniBand or RDMA over Converged Ethernet (RoCE) fabrics. Leave
`UCX_TLS` unset if you want UCX to choose transports automatically. The fabric must already be
configured for RDMA. See the
[UCX device and transport controls](https://github.com/openucx/ucx/blob/v1.21.0/docs/source/faq.md#which-transports-does-ucx-use).

Set NIC selection on both prefill and decode workers. Every rank inherits a container-wide
`UCX_NET_DEVICES` value. For a group spanning domains, set it per rank, use separate groups within
each domain, or give UCX a list such as `mlx5_0:1,mlx5_6:1`. Forcing all ranks through one NIC
can lose locality and multi-rail bandwidth.

vLLM `0.32.0` can set the NIC for each worker. Set
`VLLM_GPU_NIC_PCIE_MAPPING` to comma-separated `GPU_BDF=NIC_BDF` pairs and set
`VLLM_NIC_SELECTION_VARS=UCX_NET_DEVICES:1` alongside it. For example, for two GPUs whose inspected
PCI addresses map to the following NICs:

```bash
unset UCX_NET_DEVICES
export VLLM_GPU_NIC_PCIE_MAPPING="0000:18:00.0=0000:19:00.0,0000:3b:00.0=0000:3c:00.0"
export VLLM_NIC_SELECTION_VARS=UCX_NET_DEVICES:1
```

Replace the PCI addresses and cover every GPU assigned to the workers on that host. Set both
variables together on prefill and decode; setting only one fails initialization. The NICs must
be visible to the worker. See the
[DeepSeek-V4 per-rank mapping procedure](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4/README.md#per-rank-nic-mapping-b200--h200-disaggregated).

Do not combine this with the whole-domain HCA list above. vLLM prepends each rank's selected NIC
to an existing `UCX_NET_DEVICES` value, and UCX ignores list order: the other HCAs remain eligible.
Start with the variable unset for one-HCA-per-rank selection. If TCP is also required, retaining
only the intended TCP interface, such as `eth0`, does not add other HCAs; inspect the final
per-rank list in the logs. See the
[vLLM mapping implementation](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/executor/vllm_net_devices.py#L174)
and [UCX device-list semantics](https://github.com/openucx/ucx/blob/v1.21.0/src/ucp/core/ucp_context.c#L621).

A NIXL UCX backend `device_list` can override `UCX_NET_DEVICES`, so check the connector's config.
`NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` only configure NCCL. For a libfabric or other non-UCX
KV-transfer backend, use that backend's controls. See the
[NIXL device override](https://github.com/ai-dynamo/nixl/blob/v1.5.0/src/plugins/ucx/ucx_utils.cpp#L480)
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

With libnuma `2.0.19` or later, a denied `get_mempolicy` call can appear as "No NUMA available on
this system" or "This system does not support NUMA policy", even on a NUMA host. Node-based
CPU-only binding with `--cpunodebind` can fail too. For CPU affinity without memory-policy calls,
use `taskset -c CPU_LIST` with explicit allowed CPU IDs.

After startup, use `ps` to find the frontend and each GPU worker or scheduler. Set `WORKER_PID`
to a PID from the listing, then run the remaining checks for that process:

```bash
ps -eo pid,ppid,comm,args
taskset -apc "${WORKER_PID}"
grep -E 'Cpus_allowed_list|Mems_allowed_list' "/proc/${WORKER_PID}/status"
cat "/proc/${WORKER_PID}/numa_maps"
numastat -p "${WORKER_PID}"
```

`taskset -a` checks every thread; engines can give threads different CPU masks during startup.
The status file above describes the main thread. `Mems_allowed_list` shows permitted memory
nodes, not a binding policy. Inspect the full `numa_maps` for policy and page placement, including
high-address host buffers, and use `numastat -p` for totals by node. Check under load after engine
initialization. For UCX, temporarily set `UCX_LOG_LEVEL=info` and check the HCA, TCP interface,
and KV-payload transport.

Compare TTFT, ITL, throughput, memory placement, and KV-transfer performance with the same
workload before and after binding. If the workload offloads KV cache, also measure offload and
onboard latency and bandwidth so you can tell whether host-memory placement improved that path.
