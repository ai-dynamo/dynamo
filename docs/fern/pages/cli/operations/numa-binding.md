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

At high concurrency, expect roughly 10% higher throughput when NUMA locality is a bottleneck.

Add these settings to a working deployment, keeping its model, discovery, and KV-transfer options.

## Check the Tools in Your Image

Check `command -v numactl` in the Linux environment that will run each component. The SGLang
CUDA runtime and dedicated frontend image definitions covered here lack the executable;
vLLM and TensorRT-LLM runtime images include it. If it is missing, install the binding and
diagnostic packages during an Ubuntu image build:

```bash
apt-get update
apt-get install -y --no-install-recommends numactl libnuma1 iproute2 util-linux procps
rm -rf /var/lib/apt/lists/*
```

These packages provide `numactl`, `numastat`, `ip`, `lscpu`, `taskset`, and `ps`. UCX libraries
alone do not provide the optional `ucx_info` diagnostic.

## Choose CPU and Memory Placement

Inside the worker's container or scheduler allocation, find the CPUs and host-memory nodes
closest to each GPU:

```bash
numactl --hardware
lscpu --extended=CPU,NODE,SOCKET,CORE
nvidia-smi topo -m
nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv
```

Match GPU UUIDs or PCI addresses to the engine's `CUDA_VISIBLE_DEVICES` order before building
per-GPU mappings. `nvidia-smi -i` uses physical NVML indices, UUIDs, or PCI addresses, not positions
in that CUDA list. GPU IDs need not follow socket order; a socket can have several NUMA nodes.
For physical GPU `0`, `nvidia-smi topo -C -i 0` reports the nearest CPU node and
`nvidia-smi topo -M -i 0` reports the nearest memory node. Use a CPU-bearing node for CPU binding;
GPU HBM may appear as a memory-only node.

Apply the selected placement with `numactl` before starting the process:

| Option | Effect |
|---|---|
| `--cpunodebind=N` | Restrict CPU execution to cores belonging to NUMA node `N`. |
| `--physcpubind=CPU_LIST` | Restrict execution to specific Linux CPU IDs. |
| `--membind=N` | Allocate host memory only on node `N`; exhaustion can cause allocation failure or an out-of-memory (OOM) kill. |
| `--preferred=N` | Prefer host memory on node `N`, with fallback to other permitted nodes. |

Children inherit the policy, although engines can rebind during startup. Existing pages and
shared buffers can retain their previous placement. See the
[numactl manual](https://man7.org/linux/man-pages/man8/numactl.8.html),
[nvidia-smi Topology section](https://docs.nvidia.com/deploy/nvidia-smi/index.html), and
[hardware-coherent memory guide](https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms/).

> [!IMPORTANT]
> Apply binding to the process or rank that owns each GPU. Wrapping a whole tensor parallel (TP)
> group in a single-node `numactl` policy can put every rank on the same socket, even when its GPUs
> span several domains. Use the engine's per-worker binding or a rank-aware launcher for those groups.

## Bind the Frontend

Start with the NIC carrying client or load-balancer traffic. Run `ip -br address` in the
frontend's network namespace, identify that NIC, and set `FRONTEND_NIC` to its name. Read its
NUMA node:

```bash
cat "/sys/class/net/${FRONTEND_NIC:?Set FRONTEND_NIC}/device/numa_node"
```

For a bond, VLAN, or virtual interface, trace its physical NIC. A missing file or `-1` means
locality is unknown; inspect the host topology before choosing a node. Set `FRONTEND_NUMA_NODE`
to the known CPU-bearing node, then bind the frontend's CPU and memory there:

```bash
numactl --cpunodebind="${FRONTEND_NUMA_NODE}" --preferred="${FRONTEND_NUMA_NODE}" \
  python3 -m dynamo.frontend
```

Keep the default `--http-host 0.0.0.0`, which accepts all local IPv4 addresses. Use `--membind`
instead of `--preferred` for strict host-memory placement, or `--physcpubind=CPU_LIST` to select
part of the node's CPUs. `taskset -c CPU_LIST` provides CPU-only placement without `numactl`.
Dynamo's [startup affinity check](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/frontend/cpu_affinity.py)
warns about multiple CPU nodes; it does not apply binding.

### Optionally Select a NUMA-Local Listen Address

If clients can reach NICs on several NUMA nodes, select the address on the node chosen above
with `--http-host IP_ADDRESS` or `DYN_HTTP_HOST`. A specific listen IP disables `localhost`
access unless it is the loopback address; point clients and health checks at the selected IP.
Address selection does not bind a physical NIC or change routing and interrupt placement. See
[Frontend HTTP and Networking](../../reference/components/frontend-configuration.mdx#http-and-networking).

## Bind vLLM Workers

Dynamo accepts vLLM's native engine arguments. In vLLM `0.32.0`, binding is off by default;
`--numa-bind` binds EngineCore and multiprocessing GPU workers:

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 4 \
  --numa-bind
```

**CPU:** Automatic discovery requires at least two NUMA nodes, unrestricted CPU affinity, and
permission to call `get_mempolicy`. For Slurm tasks or Docker `--cpuset-cpus`, add an explicit
mapping: `--numa-bind-nodes 0 0 1 1` assigns GPUs `0,1` to node `0` and GPUs `2,3` to node `1`.
Without explicit nodes, failed discovery aborts startup.

Workers normally use `--cpunodebind`; to select CPUs within each domain, add one list per visible
GPU, such as `--numa-bind-cpus 0-3 4-7 48-51 52-55`. Use CPUs allowed by your allocation.
EngineCore binds to its GPU shard's domains and ignores these per-worker lists. With the default
single-GPU `uni` executor, the model runs inside EngineCore, so the lists do not narrow its affinity.

On detected Priority Core Turbo (PCT) Xeons, such as the 6776P in DGX B300, vLLM selects priority
cores with `--physcpubind` for workers and EngineCore when no CPU lists are supplied. Those cores
must fit the allowed CPU set; rejected binding can leave the process without NUMA binding.
The `dynamo.vllm` parent hosting AsyncLLM, data-parallel coordinator, and Dynamo frontend remain unbound.
If you bind the parent with `numactl` or `taskset`, pass `--numa-bind-nodes` and allow every rank's
target CPUs and memory nodes.

**Memory:** vLLM requests strict `--membind` on the selected domains. Once node IDs are known,
policy probes can fall back to CPU-only binding or no binding; check logs and effective placement.
There is no engine option for preferred memory in this version; `--preferred` belongs to `numactl`.

**NIC:** Follow [Select Local NICs for RDMA KV Transfer](#select-local-nics-for-rdma-kv-transfer)
for vLLM's per-rank GPU-to-NIC mapping. CPU binding does not select the KV-transfer NIC.

See the
[upstream NUMA guide](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/configuration/optimization.md#numa-binding-for-multi-socket-gpu-nodes)
and [binding implementation](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/utils/numa_utils.py).

## Bind SGLang Workers

Dynamo accepts SGLang's native `ServerArgs`. SGLang `0.5.22` enables automatic discovery and V2
binding by default, but the current CUDA runtime image lacks the required `numactl` binary.
Check `command -v numactl` and install it in a derived image; otherwise automatic V2 binding
silently skips with a DEBUG-level message. With the binary installed:

```bash
SGLANG_AUTO_NUMA_BIND=1 \
SGLANG_NUMA_BIND_V2=1 \
SGLANG_SET_CPU_AFFINITY=0 \
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-8B \
  --tp-size 4
```

**CPU:** V2 binds each scheduler before its Python interpreter starts. It uses `--cpunodebind`,
or `--physcpubind` for the allowed subset of CPUs on that node. For the four-GPU mapping above,
add `--numa-node 0 0 1 1`, or `numa_node: [0, 0, 1, 1]` in a native config file. The index is the
scheduler's local GPU ID; account for device visibility and `--base-gpu-id`.
Leave `SGLANG_SET_CPU_AFFINITY=0`; its separate CPU partitioning can conflict with NUMA binding.

**Memory:** V2 tries strict `--membind`, then `--preferred`, then CPU-only binding. Automatic
discovery can skip binding when NUMA support or memory-policy permissions are missing.
Set `SGLANG_NUMA_BIND_V2=0` for legacy in-process CPU affinity and preferred memory through
`libnuma`, without the binary. To disable automatic binding, set `SGLANG_AUTO_NUMA_BIND=0` and
omit the explicit node mapping.

If binding fails entirely, SGLang warns and skips it by default. `SGLANG_CRASH_ON_NUMA_BIND_FAILURE=1`
raises on binding failures, but accepts a successful weaker policy and does not catch discovery skips.
An explicit V2 node mapping forces an attempt. Check `Mems_allowed_list` and `numa_maps` after
startup; the flag does not guarantee strict memory placement.

**NIC:** Follow [Select Local NICs for RDMA KV Transfer](#select-local-nics-for-rdma-kv-transfer)
for per-GPU Mooncake device mapping or NIXL/UCX selection, according to the transfer backend.

See SGLang's [environment defaults](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/environ.py#L1565),
[binding implementation](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/utils/numa_utils.py),
and [legacy scheduler path](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/scheduler.py#L6021).

## Bind TensorRT-LLM Workers

### CPU Affinity

With an unrestricted inherited CPU mask, TensorRT-LLM selects GPU-local CPUs by default. To select
native affinity when the inherited mask is restricted, set `TLLM_NUMA_AWARE_WORKER_AFFINITY=1`;
it replaces that mask with the GPU's NVML CPU mask. Use `0` to preserve a chosen launch mask.
Leaving the variable unset with a restricted mask can instead reset affinity to all allowed CPUs.
These semantics apply to TensorRT-LLM `1.3.0rc29` and `1.4.0rc0`; see
[`configure_cpu_affinity`](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/tensorrt_llm/llmapi/utils.py#L713).

### Host-Memory Placement

Native CPU affinity does not set a memory policy. The normal `python3 -m dynamo.trtllm` launch
spawns MPI workers internally with one inherited memory policy and no per-rank `numactl` hook.
For different policies per rank, use an external launcher with the wrapper **before**
[`trtllm-llmapi-launch`](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/tensorrt_llm/llmapi/trtllm-llmapi-launch).
Calling `mpirun` directly on `python3 -m dynamo.trtllm` does not select that launch mode.

For one host with four allocated GPUs, export their **GPU UUIDs** as `CUDA_VISIBLE_DEVICES` in
the same order for every rank. Save this wrapper as `trtllm-numa-rank.sh`:

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

export TLLM_NUMA_AWARE_WORKER_AFFINITY=1
exec numactl --preferred="${numa_node}" "$@"
```

The wrapper follows TensorRT-LLM's [global-rank device mapping](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/tensorrt_llm/executor/base_worker.py#L149)
and normalizes `00000000:9D:00.0` to `0000:9d:00.0`. Unknown or CPU-less NUMA nodes stop the launch;
supply an inspected CPU-node mapping on those systems. Launch with Open MPI rebinding disabled:

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

Replace `--preferred` with `--membind` for strict local memory. Check that the allocation permits
the selected nodes and verify every rank's placement after startup. For multiple hosts, validate
the global-rank modulo mapping and visible-device order on each host.

### Optional Early CPU Binding

To also place CPU work before Python starts, replace the wrapper's final two lines with:

```bash
export TLLM_NUMA_AWARE_WORKER_AFFINITY=0
exec numactl --cpunodebind="${numa_node}" --preferred="${numa_node}" "$@"
```

Keep the Open MPI settings above so MPI does not replace the launch mask. Native affinity in
`1.4.0rc0` covers existing threads on a best-effort basis during initialization; early binding
also covers startup work. Compare performance and thread masks rather than assuming the two
paths are equivalent. For CPU-only placement when libnuma calls are blocked, use `taskset -c`
with allowed CPU IDs instead. See the [Open MPI affinity settings](https://github.com/NVIDIA/TensorRT-LLM/blob/cc63e66daf52644f1441e117bb7ba67d20566016/docs/source/deployment-guide/configuring-cpu-affinity.md#openmpi).

### NIC Locality

Select the KV-transfer NIC separately using [Local NICs for RDMA KV Transfer](#select-local-nics-for-rdma-kv-transfer).

## Select Local NICs for RDMA KV Transfer

For KV transfer over remote direct memory access (RDMA), choose network interface cards (NICs)
close to each worker's GPU and NUMA domain. Check the GPU-to-NIC PCIe path too; see the
[UCX GPU RDMA guidance](https://github.com/openucx/ucx/blob/v1.21.0/docs/source/faq.md#does-ucx-support-zero-copy-for-gpu-memory-over-rdma).

Inspect locality and map RDMA devices to Linux interfaces through sysfs. Replace `mlx5_0` with
an RDMA device available to the worker:

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

Confirm which host channel adapters (HCAs) serve the compute fabric; proximity alone does not
distinguish compute, storage, and management networks. Unified Communication X (UCX) already
selects devices using performance and NUMA locality. Keep a recipe's automatic or injected
selection unless logs or measurements justify restricting `UCX_NET_DEVICES`. If installed,
`ucx_info -d` lists available devices; `UCX_LOG_LEVEL=info` shows the selected path.

For a worker confined to one domain, set `UCX_NET_DEVICES` before startup to the corresponding
RDMA device/port list. This illustrative split applies only if the inspected topology and
fabric roles match; replace every name and exclude storage and management HCAs:

| Worker NUMA node | Example `UCX_NET_DEVICES` |
| --- | --- |
| `0` | `mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1` |
| `1` | `mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1` |

Set NIC selection on both prefill and decode workers. Every rank inherits a container-wide
list. For groups spanning domains, set it per rank or leave the appropriate devices eligible
for UCX to select. A shared list does not guarantee a local NIC for every rank. Preserve
deployment-specific allocation rules such as the
[Solar Open2 networking notes](../../recipes/model-recipes/solar-open2-250b.mdx#configuration-notes) and
[DeepSeek-V4.1 Flash device allocation](../../recipes/model-recipes/deepseek-v4-1-flash.mdx).

vLLM `0.32.0` supports per-worker NIC selection. Set `VLLM_GPU_NIC_PCIE_MAPPING` to comma-separated
`GPU_BDF=NIC_BDF` pairs and `VLLM_NIC_SELECTION_VARS=UCX_NET_DEVICES:1` alongside it. For two GPUs
whose inspected PCI addresses match this example:

```bash
unset UCX_NET_DEVICES
export VLLM_GPU_NIC_PCIE_MAPPING="0000:18:00.0=0000:19:00.0,0000:3b:00.0=0000:3c:00.0"
export VLLM_NIC_SELECTION_VARS=UCX_NET_DEVICES:1
```

Replace the addresses, cover every assigned GPU, and set both variables on prefill and decode;
setting only one fails initialization. See the
[DeepSeek-V4 per-rank mapping procedure](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v4/README.md#per-rank-nic-mapping-b200--h200-disaggregated).

Keep `UCX_NET_DEVICES` unset for one-HCA-per-rank selection. vLLM prepends the selected NIC to any
existing list, and UCX ignores list order, so combining this mapping with the whole-domain list
leaves its other HCAs eligible. See the
[vLLM mapping implementation](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/executor/vllm_net_devices.py#L174)
and [UCX device-list semantics](https://github.com/openucx/ucx/blob/v1.21.0/src/ucp/core/ucp_context.c#L621).

SGLang `0.5.22` defaults to Mooncake. Select its NICs with `--disaggregation-ib-device mlx5_0`,
a shared comma-separated list, or a per-GPU map such as
`--disaggregation-ib-device '{"0":"mlx5_0","1":"mlx5_6"}'`. Replace the mapping and cover every
worker GPU; the keys are GPU IDs, not NUMA nodes. UCX settings apply only when
`--disaggregation-transfer-backend nixl` selects NIXL with UCX. See the
[SGLang transfer options](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/arg_groups/fields/disagg.py#L65-L84).

For NIXL with UCX, a backend `device_list` can override `UCX_NET_DEVICES`; inspect the connector's
config. `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` configure NCCL only. See the
[NIXL device override](https://github.com/ai-dynamo/nixl/blob/v1.5.0/src/plugins/ucx/ucx_utils.cpp#L480).

If a UCX deployment needs TCP as well as RDMA, keep a reachable TCP interface eligible.
For a single rank whose verified local HCA is `mlx5_0`, port `1`, and TCP interface is `eth0`:

```bash
export UCX_NET_DEVICES=mlx5_0:1,eth0
export UCX_TLS=rc,cuda_ipc,cuda_copy,sm,self,tcp
```

Replace both device names for the worker. With vLLM's per-rank mapping, inherit only the TCP
interface (`UCX_NET_DEVICES=eth0`) so vLLM can add each rank's HCA. `UCX_TLS` selects transports,
not placement; this example also permits TCP payload fallback. Check UCX logs for the intended
RDMA path. See [UCX transport controls](https://github.com/openucx/ucx/blob/v1.21.0/docs/source/faq.md#which-transports-does-ucx-use).

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

Compare Time To First Token (TTFT), Inter-Token Latency (ITL), throughput, memory placement, and
KV-transfer performance with the same workload before and after binding. If the workload
offloads KV cache, also measure offload and onboard latency and bandwidth so you can tell
whether host-memory placement improved that path.
