---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Per-GPU Compute and Memory Limits with MPS
subtitle: Give a multi-GPU application a fixed share of every GPU, and pack smaller models into the rest
---

A large model that spans several GPUs often leaves part of each GPU idle. This guide shows how to cap that model at a fixed share of every GPU it uses and run smaller models, such as embedding or BERT-class servers, in the remainder. It uses CUDA Multi-Process Service (MPS) v3 for compute limits and the cgroup v2 `dmem` controller for memory limits.

You'll set up:

- The **same** compute and memory limit on every GPU an application sees, from one MPS server configuration file.
- A **different** memory limit on each GPU for the same container, through `nvidia-smi memory-limits`.

Every procedure on this page was run on a single-GPU host; the measured results are in [Verified Behavior](#verified-behavior). On a multi-GPU node, the only change is the list of GPU UUIDs or indices you pass.

## Requirements

| Requirement | Needed for | Check |
|---|---|---|
| NVIDIA driver r615 or newer (CUDA 13.4) | MPS v3: servers, namespaces, config file | `nvidia-cuda-mps-control -p 3 -h` lists `server` and `namespace` |
| Linux with cgroup v2 at `/sys/fs/cgroup` | Per-GPU memory limits | `stat -fc %T /sys/fs/cgroup` prints `cgroup2fs` |
| Kernel 6.14 or newer with the `dmem` controller | Kernel-enforced memory limits (the driver falls back to an internal `misc` backend on older kernels) | `cat /sys/fs/cgroup/dmem.capacity` lists one `nvidia/<pci>/vidmem` line per GPU |
| Root | Setting memory limits, and multi-user MPS. A single-user MPS daemon, its servers, namespaces, and clients run unprivileged | `sudo -v` |
| MIG disabled | Both features | `nvidia-smi --query-gpu=mig.mode.current --format=csv` |

Reference documentation: [MPS v3 Interface](https://docs.nvidia.com/deploy/mps/latest/mpsv3-interface.html), [MPS v3 Configuration File](https://docs.nvidia.com/deploy/mps/latest/mpsv3-configuration-file.html), and [MPS v3 Memory Partitioning](https://docs.nvidia.com/deploy/mps/latest/mpsv3-memory-partitioning.html).

## How the Pieces Fit

```text
MPS control daemon  (nvidia-cuda-mps-control -d -p 3 -a mps.toml)
└── server "shared"       allowed_devices = every GPU UUID it should own
    ├── namespace default     server-level limits (clients of <pipe>/shared)
    ├── namespace large       70% of SMs, pinned memory cap   <- large-model ranks
    └── namespace small       30% of SMs, pinned memory cap   <- small model servers

cgroup v2 dmem              per container, per GPU:   nvidia/<pci>/vidmem <bytes>
```

- **Compute** is capped by the namespace's `active_thread_percentage`. It applies to every GPU the client opens.
- **Memory** is capped in one of two ways:
  - The namespace's `pinned_memory_limit` is one value for every GPU.
  - A cgroup `dmem` limit is set separately for each GPU, and is the only way to give one container a different budget on each GPU.
- **Memory bandwidth is not partitioned.** Neither MPS nor `dmem` splits DRAM bandwidth. A memory-bound tenant slows its neighbors no matter how SMs are divided. [Interpreting GPU Sharing Results](gpu-sharing-results.mdx) shows the effect on decode throughput.

## Same Limits on Every GPU

### 1. List the GPUs

Use UUIDs everywhere. Ordinals change meaning when `CUDA_VISIBLE_DEVICES` is set.

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.total --format=csv,noheader
```

### 2. Write the server configuration

Create `mps.toml`. Put every GPU the server should own in `allowed_devices`, comma-separated:

```toml
schema_version = "1.0"

[servers.shared]
allowed_devices          = "GPU-<uuid-0>,GPU-<uuid-1>"
pinned_memory_limit      = "10G"
active_thread_percentage = "50"

[servers.shared.namespaces.large]
pinned_memory_limit      = "56G"
active_thread_percentage = "70"

[servers.shared.namespaces.small]
pinned_memory_limit      = "20G"
active_thread_percentage = "30"
```

- Limits apply **per GPU**. A client of `large` can use 70% of the SMs and 56 GiB on each GPU it opens; the check in step 6 confirms this on every GPU the client sees. Memory sizes use binary units: `10G` is 10,737,418,240 bytes.
- The `[servers.shared]` limits apply **only to the default namespace**. A named namespace does not inherit them, and it is not capped by them:
  - a namespace that sets no limits gets the whole GPU;
  - a namespace can set values above the server-level limits.

  Set both `active_thread_percentage` and `pinned_memory_limit` on every namespace you create.
- `active_thread_percentage` is a ceiling, not a reservation. Two namespaces whose percentages add up to more than 100 share SMs.

### 3. Start the daemon

Use a node-local pipe directory, such as one on the `/dev/shm` tmpfs. A network filesystem can't host the daemon's sockets.

```bash
export CUDA_MPS_PIPE_DIRECTORY=/dev/shm/mps/pipe CUDA_MPS_LOG_DIRECTORY=/dev/shm/mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d -p 3 -a mps.toml
```

The configuration is applied once at startup. Changes made later with the CLI are not written back to the file.

### 4. Confirm the server came up

```bash
nvidia-cuda-mps-control -p 3 server list
nvidia-cuda-mps-control -p 3 namespace list --server=shared
grep -i error "$CUDA_MPS_LOG_DIRECTORY"/shared/server.log
```

> [!WARNING]
> A server whose `allowed_devices` names a GPU that isn't on the node is accepted (`server create` returns success), but the server process exits with `no CUDA-capable device is detected` and disappears from `server list`. Any client pointed at it then runs **without MPS and without limits**. Always check `server list` and `server.log` after a start.

`namespace list` shows the limits each namespace enforces:

```text
NAME             SERVER        SERVER STATUS  PINNED MEMORY LIMIT   ACTIVE THREAD PERCENTAGE
small            shared        Ready          20G                   30.00
large            shared        Ready          56G                   70.00
default          shared        Ready          10G                   50.00
```

### 5. Run clients in a namespace

A client selects its namespace by pointing `CUDA_MPS_PIPE_DIRECTORY` at that namespace's directory:

```bash
CUDA_MPS_PIPE_DIRECTORY=/dev/shm/mps/pipe/shared/large python3 serve_large_model.py
```

For a container, mount only the namespace directory, pass the same path, and run as the user that owns the MPS daemon. A container user that differs from the daemon's owner, including root, gets `CUDA_ERROR_NOT_PERMITTED`. To serve clients running as other users, start the daemon as root in multi-user mode (`--multiuser-server`; see `nvidia-cuda-mps-control -h`).

```bash
docker run --rm --gpus '"device=GPU-<uuid-0>,GPU-<uuid-1>"' --user "$(id -u):$(id -g)" \
  -v /dev/shm/mps/pipe/shared/small:/dev/shm/mps/pipe/shared/small \
  -e CUDA_MPS_PIPE_DIRECTORY=/dev/shm/mps/pipe/shared/small \
  <image> python3 serve_small_model.py
```

> [!CAUTION]
> In the default compute mode, a client with a missing or misspelled pipe directory runs as a normal CUDA process with the whole GPU, and nothing reports an error. Verify every new deployment with the check below.

### 6. Verify what a client gets

Run this inside the client environment, with the same `CUDA_MPS_PIPE_DIRECTORY` and GPU selection:

```python
import torch

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    held = []
    try:
        while True:
            held.append(torch.empty(256 << 20, dtype=torch.uint8, device=i))
    except torch.OutOfMemoryError:
        pass
    print(f"GPU {i} {p.uuid}: SMs={p.multi_processor_count} "
          f"reported total={total >> 20} MiB, allocatable={256 * len(held)} MiB")
    del held
    torch.cuda.empty_cache()
```

Expect the SM count to be the namespace percentage of the GPU's SMs, rounded down to a supported count. On a 188-SM GPU, 50% gives 94 and 25% gives 46. Expect the allocatable memory to stop within a few hundred MiB of the pinned limit. That headroom is the client's own context.

> [!IMPORTANT]
> Under an MPS pinned memory limit, `cudaMemGetInfo` reports the limit as *free* memory but the **whole GPU** as *total*. Frameworks that size memory as a fraction of total, such as vLLM's `--gpu-memory-utilization`, must be given the fraction explicitly: for a 56 GiB cap on an 80 GiB GPU, pass `0.65` or less, not `0.9`. A `dmem` limit (next section) is the only cap that `cudaMemGetInfo` reports as total.

## Different Memory Limits per GPU

The namespace `pinned_memory_limit` is one value for every GPU. To give one container 40 GiB on GPU 0 and 56 GiB on GPU 1, set a cgroup `dmem` limit per GPU. `dmem` limits apply to MPS clients too, so you can combine a namespace for the compute limit with `dmem` for per-GPU memory.

### 1. Confirm the backend and find each GPU's key

```bash
cat /sys/fs/cgroup/dmem.capacity
```

```text
nvidia/00000000:02:00.0/vidmem 102641958912
```

Each GPU has its own line. `nvidia-smi memory-limits` resolves the key from `-i`, so you don't write these files yourself. Set limits only through `nvidia-smi` or NVML; writing `dmem.*` files directly skips the reservation bookkeeping the driver maintains on parent cgroups.

### 2. Know the command syntax

```bash
sudo nvidia-smi memory-limits -i <gpu index or UUID> -n /sys/fs/cgroup/<cgroup path> \
  --soft-limit <MiB> --hard-limit <MiB>
nvidia-smi memory-limits --get -i <gpu> -n /sys/fs/cgroup/<cgroup path>
```

- `memory-limits` must come **first**. `nvidia-smi -i 0 memory-limits ...` fails with `Option memory-limits is not recognized`.
- Values are **MiB integers**, `max`, or `default`. `--soft-limit=40G` is rejected.
- `-n` takes the **full path** of the cgroup directory.
- The **hard limit** is enforced at allocation time; an allocation past it returns `CUDA_ERROR_OUT_OF_MEMORY`.
- The **soft limit** is a guaranteed share. Above it, a process is borrowing, and MPS can evict it when another tenant within its own soft limit needs the memory. Set soft equal to hard for a strict partition.
- Reading limits needs no root. Setting them does.

### 3. Pick where to apply the limit

Docker puts each container in its own leaf cgroup. With the default systemd cgroup driver, `--cgroup-parent` must name a slice (`bert0.slice`), not a path (`/bert0`):

```text
docker: Error response from daemon: cgroup-parent for systemd cgroup should be a valid slice named as "xxx.slice"
```

The container's leaf cgroup gets its own `dmem.max` of `max`. That changes what the container sees depending on where you put the limit:

| Limit set on | Enforced | `cudaMemGetInfo` total inside the container | Use when |
|---|---|---|---|
| The container's leaf cgroup | Yes | The limit | The application sizes itself from GPU memory (vLLM, SGLang, TensorRT-LLM) |
| A parent slice, one per container | Yes, through the parent | **The whole GPU** | You can't act between container start and workload start, and you size the application explicitly |

A parent slice shared by several containers gives them one **shared** budget. Use one slice per container.

### 4. Option A: limit the container's own cgroup (recommended)

1. Start the container with an idle command so its cgroup exists before the workload does:

    ```bash
    docker run -d --name bert0 --gpus '"device=GPU-<uuid-0>,GPU-<uuid-1>"' \
      --cgroup-parent=bert0.slice --user "$(id -u):$(id -g)" \
      -v /dev/shm/mps/pipe/shared/small:/dev/shm/mps/pipe/shared/small \
      <image> sleep infinity
    ```

2. Find the container's cgroup:

    ```bash
    PID=$(docker inspect -f '{{.State.Pid}}' bert0)
    CG=/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/$PID/cgroup)
    echo "$CG"
    ```

    ```text
    /sys/fs/cgroup/bert0.slice/docker-<container-id>.scope
    ```

3. Set a different limit on each GPU, one call per GPU:

    ```bash
    sudo nvidia-smi memory-limits -i GPU-<uuid-0> -n "$CG" --soft-limit 40960 --hard-limit 40960
    sudo nvidia-smi memory-limits -i GPU-<uuid-1> -n "$CG" --soft-limit 57344 --hard-limit 57344
    cat "$CG/dmem.max"
    ```

    On a two-GPU node, `dmem.max` holds one line per GPU:

    ```text
    nvidia/00000000:1b:00.0/vidmem 42949672960
    nvidia/00000000:43:00.0/vidmem 60129542144
    ```

4. Start the workload in the container. It sees each GPU's limit as that GPU's total memory:

    ```bash
    docker exec -e CUDA_MPS_PIPE_DIRECTORY=/dev/shm/mps/pipe/shared/small bert0 python3 serve_small_model.py
    ```

To start the workload from the container's own entrypoint instead of `docker exec`, have the entrypoint wait for a signal, such as a file your launcher creates after step 3.

### 5. Option B: limit a per-container parent slice

Option B lets the workload start as soon as its container starts, with no wait for a limit, at the cost of the container reporting the full GPU.

1. Create the slice by starting an idle holder container in it. The holder needs no GPU, and it keeps the slice in place until the workload joins:

    ```bash
    docker run -d --name bert1-hold --cgroup-parent=bert1.slice <image> sleep infinity
    ```

2. Set the limit on the slice, one call per GPU:

    ```bash
    sudo nvidia-smi memory-limits -i GPU-<uuid-0> -n /sys/fs/cgroup/bert1.slice --soft-limit 40960 --hard-limit 40960
    sudo nvidia-smi memory-limits -i GPU-<uuid-1> -n /sys/fs/cgroup/bert1.slice --soft-limit 57344 --hard-limit 57344
    ```

3. Start the workload in the same slice, with its memory budget set explicitly:

    ```bash
    docker run -d --name bert1 --gpus '"device=GPU-<uuid-0>,GPU-<uuid-1>"' --cgroup-parent=bert1.slice \
      --user "$(id -u):$(id -g)" <image> python3 serve_small_model.py
    ```

Allocations stop at the slice's limit. `cudaMemGetInfo` inside the container still reports the whole GPU, so configure the application's memory budget yourself. For vLLM, set `--gpu-memory-utilization` below the limit divided by the GPU's total memory.

### 6. Remove limits before removing the cgroup

The driver adds each soft limit to the parent cgroups' reservations. Deleting a cgroup without zeroing its soft limit leaves that reservation on the parents. Before you stop a container or delete a slice, reset the limits on the cgroup you set them on: the container's cgroup for Option A, or the slice for Option B.

```bash
sudo nvidia-smi memory-limits -i GPU-<uuid-0> -n "$CG" --soft-limit 0 --hard-limit max
sudo nvidia-smi memory-limits -i GPU-<uuid-1> -n "$CG" --soft-limit 0 --hard-limit max
docker rm -f bert0
```

## Client-Side Per-GPU Limits

A client can lower its own pinned memory limit per GPU with `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`, keyed by ordinal or UUID:

```bash
CUDA_MPS_PIPE_DIRECTORY=/dev/shm/mps/pipe/shared/small \
CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="GPU-<uuid-0>=2G,GPU-<uuid-1>=4G" python3 serve_small_model.py
```

The variable can only lower the namespace's limit; `0=20G` under a 10 GiB namespace still allocates at most 10 GiB.

> [!WARNING]
> If any entry names a GPU the process can't see, the client drops **every** pinned memory limit, including the namespace's. Examples are `1=4G` in a process with one visible GPU, or a UUID that isn't present. Its compute limit still applies. Tensor-parallel launchers often narrow `CUDA_VISIBLE_DEVICES` per worker, which turns a correct node-wide list into an unlimited one. List only the GPUs the process sees, prefer UUID keys, and verify the result. A tenant can also escape its memory limit this way, so use `dmem` limits where enforcement matters.

## Example: One Large Model Plus Small Models on Eight GPUs

To keep a tensor-parallel model at 70% of each of eight 80 GiB GPUs and pack small model servers into the rest:

1. Create one server that owns all eight GPUs, a `large` namespace at `active_thread_percentage = "70"` and `pinned_memory_limit = "56G"`, and a `small` namespace at `"30"` and `"24G"`. Set the default namespace's server-level limits low, so any client that connects without a namespace can't take a whole GPU.
2. Point every rank of the large model at `<pipe>/shared/large`.
3. Run one small-model container per GPU, each restricted to its GPU with `--gpus '"device=GPU-<uuid-i>"'` and pointed at `<pipe>/shared/small`.
4. Where small-model containers need different memory on different GPUs, give each container a `dmem` limit per GPU (Option A).

Expect the large model's memory-bound decode to slow when the small models are busy. They share DRAM bandwidth even though their SMs are capped.

## Verified Behavior

Checked on one NVIDIA RTX PRO 6000 Blackwell (188 SMs, 96 GB) with driver 615.71.09, CUDA 13.4, Linux 7.0, systemd 255, and Docker 29.8 (systemd cgroup driver). Every case was run with exclusive use of the GPU and an idle-GPU check before each run. The `dmem` cases ran twice with identical results. Multi-GPU behavior follows from the per-GPU keys shown above; it was not run on multi-GPU hardware.

| Setup | SMs seen | Allocatable memory | `cudaMemGetInfo` total |
|---|---|---|---|
| Default namespace, server-level 50% and `10G` | 94 | 9,728 MiB | whole GPU |
| Namespace 25% and `5G`, host or container | 46 | 4,864 MiB | whole GPU |
| Named namespace with no limits | 188 | whole GPU | whole GPU |
| Named namespace 75% and `20G` under server-level 50% and `10G` | 140 | 19,968 MiB | whole GPU |
| Client `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=2G` or `GPU-<uuid>=2G` | 94 | 1,536 MiB | whole GPU |
| Client `0=2G,1=4G` with one visible GPU | 94 | **96,768 MiB (no limit)** | whole GPU |
| `dmem` 30,000 MiB on the container's own cgroup | 188 | 29,184 MiB | **30,000 MiB** |
| `dmem` 48,943 MiB on a parent slice only | 188 | 48,128 MiB | **whole GPU** |
| MPS namespace 50% with no pinned limit, plus `dmem` 3,000 MiB on the container | 94 | 2,560 MiB | **3,000 MiB** |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Client sees all SMs and all memory | Wrong or missing `CUDA_MPS_PIPE_DIRECTORY`, or the server isn't running | Check `server list`; point the client at `<pipe>/<server>/<namespace>` |
| Server missing from `server list` after start | A UUID in `allowed_devices` isn't on the node | Fix the UUID; read `<log dir>/<server>/server.log` |
| `CUDA_ERROR_NOT_PERMITTED` in a container | Container user differs from the daemon owner | Run the container as the daemon's user, or run the daemon as root in multi-user mode |
| A named namespace is unlimited | Server-level limits only apply to the default namespace | Set limits on each namespace |
| Memory limit ignored with `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` set | An entry names a GPU the process can't see | List only visible GPUs, by UUID |
| `Unable to get memory limit: Not Supported` | The cgroup has no `dmem` files, such as the root cgroup | Use a cgroup whose parent enables `+dmem` in `cgroup.subtree_control` |
| `--soft-limit must be a non-negative MebiByte value` | A unit suffix was used | Pass MiB as an integer |
| Container reports the full GPU despite a `dmem` limit | The limit is on a parent; the container's leaf has `dmem.max` = `max` | Set the limit on the leaf (Option A) or size the application explicitly |
| vLLM fails to allocate KV cache under MPS | `--gpu-memory-utilization` is applied to the physical total, not the pinned limit | Lower `--gpu-memory-utilization` to fit the limit, or use a `dmem` leaf limit |
