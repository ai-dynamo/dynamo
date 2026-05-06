---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Dynamo Snapshot: Fast Startup for Inference Workloads on Kubernetes"
subtitle: "Schwinn Saereesitthipitak, Dan Feigin, Vikram Sharma Mailthody — May 2026"
description: "Dynamo Snapshot uses CRIU, cuda-checkpoint, and a per-GPU memory service to bring inference worker startup down to 6 seconds or less."
keywords: checkpoint restore, CRIU, cuda-checkpoint, LLM inference, Kubernetes, GPU memory service, GMS, fast startup, autoscaling, cold start, Dynamo
last-updated: May 4, 2026
---

## The Cold-Start Problem
The primary objective in optimizing LLM inference is maximizing tokens per second — because every second a GPU isn't producing tokens directly translates to lost revenue.

Steady-state throughput metrics, while useful, capture only a narrow slice of the system's true behavior. In production inference environments, workloads are inherently dynamic: demand varies over time, model selections evolve, configurations are continuously tuned in pursuit of better efficiency, and failures necessitate periodic restarts. Each of these events disrupts the steady state.

Crucially, the system's responsiveness to these disruptions becomes a first-order concern. Bringing up a new inference instance — even for a single GPU under a fixed model and configuration — can take on the order of minutes. From a systems perspective, this is not merely latency; it is a prolonged interval of resource underutilization. During this time, expensive GPU capacity is allocated but not generating tokens, and therefore not generating revenue.

Here is a breakdown of the cold start time of various models for a single-GPU workload:

![Cold start time breakdown across model sizes on a single B200 GPU.](./figures/cold_start_bench.svg)

In our setup, weights are loaded from a high-bandwidth network-attached storage. For smaller models, the majority of cold-start time is consumed by upstream initialization overheads rather than weight loading itself; with lower-bandwidth storage, the weight-loading contribution grows and dominates. Even under so-called "warm start" conditions — where artifacts from mechanisms such as torch.compile and kernel warmup are cached — the observed reduction in startup time is modest. The reason is structural: much of the early-phase initialization work is inherently stateful, context-dependent, and therefore not amenable to straightforward caching.

Driving cold-start time down means chasing every contributor across the stack and getting them to cooperate — and keeping it down is an uphill battle as each inference engine adds features and has to be optimized individually.

The well-known alternative is *process-level* checkpoint/restore, which reduces the optimization surface to the checkpoint image size and storage bandwidth. Operating at the OS process level also keeps these optimizations *generic*, so they transfer across workloads.

In this post, we introduce **Dynamo Snapshot** — our solution for checkpoint/restore of AI inference workloads on Kubernetes — along with the design choices and optimizations that get us to start times of **6 seconds or less**.

## CRIU and cuda-checkpoint
A running inference worker's checkpointable state has two components:

- Device state (GPU-side): CUDA contexts, streams, device memory, virtual address mappings, etc. This is not visible to the host. To serialize this state, we use the [checkpointing capability of the CUDA driver](https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/) (which is also exposed by the `cuda-checkpoint` command line tool) to dump the device state to CPU memory of the process owning each CUDA context.
- Host state (CPU-side): CPU memory, threads, file descriptors, namespaces, etc. The Linux kernel has all the bookkeeping necessary to be able to serialize this state. We use an open-source tool, **[CRIU](https://github.com/checkpoint-restore/criu)** (Checkpoint/Restore in Userspace) to walk the Linux kernel's bookkeeping and serialize the process tree's state to disk.

These two tools compose cleanly to allow checkpoint/restore of the full inference worker state. When checkpointing:

1. `cuda-checkpoint` dumps all device state into CPU memory. It becomes a pure host process.
2. CRIU dumps all host-side process tree state to a folder in storage.

When restoring (same or different node):

1. CRIU restores the process tree according to the serialized state from the same folder (note: distributed storage like NFS/SMB allows us to fetch the checkpointed artifact from a different node).
2. `cuda-checkpoint` restores the GPU state from what is serialized in CPU memory onto the new GPUs.

![How GPU, CPU, and disk contents change at each step of cuda-checkpoint and CRIU, then in reverse on restore.](./figures/checkpoint_restore_state_panels.svg)

Since CRIU is run via an external process, **the restored workload process picks up at exactly the instruction it was at when it dumped.** This means there are no synchronization barriers between the workload and CRIU: if the workload needs to prevent being checkpointed until it is quiescent, or needs to be aware that it has been restored, those signals must be managed by an orchestrator that calls CRIU/cuda-checkpoint and/or the workload itself. We address this in the next two sections, which describe the orchestrator and the workload-side hooks respectively.

## Dynamo Snapshot: Kubernetes
In Kubernetes, workloads run inside containers, inside of pods. CRIU checkpoints typically contain references to mounts and files in the container's writable filesystem layer, so we checkpoint at the container level — the process tree state and the writable layer travel together.

Our solution is a privileged DaemonSet called `snapshot-agent`, easily installable via a Helm chart. The agent runs on every node and handles pre-checkpoint and post-restore process/overlay wiring so that it can reliably checkpoint the process tree, namespaces, overlays, etc. for `runc`-managed containers (the OCI runtime we currently target). When the agent observes a workload pod marked as a checkpoint source, it reaches in from the host and performs the checkpoint without entering the container. On restore, the agent restores the workload from inside a placeholder pod that sets up the namespaces.

To keep terminology consistent for the rest of this section:

- **Workload pod** — the pod whose process tree is being checkpointed or restored.
- **Snapshot agent** — the privileged DaemonSet pod, one per node, that performs the checkpoint or restore on the workload pod.
- **Checkpoint Job** — an ordinary Kubernetes Job whose pod runs the workload, marked as a checkpoint source.
- **Placeholder pod** — a pod created on restore that sets up the namespaces the restored workload runs inside.
- **Checkpoint node** / **restore node** — the K8s nodes where the workload is checkpointed and where it is later restored, respectively.

For each workload pod, the snapshot agent on its node performs the checkpoint or restore independently from the host side.

![On the checkpoint node, the snapshot-agent DaemonSet observes the workload pod from the host and writes the artifact to a shared PVC. On the restore node, the agent enters a placeholder pod's namespaces and restores the workload inside.](./figures/k8s_checkpoint_restore_lifecycle.svg)

**Checkpoint:**

1. The user creates a checkpoint Job. This is an ordinary Kubernetes Job whose pod runs the workload, marked as a checkpoint source.
2. The snapshot agent on the same node sees the workload pod via its checkpoint marker.
3. Once the workload's readiness probe passes (we use it as a configurable signal that determines the workload is ready to be checkpointed), the snapshot agent begins the checkpointing process.
   1. The agent inspects the running container from the host side (PIDs, namespaces, mounts, overlays, etc.) without entering it.
   2. It runs cuda-checkpoint and CRIU against the container's process tree, and captures the container's overlay-filesystem diff.
   3. The artifact is written to storage (only shared PVC is supported at the moment).
4. The workload exits, and the Job completes and is garbage-collected.

**Restore (later, on any node):**

1. A placeholder pod is created with an annotation that marks it as a restore target. It uses the same base runtime image as the workload, but with the entrypoint overridden to `sleep infinity` (i.e. sits idle) and includes snapshot tooling. This establishes the namespaces that the restored worker lives inside.
2. The snapshot agent *enters the placeholder's namespaces from the host* and applies the overlay diff, runs CRIU restore and uses `cuda-checkpoint` to resume the CUDA state from *inside* the placeholder's namespaces.
3. Once the restore is complete, the snapshot agent writes a "restore complete" signal file to a path inside the container's filesystem that the workload can (optionally) consume.
4. The restored worker has now taken over the container/pod. The workload refreshes its pod identity by reading the new pod identity from a mount. To the rest of the cluster, the placeholder pod is now a valid worker.

Note: running CRIU restore inside of the placeholder's namespace allows the restored workload pod to not need a privileged security context, which means we don't compromise on Kubernetes isolation/security best practices.

### Why not native Kubernetes checkpoint?
Some OCI runtimes — namely `runc` and `runsc` — already have built-in container-level checkpoint/restore capabilities, and `runc` in particular delegates to CRIU. Higher-level container managers like `podman` and `docker` expose checkpoint/restore by going through the underlying runtime. The checkpoints produced are full OCI images with the checkpointed process tree state baked in. However, we had a few requirements that prevented us from using these native checkpoint/restore capabilities:

- We needed to perform heavy customization and optimization of both CRIU and cuda-checkpoint (more on this in Optimizations #2 and #3).
- We couldn't rely on whether or not checkpoint/restore feature gates were exposed, even at the CRI level, because some cloud service providers do not offer control of kubelet at all. Moreover, this would also require installing CRIU on the host, which isn't always possible.
- We wanted flexibility to configure different storage backends for different parts of the checkpoint artifacts, instead of baking the checkpoint into an OCI image. Only the container's writable filesystem layer (the upperdir of the OCI overlay) and CRIU artifacts should be sufficient given a fixed "base" image.

## Dynamo Snapshot: The Workload
A Dynamo inference worker comes up in two phases:

1. **Engine initialization.** The configured inference engine (vLLM, SGLang, TensorRT-LLM) is started: communicators are initialized, weights are loaded, kernels are warmed up, graphs are compiled/captured, etc. By the end of this phase the worker is fully warm. It could serve a request, but is not yet discoverable to anything outside its own pod.
2. **Distributed runtime startup.** The worker connects to the Dynamo control plane and registers itself with the discovery backend, so the router and the rest of the graph can find it. From this point on, the worker is "live" — there are open connections to the control plane, and other components in the cluster are aware of this worker's pod identity.

If we were to implement checkpoint/restore naively, without the workload knowing it was being checkpointed, the readiness probe of the checkpoint job would correspond to a fully initialized distributed runtime that is registered to the discovery plane, which means there are active TCP connections that cannot be captured by CRIU.

The general pattern that solves this is **quiesce/resume hooks**: the workload ensures it is in a quiescent state and blocks on an external signal that fires when the restore is complete. This is a powerful abstraction for checkpoint/restore because:

1. It lets the workload clean up its resources before being checkpointed, which optimizes the final checkpoint size (and thereby decreases restore time).
2. It allows the workload to recreate resources that aren't checkpointable post-resume. This is especially important for multi-GPU and multi-node checkpoints (planned for a future release): outbound TCP connections used for RPC cannot be checkpointed in an established state since the pod IP changes between checkpoint and restore, and RDMA registrations and NIC state also need to be recreated post-restore.

In Dynamo Snapshot, we implement these hooks by configuring the readiness probe to be the presence of a "ready for checkpoint" signal file that the worker writes after the engine initializes but *before* distributed runtime startup. At this point, the worker polls for another "restore complete" signal file while the snapshot agent is checkpointing it from outside — the checkpointed state of the worker could be at any arbitrary point inside the polling loop. Because CRIU restores execution at exactly the instruction it was checkpointed at, the worker resumes inside the polling loop wherever it was, detects the signal file, and starts distributed runtime setup with no additional synchronization needed.

![Worker initializes, signals readiness, and quiesces while the snapshot-agent dumps state; on restore, the agent signals the worker to start the distributed runtime.](./figures/worker_agent_quiesce_resume_sequence.svg)

## Optimization #1: KV Cache Unmap and Release
One optimization to reduce the checkpoint size is to deallocate the KV cache memory before checkpointing. After measuring the peak GPU memory usage while weights, CUDA graphs, and other buffers/activations are allocated, inference engines allocate the remainder of the GPU memory as a large KV cache buffer.

However, since our checkpoint is taken in a quiescent state *before* the replica has served any requests, this KV cache buffer does not need to be checkpointed at all. But we need to keep the virtual address of this KV cache stable since it is baked into the CUDA graph. This means we allocate the KV cache buffer via the CUDA Virtual Memory Management API (`cuMemCreate` and `cuMemMap`); then deallocating the underlying physical allocation while keeping the virtual address stable is as simple as calling `cuMemUnmap` and `cuMemRelease`, but not `cuMemAddressFree`.

This functionality is already given to us by vLLM's `sleep()` and `wake_up()` methods, as well as SGLang's `torch_memory_saver`. Similar functionality is also exposed in TensorRT-LLM.

![Unmapping and releasing the KV cache shrinks device memory at checkpoint time to just weights and the CUDA graph buffers.](./figures/kv_cache_unmap_release.svg)

Unmap and release of the KV cache reduces the checkpoint size of Qwen3 0.6B for a B200 from ~192 GiB to ~6 GiB. The wins are most pronounced for large KV cache sizes (i.e. smaller model weights relative to GPU size).

## Optimization #2: Making CRIU Fast
So, what do the restore times look like? Surprisingly, really bad. For larger models, the restore time actually exceeds that of a cold start, defeating the entire purpose of checkpoint/restore.

![Baseline snapshot restore time across model sizes — for larger models the restore exceeds cold start.](./figures/regular_restore_criudev.svg)
<!-- TODO(figure): update regular_restore_criudev.svg to overlay the cold-start times alongside baseline restore times so the reader can read both off the same chart instead of flipping back to the cold_start_bench figure. -->

The main reason behind this is that CRIU and `cuda-checkpoint` do not copy memory at speed-of-light (SOL) speeds. In a Linux process, there are two types of memory: anonymous memory (the heap, stack, etc. of a process) and shared memory (shared between processes). For larger models, the restore bottleneck encompasses both types of memory, so we optimize both restore paths to bring the CRIU restore time down from minutes to seconds.

### Optimization #2.1: Linux Native AIO for Anonymous Memory
After CRIU has restored the shared resources (files, sockets, shmem objects, memfds, etc.), it still has to fill in each process's private memory: heap pages, stacks, anonymous mappings, and copy-on-write private file mappings. These pages are not shared; they belong to one process and need to land at the exact virtual addresses they had before checkpoint.

In upstream CRIU, that fill was a synchronous `preadv` loop. The restorer pulls one job off the list, hands it to `preadv`, and waits. The kernel issues that single read to the storage device, the device DMAs the bytes into the destination VMA pages, and `preadv` returns. Only then does the restorer move on to the next job. There's exactly one read in flight at any moment, which means the storage device is idle for most of the wall-clock time — sitting there waiting for the next request between every read. A single blocking stream barely uses what fast NVMe can do, and on network-attached storage every read pays a full round-trip before the next one starts.

![Synchronous preadv loop: one read in flight at a time, storage device idle between requests.](./figures/preadv_serial_before.svg)

We replaced the `preadv` loop with Linux native AIO. CRIU builds a list of read jobs ahead of time — each job is an `iocb` describing a file offset, a byte count, and an `iovec` pointing at the destination VMA pages. The restorer creates an AIO context, which holds many distinct read transactions simultaneously, allowing the storage device to run them concurrently across its internal channels. The restorer then submits a batch of those iocbs with `io_submit`, and keeps a window of up to 128 in flight. As completions come back via `io_getevents`, new submissions backfill the window until every job is done.

![Native AIO pipeline: up to 128 reads in flight via io_submit and io_getevents, storage device runs them concurrently.](./figures/aio_pipeline_after.svg)

On its own, AIO already accounts for the bulk of the gain — for instance, AIO alone brings CRIU restore time from ~82s to ~50s on Llama 3.3 70B FP8 and from ~119s to ~73s on GPT-OSS 120B. The remaining gap to single-digit seconds comes from the parallel memfd restore described next.

#### Direct I/O and the Page Cache
Where the storage backend supports it, both anonymous and shared memory reads use `O_DIRECT`. Restore is mostly a one-pass stream from checkpoint files into destination memory, so caching the input pages in the kernel page cache is usually wasteful. Without direct I/O, a large restore can temporarily fill the page cache with checkpoint data while also allocating the destination shmem pages, increasing memory pressure and evicting useful data for other workloads.

Even more importantly, Linux native AIO is only truly asynchronous on files opened with `O_DIRECT`. On filesystems where `O_DIRECT` is unavailable or unreliable, such as some NFS deployments, restore falls back to buffered I/O with sequential readahead so the kernel still sees a predictable streaming access pattern, but the gains from AIO are significantly reduced.

### Optimization #2.2: Parallel memfd Restore
vLLM's sleep mode reduces GPU memory pressure by moving tagged GPU allocations into pinned CPU shadow buffers. Those buffers are not ordinary Python heap memory. vLLM asks PyTorch for pinned CPU tensors, PyTorch allocates them through CUDA's pinned-memory allocator, and CUDA backs them with shared anonymous memory that is then pinned through the NVIDIA driver. Inside the Linux kernel, these are memfds — anonymous, RAM-backed files that can be mapped with `MAP_SHARED`.

For GPT-OSS 120B, we saw these buffers consume more than 120 GiB, but split up into many 2 GiB (or even smaller) buffers. These buffers are also independent. However, CRIU restored these serially — it would create one shmem-backed object, resize it, map it, read its contents from the checkpoint image, and only then move on to the next object.

The solution was to modify CRIU to first enumerate all the unique shmem-backed objects, then launch a thread pool to parallelize the restore. Each worker allocates its buffer and reads from the checkpoint independently, allowing them to use the available storage bandwidth and CPU parallelism instead of processing buffers one at a time.

### Results
On the same setup, we saw a massive improvement in CRIU restore time, and it is now significantly faster to restore from checkpoint than to cold start an inference worker:

<!-- TODO(itay-feedback): add an SOL column (checkpoint size / NFS bandwidth) once we confirm the PVC bandwidth used in this benchmark, so the reader can see how close optimized CRIU is to the storage limit. -->
| Model | Checkpoint size | CRIU Restore (baseline) | CRIU Restore (optimized) | Speedup |
| --- | --- | --- | --- | --- |
| Qwen3 0.6B | 6.2 GiB | 6.8 s | 2.4 s | 2.8x |
| Qwen3 8B | 26 GiB | 24 s | 4.7 s | 5.1x |
| Qwen3 14B | 47 GiB | 44 s | 6.8 s | 6.5x |
| Qwen3 32B | 74 GiB | 70 s | 9.9 s | 7.1x |
| Llama 3.3 70B FP8 | 86 GiB | 82 s | 11 s | 7.5x |
| GPT-OSS 120B | 129 GiB | 119 s | 15 s | 7.9x |
| Qwen2.5 72B | 164 GiB | 127 s | 20 s | 6.4x |

![Optimized snapshot restore time after AIO and parallel memfd changes — significantly faster than cold start across all model sizes.](./figures/regular_restore.svg)

At this point CRIU is no longer the bottleneck on its own, but the wall-clock time is still dominated by moving the model weights from PVC, through host memory, onto the GPU. That part is fundamentally serial: cuda-checkpoint cannot start copying weights to the GPU until CRIU has materialized them in host memory, and both halves are constrained by NFS bandwidth on one end and a single sequential `cudaMemcpy` on the other. The weights also dominate checkpoint size by a wide margin, which puts a hard ceiling on how fast restore can ever get if the weights stay inside the CRIU image.

## Optimization #3: GPU Memory Service
The idea behind the **GPU Memory Service (GMS)** is to take the weights out of the CRIU image entirely, and let weight restoration happen on its own path. This makes restore faster for three reasons that compound. First, the CRIU artifact shrinks dramatically once weights are no longer part of process memory, so CRIU restore itself completes in seconds. Second, weight restoration runs *concurrently* with CRIU rather than serially after it, so total restore time is bounded by the slower of the two paths instead of their sum. Third, the weights no longer have to follow the path of NFS → host memory → single sequential `cudaMemcpy`, and can instead use whatever channel is fastest for the cluster — sharded local SSDs, GPUDirect Storage, or peer-GPU RDMA from a node that already has the weights resident.

The subtle constraint lies in preserving execution correctness. The restored worker must observe its weight tensors at precisely the same virtual addresses as before, since these addresses are embedded within CUDA graphs captured during initialization. Consequently, any external restoration mechanism must materialize the weight bytes directly at those original virtual addresses, without introducing a copy at resume time. This requirement mirrors the KV-cache virtualization strategy discussed in Optimization #1, except this time we actually *care* about the contents of the tensors post-restore.

### Parallelizing Weight Restore
GMS is a per-GPU sidecar process (deployed as a separate container/pod) that owns physical GPU memory on behalf of inference workers, built on top of the CUDA Virtual Memory Management API. At a high level, it lets the worker hand off ownership of its weight allocations to a separate process that outlives any individual worker, so that when the worker is checkpointed, the weights are not part of its checkpointed state — and when it is restored, it can reattach to the same physical memory on the new node without copying.

The payoff is that the path that gets weights onto the GPU is now **completely decoupled** from CRIU and cuda-checkpoint. CRIU can stream process state (which is now much smaller) from the snapshot PVC at NFS bandwidth, and *in parallel*, an entirely separate `gms-loader` sidecar populates GMS with the weights for that model. The two paths converge only at the moment the worker resumes, when it reattaches to the now-populated GMS and continues serving.

![Restored pod data flow: snapshot-agent restores the CRIU artifact into the worker while gms-loader streams weights into gms-server in parallel.](./figures/gms_combined_dataflow.svg)

Where the loader gets the weights from is its own concern. In a real cluster, we want a single source of truth for which models are cached where, deduplicated downloads from external sources like HuggingFace, and ideally GPU-to-GPU RDMA between nodes that already have the weights resident. This is exactly what weight transfer engines like [ModelExpress](https://github.com/ai-dynamo/modelexpress) (MX) are built for, and the intended production path is to have `gms-loader` be a shim that exposes GMS's stored allocations directly to different weight transfer backends — peer GPU over NIXL/RDMA, disk over GPUDirect Storage, HuggingFace, etc.

We are still in the process of developing the integration with MX, among other transfer engine backends. For now, the fallback is to also use NFS for the weights, which eliminates the full host-side materialization of the weights but still causes some NFS bandwidth contention with the ongoing CRIU restore. Nonetheless, we still see a major reduction in startup time, even with this fallback.

### Full Overlap with External Weight Restoration
To demonstrate the full power of overlapping CRIU and cuda-checkpoint restore with a faster channel for restoring the weights, we implemented a proof-of-concept backend for `gms-loader` where the weights for each model are sharded across 8 SSDs on a node, and ensured the restored workload was on the same node as the checkpoint.

This setup isolates the parallel-restore claim from network storage variability and shows what the rest of the system can do when the weight source isn't the bottleneck. The loader pipelines reads with GPU copies (a pool of threads issuing `cudaMemcpyAsync` over multiple CUDA streams) so storage→host and host→GPU run continuously rather than fully materializing the weights in host memory.

<!-- TODO(itay-feedback): before showing the optimized sharded-SSD result, add an intermediate benchmark of Snapshot+GMS with the PVC backing the weights (same NFS channel as the CRIU baseline) so the comparison stays apples-to-apples. Reference: Qwen2.5-72B with PVC-backed GMS restores in ~20s vs ~40s for CRIU-only on PVC. Then frame the sharded-SSD case as the "now upgrade the channel" follow-up that hits 5–6s. -->
We saw that parallelizing the container restoration in CRIU and the weights restoration via `gms-loader` allows us to achieve under 6s restore time, even for the largest checkpoint (Qwen2.5 72B). For most other models, the startup time is well under 5 seconds.

![Snapshot restore time with GMS, weights sharded across 8 local SSDs — under 6 seconds for all model sizes including Qwen2.5 72B.](./figures/gms_sharded_ssd_restore_bench.svg)

The CRIU checkpoint now only contains the host-side state of the container's process tree and a few miscellaneous buffers that are still double-buffered. The GMS weight artifact now holds the majority of process memory.

| Model | CRIU checkpoint size (baseline) | CRIU checkpoint size (with GMS) | GMS weight artifact |
| --- | --- | --- | --- |
| Qwen3 0.6B | 6.2 GiB | 4.3 GiB | 1.2 GiB |
| Qwen3 8B | 26 GiB | 4.8 GiB | 15 GiB |
| Qwen3 14B | 47 GiB | 5.0 GiB | 28 GiB |
| Qwen3 32B | 74 GiB | 5.5 GiB | 61 GiB |
| Llama 3.3 70B FP8 | 86 GiB | 6.1 GiB | 68 GiB |
| GPT-OSS 120B | 129 GiB | 6.7 GiB | 74 GiB |
| Qwen2.5 72B | 164 GiB | 5.8 GiB | 135 GiB |

## Looking Forward
<!-- TODO: need @athreesh's input on phrasing of these roadmap bullets, especially the unreleased CUDA driver patch one. -->
Not everything in this blog post is generally available in Dynamo yet:

- **Support for Other Frameworks:** Dynamo Snapshot currently works for vLLM and SGLang. TensorRT-LLM support is work in progress.
- **CRIU optimizations:** We are currently in the process of upstreaming the AIO and parallel memfd optimizations to CRIU.
- **GMS + Snapshot Integration:** This requires an unreleased CUDA driver patch to be able to work with GPU migration (checkpoint on one GPU, restore on another) that is necessary for this to be practical. The patch will be part of an upcoming CUDA driver release.
- **GDS/P2P transfer via MX integration with GMS:** Currently we only have the PVC-source fallback for weight restoration. We are working on integrating GPUDirect Storage and peer-to-peer GPU transfer into `gms-loader`, enabling direct SSD→GPU weight loading and GPU→GPU weight transfer for autoscaling.
- **Multi-GPU support:** Single-node multi-GPU is still experimental (currently vLLM-only, with limitations) and may fail on certain hardware/driver configurations. We are actively working on robust multi-GPU and multi-node checkpoint/restore support.
