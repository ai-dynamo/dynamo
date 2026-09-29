<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Independent GMS V1 services populated by native PageBroker

Four default-config GLM restores passed on the nscale DRA B200 node, using
PageBroker to populate independently hosted GMS V1 servers. The two 16-CPU runs
reached workload Ready in **22.650 and 22.732 seconds**. Both readiness inference
and a second inference request passed. The experiment establishes the native
integration and its resource savings; it does **not** establish a restore-time
win over the prior Python-loader experiment.

[Interactive Gantts](results/default-config/pagebroker-study/timelines.html),
[CPU/contention analysis](results/default-config/pagebroker-study/cpu-analysis.md),
and [independent qualification](results/default-config/pagebroker-study/qualification.md)
contain the supporting measurements and limits.

## Corrected deployment dependency

GMS is an independent DaemonSet, with one named DRA GPU request per server
container. Each server sees local `cuda:0`. The DGD contains only the engine
container. The already-running Snapshot agent and PageBroker are separate node
services. No GMS loader process or container performs these transfers.

Generic service readiness has no selected capture. The timed sequence is:

1. Create the DGD and obtain its UID.
2. Read the Snapshot selected by that created DGD and resolve its content UID.
3. Read `/checkpoints/artifacts/<content-uid>/gms/capture.json` from the PVC.
4. Resolve named DRA requests against that capture's logical ranks; verify each
   matching V1 artifact manifest and its exact allocation IDs and sizes.
5. Submit eight native PageBroker loads. CRIU/CUDA restore proceeds independently.
6. Drain copies, release writable imports, commit each V1 allocation epoch, and
   verify every publication before opening the engine's captured `/gms/all-ready`
   gate.

The attached capture descriptor references the exact retained GMS PVC payloads;
no 448 GiB payload copy or host-RAM staging was performed. GMS servers have no
weight-PVC mount. Only the metadata coordinator and PageBroker access it.

The old resident Python-loader trials preselected their capture and dispatched
loading concurrently with DGD creation. They remain useful transfer controls but
are **optimistic end-to-end controls**; they are not pooled with this corrected
deployment timing. In the first native run, DGD creation returned at +0.250 s,
Snapshot/Content API lookup ended at +0.679 s, PVC manifest discovery ran at
+1.122–1.157 s, and the first native read started at +1.595 s. The external
harness's API/HTTP transport is included, not silently subtracted.

## Real restores

These runs retain the captured GLM engine communication configuration, including
FlashInfer mnnvl allreduce fusion and the existing logits-gather path, with no new
communication fallback. The capture's single-node NCCL overrides remain in
effect: `NCCL_IB_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`, and `NCCL_NVLS_ENABLE=0`.

All runs read **448 GiB** of saved, padded V1 allocation payload: 112 distinct
4 GiB shards, containing 224 captured allocations. The transfer window is first
native `O_DIRECT` read start through final read/copy routine completion after
draining and cleanup, including intervening queue waits. GB/s uses decimal units.

| Case | PB CPU request / limit | DGD → Ready | Weight window | Weight GB/s | CRIU | CUDA timer | Main start |
|---|---:|---:|---:|---:|---:|---:|---:|
| native16-1 | 2 / 16 | 22.650 s | 18.386 s | 26.163 | 6.252 s | 8.790 s | +1.926 s |
| native16-2 | 2 / 16 | 22.732 s | 18.084 s | 26.600 | 6.725 s | 8.865 s | +1.464 s |
| native64-1 | 2 / 64 | 25.931 s | 17.414 s | 27.623 | 6.198 s | 8.120 s | +7.030 s |
| native64-2 | 2 / 64 | 22.603 s | 17.592 s | 27.343 | 6.384 s | 8.065 s | +3.421 s |

No sampled run throttled PageBroker's CPU quota. Its CPU consumption remained
about 47 CPU-seconds per real-trial sampling window, including idle lead-in and
post-Ready collection; sampled CPU pressure was low. Increasing
the quota to 64 therefore has no demonstrated CPU bottleneck to relieve and
showed no repeatable Ready-time benefit. Keep the 16-CPU configuration.

The 25.931-second run really started main late. The host Pod was already observed
scheduled at +0.784 s, but CRI created main at +6.789 s and started it at +7.030 s.
The image was already cached. Snapshot entered its handler 29 ms after main
started. Across the four runs that gap was 7–52 ms. This evidence locates the
large variation before container start, not in controller reconciliation after
an already-running main; finer kubelet/DRA/sandbox traces would be needed to
identify the exact node-side cause.

## Equal-payload contention control

`isolated64-1` creates a zero-replica DGD, resolves the selected artifact only
after that creation, and runs the identical native GMS load with no engine Pod,
CRIU, or native CUDA restore. It uses the same resident PageBroker Pod, GPU UUIDs,
artifact digests, CPU resources, and full 448 GiB payload as the 64-CPU restore
cases. This is a transfer microbenchmark, not an engine-restore time.

The isolated payload window was **13.868 s (34.688 GB/s)**. The matching
`native64-1` window was **17.414 s**, 3.547 s / 25.6% longer. Shared-buffer wait
nearly vanished in isolation. Mean per-rank active read/copy time increased from
13.382 to 15.949 s under restore, and summed I/O request service time increased
27.6%. CUDA event wait changed little. Concurrent restore adds **41.234 GiB** of
native GPU-state transfer, plus CRIU CPU-image I/O not counted by these GPU
reports. This supports storage/shared-buffer contention rather than a CPU quota
explanation; one isolated run does not uniquely identify every shared resource.

## Startup, memory, and restore hooks

The server-only processes reached online in **0.130–0.329 s** from script start;
per-server `cuInit` took **0.027–0.198 s** in the first qualification. Actual
driver queries found no current or active primary CUDA context at readiness,
first allocation, or commit. The persistent multi-GPU PageBroker paid its
8.071-second context/ring initialization before the timer.

PageBroker reuses its qualified 32 × 128 MiB ring per GPU: **32 GiB pinned across
eight GPUs**. Removing the Python loaders eliminates their additional 32 GiB
of pinned rings and their contexts. The prior fused arrangement already had no
separate loader containers, so this changes transfer ownership and memory, not
an extra eight-container count. One GMS server per GPU remains.

GMS and native engine transfers share each GPU's ring. A FIFO lease released
after each 4 GiB GMS shard prevents the entire 56 GiB rank load from monopolizing
the ring. Queue waits are separately shown in the charts. No transfer core,
qualified NIXL revision, CRIU implementation, or CUDA shim was changed.

The deployed `nsrestore` also runs **cuInterpose reconstruction after stopping
its CUDA timer**. The following measured tail includes that hook and cleanup;
it must not be labeled idle time or attributed entirely to GMS. An audit of the
actual deployed helper, rather than a different local binary, established the
call ordering. Existing logs cannot split hook execution from cleanup or prove
whether concurrent GMS driver operations slow the hook.

## Reproducibility and scope

Snapshot source: `7e347afb62690103d53ea867265167ba9d2b886d`, against qualified
PageBroker base `ca369464c41157bb30da063748b1bea025c34d57`; the complete patch is
[pagebroker-gms-native.patch](pagebroker-gms-native.patch). The deployed Go agent
remained `b209afc332631c5a4c8f747d59dccecd32b4d03e`. Exact binary/runtime/library
hashes, build and test logs, manifests, drivers, and the helper call audit are
under [pagebroker-study](results/default-config/pagebroker-study).

The CPU suites cover native V1 interoperability and FD lifetime, malformed
messages, deadlines, manifest/path/ID admission, replay fencing, layout,
scheduling, and existing native restore behavior. The Dynamo harness suites
passed 63 tests. All four GPU trials were independently checked for exact IDs,
UUIDs/nonces, publication order, DGD/DaemonSet ownership, unchanged shim bytes,
zero restarts, and coherent inference. All four real restores passed inference.

This remains an experimental operator `podTemplate`/Engine API integration,
not production `checkpointRef` or Dynamo frontend registration. The measured
mapping is identity UUID-to-UUID; a new physical permutation was not tested in
this cohort. Services are generic and started independently, but this prototype
admits only one load per coordinator/server incarnation and rejects destructive
RW retries. Production multi-DGD lease retirement remains to be integrated.
`O_DIRECT` bypasses host page-cache payload reads; storage-system caches were not
controlled.

The original Snapshot stack and node settings were restored. Experiment Pods,
claims, ConfigMaps, private NFS mounts, and node staging directories were removed.
The matching GLM/Qwen snapshots, exact GMS PVC payloads, and new adjacent capture
descriptor remain available for subsequent experiments.
