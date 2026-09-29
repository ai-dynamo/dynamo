<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Equal-payload GMS versus PageBroker transfer study

Both transfer paths read **448 GiB (481,036,337,152 bytes)** from the exact same
112 current GMS artifact files into the same eight DRA GPUs, on s2877. Both use
one GPU per container, the same qualified NFS transport, and verified `O_DIRECT`.
No CRIU or CUDA checkpoint restore competes during these transfer-only trials.
This is client-direct PVC I/O, not RAM staging; storage-server cache/load remains
uncontrolled. A barrier starts transfers after all contexts and destination GPU
allocations exist. That barrier is a benchmark device, not a deployment proposal.

| Configuration | Runs | Mean transfer span | Range | Useful GB/s |
|---|---:|---:|---:|---:|
| GMS current, 16 lanes × 2 × 16 MiB | 3 | 16.600 s | 16.190–17.118 s | 28.98 |
| PageBroker, warm 32 × 128 MiB | 3 | 13.183 s | 12.827–13.379 s | 36.49 |
| GMS warm, 16 lanes × 2 × 16 MiB | 1 | 16.486 s | 16.486–16.486 s | 29.18 |
| GMS warm, 16 lanes × 2 × 128 MiB | 2 | 12.976 s | 12.917–13.035 s | 37.07 |
| GMS warm, 32 lanes × 2 × 64 MiB | 2 | 12.899 s | 12.843–12.955 s | 37.29 |
| GMS cold, 16 lanes × 2 × 128 MiB | 1 | 14.172 s | 14.172–14.172 s | 33.94 |
| GMS current, CPU request/limit 1/8 | 2 | 16.568 s | 16.160–16.976 s | 29.03 |
| GMS current, CPU request/limit 8/16 | 2 | 16.134 s | 16.076–16.191 s | 29.82 |

GMS's original small-buffer configuration remains slower without contention.
Prewarming it alone does not close the gap. Increasing buffer/chunk capacity,
with either 16 × two 128 MiB or 32 × two 64 MiB slots per GPU, reaches the measured
PageBroker transfer range. These tuned GMS rings use **4 GiB of pinned host memory
per GPU (32 GiB total)**, matching PageBroker; the original GMS ring uses 512 MiB
per GPU. Ring setup for warm cases is outside transfer timing and recorded per
rank. Therefore ~13 s is a warm bulk-transfer result, not pod startup, V1
publication, first inference or full restore latency.

The PB probe calls the unchanged qualified C++ TransferBuffers/NixlTransfer
implementation with its exact NIXL runtime/API revision. GMS calls the existing
PosixDirect prototype. The warm GMS variant changes only buffer lifetime, retaining
DMA drain before return; chunk/lane settings are explicit. PB reads each 4 GiB
shard into two adjacent destination extents; GMS reads its two manifest extents.
This compares the selected fused prototype backend, not every supported GMS
backend, and does not exercise a new broker→GMS control API.

After timing, every run compares the first/middle/last 4 KiB GPU page of each
2 GiB allocation to the PVC: **672 checked pages per run**. This is sampled byte
verification, not full hashing or a new full-model inference qualification.

## CPU resources and start timing

The CPU comparison raises both request and quota from 1/8 to 8/16 CPUs per rank,
then returns to 1/8 (A–B–B–A). The same claim, GPUs, bytes, mount and original
16 MiB transfer settings are retained. Actual cgroup quota and weight are saved
for every rank. Live resize was rejected by the DRA/vcluster path; the experiment
pod was recreated with the requested resources. Timers still exclude pod setup.

- PageBroker, warm 32 × 128 MiB: mean 0.434 CPU/rank during transfer; 0 throttled periods across sampled ranks.
- GMS warm, 16 lanes × 2 × 16 MiB: mean 0.231 CPU/rank during transfer; 0 throttled periods across sampled ranks.
- GMS warm, 16 lanes × 2 × 128 MiB: mean 0.261 CPU/rank during transfer; 0 throttled periods across sampled ranks.
- GMS warm, 32 lanes × 2 × 64 MiB: mean 0.277 CPU/rank during transfer; 0 throttled periods across sampled ranks.
- GMS current, CPU request/limit 1/8: mean 0.246 CPU/rank during transfer; 0 throttled periods across sampled ranks.
- GMS current, CPU request/limit 8/16: mean 0.250 CPU/rank during transfer; 0 throttled periods across sampled ranks.

These isolated results do not establish CPU starvation during actual restoration.
The main engine requests 32 CPUs and is capped at 96; each GMS rank originally
requests 1 and is capped at 8. Requests affect scheduling and relative CPU share
under contention; limits can impose quota throttling. A thread count is not a
count of busy CPU cores. Parent-cgroup/kernel work and full-engine competition
require their own measurement.

The original deployment was not simultaneous: in `dispatch-fixed-early-2`, main
placeholder start was approximately +1.520 s, while rank script starts were
+2.218 to +4.013 s (1.795 s skew). Snapshot agent start was +4.875 s. Container
Running is not restored-engine readiness. The transfer probe removes this skew.

## Contention and remaining integration

The earlier full restore standard-NIXL pairs measured 23.397 s serialized versus
25.031 s overlapping all-rank GMS loading: 1.634 s inflation, while CRIU and CUDA
also slowed. That establishes contention exists, but is a different backend/mount
cohort; it cannot be subtracted from the present tuned results to assign a cause.

A persistent PageBroker that directly acquires V1 write leases and fills exported
GMS allocations is a reasonable next integration. The current fused prototype
already combines each rank's loader and server in one process/container, so it
has no separate eight loader containers to remove. Centralizing transfer work
can reuse rings and coordinate I/O while keeping eight lightweight, one-GPU GMS
servers. See [API/ownership design](PAGEBROKER-GMS.md) for exact IDs, UUID checks,
commit/abort, shard offsets, and arbitration with native residual transfers.

The new bulk-transfer results do not prove a new pod-to-ready speedup. Retest
full restore contention after integration; do not serialize native restoration
behind a giant GMS transfer job.

## Raw evidence and failed setup attempts

[All measurements](results/transfer-comparison/comparison.json), manifests,
per-rank direct-open traces, CPU counters, setup times, provenance and replay
scripts are under `results/transfer-comparison`. `pb32-1` failed before transfer
because libaio1t64 was missing from the workload image; the exact broker runtime
dependency was then copied. `pb32-ok1` launch raced a code refresh and never
reached its all-rank barrier. Neither has a completed transfer result.
`pb32-clean1` succeeded and is retained but excluded from the main PB group
because prior failed barrier clients might still have been timing out during
its setup. Subsequent samples ran after those processes exited.

## Standalone fused V1 server + load + publication

These runs launch all rank processes from a barrier, without CRIU/native restore.
The instrumented transfer method is unchanged; its start/end are recorded inside
V1 load_weights. Publication includes V1 allocation/import/cleanup/commit.
They are not inference-readiness tests.

| Case | Start → last publication | First transfer → last transfer |
|---|---:|---:|
| isolated-v1-16-1 | 22.485 s | 17.354 s |
| isolated-v1-128-1 | 19.692 s | 13.819 s |
| isolated-v1-16-2 | 22.791 s | 16.429 s |

Experiment resources were released and the private mount removed. Agent/operator
configuration and node NUMA balancing were unchanged. Both retained snapshots
still report Ready; see `results/transfer-comparison/cleanup-verification.json`.
