<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Resident GMS V1 and restore startup

Prestarting eight one-GPU GMS V1 server/loaders in a separate DaemonSet reduced
mean DGD request to coherent workload Ready from **25.822 to 22.781 seconds**
in three paired trials: **3.042 seconds (11.8%)**. All 448 GiB of weights were
read from the PVC with O_DIRECT after the timed request. No weight allocations
or payload reads occurred during prewarming.

[Interactive Gantt and phase details](results/default-config/daemonset-study/timelines.html)
· [Measurements](results/default-config/daemonset-study/comparison.json)
· [Hot-path audit](results/default-config/daemonset-study/hot-path-analysis.md).

| Primary comparison, mean seconds (three trials each) | GMS starts with engine Pod | Resident GMS DaemonSet |
|---|---:|---:|
| DGD request → workload Ready | 25.822 | 22.781 |
| CRIU restore | 4.600 | 6.473 |
| CUDA restore phase | 8.667 | 8.286 |
| Sum of native PREPARE measurements | 8.327 | 7.965 |
| Aggregate weight transfer completion window | 17.471 | 17.482 |

Cold totals were 27.656 / 25.328 / 24.483 s; resident totals were
21.684 / 23.267 / 23.391 s. Keep the first cold run: it had longer container
startup, and removing it after seeing the result would overstate the evidence.
The first pair's 5.972 s improvement is larger than the overall mean.

The transfer window runs from the first O_DIRECT read across all ranks through
the last lane completion. Cold completion includes buffer unregister/free and
FD cleanup; resident completion drains copies but retains its buffers. This is
an effective full-payload completion rate, not an isolated copy-bandwidth
microbenchmark. Pooled rates are approximately **27.53 vs 27.52 GB/s**.
Starting every rank together lengthens some individual rank spans without
lowering aggregate throughput.

CRIU is consistently slower with resident GMS in these three pairs, by 1.873 s
on average. Earlier weight I/O overlaps more of CRIU's work; shared storage,
host memory or CPU contention is plausible, but these runs do not isolate the
resource responsible. The CUDA phase and summed native PREPARE intervals do
not show a corresponding slowdown. The qualified shim is unchanged. Every
resident engine reaches the publication gate after weights are available;
its wait is only about 18–22 microseconds. Engine restore is now critical.

## What the startup gap contains

Precise CRI timestamps distinguish actual main-container start from late
Kubernetes status observation. In the first pair, main starts at +4.938 / +2.454 s
and the handler enters at +4.950 / +2.470 s: **12 / 16 ms after main starts**.
The much longer status-observation bar is not time spent waiting for main.

The instrumented path is: DGD creation and operator child creation, scheduling
and sandbox setup, init-container lifecycle, Snapshot/Content reads and artifact
validation, finalizer/status writes, Pod-IP availability, running-container
discovery, then handler dispatch. Successful preflight and writes take tens of
milliseconds; identity discovery to handler takes approximately 4–6 ms. Increasing
general reconciliation frequency does not remove container startup or the
cached Pod-IP dependency.

The separate preinstalled-bundle experiment removes only `snapshot-cuda-install`
and mounts the same three files read-only from the node. Their SHA256 values match
the pinned installer image and qualified PVC core shim. Results are kept outside
the primary comparison. Its first run exposes a remaining real dependency:
main starts at +1.700 s, but the handler enters at +2.283 s because Pod IP has not
yet appeared in Kubernetes status.

The additional double-opt-in CRI network resolver binds a validated sandbox's
IP to its running main container. It leaves informer state unchanged, uses the
existing 50 ms polling interval, and preserves the status path by default.
With the preinstalled bundle, two trials enter the handler at **+1.798 and
+1.715 s**, **17.8 and 19.6 ms after main starts**. The neighboring control on
the same agent binary enters at +2.940 s, **1.076 s after main starts**.
This removes the measured cached-IP wait without increasing general
reconciliation frequency.

Ready times are 22.603 / 21.731 s with network discovery, versus 22.369 s for
that control. Two trials versus one, with restore-phase variation, do not
establish an additional end-to-end gain. Keep this comparison separate from
the primary residency result. See [network startup analysis](results/default-config/daemonset-study/network-hot-path-analysis.md).

## GMS lead time

The cold path includes staggered container starts, Python imports, driver
initialization, server socket setup, loader CUDA context creation, manifest
validation, exact allocation/import work, and pinned-buffer registration.
`cuInit` is only one part. The V1 server uses driver VMM allocations; the loader
requires a CUDA context for streams and copies.

Resident startup warms the main loader context and sixteen dedicated transfer
lanes, each with two 128 MiB pinned buffers. That retains **4 GiB per GPU,
32 GiB total** of pinned host memory. It does not reserve the 448 GiB of weight
allocations before the request. Every trial uses a fresh DaemonSet generation;
this prototype does not recycle a published allocator across engine instances.

In the first resident trial, prewarming takes 1.914 s per rank on average,
including cuInit 0.073 s, loader context 0.391 s and pinned rings 1.278 s.
The first timed PVC read begins at +0.806 s. Approximately 0.47 s is the remote
HTTP trigger path, followed by metadata validation and about 0.38 s per rank
of exact allocation/import work. All eight rank workers observe the trigger
within 3.9 ms. Residency removes process and buffer setup from this lead time;
it does not make allocation or request delivery free.

## Contract and scope

- The DGD is deployed by the existing Dynamo operator; ownership UIDs are
  verified through DGD → DCD → Deployment → ReplicaSet → Pod. The engine Pod
  and DaemonSet consume the same held, named DRA claim on the eight-B200 node.
- The rank-derived plan drives the CUDA UUID map and GMS rank/artifact/socket
  identities. The publication gate checks server UUIDs and exact allocation
  IDs/sizes before writing `/gms/all-ready`. A capture/generation-qualified
  one-shot HTTP trigger starts transfers concurrently with DGD creation.
- This is same-node interpod GMS using shared socket inodes and CUDA VMM FDs
  transferred over Unix sockets. It does not implement cross-node GMS transport.
  Both processes use the same UID for the `0600` sockets. Each GMS container sees
  one GPU as local `cuda:0`; captured socket identities remain stable.
- Same qualified storage transports, NUMA settings, 16 workers, 128 MiB chunks,
  GMS CPU request 1 / limit 8, engine CPU request 32 / limit 96 and 32 GiB shared
  memory. Snapshot agent and PageBroker are Ready before every timer. Images,
  claim and mounts are prepared; CPU checkpoint pages receive advisory eviction.
  NFS server cache is uncontrolled. The readiness clock includes coherent
  generation; a second independent prompt verifies restored inference afterward.
- The retained default-communication capture, CUDA shim and PageBroker are
  unchanged. The single-node capture retains its existing NCCL_IB_DISABLE=1;
  fused/allreduce paths are not disabled for this experiment.
- This is an operator-deployed custom podTemplate prototype. Native
  `checkpointRef`, standard `gpuMemoryService` rendering, Dynamo runtime worker
  re-registration and frontend routing are not exercised by this retained
  Engine API capture. No compatibility metadata was fabricated.

For an operator integration, keep node services resident, verify their generation
and rank allocation before dispatch, and start their load as soon as the DGD's
destination mapping is known. Node-installed, versioned CUDA restore bundles can
remove the per-pod copy container. Publication must remain a dependency at the
engine's first weight use, while transfers overlap engine restore. Direct
PageBroker-to-GMS loading is a separate follow-up experiment. This comparison
still uses a V1 loader inside each independent GMS server process; there are no
separate loader containers. Its measured bulk transfer window is unchanged by
residency, while loader setup moves outside the request.

## Validation and reproduction

The experiment lives on `schwinns/gms-daemonset-restore-20260929`, separate from
the discovery baseline branch. `resident_daemonset.py` renders the resident
services; `resident_server.py` warms the reusable O_DIRECT transfer pool;
`resident_coordinator.py` validates readiness and publication; the runner adds
`--resident-gms` and optional `--preinstalled-cuda-host-path` modes.

Snapshot timing instrumentation source is `d1bb736d31d0`; the separate optional
network resolver is `b209afc33263`. The complete patch is in
`snapshot-prototype.patch`. Controller/runtime/executor tests pass. Seven
coordinator protocol tests and five transfer behavior tests cover single-use
triggering, identity rejection, failure cleanup, exact offsets and absence of
pretrigger payload I/O. All twelve GPU restores validate actual interpod handles and
inference. See [independent evidence validation](results/default-config/daemonset-study/validation.json),
[build validation](results/default-config/daemonset-study/build-validation.json),
raw per-trial logs, and the archived cluster drivers under `daemonset-study/drivers`.

Cleanup independently verified: original Snapshot stack/configuration restored,
Dynamo operator unchanged, experiment workloads/claims and private mounts removed,
NUMA balancing restored, and retained GLM/Qwen captures Ready. See
[cleanup verification](results/default-config/daemonset-study/cleanup-verification.json).
