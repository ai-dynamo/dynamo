<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Results, 2026-09-28/29

This is a prototype using Dynamo GMS **V1**, SGLang's Engine API, and the qualified
Snapshot/PageBroker composition. It does not yet exercise Dynamo's distributed
frontend or a DynamoGraphDeployment. Large-model inference has passed with the constraints described below; the
RAM-staged measurements are not cold-NFS comparisons.

## PVC O_DIRECT with overlapping restore (2026-09-29)

The earlier **5.99 s** weight result was RAM-staged and is not a PVC result.
The corrected experiment loads the complete **448 GiB** matching V1 allocation
set from `snapshot-pvc`, with verified `O_DIRECT` on every payload descriptor.
No tmpfs staging or buffered-read override is used. Five old experimental
checkpoints were deleted through their controller, reclaiming about **1.9 TiB**;
metadata is archived in `results/default-config/retired-old-checkpoints.json`.

Default GLM communication settings now work before and after restore. Neither
multimem logits nor FlashInfer fused allreduce is explicitly disabled. The old
claim that their rendezvous sites were necessarily stalled was premature:
native samples showed TVM-FFI compilation/lock waits, and clean initialization
completed autotuning, graph capture, and coherent generation. Minimal 8-rank
symmetric-memory probes also passed without the shim, with the shim, and with
shim + live GMS weights. No cuInterpose change was needed for these successful
runs. The qualified single-node capture environment still retains its existing
`NCCL_IB_DISABLE=1` and monitoring/RAS settings. The existing automatic
FABRIC-to-POSIX-FD transport fallback also appears
in the previous baseline with the exact same image digest; it is separate from
the explicit communication overrides used in the historical RAM experiments.
See `default-config-debug.md` and its raw evidence for the profiling limitation.

Three alternating serialized/overlapping pairs use the same default-config
capture, claim, rank map, 112 payload files, and qualified Snapshot composition.
Each completes Berlin generation before readiness and a second Rayleigh request.
All durations below are seconds; overlapping phases must not be added together.

| Standard V1 NIXL loader, three trials each | Serialized | Overlapping | Change |
|---|---:|---:|---:|
| Pod creation to coherent readiness | 46.083 | 29.562 | **−16.521** |
| GMS first rank start to last publication | 23.397 | 25.031 | +1.634 |
| GMS server initialization, mean rank | 0.227 | 0.259 | +0.032 |
| Sockets bound to load return, mean rank | 21.261 | 22.770 | +1.509 |
| CRIU | 3.879 | 5.320 | +1.441 |
| CUDA phase | 7.761 | 8.729 | +0.968 |
| Summed native CUDA prepare/restore calls | 7.363 | 8.398 | +1.035 |
| Snapshot agent overall | 14.017 | 16.338 | +2.321 |

There is repeatable phase inflation consistent with contention, but overlap still
saves **35.9%** end to end. These measurements do not isolate the competing
resource (NFS transport, CPU, memory bandwidth or PCIe). The PVC mount uses
NFSv3, `nconnect=16`, 1 MiB read/write sizes, and one VAST address. The engine's
qualified PageBroker path retains its separate 32-connection, four-address
mount. `O_DIRECT` bypasses the client page cache; storage-server caches were not
flushed. CPU checkpoint files received the baseline's `POSIX_FADV_DONTNEED`
advice before every case. Images are already cached on the node.

The publication gate moved from ahead of Snapshot to immediately before the
captured engine calls `resume_memory_occupation`. A CPU-only sidecar verifies
every rank's server UUID and allocation IDs/sizes against the authoritative
restore plan. Only then does it write the gate. CRIU and CUDA restoration proceed
while GMS loads; no engine weight use is allowed before verification. "All eight
ranks published" means all eight V1 write sessions committed their exact captured
allocation sets, not just that eight processes or sockets started.

Against the historical qualified no-GMS baseline (four trials), CRIU was
3.655 s, CUDA phase 20.467 s, native prepare calls 10.076 s, and agent overall
25.143 s. The serialized PVC GMS run has similar CRIU time and substantially
less CUDA work; overlapping loads raise CRIU to 5.320 s. The CUDA phase and native
calls remain below the baseline. This is a historical comparison, not a fresh
alternating no-GMS control; the baseline's 25–28 s is **agent-only**, so it must
not be compared directly to the new pod-to-ready number.

The fused direct loader with four workers and NUMA affinity was also tested in
three alternating pairs on the ordinary PVC mount. Serialized means were
51.467 s pod-to-ready, 28.875 s all-rank load, 3.729 s CRIU and 7.808 s CUDA;
overlapping means were 33.474 s, 28.709 s, 4.844 s and 8.077 s respectively.
Its load variability (one serialized load took 32.045 s) prevents attributing
small loader deltas to overlap. It did not beat standard NIXL on that mount.

Charts: [serialized](results/default-config/nixl-serial-1/timeline.html) and
[overlapping](results/default-config/nixl-overlap-1/timeline.html), also PNG/SVG.
The CRIU duration is measured but its plotted position is approximate; CUDA bars
show PageBroker request intervals. `compare_trials.py` regenerates per-case
summaries and `results/default-config/comparison.json` from raw evidence.

## Completed gates

| Experiment | Result | Scope |
|---|---|---|
| Fresh-process cuInit, 8 visible B200s | 3.088–3.790 s | Five samples, same DRA node |
| Fresh-process cuInit, 1 visible B200 | 0.1245–0.1325 s | Five samples, rank-specific containers |
| Separate V1 server + NIXL loader, 4 GiB | 5.399 / 2.950 / 2.660 s | Synthetic, process startup through loader exit |
| Separate V1 server + direct loader, 4 GiB | 1.187 / 1.181 / 1.186 s | Same synthetic artifact, O_DIRECT + pinned CUDA copies |
| Fused server + direct loader, 2 GiB | 0.667 s | One smoke sample, entry to publication, warm node |
| Qwen3-0.6B TP2 engine restore | 3.134 s | Agent duration, **weights already published** |
| Qwen restore CRIU / CUDA | 1.537 / 1.193 s | Subphases of preceding measurement |
| Qwen trigger to PodReady | 7.379 s | Includes restore and first post-restore generation; excludes preload |

cuInit evidence is in `results/cuinit-*.jsonl`. Synthetic loader evidence is in
`results/loader-*.jsonl`. Synthetic byte checks sample the first, middle and final
4 KiB; they are not whole-artifact hashes. The probe's validation context remains
initialized after its first iteration, so later samples are not node-cold.

Qwen restored across DRA nodes, with rank 0 deliberately placed on destination
physical GPU 1 and rank 1 on physical GPU 0. Both GMS containers used local
`cuda:0`. Actual claim results were joined to ResourceSlices, the exact allocation
IDs/sizes and server UUIDs were checked before the restore annotation was written,
and the same map was passed to CUDA restoration and GPU mount remapping. The
restored engine generated Berlin and a coherent explanation of Rayleigh
scattering (`results/qwen/restored-inference.json`). Each rank had one 2 GiB GMS
weight allocation and about 3.617 GB of residual native CUDA checkpoint payload.
This small-model measurement is **not** a speedup comparison with the 25–28 s
GLM baseline.

Two Qwen setup failures preceded the successful run: the destination's injected
shim had the wrong file size, then the wrong executable mode. CRIU rejected both;
the engine checkpoint and restore must use matching shim contents and mode.

## Checkpointed GMS server: functional gate failed

A dedicated single-GPU V1 server was loaded with the matching Qwen allocation,
fenced, captured, restored to a different GPU, and unfenced after refreshing its
GPU UUID. Native restore took 0.828 s (CRIU 0.262 s, CUDA 0.473 s), but a subsequent
client export failed: `cuMemExportToShareableHandle: invalid argument`.
Successful metadata listing and a readiness sentinel therefore did not establish
functional GMS restoration.

Further trials enabled cuinterpose and an experimental raw-FD export path, and
then retained mapped VA anchors and rebuilt handles with
`cuMemRetainAllocationHandle` after restore. Both still failed the client export
check. The intended full-byte SHA256 verifier never reached its comparison.
These times are **failed experiments**, not a supported fast path. See
`results/qwen/server-*-bytes.txt` and captured manifests. A first attempt to
checkpoint a sidecar after the engine in the same source pod had already exited
also failed source-pod validation; dedicated server captures avoid that issue.

## GLM TP8, first qualified capture

All three restores produced Berlin and the same coherent Rayleigh-scattering
continuation as the source, with CUDA graphs enabled. Each rank's 28 matching
2 GiB allocations were published before engine restoration.

| Loader | All-rank preload span | Agent engine restore | Pod creation to coherent readiness |
|---|---:|---:|---:|
| Fused, first run | 7.197 s | 17.213 s | 67.137 s (35.624 s remote validation gap) |
| Fused, in-pod gate | 7.840 s | 14.802 s | 30.536 s |
| Default NIXL, in-pod gate | 10.329 s | 14.712 s | 34.005 s |

Preload span starts at the first GMS script timer and ends at the last rank's
publication. The timer follows interpreter startup and the initial standard-library
imports; pod timing includes that earlier startup. Pod timing includes scheduling, init/container startup, publication,
validation, restoration and the first generation. The last two trials perform
claim/artifact/server checks in the placeholder before exposing the publication
gate; they still have about 1.4 s of orchestration between observing the gate and
sending the restore trigger. They are individual trials, not statistical means.

**Storage constraint:** both NFS exports returned EDQUOT despite reported free
space. Retiring this experiment's failed server snapshots and small probes did
not clear it. Moving the 449 GiB matching weight set into a node-local 512 GiB
tmpfs cleared the quota. Its dedicated staging pod took **200.585 s** to copy;
that is explicitly outside these pre-staged restore measurements. The first copy
attempt in the low-memory agent cgroup was OOM-killed and retried in a staging
pod with a 640 GiB limit. No original artifact was removed until its complete
rank copy passed size checks. Native engine checkpoints still use the original
NFS/PageBroker stack. These results cannot establish a cold-storage speedup.

Engine GPU checkpoint payload was 147,958,267,904 bytes (18,494,783,488 per rank),
with about 57 GiB of CPU pages. Agent capture took 228.354 s, of which CRIU was
211.515 s. The subsequent experiment fixes the image's DSA cache hook name and trims
unused libc allocations before capture. Initial-capture artifacts in tmpfs were
retired after these completed trials to make room for the improved capture;
all initial plans, manifests and measured outputs remain in this branch.

### Communication compatibility

The historical RAM experiments bypassed two suspected rendezvous stalls: an
experimental plugin override routed logits gather to NCCL, and
`enforce_disable_flashinfer_allreduce_fusion=True` disabled fused allreduce.
The replacement backend for every allreduce was not instrumented. Stack locations
alone did not establish deadlock. The subsequent default-config PVC experiments
above completed both paths without these overrides, including restored inference;
the previous "underlying rendezvous hang" diagnosis is withdrawn.
When an earlier source pod released its claim, another namespace allocated all
8 source-node GPUs. No foreign workloads or agents were changed. The large-model
trial moved to s2877; the measured GLM restores used same-node capture/restore. Cross-node mapping
correctness was established only by the Qwen test above.

## GLM TP8, corrected cache-release hook

The image uses SGLang 0.5.16's `_create_index_buffers`, whereas the plugin expected
`_create_index_key_cache`. Matching the release/recreation hook to the image
reduced the native GPU payload to **44,275,073,024 bytes** (5,534,384,128 per rank),
70.1% below the first GMS capture. Trimming libc allocations had little effect:
CPU pages remained about 56 GiB. Agent capture took 237.282 s, including 226.557 s
CRIU and 9.190 s CUDA checkpoint. The exact 448 GiB GMS allocation set was saved
again for this capture; artifacts from the previous capture were not substituted.

The fused loader starts one V1 server per named DRA rank, uses local `cuda:0`, and
copies through pinned buffers. NUMA mode sets CPU affinity to the GPU's NUMA node
before creating loader threads and pinned buffers. All transfers finish and all
ranks publish before the engine restore trigger. The manifest/UUID/ID/size gate
runs inside the placeholder; no CUDA initialization is needed for that gate.

| Trial | All-rank preload span | Agent engine restore | Pod creation to coherent readiness |
|---|---:|---:|---:|
| fused-default4-2 | 6.850 s | 13.573 s | 31.583 s |
| fused-numa4-1 | 5.930 s | 13.335 s | 28.407 s |
| fused-numa4-2 | 5.968 s | 14.209 s | 28.599 s |
| fused-numa4-3 | 6.064 s | 14.800 s | 28.875 s |
| fused-numa8-1 | 7.635 s | 13.512 s | 32.248 s |
| fused-numa8-2 | 6.454 s | 15.015 s | 29.026 s |
| fused-ram-1 | 8.084 s | 13.639 s | 37.470 s |

The three four-worker NUMA trials averaged **28.627 s** pod-to-readiness
(range 28.407–28.875 s), with mean preload
5.987 s and mean agent restore 14.115 s.

For the 24 rank instances in the three four-worker NUMA trials:

| Boundary | Mean | Range |
|---|---:|---:|
| Script timer to both GMS sockets bound | 0.270 s | 0.193–0.411 s |
| Sockets bound to loader return/publication record | 4.077 s | 2.826–4.839 s |

The fused prototype runs server and loader in one process. The latter duration
is derived from the two logged boundaries and includes allocation, artifact
reads, CUDA copies, cleanup and V1 commit, plus the small serving-thread/backend
setup gap; it is not a pure DMA duration. Each rank loads 56 GiB. Rank script
starts were staggered by 1.69–1.73 s, explaining why the all-rank span is longer
than individual rank load times. "Published" means the matching allocation set
has completed loading and its V1 write session has committed, making the weights
available to readers. All eight such completions precede engine restoration.

These are warm-node, pre-staged-weight trials. The first two revised-capture
trials used `Always` image pulls; subsequent trials use `IfNotPresent` with the
same digest-pinned images already cached. Container startup skew contributes to
the all-rank preload span and pod duration. Four workers per rank with NUMA
affinity is the recommended prototype setting from this small sample; eight
workers did not improve the measured whole pipeline.

All revised-capture trials passed Berlin generation before readiness and a second
Rayleigh-scattering request afterward, with CUDA graphs enabled. They are
same-node GLM trials; the independent Qwen trial establishes cross-node rank
permutation correctness. Agent-only improvement must not be represented as an
end-to-end improvement over the prior 25.14 s **agent-only** baseline. These are historical RAM-only results. The later PVC/O_DIRECT experiment above
supersedes them for storage and default-communication validation; full Dynamo
frontend/worker deployment remains open.

Further work is a freshly alternating no-GMS control, storage-path tuning, and
production controller integration of the rank plan and publication gate.
Checkpointed GMS needs a working post-restore CUDA allocation export/import path
before performance comparison is meaningful.

## Prototype boundaries

- Snapshot patch is based on **d9b6bc72**, not PageBroker PR400's separate tree.
  `snapshot-prototype.patch` includes strict complete UUID-map validation and
  placeholder OCI `SNAPSHOT_CUDA_DEVICE_MAP` consumption.
- The external-GMS import opt-in tracks raw handles and mappings and refuses
  checkpoint while any remain. V1 sleep must release them before capture.
- An experimental dedicated `nvidia.com/gms-prototype-restore-from` annotation
  isolates this agent from host-cluster controllers that cannot see vcluster
  snapshots. No shared controller was disabled to achieve isolation.
- Direct loading and fused server startup live in the benchmark directory; these
  are not a supported production backend. The server checkpoint hook currently
  accesses private manager state and requires further lifecycle design.
- Prepublication UUID/ID/size checks do not replace validating restored inference.
  Marker files alone are insufficient, as the GMS-server experiments demonstrated.

Validation: 17 focused Python tests passed (minimal pytest configuration emits
unknown-marker warnings); CUDA/executor/podcontract Go tests passed; 14 Rust core
tests passed. Benchmark scripts pass Ruff. GMS server export remained broken in
an additional explicit ctypes dispatch experiment; no checkpointed-server timing
is counted as a functional result.

## Artifact retention and cleanup

Trial logs, rank plans, manifests, timing summaries, inference outputs, and
retired snapshot metadata are committed here. Large CUDA/CRIU payloads are not
Git artifacts. The temporary GLM RAM weight sets and their associated snapshots
were retired after the completed trials, so replay requires a fresh matching
capture. The smaller Qwen checkpoint and its durable matching artifacts were
retained. Experiment pods and GPU claims are released, and the original agent,
operator, configuration, node levers and private NFS mounts are restored; see
`results/cleanup.txt` for the recorded cleanup outcome.
