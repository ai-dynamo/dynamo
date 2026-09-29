<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Results, 2026-09-28/29

This is a prototype using Dynamo GMS **V1**, SGLang's Engine API, and the qualified
Snapshot/PageBroker composition. It does not yet exercise Dynamo's distributed
frontend or a DynamoGraphDeployment. Large-model inference has passed with the constraints described below; the
RAM-staged measurements are not cold-NFS comparisons.

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

Preload span starts at the first GMS Python process and ends at the last rank's
publication. Pod timing includes scheduling, init/container startup, publication,
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
211.515 s. The next experiment fixes the image's DSA cache hook name and trims
unused libc allocations before capture. Initial-capture artifacts in tmpfs were
retired after these completed trials to make room for the improved capture;
all initial plans, manifests and measured outputs remain in this branch.

### Communication compatibility

GLM-5.2-NVFP4 TP8 loaded and committed its weights through V1, but stalled in
PyTorch symmetric-memory rendezvous, first for the logits multimem gather and
then for FlashInfer allreduce fusion. Python stack sampling identified the call
sites. An experimental plugin patch forces the logits gather to NCCL; the third
configuration also sets `enforce_disable_flashinfer_allreduce_fusion=True` while
retaining CUDA graphs. These are communication-layout changes from the baseline.

When a stalled source pod released its claim, another namespace allocated all
8 source-node GPUs. No foreign workloads or agents were changed. The large-model
trial moved to s2877; the measured GLM restores used same-node capture/restore. Cross-node mapping
correctness was established only by the Qwen test above.

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
