<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Results, 2026-09-28/29

This is a prototype using Dynamo GMS **V1**, SGLang's Engine API, and the qualified
Snapshot/PageBroker composition. It does not yet exercise Dynamo's distributed
frontend or a DynamoGraphDeployment. No large-model speedup is claimed until the
GLM correctness and timing gates below complete.

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

## Large-model qualification in progress

GLM-5.2-NVFP4 TP8 loaded and committed its weights through V1, but stalled in
PyTorch symmetric-memory rendezvous, first for the logits multimem gather and
then for FlashInfer allreduce fusion. Python stack sampling identified the call
sites. An experimental plugin patch forces the logits gather to NCCL; the third
configuration also sets `enforce_disable_flashinfer_allreduce_fusion=True` while
retaining CUDA graphs. These are communication-layout changes from the baseline.

When a stalled source pod released its claim, another namespace allocated all
8 source-node GPUs. No foreign workloads or agents were changed. The large-model
trial moved to s2877; it will use same-node capture/restore. Cross-node mapping
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
