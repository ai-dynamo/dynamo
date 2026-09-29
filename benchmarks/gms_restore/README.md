<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# GMS V1 restore experiments

Branch: `schwinns/gms-snapshot-restore-20260928`, baseline Dynamo `f6732746a809`.

Purpose: move matching-capture weight transfers ahead of the serial engine CUDA
restore chain. Compare fresh per-rank GMS V1 servers plus artifact loaders with
checkpointed per-rank GMS V1 servers. Preserve logical rank, captured ordinal,
allocation IDs/sizes, socket identities and an explicit source/destination UUID map.
Each GMS container receives one named DRA request and uses local CUDA ordinal 0.
Engine ranks retain their captured device ordinals.

Reference evidence: `~/checkpoint-e2e-evidence/restore-profile-20260925/`.
Latest PR400 comparison: baseline mean 25.83 s (6 runs), optimized broker mean
25.14 s (4 runs), coherent inference in all runs. These are agent restore times,
not pod scheduling-to-first-token times. Workload: SGLang GLM-5.2 TP8, 8 B200s.
Composition: Snapshot d9b6bc72; PageBroker ca369464 (PR400), payload-first serial
PREPARE, patched CRIU cgroup dedupe, CUDA_DEVICE_MAX_COPY_CONNECTIONS=2,
NUMA balancing disabled, NFS read-ahead 16 MiB. Prior node changes were reverted.

Experiments record container/process startup, cuInit, server admission, allocation,
transfer, publication, CRIU, serial engine CUDA restore, wake, and first coherent
inference separately. Total elapsed time includes GMS startup and preload. A
weights-preloaded engine-only measurement is reported separately, never compared
as total elapsed time against the 25–28 s baseline.

Initial gates: DRA one-vs-eight GPU cuInit microbenchmark; GMS V1 artifact roundtrip;
explicit UUID permutation correctness; engine snapshot and coherent inference;
then repeated timing and checkpointed-GMS comparison.

## Recorded prototype

[RESULTS.md](RESULTS.md) records timing boundaries, failed trials and storage /
communication differences. `results/` holds exact pod/claim/capture plans, logs
and inference outputs. Large GPU payloads are not stored in Git.

Dynamo changes add independent V1 socket and artifact device ordinals. The
benchmark's fused server uses V1 admission/publication and an experimental pinned
copy backend. `resolve_plan.py` joins the named DRA requests to ResourceSlice UUIDs
and checks manifest hashes and allocation IDs/sizes. `publication_gate.py` performs
those checks inside the placeholder, checks every server UUID and allocation list,
and gates engine restoration on all publications. The agent consumes the same
explicit UUID map from `SNAPSHOT_CUDA_DEVICE_MAP` in the placeholder OCI spec.

`snapshot-prototype.patch` is against Snapshot d9b6bc72. Its worktree commits are
865def41 (rank map), e021a725 (quiescent raw GMS imports) and 006a3823 (isolated
experiment annotation and server raw-export experiment). The engine captures use
the e021a725 core; the Go agent includes the isolated annotation from 006a3823.
Preserve the shim's bytes and executable mode between capture and restore. The
server export experiment is not a working checkpointed-GMS implementation.

The cluster harness expects the qualified stack and matching source artifacts to
already be staged. Set `GMS_VCLUSTER_KUBECONFIG` to a private kubeconfig; it is never
recorded here. Host `kubectl` must reach translated pods. For an allocated and held
claim, the repeated-trial entry point is:

```sh
python run_glm_restore.py --case nixl-pvc-N --backend nixl --storage pvc --same-claim --fast-gate
```

`--capture-dir` selects a different captured generation. The harness intentionally
pins the experiment's node/names and RAM artifact mount; adapt the manifests to a
new cluster. Read the storage caveats before comparing these trials with the prior
cold NFS results. `sglang-evidence-image.patch` is an **image-specific experiment**:
it maps DSA index-buffer hooks to this image's older SGLang method names, disables
its stalled multimem logits path, and trims unused libc pages at release. The
source manifest also disables FlashInfer allreduce fusion. These changes retain
CUDA graphs. The logits fallback uses NCCL; disabling FlashInfer fusion does not
by itself establish the backend selected for every replacement allreduce.

For the corrected capture and tuned loader, pass:

```sh
python run_glm_restore.py --capture-dir results/glm/index-fix \
  --case fused-numa4-N --storage tmpfs --backend fused --workers 4 --numa --same-claim --fast-gate
python summarize.py results/glm
```

Each trial must start with the preceding restore pod removed while a separate
holder keeps the named claim allocated. Save that trial's agent log as `agent.txt`
before the next restore; the summarizer uses the final matching restore entry.
`summary.json` separates all-rank publication span, agent duration, and pod time.

The default-config follow-up uses durable artifacts on `snapshot-pvc`, with
`O_DIRECT` verified using `fcntl(F_GETFL)` for every payload file opened. Use
`--capture-dir results/default-config --storage pvc`. Both the standard NIXL
loader (`--backend nixl`) and fused prototype (`--backend fused`) use the V1
artifact API; only the latter substitutes the transfer backend.

`--overlap --fast-gate` starts Snapshot once the containers are running and moves
the UUID/allocation publication validator into a CPU-only sidecar. It requires a
capture whose engine waits for `/gms/all-ready` before resuming memory occupation.
Without `--overlap`, all publication and validation finish before Snapshot starts.
The flag changes scheduling, not the rank mapping or the capture artifacts.

`--qualified-pvc-mount` gives only GMS loaders a read-only bind of the existing
qualified NFS mount at `/var/lib/snapshot-restore-perf-nfs`. This is the same
`snapshot-pvc` export and exact files, with 32 connections and four NFS addresses;
it is not RAM staging. The captured engine and publication validator retain their
ordinary PVC volume. The shared PV mount configuration is not changed. The trial
records `findmnt` output and requires an NFS mount of the expected export with
`nconnect=32`. This option assumes that the qualified node setup is already done.

After collecting each case's agent/main logs, generate summaries and charts:

```bash
python3 benchmarks/gms_restore/compare_trials.py benchmarks/gms_restore/results/default-config
python3 benchmarks/gms_restore/timeline.py benchmarks/gms_restore/results/default-config/nixl-overlap-1
```

The chart generator requires Matplotlib. Generated HTML embeds the SVG and works
without network access. CRIU placement is approximate; durations and CUDA broker
intervals come from logged measurements. See `RESULTS.md` for historical versus
current comparisons and cache-state limits.

Add `--isolated-pvc-transport` to bind the identically configured second mount at
`/var/lib/schwinns-gms-0928/gms-pvc-nfs` instead. Both private mounts use
`nosharecache,nosharetransport`. This tests separate client transports for GMS and
PageBroker while retaining the same PVC export, files, and O_DIRECT requirement.
The benchmark setup/cleanup must create/unmount this experiment-owned mount.

`--early-trigger --overlap --fast-gate` places restore intent on pod creation,
validates the held claim's rank mapping beforehand, and performs evidence-only
host-pod lookup after readiness. Snapshot can then resolve its target container
without waiting for every GMS container's Running status. The captured wake gate
still protects weight use. Snapshot patch bc42fb31 is required to avoid holding
stale pod status through the runtime lookup timeout in this vcluster.
See `DGD-RESTORE.md` for current operator behavior and the production integration
plan; these tests are still the Engine API prototype, not a complete DGD run.
