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
