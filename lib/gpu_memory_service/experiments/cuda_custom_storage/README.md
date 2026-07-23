<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GMS CUDA CustomStorage round-trip experiment

This experiment checks whether CUDA CustomStorage can checkpoint and restore GPU
memory owned by a GMS server. The interesting part is that GMS owns the
allocation, while a separate client imports the allocation and writes its
contents.

```text
writer ──creates, imports, and writes──> GMS allocation
                                           │
controller ──checkpoints GMS server────────┘
                                           │
reader ──reconnects, imports, and verifies─┘
```

## What the experiment does

The controller starts a real GMS server as a child process, then runs this
sequence:

1. A writer connects through the existing GMS RPC API, creates one allocation,
   fills it with a deterministic byte pattern, stores identifying metadata,
   unmaps it, and commits it.
2. The controller checkpoints only the GMS server and copies the CUDA-provided
   CustomStorage extent into an artifact file.
3. The controller restores the same server process from that extent.
4. A new reader process reconnects, imports the restored allocation, and verifies
   allocation metadata, layout hash, and every byte through CUDA driver copies.

Checkpoint bytes do not pass through GMS RPC or the controller's synchronization
pipes. CUDA exposes the serialized GPU state through a device pointer and stream;
the controller copies those bytes to storage, then copies them back to CUDA
during restore.

## Files

- `gms_roundtrip.cpp` controls the GMS server, CUDA checkpoint/restore state
  transitions, artifact I/O, timeouts, and result checks.
- `gms_client.py` acts as either the writer before checkpoint or the fresh reader
  after restore using the existing GMS client API.
- `cuda_checkpoint_compat.h` defines the unreleased CustomStorage declarations
  needed by the test environment.
- `Makefile` builds the controller.

## Build and run

From the repository root in an environment containing the GMS Python dependencies,
NumPy, and a CUDA driver with the 13.4 CustomStorage API:

```bash
make -C lib/gpu_memory_service/experiments/cuda_custom_storage

artifact_parent="$(mktemp -d)"
timeout 130s \
  lib/gpu_memory_service/experiments/cuda_custom_storage/gms-custom-storage-roundtrip \
  --artifact-dir "${artifact_parent}/artifact" \
  --socket-path "${artifact_parent}/gms.sock" \
  --repo-root "${PWD}" \
  --bytes 67108864
```

Success ends with `gms_custom_storage_roundtrip=passed`. The artifact directory
contains the CUDA extent and a manifest recording the requested GPU UUID, CUDA
storage pointer, stream, and extent size.

## Result

The 64 MiB probe passed on an nscale B200 using driver 595.58.03 and the
`nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` image, whose forward-compatible
driver reported CUDA 13.4. CUDA returned one 64 MiB CustomStorage extent. The
fresh reader recovered the original allocation ID, layout slot, layout hash,
and every byte after the complete checkpoint and restore state transition.

## What this proves

- CUDA CustomStorage includes the GPU allocation owned by the GMS server.
- The checkpoint bytes can be copied to a file and supplied back to CUDA.
- The restored allocation keeps its contents.
- A fresh GMS client can reconnect, import the allocation, and read the expected
  metadata and bytes.

## What this does not prove

- The GMS server's host state can be restored.
- The original GMS server can be replaced by a new or restored process.
- The artifact format or compatibility declarations are suitable as permanent
  interfaces.
- The same behavior works with multiple allocations, GPUs, or clients.
