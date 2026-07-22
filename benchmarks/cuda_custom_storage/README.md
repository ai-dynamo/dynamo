<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CUDA CustomStorage standalone round-trip

This single-GPU harness validates the CUDA 13.4 CustomStorage checkpoint contract without
Snapshot, CRIU, GMS, HVBM, NIXL, or Kubernetes.

It forks a standalone CUDA workload that allocates a deterministic local device
buffer. The controller process then:

1. locks the workload;
2. requests a CustomStorage checkpoint;
3. requires one driver-provided device extent and records its requested device ordinal,
   UUID, pointer, size, and stream;
4. copies those checkpoint extents to POSIX files through a bounded pinned buffer;
5. completes the checkpoint operation;
6. validates the complete artifact before mutating restore state;
7. restores every extent and unlocks the workload; and
8. asserts the documented RUNNING → LOCKED → CHECKPOINTED → LOCKED → RUNNING states;
9. asks the workload to verify its original bytes and execute another CUDA operation.

An internal 120-second watchdog kills the child workload and exits the controller if a
driver call, pipe operation, or workload exit hangs. Multi-GPU restore is an explicit
non-goal of this proof.

The driver-provided CustomStorage pointer is checkpoint storage, not necessarily an
application allocation address. Correctness is therefore established by application
byte verification after a complete checkpoint/restore cycle, not by matching the
application pointer against the returned extent pointer.

## Requirements

- Linux on x86-64
- CUDA driver exposing the CUDA 13.4 CustomStorage API
- CUDA toolkit headers and driver stubs under `CUDA_HOME`
- Permission to checkpoint the child process

The compatibility header temporarily permits a build image with pre-13.4 headers when
the runtime driver already exposes the 13.4 symbol. Prefer the released CUDA 13.4
headers as soon as they are available.

## Build

```bash
make -C benchmarks/cuda_custom_storage
```

## Verify a normal round-trip

The artifact directory must not already exist:

```bash
artifact_dir="$(mktemp -d)/artifact"
timeout 130s benchmarks/cuda_custom_storage/cuda-custom-storage-roundtrip \
  --artifact-dir "${artifact_dir}" \
  --bytes 67108864
```

Expected output lists the CUDA-provided extents and ends with `roundtrip=passed` plus
separate CUDA API, D2H, filesystem write/fsync, filesystem read, H2D, operation
completion, and total timings.
The artifact contains `manifest.txt` plus one `extent-N.bin` file per returned device
extent.

## Verify truncated-artifact rejection

Use a new artifact directory:

```bash
artifact_dir="$(mktemp -d)/artifact"
timeout 130s benchmarks/cuda_custom_storage/cuda-custom-storage-roundtrip \
  --artifact-dir "${artifact_dir}" \
  --bytes 67108864 \
  --truncate-before-restore
```

The harness intentionally truncates the first extent, rejects it before invoking CUDA
restore, terminates the now-checkpointed test workload, and reports
`corruption_check=passed`.

## Verify that restored state depends on artifact bytes

Use another new artifact directory:

```bash
artifact_dir="$(mktemp -d)/artifact"
timeout 130s benchmarks/cuda_custom_storage/cuda-custom-storage-roundtrip \
  --artifact-dir "${artifact_dir}" \
  --bytes 67108864 \
  --corrupt-before-restore
```

This preserves the extent's declared size but overwrites its contents. The test passes
only if CUDA restore/completion rejects the bytes or the workload explicitly rejects
the restored application state.

## Explicit non-goals

- Snapshot or CRIU orchestration
- GMS/HVBM allocation ownership or metadata
- NIXL and transfer-performance optimization
- daemonization or multi-tenant policy
- portable GPU remapping
- multi-GPU checkpoint or restore
- production artifact compatibility

Those layers should consume or extend the verified CUDA contract only after this
standalone behavior is reproduced on the target CUDA 13.4 environment.
