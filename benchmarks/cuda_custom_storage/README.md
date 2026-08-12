<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CUDA CustomStorage standalone round-trip

This single-GPU harness validates the CUDA 13.4 CustomStorage checkpoint contract without
Snapshot, CRIU, GMS, HVBM, NIXL, or Kubernetes.

The target-PID CustomStorage operation and capability handling are benchmark-local. The
harness deliberately supplies its own single-buffer POSIX transfer instead of depending
on Snapshot's daemon, NIXL engine, or artifact format. Production integration belongs in
Snapshot only when the snapshot helper consumes this C++ path.

It forks a standalone CUDA workload that allocates a deterministic local device
buffer. The controller process then:

1. locks the workload;
2. requests a CustomStorage checkpoint;
3. requires one driver-provided device extent and reports its requested device ordinal,
   UUID, pointer, size, and stream;
4. copies that extent to one POSIX file through one bounded pinned buffer;
5. completes the checkpoint operation;
6. validates the file size before mutating restore state;
7. restores the extent and unlocks the workload;
8. asserts the documented RUNNING → LOCKED → CHECKPOINTED → LOCKED → RUNNING states; and
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
- CUDA 13.4 or newer toolkit headers and driver stubs under `CUDA_HOME`
- Permission to checkpoint the child process

The benchmark intentionally does not recreate CUDA 13.4 declarations when compiling
against older toolkit headers.

## Source layout

- `roundtrip.cpp` owns argument parsing and controller orchestration.
- `workload.cpp` owns the child CUDA workload and its typed result protocol.
- `storage.cpp` owns POSIX extent transfer and checkpoint/restore sequencing.
- `custom_storage_operation.cpp` owns the benchmark-local CUDA operation state.
- `roundtrip_common.cpp` owns process, pipe, watchdog, and CUDA utility code.

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

Expected output reports the CUDA-provided extent and ends with `roundtrip=passed`.
The output directory contains one `checkpoint.bin` file. Its metadata remains in
memory for the duration of the test; the harness does not define a persistent artifact
format.

## Verify truncated-artifact rejection

Use a new artifact directory:

```bash
artifact_dir="$(mktemp -d)/artifact"
timeout 130s benchmarks/cuda_custom_storage/cuda-custom-storage-roundtrip \
  --artifact-dir "${artifact_dir}" \
  --bytes 67108864 \
  --truncate-before-restore
```

The harness intentionally truncates the checkpoint file, rejects it before invoking CUDA
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
