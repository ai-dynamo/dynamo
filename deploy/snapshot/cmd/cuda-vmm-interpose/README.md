<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Same-node POSIX CUDA VMM checkpoint and restore

This interposer implements checkpoint and restore for CUDA VMM allocations
shared between processes on one node with
`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`.

## Contract

Launch every participating CUDA process with:

```bash
export DYN_SNAPSHOT_JOB_ID=<32-lowercase-hex-job-id>
snapshot-cuda-vmm-launch COMMAND [ARG...]
```

For checkpoint jobs created by `snapshotctl`, enable this wrapper with:

```bash
snapshotctl checkpoint --cuda-vmm-interpose [OTHER FLAGS]
```

This experimental flag is an explicit launch-time opt-in because the shim
must be preloaded before the application creates CUDA VMM allocations.
`snapshotctl` injects the job identity and interpose launcher. The interposer
does not depend on `cuda-checkpoint --launch-job` or its job file.

`DYN_SNAPSHOT_PARTICIPANT_ID` may provide a stable 32-character lowercase
hex participant ID. Otherwise, the shim creates one when the process starts.

The orchestrator must externally quiesce the application at the checkpoint
boundary. Allocation sharing, import, mapping, access updates, kernel work,
communication-library setup, and teardown must not be in flight.

During ordinary execution:

- CUDA generic allocation handles are real driver handles.
- POSIX exports return sealed capability FDs containing the job, creator,
  allocation, endpoint, and authorization identities.
- A POSIX import resolves the capability through the creator's local Unix
  endpoint. A transient raw CUDA FD is passed with `SCM_RIGHTS`, imported,
  closed immediately, and never returned to the application.

`posix.c` owns the sealed capability format and remote creator exchange.
`interpose.c` owns CUDA interception, allocation state, and local raw exports.

Before native CUDA checkpoint, the snapshot agent asks the native coordinator
to validate the complete participant topology. The original `cuMemCreate`
participant is always the saver and must still have a live creator handle or
mapping (the creator-anchor invariant). The shim then puts every allocation
in checkpoint normal form: one unmapped creator-local carrier remains,
managed mappings are removed without freeing VA reservations, and importer
handles are released. Native CUDA alone saves and restores allocation bytes.

On restore, native CUDA first restores creator allocations. CUDA processes are
then unlocked so the shim can replay VMM topology while the application stays
behind the restore-complete sentinel. Creators are reconstructed before
importers; importers resolve fresh creator exports. The shim remaps every
allocation at its saved VA, restores effective access and required handle
translations, and performs final topology validation. Any failed validation
fails closed. There is no rollback after checkpoint preparation mutates CUDA
state.

The native coordinator atomically writes `cuda-vmm.state`, its opaque durable
topology sidecar, in the checkpoint directory. Go only orders native VMM
prepare and restore; it does not serialize or inspect the topology. The
sidecar contains no raw FDs or allocation contents.

## Testing

Run the native interposer integration test from the repository root:

```bash
uv run --project deploy/snapshot/cmd/cuda-vmm-interpose/tests \
  pytest deploy/snapshot/cmd/cuda-vmm-interpose/tests/test_cucheckpoint.py -v
```

## Qualification

Disjoint GPU qualification is limited to source GPUs 0/1 restored onto
destination GPUs 4/5 with user-mode `libcuda` 615.65 and kernel RM 595.58.03.

## Limitations and future extensions

> [!WARNING]
> Raw CUDA/RM export FDs outside the capability path are not enumerated or
> repaired. Checkpoint and restore can appear successful before a later
> invisible FD import references old allocation state. The result can be split
> allocation state, data divergence, or GPU memory leakage.

### Potentially silent gaps

The following gaps can bypass bookkeeping or silently preserve stale state:

- An un-interposed raw export or import call bypasses bookkeeping.
- Raw FD aliases created with `dup` or `fcntl` are invisible.
- Raw FDs cached, queued, or transferred with `SCM_RIGHTS` outside the adapter
  can later import stale pre-restore state.
- An unshimmed broker or helper makes the sharing topology incomplete.
- An uncovered CUDA runtime symbol-resolution path can bypass wrappers.
- An old generic handle retained by an uncovered consumer can be stale.
- NVIDIA-specific `ioctl`, `fstat`, or `poll` semantics on a capability FD are
  unsupported.
- Checkpoint during sharing or communicator construction is unsupported.

Applications must finish setup and externally quiesce the complete process
group first.

Children created with `fork()` and no subsequent `exec()` lazily receive a fresh
random process-local participant identity, a child-PID control socket and thread,
and empty allocation, handle, and mapping bookkeeping on their first intercepted
VMM operation. This supports the fork-before-CUDA-initialization lifecycle used
by vLLM. An explicitly configured parent participant ID is never reused by a
forked child.

Forking after the shim has tracked VMM allocations, handles, or mappings is
unsupported. The child deliberately discards its inherited copies without
freeing them or calling CUDA; inherited tracked CUDA state is not reconciled or
preserved.

### Fail-closed limitations

The coordinator fails closed for a missing creator anchor, incomplete
participants or topology, more than eight access descriptors, access updates
that do not exactly match a tracked mapping range, and reconstruction or final
validation failures. There is no rollback after checkpoint preparation mutates
CUDA state.

Multicast objects are not checkpointable yet. While interposition is enabled,
all `cuMulticast*` entry points return `CUDA_ERROR_NOT_SUPPORTED`, including
calls resolved through `dlsym()` or `cuGetProcAddress()`. NCCL workloads must set
`NCCL_NVLS_ENABLE=0`; otherwise NCCL initialization fails intentionally rather
than creating multicast state that cannot be restored.

### Future compatibility

FABRIC/IMEX handles, multicast checkpointing, raw-FD compatibility, and
multi-node rendezvous are not implemented. The durable sidecar preserves
allocation type, location, requested-handle type, mapping, access, and creator
topology needed for those future compatibility seams. Creator endpoints and
authorization tokens remain transient live/capability data and are not sidecar
fields. The sidecar intentionally contains neither raw FDs nor allocation
bytes.
