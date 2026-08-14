<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Same-node POSIX CUDA VMM and multicast checkpoint and restore

This interposer implements checkpoint and restore for CUDA VMM allocations
and complete CUDA multicast groups shared between processes on one node with
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
`snapshotctl` injects the job identity and interpose launcher. Single-GPU
POSIX VMM checkpointing does not require a CUDA launch-job file. Multi-GPU
checkpointing must also launch through the existing
`cuda-checkpoint --launch-job` path; manual `snapshotctl` users must pass both
`--cuda-checkpoint-wrap` and `--cuda-vmm-interpose`. The launch-job wrapper is
outermost and `snapshot-cuda-vmm-launch` remains the inner wrapper.
Operator-created multi-GPU checkpoint jobs enable the launch-job wrapper
automatically; VMM interposition remains a separate opt-in.

Legacy CUDA IPC remains driver-owned and is unsupported by this shim.

`DYN_SNAPSHOT_PARTICIPANT_ID` may provide a stable 32-character lowercase
hex participant ID. Otherwise, the shim creates one when the process starts.

The orchestrator must externally quiesce the application at the checkpoint
boundary. Allocation sharing, import, mapping, access updates, multicast
setup/teardown, kernel work, and communication-library setup must not be in
flight.

During ordinary execution:

- CUDA generic handles for tracked POSIX resources are tagged logical tokens.
  The handle table distinguishes unicast allocations from multicast objects;
  corresponding real driver handles remain internal.
- Untracked CUDA generic handles remain real driver handles.
- A POSIX-capable unicast allocation becomes checkpoint-managed when its shim
  capability is exported.
- POSIX exports return sealed capability FDs containing the job, creator,
  resource, endpoint, and authorization identities.
- A capability import resolves through the creator's local Unix endpoint. A
  transient raw CUDA FD is passed with `SCM_RIGHTS`, imported, closed
  immediately, and never returned to the application.
- A raw external POSIX import is passed directly to CUDA and is not tracked.
- POSIX multicast create/export/import, device membership, `BindMem`,
  `BindAddr`, unbind, generic mapping, and mapping access are passed to CUDA
  and recorded. Device-explicit `_v2` bind variants are included when the
  build headers expose them.

Handle ownership is explicit:

| Source | Application receives | Managed table | Driver-handle owner |
| --- | --- | --- | --- |
| POSIX-capable create, tracked retain, or capability import | Tagged logical token | Logical token to UC or MC resource | Shim |
| Raw external import or other pass-through | Real driver handle | None | Application |
| UC checkpoint carrier | Nothing | No separate entry | Shim |
| Transient MC teardown/restore handle | Nothing | No durable entry | Shim |

`posix.c` owns the sealed capability format and remote creator exchange.
`multicast.c` owns the complete multicast lifecycle: identities and handles,
team membership, UC bindings, mappings and access, teardown, fresh same-node
FD exchange, reconstruction, and validation. `interpose.c` owns generic CUDA
interception and unicast allocation state.

## Multicast is a topology overlay

A multicast object is topology layered over restored unicast members. It does
not own separately checkpointed bytes:

```text
multicast object -> UC member A -> creator A's native cuCheckpoint bytes
                 -> UC member B -> creator B's native cuCheckpoint bytes
```

The recorded multicast creator is only the deterministic control-plane
participant that creates and exports a fresh object during restore. It is not
a byte saver or checkpoint anchor. No old multicast object or handle remains
at the native checkpoint boundary. If the application released its multicast
handle but retained a mapping, the shim derives only a short-lived teardown
handle from that mapping and releases it before native checkpoint.

## Checkpoint and restore ordering

Before native CUDA checkpoint, the coordinator validates the complete
participant topology. Every original `cuMemCreate` participant remains the
saver for its unicast allocation and must still have a live creator handle or
mapping (the creator-anchor invariant).

Multicast teardown is the first destructive stage:

1. unmap every multicast VA without freeing its reservation;
2. unbind every recorded UC member;
3. release every imported/creator multicast handle and old object;
4. retain only stable logical identity and topology metadata.

The coordinator waits for every participant to report multicast detached.
Only after that barrier does the shim normalize unicast allocations. Each
creator keeps one unmapped local carrier, all managed UC mappings are removed
without freeing VA reservations, and importer handles are released. Native
`cuCheckpoint` alone saves and restores unicast allocation bytes.

On restore, native CUDA first restores creator allocations. CUDA processes are
then unlocked so the shim can reconstruct topology while the application
remains behind the restore-complete sentinel:

1. restore UC creator handles and mappings;
2. import current-generation UC handles and restore importer mappings;
3. have the recorded MC metadata creator create a fresh POSIX object;
4. import that fresh object in all other participants;
5. add every recorded device to the complete team before replaying bindings;
6. replay `BindMem` and `BindAddr` against restored UC handles or addresses;
7. map each original multicast VA and restore access;
8. install logical-to-current-handle translation and validate before resume.

The coordinator rejects partial groups. All checkpoint participants must
report the multicast object, exactly one metadata creator, the complete unique
device set, and bindings for every device. Reconstruction and final
validation fail closed. There is no rollback after checkpoint preparation
mutates CUDA state.

The native coordinator atomically writes `cuda-vmm.state`, its opaque durable
topology sidecar, in the checkpoint directory. Go only orders native VMM
prepare and restore; it does not serialize or inspect CUDA topology. The
sidecar contains no raw FDs or allocation contents. Its MC records contain
only stable identity and creator, properties, devices and participants, UC
member identities, binding offsets/sizes/flags/API variant, mapping
VAs/offsets/flags/access, and logical-handle liveness.

## Interception and symbol resolution

The shim covers direct Driver API symbols and these resolver paths:

- explicit `dlsym()` lookups from `libcuda.so` and `libcudart.so`;
- `cuGetProcAddress`, `cuGetProcAddress_v2`, and `_ptsz`;
- `cudaGetDriverEntryPoint` and `cudaGetDriverEntryPointByVersion`, including
  the `_ptsz` exports present in CUDA 13.

The multicast surface includes create, add-device, `BindMem`, `BindAddr`,
unbind, granularity, generic export/import/release/retain/property consumers,
and generic map/unmap/access consumers. Resolver requests at CUDA 13.1 or
later are directed to the device-explicit bind ABI when the installed headers
expose it.

## Testing

Run the native interposer integration tests from the repository root:

```bash
cuda-checkpoint --launch-job \
  uv run --project deploy/snapshot/cmd/cuda-vmm-interpose/tests \
    pytest deploy/snapshot/cmd/cuda-vmm-interpose/tests/test_cucheckpoint.py -v
```

The project contains a non-multimem POSIX baseline and a two-GPU multicast
case. The baseline sets `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` and uses the
non-multicast collective, although PyTorch 2.11 may still initialize dormant
multicast topology during rendezvous. The multicast case requires PyTorch
symmetric memory to select multicast, runs a real multimem collective,
replaces its local UC member with a `cuMulticastBindAddr` binding, captures the
collective in a CUDA graph, performs native `cuCheckpoint`, and replays the
same graph after reconstruction. It fails rather than accepting PyTorch's
non-multicast fallback. The test also requires and verifies the shared
`CUDA_CHECKPOINT_JOB_FILE` established by `--launch-job`; the pytest controller
and both CUDA workers use that same native checkpoint job.

## Qualification

Disjoint GPU qualification is limited to source GPUs 0/1 restored onto
destination GPUs 4/5 with user-mode `libcuda` 615.65 and kernel RM 595.58.03.

## Limitations and future extensions

> [!WARNING]
> Raw external POSIX imports are passed through and are not tracked or
> reconstructed. Every mapping and handle from such an import must be unmapped
> and released before checkpoint prepare, as the GMS saver/sleep flow does.
> If retained, native checkpoint may fail or restore may later see stale or
> incomplete sharing.

The shim reserves generic handles whose top 16 bits are `0xd94d` for logical
tokens. If CUDA returns a raw pass-through handle in that range, the shim
releases it and returns `CUDA_ERROR_INVALID_HANDLE`.

### Potentially silent gaps

- An un-interposed raw export or import bypasses bookkeeping.
- Raw FD aliases made with `dup`, `fcntl`, or out-of-adapter `SCM_RIGHTS` are
  invisible and can later import stale state.
- An unshimmed broker or helper makes the sharing topology incomplete.
- An uncovered CUDA runtime symbol-resolution path can bypass wrappers.
- An old generic handle retained by an uncovered consumer can be stale.
- NVIDIA-specific `ioctl`, `fstat`, or `poll` semantics on a capability FD are
  unsupported.
- `BindAddr` ranges outside tracked VMM mappings rely on native
  `cuCheckpoint` to restore their underlying allocation at the identical VA.
- Access updates that do not exactly match a tracked mapping are passed
  through but are not recorded for restore.
- Checkpoint during sharing or communicator construction is unsupported.

Applications must finish setup and externally quiesce the complete process
group first.

Children made with `fork()` and no subsequent `exec()` lazily receive a fresh
random process-local participant identity, child-PID control socket and
thread, and empty allocation, handle, mapping, and multicast bookkeeping on
their first intercepted VMM operation. This supports the fork-before-CUDA
initialization lifecycle used by vLLM. An explicitly configured parent
participant ID is never reused by a forked child.

Forking after the shim has tracked CUDA state is unsupported. The child
deliberately discards inherited bookkeeping without freeing resources or
calling CUDA.

### Fail-closed limitations

The coordinator fails closed for a missing UC creator anchor, incomplete
participants or topology, more than eight access descriptors, reconstruction
failures, or final validation failures. Multicast additionally fails closed
for non-POSIX handle types, cross-node or partial groups, duplicate or missing
devices, missing member bindings, or more than one metadata creator.

Only complete same-node POSIX multicast groups are supported. The shim does
not coordinate a subset of an NCCL/PyTorch team or restore one participant
independently. A multicast identity is tracked in one CUDA context per
process.

### Future compatibility

FABRIC/MNNVL multicast belongs behind a future `fabric.c` boundary with IMEX
authorization and a cross-node rendezvous service. FABRIC/IMEX handles,
raw-FD compatibility, and multi-node rendezvous are not implemented here.
Creator endpoints and authorization tokens remain transient live/capability
data and are not sidecar fields. The sidecar intentionally contains neither
raw FDs nor allocation bytes.
