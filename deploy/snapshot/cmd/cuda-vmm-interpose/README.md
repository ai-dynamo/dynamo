<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Snapshot CUDA POSIX-FD VMM interposer

This launcher-scoped `LD_PRELOAD` interposer checkpoints complete CUDA Driver
API VMM allocations shared through POSIX file descriptors. It is transparent
to inference frameworks: applications do not call checkpoint hooks. The
interposer composes with `cuda-checkpoint --launch-job`; Redis replaces the
CUDA job-file role only for shim-managed POSIX-FD VMM state. CUDA checkpoint
continues to own unrelated CUDA state, including legacy CUDA IPC.

## Enablement and flow

Opt in with `snapshotctl checkpoint --cuda-vmm-interpose`. The target command
is composed as:

```text
cuda-checkpoint --launch-job dynamo-snapshot-cuda-vmm-launch APPLICATION
```

The launcher prepends
`/usr/local/lib/dynamo/libdynamo_snapshot_cuda_vmm.so` to `LD_PRELOAD` and
executes the application. The image does not use `/etc/ld.so.preload`.

The application process tree must already be quiescent before the coordinator
issues the first VMM `INSPECT` or any other checkpoint-prepare operation. It
must remain quiescent throughout graph capture and all importer/owner detach
operations. At that boundary:

1. Each shim reports owner/importer mappings to the Snapshot coordinator over
   its runtime `/snapshot-control/cuda-vmm-PID.sock`.
2. The coordinator groups this captured generation by the allocation UUID in
   each shim record, assigns logical resource IDs, and validates exactly one
   owner per UUID.
3. Each owner's bytes are copied once into Redis.
4. Importer mappings are detached in resource/participant order, followed by owners.
   CUDA virtual-address reservations remain in place.
5. The coordinator records metadata and content in Redis, runs `SAVE`, and
   copies the RDB into the checkpoint as `cuda-vmm.rdb`.
6. Normal `cuda-checkpoint` and CRIU capture all remaining supported state.

At restore, CUDA contexts are restored but initially remain locked. Restored
shims rejoin the persistent graph by stable participant ID rather than runtime
PID. While the CUDA process lock is held, the coordinator poisons the dedicated
Redis database, loads the RDB, verifies the expected generation and
manifest-bound state digest, and validates all graph metadata and allocation
bytes. The non-restored coordinator also verifies its `NODE_NAME` against the
source node, and each restored shim queries the real driver for the UUID
currently mapped to every source ordinal. M1 rejects node changes, GPU changes,
ordinal reordering, and access descriptors that are not exactly one read/write
DEVICE descriptor on the allocation GPU.

Processes launched through the shim but without capture-visible shared VMM
mappings, such as a launcher, are not durable VMM graph participants. After
native CUDA restore, the coordinator accepts such an endpoint only when
`QUERY_PLACEMENT` proves that it has zero detached managed placements. An
unexpected endpoint with any managed placement fails closed. This does not
support partial process restore: every durable ledger participant must still
restore on the source node with its exact captured GPU placement.

After all preflight checks pass, the coordinator unlocks the CUDA processes
immediately before the first owner replay request. This is required because
owner and importer replay invokes ordinary CUDA Driver APIs in the restored
processes. During this brief unlocked window, it recreates all owner
allocations before any importer, restores each owner at its exact VA, installs
temporary owner-location read/write access only while copying bytes, then
applies the recorded final access. Importers receive only their exact recorded
access. Every fresh broker FD is released and every shim passes one final
health check before the restore succeeds. Application-released generic handles
use temporary restore handles, which are released after remap. Application-live
logical handles are rebound to their new real handle and can be released
normally after restore.

The application never observes a real UMD
`CUmemGenericAllocationHandle` for managed allocations. The shim returns a
tagged process-local logical token from supported `cuMemCreate` and
`cuMemImportFromShareableHandle` calls and translates it for map, export,
release, and allocation-property queries. Unknown tagged tokens fail closed;
logical tokens are never passed to the UMD.

The application also never observes a raw CUDA POSIX export FD. On the first
successful owner export, the shim assigns that allocation a random nonzero
128-bit UUID. Every export of the same allocation returns a fresh fixed-size
sealed memfd containing the UUID, exact owner control-socket path, and owner
participant ID. The application can transport this opaque capability with
`SCM_RIGHTS`; it remains caller-owned and open after import.
Capabilities are bearer tokens: sealing protects their contents, while control
socket directory permissions remain the authorization boundary. An importer
accepts only the exact canonical
`CONTROL-DIRECTORY/cuda-vmm-PID.sock` shape in its own configured control
directory (`DYN_SNAPSHOT_CONTROL_DIR`, or `/snapshot-control` by default),
with a positive canonical decimal namespace PID.

An importer validates only this sealed capability. For another process's
owner, it releases its process-global shim mutex, requests a fresh raw CUDA FD
from the named owner over the bounded-time control socket, reacquires the
mutex, revalidates active state, and imports that FD. A same-process import
exports locally without calling its own control socket. The owner binds the
broker response to the operation, allocation UUID, and participant ID. The
owner and importer close their raw FD copies on every path and never persist
them. Malformed tokens, unavailable owners, broker mismatches, and raw CUDA
FDs passed directly to the import wrapper fail closed without fallback.

## Redis and RDB contract

Redis is a dedicated, explicitly configured endpoint for one Snapshot job. The
coordinator issues `FLUSHDB`; do not point it at a shared database.

| Environment variable | Contract |
|---|---|
| `DYN_SNAPSHOT_CUDA_VMM_REDIS_ADDR` | Required `host:port` reachable by both checkpoint and restore coordinators |
| `DYN_SNAPSHOT_CUDA_VMM_REDIS_PASSWORD` | Optional Redis `AUTH` password |
| `DYN_SNAPSHOT_CUDA_VMM_REDIS_RDB_PATH` | Required only at checkpoint: path to the endpoint's RDB, readable by the coordinator after synchronous `SAVE` |
| `DYN_SNAPSHOT_CUDA_VMM_REDIS_RESTORE_COMMAND` | Required at restore; executable invoked as `COMMAND CHECKPOINT_DIR/cuda-vmm.rdb` |

The restore command must atomically replace the dedicated endpoint's database
and return only when it accepts connections with the loaded generation. For a
remote Redis server, operators must provide the shared RDB path and restore
command through external orchestration. There is no service manager, remote
copy framework, or CUDA job-file fallback.

Redis keys are generation-scoped and contain one JSON graph/state record plus
one binary value per logical allocation. The typed JSON graph contains stable
participant and allocation-resource IDs, logical node placement, stable GPU
UUID placement, exact VAs, allocation-property bytes, requested handle type,
application-handle liveness, and access descriptors. It does not persist
PIDs, FD numbers, capture allocation UUIDs, real CUDA handles, CUDA contexts,
or opaque transport handles. Device ordinals can occur inside opaque
allocation-property and access-descriptor replay records, but they are not
durable identity. M1 admits only an access descriptor whose ordinal matches
the allocation's transient validated device ordinal; restore uses an ordinal
only after current UUID validation. A
deterministic SHA-256 digest over that graph
metadata and every allocation byte is stored in Redis and the checkpoint
manifest. The artifact requires the same architecture, CUDA ABI, and shim
build.

## Supported M1 contract

- CUDA Driver API POSIX-FD VMM only.
- One owner physical allocation, one complete offset-zero mapping per
  participant, all participants launched through the shim, and one
  checkpoint/restore cycle.
- One full-range access update per managed mapping, issued either for that
  mapping alone or once across an exact contiguous union of complete managed
  mappings. Every mapping in a combined update records the same descriptor
  array independently.
- Managed generic handles are stable shim tokens. They may remain live across
  checkpoint, or may follow the mapping-only lifecycle after application
  release.
- POSIX sharing handles are sealed shim capabilities. Repeated owner exports
  reuse one allocation UUID but return independent capability FDs.
- Legacy CUDA IPC passes through unchanged to CUDA checkpoint's native
  job-file support.
- `cuMulticastBindMem` and its available `_v2` resolver entry are poison-only:
  binding a managed allocation is rejected before the UMD because M1 does not
  capture or replay multicast state. Calls with unrelated real handles pass
  through unchanged.

## Detected and fail-closed

The implementation detects and rejects:

- missing or mixed launcher opt-in and process-set drift;
- non-POSIX shared-handle types, unknown typed resource kinds, unknown tagged
  logical handles, raw/unsealed/malformed sharing FDs, and untracked real
  handles used for managed map/export;
- any successful `cuMemRetainAllocationHandle` call, even though the real
  result is returned unchanged;
- partial, repeated, overlapping, or nonzero-offset tracked mappings and
  incomplete access metadata;
- partial, gapped, mixed managed/unmanaged, or repeated access-update ranges;
- host, other-device, multi-device, non-read/write, or otherwise non-FlashInfer
  access descriptors; M1 requires exactly one read/write DEVICE descriptor on
  the allocation GPU;
- missing owners/importers or local detached records that do not match the
  allocation UUID, logical object, role, exact VA, and size;
- zero allocation UUIDs, multiple owners for one UUID, owner endpoint or
  participant mismatches, unavailable owners, broker timeouts, responses
  without exactly one raw FD, and any attempt to fall back to importing the
  application capability directly;
- fork after CUDA initialization as observed by `pthread_atfork`;
- missing current `NODE_NAME`, target-node changes, GPU UUID changes at a
  source ordinal, and ordinal reordering;
- inconsistent manifest VMM fields, a missing/empty/non-regular
  `cuda-vmm.rdb`, or an RDB artifact without manifest opt-in;
- a checkpoint-side RDB source that is not a regular non-empty file after
  synchronous `SAVE`;
- a restore command that leaves the pre-restore Redis poison in place, or
  loaded state whose generation, ledger digest, graph metadata, or allocation
  bytes do not match the manifest;
- CUDA detach/replay, Redis, configuration, socket, and broker-FD cleanup
  failures;
- failure to unlock the CUDA processes at the validated preflight-to-replay
  boundary;
- a second prepare/restore cycle.

Preflight errors abort while CUDA remains locked, and an unlock error prevents
all owner/importer replay. Replay and final-health errors occur after CUDA
unlock. Partial detach/restore has no rollback: Snapshot fails closed by
withholding the `restore-complete` sentinel and tearing down the restore
container.

`UnlockProcessTree` unlocks one process at a time. If a later PID fails, earlier
PIDs can remain running, but no VMM replay is sent, `restore-complete` is
withheld, and the restore target is terminated. Application quiescence,
including the unmanaged/bypass CUDA risk, must therefore hold until teardown.

The wrappers are limited to `cuMemCreate`, `cuMemRelease`, `cuMemMap`,
`cuMemUnmap`, `cuMemSetAccess`, `cuMemExportToShareableHandle`,
`cuMemImportFromShareableHandle`, the poison-only
`cuMemRetainAllocationHandle`, `cuMemGetAllocationPropertiesFromHandle`,
poison-only `cuMulticastBindMem` variants, and CUDA driver/runtime resolver
entry points needed to return those wrappers. The shim narrowly interposes
`dlsym` for cuda-python's explicit `libcuda` lookup of
`cuGetProcAddress`, `cuGetProcAddress_v2`, or
`cuGetProcAddress_v2_ptsz`; all other `dlsym` names pass through unchanged.
Context and address-reservation APIs are not wrapped.

The shim does not interpose any `cuIpc*` API. Legacy CUDA IPC symbols and
calls pass through unchanged and remain covered by CUDA checkpoint's native
job-file behavior.

## Unobservable caller assumptions

The shim deliberately does not trace FD, socket, or loader syscalls. Callers
must enforce these assumptions:

| Assumption | Possible failure when violated |
|---|---|
| The process tree is quiesced before the first VMM `INSPECT` or any other checkpoint-prepare operation and remains quiesced through graph capture and all importer/owner detach operations. At its restored control point, the application remains quiescent through the unlocked owner/importer replay and final-health window. | This is an unsupported contract violation and can silently omit a live mapping from the captured graph, producing an incomplete or corrupt checkpoint. In particular, a remote import that finishes after its participant was inspected can commit a mapping absent from the captured graph. Other concurrent managed calls can cause rejection or CUDA errors; bypassed or otherwise unmanaged CUDA calls can race replay and cause silent mapping or data corruption. |
| Every application POSIX capability FD and alias is closed before prepare. FlashInfer's self-allgather result slot can be application-owned even when it is not imported, so the workload must close it too. | M1 cannot reliably verify this. A violation can preserve a stale bearer capability outside the captured graph and may be caught later by CRIU or yield undefined behavior. |
| No FD is queued in `SCM_RIGHTS`. | CRIU can restore a stale capability that does not refer to the newly brokered allocation. |
| No FD has been received but not imported. | The untracked capability can survive outside the logical graph and later import stale backing. |
| Applications treat sharing FDs as opaque transport capabilities and do not require raw CUDA-FD `fstat`, `ioctl`, or `poll` behavior. | The returned FD is a sealed memfd, not the NVIDIA character-device export. Unsupported inspection or raw-FD bypass cannot be associated safely and must not be used. |
| Every managed allocation has one complete offset-zero mapping and one full-range access update, either individually or through an exact contiguous union. | Partial, gapped, mixed, repeated, or unobserved shape cannot be reconstructed exactly. Observed wrapped violations fail admission; bypassed calls can replay incorrectly without detection. |
| Shared resources use only the M1 POSIX-FD allocation APIs. | IMEX/fabric handles, multicast, mempools, external memory, arrays, or unwrapped allocation-derived handles bypass the graph and can restore incomplete or stale sharing state. |
| Every participating process uses the launcher and resolves managed APIs through the preload/resolver path. | Static binaries, alternate loader namespaces, preload suppression, explicit-handle `dlvsym`, or direct explicit-handle lookup of a managed API can bypass tracking and produce an incomplete graph. cuda-python's explicit `dlsym` lookup of the CUDA resolver is supported. |
| A process does not fork before CUDA use and then continue without `exec`. | The child inherits the parent's participant ID, listener FD, and socket path but not the control thread. It cannot create an independent endpoint, and checkpoint eventually fails through participant/process-set drift. Fork followed by `exec` reinitializes the launcher-scoped shim. |
| Checkpoint and restore run on one logical node. | M1 has no cross-node barrier, capability transport, or placement; changing nodes makes local POSIX-FD brokerage and device ordinals invalid. |
| The restore uses the same architecture, CUDA ABI, shim build, and compatible GPU placement. | Opaque CUDA property/access bytes or device ordinals can have different meaning. Detectable preflight mismatches fail while locked; replay-time CUDA failures fail the restore and trigger target teardown. |
| Redis is dedicated to this job and the restore command atomically loads the supplied RDB. | Another writer or non-atomic load can replace/mix state; digest, generation, or poison checks reject detectable drift. |

Queued `SCM_RIGHTS` FDs and received-but-not-imported FDs are unsupported and
unobservable to M1.

The non-restored Snapshot coordinator reads its existing `NODE_NAME` for both
source metadata and current restore placement; restore fails closed when that
identity is unavailable. M1 remains single-node.

Deployment admission and workload discipline must enforce these assumptions.
