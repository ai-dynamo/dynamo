<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Snapshot CUDA shared VMM interposer

This launcher-scoped `LD_PRELOAD` interposer checkpoints complete CUDA Driver
API VMM allocations shared through either POSIX file descriptors or single-node
`CU_MEM_HANDLE_TYPE_FABRIC`/IMEX handles. It also checkpoints the exact
same-node, handle-based CUDA multicast lifecycle used by FlashInfer. It is
transparent to inference frameworks: applications do not call checkpoint
hooks. The interposer composes with `cuda-checkpoint --launch-job`; Redis
replaces the CUDA job-file role only for shim-managed VMM state. CUDA
checkpoint continues to own unrelated CUDA state, including legacy CUDA IPC.

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
2. The coordinator groups this captured generation by allocation or multicast
   UUID, assigns logical resource IDs, resolves every multicast
   backing-allocation reference, and validates exactly one owner per resource.
3. Each allocation owner's bytes are copied once into Redis. Multicast
   resources are byte-less and have no content key.
4. Multicast importers and owners are unbound, unmapped, and released first.
   Allocation importers and owners are then detached while CUDA
   virtual-address reservations remain in place.
5. The coordinator records metadata and content in Redis, runs `SAVE`, and
   copies the RDB into the checkpoint as `cuda-vmm.rdb`.
6. Normal `cuda-checkpoint` and CRIU capture all remaining supported state.

At restore, CUDA contexts are restored but initially remain locked. Restored
shims rejoin the persistent graph by stable participant ID rather than runtime
PID. While the CUDA process lock is held, the coordinator poisons the dedicated
Redis database, loads the RDB, verifies the expected generation and
manifest-bound state digest, and validates all graph metadata and allocation
bytes. The non-restored coordinator also verifies its `NODE_NAME` against the
source node. It builds a complete identity-first source-to-target placement from
the manifest source UUIDs and target UUIDs in current runtime ordinal order,
then sends each restored shim only its expected source entries. Each shim
validates all detached allocation metadata before atomically updating the
target ordinal in allocation properties and access descriptors. The saved
source UUID remains the allocation identity; restored shims do not enumerate
CUDA devices during placement setup.

Processes launched through the shim but without capture-visible shared VMM
mappings, such as a launcher, are not durable VMM graph participants. After
native CUDA restore, the coordinator accepts such an endpoint only when
an empty `SET_PLACEMENT` proves that it has zero detached managed placements. An
unexpected endpoint with any managed placement fails closed. This does not
support partial process restore: every durable ledger participant must still
restore on the source node under the authoritative placement plan.

After all preflight checks pass, the coordinator unlocks the CUDA processes
immediately before the first owner replay request. This is required because
replay invokes ordinary CUDA Driver APIs in the restored processes. During
this brief unlocked window:

1. It recreates all allocation owners before any allocation importer, restores
   each owner at its exact VA, copies owner bytes under temporary access, and
   applies every participant's recorded final access.
2. It creates multicast owners, imports multicast importers, and adds each
   participant's recorded local device. No multicast bind or map occurs until
   every participant has completed this membership phase.
3. It replays each exact full-range bind, mapping, and access update.
4. It releases temporary backing-allocation handles only after every multicast
   resource has bound. This preserves allocations shared by FlashInfer's full
   and tensor-parallel multicast groups.
5. It verifies every shim's final health and only then marks Redis restored.

After unlock, any multicast owner/importer, bind, broker, finalize, or health
failure triggers a best-effort abort request to every restored process. The
abort leaves each shim failed and removes recreated/imported multicast groups,
their mappings and binds, and temporary retained backing handles. This
distributed cleanup is specific to the multicast restore transaction; it does
not roll back unrelated ordinary allocation replay.

Every fresh broker value is closed or wiped. Application-released allocation
handles referenced by multicast binds are retained only through the explicit
finalize phase. Application-live logical handles are rebound to their new real
handle and can be released normally after restore.

The application never observes a real UMD
`CUmemGenericAllocationHandle` for managed allocations or multicast objects.
The shim returns a tagged process-local logical token from supported
`cuMemCreate`, `cuMulticastCreate`, and `cuMemImportFromShareableHandle` calls
and translates it for supported operations. Unknown tagged tokens fail closed;
logical tokens are never passed to the UMD.

The application never observes a raw CUDA POSIX export FD or raw CUDA FABRIC
handle. On the first successful owner export, the shim assigns that allocation
a random nonzero 128-bit UUID.

- A POSIX export returns a fresh 256-byte sealed memfd capability containing
  the UUID, object-kind discriminator, exact owner control-socket path, and
  owner participant ID. The application can transport this opaque capability
  with `SCM_RIGHTS`; it remains caller-owned and open after import.
- A FABRIC export calls the real driver on every attempt, then returns an exact
  64-byte logical token containing a magic/version/type discriminator, UUID,
  32 lowercase hexadecimal participant bytes, positive namespace PID, and
  object kind. The driver-produced 64 bytes are immediately discarded.
  Repeated exports reuse the logical UUID and route identity.

The POSIX capability and FABRIC token are bearer values. Control-socket
directory permissions remain the authorization boundary. The FABRIC PID is
only a transient local routing hint; durable allocation identity is the
participant ID plus allocation UUID. The importer derives the canonical
`CONTROL-DIRECTORY/cuda-vmm-PID.sock` path from its own configured control
directory (`DYN_SNAPSHOT_CONTROL_DIR`, or `/snapshot-control` by default).

For another process's owner, the importer releases its process-global shim
mutex during bounded control-socket I/O, reacquires the mutex, revalidates
active state and identity, then passes the freshly brokered raw handle directly
to the real driver. A same-process import exports locally without calling its
own socket. The private protocol is typed:

| Handle type | Broker result |
|---|---|
| POSIX FD (`1`) | Exact allocation or multicast object kind, one `SCM_RIGHTS` FD, and no payload |
| FABRIC (`8`) | Exact allocation or multicast object kind, no FD, and exactly 64 payload bytes |

The owner binds responses to operation, exact handle type, allocation UUID, and
participant ID. Raw FD copies are closed and raw FABRIC bytes are wiped and
discarded on every path. Malformed logical tokens, raw CUDA handles passed to
the application import wrapper, unavailable owners, and response-shape or
identity mismatches fail closed without fallback or live Redis lookup.

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
participant and resource IDs, resource kind, logical node placement, stable GPU
UUID placement, exact VAs, allocation-property bytes, requested handle type,
application-handle liveness, access descriptors, and multicast object,
membership, and backing-allocation binding metadata. Multicast groups have no
`resource:<id>` content value or content digest object; their serialized
metadata remains covered by the ledger digest. The ledger does not persist
PIDs, FD numbers, capture UUIDs, real CUDA handles, CUDA contexts, socket
paths, or POSIX/FABRIC transport bytes. Device ordinals can occur inside opaque
allocation-property and access-descriptor replay records, but they are not
durable identity. The shim admits only an access descriptor whose ordinal matches
the allocation's transient validated device ordinal; restore uses an ordinal
only after validating the controller-supplied source/target UUID plan. A
deterministic SHA-256 digest over that graph
metadata and every allocation byte is stored in Redis and the checkpoint
manifest. Redis stores the resource graph and allocation contents, never POSIX
FDs, logical FABRIC tokens, raw FABRIC handles, PIDs, or socket paths. Ledger
version 3 and private control protocol version 5 are required; cross-version
artifact restore is unsupported. The artifact requires the same architecture,
CUDA ABI, and shim build.

## Supported contract

- CUDA Driver API VMM with `requestedHandleTypes` exactly POSIX FD (`1`) or
  exactly FABRIC (`8`). Zero, combined bitmasks such as `9`, and every other
  type are rejected.
- FABRIC is single-node IMEX only. Exporter and importer must use
  fabric-ready hardware and have access to the same IMEX channel.
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
- Every owner and importer for one logical resource uses the same exact handle
  type. POSIX sharing handles are sealed shim capabilities; FABRIC sharing
  handles are exact 64-byte shim tokens. Repeated owner exports reuse one
  allocation UUID.
- Individual allocation contents are limited to Redis's 512 MiB bulk limit;
  restore frames add only one exact record, property, and access descriptor.
- Legacy CUDA IPC passes through unchanged to CUDA checkpoint's native
  job-file support.
- Handle-based multicast supports one owner and one or more importers on the
  same logical node. Each participant adds exactly one stable member GPU,
  binds one tracked allocation at offsets zero for the complete object size
  with flags zero, maps the complete object once, and applies one full-range
  same-GPU read/write access update. Both legacy `cuMulticastBindMem` and its
  available `_v2` form are supported; `_v2` must name the recorded member.
- Multiple multicast resources can bind the same backing allocation, as in
  FlashInfer's full and tensor-parallel groups. `cuMulticastGetGranularity`
  and entirely unmanaged multicast objects pass through unchanged.

On restore, owner reconstruction remains one path: create, exact-VA map,
temporary write access, content copy, final access, and logical-handle rebind.
The owner then returns either one transient POSIX FD or one transient 64-byte
FABRIC value. A FABRIC importer receives those bytes only after its durable
record/access metadata, calls the real import API, wipes the bytes, maps,
applies access, and rebinds. No raw transport value enters allocation state,
Redis, JSON, the digest, or a file.

## Detected and fail-closed

The implementation detects and rejects:

- missing or mixed launcher opt-in and process-set drift;
- zero, mixed, combined, or unsupported shared-handle types, unknown typed
  resource kinds, unknown tagged
  logical handles, raw/unsealed/malformed sharing FDs, and untracked real
  handles used for managed map/export;
- any successful `cuMemRetainAllocationHandle` call, even though the real
  result is returned unchanged;
- partial, repeated, overlapping, or nonzero-offset tracked mappings and
  incomplete access metadata;
- missing, duplicate, or post-bind multicast membership; mixed
  managed/unmanaged multicast operations; partial, repeated, nonzero-offset,
  wrong-device, wrong-backing, or wrong-size multicast binds; incomplete
  multicast graphs; and managed `cuMulticastBindAddr` variants;
- partial, gapped, mixed managed/unmanaged, or repeated access-update ranges;
- host, other-device, multi-device, non-read/write, or otherwise non-FlashInfer
  access descriptors; the shim requires exactly one read/write DEVICE descriptor on
  the allocation GPU;
- missing owners/importers or local detached records that do not match the
  allocation UUID, logical object, role, exact VA, and size;
- zero allocation UUIDs, invalid FABRIC token magic/version/type/participant/PID
  or object-kind discrimination, multiple owners for one UUID, owner endpoint
  or participant mismatches, unavailable owners, broker timeouts, FABRIC
  responses carrying an FD or not exactly 64 bytes, POSIX responses carrying a
  payload or not exactly one FD, and any attempt to fall back to importing an
  application token directly;
- real CUDA FABRIC export/import failure, including
  `CUDA_ERROR_NOT_PERMITTED` when the CUDA FABRIC-support attribute, fabric
  state, or accessible IMEX channel is absent or mismatched, or when exporter
  and importer do not share the same channel;
- fork after CUDA initialization as observed by `pthread_atfork`;
- missing current `NODE_NAME`, target-node changes, or an invalid, incomplete,
  duplicate, or participant-inconsistent authoritative GPU placement plan;
- inconsistent manifest VMM fields, a missing/empty/non-regular
  `cuda-vmm.rdb`, or an RDB artifact without manifest opt-in;
- a checkpoint-side RDB source that is not a regular non-empty file after
  synchronous `SAVE`;
- a restore command that leaves the pre-restore Redis poison in place, or
  loaded state whose generation, ledger digest, graph metadata, or allocation
  bytes do not match the manifest;
- CUDA detach/replay, Redis, configuration, socket, broker-FD cleanup, and
  FABRIC-byte cleanup
  failures;
- failure to unlock the CUDA processes at the validated preflight-to-replay
  boundary;
- a second prepare/restore cycle.

Preflight errors abort while CUDA remains locked, and an unlock error prevents
all owner/importer replay. Replay and final-health errors occur after CUDA
unlock. On a partial multicast restore, the coordinator best-effort sends the
idempotent abort operation to every restored process; each shim unbinds, unmaps,
and releases newly created/imported groups and temporary allocation handles.
Cleanup failures are reported with the original replay failure. The shims
remain failed, Snapshot withholds the `restore-complete` sentinel, and the
restore container is torn down. The abort does not undo unrelated ordinary
allocation replay.

`UnlockProcessTree` unlocks one process at a time. If a later PID fails, earlier
PIDs can remain running, but no VMM replay is sent, `restore-complete` is
withheld, and the restore target is terminated. Application quiescence,
including the unmanaged/bypass CUDA risk, must therefore hold until teardown.

The wrappers are limited to `cuMemCreate`, `cuMemRelease`, `cuMemMap`,
`cuMemUnmap`, `cuMemSetAccess`, `cuMemExportToShareableHandle`,
`cuMemImportFromShareableHandle`, the poison-only
`cuMemRetainAllocationHandle`, `cuMemGetAllocationPropertiesFromHandle`,
`cuMulticastCreate`, `cuMulticastAddDevice`, `cuMulticastBindMem`,
`cuMulticastUnbind`, poison-only `cuMulticastBindAddr` variants, and CUDA
driver/runtime resolver
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
| Every application POSIX capability FD and alias is closed before prepare. FlashInfer's self-allgather result slot can be application-owned even when it is not imported, so the workload must close it too. | The shim cannot reliably verify this. A violation can preserve a stale bearer capability outside the captured graph and may be caught later by CRIU or yield undefined behavior. |
| No FABRIC logical token remains queued, retained, or unconsumed at the checkpoint boundary. | The 64-byte value has no observable lifecycle. A stale token can survive outside the captured graph and route to detached or replaced owner state. |
| No FD is queued in `SCM_RIGHTS`. | CRIU can restore a stale capability that does not refer to the newly brokered allocation. |
| No FD has been received but not imported. | The untracked capability can survive outside the logical graph and later import stale backing. |
| Applications treat sharing FDs as opaque transport capabilities and do not require raw CUDA-FD `fstat`, `ioctl`, or `poll` behavior. | The returned FD is a sealed memfd, not the NVIDIA character-device export. Unsupported inspection or raw-FD bypass cannot be associated safely and must not be used. |
| Every managed allocation has one complete offset-zero mapping and one full-range access update, either individually or through an exact contiguous union. | Partial, gapped, mixed, repeated, or unobserved shape cannot be reconstructed exactly. Observed wrapped violations fail admission; bypassed calls can replay incorrectly without detection. |
| Shared resources use only exact POSIX-FD or FABRIC handle types through the wrapped Driver APIs. Multicast follows the one-member, one-full-bind, one-map contract above. | Address-bound multicast, multicast shapes outside that contract, mempools, external memory, arrays, native FlashInfer checkpoint hooks, retained/queued transport objects, or unwrapped allocation-derived handles bypass the graph and can restore incomplete or stale sharing state. |
| Every participating process uses the launcher and resolves managed APIs through the preload/resolver path. | Static binaries, alternate loader namespaces, preload suppression, explicit-handle `dlvsym`, or direct explicit-handle lookup of a managed API can bypass tracking and produce an incomplete graph. cuda-python's explicit `dlsym` lookup of the CUDA resolver is supported. |
| A process does not fork before CUDA use and then continue without `exec`. | The child inherits the parent's participant ID, listener FD, and socket path but not the control thread. It cannot create an independent endpoint, and checkpoint eventually fails through participant/process-set drift. Fork followed by `exec` reinitializes the launcher-scoped shim. |
| Checkpoint and restore run on one logical node, with the same accessible IMEX channel and ready fabric state for FABRIC resources. | Cross-node routing is unsupported. A node, GPU, IMEX channel, or fabric-state change invalidates local brokerage and can cause `CUDA_ERROR_NOT_PERMITTED` or restore failure. |
| The restore uses the same architecture, CUDA ABI, and shim build. | Identity and same-node remapped GPU placements are reconciled while locked from the controller plan. Other opaque CUDA property/access changes fail preflight; replay-time CUDA failures fail the restore and trigger target teardown. |
| Redis is dedicated to this job and the restore command atomically loads the supplied RDB. | Another writer or non-atomic load can replace/mix state; digest, generation, or poison checks reject detectable drift. |

The non-restored Snapshot coordinator reads its existing `NODE_NAME` for both
source metadata and current restore placement; restore fails closed when that
identity is unavailable. The implementation remains single-node.

## Non-goals

The interposer does not support cross-node routing, multinode IMEX,
`cuMulticastBindAddr` replay, partial or repeated multicast membership/binds,
CUDA mempools, external memory, arrays, unobservable retained, dangling,
queued, or received-but-not-imported transport capabilities, repeated
checkpoint/restore cycles, cross-version artifacts, or native application
checkpoint hooks, including native FlashInfer hooks. It does not remove
CUDA's native checkpoint responsibility for unrelated CUDA state or legacy
CUDA IPC.

Deployment admission and workload discipline must enforce these assumptions.
