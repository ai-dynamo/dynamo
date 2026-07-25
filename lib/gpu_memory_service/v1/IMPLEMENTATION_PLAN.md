# GMS Snapshot V1 Stacked Implementation Plan

## Stack

This change is intentionally split into two reviewable layers:

1. **Core and Snapshot V1** adds the final mechanisms shared by both profiles
   and the Snapshot-specific policy that consumes them. The existing V0
   implementation and behavior remain unchanged in this layer.
2. **V0 core migration** changes V0 to consume those mechanisms and deletes
   only the implementations they replace.

The temporary duplication between `core/` and V0 is deliberate. It lets the
lower layer introduce and validate the common interface without mixing in a
large V0 refactor.

## Lower layer: core and Snapshot V1

The shared core owns:

- rank-local physical allocation handles and transient export FDs;
- socket sessions, writer priority, `RW`, `RO`, and `RW_OR_RO` admission;
- same-socket RW-to-RO commit and disconnect cleanup;
- neutral reserve/import/map/access/unmap/release operations;
- generic Torch allocator and MemPool construction; and
- alias-preserving tensor isolation.

Snapshot V1 owns:

- fresh RW construction, commit, sleep, restored wake, and retirement policy;
- exact checkpointed allocation ID and virtual-address reuse;
- live TensorImpl discovery, copy-out, and accounting;
- broad vLLM weight capture and native vLLM KV ordering; and
- the V1 sidecar entry point and packaging.

Snapshot V1 exposes no lock selection. Fresh construction opens RW, commit
holds the same socket as RO, sleep unmaps and closes RO, and restored wake
opens RO and imports the exact IDs at the checkpointed virtual addresses.

## Upper layer: V0 migration

The upper layer makes V0 consume shared:

- server allocation and session/lease ownership;
- local VMM mapping operations;
- Torch allocator construction; and
- tensor isolation.

V0 keeps its typed wire protocol, metadata and layout reconstruction,
meta-model and structural matching, scratch/mutable KV behavior, framework
accounting, compatibility behavior, and existing public interfaces.

## Failure model

Both layers are fail-stop. There is no rollback, retry, reconnect, replay,
fallback, compensating protocol, mapping validation, or two-phase commit.
Cleanup is limited to unpublished local resources, such as releasing an
imported handle when the immediate map fails.

## Validation

Each branch is validated independently:

- lower layer: focused core/V1 tests and a standalone wheel build;
- upper layer: focused V0 session, runtime, mapping, allocator, and tensor
  regressions.

The previously combined tree remains a baseline only; its results do not
validate either reconstructed branch head.
