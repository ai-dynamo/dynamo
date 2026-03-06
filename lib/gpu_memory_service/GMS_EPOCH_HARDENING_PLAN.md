# GMS Epoch Hardening Plan

## Goal
Eliminate control-plane/data-plane races by making committed allocations immutable and ensuring every RW session writes into a fresh epoch.

## Invariants
1. `committed_epoch_id` is RO-visible only while no RW epoch exists.
2. `active_rw_epoch_id` exists only while a RW lock is held.
3. RW allocate/free/metadata mutations apply only to `active_rw_epoch_id`.
4. RO export/get/list/metadata reads apply only to `committed_epoch_id`.
5. Every allocation is tagged with `epoch_id`.
6. Commit is a publish barrier:
   1. client `synchronize()`
   2. client downgrades mapped pages to RO
   3. server atomically promotes `active_rw_epoch_id -> committed_epoch_id`
7. RW connect invalidates committed visibility immediately (`committed=false`).
8. RW abort/disconnect drops `active_rw_epoch_id` and leaves the server in `EMPTY`.
   Treat each transition to `EMPTY` as the start of a new uncommitted epoch boundary.
9. Last `RO_DISCONNECT` returns lock FSM to `COMMITTED` only for an actually committed epoch.
10. `allocation.epoch_id` is write-once at allocate-time and never mutated.

## Concrete Changes
### 1. Server epoch model
- Add server epoch state in request handler:
  - `committed_epoch_id: Optional[str]`
  - `active_rw_epoch_id: Optional[str]`
  - epoch-scoped metadata map.
- Add handler hooks:
  - `on_rw_connect()`: create new active epoch.
  - `on_commit()`: publish active epoch as committed.
  - `on_rw_abort()`: clear active epoch only.

### 2. Allocation epoch tracking
- Extend `AllocationInfo` with `epoch_id`.
- Add epoch-aware memory manager operations:
  - `allocate(size, tag, epoch_id)`
  - `get_allocation(allocation_id, epoch_id=None)`
  - `list_allocations(tag=None, epoch_id=None)`
  - `clear_epoch(epoch_id)`

### 3. RPC routing by lock mode
- On handshake grant RW, call `on_rw_connect()`.
- Route RO-visible operations to committed epoch.
- Route RW-visible operations to active epoch.
- Keep state machine lock semantics (EMPTY/RW/COMMITTED/RO) unchanged for this slice.

### 4. Client commit hardening
- Move synchronization into `GMSClientMemoryManager.commit()`.
- Before RPC commit, downgrade all mapped allocations to RO access via `cuMemSetAccess`.
- Keep commit close/reconnect flow unchanged.

### 5. Protocol visibility
- Include `epoch_id` in allocation responses for observability.

## Follow-up (next slice)
1. OOM wait/retry for new RW allocations.
2. Epoch GC policy and memory-pressure cleanup.
3. Fault-injection tests for abort, disconnect, and publish races.
