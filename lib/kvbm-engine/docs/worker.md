# Worker Module

The worker module defines execution primitives for data transfers between
storage tiers. Workers own the physical resources (transfer managers, layout
handles) needed to move blocks via RDMA, local copy, or object storage.

## Trait Hierarchy

```text
WorkerTransfers          Worker
  (execution)       (metadata + handles)
       │                    │
       └────────┬───────────┘
                │
         ObjectBlockOps
          (G4 storage)
```

- **`WorkerTransfers`** – core execution trait. Provides `execute_local_transfer`,
  `execute_remote_onboard`, `execute_remote_offload`, `connect_remote`, and
  `execute_remote_onboard_for_instance`.
- **`Worker`** – extends `WorkerTransfers + ObjectBlockOps`. Adds layout handle
  accessors (`g1_handle`, `g2_handle`, `g3_handle`) and metadata import/export.

## PhysicalWorker (aka DirectWorker)

`PhysicalWorker` is the fundamental single-worker implementation. It directly
owns a `TransferManager` and layout handles for executing data movement.

### Builder

```rust,ignore
let worker = PhysicalWorker::builder()
    .manager(transfer_manager)   // required
    .g1_handle(g1)               // optional – GPU tier
    .g2_handle(g2)               // optional – host tier
    .g3_handle(g3)               // optional – disk tier
    .rank(0)                     // optional – for SPMD key prefixing
    .object_client(s3_client)    // optional – for G4 operations
    .build()?;
```

| Field | Required | Purpose |
|-------|----------|---------|
| `manager` | yes | `TransferManager` for executing transfers |
| `g1_handle` | no | GPU/HBM layout handle |
| `g2_handle` | no | Host/pinned-DRAM layout handle |
| `g3_handle` | no | Disk/NVMe layout handle |
| `rank` | no | Worker rank for SPMD key prefixing |
| `object_client` | no | G4 object storage client |

`DirectWorker` is a compatibility alias for `PhysicalWorker`.

### Execution State vs Coordination State

PhysicalWorker maintains **execution state** – the handles and manager needed
to actually perform transfers. This is distinct from **coordination state**
which the leader tracks in `CoordinatedWorker`. When a leader wraps a
PhysicalWorker in a CoordinatedWorker, handles exist in both places
intentionally: PhysicalWorker needs them to call TransferManager, while
CoordinatedWorker provides a uniform API for both local and remote workers.

## CoordinatedWorker

`CoordinatedWorker` is the leader's view of a worker. It wraps any `Worker`
implementation and adds coordination state:

- Local layout handles (populated via `apply_layout_response`)
- Remote handle mappings for cross-leader RDMA transfers
- Worker rank and host instance tracking

This wrapper lets the leader use the same API regardless of whether the
underlying worker is local (`PhysicalWorker`) or remote (`VeloWorkerClient`).

## VeloWorkerClient / VeloWorkerService

The Velo (RPC) layer enables remote worker execution:

- **`VeloWorkerService`** – wraps a `PhysicalWorker` and exposes RPC handlers
  for `execute_local_transfer`, `export_metadata`, `import_metadata`, etc.
- **`VeloWorkerClient`** – implements `WorkerTransfers` by sending RPC
  requests to a remote `VeloWorkerService`.

Together they allow the leader to drive workers on remote nodes as if they
were local.
