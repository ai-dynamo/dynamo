# Worker Group Module

The worker group module provides abstractions for driving multiple workers
in parallel from a single leader.

## ParallelWorkers Trait

`ParallelWorkers` extends `WorkerTransfers + ObjectBlockOps` for cohorts of
workers. It adds:

- `export_metadata()` → `Vec<SerializedLayoutResponse>` (one per rank)
- `import_metadata(Vec<SerializedLayout>)` → `Vec<ImportMetadataResponse>`
- `worker_count()` → number of workers
- `workers()` → slice of underlying `Arc<dyn Worker>`

## SpmdParallelWorkers

`SpmdParallelWorkers` implements the SPMD (Single Program, Multiple Data)
execution model: the same operation is broadcast to every worker in parallel,
and results are aggregated.

### Fan-out Execution

Every `WorkerTransfers` method (local transfer, remote onboard, remote
offload) iterates over all workers and calls the same operation on each.
Workers execute in parallel – each resolves the shared logical layout handle
to its own physical layout.

### Rank-aware Routing

For `connect_remote`, each worker receives its rank-specific metadata slice.
Remote handle mappings are stored as `(InstanceId, worker_idx,
LogicalLayoutHandle) → LayoutHandle`, so `execute_remote_onboard_for_instance`
can look up the correct remote handle for each worker by rank.

### Event Aggregation

Transfer completion notifications from individual workers are aggregated into
a single `TransferCompleteNotification` via the event system. The aggregated
notification fires only when all workers have completed.

### ObjectBlockOps Aggregation

- `has_blocks`: queries all workers, returns results from worker 0 (all should
  agree in SPMD semantics).
- `put_blocks` / `get_blocks`: executes on all workers in parallel. A key
  succeeds only if **all** workers succeed for that key.

### Construction

```rust,ignore
let parallel = SpmdParallelWorkers::new(
    workers,        // Vec<Arc<dyn Worker>>, one per rank
    event_manager,  // Arc<EventManager> for aggregation
    runtime_handle, // tokio::runtime::Handle for spawning
);
```
