# Offload Module Developer Guide

This document provides implementation details for developers working on the offload pipeline. For high-level concepts and policy statements, see [README.md](README.md).

## Container-Based Architecture

### OffloadContainer

The container is the fundamental unit that flows through the pipeline:

```rust
struct OffloadContainer<T: BlockMetadata> {
    /// Source blocks to transfer
    blocks: Vec<SourceBlock<T>>,
    /// Precondition event - Some before PreconditionAwaiter, None after
    precondition: Option<EventHandle>,
    /// Cancellation token (cloned from TransferHandle)
    cancel_token: CancellationToken,
}

impl<T: BlockMetadata> OffloadContainer<T> {
    /// Check if this container has been cancelled
    fn is_cancelled(&self) -> bool {
        self.cancel_token.is_requested()
    }

    /// Upgrade all blocks from Weak → Strong
    /// Returns None if any block was evicted
    fn upgrade(self) -> Option<UpgradedContainer<T>> {
        // Implementation upgrades each SourceBlock
    }
}
```

### OffloadBatch

Batches group multiple containers for efficient transfer:

```rust
struct OffloadBatch<T: BlockMetadata> {
    containers: Vec<OffloadContainer<T>>,
}

impl<T: BlockMetadata> OffloadBatch<T> {
    /// Total blocks across all containers
    fn total_blocks(&self) -> usize {
        self.containers.iter().map(|c| c.blocks.len()).sum()
    }

    /// Remove cancelled containers, return count removed
    fn sweep_cancelled(&mut self) -> usize {
        let before = self.containers.len();
        self.containers.retain(|c| !c.is_cancelled());
        before - self.containers.len()
    }

    /// Check if batch is empty
    fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }
}
```

### Data Transformations Per Stage

| Stage | Input | Output | Transform |
|-------|-------|--------|-----------|
| Enqueue | `Vec<SourceBlock<T>>` | `OffloadContainer<T>` | Wrap with token + precondition |
| PolicyEvaluator | `OffloadContainer<T>` | `OffloadContainer<T>` | Filter `blocks` vec |
| PreconditionAwaiter | `OffloadContainer<T>` | `OffloadContainer<T>` | Await event, set `precondition = None` |
| Batcher | `OffloadContainer<T>` | `OffloadBatch<T>` | Group by total block count |
| TransferExecutor | `OffloadBatch<T>` | `Vec<ImmutableBlock<T>>` | Sweep → Upgrade → Flat map |

---

## Token-Based Cancellation

### Token Lifecycle

1. **Creation**: At enqueue, create a `CancellationToken` pair
2. **Distribution**: Handle gets the token, container gets a clone
3. **Propagation**: Token travels with container through pipeline
4. **Termination**: Token is consumed at upgrade (commitment point)

```rust
// At enqueue
let (cancel_token, cancel_updater) = CancellationToken::new();

// Give to handle
let handle = TransferHandle { cancel_token: cancel_token.clone(), ... };

// Give to container
let container = OffloadContainer {
    blocks,
    precondition: Some(event),
    cancel_token: cancel_token.clone(),
};
```

### CancellationToken API

```rust
impl CancellationToken {
    /// Request cancellation (called by handle)
    fn request(&self);

    /// Check if cancellation requested
    fn is_requested(&self) -> bool;

    /// Await cancellation request (for select!)
    async fn wait_requested(&self);

    /// Await confirmation that all blocks released
    fn wait_confirmed(&self) -> CancelConfirmation;
}
```

### PreconditionAwaiter Select Pattern

The awaiter uses `select!` to handle both event completion and cancellation:

```rust
async fn process(&self, mut container: OffloadContainer<T>) {
    // Fast path: event already satisfied
    if let Some(ref event) = container.precondition {
        if event.is_done() {
            container.precondition = None;
            self.output_queue.push(container);
            return;
        }
    }

    // Slow path: select on event OR cancellation
    if let Some(event) = container.precondition.take() {
        tokio::select! {
            _ = event.wait() => {
                // Event satisfied, propagate
                self.output_queue.push(container);
            }
            _ = container.cancel_token.wait_requested() => {
                // Cancelled while waiting - drop container
                tracing::debug!("Container cancelled during precondition wait");
                // container dropped here
            }
        }
    } else {
        // No precondition, pass through
        self.output_queue.push(container);
    }
}
```

### CancellableQueue Sweep Mechanics

The queue supports active cancellation via sweeping:

```rust
impl<T: HasCancellationToken> CancellableQueue<T> {
    /// Push item, reject if already cancelled
    fn push(&self, item: T) -> bool {
        if item.cancel_token().is_requested() {
            return false;
        }
        self.inner.push(item);
        true
    }

    /// Pop, skipping cancelled items
    fn pop_valid(&self) -> Option<T> {
        loop {
            match self.inner.pop() {
                Some(item) if item.cancel_token().is_requested() => continue,
                other => return other,
            }
        }
    }

    /// Remove all cancelled items
    fn sweep(&self) -> usize {
        let mut removed = 0;
        let mut kept = Vec::new();

        while let Some(item) = self.inner.pop() {
            if item.cancel_token().is_requested() {
                removed += 1;
            } else {
                kept.push(item);
            }
        }

        for item in kept {
            self.inner.push(item);
        }
        removed
    }
}
```

### Batch-Level Sweep

For `CancellableQueue<OffloadBatch<T>>`, sweeping removes cancelled containers within batches:

```rust
fn sweep(&self) -> usize {
    let mut removed_containers = 0;
    let mut kept_batches = Vec::new();

    while let Some(mut batch) = self.inner.pop() {
        // Remove cancelled containers from this batch
        removed_containers += batch.sweep_cancelled();

        // Keep batch if it still has containers
        if !batch.is_empty() {
            kept_batches.push(batch);
        }
    }

    for batch in kept_batches {
        self.inner.push(batch);
    }
    removed_containers
}
```

### Cancellation at Each Stage

| Stage | Mechanism | Behavior |
|-------|-----------|----------|
| PolicyEvaluator | Token check | Check `is_cancelled()` between block evaluations |
| PreconditionAwaiter | `select!` | Immediate drop if cancelled while waiting |
| Batcher Queue | CancellableQueue | Sweep removes cancelled containers |
| Executor Queue | CancellableQueue | Sweep removes cancelled containers from batches |
| TransferExecutor | Final sweep | `batch.sweep_cancelled()` before upgrade |

### Cancellation Boundary at Upgrade

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CANCELLABLE ZONE                                 │
│                                                                         │
│  Enqueue → PolicyEval → PrecondAwaiter → Batcher → ExecutorQueue        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                                                 ┌───────────────────┐
                                                 │  sweep_cancelled  │
                                                 │  (last check)     │
                                                 └───────────────────┘
                                                            │
                                                            ▼
═══════════════════════════════════════════════════════════════════════════
                              UPGRADE BOUNDARY
═══════════════════════════════════════════════════════════════════════════
                                                            │
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMMITTED ZONE                                   │
│                                                                         │
│  Upgrade → Flat Map → Transfer                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## TransferExecutor Design

### Sweep → Upgrade → Flat Map → Transfer

```rust
impl<T: BlockMetadata, D: TransferDestination> TransferExecutor<T, D> {
    async fn run(self) {
        while let Some(mut batch) = self.input_queue.pop() {
            // 1. SWEEP: Last cancellation check
            batch.sweep_cancelled();

            if batch.is_empty() {
                continue;
            }

            // 2. UPGRADE: Weak → Strong (commitment point)
            let upgraded: Vec<UpgradedContainer<T>> = batch
                .containers
                .into_iter()
                .filter_map(|c| c.upgrade())
                .collect();

            if upgraded.is_empty() {
                continue;
            }

            // 3. FLAT MAP: Consolidate into single vec
            let all_blocks: Vec<ImmutableBlock<T>> = upgraded
                .into_iter()
                .flat_map(|c| c.blocks)
                .collect();

            // 4. TRANSFER: Execute via destination
            self.destination.execute_transfer(all_blocks).await;
        }
    }
}
```

### Generic TransferDestination Trait

```rust
trait TransferDestination {
    type Output;

    async fn execute_transfer(
        &self,
        blocks: Vec<ImmutableBlock<T>>,
        src_layout: LogicalLayoutHandle,
    ) -> Result<Self::Output>;
}
```

### Block Destination (G2, G3)

For transfers to another `BlockManager`:

```rust
struct BlockDestination<Dst: BlockMetadata> {
    leader: Arc<InstanceLeader>,
    dst_manager: Arc<BlockManager<Dst>>,
    src_layout: LogicalLayoutHandle,
    dst_layout: LogicalLayoutHandle,
}

impl<Dst: BlockMetadata> TransferDestination for BlockDestination<Dst> {
    type Output = Vec<ImmutableBlock<Dst>>;

    async fn execute_transfer(&self, blocks: Vec<ImmutableBlock<_>>) -> Result<Self::Output> {
        // 1. Allocate destination blocks
        let dst_blocks = self.dst_manager.allocate_blocks(blocks.len())?;

        // 2. Execute transfer via leader
        let notification = self.leader.execute_local_transfer(
            self.src_layout,
            self.dst_layout,
            src_block_ids,
            dst_block_ids,
        )?;
        notification.await?;

        // 3. Register destination blocks
        let registered = dst_blocks.into_iter()
            .zip(sequence_hashes)
            .map(|(block, hash)| self.dst_manager.register_with_hash(block, hash))
            .collect();

        Ok(registered)
    }
}
```

### Object Destination (G4)

For transfers to object storage:

```rust
struct ObjectDestination {
    object_ops: Arc<dyn ObjectBlockOps>,
    src_layout: LogicalLayoutHandle,
    lock_manager: Option<Arc<dyn ObjectLockManager>>,
}

impl TransferDestination for ObjectDestination {
    type Output = Vec<SequenceHash>;

    async fn execute_transfer(&self, blocks: Vec<ImmutableBlock<_>>) -> Result<Self::Output> {
        // 1. Extract keys and block IDs
        let keys: Vec<SequenceHash> = blocks.iter().map(|b| b.sequence_hash()).collect();
        let block_ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();

        // 2. Execute object put
        let results = self.object_ops.put_blocks(keys.clone(), self.src_layout, block_ids).await;

        // 3. Handle lock management
        if let Some(lock_manager) = &self.lock_manager {
            for hash in &successful_hashes {
                lock_manager.create_meta(*hash).await?;
                lock_manager.release_lock(*hash).await?;
            }
        }

        Ok(successful_hashes)
    }
}
```

---

## Batcher Design

### Grouping Containers

The batcher accumulates containers and flushes when:
- Total blocks reach `max_batch_size`
- Flush interval expires and `min_batch_size` is met
- All blocks for a transfer have been processed (sentinel flush)

```rust
struct Batcher<T: BlockMetadata> {
    config: BatchConfig,
    input_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
    output_queue: Arc<CancellableQueue<OffloadBatch<T>>>,
    current_batch: OffloadBatch<T>,
}

impl<T: BlockMetadata> Batcher<T> {
    async fn run(mut self) {
        let mut flush_timer = tokio::time::interval(self.config.flush_interval);

        loop {
            tokio::select! {
                _ = flush_timer.tick() => {
                    self.try_flush().await;
                }
                Some(container) = self.input_queue.pop_valid() => {
                    self.current_batch.containers.push(container);

                    if self.current_batch.total_blocks() >= self.config.max_batch_size {
                        self.flush().await;
                    }
                }
            }
        }
    }

    async fn try_flush(&mut self) {
        if self.current_batch.total_blocks() >= self.config.min_batch_size {
            self.flush().await;
        }
    }

    async fn flush(&mut self) {
        if self.current_batch.is_empty() {
            return;
        }

        let batch = std::mem::replace(
            &mut self.current_batch,
            OffloadBatch { containers: Vec::new() },
        );

        self.output_queue.push(batch);
    }
}
```

### Preserving Per-Container Cancellability

Each container retains its own `cancel_token`. When the batch is in the executor queue:

1. **Sweep at queue level**: Removes cancelled containers from batches
2. **Sweep at executor**: Final check before upgrade
3. **Partial cancellation**: Some containers may be cancelled while others proceed

---

## Extension Rules

### Adding a New Policy

1. Implement the `OffloadPolicy` trait
2. Add to pipeline configuration
3. Policy must be fast or async-compatible

```rust
trait OffloadPolicy<T: BlockMetadata>: Send + Sync {
    fn name(&self) -> &str;
    fn evaluate(&self, ctx: &EvalContext<T>) -> impl Future<Output = Result<bool>>;
}
```

### Adding a New Destination Type

1. Implement `TransferDestination` trait
2. Create a new pipeline variant or use generic executor
3. Handle destination-specific registration/cleanup

### Maintaining Cancellation Invariants

When modifying the pipeline:

1. **Never skip the upgrade boundary** - It's the commitment point
2. **Always sweep before upgrade** - Last chance to cancel
3. **Token must travel with container** - Don't strip it prematurely
4. **Batches preserve container identity** - Until flat map

---

## Testing Guidance

### Unit Tests

- Test each stage in isolation
- Mock `CancellationToken` for cancel scenarios
- Verify sweep removes correct items

### Integration Tests

- Test full pipeline with cancel at each stage
- Verify no orphaned blocks after cancellation
- Test partial batch cancellation

### Performance Tests

- Measure overhead of cancellation checks
- Benchmark sweep operation at scale
- Profile upgrade → flat map → transfer path





