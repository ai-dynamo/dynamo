# Offload Module

The offload module manages the asynchronous transfer of KV cache blocks between storage tiers. It provides a pipeline-based architecture for evaluating, batching, and executing block transfers with full cancellation support.

## Overview

Offloading moves blocks from a source tier (e.g., GPU memory) to a destination tier (e.g., host memory, remote storage, or object storage). The pipeline ensures:

- **Policy-based filtering**: Only blocks meeting criteria are transferred
- **Batched execution**: Blocks are grouped for efficient transfer
- **Cancellation support**: Transfers can be cancelled at any point before commitment
- **Precondition synchronization**: Transfers wait for forward pass completion

## Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ PolicyEvaluator │────►│ PreconditionAwaiter │────►│       Batcher       │────►│ TransferExecutor │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘     └──────────────────┘
                                                             ▲                          ▲
                                                             │                          │
                                                    CancellableQueue          CancellableQueue
                                                             │                          │
                                                             └──────── CancelSweeper ───┘
```

### Stages

| Stage | Purpose |
|-------|---------|
| **PolicyEvaluator** | Filters blocks based on configured policies (frequency, presence, etc.) |
| **PreconditionAwaiter** | Waits for forward pass completion before proceeding |
| **Batcher** | Groups containers into batches based on total block count |
| **TransferExecutor** | Upgrades blocks and executes the actual transfer |

## Container Data Model

The fundamental unit flowing through the pipeline is an **OffloadContainer**:

```rust
struct OffloadContainer<T: BlockMetadata> {
    /// The blocks to offload
    blocks: Vec<SourceBlock<T>>,
    /// Precondition event (forward pass completion)
    precondition: Option<EventHandle>,
    /// Cancellation token
    cancel_token: CancellationToken,
}
```

Containers are grouped into batches for efficient transfer:

```rust
struct OffloadBatch<T: BlockMetadata> {
    /// Multiple containers, each independently cancellable
    containers: Vec<OffloadContainer<T>>,
}
```

## Policy Statements

The following policies govern the behavior of this module. See [AGENTS.md](AGENTS.md) for governance rules.

### P1: Container is the Unit of Cancellation

Individual blocks within a container are not independently cancellable. When a container is cancelled, all its blocks are cancelled together.

### P2: Token Travels with Container

Each container carries its own `CancellationToken`, cloned from the `TransferHandle` at enqueue time. The token travels with the container through all pipeline stages until upgrade.

### P3: Upgrade is the Commitment Boundary

The upgrade step (Weak → Strong) is the point of no return:

- **Before upgrade**: Containers can be cancelled via sweep or token check
- **After upgrade**: We own the blocks; cancellation no longer applies

### P4: Sweep Before Upgrade

The last cancellation check occurs immediately before upgrade. The `TransferExecutor` calls `batch.sweep_cancelled()` to remove cancelled containers before committing.

### P5: Flat Map After Upgrade

After upgrade, all blocks from all containers are consolidated into a single `Vec<ImmutableBlock<T>>` for efficient batch transfer. Per-container identity is lost at this point.

### P6: PreconditionAwaiter Uses Select

The precondition awaiter can be cancelled via `select!` on both the precondition event and the cancellation token. If cancelled while waiting, the container is dropped immediately.

## Configuration

Pipeline behavior is controlled via `PipelineConfig`:

| Option | Default | Description |
|--------|---------|-------------|
| `batch_config.max_batch_size` | 64 | Maximum blocks per batch |
| `batch_config.min_batch_size` | 8 | Minimum blocks before flush |
| `batch_config.flush_interval` | 10ms | Time before flushing partial batch |
| `policy_timeout` | 100ms | Timeout for policy evaluation |
| `sweep_interval` | 10ms | Interval for cancel sweeper |
| `max_concurrent_transfers` | 1 | Concurrent transfer batches |

## Usage

### Enqueueing Blocks

```rust
let handle = pipeline.enqueue(source_blocks, precondition_event);

// Track progress
println!("Status: {:?}", handle.status());

// Wait for completion
let result = handle.wait().await?;
```

### Cancelling a Transfer

```rust
// Request cancellation and wait for confirmation
handle.cancel().await;
// All blocks are now released
```

## Related Documentation

- [DEVELOPER.md](DEVELOPER.md) - Implementation details and extension rules
- [AGENTS.md](AGENTS.md) - AI agent governance rules





