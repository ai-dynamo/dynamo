# Block Manager

The `KvBlockManager` is the central orchestrator of the KV cache management system. It provides a unified interface for managing blocks across different storage tiers and coordinates all block-related operations.

## Overview

The `KvBlockManager` is a generic structure that can work with different metadata types and storage configurations. It manages the lifecycle of KV cache blocks from allocation to cleanup, providing efficient access patterns for LLM inference.

## Core Structure

```rust
pub struct KvBlockManager<Metadata: BlockMetadata> {
    state: Arc<state::KvBlockManagerState<locality::Local, Metadata>>,
    _cancellation_token: Arc<CancelOnLastDrop>,
    block_size: usize,
}
```

### Type Parameters

- **`Metadata`**: The metadata type associated with blocks. Must implement `BlockMetadata` trait.

### Fields

- **`state`**: Shared state containing all block pools and configuration
- **`_cancellation_token`**: Token for graceful shutdown of background tasks
- **`block_size`**: Size of each block in tokens

## Creation and Configuration

### Basic Configuration

```rust
use dynamo_llm::block_manager::{
    KvBlockManager, KvBlockManagerConfig, KvManagerModelConfig, KvManagerRuntimeConfig
};

// Create runtime configuration
let runtime_config = KvManagerRuntimeConfig::builder()
    .worker_id(0)
    .enable_nixl()  // Enable NIXL for remote storage
    .build();

// Create model configuration
let model_config = KvManagerModelConfig::builder()
    .num_layers(32)
    .outer_dim(2)      // Key and Value dimensions
    .page_size(16)     // Tokens per block
    .inner_dim(4096)   // Hidden dimension size
    .dtype(DType::FP16)
    .build();

// Create main configuration
let config = KvBlockManagerConfig::builder()
    .runtime(runtime_config)
    .model(model_config)
    .build()?;

// Create block manager
let block_manager = KvBlockManager::new(config).await?;
```

### Advanced Configuration

```rust
// Configure storage layouts
let device_layout = KvManagerLayoutConfig::builder()
    .num_blocks(1000)
    .layout_type(LayoutType::FullyContiguous)
    .allocator(DeviceAllocator::default())
    .build();

let host_layout = KvManagerLayoutConfig::builder()
    .num_blocks(500)
    .layout_type(LayoutType::Paged)
    .allocator(PinnedAllocator::default())
    .build();

let config = KvBlockManagerConfig::builder()
    .runtime(runtime_config)
    .model(model_config)
    .device_layout(device_layout)
    .host_layout(host_layout)
    .build()?;
```

## Core Operations

### Block Allocation

Allocate new blocks for inference:

```rust
// Allocate blocks from device pool
let device_pool = block_manager.device().unwrap();
let mut_blocks = device_pool.allocate_blocks_blocking(4)?;

// Allocate blocks from host pool
let host_pool = block_manager.host().unwrap();
let host_blocks = host_pool.allocate_blocks_blocking(2)?;
```

### Block Registration

Register completed blocks for reuse:

```rust
// Register blocks after computation
let immutable_blocks = device_pool.register_blocks_blocking(mut_blocks)?;

// Get sequence hashes for matching
let sequence_hashes: Vec<SequenceHash> = immutable_blocks
    .iter()
    .map(|block| block.sequence_hash())
    .collect();
```

### Block Matching

Find existing blocks that match sequence hashes:

```rust
// Match blocks by sequence hash
let matched_blocks = device_pool.match_sequence_hashes_blocking(&sequence_hashes)?;

// Check if all blocks were found
if matched_blocks.len() == sequence_hashes.len() {
    println!("All blocks found in cache!");
} else {
    println!("Some blocks need to be computed");
}
```

### Storage Access

Access different storage pools:

```rust
// Get device pool (GPU memory)
if let Some(device_pool) = block_manager.device() {
    println!("Device blocks: {}", device_pool.total_blocks());
    println!("Available: {}", device_pool.available_blocks());
}

// Get host pool (CPU memory)
if let Some(host_pool) = block_manager.host() {
    println!("Host blocks: {}", host_pool.total_blocks());
}

// Get disk pool (NVMe storage)
if let Some(disk_pool) = block_manager.disk() {
    println!("Disk blocks: {}", disk_pool.total_blocks());
}
```

## Distributed Operations

### Block Export/Import

For distributed scenarios, blocks can be exported and imported:

```rust
// Export local block configuration
let serialized_blockset = block_manager.export_local_blockset()?;

// Import remote block configuration
block_manager.import_remote_blockset(serialized_blockset)?;
```

### Remote Block Access

Access blocks from remote workers:

```rust
// Get immutable remote blocks
let remote_blocks = block_manager.get_remote_blocks_immutable(&block_descriptors)?;

// Get mutable remote blocks
let mut_remote_blocks = block_manager.get_remote_blocks_mutable(&block_descriptors)?;
```

## Block Lifecycle

### 1. Allocation Phase

```rust
// Request blocks from pool
let mut_blocks = pool.allocate_blocks_blocking(count)?;

// Blocks are now in Mutable state
for block in &mut_blocks {
    // Fill block with computed KV cache data
    fill_block_with_data(block, kv_data);
}
```

### 2. Registration Phase

```rust
// Register blocks for reuse
let immutable_blocks = pool.register_blocks_blocking(mut_blocks)?;

// Blocks are now in Immutable state and can be shared
for block in &immutable_blocks {
    let sequence_hash = block.sequence_hash();
    // Store sequence hash for future matching
}
```

### 3. Usage Phase

```rust
// Match blocks by sequence hash
let matched_blocks = pool.match_sequence_hashes_blocking(&hashes)?;

// Use blocks for inference
for block in &matched_blocks {
    // Access block data for attention computation
    let layer_data = block.layer_view(0, 0)?;
    compute_attention(layer_data);
}
```

### 4. Cleanup Phase

```rust
// Blocks are automatically returned to pool when dropped
// No explicit cleanup needed
```

## Error Handling

The block manager provides comprehensive error handling:

```rust
use dynamo_llm::block_manager::BlockPoolError;

match pool.allocate_blocks_blocking(count) {
    Ok(blocks) => {
        // Success - use blocks
    }
    Err(BlockPoolError::NotEnoughBlocksAvailable(requested, available)) => {
        // Handle insufficient blocks
        println!("Requested {} blocks, only {} available", requested, available);
    }
    Err(BlockPoolError::ProgressEngineShutdown) => {
        // Handle shutdown
        println!("Block manager is shutting down");
    }
    Err(e) => {
        // Handle other errors
        println!("Unexpected error: {}", e);
    }
}
```

## Performance Considerations

### Block Reuse

The block manager optimizes memory usage through intelligent block reuse:

```rust
// Check cache hit rate
let total_requests = 1000;
let cache_hits = 750;
let hit_rate = cache_hits as f64 / total_requests as f64;
println!("Cache hit rate: {:.2}%", hit_rate * 100.0);
```

### Memory Efficiency

Monitor memory usage across storage tiers:

```rust
// Check memory usage
let device_usage = block_manager.device()
    .map(|pool| pool.total_blocks() - pool.available_blocks())
    .unwrap_or(0);
let device_total = block_manager.device()
    .map(|pool| pool.total_blocks())
    .unwrap_or(0);

println!("Device memory usage: {}/{} blocks", device_usage, device_total);
```

## Thread Safety

The `KvBlockManager` is designed for concurrent access:

```rust
use std::sync::Arc;
use tokio::spawn;

let block_manager = Arc::new(block_manager);

// Spawn multiple tasks
let handles: Vec<_> = (0..4).map(|i| {
    let bm = block_manager.clone();
    spawn(async move {
        let pool = bm.device().unwrap();
        let blocks = pool.allocate_blocks_blocking(2).unwrap();
        // Process blocks...
    })
}).collect();

// Wait for all tasks
for handle in handles {
    handle.await.unwrap();
}
```

## Next Steps

- [Configuration System](configuration.md) - Learn about configuration options
- [Block Pool](block_pool.md) - Understand block lifecycle management
- [Storage System](storage.md) - Explore storage backends
- [Python API](python/overview.md) - Use the Python interface 