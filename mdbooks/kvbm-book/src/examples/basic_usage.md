# Basic Usage Examples

This section provides practical examples of how to use the KV Block Manager for common scenarios. These examples demonstrate the core functionality and best practices.

## Rust Examples

### Example 1: Basic Block Manager Setup

```rust
use dynamo_llm::block_manager::{
    KvBlockManager, KvBlockManagerConfig, KvManagerModelConfig, KvManagerRuntimeConfig
};
use dynamo_llm::common::dtype::DType;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime configuration
    let runtime_config = KvManagerRuntimeConfig::builder()
        .worker_id(0)
        .build();

    // Create model configuration
    let model_config = KvManagerModelConfig::builder()
        .num_layers(32)
        .outer_dim(2)      // Key and Value
        .page_size(16)     // Tokens per block
        .inner_dim(4096)   // Hidden dimension
        .dtype(DType::FP16)
        .build();

    // Create block manager configuration
    let config = KvBlockManagerConfig::builder()
        .runtime(runtime_config)
        .model(model_config)
        .build()?;

    // Create block manager
    let block_manager = KvBlockManager::new(config).await?;

    println!("Block manager created successfully!");
    println!("Block size: {} tokens", block_manager.block_size());

    Ok(())
}
```

### Example 2: Allocating and Using Blocks

```rust
use dynamo_llm::block_manager::{KvBlockManager, BlockPoolError};

async fn allocate_and_use_blocks(block_manager: &KvBlockManager<BasicMetadata>) -> Result<(), Box<dyn std::error::Error>> {
    // Get device pool
    let device_pool = block_manager.device()
        .ok_or("Device pool not configured")?;

    println!("Total device blocks: {}", device_pool.total_blocks());
    println!("Available device blocks: {}", device_pool.available_blocks());

    // Allocate blocks
    let mut_blocks = device_pool.allocate_blocks_blocking(4)?;
    println!("Allocated {} blocks", mut_blocks.len());

    // Process each block
    for (i, block) in mut_blocks.iter().enumerate() {
        println!("Processing block {}", i);
        
        // Get block information
        let num_layers = block.num_layers();
        let page_size = block.page_size();
        let inner_dim = block.inner_dim();
        
        println!("  Layers: {}, Page size: {}, Inner dim: {}", 
                num_layers, page_size, inner_dim);

        // Access block data for each layer
        for layer_idx in 0..num_layers {
            let layer_view = block.layer_view(layer_idx, 0)?;
            println!("  Layer {} data size: {} bytes", layer_idx, layer_view.len());
            
            // Here you would typically fill the layer with KV cache data
            // fill_layer_with_kv_data(layer_view, kv_data);
        }
    }

    // Register blocks for reuse
    let immutable_blocks = device_pool.register_blocks_blocking(mut_blocks)?;
    println!("Registered {} blocks", immutable_blocks.len());

    // Get sequence hashes for future matching
    let sequence_hashes: Vec<SequenceHash> = immutable_blocks
        .iter()
        .map(|block| block.sequence_hash())
        .collect();

    println!("Sequence hashes: {:?}", sequence_hashes);

    Ok(())
}
```

### Example 3: Block Matching and Reuse

```rust
use dynamo_llm::block_manager::tokens::SequenceHash;

async fn demonstrate_block_reuse(block_manager: &KvBlockManager<BasicMetadata>) -> Result<(), Box<dyn std::error::Error>> {
    let device_pool = block_manager.device()
        .ok_or("Device pool not configured")?;

    // Simulate sequence hashes from a previous computation
    let sequence_hashes = vec![
        SequenceHash::new([1, 2, 3, 4, 5, 6, 7, 8]),
        SequenceHash::new([9, 10, 11, 12, 13, 14, 15, 16]),
    ];

    // Try to match existing blocks
    match device_pool.match_sequence_hashes_blocking(&sequence_hashes) {
        Ok(matched_blocks) => {
            if matched_blocks.len() == sequence_hashes.len() {
                println!("All blocks found in cache! Cache hit rate: 100%");
                
                // Use the matched blocks directly
                for (i, block) in matched_blocks.iter().enumerate() {
                    println!("Using cached block {} with hash {:?}", i, block.sequence_hash());
                }
            } else {
                println!("Partial cache hit: {}/{} blocks found", 
                        matched_blocks.len(), sequence_hashes.len());
                
                // Allocate new blocks for missing sequences
                let missing_count = sequence_hashes.len() - matched_blocks.len();
                let new_blocks = device_pool.allocate_blocks_blocking(missing_count)?;
                println!("Allocated {} new blocks for missing sequences", new_blocks.len());
            }
        }
        Err(BlockPoolError::NotEnoughBlocksAvailable(requested, available)) => {
            println!("Not enough blocks available. Requested: {}, Available: {}", 
                    requested, available);
        }
        Err(e) => {
            println!("Error matching blocks: {}", e);
        }
    }

    Ok(())
}
```

### Example 4: Multi-tier Storage

```rust
use dynamo_llm::block_manager::{
    KvManagerLayoutConfig, LayoutType
};
use dynamo_llm::block_manager::storage::{
    DeviceAllocator, PinnedAllocator, DiskAllocator
};

async fn setup_multi_tier_storage() -> Result<KvBlockManager<BasicMetadata>, Box<dyn std::error::Error>> {
    // Runtime and model configuration
    let runtime_config = KvManagerRuntimeConfig::builder()
        .worker_id(0)
        .build();

    let model_config = KvManagerModelConfig::builder()
        .num_layers(32)
        .outer_dim(2)
        .page_size(16)
        .inner_dim(4096)
        .dtype(DType::FP16)
        .build();

    // Device layout (GPU memory) - Fast access, limited capacity
    let device_layout = KvManagerLayoutConfig::builder()
        .num_blocks(1000)
        .layout_type(LayoutType::FullyContiguous)
        .allocator(DeviceAllocator::default())
        .build();

    // Host layout (CPU memory) - Medium speed, larger capacity
    let host_layout = KvManagerLayoutConfig::builder()
        .num_blocks(5000)
        .layout_type(LayoutType::Paged)
        .allocator(PinnedAllocator::default())
        .build();

    // Disk layout (NVMe storage) - Slow access, very large capacity
    let disk_layout = KvManagerLayoutConfig::builder()
        .num_blocks(50000)
        .layout_type(LayoutType::Paged)
        .allocator(DiskAllocator::new("/mnt/nvme/kv_cache"))
        .build();

    // Create block manager with multi-tier storage
    let config = KvBlockManagerConfig::builder()
        .runtime(runtime_config)
        .model(model_config)
        .device_layout(device_layout)
        .host_layout(host_layout)
        .disk_layout(disk_layout)
        .build()?;

    let block_manager = KvBlockManager::new(config).await?;

    // Print storage information
    if let Some(device_pool) = block_manager.device() {
        println!("Device storage: {} blocks", device_pool.total_blocks());
    }
    
    if let Some(host_pool) = block_manager.host() {
        println!("Host storage: {} blocks", host_pool.total_blocks());
    }
    
    if let Some(disk_pool) = block_manager.disk() {
        println!("Disk storage: {} blocks", disk_pool.total_blocks());
    }

    Ok(block_manager)
}
```

### Example 5: Error Handling and Recovery

```rust
use dynamo_llm::block_manager::{BlockPoolError, BlockError};

async fn robust_block_allocation(block_manager: &KvBlockManager<BasicMetadata>) -> Result<(), Box<dyn std::error::Error>> {
    let device_pool = block_manager.device()
        .ok_or("Device pool not configured")?;

    // Try to allocate blocks with fallback strategies
    let mut allocated_blocks = Vec::new();
    let requested_blocks = 10;

    // Strategy 1: Try device memory first
    match device_pool.allocate_blocks_blocking(requested_blocks) {
        Ok(blocks) => {
            println!("Successfully allocated {} blocks from device memory", blocks.len());
            allocated_blocks = blocks;
        }
        Err(BlockPoolError::NotEnoughBlocksAvailable(requested, available)) => {
            println!("Device memory insufficient. Requested: {}, Available: {}", requested, available);
            
            // Strategy 2: Try to allocate what's available
            match device_pool.allocate_blocks_blocking(available) {
                Ok(blocks) => {
                    println!("Allocated {} blocks from device memory", blocks.len());
                    allocated_blocks = blocks;
                }
                Err(e) => {
                    println!("Failed to allocate from device memory: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Device allocation failed: {}", e);
        }
    }

    // Strategy 3: Fallback to host memory if needed
    if allocated_blocks.is_empty() {
        if let Some(host_pool) = block_manager.host() {
            match host_pool.allocate_blocks_blocking(requested_blocks) {
                Ok(blocks) => {
                    println!("Allocated {} blocks from host memory", blocks.len());
                    allocated_blocks = blocks;
                }
                Err(e) => {
                    println!("Host allocation failed: {}", e);
                }
            }
        }
    }

    // Strategy 4: Fallback to disk storage if needed
    if allocated_blocks.is_empty() {
        if let Some(disk_pool) = block_manager.disk() {
            match disk_pool.allocate_blocks_blocking(requested_blocks) {
                Ok(blocks) => {
                    println!("Allocated {} blocks from disk storage", blocks.len());
                    allocated_blocks = blocks;
                }
                Err(e) => {
                    println!("Disk allocation failed: {}", e);
                }
            }
        }
    }

    if allocated_blocks.is_empty() {
        return Err("Failed to allocate blocks from any storage tier".into());
    }

    println!("Successfully allocated {} blocks", allocated_blocks.len());
    
    // Process blocks...
    for block in allocated_blocks {
        // Process block
    }

    Ok(())
}
```

## Python Examples

### Example 1: Basic Python Usage

```python
import dynamo_llm
import torch

def basic_block_manager_example():
    # Create block manager
    block_manager = dynamo_llm.BlockManager(
        num_layers=32,
        page_size=16,
        inner_dim=4096,
        dtype=dynamo_llm.DType.FP16
    )
    
    print(f"Block manager created with block size: {block_manager.block_size}")
    
    # Allocate blocks
    blocks = block_manager.allocate_blocks(4)
    print(f"Allocated {len(blocks)} blocks")
    
    # Process blocks
    for i, block in enumerate(blocks):
        print(f"Block {i} has {len(block)} layers")
        
        # Access layer data
        for layer_idx in range(len(block)):
            layer = block[layer_idx]
            
            # Convert to PyTorch tensor
            tensor = torch.from_dlpack(layer.__dlpack__())
            print(f"  Layer {layer_idx} shape: {tensor.shape}")
            
            # Perform operations
            mean = tensor.mean().item()
            std = tensor.std().item()
            print(f"  Layer {layer_idx} stats: mean={mean:.4f}, std={std:.4f}")

if __name__ == "__main__":
    basic_block_manager_example()
```

### Example 2: Block Pool Management

```python
import dynamo_llm

def block_pool_example():
    # Create block manager
    block_manager = dynamo_llm.BlockManager(
        num_layers=32,
        page_size=16,
        inner_dim=4096
    )
    
    # Access different pools
    device_pool = block_manager.get_device_pool()
    host_pool = block_manager.get_host_pool()
    
    if device_pool:
        print(f"Device pool: {device_pool.total_blocks()} total, "
              f"{device_pool.available_blocks()} available")
        
        # Allocate from device pool
        device_blocks = device_pool.allocate_blocks(2)
        print(f"Allocated {len(device_blocks)} blocks from device")
        
        # Check memory usage
        usage = device_pool.usage()
        print(f"Device memory usage: {usage:.2%}")
    
    if host_pool:
        print(f"Host pool: {host_pool.total_blocks()} total, "
              f"{host_pool.available_blocks()} available")
        
        # Allocate from host pool
        host_blocks = host_pool.allocate_blocks(1)
        print(f"Allocated {len(host_blocks)} blocks from host")

if __name__ == "__main__":
    block_pool_example()
```

### Example 3: Memory Monitoring

```python
import dynamo_llm
import time

def monitor_memory_usage():
    # Create block manager
    block_manager = dynamo_llm.BlockManager(
        num_layers=32,
        page_size=16,
        inner_dim=4096
    )
    
    device_pool = block_manager.get_device_pool()
    if not device_pool:
        print("Device pool not available")
        return
    
    print("Monitoring memory usage...")
    print("Time | Total | Available | Used | Usage %")
    print("-" * 50)
    
    for i in range(10):
        total = device_pool.total_blocks()
        available = device_pool.available_blocks()
        used = total - available
        usage = used / total * 100
        
        print(f"{i:4d} | {total:5d} | {available:9d} | {used:4d} | {usage:6.1f}%")
        
        # Allocate some blocks to see usage change
        if i % 3 == 0:
            try:
                blocks = device_pool.allocate_blocks(10)
                print(f"     Allocated {len(blocks)} blocks")
            except Exception as e:
                print(f"     Allocation failed: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    monitor_memory_usage()
```

### Example 4: Error Handling

```python
import dynamo_llm

def error_handling_example():
    try:
        # Create block manager
        block_manager = dynamo_llm.BlockManager(
            num_layers=32,
            page_size=16,
            inner_dim=4096
        )
        
        # Try to allocate more blocks than available
        device_pool = block_manager.get_device_pool()
        if device_pool:
            total_blocks = device_pool.total_blocks()
            
            try:
                # Try to allocate more than available
                blocks = device_pool.allocate_blocks(total_blocks + 10)
                print(f"Allocated {len(blocks)} blocks")
            except dynamo_llm.BlockManagerError as e:
                print(f"Block manager error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
        
    except Exception as e:
        print(f"Failed to create block manager: {e}")

if __name__ == "__main__":
    error_handling_example()
```

## Performance Examples

### Example 1: Batch Processing

```rust
use dynamo_llm::block_manager::{KvBlockManager, BasicMetadata};
use std::sync::Arc;
use tokio::spawn;

async fn batch_processing_example(block_manager: Arc<KvBlockManager<BasicMetadata>>) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 10;
    let num_batches = 5;
    
    let mut handles = Vec::new();
    
    // Spawn multiple batch processing tasks
    for batch_id in 0..num_batches {
        let bm = block_manager.clone();
        let handle = spawn(async move {
            let device_pool = bm.device().unwrap();
            let blocks = device_pool.allocate_blocks_blocking(batch_size).unwrap();
            
            println!("Batch {}: Processing {} blocks", batch_id, blocks.len());
            
            // Process blocks
            for block in blocks {
                // Simulate processing
                for layer_idx in 0..block.num_layers() {
                    let _layer_view = block.layer_view(layer_idx, 0).unwrap();
                    // Process layer data
                }
            }
            
            println!("Batch {}: Completed", batch_id);
        });
        
        handles.push(handle);
    }
    
    // Wait for all batches to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    println!("All batches completed!");
    Ok(())
}
```

### Example 2: Memory-Efficient Processing

```python
import dynamo_llm
import gc

def memory_efficient_processing():
    block_manager = dynamo_llm.BlockManager(
        num_layers=32,
        page_size=16,
        inner_dim=4096
    )
    
    device_pool = block_manager.get_device_pool()
    if not device_pool:
        return
    
    print("Memory-efficient processing...")
    
    for i in range(100):
        # Allocate blocks
        blocks = device_pool.allocate_blocks(5)
        
        # Process blocks
        for block in blocks:
            for layer in block:
                # Process layer data
                tensor = torch.from_dlpack(layer.__dlpack__())
                result = torch.softmax(tensor, dim=-1)
                # Use result...
        
        # Explicitly delete blocks to return to pool
        del blocks
        
        # Force garbage collection
        gc.collect()
        
        # Monitor memory usage
        if i % 10 == 0:
            usage = device_pool.usage()
            print(f"Iteration {i}: Memory usage {usage:.2%}")

if __name__ == "__main__":
    memory_efficient_processing()
```

## Next Steps

- [Advanced Usage](advanced_usage.md) - More complex examples and patterns
- [vLLM Integration](vllm_integration.md) - Production deployment examples
- [Performance Optimization](advanced/performance.md) - Performance tuning guides
- [API Reference](api/rust.md) - Complete API documentation 