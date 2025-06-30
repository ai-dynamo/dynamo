# KV Block Manager

The KV Block Manager is a high-performance, memory-efficient system for managing key-value (KV) cache blocks in Large Language Model (LLM) inference. It provides a unified interface for managing blocks across different storage types including GPU memory, CPU memory, and remote storage.

## Key Features

- **Multi-tier Storage**: Support for GPU memory, CPU memory, local NVMe, and remote storage
- **Block Reuse**: Intelligent caching and reuse of KV blocks to reduce memory footprint
- **Distributed Support**: Built-in support for distributed inference across multiple workers
- **Python Integration**: Native Python bindings with DLPack support for seamless integration
- **vLLM Compatibility**: Direct integration with vLLM for production deployments
- **Performance Optimized**: Designed for high-throughput inference with minimal latency

## Architecture Overview

The KV Block Manager consists of several core components:

- **Block Manager**: The main orchestrator that manages block allocation and lifecycle
- **Block Pool**: Efficient pool management for block reuse and allocation
- **Storage System**: Unified interface for different storage backends
- **Layout Management**: Flexible data layout strategies for optimal performance
- **Offloading**: Intelligent block movement between storage tiers
- **Distributed Management**: Coordination across multiple workers

## Quick Start

### Rust Usage

```rust
use dynamo_llm::block_manager::{
    KvBlockManager, KvBlockManagerConfig, KvManagerModelConfig, KvManagerRuntimeConfig
};

// Create configuration
let config = KvBlockManagerConfig::builder()
    .runtime(KvManagerRuntimeConfig::builder()
        .worker_id(0)
        .build())
    .model(KvManagerModelConfig::builder()
        .num_layers(32)
        .page_size(16)
        .inner_dim(4096)
        .build())
    .build()?;

// Create block manager
let block_manager = KvBlockManager::new(config).await?;
```

### Python Usage

```python
import dynamo_llm

# Create block manager
block_manager = dynamo_llm.BlockManager(
    num_layers=32,
    page_size=16,
    inner_dim=4096
)

# Allocate blocks
blocks = block_manager.allocate_blocks(4)

# Use blocks for inference
for block in blocks:
    # Access layer data
    layer_data = block[0]  # Get first layer
    # Process layer...
```

## Storage Tiers

The KV Block Manager supports multiple storage tiers:

- **G1 (GPU Memory)**: Fastest access, limited capacity
- **G2 (CPU Memory)**: Medium speed, larger capacity
- **G3 (Local NVMe)**: Slower access, very large capacity
- **G4 (Remote Storage)**: Slowest access, unlimited capacity

## Performance Characteristics

- **Latency**: Sub-millisecond block allocation and access
- **Throughput**: Support for thousands of concurrent requests
- **Memory Efficiency**: Intelligent block reuse reduces memory footprint by 50-80%
- **Scalability**: Linear scaling with additional workers

## Integration

The KV Block Manager integrates seamlessly with:

- **vLLM**: Production-ready inference serving
- **PyTorch**: Native tensor operations via DLPack
- **Custom Inference Engines**: Flexible API for custom implementations
- **Distributed Systems**: Built-in support for multi-node deployments

## Next Steps

- [Core Architecture](core/overview.md) - Learn about the internal architecture
- [Python API](python/overview.md) - Explore the Python interface
- [Examples](examples/basic_usage.md) - See practical usage examples
- [API Reference](api/rust.md) - Detailed API documentation 