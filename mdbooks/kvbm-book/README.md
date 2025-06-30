# KV Block Manager Documentation

This is the official documentation for the KV Block Manager, a high-performance system for managing key-value (KV) cache blocks in Large Language Model (LLM) inference.

## Overview

The KV Block Manager provides:

- **Multi-tier Storage**: Support for GPU memory, CPU memory, local NVMe, and remote storage
- **Block Reuse**: Intelligent caching and reuse of KV blocks to reduce memory footprint
- **Distributed Support**: Built-in support for distributed inference across multiple workers
- **Python Integration**: Native Python bindings with DLPack support
- **vLLM Compatibility**: Direct integration with vLLM for production deployments

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
```

## Documentation Structure

### Core Architecture
- [Overview](src/core/overview.md) - High-level architecture overview
- [Block Manager](src/core/block_manager.md) - Main orchestrator component
- [Configuration](src/core/configuration.md) - Configuration system
- [Storage System](src/core/storage.md) - Storage backends and management
- [Block Pool](src/core/block_pool.md) - Block lifecycle management
- [Block Data](src/core/block_data.md) - Block data structures and views
- [Layout Management](src/core/layout.md) - Data layout strategies
- [Offloading](src/core/offload.md) - Block movement between tiers
- [Distributed Management](src/core/distributed.md) - Distributed operations
- [Events and Metrics](src/core/events_metrics.md) - Monitoring and observability

### Python Bindings
- [Python API Overview](src/python/overview.md) - Python interface overview
- [Block Interface](src/python/block.md) - Block and Layer classes
- [Layer Interface](src/python/layer.md) - Layer data access
- [DLPack Integration](src/python/dlpack.md) - Tensor interoperability
- [vLLM Integration](src/python/vllm.md) - Production deployment
- [Block Lists](src/python/block_list.md) - Block collection management

### Advanced Topics
- [Performance Optimization](src/advanced/performance.md) - Performance tuning
- [Memory Management](src/advanced/memory.md) - Memory optimization
- [Error Handling](src/advanced/errors.md) - Error handling patterns
- [Best Practices](src/advanced/best_practices.md) - Development best practices

### API Reference
- [Rust API Reference](src/api/rust.md) - Complete Rust API documentation
- [Python API Reference](src/api/python.md) - Complete Python API documentation
- [Configuration Reference](src/api/config.md) - Configuration options

### Examples
- [Basic Usage](src/examples/basic_usage.md) - Basic usage examples
- [Advanced Usage](src/examples/advanced_usage.md) - Advanced patterns
- [vLLM Integration](src/examples/vllm_integration.md) - Production examples

## Building the Documentation

This documentation is built using [mdBook](https://rust-lang.github.io/mdBook/).

### Prerequisites

```bash
# Install mdBook
cargo install mdbook
```

### Building

```bash
# Build the documentation
mdbook build

# Serve the documentation locally
mdbook serve
```

The built documentation will be available in the `book/` directory.

## Contributing

To contribute to the documentation:

1. Edit the markdown files in the `src/` directory
2. Test your changes locally with `mdbook serve`
3. Submit a pull request with your changes

## License

This documentation is licensed under the Apache License, Version 2.0.

## Support

For questions and support:

- [GitHub Issues](https://github.com/nvidia/dynamo/issues)
- [Documentation Issues](https://github.com/nvidia/dynamo/issues?q=label%3Adocumentation)
- [Discussions](https://github.com/nvidia/dynamo/discussions) 