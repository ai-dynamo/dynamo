# Configuration System

The KV Block Manager uses a comprehensive configuration system that allows fine-grained control over all aspects of block management, storage, and performance. This document covers all configuration options and their usage.

## Configuration Structure

The configuration system is built around several key structures:

```rust
pub struct KvBlockManagerConfig {
    pub runtime: KvManagerRuntimeConfig,
    pub model: KvManagerModelConfig,
    pub device_layout: Option<KvManagerLayoutConfig<DeviceStorage>>,
    pub host_layout: Option<KvManagerLayoutConfig<PinnedStorage>>,
    pub disk_layout: Option<KvManagerLayoutConfig<DiskStorage>>,
    pub event_manager: Option<Arc<dyn EventManager>>,
}
```

## Runtime Configuration

The `KvManagerRuntimeConfig` controls the runtime behavior of the block manager:

```rust
#[derive(Debug, Clone, Builder, Validate)]
pub struct KvManagerRuntimeConfig {
    pub worker_id: u64,
    pub cancellation_token: CancellationToken,
    pub nixl: NixlOptions,
    pub async_runtime: Option<Arc<tokio::runtime::Runtime>>,
    pub metrics_registry: Arc<Registry>,
}
```

### Basic Runtime Configuration

```rust
let runtime_config = KvManagerRuntimeConfig::builder()
    .worker_id(0)
    .build();
```

### Advanced Runtime Configuration

```rust
use prometheus::Registry;
use tokio::runtime::Runtime;

// Create custom runtime
let runtime = Arc::new(Runtime::new()?);

// Create metrics registry
let metrics_registry = Arc::new(Registry::new());

// Configure NIXL
let nixl_agent = NixlAgent::new("my_agent")?;

let runtime_config = KvManagerRuntimeConfig::builder()
    .worker_id(0)
    .use_nixl_agent(nixl_agent)
    .async_runtime(runtime)
    .metrics_registry(metrics_registry)
    .build();
```

### NIXL Options

NIXL (Network Interface for eXternal Libraries) provides remote storage capabilities:

```rust
#[derive(Debug, Clone)]
pub enum NixlOptions {
    /// Enable NIXL and create a new NIXL agent
    Enabled,
    
    /// Enable NIXL and use the provided NIXL agent
    EnabledWithAgent(NixlAgent),
    
    /// Disable NIXL
    Disabled,
}
```

Usage examples:

```rust
// Enable NIXL with default agent
let config = KvManagerRuntimeConfig::builder()
    .worker_id(0)
    .enable_nixl()
    .build();

// Enable NIXL with custom agent
let agent = NixlAgent::new("custom_agent")?;
let config = KvManagerRuntimeConfig::builder()
    .worker_id(0)
    .use_nixl_agent(agent)
    .build();

// Disable NIXL
let config = KvManagerRuntimeConfig::builder()
    .worker_id(0)
    .disable_nixl()
    .build();
```

## Model Configuration

The `KvManagerModelConfig` defines the model-specific parameters:

```rust
#[derive(Debug, Clone, Builder, Validate)]
pub struct KvManagerModelConfig {
    #[validate(range(min = 1))]
    pub num_layers: usize,
    
    #[validate(range(min = 1, max = 2))]
    pub outer_dim: usize,
    
    #[validate(range(min = 1))]
    pub page_size: usize,
    
    #[validate(range(min = 1))]
    pub inner_dim: usize,
    
    #[builder(default = "DType::FP16")]
    pub dtype: DType,
}
```

### Model Configuration Examples

```rust
// Basic model configuration
let model_config = KvManagerModelConfig::builder()
    .num_layers(32)
    .outer_dim(2)      // Key and Value
    .page_size(16)     // Tokens per block
    .inner_dim(4096)   // Hidden dimension
    .build();

// Advanced model configuration
let model_config = KvManagerModelConfig::builder()
    .num_layers(48)
    .outer_dim(2)
    .page_size(32)     // Larger blocks for better efficiency
    .inner_dim(8192)   // Larger hidden dimension
    .dtype(DType::BF16) // Use bfloat16
    .build();
```

### Validation Rules

The model configuration includes validation rules:

- `num_layers`: Must be at least 1
- `outer_dim`: Must be between 1 and 2 (typically 2 for Key/Value)
- `page_size`: Must be at least 1
- `inner_dim`: Must be at least 1

## Layout Configuration

The `KvManagerLayoutConfig` controls how blocks are organized in memory:

```rust
#[derive(Builder, Validate)]
pub struct KvManagerLayoutConfig<S: Storage + NixlRegisterableStorage> {
    #[validate(range(min = 1))]
    pub num_blocks: usize,
    
    #[builder(default = "LayoutType::FullyContiguous")]
    pub layout_type: LayoutType,
    
    pub storage: Option<Vec<S>>,
    pub allocator: Option<Arc<dyn StorageAllocator<S>>>,
    pub logical: Option<BlockParallelismStrategy>,
}
```

### Layout Types

```rust
#[derive(Debug, Clone)]
pub enum LayoutType {
    /// Simple contiguous memory layout
    FullyContiguous,
    
    /// Block-based memory organization
    Paged,
    
    /// Custom layout strategy
    Custom(Arc<dyn LayoutStrategy>),
}
```

### Layout Configuration Examples

#### Device Layout (GPU Memory)

```rust
use dynamo_llm::block_manager::storage::DeviceAllocator;

let device_layout = KvManagerLayoutConfig::builder()
    .num_blocks(1000)
    .layout_type(LayoutType::FullyContiguous)
    .allocator(DeviceAllocator::default())
    .build();
```

#### Host Layout (CPU Memory)

```rust
use dynamo_llm::block_manager::storage::PinnedAllocator;

let host_layout = KvManagerLayoutConfig::builder()
    .num_blocks(500)
    .layout_type(LayoutType::Paged)
    .allocator(PinnedAllocator::default())
    .build();
```

#### Disk Layout (NVMe Storage)

```rust
use dynamo_llm::block_manager::storage::DiskAllocator;

let disk_layout = KvManagerLayoutConfig::builder()
    .num_blocks(10000)
    .layout_type(LayoutType::Paged)
    .allocator(DiskAllocator::new("/path/to/storage"))
    .build();
```

### Block Parallelism Strategy

For distributed scenarios, you can configure block parallelism:

```rust
#[derive(Debug, Clone)]
pub enum BlockParallelismStrategy {
    /// KV blocks are sharded across all workers
    LeaderWorkerSharded,
}
```

```rust
let device_layout = KvManagerLayoutConfig::builder()
    .num_blocks(1000)
    .layout_type(LayoutType::FullyContiguous)
    .logical(BlockParallelismStrategy::LeaderWorkerSharded)
    .build();
```

## Complete Configuration Example

Here's a complete example showing all configuration options:

```rust
use dynamo_llm::block_manager::{
    KvBlockManager, KvBlockManagerConfig, KvManagerModelConfig, KvManagerRuntimeConfig,
    KvManagerLayoutConfig, LayoutType, BlockParallelismStrategy
};
use dynamo_llm::block_manager::storage::{DeviceAllocator, PinnedAllocator, DiskAllocator};
use dynamo_llm::common::dtype::DType;

// Runtime configuration
let runtime_config = KvManagerRuntimeConfig::builder()
    .worker_id(0)
    .enable_nixl()
    .build();

// Model configuration
let model_config = KvManagerModelConfig::builder()
    .num_layers(32)
    .outer_dim(2)
    .page_size(16)
    .inner_dim(4096)
    .dtype(DType::FP16)
    .build();

// Device layout (GPU memory)
let device_layout = KvManagerLayoutConfig::builder()
    .num_blocks(1000)
    .layout_type(LayoutType::FullyContiguous)
    .allocator(DeviceAllocator::default())
    .build();

// Host layout (CPU memory)
let host_layout = KvManagerLayoutConfig::builder()
    .num_blocks(500)
    .layout_type(LayoutType::Paged)
    .allocator(PinnedAllocator::default())
    .build();

// Disk layout (NVMe storage)
let disk_layout = KvManagerLayoutConfig::builder()
    .num_blocks(10000)
    .layout_type(LayoutType::Paged)
    .allocator(DiskAllocator::new("/mnt/nvme/kv_cache"))
    .build();

// Main configuration
let config = KvBlockManagerConfig::builder()
    .runtime(runtime_config)
    .model(model_config)
    .device_layout(device_layout)
    .host_layout(host_layout)
    .disk_layout(disk_layout)
    .build()?;

// Create block manager
let block_manager = KvBlockManager::new(config).await?;
```

## Configuration Validation

The configuration system includes comprehensive validation:

```rust
// Validation will catch configuration errors
match KvBlockManagerConfig::builder()
    .runtime(runtime_config)
    .model(model_config)
    .build() {
    Ok(config) => {
        println!("Configuration is valid");
    }
    Err(e) => {
        println!("Configuration error: {}", e);
    }
}
```

### Common Validation Errors

- **Invalid model parameters**: Layer count, dimensions, or page size out of range
- **Missing storage configuration**: No storage layouts provided
- **Conflicting options**: Multiple storage/allocator options specified
- **Invalid NIXL configuration**: NIXL agent creation failed

## Configuration Best Practices

### 1. Memory Sizing

```rust
// Calculate optimal block counts based on available memory
let gpu_memory_gb = 24;
let block_size_bytes = num_layers * outer_dim * page_size * inner_dim * dtype_size;
let max_blocks = (gpu_memory_gb * 1024 * 1024 * 1024) / block_size_bytes;

let device_layout = KvManagerLayoutConfig::builder()
    .num_blocks(max_blocks * 80 / 100) // Use 80% of available memory
    .build();
```

### 2. Multi-tier Storage

```rust
// Configure storage hierarchy
let device_layout = KvManagerLayoutConfig::builder()
    .num_blocks(1000)  // Fast access, limited capacity
    .build();

let host_layout = KvManagerLayoutConfig::builder()
    .num_blocks(5000)  // Medium speed, larger capacity
    .build();

let disk_layout = KvManagerLayoutConfig::builder()
    .num_blocks(50000) // Slow access, very large capacity
    .build();
```

### 3. Distributed Configuration

```rust
// Configure for distributed inference
let runtime_config = KvManagerRuntimeConfig::builder()
    .worker_id(worker_id)
    .enable_nixl()
    .build();

let device_layout = KvManagerLayoutConfig::builder()
    .num_blocks(1000)
    .logical(BlockParallelismStrategy::LeaderWorkerSharded)
    .build();
```

## Environment-based Configuration

You can create configuration based on environment variables:

```rust
fn create_config_from_env() -> Result<KvBlockManagerConfig> {
    let worker_id = std::env::var("WORKER_ID")
        .unwrap_or_else(|_| "0".to_string())
        .parse()?;
    
    let num_layers = std::env::var("NUM_LAYERS")
        .unwrap_or_else(|_| "32".to_string())
        .parse()?;
    
    let gpu_memory_gb = std::env::var("GPU_MEMORY_GB")
        .unwrap_or_else(|_| "24".to_string())
        .parse()?;
    
    let runtime_config = KvManagerRuntimeConfig::builder()
        .worker_id(worker_id)
        .build();
    
    let model_config = KvManagerModelConfig::builder()
        .num_layers(num_layers)
        .outer_dim(2)
        .page_size(16)
        .inner_dim(4096)
        .build();
    
    let device_layout = KvManagerLayoutConfig::builder()
        .num_blocks(calculate_blocks(gpu_memory_gb))
        .allocator(DeviceAllocator::default())
        .build();
    
    KvBlockManagerConfig::builder()
        .runtime(runtime_config)
        .model(model_config)
        .device_layout(device_layout)
        .build()
}
```

## Next Steps

- [Storage System](storage.md) - Learn about storage backends
- [Block Pool](block_pool.md) - Understand block lifecycle management
- [Layout Management](layout.md) - Explore data layout strategies
- [Python Configuration](python/overview.md) - Configure from Python 