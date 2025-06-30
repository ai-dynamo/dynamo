# Rust API Reference

This document provides a comprehensive reference for the Rust API of the KV Block Manager.

## Core Types

### KvBlockManager

The main entry point for block management operations.

```rust
pub struct KvBlockManager<Metadata: BlockMetadata> {
    state: Arc<state::KvBlockManagerState<locality::Local, Metadata>>,
    _cancellation_token: Arc<CancelOnLastDrop>,
    block_size: usize,
}
```

#### Methods

```rust
impl<Metadata: BlockMetadata> KvBlockManager<Metadata> {
    /// Create a new KvBlockManager
    pub async fn new(config: KvBlockManagerConfig) -> Result<Self>
    
    /// Get the block size in tokens
    pub fn block_size(&self) -> usize
    
    /// Get the worker ID
    pub fn worker_id(&self) -> WorkerID
    
    /// Export local blockset configuration
    pub fn export_local_blockset(&self) -> Result<SerializedNixlBlockSet>
    
    /// Import remote blockset configuration
    pub fn import_remote_blockset(&self, serialized_blockset: SerializedNixlBlockSet) -> Result<()>
    
    /// Get immutable remote blocks
    pub fn get_remote_blocks_immutable(&self, bds: &BlockDescriptorList) -> Result<Vec<RemoteBlock<IsImmutable>>>
    
    /// Get mutable remote blocks
    pub fn get_remote_blocks_mutable(&self, bds: &BlockDescriptorList) -> Result<Vec<RemoteBlock<IsMutable>>>
    
    /// Get device block pool
    pub fn device(&self) -> Option<&BlockPool<DeviceStorage, locality::Local, Metadata>>
    
    /// Get host block pool
    pub fn host(&self) -> Option<&BlockPool<PinnedStorage, locality::Local, Metadata>>
    
    /// Get disk block pool
    pub fn disk(&self) -> Option<&BlockPool<DiskStorage, locality::Local, Metadata>>
}
```

### BlockPool

Manages the lifecycle of blocks in a specific storage backend.

```rust
pub struct BlockPool<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    priority_tx: tokio::sync::mpsc::UnboundedSender<PriorityRequest<S, L, M>>,
    ctrl_tx: tokio::sync::mpsc::UnboundedSender<ControlRequest<S, L, M>>,
    available_blocks_counter: Arc<AtomicU64>,
    total_blocks_counter: Arc<AtomicU64>,
}
```

#### Methods

```rust
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> BlockPool<S, L, M> {
    /// Create a new BlockPool
    pub fn new(args: BlockPoolArgs<S, L, M>) -> Result<Self>
    
    /// Get total number of blocks
    pub fn total_blocks(&self) -> u64
    
    /// Get number of available blocks
    pub fn available_blocks(&self) -> u64
    
    /// Allocate blocks (async)
    pub async fn allocate_blocks(&self, count: usize) -> Result<Vec<MutableBlock<S, L, M>>, BlockPoolError>
    
    /// Allocate blocks (blocking)
    pub fn allocate_blocks_blocking(&self, count: usize) -> Result<Vec<MutableBlock<S, L, M>>, BlockPoolError>
    
    /// Register blocks (async)
    pub async fn register_blocks(&self, blocks: Vec<MutableBlock<S, L, M>>) -> Result<Vec<ImmutableBlock<S, L, M>>, BlockPoolError>
    
    /// Register blocks (blocking)
    pub fn register_blocks_blocking(&self, blocks: Vec<MutableBlock<S, L, M>>) -> Result<Vec<ImmutableBlock<S, L, M>>, BlockPoolError>
    
    /// Match sequence hashes (async)
    pub async fn match_sequence_hashes(&self, sequence_hashes: &[SequenceHash]) -> Result<Vec<ImmutableBlock<S, L, M>>, BlockPoolError>
    
    /// Match sequence hashes (blocking)
    pub fn match_sequence_hashes_blocking(&self, sequence_hashes: &[SequenceHash]) -> Result<Vec<ImmutableBlock<S, L, M>>, BlockPoolError>
    
    /// Add blocks to pool
    pub async fn add_blocks(&self, blocks: Vec<Block<S, L, M>>) -> Result<(), BlockPoolError>
    
    /// Add blocks to pool (blocking)
    pub fn add_blocks_blocking(&self, blocks: Vec<Block<S, L, M>>) -> Result<(), BlockPoolError>
}
```

### MutableBlock

Represents a uniquely owned block that can be modified.

```rust
pub struct MutableBlock<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    block: Block<S, L, M>,
    state: BlockState,
}
```

#### Methods

```rust
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> MutableBlock<S, L, M> {
    /// Get block ID
    pub fn block_id(&self) -> BlockId
    
    /// Get number of layers
    pub fn num_layers(&self) -> usize
    
    /// Get page size
    pub fn page_size(&self) -> usize
    
    /// Get inner dimension
    pub fn inner_dim(&self) -> usize
    
    /// Get number of outer dimensions
    pub fn num_outer_dims(&self) -> usize
    
    /// Get layer view
    pub fn layer_view(&self, layer_idx: usize, outer_idx: usize) -> BlockResult<view::LayerView<S>>
    
    /// Get mutable layer view
    pub fn layer_view_mut(&mut self, layer_idx: usize, outer_idx: usize) -> BlockResult<view::LayerViewMut<S>>
    
    /// Get block view
    pub fn block_view(&self) -> BlockResult<view::BlockView<S>>
    
    /// Get mutable block view
    pub fn block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<S>>
    
    /// Get sequence hash
    pub fn sequence_hash(&self) -> SequenceHash
    
    /// Check if block is complete
    pub fn is_complete(&self) -> bool
    
    /// Mark block as complete
    pub fn mark_complete(&mut self)
}
```

### ImmutableBlock

Represents a shared, immutable reference to a block.

```rust
pub struct ImmutableBlock<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    block: Arc<Block<S, L, M>>,
    state: BlockState,
}
```

#### Methods

```rust
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> ImmutableBlock<S, L, M> {
    /// Get block ID
    pub fn block_id(&self) -> BlockId
    
    /// Get number of layers
    pub fn num_layers(&self) -> usize
    
    /// Get page size
    pub fn page_size(&self) -> usize
    
    /// Get inner dimension
    pub fn inner_dim(&self) -> usize
    
    /// Get number of outer dimensions
    pub fn num_outer_dims(&self) -> usize
    
    /// Get layer view
    pub fn layer_view(&self, layer_idx: usize, outer_idx: usize) -> BlockResult<view::LayerView<S>>
    
    /// Get block view
    pub fn block_view(&self) -> BlockResult<view::BlockView<S>>
    
    /// Get sequence hash
    pub fn sequence_hash(&self) -> SequenceHash
}
```

## Configuration Types

### KvBlockManagerConfig

Main configuration for the block manager.

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

### KvManagerRuntimeConfig

Runtime configuration for the block manager.

```rust
pub struct KvManagerRuntimeConfig {
    pub worker_id: u64,
    pub cancellation_token: CancellationToken,
    pub nixl: NixlOptions,
    pub async_runtime: Option<Arc<tokio::runtime::Runtime>>,
    pub metrics_registry: Arc<Registry>,
}
```

### KvManagerModelConfig

Model-specific configuration.

```rust
pub struct KvManagerModelConfig {
    pub num_layers: usize,
    pub outer_dim: usize,
    pub page_size: usize,
    pub inner_dim: usize,
    pub dtype: DType,
}
```

### KvManagerLayoutConfig

Layout configuration for a storage type.

```rust
pub struct KvManagerLayoutConfig<S: Storage + NixlRegisterableStorage> {
    pub num_blocks: usize,
    pub layout_type: LayoutType,
    pub storage: Option<Vec<S>>,
    pub allocator: Option<Arc<dyn StorageAllocator<S>>>,
    pub logical: Option<BlockParallelismStrategy>,
}
```

## Storage Types

### Storage Trait

Core trait for storage backends.

```rust
pub trait Storage: Debug + Send + Sync + 'static {
    fn storage_type(&self) -> StorageType;
    fn addr(&self) -> u64;
    fn size(&self) -> usize;
    unsafe fn as_ptr(&self) -> *const u8;
    unsafe fn as_mut_ptr(&mut self) -> *mut u8;
}
```

### StorageType

Enumeration of supported storage types.

```rust
pub enum StorageType {
    System,
    Device(u32),
    Pinned,
    Disk(u64),
    Nixl,
    Null,
}
```

### Storage Allocators

```rust
// System memory allocator
pub struct SystemAllocator;

// Device memory allocator
pub struct DeviceAllocator;

// Pinned memory allocator
pub struct PinnedAllocator;

// Disk storage allocator
pub struct DiskAllocator;

// NIXL storage allocator
pub struct NixlAllocator;
```

## Error Types

### BlockPoolError

Errors that can occur during block pool operations.

```rust
pub enum BlockPoolError {
    BlockNotComplete,
    NotEnoughBlocksAvailable(usize, usize),
    InvalidMutableBlock(String),
    FailedToRegisterBlock(String),
    ProgressEngineShutdown,
    BlockError(BlockError),
}
```

### BlockError

Errors that can occur during block operations.

```rust
pub enum BlockError {
    ViewsNotAvailableOnLogicalBlocks,
    InvalidLayerIndex(usize),
    InvalidOuterIndex(usize),
    StorageError(StorageError),
}
```

### StorageError

Errors that can occur during storage operations.

```rust
pub enum StorageError {
    AllocationFailed(String),
    NotAccessible(String),
    InvalidConfig(String),
    OperationFailed(String),
    Cuda(DriverError),
    RegistrationKeyExists(String),
    HandleNotFound(String),
    NixlError(NixlError),
    OutOfBounds(String),
}
```

## View Types

### LayerView

Read-only view of a layer's data.

```rust
pub struct LayerView<S: Storage> {
    storage: S,
    offset: usize,
    size: usize,
}
```

### LayerViewMut

Mutable view of a layer's data.

```rust
pub struct LayerViewMut<S: Storage> {
    storage: S,
    offset: usize,
    size: usize,
}
```

### BlockView

Read-only view of an entire block's data.

```rust
pub struct BlockView<S: Storage> {
    storage: S,
    offset: usize,
    size: usize,
}
```

### BlockViewMut

Mutable view of an entire block's data.

```rust
pub struct BlockViewMut<S: Storage> {
    storage: S,
    offset: usize,
    size: usize,
}
```

## Event System

### EventManager Trait

Trait for handling block-related events.

```rust
pub trait EventManager: Send + Sync + 'static {
    fn block_allocated(&self, block_id: BlockId);
    fn block_registered(&self, block_id: BlockId, sequence_hash: SequenceHash);
    fn block_matched(&self, block_id: BlockId, sequence_hash: SequenceHash);
    fn block_evicted(&self, block_id: BlockId);
}
```

### NullEventManager

Default event manager that does nothing.

```rust
pub struct NullEventManager;

impl EventManager for NullEventManager {
    fn block_allocated(&self, _block_id: BlockId) {}
    fn block_registered(&self, _block_id: BlockId, _sequence_hash: SequenceHash) {}
    fn block_matched(&self, _block_id: BlockId, _sequence_hash: SequenceHash) {}
    fn block_evicted(&self, _block_id: BlockId) {}
}
```

## Metrics

### BlockManagerMetrics

Metrics collection for the block manager.

```rust
pub struct BlockManagerMetrics {
    pool_metrics: Arc<PoolMetrics>,
    allocation_counter: Counter,
    registration_counter: Counter,
    match_counter: Counter,
    eviction_counter: Counter,
}
```

### PoolMetrics

Metrics for a specific block pool.

```rust
pub struct PoolMetrics {
    total_blocks: Gauge,
    available_blocks: Gauge,
    allocation_duration: Histogram,
    registration_duration: Histogram,
    match_duration: Histogram,
}
```

## Utility Types

### SequenceHash

Hash representing a sequence of tokens.

```rust
pub struct SequenceHash([u8; 8]);

impl SequenceHash {
    pub fn new(hash: [u8; 8]) -> Self;
    pub fn as_bytes(&self) -> &[u8; 8];
}
```

### BlockId

Unique identifier for a block.

```rust
pub type BlockId = usize;
```

### WorkerID

Unique identifier for a worker.

```rust
pub type WorkerID = u64;
```

### DType

Data type enumeration.

```rust
pub enum DType {
    FP16,
    FP32,
    BF16,
    INT8,
    INT16,
    INT32,
    INT64,
}
```

## Constants

```rust
// Default values
pub const DEFAULT_PAGE_SIZE: usize = 16;
pub const DEFAULT_INNER_DIM: usize = 4096;
pub const DEFAULT_NUM_LAYERS: usize = 32;
pub const DEFAULT_OUTER_DIM: usize = 2;

// Limits
pub const MAX_PAGE_SIZE: usize = 1024;
pub const MAX_INNER_DIM: usize = 32768;
pub const MAX_NUM_LAYERS: usize = 128;
pub const MAX_OUTER_DIM: usize = 2;
```

## Type Aliases

```rust
// Common type aliases
pub type ReferenceBlockManager = KvBlockManager<BasicMetadata>;
pub type MutableBlocks<S, L, M> = Vec<MutableBlock<S, L, M>>;
pub type ImmutableBlocks<S, L, M> = Vec<ImmutableBlock<S, L, M>>;
pub type BlockPoolResult<T> = Result<T, BlockPoolError>;
pub type StorageResult<T> = Result<T, StorageError>;
```

## Builder Patterns

### KvBlockManagerConfigBuilder

```rust
impl KvBlockManagerConfigBuilder {
    pub fn runtime(mut self, runtime: KvManagerRuntimeConfig) -> Self;
    pub fn model(mut self, model: KvManagerModelConfig) -> Self;
    pub fn device_layout(mut self, layout: KvManagerLayoutConfig<DeviceStorage>) -> Self;
    pub fn host_layout(mut self, layout: KvManagerLayoutConfig<PinnedStorage>) -> Self;
    pub fn disk_layout(mut self, layout: KvManagerLayoutConfig<DiskStorage>) -> Self;
    pub fn event_manager(mut self, manager: Arc<dyn EventManager>) -> Self;
    pub fn build(self) -> Result<KvBlockManagerConfig, String>;
}
```

### KvManagerRuntimeConfigBuilder

```rust
impl KvManagerRuntimeConfigBuilder {
    pub fn worker_id(mut self, id: u64) -> Self;
    pub fn enable_nixl(mut self) -> Self;
    pub fn use_nixl_agent(mut self, agent: NixlAgent) -> Self;
    pub fn disable_nixl(mut self) -> Self;
    pub fn async_runtime(mut self, runtime: Arc<tokio::runtime::Runtime>) -> Self;
    pub fn metrics_registry(mut self, registry: Arc<Registry>) -> Self;
    pub fn build(self) -> KvManagerRuntimeConfig;
}
```

### KvManagerModelConfigBuilder

```rust
impl KvManagerModelConfigBuilder {
    pub fn num_layers(mut self, layers: usize) -> Self;
    pub fn outer_dim(mut self, dim: usize) -> Self;
    pub fn page_size(mut self, size: usize) -> Self;
    pub fn inner_dim(mut self, dim: usize) -> Self;
    pub fn dtype(mut self, dtype: DType) -> Self;
    pub fn build(self) -> KvManagerModelConfig;
}
```

## Next Steps

- [Python API Reference](python.md) - Python bindings reference
- [Configuration Reference](config.md) - Configuration options
- [Examples](examples/basic_usage.md) - Usage examples
- [Best Practices](advanced/best_practices.md) - Best practices guide 