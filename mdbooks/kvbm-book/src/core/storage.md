# Storage System

The KV Block Manager provides a unified storage abstraction that supports multiple storage backends including GPU memory, CPU memory, local NVMe storage, and remote storage. This document covers the storage architecture and how to use different storage types.

## Storage Architecture

The storage system is built around several key abstractions:

```rust
pub trait Storage: Debug + Send + Sync + 'static {
    fn storage_type(&self) -> StorageType;
    fn addr(&self) -> u64;
    fn size(&self) -> usize;
    unsafe fn as_ptr(&self) -> *const u8;
    unsafe fn as_mut_ptr(&mut self) -> *mut u8;
}
```

## Storage Types

The system supports multiple storage types through the `StorageType` enum:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum StorageType {
    /// System memory
    System,
    
    /// CUDA device memory
    Device(u32),
    
    /// CUDA page-locked host memory
    Pinned,
    
    /// Disk memory
    Disk(u64),
    
    /// Remote memory accessible through NIXL
    Nixl,
    
    /// Null storage (for testing)
    Null,
}
```

## Storage Backends

### 1. System Storage

System memory storage for CPU-based operations:

```rust
use dynamo_llm::block_manager::storage::{SystemStorage, SystemAllocator};

// Create system storage
let storage = SystemStorage::new(1024 * 1024)?; // 1MB

// Use system allocator
let allocator = SystemAllocator;
let storage = allocator.allocate(1024 * 1024)?;
```

### 2. Device Storage (GPU Memory)

CUDA device memory for GPU operations:

```rust
use dynamo_llm::block_manager::storage::{DeviceStorage, DeviceAllocator};

// Create device storage
let device_id = 0;
let storage = DeviceStorage::new(1024 * 1024, device_id)?; // 1MB on GPU 0

// Use device allocator
let allocator = DeviceAllocator::default();
let storage = allocator.allocate(1024 * 1024)?;
```

### 3. Pinned Storage (Page-Locked Memory)

Page-locked host memory for efficient CPU-GPU transfers:

```rust
use dynamo_llm::block_manager::storage::{PinnedStorage, PinnedAllocator};

// Create pinned storage
let storage = PinnedStorage::new(1024 * 1024)?; // 1MB pinned memory

// Use pinned allocator
let allocator = PinnedAllocator::default();
let storage = allocator.allocate(1024 * 1024)?;
```

### 4. Disk Storage (NVMe)

Local NVMe storage for large capacity:

```rust
use dynamo_llm::block_manager::storage::{DiskStorage, DiskAllocator};

// Create disk storage
let path = "/mnt/nvme/kv_cache";
let storage = DiskStorage::new(path, 1024 * 1024 * 1024)?; // 1GB

// Use disk allocator
let allocator = DiskAllocator::new(path);
let storage = allocator.allocate(1024 * 1024)?;
```

### 5. NIXL Storage (Remote)

Remote storage accessible through NIXL protocol:

```rust
use dynamo_llm::block_manager::storage::nixl::{NixlStorage, NixlAllocator};
use nixl_sys::Agent as NixlAgent;

// Create NIXL agent
let agent = NixlAgent::new("remote_storage")?;

// Create NIXL storage
let storage = NixlStorage::new(&agent, 1024 * 1024)?;

// Use NIXL allocator
let allocator = NixlAllocator::new(agent);
let storage = allocator.allocate(1024 * 1024)?;
```

## Storage Allocators

The system provides allocators for each storage type:

```rust
pub trait StorageAllocator<S: Storage>: Send + Sync {
    fn allocate(&self, size: usize) -> Result<S, StorageError>;
}
```

### Allocator Examples

```rust
use dynamo_llm::block_manager::storage::{
    SystemAllocator, DeviceAllocator, PinnedAllocator, DiskAllocator
};

// System memory allocator
let system_allocator = SystemAllocator;
let system_storage = system_allocator.allocate(1024 * 1024)?;

// Device memory allocator
let device_allocator = DeviceAllocator::default();
let device_storage = device_allocator.allocate(1024 * 1024)?;

// Pinned memory allocator
let pinned_allocator = PinnedAllocator::default();
let pinned_storage = pinned_allocator.allocate(1024 * 1024)?;

// Disk allocator
let disk_allocator = DiskAllocator::new("/mnt/nvme/kv_cache");
let disk_storage = disk_allocator.allocate(1024 * 1024)?;
```

## Memory Operations

### Basic Memory Operations

```rust
use dynamo_llm::block_manager::storage::StorageMemset;

// Set memory to zero
storage.memset(0, 0, storage.size())?;

// Set specific region
storage.memset(255, 1000, 100)?; // Set 100 bytes starting at offset 1000 to 255
```

### Memory Views

```rust
// Get read-only view
let ptr = unsafe { storage.as_ptr() };
let slice = unsafe { std::slice::from_raw_parts(ptr, storage.size()) };

// Get mutable view
let mut_ptr = unsafe { storage.as_mut_ptr() };
let mut_slice = unsafe { std::slice::from_raw_parts_mut(mut_ptr, storage.size()) };
```

## Storage Registration

Some storage types can be registered with external libraries:

```rust
pub trait RegisterableStorage: Storage + Send + Sync + 'static {
    fn register(&mut self, key: &str, handle: Box<dyn RegistationHandle>) -> Result<(), StorageError>;
    fn is_registered(&self, key: &str) -> bool;
    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle>;
}
```

### NIXL Registration

```rust
use dynamo_llm::block_manager::storage::nixl::NixlRegisterableStorage;
use nixl_sys::Agent as NixlAgent;

// Create NIXL agent
let agent = NixlAgent::new("my_agent")?;

// Create storage
let mut storage = PinnedStorage::new(1024 * 1024)?;

// Register with NIXL
storage.nixl_register(&agent, None)?;

// Check registration
if storage.is_registered("nixl") {
    println!("Storage is registered with NIXL");
}
```

## Storage Configuration

### Multi-tier Storage Setup

```rust
use dynamo_llm::block_manager::{
    KvBlockManagerConfig, KvManagerLayoutConfig, LayoutType
};
use dynamo_llm::block_manager::storage::{
    DeviceAllocator, PinnedAllocator, DiskAllocator
};

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
```

### Memory Sizing

```rust
fn calculate_storage_requirements(
    num_layers: usize,
    outer_dim: usize,
    page_size: usize,
    inner_dim: usize,
    dtype_size: usize,
    num_blocks: usize
) -> usize {
    let block_size = num_layers * outer_dim * page_size * inner_dim * dtype_size;
    block_size * num_blocks
}

// Example calculation
let block_size = calculate_storage_requirements(32, 2, 16, 4096, 2, 1000);
println!("Total storage required: {} bytes ({:.2} GB)", 
         block_size, block_size as f64 / (1024.0 * 1024.0 * 1024.0));
```

## Error Handling

The storage system provides comprehensive error handling:

```rust
use dynamo_llm::block_manager::storage::StorageError;

match storage.allocate(1024 * 1024) {
    Ok(storage) => {
        println!("Storage allocated successfully");
    }
    Err(StorageError::AllocationFailed(msg)) => {
        println!("Allocation failed: {}", msg);
    }
    Err(StorageError::NotAccessible(msg)) => {
        println!("Storage not accessible: {}", msg);
    }
    Err(StorageError::InvalidConfig(msg)) => {
        println!("Invalid configuration: {}", msg);
    }
    Err(e) => {
        println!("Unexpected error: {}", e);
    }
}
```

## Performance Considerations

### Memory Access Patterns

```rust
// Optimize for sequential access
let storage = SystemStorage::new(1024 * 1024)?;
let ptr = unsafe { storage.as_ptr() };

// Sequential read
for i in 0..storage.size() {
    let byte = unsafe { *ptr.add(i) };
    // Process byte
}

// Random access (slower)
let indices = vec![100, 500, 1000, 2000];
for &index in &indices {
    let byte = unsafe { *ptr.add(index) };
    // Process byte
}
```

### Transfer Optimization

```rust
// Use pinned memory for CPU-GPU transfers
let pinned_storage = PinnedAllocator::default().allocate(1024 * 1024)?;

// Efficient transfer to GPU
let device_storage = DeviceAllocator::default().allocate(1024 * 1024)?;

// Transfer data (implementation depends on CUDA bindings)
transfer_to_device(&pinned_storage, &device_storage)?;
```

### Storage Tiering

```rust
// Implement storage tiering based on access patterns
fn get_optimal_storage(access_frequency: f64, data_size: usize) -> Box<dyn StorageAllocator> {
    match (access_frequency, data_size) {
        (freq, _) if freq > 0.8 => {
            // High frequency access -> GPU memory
            Box::new(DeviceAllocator::default())
        }
        (freq, size) if freq > 0.3 && size < 100 * 1024 * 1024 => {
            // Medium frequency, small size -> Pinned memory
            Box::new(PinnedAllocator::default())
        }
        _ => {
            // Low frequency or large size -> Disk storage
            Box::new(DiskAllocator::new("/mnt/nvme/kv_cache"))
        }
    }
}
```

## Testing and Debugging

### Null Storage for Testing

```rust
use dynamo_llm::block_manager::storage::tests::{NullDeviceStorage, NullDeviceAllocator};

// Create null storage for testing
let null_storage = NullDeviceStorage::new(1024);
let null_allocator = NullDeviceAllocator;

// Test allocation
let test_storage = null_allocator.allocate(1024)?;
assert_eq!(test_storage.storage_type(), StorageType::Null);
```

### Storage Validation

```rust
fn validate_storage<S: Storage>(storage: &S) -> Result<(), StorageError> {
    // Check size
    if storage.size() == 0 {
        return Err(StorageError::InvalidConfig("Storage size cannot be zero".to_string()));
    }
    
    // Check address
    if storage.addr() == 0 {
        return Err(StorageError::InvalidConfig("Storage address cannot be null".to_string()));
    }
    
    // Check pointer validity
    let ptr = unsafe { storage.as_ptr() };
    if ptr.is_null() {
        return Err(StorageError::InvalidConfig("Storage pointer is null".to_string()));
    }
    
    Ok(())
}
```

## Next Steps

- [Block Pool](block_pool.md) - Understand how storage is used in block management
- [Layout Management](layout.md) - Learn about data layout strategies
- [Offloading](offload.md) - Explore block movement between storage tiers
- [Python Storage](python/overview.md) - Use storage from Python 