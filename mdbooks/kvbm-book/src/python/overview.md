# Python API Overview

The KV Block Manager provides comprehensive Python bindings that enable seamless integration with Python-based machine learning workflows. The Python API offers the same functionality as the Rust implementation while providing a more familiar interface for Python developers.

## Key Features

- **Native Python Interface**: Clean, Pythonic API design
- **DLPack Integration**: Direct tensor interoperability with PyTorch, NumPy, and other frameworks
- **vLLM Integration**: Production-ready integration with vLLM inference serving
- **Memory Safety**: Automatic memory management with Python's garbage collection
- **Async Support**: Full support for async/await patterns
- **Type Hints**: Comprehensive type annotations for better IDE support

## Installation

The Python bindings are included with the main Dynamo package:

```bash
pip install dynamo-llm
```

## Basic Usage

### Creating a Block Manager

```python
import dynamo_llm

# Create a block manager with basic configuration
block_manager = dynamo_llm.BlockManager(
    num_layers=32,
    page_size=16,
    inner_dim=4096
)

# Create with advanced configuration
block_manager = dynamo_llm.BlockManager(
    num_layers=32,
    page_size=16,
    inner_dim=4096,
    dtype=dynamo_llm.DType.FP16,
    device_id=0
)
```

### Allocating Blocks

```python
# Allocate blocks for inference
blocks = block_manager.allocate_blocks(4)

# Blocks are returned as a list of Block objects
for block in blocks:
    print(f"Block has {len(block)} layers")
    
    # Access individual layers
    for layer_idx in range(len(block)):
        layer = block[layer_idx]
        print(f"Layer {layer_idx} shape: {layer.shape}")
```

### Working with Block Data

```python
# Get a block and access its data
block = blocks[0]
layer = block[0]  # Get first layer

# Convert to PyTorch tensor
import torch
tensor = torch.from_dlpack(layer.__dlpack__())
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor dtype: {tensor.dtype}")

# Convert to NumPy array
import numpy as np
array = np.from_dlpack(layer.__dlpack__())
print(f"Array shape: {array.shape}")
```

## Core Classes

### BlockManager

The main entry point for block management operations.

```python
class BlockManager:
    def __init__(
        self,
        num_layers: int,
        page_size: int,
        inner_dim: int,
        dtype: DType = DType.FP16,
        device_id: int = 0
    ):
        ...
    
    def allocate_blocks(self, count: int) -> List[Block]:
        """Allocate new blocks for inference."""
        ...
    
    def get_device_pool(self) -> Optional[BlockPool]:
        """Get the device (GPU) block pool."""
        ...
    
    def get_host_pool(self) -> Optional[BlockPool]:
        """Get the host (CPU) block pool."""
        ...
```

### Block

Represents a KV cache block containing multiple layers.

```python
class Block:
    def __len__(self) -> int:
        """Return the number of layers in the block."""
        ...
    
    def __getitem__(self, index: int) -> Layer:
        """Get a layer by index."""
        ...
    
    def __iter__(self) -> Iterator[Layer]:
        """Iterate over layers."""
        ...
    
    def to_list(self) -> List[Layer]:
        """Convert block to a list of layers."""
        ...
    
    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
        """Export block data as DLPack tensor."""
        ...
```

### Layer

Represents a single layer within a block.

```python
class Layer:
    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
        """Export layer data as DLPack tensor."""
        ...
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the layer data."""
        ...
    
    @property
    def dtype(self) -> DType:
        """Get the data type of the layer."""
        ...
```

## Advanced Usage

### Block Pool Management

```python
# Access different storage pools
device_pool = block_manager.get_device_pool()
host_pool = block_manager.get_host_pool()

if device_pool:
    print(f"Device blocks: {device_pool.total_blocks()}")
    print(f"Available: {device_pool.available_blocks()}")
    
    # Allocate from specific pool
    device_blocks = device_pool.allocate_blocks(2)
```

### Block Registration and Matching

```python
# Register blocks after computation
immutable_blocks = device_pool.register_blocks(device_blocks)

# Get sequence hashes for matching
sequence_hashes = [block.sequence_hash() for block in immutable_blocks]

# Match blocks by sequence hash
matched_blocks = device_pool.match_sequence_hashes(sequence_hashes)

if len(matched_blocks) == len(sequence_hashes):
    print("All blocks found in cache!")
else:
    print("Some blocks need to be computed")
```

### Memory Management

```python
# Check memory usage
if device_pool:
    usage = device_pool.usage()
    print(f"Device memory usage: {usage:.2%}")

# Monitor block allocation
total_blocks = device_pool.total_blocks()
available_blocks = device_pool.available_blocks()
used_blocks = total_blocks - available_blocks
print(f"Used blocks: {used_blocks}/{total_blocks}")
```

## DLPack Integration

The Python bindings provide full DLPack support for seamless tensor interoperability:

### PyTorch Integration

```python
import torch
import dynamo_llm

# Create block manager
block_manager = dynamo_llm.BlockManager(
    num_layers=32,
    page_size=16,
    inner_dim=4096
)

# Allocate blocks
blocks = block_manager.allocate_blocks(1)
block = blocks[0]

# Convert to PyTorch tensors
for layer_idx in range(len(block)):
    layer = block[layer_idx]
    
    # Convert to PyTorch tensor
    tensor = torch.from_dlpack(layer.__dlpack__())
    
    # Perform operations
    result = torch.softmax(tensor, dim=-1)
    
    # Convert back if needed
    # Note: This would require additional implementation
```

### NumPy Integration

```python
import numpy as np
import dynamo_llm

# Get layer data as NumPy array
layer = block[0]
array = np.from_dlpack(layer.__dlpack__())

# Perform NumPy operations
mean = np.mean(array)
std = np.std(array)
normalized = (array - mean) / std
```

### CuPy Integration (GPU Arrays)

```python
import cupy as cp
import dynamo_llm

# Convert to CuPy array for GPU operations
layer = block[0]
gpu_array = cp.from_dlpack(layer.__dlpack__())

# Perform GPU operations
result = cp.linalg.norm(gpu_array, axis=-1)
```

## vLLM Integration

The KV Block Manager provides direct integration with vLLM:

```python
import dynamo_llm
from dynamo_llm.vllm import KvbmCacheManager

# Create block manager
block_manager = dynamo_llm.BlockManager(
    num_layers=32,
    page_size=16,
    inner_dim=4096
)

# Create vLLM cache manager
cache_manager = KvbmCacheManager(block_manager)

# Create a request
request = KvbmRequest(
    request_id="req_001",
    salt_hash=12345
)

# Create slot for request
tokens = [1, 2, 3, 4, 5]  # Token IDs
sequence_hashes = cache_manager.create_slot(request, tokens)

# Get computed blocks
computed_blocks = cache_manager.get_computed_blocks(sequence_hashes)

# Update slot with new tokens
update = SlotUpdate(
    request_id="req_001",
    request_num_tokens=10,
    request_num_computed_tokens=5,
    tokens_to_append=[6, 7, 8, 9, 10],
    num_new_tokens=5
)

new_blocks = cache_manager.alloctate_slots(update)

# Clean up
cache_manager.free("req_001")
```

## Error Handling

The Python API provides comprehensive error handling:

```python
import dynamo_llm

try:
    # Create block manager
    block_manager = dynamo_llm.BlockManager(
        num_layers=32,
        page_size=16,
        inner_dim=4096
    )
    
    # Allocate blocks
    blocks = block_manager.allocate_blocks(1000)  # May fail if not enough memory
    
except dynamo_llm.BlockManagerError as e:
    print(f"Block manager error: {e}")
    
except dynamo_llm.StorageError as e:
    print(f"Storage error: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Memory Efficiency

```python
# Monitor memory usage
def monitor_memory(block_manager):
    device_pool = block_manager.get_device_pool()
    if device_pool:
        usage = device_pool.usage()
        print(f"Memory usage: {usage:.2%}")
        
        if usage > 0.9:  # 90% usage
            print("Warning: High memory usage!")

# Use context managers for automatic cleanup
def process_blocks(block_manager, count):
    with block_manager.allocate_blocks(count) as blocks:
        # Process blocks
        for block in blocks:
            # ... processing ...
            pass
    # Blocks automatically returned to pool
```

### Batch Processing

```python
# Process multiple blocks efficiently
def process_batch(block_manager, batch_size):
    blocks = block_manager.allocate_blocks(batch_size)
    
    try:
        # Process all blocks
        results = []
        for block in blocks:
            result = process_block(block)
            results.append(result)
        
        return results
    
    finally:
        # Ensure blocks are returned to pool
        for block in blocks:
            del block
```

## Type Hints and IDE Support

The Python API includes comprehensive type hints:

```python
from typing import List, Optional, Tuple
import dynamo_llm

def process_layers(block: dynamo_llm.Block) -> List[float]:
    """Process all layers in a block and return results."""
    results: List[float] = []
    
    for layer in block:
        # Type hints help with IDE autocomplete
        tensor = torch.from_dlpack(layer.__dlpack__())
        result = tensor.mean().item()
        results.append(result)
    
    return results

def allocate_with_fallback(
    block_manager: dynamo_llm.BlockManager,
    count: int
) -> Optional[List[dynamo_llm.Block]]:
    """Allocate blocks with fallback to host memory."""
    try:
        return block_manager.allocate_blocks(count)
    except dynamo_llm.BlockManagerError:
        # Fallback to host memory
        host_pool = block_manager.get_host_pool()
        if host_pool:
            return host_pool.allocate_blocks(count)
        return None
```

## Next Steps

- [Block Interface](block.md) - Learn about Block and Layer classes
- [DLPack Integration](dlpack.md) - Understand tensor interoperability
- [vLLM Integration](vllm.md) - Production deployment with vLLM
- [Examples](examples/basic_usage.md) - Practical usage examples 