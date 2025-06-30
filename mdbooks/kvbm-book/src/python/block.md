# Block and Layer Interfaces

The Python bindings provide `Block` and `Layer` classes that represent KV cache blocks and their individual layers. These classes offer a Pythonic interface for accessing and manipulating block data.

## Block Class

The `Block` class represents a KV cache block containing multiple layers. It provides sequence-like access to layers and supports DLPack for tensor interoperability.

### Basic Usage

```python
import dynamo_llm

# Create block manager and allocate blocks
block_manager = dynamo_llm.BlockManager(
    num_layers=32,
    page_size=16,
    inner_dim=4096
)

blocks = block_manager.allocate_blocks(1)
block = blocks[0]

# Get block information
print(f"Block has {len(block)} layers")
print(f"Block device ID: {block.device_id}")
print(f"Block dtype: {block.dtype}")
```

### Layer Access

```python
# Access layers by index
layer_0 = block[0]
layer_1 = block[1]
layer_2 = block[2]

# Iterate over all layers
for layer_idx, layer in enumerate(block):
    print(f"Layer {layer_idx}: {layer.shape}")

# Convert to list of layers
layers = block.to_list()
print(f"Converted to list with {len(layers)} layers")
```

### DLPack Integration

```python
import torch
import numpy as np

# Convert entire block to PyTorch tensor
block_tensor = torch.from_dlpack(block.__dlpack__())
print(f"Block tensor shape: {block_tensor.shape}")

# Convert to NumPy array
block_array = np.from_dlpack(block.__dlpack__())
print(f"Block array shape: {block_array.shape}")

# Get device information
device_info = block.__dlpack_device__()
print(f"Block device: {device_info}")
```

### Block Properties

```python
# Get block metadata
print(f"Number of layers: {len(block)}")
print(f"Device ID: {block.device_id}")
print(f"Data type: {block.dtype}")

# Check if block is valid
if block:
    print("Block is valid")
else:
    print("Block is invalid")
```

## Layer Class

The `Layer` class represents a single layer within a block. It provides access to the layer's data and supports DLPack for tensor operations.

### Basic Usage

```python
# Get a layer from a block
layer = block[0]

# Get layer information
print(f"Layer shape: {layer.shape}")
print(f"Layer dtype: {layer.dtype}")
print(f"Layer device ID: {layer.device_id}")
```

### Data Access

```python
import torch

# Convert layer to PyTorch tensor
tensor = torch.from_dlpack(layer.__dlpack__())
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor dtype: {tensor.dtype}")

# Perform tensor operations
mean = tensor.mean()
std = tensor.std()
max_val = tensor.max()
min_val = tensor.min()

print(f"Layer statistics: mean={mean:.4f}, std={std:.4f}")
print(f"Value range: [{min_val:.4f}, {max_val:.4f}]")
```

### NumPy Integration

```python
import numpy as np

# Convert layer to NumPy array
array = np.from_dlpack(layer.__dlpack__())
print(f"Array shape: {array.shape}")
print(f"Array dtype: {array.dtype}")

# Perform NumPy operations
mean = np.mean(array)
std = np.std(array)
percentiles = np.percentile(array, [25, 50, 75])

print(f"NumPy statistics: mean={mean:.4f}, std={std:.4f}")
print(f"Percentiles: 25%={percentiles[0]:.4f}, 50%={percentiles[1]:.4f}, 75%={percentiles[2]:.4f}")
```

### CuPy Integration (GPU Arrays)

```python
import cupy as cp

# Convert layer to CuPy array for GPU operations
gpu_array = cp.from_dlpack(layer.__dlpack__())
print(f"GPU array shape: {gpu_array.shape}")

# Perform GPU operations
norm = cp.linalg.norm(gpu_array)
eigenvals = cp.linalg.eigvals(gpu_array)

print(f"GPU norm: {norm:.4f}")
print(f"Eigenvalues shape: {eigenvals.shape}")
```

## Advanced Usage

### Batch Processing

```python
import torch
import dynamo_llm

def process_block_layers(block):
    """Process all layers in a block."""
    results = []
    
    for layer_idx in range(len(block)):
        layer = block[layer_idx]
        tensor = torch.from_dlpack(layer.__dlpack__())
        
        # Process layer data
        result = torch.softmax(tensor, dim=-1)
        results.append(result)
    
    return results

# Process multiple blocks
blocks = block_manager.allocate_blocks(4)
all_results = []

for block in blocks:
    block_results = process_block_layers(block)
    all_results.extend(block_results)

print(f"Processed {len(all_results)} layers total")
```

### Memory-Efficient Processing

```python
def memory_efficient_layer_processing(block):
    """Process layers with minimal memory usage."""
    for layer_idx in range(len(block)):
        layer = block[layer_idx]
        
        # Process layer in chunks to avoid memory issues
        tensor = torch.from_dlpack(layer.__dlpack__())
        
        # Process in smaller chunks if needed
        chunk_size = 1000
        for i in range(0, tensor.shape[0], chunk_size):
            chunk = tensor[i:i+chunk_size]
            # Process chunk
            result = torch.softmax(chunk, dim=-1)
            # Use result...
        
        # Explicitly delete tensor to free memory
        del tensor
```

### Custom Layer Processing

```python
class LayerProcessor:
    def __init__(self, model_config):
        self.model_config = model_config
    
    def process_layer(self, layer, layer_idx):
        """Process a single layer with custom logic."""
        tensor = torch.from_dlpack(layer.__dlpack__())
        
        # Apply layer-specific processing
        if layer_idx < self.model_config.num_layers // 2:
            # Early layers: apply attention
            result = self.apply_attention(tensor)
        else:
            # Later layers: apply feedforward
            result = self.apply_feedforward(tensor)
        
        return result
    
    def apply_attention(self, tensor):
        # Custom attention logic
        return torch.softmax(tensor, dim=-1)
    
    def apply_feedforward(self, tensor):
        # Custom feedforward logic
        return torch.relu(tensor)

# Use custom processor
processor = LayerProcessor(model_config)
blocks = block_manager.allocate_blocks(2)

for block in blocks:
    for layer_idx in range(len(block)):
        layer = block[layer_idx]
        result = processor.process_layer(layer, layer_idx)
        # Use result...
```

## Error Handling

### Block Access Errors

```python
try:
    # Try to access non-existent layer
    layer = block[100]  # Index out of range
except IndexError as e:
    print(f"Layer access error: {e}")

try:
    # Try to access invalid block
    invalid_block = None
    layer = invalid_block[0]
except AttributeError as e:
    print(f"Block access error: {e}")
```

### DLPack Errors

```python
try:
    # Try to convert layer to tensor
    tensor = torch.from_dlpack(layer.__dlpack__())
except Exception as e:
    print(f"DLPack conversion error: {e}")
    # Fallback to other methods
```

### Memory Errors

```python
try:
    # Try to process large layer
    tensor = torch.from_dlpack(layer.__dlpack__())
    result = torch.softmax(tensor, dim=-1)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU memory exhausted, processing in chunks")
        # Implement chunked processing
    else:
        raise
```

## Performance Considerations

### Efficient Iteration

```python
# Efficient iteration over layers
for layer in block:
    # Process layer
    tensor = torch.from_dlpack(layer.__dlpack__())
    # ... processing ...

# Less efficient: accessing by index
for i in range(len(block)):
    layer = block[i]  # Additional overhead
    tensor = torch.from_dlpack(layer.__dlpack__())
    # ... processing ...
```

### Memory Management

```python
def process_layers_with_cleanup(block):
    """Process layers with explicit memory cleanup."""
    results = []
    
    for layer in block:
        # Convert to tensor
        tensor = torch.from_dlpack(layer.__dlpack__())
        
        # Process tensor
        result = torch.softmax(tensor, dim=-1)
        results.append(result)
        
        # Explicitly delete tensor to free memory
        del tensor
    
    return results
```

### Batch Processing

```python
def batch_process_layers(blocks, batch_size=4):
    """Process multiple layers in batches."""
    all_layers = []
    
    # Collect all layers
    for block in blocks:
        for layer in block:
            all_layers.append(layer)
    
    # Process in batches
    results = []
    for i in range(0, len(all_layers), batch_size):
        batch = all_layers[i:i+batch_size]
        
        # Convert batch to tensors
        tensors = [torch.from_dlpack(layer.__dlpack__()) for layer in batch]
        
        # Process batch
        batch_results = torch.stack(tensors)
        processed = torch.softmax(batch_results, dim=-1)
        
        results.extend(processed)
    
    return results
```

## Type Hints

The Python API includes comprehensive type hints for better IDE support:

```python
from typing import List, Optional, Tuple
import dynamo_llm

def process_block(block: dynamo_llm.Block) -> List[torch.Tensor]:
    """Process a block and return list of processed tensors."""
    results: List[torch.Tensor] = []
    
    for layer in block:
        tensor = torch.from_dlpack(layer.__dlpack__())
        result = torch.softmax(tensor, dim=-1)
        results.append(result)
    
    return results

def get_layer_info(layer: dynamo_llm.Layer) -> Tuple[Tuple[int, ...], dynamo_llm.DType]:
    """Get layer shape and data type."""
    return layer.shape, layer.dtype
```

## Next Steps

- [DLPack Integration](dlpack.md) - Learn about tensor interoperability
- [vLLM Integration](vllm.md) - Production deployment with vLLM
- [Block Lists](block_list.md) - Manage collections of blocks
- [Examples](examples/basic_usage.md) - Practical usage examples 