# Speculative Decoding Support in vLLM Backend

This document describes the speculative decoding support added to the Dynamo vLLM backend, which provides compatibility with TRT-LLM configuration formats while leveraging vLLM's native speculative decoding capabilities.

## Overview

Speculative decoding is a technique that improves inference latency by using a smaller "draft" model to propose tokens that are then verified by the main model in parallel. This implementation adds support for:

- **Eagle Speculative Decoding**: Uses a dedicated draft model for token proposals
- **MTP (Multi-Token Prediction)**: Maps to vLLM's n-gram lookup for similar functionality
- **YAML Configuration**: Compatible with TRT-LLM configuration format

## Configuration

### Using YAML Configuration Files

You can now use YAML configuration files similar to TRT-LLM format:

```bash
python -m dynamo.vllm \
    --model meta-llama/Llama-2-7b-hf \
    --extra-engine-args components/backends/vllm/engine_configs/eagle_decode.yaml
```

### Example YAML Configurations

#### Eagle Speculative Decoding

```yaml
# eagle_decode.yaml
tensor_parallel_size: 4
max_batch_size: 256
max_num_tokens: 1024
max_seq_len: 8704
gpu_memory_utilization: 0.5

# Enable Speculative Decoding
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model_dir: nvidia/Llama-4-Maverick-17B-128E-Eagle3
  eagle3_one_model: true

kv_cache_config:
  free_gpu_memory_fraction: 0.5
```

#### MTP (Multi-Token Prediction)

```yaml
# mtp_decode.yaml  
tensor_parallel_size: 4
max_batch_size: 256
max_num_tokens: 512
max_seq_len: 8704

# MTP maps to n-gram lookup in vLLM
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 1

kv_cache_config:
  free_gpu_memory_fraction: 0.85
```

## Configuration Mapping

### TRT-LLM to vLLM Parameter Mapping

| TRT-LLM Parameter | vLLM Parameter | Description |
|------------------|----------------|-------------|
| `max_batch_size` | `max_num_seqs` | Maximum number of sequences |
| `max_num_tokens` | `max_num_batched_tokens` | Maximum batched tokens |
| `max_seq_len` | `max_model_len` | Maximum sequence length |
| `tensor_parallel_size` | `tensor_parallel_size` | Tensor parallelism |
| `pipeline_parallel_size` | `pipeline_parallel_size` | Pipeline parallelism |

### Speculative Config Mapping

#### Eagle Configuration

| TRT-LLM | vLLM | Description |
|---------|------|-------------|
| `decoding_type: Eagle` | `speculative_model: <model_path>` | Uses draft model |
| `max_draft_len: N` | `num_speculative_tokens: N` | Number of speculative tokens |
| `speculative_model_dir: <path>` | `speculative_model: <path>` | Draft model path |

#### MTP Configuration

| TRT-LLM | vLLM | Description |
|---------|------|-------------|
| `decoding_type: MTP` | `speculative_model: "[ngram]"` | Uses n-gram lookup |
| `num_nextn_predict_layers: N` | `num_speculative_tokens: N*2` | Approximate mapping |

## Usage Examples

### Basic Usage

```bash
# Using Eagle speculative decoding
python -m dynamo.vllm \
    --model meta-llama/Llama-2-7b-hf \
    --extra-engine-args engine_configs/eagle_decode.yaml

# Using MTP (mapped to n-gram)
python -m dynamo.vllm \
    --model meta-llama/Llama-2-7b-hf \
    --extra-engine-args engine_configs/mtp_decode.yaml
```

### Disaggregated Serving

```bash
# Prefill worker with speculative decoding
python -m dynamo.vllm \
    --model meta-llama/Llama-2-7b-hf \
    --is-prefill-worker \
    --extra-engine-args engine_configs/eagle_prefill.yaml

# Decode worker with speculative decoding  
python -m dynamo.vllm \
    --model meta-llama/Llama-2-7b-hf \
    --extra-engine-args engine_configs/eagle_decode.yaml
```

## Implementation Details

### YAML Processing

The implementation adds a new `--extra-engine-args` parameter that accepts a YAML file path. The YAML file is processed by:

1. `update_vllm_args_with_extra_options()`: Loads YAML and applies configurations
2. `convert_trtllm_speculative_config_to_vllm()`: Converts TRT-LLM format to vLLM format

### Compatibility Notes

- **Eagle**: Maps directly to vLLM's draft model speculative decoding
- **MTP**: No direct equivalent in vLLM, maps to n-gram lookup decoding
- **Configuration**: Maintains TRT-LLM YAML structure for consistency

### Limitations

1. **MTP Mapping**: MTP doesn't have a direct vLLM equivalent, so it maps to n-gram lookup
2. **Model Compatibility**: Draft models must be compatible with the main model's vocabulary
3. **Performance**: Effectiveness depends on the quality of the draft model or n-gram patterns

## Testing

Run the test suite to verify the implementation:

```bash
python test_vllm_speculative.py
```

This tests:
- Eagle configuration conversion
- MTP configuration conversion  
- YAML file loading and parameter mapping

## Performance Tips

1. **Draft Model Selection**: Choose draft models that are small but accurate for your use case
2. **Token Count**: Start with 3-5 speculative tokens and adjust based on acceptance rate
3. **Memory Usage**: Speculative decoding requires additional memory for the draft model
4. **Monitoring**: Use vLLM's metrics to monitor acceptance rates and adjust accordingly

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're using the local Dynamo installation, not pip version
2. **Model Compatibility**: Verify draft and main models use the same vocabulary
3. **Memory Issues**: Reduce `gpu_memory_utilization` if running out of memory
4. **Performance**: Monitor acceptance rates; low rates may indicate poor draft model choice

### Debugging

Enable debug logging to see configuration processing:

```bash
export PYTHONPATH=/path/to/dynamo/components/backends/vllm/src:$PYTHONPATH
python -m dynamo.vllm --extra-engine-args config.yaml --log-level DEBUG
```

## Migration from TRT-LLM

To migrate from TRT-LLM to vLLM with speculative decoding:

1. **Copy Configuration**: Use your existing TRT-LLM YAML files as-is
2. **Update Model Paths**: Ensure draft model paths are accessible to vLLM
3. **Test Performance**: Compare latency and throughput with your workload
4. **Adjust Parameters**: Fine-tune based on vLLM-specific behavior

The goal is to provide a seamless migration path while maintaining the low latency benefits of speculative decoding. 