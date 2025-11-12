# TensorRT-LLM Engine Configurations

This directory contains TensorRT-LLM engine configuration files for various model deployments.


## Usage

These YAML configuration files can be passed to TensorRT-LLM workers using the `--extra-engine-args` parameter:

```bash
python3 -m dynamo.trtllm \
    --extra-engine-args "${ENGINE_ARGS}" \
    ...
```

Where `ENGINE_ARGS` points to one of the configuration files in this directory.

## Configuration Types

### Aggregated (agg/)
Single-node configurations that combine prefill and decode operations:
- **simple/**: Basic aggregated setup
- **mtp/**: Multi-token prediction configurations
- **wide_ep/**: Wide expert parallel configurations

### Disaggregated (disagg/)
Separate configurations for prefill and decode workers:
- **simple/**: Basic prefill/decode split
- **mtp/**: Multi-token prediction with separate prefill/decode
- **wide_ep/**: Wide expert parallel with expert load balancer

## Key Configuration Parameters

- **Parallelism**: `tensor_parallel_size`, `moe_expert_parallel_size`, `pipeline_parallel_size`
- **Memory**: `kv_cache_config.free_gpu_memory_fraction`, `kv_cache_config.dtype`
- **Batching**: `max_batch_size`, `max_num_tokens`, `max_seq_len`
- **Scheduling**: `disable_overlap_scheduler`, `cuda_graph_config`

## Notes

- For disaggregated setups, ensure `kv_cache_config.dtype` matches between prefill and decode configs
- WideEP configurations require an expert load balancer config (`eplb.yaml`)
- Adjust `free_gpu_memory_fraction` based on your workload and attention DP settings
