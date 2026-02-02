# Mocker engine

The mocker engine is a mock vLLM implementation designed for testing and development purposes. It simulates realistic token generation timing without requiring actual model inference, making it useful for:

- Testing distributed system components without GPU resources
- Benchmarking infrastructure and networking overhead
- Developing and debugging Dynamo components
- Load testing and performance analysis

## Basic usage

The mocker engine now supports a vLLM-style CLI interface with individual arguments for all configuration options.

### Required arguments:
- `--model-path`: Path to model directory or HuggingFace model ID (required for tokenizer)

### MockEngineArgs parameters (vLLM-style):
- `--num-gpu-blocks-override`: Number of GPU blocks for KV cache (default: 16384)
- `--block-size`: Token block size for KV cache blocks (default: 64)
- `--max-num-seqs`: Maximum number of sequences per iteration (default: 256)
- `--max-num-batched-tokens`: Maximum number of batched tokens per iteration (default: 8192)
- `--enable-prefix-caching` / `--no-enable-prefix-caching`: Enable/disable automatic prefix caching (default: True)
- `--enable-chunked-prefill` / `--no-enable-chunked-prefill`: Enable/disable chunked prefill (default: True)
- `--watermark`: KV cache watermark threshold as a fraction (default: 0.01)
- `--speedup-ratio`: Speed multiplier for token generation (default: 1.0). Higher values make the simulation engines run faster
- `--data-parallel-size`: Number of data parallel workers to simulate (default: 1)
- `--num-workers`: Number of mocker workers to launch in the same process (default: 1). All workers share the same tokio runtime and thread pool
- `--stagger-delay`: Delay in seconds between launching each worker to avoid overwhelming etcd/NATS/frontend. Set to 0 to disable staggering. Use -1 for auto mode (stagger dependent on number of workers). Default: -1 (auto)
- `--is-prefill-worker` / `--is-decode-worker`: Whether the worker is a prefill or decode worker for disaggregated deployment. If not specified, mocker will be in aggregated mode.

### Example with individual arguments (vLLM-style):
```bash
# Start mocker with custom configuration
python -m dynamo.mocker \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-gpu-blocks-override 8192 \
  --block-size 16 \
  --speedup-ratio 10.0 \
  --max-num-seqs 512 \
  --num-workers 4 \
  --enable-prefix-caching

# Start frontend server
python -m dynamo.frontend --http-port 8000
```

> [!Note]
> Each mocker instance runs as a single process, and each DP worker (specified by `--data-parallel-size`) is spawned as a lightweight async task within that process. For benchmarking (e.g., router testing), you can use `--num-workers` to launch multiple mocker engines in the same process, which is more efficient than launching separate processes since they all share the same tokio runtime and thread pool.

## Performance modeling with planner profile data

By default, the mocker uses hardcoded polynomial formulas to estimate prefill and decode timing. For more realistic simulations, you can load performance data from actual profiling results using `--planner-profile-data`:

```bash
python -m dynamo.mocker \
  --model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
  --planner-profile-data tests/planner/profiling_results/H200_TP1P_TP1D \
  --speedup-ratio 1.0
```

The profile results directory should contain `selected_prefill_interpolation/` and `selected_decode_interpolation/` subdirectories with `raw_data.npz` files. This works seamlessly in Kubernetes where profile data is mounted via ConfigMap or PersistentVolume.

To generate profiling data for your own model/hardware configuration, run the profiler (see [SLA-driven profiling documentation](../../../../docs/benchmarks/sla_driven_profiling.md) for details):

```bash
python benchmarks/profiler/profile_sla.py \
  --profile-config your_profile_config.yaml
```

Then use the resulting profile results directory directly with `--planner-profile-data`.

## Deploying Mocker in K8s

We provide the example DGD yaml configurations for aggregated and disaggregated deployment in `examples/backends/mocker/deploy/`. You can deploy the mocker engine in K8s by running:

```bash
kubectl apply -f examples/backends/mocker/deploy/agg.yaml # or, for disaggregated
kubectl apply -f examples/backends/mocker/deploy/disagg.yaml
```
