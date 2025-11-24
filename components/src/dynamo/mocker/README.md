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

By default, the mocker uses hardcoded polynomial formulas to estimate prefill and decode timing. For more realistic simulations, you can load performance data from actual profiling results.

### Using profiled performance data

Add the `--planner-profile-data` flag to load an NPZ file containing interpolation grids from the planner profiler:

```bash
python -m dynamo.mocker \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --planner-profile-data /path/to/profiling_results/perf_data.npz \
  --speedup-ratio 1.0
```

The NPZ file should contain the following arrays:
- `prefill_isl`: 1D array of input sequence lengths
- `prefill_ttft_ms`: 1D array of time-to-first-token values (ms)
- `decode_active_kv_tokens`: 1D array of active KV token counts
- `decode_context_length`: 1D array of context lengths
- `decode_itl`: 2D array of inter-token latencies (ms)

### Generating performance data from profiler results

#### Option 1: Use existing pre-swept results

The repository includes pre-swept profiling results for common models and hardware configurations. For example, to use Llama-3.1-8B-Instruct-FP8 on H200 SXM:

```bash
# Convert existing pre-swept results to mocker-compatible NPZ format
python components/src/dynamo/mocker/utils/planner_profiler_perf_data_converter.py \
  --profile_results_dir tests/planner/profiling_results/H200_TP1P_TP1D \
  --output_dir mocker_perf_data \
  --resolution 100

# Use the generated NPZ with mocker
python -m dynamo.mocker \
  --model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
  --planner-profile-data mocker_perf_data/perf_data.npz
```

#### Option 2: Generate from custom profiler runs

To convert your own profiler results into the NPZ format suitable for the mocker, you'll need to run the profiler (see [SLA-driven profiling documentation](../../../../docs/benchmarks/sla_driven_profiling.md) for details). Note that this is generally run in a Kubernetes environment.

```bash
# Run the profiler
python benchmarks/profiler/profile_sla.py \
  --profile-config your_profile_config.yaml

# Convert profiler results to mocker-compatible NPZ format
python components/src/dynamo/mocker/utils/planner_profiler_perf_data_converter.py \
  --profile_results_dir profiling_results/selected_prefill_interpolation \
  --output_dir profiling_results \
  --resolution 100

# This creates profiling_results/perf_data.npz
```

The converter script combines prefill and decode interpolation data into a single NPZ file with the appropriate array structure.

### How it works

When you provide `--planner-profile-data`:
1. The mocker loads the NPZ file during initialization
2. Prefill timing uses 1D linear interpolation on the ISL grid
3. Decode timing uses 2D bilinear interpolation on (active_kv_tokens, context_length)

Without `--planner-profile-data`, the mocker falls back to the default polynomial formulas for backward compatibility.