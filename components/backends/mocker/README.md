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

### Example with individual arguments (vLLM-style):
```bash
# Start mocker with custom configuration
python -m dynamo.mocker \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-gpu-blocks-override 8192 \
  --block-size 16 \
  --speedup-ratio 10.0 \
  --max-num-seqs 512 \
  --enable-prefix-caching

# Start frontend server
python -m dynamo.frontend --http-port 8080
```

### Legacy JSON file support:
For backward compatibility, you can still provide configuration via a JSON file:

```bash
echo '{"speedup_ratio": 10.0, "num_gpu_blocks": 8192}' > mocker_args.json
python -m dynamo.mocker \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --extra-engine-args mocker_args.json
```

Note: If `--extra-engine-args` is provided, it overrides all individual CLI arguments.