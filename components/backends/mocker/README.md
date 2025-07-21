# Mocker engine

The mocker engine is a mock vLLM implementation designed for testing and development purposes. It simulates realistic token generation timing without requiring actual model inference, making it useful for:

- Testing distributed system components without GPU resources
- Benchmarking infrastructure and networking overhead
- Developing and debugging Dynamo components
- Load testing and performance analysis

**Basic usage:**

The `--model-path` is required but can point to any valid model path - the mocker doesn't actually load the model weights (but the pre-processor needs the tokenizer). The arguments `block_size`, `num_gpu_blocks`, `max_num_seqs`, `max_num_batched_tokens`, `enable_prefix_caching`, and `enable_chunked_prefill` are common arguments shared with the real VLLM engine.

And below are arguments that are mocker-specific:
- `speedup_ratio`: Speed multiplier for token generation (default: 1.0). Higher values make the simulation engines run faster.
- `dp_size`: Number of data parallel workers to simulate (default: 1)
- `watermark`: KV cache watermark threshold as a fraction (default: 0.01). This argument also exists for the real VLLM engine but cannot be passed as an engine arg.

```bash
echo '{"speedup_ratio": 10.0}' > mocker_args.json
python -m dynamo.mocker --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --extra-engine-args mocker_args.json
python -m dynamo.frontend --http-port 8080
```