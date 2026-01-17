# KV Behavior & Model Determinism Tests (kvbm)

## Overview

This suite validates the determinism properties of the API-backed LLM under fixed sampling parameters and, optionally, across prefix cache resets. The tests can automatically start a local LLM server—either a vLLM server or a TensorRT-LLM server—warm it up, and compare responses for identical prompts over multiple iterations. The suite also automatically detects whether the vLLM or TensorRT-LLM wheel is installed and starts the corresponding server.

## Files

- `test_determinism.py` — comprehensive determinism tests with automatic LLM server lifecycle and warmup.
  - `test_determinism_with_cache_reset` — run test with warmup, reset cache, then run again without warmup to test determinism across cache reset boundary
  - `test_concurrent_determinism_with_ifeval` — send parametrized number of IFEval prompts (default: 120) with controlled concurrency, with warmup, then reset cache and test again without warmup to validate determinism across cache reset

## Markers

- `kvbm` — KV behavior and model determinism tests
- `e2e` — end-to-end tests
- `slow` — tests may take a while due to warmup/iterations
- `nightly` — preferred for nightly runs

## How It Works

- A `LLMServerManager` fixture (`llm_server`) launches `vllm serve` or `trtllm-serve` with the Dynamo connector and optional cache block overrides.
- A `tester` fixture binds the test client to the running server's base URL.
- The test performs a comprehensive warmup across prompts, then executes repeated requests and checks that responses are identical (deterministic). An optional cache reset phase re-validates determinism across the reset boundary.

## Running

Run all kvbm tests:

```bash
pytest -v -m "kvbm" -s
```

Run the determinism test file directly inside dynamo repo:

```bash
pytest -v tests/kvbm_integration/test_determinism_agg.py -s

# disagg needs 2 GPUs to run
pytest -v tests/kvbm_integration/test_determinism_disagg.py -s
```

## Configuration

Environment variables control server settings and test load:

- Server/model
  - `KVBM_MODEL_ID` (default: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)
  - `KVBM_SERVER_PORT` (default: `8000`)
  - `KVBM_SERVER_START_TIMEOUT` (default: `300` seconds)

- Cache size overrides
  - `KVBM_CPU_BLOCKS` (used via test parametrization; default: `10000`)
  - `--num-gpu-blocks-override` is applied when `gpu_blocks` is parametrized

- Request/test parameters
  - `KVBM_MAX_TOKENS` (default: `48`) - single integer for max tokens per request
  - `KVBM_SEED` (default: `42`)
  - `KVBM_MAX_ITERATIONS` (default: `500`)
  - `KVBM_WORD_COUNT` (default: `200`)
  - `KVBM_CONTROL_INTERVAL` (default: `10`)
  - `KVBM_SHAKESPEARE_INTERVAL` (default: `1`)
  - `KVBM_RANDOM_INTERVAL` (default: `7`)
  - `KVBM_HTTP_TIMEOUT` (default: `30` seconds)
  - `KVBM_SHAKESPEARE_URL` (default: MIT OCW Shakespeare text)

- Concurrent testing (only for `test_concurrent_determinism_with_ifeval`)
  - `KVBM_CONCURRENT_REQUESTS` (default: `3`) - comma-separated list for parametrization of max concurrent workers
  - `KVBM_IFEVAL_PROMPTS` (default: `120`) - comma-separated list for parametrization of number of IFEval prompts

### Example

```bash
KVBM_MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
KVBM_CPU_BLOCKS=10000 \
KVBM_MAX_ITERATIONS=100 \
KVBM_MAX_TOKENS=48 \
KVBM_CONCURRENT_REQUESTS="10,25,50" \
KVBM_IFEVAL_PROMPTS="50,120,200" \
pytest -v -m "kvbm" -s
```

## Requirements

- `vllm` executable available in PATH inside the test environment.
- The connector module path must be valid: `kvbm.vllm_integration.connector`.
- NATS and etcd services (provided automatically by the `runtime_services` fixture).
- `datasets` library for IFEval concurrent testing (included in test dependencies).
- For containerized workflows, follow the top-level `tests/README.md` guidance to build/run the appropriate image, then execute pytest inside the container.

## Notes

- Warmup is critical to avoid initialization effects impacting determinism.
- For faster local iteration, reduce `KVBM_MAX_ITERATIONS` and/or increase intervals.
- Logs are written under the per-test directory created by `tests/conftest.py` and include the LLM server stdout/stderr.
- Tests use the static port defined by `KVBM_SERVER_PORT` for LLM server communication.
