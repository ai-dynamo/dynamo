# Dynamo Testing Framework

## Overview

This document outlines the testing framework for the Dynamo runtime system, including test discovery, organization, and best practices.

## Directory Structure

```
tests/
├── serve/              # E2E tests using dynamo serve
│   ├── conftest.py     # test fixtures as needed for specific test area
├── run/                # E2E tests using dynamo run
│   ├── conftest.py     # test fixtures as needed for specific test area
├── conftest.py         # Shared fixtures and configuration
└── README.md           # This file
```

## Test Discovery

Pytest automatically discovers tests based on their naming convention. All test files must follow this pattern:

```
test_<component>.py
```

Where:
- `component`: The component being tested (e.g., planner, kv_router)
  - For e2e tests, this could be the API or simply "dynamo"

## Running Tests

To run all tests:
```bash
pytest
```

To run only specific tests:
```bash
# Run only vLLM tests
pytest -v -m vllm

# Run only e2e tests
pytest -v -m e2e

# Run tests for a specific component
pytest -v -m planner

# Run with print statements visible
pytest -s
```

## Test Markers

Markers help control which tests run under different conditions. Add these decorators to your test functions:

```
markers = [
    "pre_merge: marks tests to run before merging",
    "nightly: marks tests to run nightly",
    "weekly: marks tests to run weekly",
    "gpu_1: marks tests to run on GPU",
    "gpu_2: marks tests to run on 2GPUs",
    "e2e: marks tests as end-to-end tests",
    "vllm: marks tests as requiring vllm",
    "sglang: marks tests as requiring sglang",
    "slow: marks tests as known to be slow"
]
```

### Frequency-based markers
- `@pytest.mark.nightly` - Tests run nightly
- `@pytest.mark.weekly` - Tests run weekly
- `@pytest.mark.pre_merge` - Tests run before merging PRs

### Role-based markers
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.stress` - Stress/load tests
- `@pytest.mark.benchmark` - Performance benchmark tests

### Component-specific markers
- `@pytest.mark.vllm` - Framework tests
- `@pytest.mark.planner` - Planner component tests
- `@pytest.mark.kv_router` - KV Router component tests
- etc.

### Execution-related markers
- `@pytest.mark.slow` - Tests that take a long time to run
- `@pytest.mark.skip(reason="Example: KV Manager is under development")` - Skip these tests
- `@pytest.mark.xfail(reason="Expected to fail because...")` - Tests expected to fail

## Environment Setup

### Requirements
- etcd service
- nats-server service
- Python dependencies: pytest, requests, transformers, huggingface_hub
- For GPU tests: CUDA-compatible GPU with appropriate drivers

### Environment Variables
- `HF_TOKEN` - Your HuggingFace API token to avoid rate limits
  - Get a token from https://huggingface.co/settings/tokens
  - Set it before running tests: `export HF_TOKEN=your_token_here`

### Model Download Cache

The tests will automatically use a local cache at `~/.cache/huggingface` to avoid
repeated downloads of model files. This cache is shared across test runs to improve performance.

## Troubleshooting

Common issues and solutions:

1. **"Model registration timed out"** - Increase the timeout in `conftest.py` or ensure your GPU has enough memory.

2. **"HTTP server failed to start"** - Check that no other services are using the same port.

3. **"Service health check timed out"** - Verify that the component registration order matches test expectations.

4. **"429 Too Many Requests"** - You're hitting HuggingFace rate limits. Set the `HF_TOKEN` environment variable or try again later.
