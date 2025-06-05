# Dynamo Testing Framework

## Overview

This document outlines the testing framework for the Dynamo runtime system, including test discovery, organization, and best practices.

## Directory Structure

```bash
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

```bash
test_<component_or_flow>.py
```

Where:
- `component_or_flow`: The component or flow being tested (e.g., planner, kv_router)
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
- `@pytest.mark.sglang` - Framework tests
- `@pytest.mark.tensorrtllm` - Framework tests
- `@pytest.mark.planner` - Planner component tests
- `@pytest.mark.kv_router` - KV Router component tests
- etc.

### Execution-related markers
- `@pytest.mark.slow` - Tests that take a long time to run
- `@pytest.mark.skip(reason="Example: KV Manager is under development")` - Skip these tests
- `@pytest.mark.xfail(reason="Expected to fail because...")` - Tests expected to fail

## Environment Setup

Tests are designed to run in the appropriate framework container built
via ```./container/build.sh --framework X``` and run via
```./container/run.sh --mount-workspace -it -- pytest```.


### Environment Variables
- `HF_TOKEN` - Your HuggingFace API token to avoid rate limits
  - Get a token from [HuggingFace Settings](https://huggingface.co/settings/tokens)
  - Set it before running tests: `export HF_TOKEN=your_token_here`

### Model Download Cache

The tests will automatically use a local cache at `~/.cache/huggingface` to avoid
repeated downloads of model files. This cache is shared across test runs to improve performance.

