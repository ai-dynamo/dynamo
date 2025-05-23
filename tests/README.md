# Dynamo Testing Framework

## Overview

This document outlines the testing framework for the Dynamo runtime system, including test discovery, organization, and best practices.

## Directory Structure

```
tests/
├── unit/               # Unit tests for individual components
│   ├── conftest.py     # Unit test fixtures and configuration
│   └── unittest_utils.py
├── integration/        # Integration tests between components
│   ├── conftest.py     # Integration test fixtures and configuration
│   └── integtest_utils.py
├── e2e/                # End-to-end system tests
│   ├── conftest.py     # E2E test fixtures and configuration
│   └── e2etest_utils.py
├── conftest.py         # Shared fixtures and configuration
└── README.md           # This file
```

## Test Discovery

Pytest automatically discovers tests based on their naming convention. All test files must follow this pattern:

```
test_<type>_<component>_<flow>.py
```

Where:
- `type`: Test category (e2e, integ, unit)
- `component`: The component being tested (e.g., planner, kv_router)
  - For e2e tests, this could be the API or simply "dynamo"
- `flow`: The specific functionality or scenario being tested

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
- `@pytest.mark.planner` - Planner component tests
- `@pytest.mark.kv_router` - KV Router component tests
- etc.

### Execution-related markers
- `@pytest.mark.slow` - Tests that take a long time to run
- `@pytest.mark.skip(reason="Example: KV Manager is under development")` - Skip these tests
- `@pytest.mark.xfail(reason="Expected to fail because...")` - Tests expected to fail

## Writing Tests

### Test Classes
When using class-based tests, utilize the setup and teardown methods:

```python
class TestComponent:
    def setup_method(self):
        # Setup code runs before each test method
        self.component = Component()
        
    def teardown_method(self):
        # Cleanup code runs after each test method
        self.component.cleanup()
        
    def test_feature(self):
        # Test code here
        assert self.component.feature() == expected_result
```

### Testing Error Cases
To test scenarios expected to fail:

```python
def test_integ_planner_fail_to_forward():
    with pytest.mark.raises(RoutingError):
        planner_func(route_req, cache_context)
```

### Mocking
Use mocking to isolate the system under test:

```python
@patch('module.external_service')
def test_with_mock(mock_service):
    # Configure the mock
    mock_service.return_value = expected_data
    
    # Test code that uses the external service
    result = function_under_test()
    
    # Assertions
    assert result == expected_result
    mock_service.assert_called_once()
```

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