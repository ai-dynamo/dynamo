# Dynamo vLLM Backend Tests

This directory contains unit tests for the Dynamo vLLM backend components.

## Running Tests

### Run all tests
```bash
cd components/backends/vllm/src/dynamo/tests
python -m pytest
```

### Run specific test file
```bash
python -m pytest test_ports.py
```

### Run with coverage
```bash
python -m pytest --cov=dynamo.vllm.ports --cov-report=term
```

### Run specific test class or method
```bash
python -m pytest test_ports.py::TestPortBinding
python -m pytest test_ports.py::TestPortBinding::test_single_port_binding
```

## Dependencies

The tests require:
- `pytest` - Test framework
- `pytest-asyncio` - For async test support
- `pytest-cov` - For coverage reports (optional)

Install with:
```bash
pip install pytest pytest-asyncio pytest-cov
```
