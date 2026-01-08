# Scale Testing Tool

A scale testing tool that spins up configurable numbers of Dynamo mocker instances as Python processes with shared NATS/etcd infrastructure and includes basic load generation to verify functionality.

## Overview

This tool allows you to:
- Start multiple Dynamo mocker deployments (each with its own frontend)
- Share a single NATS and etcd infrastructure across all deployments
- Isolate each deployment using unique `DYN_NAMESPACE` values
- Generate load across all frontends to verify functionality
- Collect and report latency/throughput metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Scale Test Tool                            │
└─────────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Shared     │     │  Shared     │     │    Load     │
    │  NATS       │     │  etcd       │     │  Generator  │
    └─────────────┘     └─────────────┘     └─────────────┘
           │                   │                   │
           └─────────┬─────────┘                   │
                     │                             │
        ┌────────────┼────────────┐                │
        ▼            ▼            ▼                │
   ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │Mocker 1 │  │Mocker 2 │  │Mocker N │          │
   │ns:      │  │ns:      │  │ns:      │          │
   │scale-   │  │scale-   │  │scale-   │          │
   │test-1   │  │test-2   │  │test-N   │          │
   └────┬────┘  └────┬────┘  └────┬────┘          │
        │            │            │                │
        ▼            ▼            ▼                │
   ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │Frontend │  │Frontend │  │Frontend │◄─────────┘
   │:8001    │  │:8002    │  │:800N    │
   └─────────┘  └─────────┘  └─────────┘
```

## Quick Start

### Run a Full Test

Start 10 mocker deployments, generate load for 60 seconds at 2 QPS, then cleanup:

```bash
python -m tests.scale_test run --count 10 --duration 60 --qps 2
```

### Start Deployments for Manual Testing

Start 5 deployments and keep them running for manual testing:

```bash
python -m tests.scale_test start --count 5
```

Press `Ctrl+C` to stop and cleanup.

### Cleanup Leftover Processes

If processes were left running (e.g., after a crash):

```bash
python -m tests.scale_test cleanup
```

## Commands

### `start`

Start N deployments and wait for manual testing.

```bash
python -m tests.scale_test start [options]
```

Options:
- `--count N`: Number of deployments (default: 5)
- `--model-path PATH`: Model path for tokenizer (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `--speedup-ratio RATIO`: Mocker speedup multiplier (default: 10.0)
- `--base-port PORT`: Starting frontend port (default: 8001)
- `--display-output`: Display process output to console

### `run`

Run a full test: start + load generation + cleanup.

```bash
python -m tests.scale_test run [options]
```

All options from `start`, plus:
- `--duration SECS`: Load test duration in seconds (default: 60)
- `--qps RATE`: Queries per second to send (default: 1.0)

### `cleanup`

Cleanup any leftover scale test processes.

```bash
python -m tests.scale_test cleanup
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--count` | 5 | Number of mocker/frontend pairs to deploy |
| `--model-path` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Model path for tokenizer |
| `--speedup-ratio` | 10.0 | Mocker speedup (higher = faster simulation) |
| `--base-port` | 8001 | Starting port for frontends |
| `--duration` | 60 | Load test duration (seconds) |
| `--qps` | 1.0 | Queries per second to generate |
| `-v, --verbose` | false | Enable debug logging |
| `--display-output` | false | Show process output in console |

## Example Output

```
$ python -m tests.scale_test run --count 10 --duration 120 --qps 2

Starting shared NATS and etcd...
NATS started on port 4222
etcd started on port 2379

Starting 10 mocker processes...
Mocker 1 started (namespace: scale-test-1)
Mocker 2 started (namespace: scale-test-2)
...

Starting 10 frontend processes...
Frontend 1 started on port 8001
Frontend 2 started on port 8002
...

All services ready!

Generating load for 120 seconds at 2 QPS...
Load generation complete.

======================================================================
LOAD GENERATION RESULTS
======================================================================

Frontend http://localhost:8001:
  Requests: 24
  Successful: 24
  Errors: 0 (0.0%)
  Avg latency: 45.2ms
  P50 latency: 42.1ms
  P99 latency: 78.3ms

Frontend http://localhost:8002:
  Requests: 24
  Successful: 24
  Errors: 0 (0.0%)
  Avg latency: 47.1ms
  ...

----------------------------------------------------------------------
Total requests: 240
Total errors: 0
Overall error rate: 0.0%
======================================================================

Cleaning up...
All processes terminated.
```

## Namespace Isolation

Each mocker/frontend pair uses a unique namespace to ensure isolation:

```python
env["DYN_NAMESPACE"] = f"scale-test-{deployment_id}"
env["NATS_SERVER"] = "nats://localhost:4222"
env["ETCD_ENDPOINTS"] = "http://localhost:2379"
```

This allows all deployments to share the same NATS/etcd infrastructure while maintaining logical separation.

## Troubleshooting

### Processes not starting
- Check if ports are already in use: `lsof -i :8001-8020`
- Run cleanup: `python -m tests.scale_test cleanup`

### High error rates
- Increase `--speedup-ratio` for faster mocker responses
- Decrease `--qps` to reduce load
- Check logs in the temporary log directory (printed at startup)

### Services not becoming ready
- Increase timeout by checking the ScaleManager timeout parameter
- Ensure NATS and etcd binaries are installed and in PATH
- Check system resources (memory, CPU)

## Programmatic Usage

You can also use the components directly in Python:

```python
from tests.scale_test import ScaleManager, LoadGenerator

# Using context manager for automatic cleanup
with ScaleManager(num_deployments=5) as manager:
    urls = manager.get_frontend_urls()
    print(f"Frontends: {urls}")

    # Run load generation
    import asyncio
    from tests.scale_test.load_generator import run_load_test

    asyncio.run(run_load_test(
        frontend_urls=urls,
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        duration_sec=30,
        qps=1.0,
    ))
```

## Dependencies

- Python 3.10+
- NATS server (`nats-server` binary in PATH)
- etcd (`etcd` binary in PATH)
- OpenAI Python client
- psutil
- requests


