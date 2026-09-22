<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shutdown validation

Run from a development environment with Rust, rebuilt editable Dynamo bindings, pytest, pytest-asyncio, and pytest-timeout:

```bash
cd lib/bindings/python
maturin develop --uv
cd ../../..
SHUTDOWN_PYTHON=.venv/bin/python bash tests/runtime/validate_shutdown.sh
```

The runner checks runtime teardown and worker stage contracts, then exercises shutdown end to end through real Rust bindings, TCP streams, OS signals, and subprocess exits. It needs permission to bind local sockets but no GPU, model download, etcd, or NATS service.

| Scenario | Required outcome |
| --- | --- |
| Concurrent runtime shutdown callers | All await the same teardown; cancelling one waiter does not cancel teardown. |
| SDK worker hosted by another application | Clean shutdown returns and the host survives beyond the former watchdog deadline. |
| SDK cleanup holds the GIL | The native watchdog still forces exit code 70. |
| Python pull and push streaming | SIGTERM removes discovery, completes admitted work, then cleans the engine and runtime. |
| Positive remaining budget below five seconds | Cleanup gets its five-second reserve instead of being truncated to the remainder. |
| Second SIGTERM | Immediate exit code 70. |
| Idle and slow embedding children | A process-group signal is forwarded once; idle event loops wake, children exit cleanly, and parent cleanup follows. |
| Stale router addresses a closed admission gate | Both response transports preserve the typed Unavailable rejection. |

`shutdown_probe.py` is the subprocess helper; use pytest or the runner rather than invoking its signal modes in a shared process group. Every subprocess has a timeout, and embedding children have their own bounded lifetime.

These checks use a CPU probe engine and the shared Python shutdown coordinator used by vLLM, SGLang, and TensorRT-LLM. The embedding scenarios import the production process supervisor while stubbing unused vLLM engine-construction imports. They do not instantiate WorkerFactory or validate GPU engine teardown, distributed KV transfer, external discovery propagation, or Kubernetes termination. Those require backend-specific serving environments and remain separate deployment validation.
