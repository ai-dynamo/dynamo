## Overview

Pipeline Architecture:

```
Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
      │ nova/distributed-runtime
      ▼
┌─────────────┐
│   Middle    │
└─────────────┘
      │ nova/distributed-runtime
      ▼
┌─────────────┐
│  Backend    │
└─────────────┘
```


## Unified serve
Launch all three services using a single command -

```bash
cd /workspace/examples/python_rs/llm/vllm

compoundai serve sdk_basic_service.basic:Frontend
```
