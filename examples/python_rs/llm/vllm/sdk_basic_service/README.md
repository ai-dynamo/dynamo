## Overview

Pipeline Architecture:

```
Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
      │ dynemo/distributed-runtime
      ▼
┌─────────────┐
│   Middle    │
└─────────────┘
      │ dynemo/distributed-runtime
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
