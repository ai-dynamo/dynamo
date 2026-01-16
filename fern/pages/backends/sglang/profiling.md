---
title: "Profiling SGLang Workers in Dynamo"
---

{/*
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/}

Dynamo exposes profiling endpoints for SGLang workers via the system server's `/engine/*` routes. This allows you to start and stop PyTorch profiling on running inference workers without restarting them.

These endpoints wrap SGLang's internal `TokenizerManager.start_profile()` and `stop_profile()` methods. See SGLang's documentation for the full list of supported parameters.

## Quick Start

1. **Start profiling:**

```bash
curl -X POST http://localhost:9090/engine/start_profile \
  -H "Content-Type: application/json" \
  -d '{"output_dir": "/tmp/profiler_output"}'
```

2. **Run some inference requests to generate profiling data**

3. **Stop profiling:**

```bash
curl -X POST http://localhost:9090/engine/stop_profile
```

4. **View the traces:**

The profiler outputs Chrome trace files in the specified `output_dir`. You can view them using:
- Chrome's `chrome://tracing`
- [Perfetto UI](https://ui.perfetto.dev/)
- TensorBoard with the PyTorch Profiler plugin

## Test Script

A test script is provided at [`examples/backends/sglang/test_sglang_profile.py`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/test_sglang_profile.py) that demonstrates the full profiling workflow:

```bash
python examples/backends/sglang/test_sglang_profile.py
```

