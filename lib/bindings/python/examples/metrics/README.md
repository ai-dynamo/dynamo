<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Python Metrics Examples

Example scripts demonstrating how to create and use Prometheus metrics in Python using the Dynamo metrics API.

## Documentation

See the **[Metrics Developer Guide - Python Section](../../../../../docs/observability/metrics-developer-guide.md#metrics-api-in-python)** for complete documentation.

## Running Examples

```bash
cd ~/dynamo/lib/bindings/python/examples/metrics

# Background thread updates
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 ./server_with_loop.py

# Callback-based updates
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 ./server_with_callback.py

# Check Prometheus Exposition Format text metrics
curl http://localhost:8081/metrics
```
