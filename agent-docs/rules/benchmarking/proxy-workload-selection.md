<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Proxy Workload Selection

Choose benchmark input in this order:

1. Exact user-provided trace.
2. Exact user-provided request shape and traffic controls.
3. Closest compatible Dynamo recipe workload, labeled `recipe_proxy`.

Use a proxy only when no exact workload is available. It supports exploration, not target-workload validation.

## Proxy Fit

Compare candidate proxies by:

- endpoint, tokenizer, prompt format, and streaming behavior;
- input- and output-length distributions;
- arrival schedule, request rate, or concurrency; and
- prefix and cache-reuse structure.

Record the source, path, SHA256, selection reason, and known mismatches. Preserve the proxy unchanged; any workload
transformation starts a new benchmark series.
