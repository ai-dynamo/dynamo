<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Continuum KV hint unit tests and microbenchmarks

This directory contains the focused checks used before workload-level experiments:

```text
dynamo-vllm-boundary/              in-process Dynamo-to-vLLM boundary smoke
dynamo-vllm-e2e-hints-microbench/ Dynamo vLLM constrained-cache hint matrix
session-prefix-indexer/            SessionPrefixIndexer CPU benchmark and sanity example
weka-correctness/                  reduced WEKA lifecycle and end-to-end correctness replay
```

These checks validate protocol plumbing, action semantics, and indexing behavior. They are not statistically meaningful workload-performance experiments.
