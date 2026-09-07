<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Session KV hint policy example

`SessionKvHintPolicy` is a linked post-selection policy. Call this crate's `register(HttpFrontend::default(), ...)` helper; Dynamo continues to use its configured worker selector.

The policy resolves the materialized session lineage into decimal-string external block hashes. Explicit session eviction intent emits `kv.evict`; an optional fixed-retention configuration emits `kv.retain`. Both actions execute after the carrying request completes, and the selected vLLM worker resolves the hashes it owns and ignores non-local hashes.
