<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Session KV hint policy example

`SessionKvHintPolicy` is a linked post-selection policy. Register it on `HttpFrontend`; Dynamo continues to use its configured worker selector.

The policy resolves the selected worker's known session lineage into external block hashes. A final session request emits `kv.evict`; an optional fixed-retention configuration emits `kv.retain`. Both actions execute after the carrying request completes and include that request's blocks.
