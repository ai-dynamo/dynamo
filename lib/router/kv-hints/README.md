<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-hints

`dynamo-kv-hints` defines the hint the router sends with a request when the chosen worker can fetch cached KV blocks from another worker instead of recomputing them: which sources to pull from, and the runtime-config keys a worker advertises the capability under. It depends on `dynamo-kv-indexer` for the transfer candidates the index produces; the kernel (`dynamo-scheduler`) decides when to attach a hint.
