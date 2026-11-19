<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-session-affinity

`dynamo-session-affinity` keeps the table that pins a session to a worker, so later turns of a conversation return to the worker that holds its KV cache. Bindings carry leases and versions so they can be released, rebound when a worker fails, and replicated between router instances; the replication transport lives in `dynamo-kv-router`.
