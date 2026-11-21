<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-admission

`dynamo-admission` decides, when a request arrives, whether to send it now, hold it until a worker has room, or reject it. Held requests wait in per-class queues ordered by policy (FCFS, LCFS, WSPT, or earliest deadline) that take weighted turns, and a classifier plugin can assign each request its class. The YAML policy file and the plugin registry live in `dynamo-kv-router`.
