<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-slot-tracker

`dynamo-slot-tracker` keeps track of the requests currently running on each worker rank, with their KV blocks and remaining prefill, so the router can estimate load and decide when a worker is full. It also defines the replication wire that keeps the views of several router replicas in sync. The replica mesh and HTTP server that use it live in `dynamo-kv-router`.
