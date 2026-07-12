<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CUDA checkpoint helper

`cuda-checkpoint-helper` is the long-running daemon used by the snapshot agent
for CUDA checkpoint and restore operations. When CUDA CustomStorage is
available, it transfers checkpoint extents through the NIXL POSIX backend.

The production transfer path supports deterministic file sharding and a
bounded ring of pinned transfer buffers. Buffer count and chunk size are
validated per device and per operation before workers start.

Successful CustomStorage operations emit a `cuda_custom_storage_transfer` JSON
payload. Its timing fields cover:

- storage directory and manifest validation
- CUDA initialization, device enumeration, and primary-context management
- the synchronous CUDA checkpoint or restore API call
- transfer-job construction and worker orchestration
- post-transfer validation and operation completion

Totals contain subphases and must not be added to them. Worker service times
are sums across workers and can overlap in wall time; the `timing_scope` field
records these rules.

The helper resolves CustomStorage capability once at daemon startup. Primary
contexts are released after CUDA acknowledges a successful operation, and the
payload records release success and status. A release failure is logged and
reported but is not retried after the operation has been acknowledged.
