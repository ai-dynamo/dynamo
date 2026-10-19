<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native replay routing

This adapter belongs to Dynamo and calls its workspace LocalScheduler and
SessionAffinity. AISimulate owns the shared Replayer, lifecycle, and virtual time.
The ai-dynamo integration supplies this composition to AISimulate; the AISimulate
wheel contains no Dynamo routing implementation.

Moved and modified from ai-dynamo/aisimulate at immutable revision
8b00c0337bb4f33fb534f3528de8546ea9fa7082, `crates/python/src/native_router/`
and the shared native execution function in `crates/python/src/lib.rs`:
https://github.com/ai-dynamo/aisimulate/tree/8b00c0337bb4f33fb534f3528de8546ea9fa7082/crates/python/src
Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
Existing file headers preserve the original Dynamo sources and revisions.
Changes use the local Dynamo workspace dependencies and report their package
version, rather than a separately pinned copy compiled into an AISimulate wheel.
