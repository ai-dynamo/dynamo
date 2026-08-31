// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test harness utilities (feature-gated behind `testing`).
//!
//! Populated by Wave 2:
//! - `TestConsolidator`: ephemeral-port builder.
//! - `ZmqPubHandle` / `ZmqSubHandle`: fake producers/consumers for integration tests.
//! - `KvbmEventInjector`: wraps a `broadcast::Sender<KvCacheEvent>` for simulated KVBM events.
//! - `load_fixture`: msgpack fixture loader for e2e replay.
//! - `new_test_tracing`: once_cell-guarded tracing init.

// Intentionally empty at scaffold stage.
