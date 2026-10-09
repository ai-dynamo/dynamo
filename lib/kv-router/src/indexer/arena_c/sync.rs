// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomics used by the protocol code in `table.rs`.
//!
//! The loom models in `lib/kv-router/loom-arena-c` compile `table.rs` unchanged against
//! their own `sync` module, which re-exports loom's atomics instead.

pub(super) use std::sync::atomic::{AtomicU64, Ordering};
