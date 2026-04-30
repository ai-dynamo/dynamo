// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kvbm_audit` tracing target for kvbm-engine.
//!
//! Mirror of `kvbm_connector::connector::leader::audit` so session
//! and worker-side events land on the same `target: "kvbm_audit"`
//! sink the `cd-trace.py` parser reads. Kept as an in-crate macro
//! (no cross-crate dependency) so kvbm-engine doesn't pull
//! kvbm-connector.

#[macro_export]
macro_rules! engine_audit {
    ($event:expr, $($field:tt)*) => {
        tracing::info!(
            target: "kvbm_audit",
            event = $event,
            $($field)*
        )
    };
}
