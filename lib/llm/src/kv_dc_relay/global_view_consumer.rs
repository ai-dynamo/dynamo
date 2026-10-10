// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process Relay-backed Global View for the first aggregated router POC.
//! Catalog, CKF, stats, and reconnect lifecycle are owned by this module.

mod coordinator;
pub mod inbound;
mod runtime;
mod scorer;
pub mod stats;
mod stream;

pub use coordinator::{run_relay_view, run_relay_view_epoch, run_relay_view_with_stats};
pub use runtime::{GlobalViewRuntime, RelayDgdSource};
pub use scorer::RelayCkfOverlapStore;
pub use stream::run_exact_aggregated_producer;
