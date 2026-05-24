// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ConditionalDisagg feature — hub-side manager and client-side wrapper.

/// `kvbmctl` client CLI for this feature. Gated behind the `kvbmctl` feature.
#[cfg(feature = "kvbmctl")]
pub mod cli;
pub mod client;
pub mod dispatcher;
pub mod load_aware;
pub mod manager;
pub mod protocol;

pub use client::ConditionalDisaggClient;
pub use dispatcher::{
    DispatchOutcome, HttpVllmDispatcher, PrefillRequestDispatcher, RecordingDispatcher,
};
pub use load_aware::LoadAwareHttpDispatcher;
pub use manager::ConditionalDisaggManager;
pub use protocol::{ConditionalDisaggInstancesResponse, ROUTE_PREFIX};
