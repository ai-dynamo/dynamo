// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ConditionalDisagg feature — hub-side manager and client-side wrapper.

pub mod client;
pub mod dispatcher;
pub mod manager;
pub mod registry;
pub mod selector;

pub use client::ConditionalDisaggClient;
pub use dispatcher::{
    DispatchOutcome, HttpVllmDispatcher, PrefillRequestDispatcher, RecordingDispatcher,
};
pub use manager::ConditionalDisaggManager;
pub use registry::{CdPeerRegistry, CdRegistryError};
pub use selector::{
    LeastLoadedSelector, LoadPermit, PrefillPeerEntry, PrefillPeerSource,
    PrefillWorkerSelector, RoundRobinSelector, SelectedWorker,
};
