// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill router feature.
//!
//! Owns selection-and-execution of prefill requests across a fleet of
//! registered prefill workers. The hub's CD feature publishes prefill
//! requests onto its messenger queue; this feature consumes them via the
//! [`PrefillRequestDispatcher`] handshake and routes each one to a worker
//! advertised through [`PrefillRouterConfig`] at registration.
//!
//! Selection is load-aware (lowest `load_net_new`, then lowest `inflight`,
//! then lowest index), capped by a per-worker concurrency limit and a
//! fleet-wide semaphore that propagates backpressure to the queue.
//!
//! Execution sits behind [`PrefillExecutionBackend`] so a worker can be
//! reached over different transports — HTTP today, velo unary later.

pub mod dispatcher;
pub mod execution;
pub mod manager;
pub mod protocol;
pub mod router;
pub mod selection;

pub use dispatcher::{DispatchOutcome, PrefillRequestDispatcher, RecordingDispatcher};
pub use execution::{HttpExecutionBackend, PrefillExecutionBackend, VeloExecutionBackend};
pub use manager::PrefillRouterManager;
pub use protocol::{
    PREFILL_DISPATCH_HANDLER, PrefillBackendAdvertisement, PrefillDispatchRequest,
    PrefillDispatchResponse, PrefillRouterConfig, PrefillTargetSummary, ROUTE_PREFIX,
    TargetsResponse, VllmHttpEndpoint,
};
pub use router::PrefillRouter;
pub use selection::{Selector, SelectorConfig};
