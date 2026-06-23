// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing-neutral session affinity and backend lifecycle coordination.

mod coordinator;
mod lifecycle;
mod push_router;
#[cfg(test)]
mod tests;

pub use coordinator::{SessionCoordinator, SessionOperation};
pub use lifecycle::{EventSessionLifecycle, LifecycleError, SessionLifecycleBackend};
pub use push_router::{SessionPushRouter, explicit_target};

use std::{sync::Arc, time::Duration};

use dynamo_runtime::protocols::annotated::Annotated;
use tokio::time::Instant;

use crate::protocols::common::llm_backend::LLMEngineOutput;

const SESSION_TIMEOUT_FALLBACK_BUFFER: Duration = Duration::from_secs(30);

type LlmResponse = Annotated<LLMEngineOutput>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionTarget {
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SessionKind {
    RouterOnly,
    EngineBacked,
}

#[derive(Clone, Debug)]
struct SessionBinding {
    target: SessionTarget,
    kind: SessionKind,
    timeout: Duration,
    expires_at: Instant,
}

#[derive(Clone, Debug)]
struct OpeningState {
    revision: u64,
    kind: SessionKind,
    requested_target: Option<SessionTarget>,
    target: Option<SessionTarget>,
    timeout: Duration,
    notify: Arc<tokio::sync::Notify>,
}

#[derive(Clone, Debug)]
struct BoundState {
    revision: u64,
    binding: SessionBinding,
    active_leases: usize,
}

#[derive(Clone, Debug)]
struct ClosingState {
    revision: u64,
    binding: SessionBinding,
    remove_at: Instant,
    retry_started: bool,
}

#[derive(Clone, Debug)]
enum SessionState {
    Opening(OpeningState),
    Bound(BoundState),
    Closing(ClosingState),
}
