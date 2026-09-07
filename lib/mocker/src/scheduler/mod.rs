// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-facing protocol for the shared AISimulate generalized engine.
//!
//! Engine scheduling, native KV accounting, preemption, and timing live in
//! `aisimulate_core::engine`. This module retains only the asynchronous compatibility
//! contract consumed by Dynamo's Live Mocker and handoff driver.

mod metrics;
mod protocol;

use crate::common::protocols::DirectRequest;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

pub use crate::common::protocols::ForwardPassSnapshot;
pub use metrics::MockerMetrics;
pub use protocol::{
    SchedulerCommand, SchedulerCommandEffects, SchedulerCommandResult, SchedulerLifecycleEvent,
};

#[derive(Debug, Clone)]
pub(crate) struct AdmissionEvent {
    pub(crate) uuid: Uuid,
    pub(crate) reused_input_tokens: usize,
}

pub struct SchedulerCommandEnvelope {
    pub command: SchedulerCommand,
    pub reply: oneshot::Sender<anyhow::Result<SchedulerCommandEffects>>,
}

/// Visibility point retained by Dynamo's replay-artifact adapter. Native
/// engine observations are captured at the generalized-engine boundary; this
/// enum only selects the timestamp used when rendering legacy artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterEventVisibility {
    PassStart,
    PassEnd,
}

pub struct SchedulerCancellationEnvelope {
    pub request_id: Uuid,
    pub discard_pending_output: bool,
    pub reply: oneshot::Sender<anyhow::Result<SchedulerCommandEffects>>,
}

impl From<SchedulerCancellationEnvelope> for SchedulerCommandEnvelope {
    fn from(cancellation: SchedulerCancellationEnvelope) -> Self {
        Self {
            command: SchedulerCommand::CancelRequest {
                request_id: cancellation.request_id,
            },
            reply: cancellation.reply,
        }
    }
}

/// Engine-agnostic asynchronous scheduler interface retained for Dynamo.
pub trait SchedulerHandle: Send + Sync {
    /// Send a request to the scheduler's waiting queue.
    fn receive(&self, request: DirectRequest);

    /// Get a clone of the compatibility request sender channel.
    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest>;

    fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics>;

    fn command_sender(&self) -> mpsc::Sender<SchedulerCommandEnvelope>;

    fn cancellation_sender(&self) -> mpsc::Sender<SchedulerCancellationEnvelope>;

    fn take_lifecycle_receiver(&mut self) -> Option<mpsc::Receiver<SchedulerLifecycleEvent>>;
}

pub(crate) fn handoff_channel_capacity(args: &crate::common::protocols::MockEngineArgs) -> usize {
    args.effective_handoff_capacity()
        .checked_mul(2)
        .expect("mocker handoff channel capacity overflow")
}
