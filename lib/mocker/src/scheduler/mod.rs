// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-facing protocol for the shared AISimulate generalized engine.
//!
//! Engine scheduling, native KV accounting, preemption, and timing live in
//! `aisimulate-engine`. This module retains only the asynchronous compatibility
//! contract consumed by Dynamo's Live Mocker and handoff driver.

mod metrics;
mod protocol;

use crate::common::protocols::{DirectRequest, OutputSignal};
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

#[derive(Debug)]
pub(crate) enum LiveEngineEvent {
    Admissions(Vec<AdmissionEvent>),
    Outputs {
        signals: Vec<OutputSignal>,
        /// Acknowledge only after the request-route dispatcher has attempted
        /// delivery. The grouped pass boundary waits on this signal, so the
        /// next pass cannot overtake route cleanup for the current one.
        delivered: oneshot::Sender<Vec<OutputSignal>>,
    },
}

/// Visibility point retained by Dynamo's replay-artifact adapter. Native
/// engine observations are captured at the generalized-engine boundary; this
/// enum only selects the timestamp used when rendering legacy artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterEventVisibility {
    PassStart,
    PassEnd,
}

#[derive(Clone)]
pub(crate) enum SchedulerEventSender {
    Outputs(mpsc::UnboundedSender<Vec<OutputSignal>>),
    Ordered {
        tx: mpsc::Sender<LiveEngineEvent>,
        forward_admissions: bool,
    },
}

#[derive(Debug)]
pub(crate) enum SchedulerEventSendError {
    OutputClosed(Vec<OutputSignal>),
    OrderedLaneClosed,
}

impl SchedulerEventSender {
    pub(crate) async fn send_admissions(
        &self,
        admissions: &[AdmissionEvent],
    ) -> Result<(), SchedulerEventSendError> {
        if admissions.is_empty() {
            return Ok(());
        }
        match self {
            Self::Outputs(_) => Ok(()),
            Self::Ordered {
                forward_admissions: false,
                ..
            } => Ok(()),
            Self::Ordered { tx, .. } => tx
                .send(LiveEngineEvent::Admissions(admissions.to_vec()))
                .await
                .map_err(|_| SchedulerEventSendError::OrderedLaneClosed),
        }
    }

    pub(crate) async fn send_outputs(
        &self,
        signals: Vec<OutputSignal>,
    ) -> Result<(), SchedulerEventSendError> {
        match self {
            Self::Outputs(tx) => tx
                .send(signals)
                .map_err(|error| SchedulerEventSendError::OutputClosed(error.0)),
            Self::Ordered { tx, .. } => {
                let (delivered, acknowledged) = oneshot::channel();
                tx.send(LiveEngineEvent::Outputs { signals, delivered })
                    .await
                    .map_err(|_| SchedulerEventSendError::OrderedLaneClosed)?;
                let failed = acknowledged
                    .await
                    .map_err(|_| SchedulerEventSendError::OrderedLaneClosed)?;
                if failed.is_empty() {
                    Ok(())
                } else {
                    Err(SchedulerEventSendError::OutputClosed(failed))
                }
            }
        }
    }
}

impl From<mpsc::UnboundedSender<Vec<OutputSignal>>> for SchedulerEventSender {
    fn from(tx: mpsc::UnboundedSender<Vec<OutputSignal>>) -> Self {
        Self::Outputs(tx)
    }
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
    fn request_sender(&self) -> mpsc::Sender<DirectRequest>;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ordered_output_send_waits_for_route_delivery_ack() {
        let (tx, mut rx) = mpsc::channel(1);
        let sender = SchedulerEventSender::Ordered {
            tx,
            forward_admissions: false,
        };
        let send = tokio::spawn(async move {
            sender
                .send_outputs(vec![OutputSignal {
                    uuid: Uuid::from_u128(1),
                    token_id: Some(2),
                    completed: true,
                    rejected: false,
                    handoff_delay_ms: None,
                }])
                .await
        });

        let Some(LiveEngineEvent::Outputs { signals, delivered }) = rx.recv().await else {
            panic!("expected an ordered output batch");
        };
        assert_eq!(signals.len(), 1);
        tokio::task::yield_now().await;
        assert!(
            !send.is_finished(),
            "enqueueing the output must not acknowledge route delivery"
        );

        delivered.send(Vec::new()).unwrap();
        send.await.unwrap().unwrap();
    }
}
