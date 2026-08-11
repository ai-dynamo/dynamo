// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-facing protocol for the shared AISimulate generalized engine.
//!
//! Engine scheduling, native KV accounting, preemption, and timing live in
//! `aisimulate_core::engine`. This module retains only the asynchronous compatibility
//! contract consumed by Dynamo's Live Mocker and handoff driver.

mod metrics;
mod protocol;

use crate::common::protocols::{DirectRequest, OutputSignal};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
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
        cancel: CancellationToken,
    },
}

#[derive(Debug)]
pub(crate) enum SchedulerEventSendError {
    OutputClosed(Vec<OutputSignal>),
    OrderedLaneClosed,
    Cancelled,
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
            Self::Ordered { tx, cancel, .. } => {
                tokio::select! {
                    biased;
                    result = tx.send(LiveEngineEvent::Admissions(admissions.to_vec())) => {
                        result.map_err(|_| {
                            if cancel.is_cancelled() {
                                SchedulerEventSendError::Cancelled
                            } else {
                                SchedulerEventSendError::OrderedLaneClosed
                            }
                        })
                    }
                    _ = cancel.cancelled() => Err(SchedulerEventSendError::Cancelled),
                }
            }
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
            Self::Ordered { tx, cancel, .. } => {
                let (delivered, acknowledged) = oneshot::channel();
                tokio::select! {
                    biased;
                    result = tx.send(LiveEngineEvent::Outputs { signals, delivered }) => {
                        result.map_err(|_| {
                            if cancel.is_cancelled() {
                                SchedulerEventSendError::Cancelled
                            } else {
                                SchedulerEventSendError::OrderedLaneClosed
                            }
                        })?;
                    }
                    _ = cancel.cancelled() => return Err(SchedulerEventSendError::Cancelled),
                }
                let failed = tokio::select! {
                    biased;
                    result = acknowledged => {
                        result.map_err(|_| {
                            if cancel.is_cancelled() {
                                SchedulerEventSendError::Cancelled
                            } else {
                                SchedulerEventSendError::OrderedLaneClosed
                            }
                        })?
                    }
                    _ = cancel.cancelled() => return Err(SchedulerEventSendError::Cancelled),
                };
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
            cancel: CancellationToken::new(),
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

    #[tokio::test]
    async fn dropped_ordered_output_ack_is_orderly_after_cancellation() {
        let (tx, mut rx) = mpsc::channel(1);
        let cancel = CancellationToken::new();
        let sender = SchedulerEventSender::Ordered {
            tx,
            forward_admissions: false,
            cancel: cancel.clone(),
        };
        let send = tokio::spawn(async move {
            sender
                .send_outputs(vec![OutputSignal {
                    uuid: Uuid::from_u128(2),
                    token_id: Some(3),
                    completed: true,
                    rejected: false,
                    handoff_delay_ms: None,
                }])
                .await
        });

        let Some(LiveEngineEvent::Outputs { delivered, .. }) = rx.recv().await else {
            panic!("expected an ordered output batch");
        };
        cancel.cancel();
        drop(delivered);
        assert!(matches!(
            send.await.unwrap(),
            Err(SchedulerEventSendError::Cancelled)
        ));
    }

    #[tokio::test]
    async fn dropped_ordered_output_ack_without_cancellation_is_an_error() {
        let (tx, mut rx) = mpsc::channel(1);
        let sender = SchedulerEventSender::Ordered {
            tx,
            forward_admissions: false,
            cancel: CancellationToken::new(),
        };
        let send = tokio::spawn(async move {
            sender
                .send_outputs(vec![OutputSignal {
                    uuid: Uuid::from_u128(3),
                    token_id: Some(4),
                    completed: true,
                    rejected: false,
                    handoff_delay_ms: None,
                }])
                .await
        });

        let Some(LiveEngineEvent::Outputs { delivered, .. }) = rx.recv().await else {
            panic!("expected an ordered output batch");
        };
        drop(delivered);
        assert!(matches!(
            send.await.unwrap(),
            Err(SchedulerEventSendError::OrderedLaneClosed)
        ));
    }
}
