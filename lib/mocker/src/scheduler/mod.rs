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
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
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
        enqueued_at: Instant,
        /// Acknowledge only after the request-route dispatcher has attempted
        /// delivery. The grouped pass boundary waits on this signal, so the
        /// next pass cannot overtake route cleanup for the current one.
        delivered: oneshot::Sender<OutputDeliveryAck>,
    },
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct OutputRouteTiming {
    pub(crate) predecessor_busy: Duration,
    pub(crate) receiver_wake: Duration,
    pub(crate) gate_wait: Duration,
    pub(crate) route_wall: Duration,
    pub(crate) route_thread_cpu: Option<Duration>,
    pub(crate) dispatcher_residual: Duration,
    pub(crate) signals: u64,
    pub(crate) terminals: u64,
    pub(crate) route_found: u64,
    pub(crate) route_missing: u64,
    pub(crate) delivered: u64,
    pub(crate) full: u64,
    pub(crate) closed: u64,
    pub(crate) terminal_removals: u64,
}

#[derive(Debug)]
pub(crate) struct OutputDeliveryAck {
    pub(crate) failed: Vec<OutputSignal>,
    pub(crate) timing: OutputRouteTiming,
    pub(crate) acknowledged_at: Instant,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct OutputPublishTiming {
    pub(crate) wall: Duration,
    pub(crate) admission: Duration,
    pub(crate) reserve: Duration,
    pub(crate) reserve_waited: bool,
    pub(crate) predecessor_busy: Duration,
    pub(crate) receiver_wake: Duration,
    pub(crate) gate_wait: Duration,
    pub(crate) route_wall: Duration,
    pub(crate) route_thread_cpu: Option<Duration>,
    pub(crate) dispatcher_residual: Duration,
    pub(crate) ack_wake: Duration,
    pub(crate) residual_ms: f64,
    pub(crate) signals: u64,
    pub(crate) uninstrumented_signals: u64,
    pub(crate) terminals: u64,
    pub(crate) route_found: u64,
    pub(crate) route_missing: u64,
    pub(crate) delivered: u64,
    pub(crate) full: u64,
    pub(crate) closed: u64,
    pub(crate) terminal_removals: u64,
}

#[derive(Debug)]
pub(crate) struct OutputSendResult {
    pub(crate) failed: Vec<OutputSignal>,
    pub(crate) timing: OutputPublishTiming,
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

    #[cfg(test)]
    pub(crate) async fn send_outputs(
        &self,
        signals: Vec<OutputSignal>,
    ) -> Result<(), SchedulerEventSendError> {
        let result = self.send_outputs_timed(signals).await?;
        if result.failed.is_empty() {
            Ok(())
        } else {
            Err(SchedulerEventSendError::OutputClosed(result.failed))
        }
    }

    pub(crate) async fn send_outputs_timed(
        &self,
        signals: Vec<OutputSignal>,
    ) -> Result<OutputSendResult, SchedulerEventSendError> {
        let started_at = Instant::now();
        match self {
            Self::Outputs(tx) => {
                let signal_count = signals.len() as u64;
                tx.send(signals)
                    .map_err(|error| SchedulerEventSendError::OutputClosed(error.0))?;
                Ok(OutputSendResult {
                    failed: Vec::new(),
                    timing: OutputPublishTiming {
                        wall: started_at.elapsed(),
                        signals: signal_count,
                        uninstrumented_signals: signal_count,
                        ..OutputPublishTiming::default()
                    },
                })
            }
            Self::Ordered { tx, cancel, .. } => {
                let (permit, reserve, reserve_waited) = match tx.try_reserve() {
                    Ok(permit) => (permit, Duration::ZERO, false),
                    Err(mpsc::error::TrySendError::Full(())) => {
                        let reserve_started_at = Instant::now();
                        let permit = tokio::select! {
                            biased;
                            result = tx.reserve() => {
                                result.map_err(|_| {
                                    if cancel.is_cancelled() {
                                        SchedulerEventSendError::Cancelled
                                    } else {
                                        SchedulerEventSendError::OrderedLaneClosed
                                    }
                                })?
                            }
                            _ = cancel.cancelled() => {
                                return Err(SchedulerEventSendError::Cancelled);
                            }
                        };
                        (permit, reserve_started_at.elapsed(), true)
                    }
                    Err(mpsc::error::TrySendError::Closed(())) => {
                        return Err(if cancel.is_cancelled() {
                            SchedulerEventSendError::Cancelled
                        } else {
                            SchedulerEventSendError::OrderedLaneClosed
                        });
                    }
                };
                let (delivered, acknowledged) = oneshot::channel();
                let enqueued_at = Instant::now();
                permit.send(LiveEngineEvent::Outputs {
                    signals,
                    enqueued_at,
                    delivered,
                });
                let admission = enqueued_at.saturating_duration_since(started_at);
                let acknowledged = tokio::select! {
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
                let ack_wake = acknowledged.acknowledged_at.elapsed();
                let wall = started_at.elapsed();
                let timing = acknowledged.timing;
                let accounted = admission
                    .saturating_add(timing.predecessor_busy)
                    .saturating_add(timing.receiver_wake)
                    .saturating_add(timing.gate_wait)
                    .saturating_add(timing.route_wall)
                    .saturating_add(timing.dispatcher_residual)
                    .saturating_add(ack_wake);
                Ok(OutputSendResult {
                    failed: acknowledged.failed,
                    timing: OutputPublishTiming {
                        wall,
                        admission,
                        reserve,
                        reserve_waited,
                        predecessor_busy: timing.predecessor_busy,
                        receiver_wake: timing.receiver_wake,
                        gate_wait: timing.gate_wait,
                        route_wall: timing.route_wall,
                        route_thread_cpu: timing.route_thread_cpu,
                        dispatcher_residual: timing.dispatcher_residual,
                        ack_wake,
                        residual_ms: (wall.as_secs_f64() - accounted.as_secs_f64()) * 1_000.0,
                        signals: timing.signals,
                        uninstrumented_signals: 0,
                        terminals: timing.terminals,
                        route_found: timing.route_found,
                        route_missing: timing.route_missing,
                        delivered: timing.delivered,
                        full: timing.full,
                        closed: timing.closed,
                        terminal_removals: timing.terminal_removals,
                    },
                })
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
                .send_outputs_timed(vec![OutputSignal {
                    uuid: Uuid::from_u128(1),
                    token_id: Some(2),
                    completed: true,
                    rejected: false,
                    handoff_delay_ms: None,
                    cached_tokens: None,
                }])
                .await
        });

        let Some(LiveEngineEvent::Outputs {
            signals, delivered, ..
        }) = rx.recv().await
        else {
            panic!("expected an ordered output batch");
        };
        assert_eq!(signals.len(), 1);
        tokio::task::yield_now().await;
        assert!(
            !send.is_finished(),
            "enqueueing the output must not acknowledge route delivery"
        );

        delivered
            .send(OutputDeliveryAck {
                failed: Vec::new(),
                timing: OutputRouteTiming {
                    predecessor_busy: Duration::from_millis(1),
                    receiver_wake: Duration::from_millis(2),
                    gate_wait: Duration::from_millis(3),
                    route_wall: Duration::from_millis(4),
                    route_thread_cpu: Some(Duration::from_millis(3)),
                    dispatcher_residual: Duration::from_millis(5),
                    signals: 1,
                    terminals: 1,
                    route_found: 1,
                    delivered: 1,
                    terminal_removals: 1,
                    ..OutputRouteTiming::default()
                },
                acknowledged_at: Instant::now(),
            })
            .unwrap();
        let result = send.await.unwrap().unwrap();
        assert!(result.failed.is_empty());
        assert_eq!(result.timing.predecessor_busy, Duration::from_millis(1));
        assert_eq!(result.timing.receiver_wake, Duration::from_millis(2));
        assert_eq!(result.timing.gate_wait, Duration::from_millis(3));
        assert_eq!(result.timing.route_wall, Duration::from_millis(4));
        assert_eq!(
            result.timing.route_thread_cpu,
            Some(Duration::from_millis(3))
        );
        assert_eq!(result.timing.dispatcher_residual, Duration::from_millis(5));
        assert_eq!(result.timing.signals, 1);
        assert_eq!(result.timing.terminals, 1);
        assert_eq!(result.timing.route_found, 1);
        assert_eq!(result.timing.delivered, 1);
        assert_eq!(result.timing.terminal_removals, 1);
        assert_eq!(result.timing.uninstrumented_signals, 0);
        assert!(!result.timing.reserve_waited);
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
                    cached_tokens: None,
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
                    cached_tokens: None,
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
