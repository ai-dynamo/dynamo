// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::{FpmPublisher, KvEventPublishers, OutputSignal};
use crate::scheduler::kv_event_sink::{
    DeferredKvPublish, DeferredKvPublishBuffer, publish_deferred_fpm, publish_deferred_kv_events,
};
use crate::scheduler::vllm::MockerMetrics;
use crate::scheduler::{
    AdmissionEvent, EnginePassResult, RouterEventVisibility, SchedulerCommand,
    SchedulerCommandEffects, SchedulerCommandEnvelope, SchedulerLifecycleEvent,
};
use tokio::sync::{mpsc, watch};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PublishedEffect {
    Admissions,
    Kv,
    Fpm,
    Outputs,
    Accounting,
    Ack,
    Lifecycle,
    Metrics,
}

pub(crate) trait LiveBoundaryCore {
    fn apply_live_command(
        &mut self,
        command: SchedulerCommand,
        allow_destination_admission: bool,
        now_ms: f64,
    ) -> anyhow::Result<SchedulerCommandEffects>;

    fn retry_live_destinations(&mut self, now_ms: f64) -> Vec<SchedulerLifecycleEvent>;

    fn live_metrics(&self) -> MockerMetrics;

    fn pass_boundary_metrics(&self, pass_metrics: MockerMetrics) -> MockerMetrics;

    fn output_delivery_failed(&mut self, _signals: Vec<OutputSignal>) {}

    #[cfg(feature = "kvbm-offload")]
    fn advance_live_offload(
        &mut self,
        _now_ms: f64,
        _allow_destination_admission: bool,
    ) -> crate::scheduler::OffloadTickEffects {
        crate::scheduler::OffloadTickEffects {
            kv_events: Vec::new(),
            lifecycle_events: Vec::new(),
        }
    }
}

/// Owns every effect that becomes externally visible at a live scheduler
/// boundary. Pass effects are captured before modeled sleep so an admissible
/// mid-pass command cannot publish or consume state derived from that pass.
pub(crate) struct LiveEffectsPublisher {
    output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
    admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
    lifecycle_tx: mpsc::Sender<SchedulerLifecycleEvent>,
    metrics_tx: watch::Sender<MockerMetrics>,
    kv_event_publishers: KvEventPublishers,
    fpm_publisher: FpmPublisher,
    captured_kv_events: DeferredKvPublishBuffer,
    #[cfg(test)]
    publication_log: Option<std::sync::Arc<std::sync::Mutex<Vec<PublishedEffect>>>>,
}

pub(crate) struct PendingLivePass {
    pass: EnginePassResult,
    kv_events: Vec<DeferredKvPublish>,
    admissions_published: bool,
    pass_start_kv_published: bool,
}

impl PendingLivePass {
    pub(crate) fn end_ms(&self) -> f64 {
        self.pass.end_ms
    }

    pub(crate) fn made_progress_since(&self, metrics_before: &MockerMetrics) -> bool {
        self.pass.completed_requests > 0
            || !self.pass.admissions.is_empty()
            || !self.pass.output_signals.is_empty()
            || !self.pass.lifecycle_events.is_empty()
            || !self.pass.kv_events.is_empty()
            || !self.kv_events.is_empty()
            || self.pass.mocker_metrics != *metrics_before
    }
}

impl LiveEffectsPublisher {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
        lifecycle_tx: mpsc::Sender<SchedulerLifecycleEvent>,
        metrics_tx: watch::Sender<MockerMetrics>,
        kv_event_publishers: KvEventPublishers,
        fpm_publisher: FpmPublisher,
        captured_kv_events: DeferredKvPublishBuffer,
    ) -> Self {
        Self {
            output_tx,
            admission_tx,
            lifecycle_tx,
            metrics_tx,
            kv_event_publishers,
            fpm_publisher,
            captured_kv_events,
            #[cfg(test)]
            publication_log: None,
        }
    }

    #[cfg(test)]
    fn with_publication_log(
        mut self,
        log: std::sync::Arc<std::sync::Mutex<Vec<PublishedEffect>>>,
    ) -> Self {
        self.publication_log = Some(log);
        self
    }

    pub(crate) fn capture_pass(&self, pass: EnginePassResult) -> PendingLivePass {
        PendingLivePass {
            pass,
            kv_events: self.captured_kv_events.drain(),
            admissions_published: false,
            pass_start_kv_published: false,
        }
    }

    /// Publish effects whose scheduler contract makes them visible before the
    /// modeled GPU pass runs. Outputs and lifecycle effects remain at pass end.
    pub(crate) fn publish_pass_start(&self, pending: &mut PendingLivePass) {
        self.publish_admissions(&pending.pass.admissions);
        pending.admissions_published = true;
        if pending.pass.router_event_visibility == RouterEventVisibility::PassStart {
            self.publish_router_effects(std::mem::take(&mut pending.kv_events), None);
            pending.pass_start_kv_published = true;
        }
    }

    /// Publishes a completed pass as one ordered transaction:
    /// admissions/accounting, KV/FPM, outputs, lifecycle, then metrics.
    pub(crate) async fn publish_pass<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        mut pending: PendingLivePass,
    ) {
        if !pending.admissions_published {
            self.publish_admissions(&pending.pass.admissions);
        }

        // Mid-pass command effects remain part of this pass transaction and
        // become visible at pass end, even when pass-start effects are already
        // published.
        let midpass_kv_events = self.captured_kv_events.drain();
        if pending.pass_start_kv_published {
            self.publish_router_effects(midpass_kv_events, pending.pass.fpm.take());
        } else {
            pending.kv_events.extend(midpass_kv_events);
            self.publish_router_effects(pending.kv_events, pending.pass.fpm.take());
        }

        #[cfg(debug_assertions)]
        {
            let completed_signals = pending
                .pass
                .output_signals
                .iter()
                .filter(|signal| signal.completed)
                .count();
            let accept_length =
                crate::scheduler::accept_length_sample(&pending.pass.output_signals);
            debug_assert_eq!(pending.pass.completed_requests, completed_signals);
            debug_assert_eq!(
                (
                    pending.pass.accept_length_output_tokens,
                    pending.pass.accept_length_decode_forwards,
                ),
                accept_length
            );
        }
        self.publish_outputs(core, pending.pass.output_signals);
        // Live completion/accept accounting is carried by the admission and
        // output channels; the scalar replay counters have no separate live
        // consumer, but are committed at this same boundary.
        self.record(PublishedEffect::Accounting);
        // vLLM may release request-owned blocks when its output receiver has
        // disappeared. Those cleanup events belong to this same boundary.
        self.publish_router_effects(self.captured_kv_events.drain(), None);
        self.publish_lifecycle(pending.pass.lifecycle_events).await;
        let metrics = core.pass_boundary_metrics(pending.pass.mocker_metrics);
        self.record(PublishedEffect::Metrics);
        let _ = self.metrics_tx.send(metrics);
    }

    pub(crate) async fn apply_command<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        envelope: SchedulerCommandEnvelope,
        allow_destination_admission: bool,
        now_ms: f64,
    ) {
        let SchedulerCommandEnvelope { command, reply } = envelope;
        let publish_midpass_metrics = matches!(&command, SchedulerCommand::CancelRequest { .. });
        let result = core.apply_live_command(command, allow_destination_admission, now_ms);
        match result {
            Ok(mut effects) => {
                let lifecycle_events = std::mem::take(&mut effects.lifecycle_events);
                if allow_destination_admission {
                    self.publish_router_effects(self.captured_kv_events.drain(), None);
                } else {
                    assert!(
                        lifecycle_events.is_empty(),
                        "mid-pass scheduler command produced lifecycle effects"
                    );
                }
                self.record(PublishedEffect::Ack);
                let _ = reply.send(Ok(effects));
                if allow_destination_admission {
                    self.publish_lifecycle(lifecycle_events).await;
                }
                if allow_destination_admission || publish_midpass_metrics {
                    self.record(PublishedEffect::Metrics);
                    let _ = self.metrics_tx.send(core.live_metrics());
                }
            }
            Err(error) => {
                self.record(PublishedEffect::Ack);
                let _ = reply.send(Err(error));
            }
        }
    }

    pub(crate) async fn retry_destinations<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        now_ms: f64,
    ) -> bool {
        let lifecycle_events = core.retry_live_destinations(now_ms);
        let kv_events = self.captured_kv_events.drain();
        let made_progress = !lifecycle_events.is_empty() || !kv_events.is_empty();
        self.publish_router_effects(kv_events, None);
        self.publish_lifecycle(lifecycle_events).await;
        self.record(PublishedEffect::Metrics);
        let _ = self.metrics_tx.send(core.live_metrics());
        made_progress
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) async fn advance_offload<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        now_ms: f64,
        allow_destination_admission: bool,
    ) -> bool {
        let effects = core.advance_live_offload(now_ms, allow_destination_admission);
        assert!(
            effects.kv_events.is_empty(),
            "live offload progress unexpectedly used offline KV capture"
        );
        if !allow_destination_admission {
            assert!(
                effects.lifecycle_events.is_empty(),
                "busy offload progress admitted a destination"
            );
        }
        let kv_events = self.captured_kv_events.drain();
        let made_progress = !kv_events.is_empty() || !effects.lifecycle_events.is_empty();
        self.publish_router_effects(kv_events, None);
        self.publish_lifecycle(effects.lifecycle_events).await;
        self.record(PublishedEffect::Metrics);
        let _ = self.metrics_tx.send(core.live_metrics());
        made_progress
    }

    fn publish_admissions(&self, admissions: &[AdmissionEvent]) {
        let Some(tx) = self.admission_tx.as_ref() else {
            return;
        };
        if !admissions.is_empty() {
            self.record(PublishedEffect::Admissions);
        }
        for admission in admissions {
            let _ = tx.send(admission.clone());
        }
    }

    fn publish_router_effects(
        &self,
        kv_events: Vec<DeferredKvPublish>,
        fpm: Option<crate::common::protocols::ForwardPassSnapshot>,
    ) {
        if !kv_events.is_empty() {
            self.record(PublishedEffect::Kv);
        }
        publish_deferred_kv_events(&self.kv_event_publishers, kv_events);
        if let Some(fpm) = fpm {
            self.record(PublishedEffect::Fpm);
            publish_deferred_fpm(&self.fpm_publisher, vec![fpm]);
        }
    }

    fn publish_outputs<C: LiveBoundaryCore>(&self, core: &mut C, signals: Vec<OutputSignal>) {
        let Some(tx) = self.output_tx.as_ref() else {
            return;
        };
        if signals.is_empty() {
            return;
        }
        self.record(PublishedEffect::Outputs);
        if let Err(error) = tx.send(signals) {
            core.output_delivery_failed(error.0);
        }
    }

    async fn publish_lifecycle(&self, events: Vec<SchedulerLifecycleEvent>) {
        if !events.is_empty() {
            self.record(PublishedEffect::Lifecycle);
        }
        for event in events {
            if self.lifecycle_tx.send(event).await.is_err() {
                break;
            }
        }
    }

    fn record(&self, effect: PublishedEffect) {
        #[cfg(test)]
        if let Some(log) = &self.publication_log {
            log.lock().unwrap().push(effect);
        }
        #[cfg(not(test))]
        let _ = effect;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::handoff::HandoffId;
    use crate::common::protocols::ForwardPassSnapshot;
    use crate::scheduler::{SchedulerCommandResult, capture_deferred_kv_publish_sink};
    use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};
    use std::sync::{Arc, Mutex};
    use uuid::Uuid;

    struct FakeCore {
        publishers: KvEventPublishers,
        command_effects: bool,
        midpass_kv_effects: bool,
    }

    impl FakeCore {
        fn publish_kv(&self, event_id: u64) {
            self.publishers
                .publish(
                    KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Cleared,
                        dp_rank: 0,
                    },
                    None,
                )
                .unwrap();
        }
    }

    impl LiveBoundaryCore for FakeCore {
        fn apply_live_command(
            &mut self,
            _command: SchedulerCommand,
            _allow_destination_admission: bool,
            _now_ms: f64,
        ) -> anyhow::Result<SchedulerCommandEffects> {
            let mut effects = SchedulerCommandEffects::new(SchedulerCommandResult::Applied);
            if self.command_effects || self.midpass_kv_effects {
                self.publish_kv(2);
            }
            if self.command_effects {
                effects
                    .lifecycle_events
                    .push(SchedulerLifecycleEvent::DestinationReserved {
                        handoff_id: HandoffId::from(Uuid::from_u128(2)),
                        request_id: Uuid::from_u128(3),
                        transferable_prompt_tokens: 4,
                    });
            }
            Ok(effects)
        }

        fn retry_live_destinations(&mut self, _now_ms: f64) -> Vec<SchedulerLifecycleEvent> {
            Vec::new()
        }

        fn live_metrics(&self) -> MockerMetrics {
            MockerMetrics::default()
        }

        fn pass_boundary_metrics(&self, pass_metrics: MockerMetrics) -> MockerMetrics {
            pass_metrics
        }

        fn output_delivery_failed(&mut self, _signals: Vec<OutputSignal>) {
            self.publish_kv(3);
        }
    }

    fn pass() -> EnginePassResult {
        let request_id = Uuid::from_u128(1);
        EnginePassResult {
            end_ms: 1.0,
            completed_requests: 1,
            output_signals: vec![OutputSignal {
                uuid: request_id,
                token_id: None,
                completed: true,
                rejected: false,
                handoff_delay_ms: None,
            }],
            admissions: vec![AdmissionEvent {
                uuid: request_id,
                reused_input_tokens: 0,
            }],
            lifecycle_events: vec![SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: HandoffId::from(Uuid::from_u128(2)),
                request_id,
                transferable_prompt_tokens: 4,
            }],
            mocker_metrics: MockerMetrics::default(),
            router_event_visibility: RouterEventVisibility::PassEnd,
            kv_events: Vec::new(),
            fpm: Some(ForwardPassSnapshot::default()),
            accept_length_output_tokens: 1,
            accept_length_decode_forwards: 1,
        }
    }

    fn publisher(
        output_tx: mpsc::UnboundedSender<Vec<OutputSignal>>,
        captured: DeferredKvPublishBuffer,
        log: Arc<Mutex<Vec<PublishedEffect>>>,
    ) -> LiveEffectsPublisher {
        let (admission_tx, _admission_rx) = mpsc::unbounded_channel();
        let (lifecycle_tx, _lifecycle_rx) = mpsc::channel(4);
        let (metrics_tx, _metrics_rx) = watch::channel(MockerMetrics::default());
        LiveEffectsPublisher::new(
            Some(output_tx),
            Some(admission_tx),
            lifecycle_tx,
            metrics_tx,
            KvEventPublishers::default(),
            FpmPublisher::default(),
            captured,
        )
        .with_publication_log(log)
    }

    #[tokio::test]
    async fn pass_effects_publish_once_in_boundary_order_and_isolate_midpass_ack() {
        let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
        let mut core = FakeCore {
            publishers: buffering_publishers,
            command_effects: false,
            midpass_kv_effects: false,
        };
        core.publish_kv(1);
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        drop(output_rx);
        let log = Arc::new(Mutex::new(Vec::new()));
        let publisher = publisher(output_tx, captured, log.clone());
        let mut pending = publisher.capture_pass(pass());
        publisher.publish_pass_start(&mut pending);

        let (reply, reply_rx) = tokio::sync::oneshot::channel();
        publisher
            .apply_command(
                &mut core,
                SchedulerCommandEnvelope {
                    command: SchedulerCommand::CancelSource {
                        handoff_id: HandoffId::from(Uuid::from_u128(2)),
                    },
                    reply,
                },
                false,
                1.0,
            )
            .await;
        assert_eq!(
            reply_rx.await.unwrap().unwrap().result,
            SchedulerCommandResult::Applied
        );
        assert_eq!(
            log.lock().unwrap().as_slice(),
            &[PublishedEffect::Admissions, PublishedEffect::Ack]
        );

        publisher.publish_pass(&mut core, pending).await;
        assert_eq!(
            log.lock().unwrap().as_slice(),
            &[
                PublishedEffect::Admissions,
                PublishedEffect::Ack,
                PublishedEffect::Kv,
                PublishedEffect::Fpm,
                PublishedEffect::Outputs,
                PublishedEffect::Accounting,
                PublishedEffect::Kv,
                PublishedEffect::Lifecycle,
                PublishedEffect::Metrics,
            ]
        );
    }

    #[tokio::test]
    async fn controlled_pass_start_router_effects_precede_midpass_ack_without_duplicates() {
        let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
        let mut core = FakeCore {
            publishers: buffering_publishers,
            command_effects: false,
            midpass_kv_effects: true,
        };
        core.publish_kv(1);
        let (output_tx, _output_rx) = mpsc::unbounded_channel();
        let log = Arc::new(Mutex::new(Vec::new()));
        let publisher = publisher(output_tx, captured, log.clone());
        let mut pass = pass();
        pass.router_event_visibility = RouterEventVisibility::PassStart;
        let mut pending = publisher.capture_pass(pass);

        publisher.publish_pass_start(&mut pending);
        let (reply, reply_rx) = tokio::sync::oneshot::channel();
        publisher
            .apply_command(
                &mut core,
                SchedulerCommandEnvelope {
                    command: SchedulerCommand::CancelSource {
                        handoff_id: HandoffId::from(Uuid::from_u128(2)),
                    },
                    reply,
                },
                false,
                1.0,
            )
            .await;
        assert_eq!(
            reply_rx.await.unwrap().unwrap().result,
            SchedulerCommandResult::Applied
        );
        assert_eq!(
            log.lock().unwrap().as_slice(),
            &[
                PublishedEffect::Admissions,
                PublishedEffect::Kv,
                PublishedEffect::Ack,
            ]
        );

        publisher.publish_pass(&mut core, pending).await;
        assert_eq!(
            log.lock().unwrap().as_slice(),
            &[
                PublishedEffect::Admissions,
                PublishedEffect::Kv,
                PublishedEffect::Ack,
                PublishedEffect::Kv,
                PublishedEffect::Fpm,
                PublishedEffect::Outputs,
                PublishedEffect::Accounting,
                PublishedEffect::Lifecycle,
                PublishedEffect::Metrics,
            ]
        );
    }

    #[tokio::test]
    async fn command_effects_publish_kv_before_ack_then_lifecycle_and_metrics() {
        let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
        let mut core = FakeCore {
            publishers: buffering_publishers,
            command_effects: true,
            midpass_kv_effects: false,
        };
        let (output_tx, _output_rx) = mpsc::unbounded_channel();
        let log = Arc::new(Mutex::new(Vec::new()));
        let publisher = publisher(output_tx, captured, log.clone());
        let (reply, reply_rx) = tokio::sync::oneshot::channel();

        publisher
            .apply_command(
                &mut core,
                SchedulerCommandEnvelope {
                    command: SchedulerCommand::ActivateDestination {
                        handoff_id: HandoffId::from(Uuid::from_u128(2)),
                    },
                    reply,
                },
                true,
                1.0,
            )
            .await;

        assert_eq!(
            reply_rx.await.unwrap().unwrap().result,
            SchedulerCommandResult::Applied
        );
        assert_eq!(
            log.lock().unwrap().as_slice(),
            &[
                PublishedEffect::Kv,
                PublishedEffect::Ack,
                PublishedEffect::Lifecycle,
                PublishedEffect::Metrics,
            ]
        );
    }
}
