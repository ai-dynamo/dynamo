// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

#[cfg(all(test, feature = "kvbm-offload"))]
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::scheduler::{
    AdmissionEvent, LiveBoundaryCore, LivePassExecution, LiveSchedulerState,
    SchedulerCancellationEnvelope, SchedulerCommand, SchedulerCommandEffects,
    SchedulerCommandEnvelope, SchedulerHandle, SchedulerLifecycleEvent, SchedulerOutputSender,
    spawn_live_scheduler,
};

use super::core::VllmCore;

#[derive(Clone, Default, Debug, PartialEq)]
pub struct MockerMetrics {
    pub dp_rank: dynamo_kv_router::protocols::DpRank,
    pub active_decode_blocks: u64,
    pub total_blocks: u64,
    pub gpu_cache_usage_perc: f64,
    pub running_requests: u64,
    pub waiting_requests: u64,
    pub vllm_preemptions_total: u64,
    pub sglang_cache_hit_tokens: u64,
    pub sglang_cache_total_tokens: u64,
}

impl MockerMetrics {
    pub fn new(
        dp_rank: dynamo_kv_router::protocols::DpRank,
        active_decode_blocks: u64,
        total_blocks: u64,
    ) -> Self {
        Self::from_parts(dp_rank, active_decode_blocks, total_blocks, 0, 0, 0, 0, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        dp_rank: dynamo_kv_router::protocols::DpRank,
        active_decode_blocks: u64,
        total_blocks: u64,
        running_requests: u64,
        waiting_requests: u64,
        vllm_preemptions_total: u64,
        sglang_cache_hit_tokens: u64,
        sglang_cache_total_tokens: u64,
    ) -> Self {
        let gpu_cache_usage_perc = if total_blocks == 0 {
            0.0
        } else {
            active_decode_blocks as f64 / total_blocks as f64
        };
        Self {
            dp_rank,
            active_decode_blocks,
            total_blocks,
            gpu_cache_usage_perc,
            running_requests,
            waiting_requests,
            vllm_preemptions_total,
            sglang_cache_hit_tokens,
            sglang_cache_total_tokens,
        }
    }
}

#[derive(Clone)]
pub struct Scheduler {
    inner: LiveSchedulerState,
}

impl Scheduler {
    pub fn new(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        fpm_publisher: FpmPublisher,
    ) -> Self {
        Self::new_with_output_sender(
            args,
            dp_rank,
            output_tx.map(SchedulerOutputSender::from),
            kv_event_publishers,
            cancellation_token,
            fpm_publisher,
        )
    }

    pub(crate) fn new_with_output_sender(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<SchedulerOutputSender>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        fpm_publisher: FpmPublisher,
    ) -> Self {
        Self::new_internal(
            args,
            dp_rank,
            output_tx,
            kv_event_publishers,
            cancellation_token,
            None,
            fpm_publisher,
        )
    }

    pub(crate) fn new_with_admission(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
        fpm_publisher: FpmPublisher,
    ) -> Self {
        Self::new_internal(
            args,
            dp_rank,
            output_tx.map(SchedulerOutputSender::from),
            kv_event_publishers,
            cancellation_token,
            admission_tx,
            fpm_publisher,
        )
    }

    fn new_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<SchedulerOutputSender>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
        fpm_publisher: FpmPublisher,
    ) -> Self {
        Self {
            inner: spawn_live_scheduler(
                args,
                dp_rank,
                output_tx,
                kv_event_publishers,
                cancellation_token,
                admission_tx,
                fpm_publisher,
                VllmCore::new_with_sink,
            ),
        }
    }
}

impl SchedulerHandle for Scheduler {
    fn receive(&self, request: DirectRequest) {
        self.inner.receive(request);
    }

    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest> {
        self.inner.request_sender()
    }

    fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics> {
        self.inner.metrics_receiver()
    }

    fn command_sender(&self) -> mpsc::Sender<SchedulerCommandEnvelope> {
        self.inner.command_sender()
    }

    fn cancellation_sender(&self) -> mpsc::Sender<SchedulerCancellationEnvelope> {
        self.inner.cancellation_sender()
    }

    fn take_lifecycle_receiver(&mut self) -> Option<mpsc::Receiver<SchedulerLifecycleEvent>> {
        self.inner.take_lifecycle_receiver()
    }
}

impl LiveBoundaryCore for VllmCore {
    fn initialize_live(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            #[cfg(feature = "kvbm-offload")]
            if let Err(error) = self.init_offload_live().await {
                tracing::error!("kvbm-offload live init failed: {error}");
            }
        })
    }

    fn live_is_empty(&self) -> bool {
        self.is_empty()
    }

    fn receive_live_request(&mut self, request: DirectRequest) {
        self.receive(request);
    }

    fn apply_live_command(
        &mut self,
        command: SchedulerCommand,
        allow_destination_admission: bool,
        now_ms: f64,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        self.apply_command_effects_at(command, allow_destination_admission, Some(now_ms))
    }

    fn retry_live_destinations(&mut self, now_ms: f64) -> Vec<SchedulerLifecycleEvent> {
        self.retry_pending_destinations_at(Some(now_ms))
    }

    fn live_metrics(&self) -> MockerMetrics {
        self.mocker_metrics()
    }

    fn pass_boundary_metrics(&self, _pass_metrics: MockerMetrics) -> MockerMetrics {
        self.mocker_metrics()
    }

    fn live_internal_deadline_ms(&self) -> Option<f64> {
        #[cfg(feature = "kvbm-offload")]
        {
            self.earliest_offload_deadline()
        }
        #[cfg(not(feature = "kvbm-offload"))]
        {
            None
        }
    }

    fn execute_live_pass(&mut self, scheduler_start: &Instant) -> LivePassExecution {
        // Wall-clock elapsed time drives the PS bandwidth models across
        // vLLM passes; SGLang reports a duration directly from each pass.
        let now_ms = scheduler_start.elapsed().as_secs_f64() * 1000.0;
        let pass = self.execute_pass_internal(None, now_ms, None);
        let duration = std::time::Duration::from_secs_f64((pass.end_ms - now_ms).max(0.0) / 1000.0);
        LivePassExecution { pass, duration }
    }

    fn output_delivery_failed(&mut self, signals: Vec<OutputSignal>) {
        for signal in signals {
            self.drop_request(signal.uuid);
        }
    }

    #[cfg(feature = "kvbm-offload")]
    fn advance_live_offload(
        &mut self,
        now_ms: f64,
        allow_destination_admission: bool,
    ) -> crate::scheduler::OffloadTickEffects {
        if allow_destination_admission {
            self.tick_offload_only(now_ms)
        } else {
            self.tick_offload_transport_only(now_ms)
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "kvbm-offload")]
    use super::*;

    #[cfg(feature = "kvbm-offload")]
    use crate::common::protocols::KvCacheEventSink;
    #[cfg(feature = "kvbm-offload")]
    use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, StorageTier};
    #[cfg(feature = "kvbm-offload")]
    use std::sync::{Arc, Mutex};

    #[cfg(feature = "kvbm-offload")]
    #[derive(Default)]
    struct CapturingKvSink {
        events: Mutex<Vec<(StorageTier, KvCacheEvent)>>,
    }

    #[cfg(feature = "kvbm-offload")]
    impl KvCacheEventSink for CapturingKvSink {
        fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
            self.events
                .lock()
                .unwrap()
                .push((StorageTier::Device, event));
            Ok(())
        }

        fn publish_with_storage_tier(
            &self,
            event: KvCacheEvent,
            storage_tier: StorageTier,
        ) -> anyhow::Result<()> {
            self.events.lock().unwrap().push((storage_tier, event));
            Ok(())
        }
    }

    #[cfg(feature = "kvbm-offload")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn idle_destination_reservation_wakes_at_offload_deadline() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(2)
            .block_size(4)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_prefix_caching(true)
            .worker_type(crate::common::protocols::WorkerType::Decode)
            .speedup_ratio(1000.0)
            .kv_bytes_per_token(Some(250_000))
            .num_g2_blocks(Some(4))
            .bandwidth_g1_to_g2_gbps(Some(1.0))
            .build()
            .unwrap();
        let sink = Arc::new(CapturingKvSink::default());
        let publishers = KvEventPublishers::new(Some(sink.clone()), None);
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let mut scheduler = Scheduler::new(
            args,
            0,
            Some(output_tx),
            publishers,
            None,
            FpmPublisher::default(),
        );
        let mut lifecycle_rx = scheduler.take_lifecycle_receiver().unwrap();

        scheduler.receive(DirectRequest {
            tokens: vec![1; 4],
            max_output_tokens: 1,
            uuid: Some(uuid::Uuid::from_u128(1)),
            ..Default::default()
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let seed = output_rx
                    .recv()
                    .await
                    .expect("output channel should stay open");
                if seed.iter().any(|signal| signal.completed) {
                    break;
                }
            }
        })
        .await
        .expect("seed request should complete");
        sink.events.lock().unwrap().clear();

        let source_handoff_id = crate::common::handoff::HandoffId::from(uuid::Uuid::from_u128(10));
        let source_request_id = uuid::Uuid::from_u128(11);
        let (source_reply, source_reply_rx) = tokio::sync::oneshot::channel();
        scheduler
            .command_sender()
            .send(SchedulerCommandEnvelope {
                command: SchedulerCommand::SubmitHandoffPrefill {
                    handoff_id: source_handoff_id,
                    request: DirectRequest {
                        tokens: vec![3; 3],
                        max_output_tokens: 1,
                        uuid: Some(source_request_id),
                        ..Default::default()
                    },
                },
                reply: source_reply,
            })
            .await
            .unwrap();
        let submitted = source_reply_rx.await.unwrap().unwrap();
        assert!(matches!(
            submitted.result,
            crate::scheduler::SchedulerCommandResult::Submitted(observed)
                if observed == source_request_id
        ));
        let held = tokio::time::timeout(Duration::from_secs(1), lifecycle_rx.recv())
            .await
            .expect("source hold should complete")
            .expect("lifecycle channel should stay open");
        assert!(matches!(
            held,
            SchedulerLifecycleEvent::SourceHeld {
                handoff_id: observed,
                request_id: observed_request,
                ..
            } if observed == source_handoff_id && observed_request == source_request_id
        ));

        let handoff_id = crate::common::handoff::HandoffId::from(uuid::Uuid::from_u128(2));
        let request_id = uuid::Uuid::from_u128(3);
        let (reply, reply_rx) = tokio::sync::oneshot::channel();
        scheduler
            .command_sender()
            .send(SchedulerCommandEnvelope {
                command: SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: DirectRequest {
                        tokens: vec![2; 4],
                        max_output_tokens: 1,
                        uuid: Some(request_id),
                        ..Default::default()
                    },
                },
                reply,
            })
            .await
            .unwrap();
        let accepted = reply_rx.await.unwrap().unwrap();
        assert!(matches!(
            accepted.result,
            crate::scheduler::SchedulerCommandResult::DestinationAccepted {
                request_id: observed,
            } if observed == request_id
        ));
        assert!(accepted.lifecycle_events.is_empty());

        let reserved = tokio::time::timeout(Duration::from_secs(1), lifecycle_rx.recv())
            .await
            .expect("idle offload deadline should wake the scheduler")
            .expect("lifecycle channel should stay open");
        assert!(matches!(
            reserved,
            SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: observed,
                request_id: observed_request,
                ..
            } if observed == handoff_id && observed_request == request_id
        ));
        let host_stores = || {
            sink.events
                .lock()
                .unwrap()
                .iter()
                .filter(|(tier, event)| {
                    *tier == StorageTier::HostPinned
                        && matches!(&event.data, KvCacheEventData::Stored(_))
                })
                .count()
        };
        assert_eq!(host_stores(), 1);

        let (barrier_reply, barrier_rx) = tokio::sync::oneshot::channel();
        scheduler
            .command_sender()
            .send(SchedulerCommandEnvelope {
                command: SchedulerCommand::CancelDestination {
                    handoff_id: crate::common::handoff::HandoffId::from(uuid::Uuid::from_u128(99)),
                },
                reply: barrier_reply,
            })
            .await
            .unwrap();
        let barrier = barrier_rx.await.unwrap().unwrap();
        assert_eq!(
            barrier.result,
            crate::scheduler::SchedulerCommandResult::Noop
        );
        assert!(lifecycle_rx.try_recv().is_err());
        assert_eq!(host_stores(), 1);

        for command in [
            SchedulerCommand::CancelDestination { handoff_id },
            SchedulerCommand::CancelSource {
                handoff_id: source_handoff_id,
            },
        ] {
            let (reply, reply_rx) = tokio::sync::oneshot::channel();
            scheduler
                .command_sender()
                .send(SchedulerCommandEnvelope { command, reply })
                .await
                .unwrap();
            let cleanup = reply_rx.await.unwrap().unwrap();
            assert_eq!(
                cleanup.result,
                crate::scheduler::SchedulerCommandResult::Applied
            );
        }
    }
}
