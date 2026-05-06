// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::common::utils::sleep_until_precise;
use crate::scheduler::{
    AdmissionEvent, DeferredFpmBuffer, RouterEventVisibility, SchedulerHandle,
    capture_deferred_kv_publish_sink, publish_deferred_fpm, publish_deferred_kv_events,
};

use super::core::VllmCore;

#[derive(Clone, Default, Debug)]
pub struct MockerMetrics {
    pub dp_rank: dynamo_kv_router::protocols::DpRank,
    pub active_decode_blocks: u64,
    pub total_blocks: u64,
    pub gpu_cache_usage_perc: f64,
}

impl MockerMetrics {
    pub fn new(
        dp_rank: dynamo_kv_router::protocols::DpRank,
        active_decode_blocks: u64,
        total_blocks: u64,
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
        }
    }
}

#[derive(Clone)]
pub struct Scheduler {
    request_tx: mpsc::UnboundedSender<DirectRequest>,
    metrics_rx: tokio::sync::watch::Receiver<MockerMetrics>,
    _cancel_guard: Arc<CancelGuard>,
}

struct CancelGuard(CancellationToken);

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
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
            output_tx,
            kv_event_publishers,
            cancellation_token,
            admission_tx,
            fpm_publisher,
        )
    }

    fn new_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
        fpm_publisher: FpmPublisher,
    ) -> Self {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<DirectRequest>();
        let total_blocks = args.num_gpu_blocks as u64;
        let initial_metrics = MockerMetrics::new(dp_rank, 0, total_blocks);
        let (metrics_tx, metrics_rx) = tokio::sync::watch::channel(initial_metrics);

        let cancel_token = cancellation_token.unwrap_or_default();
        let cancel_token_clone = cancel_token.clone();
        let cancel_guard = Arc::new(CancelGuard(cancel_token));

        tokio::spawn(async move {
            let (deferred_kv_events, buffering_publishers) =
                capture_deferred_kv_publish_sink(kv_event_publishers.raw_enabled());
            let deferred_fpm = DeferredFpmBuffer::default();
            let mut core = VllmCore::new_with_sink(args, dp_rank, buffering_publishers);
            #[cfg(feature = "kvbm-offload")]
            if let Err(e) = core.init_offload_live().await {
                tracing::error!("kvbm-offload live init failed: {e}");
            }
            // Wall-clock origin for this scheduler's simulated time. Drives
            // `engine.tick(now_ms)` so the PS bandwidth models advance
            // in real time across passes.
            let scheduler_start = Instant::now();

            loop {
                if receive_requests(&mut core, &mut request_rx, &cancel_token_clone)
                    .await
                    .is_none()
                {
                    break;
                }

                let iteration_start = Instant::now();
                let now_ms = scheduler_start.elapsed().as_secs_f64() * 1000.0;
                let pass = core.execute_pass_internal(None, now_ms, admission_tx.as_ref());
                let total_time =
                    std::time::Duration::from_secs_f64((pass.end_ms - now_ms).max(0.0) / 1000.0);
                if let Some(fpm) = pass.fpm {
                    deferred_fpm.push(fpm);
                }
                if pass.router_event_visibility == RouterEventVisibility::PassStart {
                    publish_deferred_kv_events(&kv_event_publishers, deferred_kv_events.drain());
                    publish_deferred_fpm(&fpm_publisher, deferred_fpm.drain());
                }
                if total_time > std::time::Duration::ZERO {
                    sleep_until_precise(iteration_start + total_time).await;
                }
                if pass.router_event_visibility == RouterEventVisibility::PassEnd {
                    publish_deferred_kv_events(&kv_event_publishers, deferred_kv_events.drain());
                    publish_deferred_fpm(&fpm_publisher, deferred_fpm.drain());
                }
                flush_output_signals(&mut core, &output_tx, pass.output_signals);
                publish_deferred_kv_events(&kv_event_publishers, deferred_kv_events.drain());
                publish_deferred_fpm(&fpm_publisher, deferred_fpm.drain());
                let _ = metrics_tx.send(MockerMetrics::new(
                    dp_rank,
                    core.kv_manager.num_active_blocks() as u64,
                    total_blocks,
                ));
            }
        });

        Self {
            request_tx,
            metrics_rx,
            _cancel_guard: cancel_guard,
        }
    }
}

impl SchedulerHandle for Scheduler {
    fn receive(&self, request: DirectRequest) {
        let _ = self.request_tx.send(request);
    }

    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest> {
        self.request_tx.clone()
    }

    fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics> {
        self.metrics_rx.clone()
    }
}

async fn receive_requests(
    core: &mut VllmCore,
    request_rx: &mut mpsc::UnboundedReceiver<DirectRequest>,
    cancel_token: &CancellationToken,
) -> Option<()> {
    if cancel_token.is_cancelled() {
        return None;
    }

    if core.is_empty() {
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => return None,
            result = request_rx.recv() => {
                let request = result?;
                core.receive(request);
            }
        }
    }

    while let Ok(request) = request_rx.try_recv() {
        core.receive(request);
    }

    Some(())
}

fn flush_output_signals(
    core: &mut VllmCore,
    output_tx: &Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
    output_signals: Vec<OutputSignal>,
) {
    let Some(tx) = output_tx.as_ref() else {
        return;
    };

    if output_signals.is_empty() {
        return;
    }

    if let Err(error) = tx.send(output_signals) {
        for signal in error.0 {
            core.drop_request(signal.uuid);
        }
    }
}
