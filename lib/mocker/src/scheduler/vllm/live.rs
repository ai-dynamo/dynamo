// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{DirectRequest, KvEventPublishers, MockEngineArgs, OutputSignal};
use crate::common::utils::sleep_until_precise;
use crate::scheduler::{
    AdmissionEvent, RouterEventVisibility, SchedulerHandle, capture_deferred_kv_publish_sink,
    publish_deferred_kv_events,
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
        output_tx: Option<mpsc::UnboundedSender<OutputSignal>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
    ) -> Self {
        Self::new_internal(
            args,
            dp_rank,
            output_tx,
            kv_event_publishers,
            cancellation_token,
            None,
        )
    }

    pub(crate) fn new_with_admission(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<OutputSignal>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
    ) -> Self {
        Self::new_internal(
            args,
            dp_rank,
            output_tx,
            kv_event_publishers,
            cancellation_token,
            admission_tx,
        )
    }

    fn new_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<OutputSignal>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
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
            let mut core = VllmCore::new_with_sink(args, dp_rank, buffering_publishers);

            loop {
                if receive_requests(&mut core, &mut request_rx, &cancel_token_clone)
                    .await
                    .is_none()
                {
                    break;
                }

                let iteration_start = Instant::now();
                let pass = core.execute_pass_internal(None, 0.0, admission_tx.as_ref());
                let total_time = std::time::Duration::from_secs_f64(pass.end_ms / 1000.0);
                if pass.router_event_visibility == RouterEventVisibility::PassStart {
                    publish_deferred_kv_events(&kv_event_publishers, deferred_kv_events.drain());
                }
                if total_time > std::time::Duration::ZERO {
                    sleep_until_precise(iteration_start + total_time).await;
                }
                if pass.router_event_visibility == RouterEventVisibility::PassEnd {
                    publish_deferred_kv_events(&kv_event_publishers, deferred_kv_events.drain());
                }
                flush_output_signals(&mut core, &output_tx, &pass.output_signals);
                publish_deferred_kv_events(&kv_event_publishers, deferred_kv_events.drain());
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
    output_tx: &Option<mpsc::UnboundedSender<OutputSignal>>,
    output_signals: &[OutputSignal],
) {
    let Some(tx) = output_tx.as_ref() else {
        return;
    };

    for signal in output_signals {
        if tx.send(signal.clone()).is_ok() {
            continue;
        }
        core.drop_request(signal.uuid);
    }
}
