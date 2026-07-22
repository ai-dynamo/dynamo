// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::scheduler::{
    AdmissionEvent, LiveBoundaryCore, LivePassExecution, LiveSchedulerState, MockerMetrics,
    SchedulerCommand, SchedulerCommandEffects, SchedulerCommandEnvelope, SchedulerHandle,
    SchedulerLifecycleEvent, spawn_live_scheduler,
};

use super::core::SglangCore;

#[derive(Clone)]
pub struct SglangScheduler {
    inner: LiveSchedulerState,
}
impl SglangScheduler {
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
        Self {
            inner: spawn_live_scheduler(
                args,
                dp_rank,
                output_tx,
                kv_event_publishers,
                cancellation_token,
                admission_tx,
                fpm_publisher,
                SglangCore::new_with_sink,
            ),
        }
    }
}

impl SchedulerHandle for SglangScheduler {
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

    fn take_lifecycle_receiver(&mut self) -> Option<mpsc::Receiver<SchedulerLifecycleEvent>> {
        self.inner.take_lifecycle_receiver()
    }
}

impl LiveBoundaryCore for SglangCore {
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
        _now_ms: f64,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        self.apply_command_effects(command, allow_destination_admission)
    }

    fn retry_live_destinations(&mut self, _now_ms: f64) -> Vec<SchedulerLifecycleEvent> {
        self.retry_pending_destinations()
    }

    fn live_metrics(&self) -> MockerMetrics {
        self.mocker_metrics()
    }

    fn pass_boundary_metrics(&self, mut pass_metrics: MockerMetrics) -> MockerMetrics {
        let current = self.mocker_metrics();
        pass_metrics.active_decode_blocks = current.active_decode_blocks;
        pass_metrics.gpu_cache_usage_perc = current.gpu_cache_usage_perc;
        pass_metrics.running_requests = current.running_requests;
        pass_metrics.waiting_requests = current.waiting_requests;
        pass_metrics
    }

    fn execute_live_pass(&mut self, _scheduler_start: &Instant) -> LivePassExecution {
        let pass = self.execute_pass_internal(None, 0.0);
        let duration = std::time::Duration::from_secs_f64(pass.end_ms / 1000.0);
        LivePassExecution { pass, duration }
    }
}
