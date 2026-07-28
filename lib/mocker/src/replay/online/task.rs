// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::DirectRequest;
use crate::live::LiveEngine;
use crate::replay::ReplayTerminalStatus;

use super::ReplayRouter;
use super::recorder::{RecorderSender, TerminalObservation};
use super::state::{SharedLiveRuntimeStats, WorkloadDispatchState, now_ms, request_uuid};

#[derive(Clone)]
pub(super) struct RequestTaskContext {
    pub(super) engines: Arc<[LiveEngine]>,
    pub(super) router: Arc<ReplayRouter>,
    pub(super) recorder: RecorderSender,
    pub(super) stats: Arc<SharedLiveRuntimeStats>,
    pub(super) workload: Option<Arc<WorkloadDispatchState>>,
    pub(super) cancel: CancellationToken,
    pub(super) start: Instant,
}

/// Releases a `WorkloadDriver` cap slot on drop if `mark_completed` was not called.
pub(super) struct InFlightGuard {
    dispatch: Arc<WorkloadDispatchState>,
    uuid: Uuid,
    completed: bool,
}

impl InFlightGuard {
    pub(super) fn new(dispatch: Arc<WorkloadDispatchState>, uuid: Uuid) -> Self {
        Self {
            dispatch,
            uuid,
            completed: false,
        }
    }

    pub(super) fn mark_completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if let Ok(mut driver) = self.dispatch.driver.lock() {
            driver.release_cap_slot(self.uuid, now_ms(self.dispatch.start));
        }
        self.dispatch.wakeup.notify_waiters();
    }
}

pub(super) async fn wait_for_workload_progress<F>(
    next_ready_ms: Option<f64>,
    start: Instant,
    mut wake: Pin<&mut F>,
) where
    F: Future<Output = ()>,
{
    match next_ready_ms {
        Some(next_ready_ms) => {
            let deadline = start + tokio::time::Duration::from_secs_f64(next_ready_ms / 1000.0);
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {}
                _ = wake.as_mut() => {}
            }
        }
        None => {
            wake.as_mut().await;
        }
    }
}

pub(super) async fn run_request_task(
    ctx: RequestTaskContext,
    request: DirectRequest,
    mut guard: Option<InFlightGuard>,
) -> Result<()> {
    if ctx.cancel.is_cancelled() {
        bail!("online replay cancelled");
    }
    let uuid = request_uuid(&request)?;
    let worker_idx = ctx
        .router
        .select_worker(&request, ctx.engines.len())
        .await?;
    if ctx.cancel.is_cancelled() {
        bail!("online replay cancelled");
    }
    ensure!(
        worker_idx < ctx.engines.len(),
        "online replay selected unknown worker index {worker_idx}"
    );

    let mut live_request = ctx.engines[worker_idx]
        .submit(request)
        .await
        .with_context(|| {
            format!("online replay failed to submit request {uuid} to worker {worker_idx}")
        })?;
    if ctx.cancel.is_cancelled() {
        bail!("online replay cancelled");
    }
    ctx.stats.record_dispatch(worker_idx);
    ctx.recorder.record_decode_assignment(uuid, worker_idx)?;

    let mut first_token_seen = false;
    let mut token_times_ms = Vec::new();
    let (terminal_time_ms, status) = loop {
        let observed = live_request.recv_observed().await.ok_or_else(|| {
            anyhow::anyhow!(
                "online replay request {uuid} output stream closed before terminal delivery"
            )
        })?;
        let output = observed.event;
        ensure!(
            output.uuid == uuid,
            "online replay request {uuid} received output for {}",
            output.uuid
        );

        let output_time_ms = observed
            .observed_at
            .saturating_duration_since(ctx.start)
            .as_secs_f64()
            * 1000.0;
        if !output.rejected && output.token_id.is_some() {
            token_times_ms.push(output_time_ms);
            if !first_token_seen {
                first_token_seen = true;
                let marked = ctx.router.on_first_token(uuid).await?;
                if marked {
                    ctx.stats.record_prefill_marked();
                }
            }
        }
        if output.completed {
            let status = if output.rejected {
                ReplayTerminalStatus::Rejected
            } else {
                ReplayTerminalStatus::Completed
            };
            break (output_time_ms, status);
        }
    };

    ctx.recorder.record_terminal(TerminalObservation {
        uuid,
        token_times_ms,
        terminal_time_ms,
        status,
    })?;
    let freed = ctx.router.on_complete(uuid).await?;
    if freed {
        ctx.stats.record_freed();
    }
    ctx.stats.record_completion();

    if let Some(workload) = ctx.workload.as_ref() {
        let completion_ms = now_ms(workload.start);
        workload
            .driver
            .lock()
            .unwrap()
            .on_complete(uuid, completion_ms)?;
        workload.wakeup.notify_waiters();
        if let Some(guard) = guard.as_mut() {
            guard.mark_completed();
        }
    }
    Ok(())
}
