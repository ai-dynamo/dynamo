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

use super::recorder::{RecorderSender, TerminalObservation};
use super::state::{
    SharedLiveRuntimeStats, WorkloadDispatchState, deadline_from_ms, now_ms, request_uuid,
};
use super::{ReplayPlacement, ReplayRouter};

#[derive(Clone)]
pub(super) struct RequestTaskContext {
    pub(super) engines: Arc<[LiveEngine]>,
    pub(super) num_workers: usize,
    pub(super) dp_size: usize,
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
) -> Result<()>
where
    F: Future<Output = ()>,
{
    match next_ready_ms {
        Some(next_ready_ms) => {
            // `deadline_from_ms`, not `Duration::from_secs_f64`: driver ready times are
            // derived from caller-supplied trace timing and never pass through
            // `normalize_trace_requests`, and `from_secs_f64` panics on NaN/infinite/negative.
            let deadline = deadline_from_ms(start, next_ready_ms)?;
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {}
                _ = wake.as_mut() => {}
            }
        }
        None => {
            wake.as_mut().await;
        }
    }
    Ok(())
}

/// One request with its routing decision already made.
///
/// Routing is resolved by [`route_request`] on the submission path and carried into
/// the spawned task, so a task never makes a routing decision itself.
pub(super) struct RoutedRequest {
    request: DirectRequest,
    uuid: Uuid,
    worker_idx: usize,
    dp_rank: usize,
    engine_idx: usize,
}

/// Resolve one request's placement.
///
/// This MUST stay on the submission path, awaited in trace order, and must not move
/// back inside [`run_request_task`]. Request tasks are spawned onto a multi-thread
/// runtime, so two rows with coincident or near-coincident arrivals would otherwise
/// race into the router with no ordering guarantee: round-robin's shared
/// `AtomicUsize` and the KV router's shared scheduler would both hand out
/// assignments in OS-scheduler order rather than trace order, making the resulting
/// cache-hit structure depend on thread timing. Resolving here makes the routing
/// decision a function of the trace; only the per-request work after it stays
/// concurrent.
///
/// This does not make an online KV run fully reproducible -- `on_first_token` and
/// `on_complete` still mutate shared scheduler state from concurrent tasks at
/// wall-clock-dependent instants, which is inherent to a live replay. Offline
/// replay is the deterministic path. It does remove the dispatch-order race, which
/// is the part that has no business being nondeterministic.
pub(super) async fn route_request(
    ctx: &RequestTaskContext,
    request: DirectRequest,
) -> Result<RoutedRequest> {
    if ctx.cancel.is_cancelled() {
        bail!("online replay cancelled");
    }
    let uuid = request_uuid(&request)?;
    let ReplayPlacement {
        worker_idx,
        dp_rank,
    } = ctx
        .router
        .select_worker(&request, ctx.num_workers, ctx.dp_size)
        .await?;
    if ctx.cancel.is_cancelled() {
        bail!("online replay cancelled");
    }
    ensure!(
        worker_idx < ctx.num_workers,
        "online replay selected unknown worker index {worker_idx}"
    );
    ensure!(
        dp_rank < ctx.dp_size,
        "online replay selected unknown DP rank {dp_rank} for worker {worker_idx}"
    );
    let engine_idx = worker_idx
        .checked_mul(ctx.dp_size)
        .and_then(|base| base.checked_add(dp_rank))
        .ok_or_else(|| anyhow::anyhow!("online replay rank-handle index overflow"))?;
    ensure!(
        engine_idx < ctx.engines.len(),
        "online replay has no rank handle for worker {worker_idx}, DP rank {dp_rank}"
    );
    // Recorded here, not after `submit`, so `dispatch_history` is the routing
    // sequence in trace order rather than whichever task won the race to submit.
    ctx.stats.record_dispatch(worker_idx);
    ctx.recorder.record_decode_assignment(uuid, worker_idx)?;
    Ok(RoutedRequest {
        request,
        uuid,
        worker_idx,
        dp_rank,
        engine_idx,
    })
}

pub(super) async fn run_request_task(
    ctx: RequestTaskContext,
    routed: RoutedRequest,
    mut guard: Option<InFlightGuard>,
) -> Result<()> {
    let RoutedRequest {
        request,
        uuid,
        worker_idx,
        dp_rank,
        engine_idx,
    } = routed;
    if ctx.cancel.is_cancelled() {
        bail!("online replay cancelled");
    }

    let mut live_request = ctx.engines[engine_idx]
        .submit(request)
        .await
        .with_context(|| {
            format!(
                "online replay failed to submit request {uuid} to worker {worker_idx}, DP rank {dp_rank}"
            )
        })?;
    if ctx.cancel.is_cancelled() {
        bail!("online replay cancelled");
    }

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
            .on_terminal(uuid, completion_ms, status)?;
        workload.wakeup.notify_waiters();
        if let Some(guard) = guard.as_mut() {
            guard.mark_completed();
        }
    }
    Ok(())
}
