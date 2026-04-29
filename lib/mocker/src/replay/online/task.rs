// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use tokio::sync::mpsc;
use tokio::time::Instant;
use uuid::Uuid;

use crate::common::protocols::DirectRequest;

use super::ReplayRouter;
use super::state::{
    RequestRegistry, RequestState, SharedLiveRuntimeStats, WorkloadDispatchState, now_ms,
    request_uuid,
};

#[derive(Clone)]
pub(super) struct RequestTaskContext {
    pub(super) senders: Arc<[mpsc::UnboundedSender<DirectRequest>]>,
    pub(super) router: Arc<ReplayRouter>,
    pub(super) requests: RequestRegistry,
    pub(super) stats: Arc<SharedLiveRuntimeStats>,
    pub(super) workload: Option<Arc<WorkloadDispatchState>>,
}

/// Releases a `WorkloadDriver` cap slot on drop if `mark_completed` was not called.
/// Preserves the drop-safety of the old `OwnedSemaphorePermit` so a cancelled or
/// panicking request task can't leak capacity.
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
            driver.release_cap_slot(self.uuid);
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
    let uuid = request_uuid(&request)?;

    let worker_idx = ctx
        .router
        .select_worker(&request, ctx.senders.len())
        .await?;
    if worker_idx >= ctx.senders.len() {
        bail!("online replay selected unknown worker index {worker_idx}");
    }

    let state = Arc::new(RequestState::default());
    ctx.requests.insert(uuid, Arc::clone(&state));
    if let Err(error) = ctx.senders[worker_idx].send(request) {
        ctx.requests.remove(&uuid);
        return Err(anyhow!(
            "online replay failed to dispatch request to worker {worker_idx}: {error}"
        ));
    }

    ctx.stats.record_dispatch(worker_idx);
    state.wait_for_completion().await;
    ctx.stats.record_completion();
    ctx.requests.remove(&uuid);
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
