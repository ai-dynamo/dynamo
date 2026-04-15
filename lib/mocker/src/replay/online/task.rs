// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::time::Instant;

use crate::common::protocols::DirectRequest;

use super::ReplayRouter;
use super::state::{
    LiveReplayMode, RequestRegistry, RequestState, SharedLiveRuntimeStats, WorkloadDispatchState,
    now_ms, request_uuid,
};

#[derive(Clone)]
pub(super) struct RequestTaskContext {
    pub(super) senders: Arc<[mpsc::UnboundedSender<DirectRequest>]>,
    pub(super) router: Arc<ReplayRouter>,
    pub(super) requests: RequestRegistry,
    pub(super) stats: Arc<SharedLiveRuntimeStats>,
    pub(super) workload: Option<Arc<WorkloadDispatchState>>,
}

pub(super) async fn wait_for_workload_progress<F>(
    mode: LiveReplayMode,
    semaphore: Option<&Semaphore>,
    next_ready_ms: Option<f64>,
    start: Instant,
    mut wake: Pin<&mut F>,
) where
    F: Future<Output = ()>,
{
    match (mode, semaphore, next_ready_ms) {
        (LiveReplayMode::Trace, _, Some(next_ready_ms)) => {
            let deadline = start + tokio::time::Duration::from_secs_f64(next_ready_ms / 1000.0);
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {}
                _ = wake.as_mut() => {}
            }
        }
        (LiveReplayMode::Trace, _, None) => {
            wake.as_mut().await;
        }
        (LiveReplayMode::Concurrency { .. }, Some(semaphore), Some(next_ready_ms)) => {
            if semaphore.available_permits() == 0 {
                wake.as_mut().await;
            } else {
                let deadline = start + tokio::time::Duration::from_secs_f64(next_ready_ms / 1000.0);
                tokio::select! {
                    _ = tokio::time::sleep_until(deadline) => {}
                    _ = wake.as_mut() => {}
                }
            }
        }
        (LiveReplayMode::Concurrency { .. }, Some(_semaphore), None) => {
            wake.as_mut().await;
        }
        (LiveReplayMode::Concurrency { .. }, None, _) => {
            unreachable!("concurrency mode must have a semaphore");
        }
    }
}

pub(super) async fn run_request_task(
    ctx: RequestTaskContext,
    request: DirectRequest,
    permit: Option<OwnedSemaphorePermit>,
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
    }
    drop(permit);
    Ok(())
}
