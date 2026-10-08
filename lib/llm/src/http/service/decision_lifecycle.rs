// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_runtime::engine::AsyncEngineContext;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tracing::Instrument;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AdmissionError {
    InvalidBranchCount,
    RequestTooLarge,
    CapacityExhausted,
    AdmissionClosed,
}

/// Reject excess load immediately, matching frontend admission rather than queueing.
/// The caller must initialize the semaphore and `limit` from the same bounded configuration.
pub(crate) fn acquire_admission(
    admission: Arc<Semaphore>,
    branches: u32,
    limit: usize,
) -> Result<OwnedSemaphorePermit, AdmissionError> {
    if branches == 0 {
        return Err(AdmissionError::InvalidBranchCount);
    }
    if branches as usize > limit {
        return Err(AdmissionError::RequestTooLarge);
    }
    admission
        .try_acquire_many_owned(branches)
        .map_err(|cause| match cause {
            TryAcquireError::NoPermits => AdmissionError::CapacityExhausted,
            TryAcquireError::Closed => AdmissionError::AdmissionClosed,
        })
}

pub(crate) async fn run_until_killed<T>(
    context: &dyn AsyncEngineContext,
    operation: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::pin!(operation);
    tokio::select! {
        // Preserve completed results when completion and cancellation are both ready.
        biased;
        result = &mut operation => Some(result),
        _ = context.killed() => None,
    }
}

pub(crate) async fn spawn_blocking_with_permit<T, E, F>(
    permit: OwnedSemaphorePermit,
    task: F,
) -> Result<Result<(T, OwnedSemaphorePermit), E>, tokio::task::JoinError>
where
    T: Send + 'static,
    E: Send + 'static,
    F: FnOnce() -> Result<T, E> + Send + 'static,
{
    tokio::task::spawn_blocking(move || task().map(|output| (output, permit))).await
}

#[derive(Debug)]
pub(crate) enum DispatchWaitError {
    Task(tokio::task::JoinError),
    Deadline,
}

/// The task must own admission until its work completes and should observe `parent.kill()`.
/// Cleanup runs separately so the response does not wait for the cancellation grace.
/// Local completion does not establish physical GPU settlement.
pub(crate) async fn dispatch_with_deadline<T: Send + 'static>(
    mut task: tokio::task::JoinHandle<T>,
    deadline: tokio::time::Instant,
    parent: Arc<dyn AsyncEngineContext>,
) -> Result<T, DispatchWaitError> {
    match tokio::time::timeout_at(deadline, &mut task).await {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(cause)) => {
            parent.kill();
            Err(DispatchWaitError::Task(cause))
        }
        Err(_) => {
            parent.kill();
            tokio::spawn(drain_cancelled_task(task).in_current_span());
            Err(DispatchWaitError::Deadline)
        }
    }
}

async fn drain_cancelled_task<T: Send + 'static>(mut task: tokio::task::JoinHandle<T>) {
    let result = match tokio::time::timeout(std::time::Duration::from_secs(1), &mut task).await {
        Ok(result) => result,
        Err(_) => {
            tracing::warn!(
                "Decision task exceeded cancellation grace; retaining admission until completion"
            );
            task.await
        }
    };
    if let Err(cause) = result {
        tracing::error!(error = %cause, "Cancelled decision task failed during cleanup");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{engine::AsyncEngineContextProvider, pipeline::Context};
    use std::{future::pending, time::Duration};
    use tokio::sync::oneshot;

    #[test]
    fn admission_distinguishes_unfulfillable_requests_from_busy_capacity() {
        let admission = Arc::new(Semaphore::new(2));
        assert_eq!(
            acquire_admission(admission.clone(), 3, 2).unwrap_err(),
            AdmissionError::RequestTooLarge,
        );
        let permit = acquire_admission(admission.clone(), 2, 2).unwrap();
        assert_eq!(
            acquire_admission(admission.clone(), 1, 2).unwrap_err(),
            AdmissionError::CapacityExhausted,
        );
        drop(permit);
        assert!(acquire_admission(admission, 1, 2).is_ok());
    }

    #[test]
    fn admission_rejects_empty_and_closed_requests_without_reporting_overload() {
        let admission = Arc::new(Semaphore::new(1));
        assert_eq!(
            acquire_admission(admission.clone(), 0, 1).unwrap_err(),
            AdmissionError::InvalidBranchCount,
        );
        admission.close();
        assert_eq!(
            acquire_admission(admission, 1, 1).unwrap_err(),
            AdmissionError::AdmissionClosed,
        );
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_returns_without_waiting_for_noncooperative_work() {
        let parent = Context::new(()).context();
        let (release_tx, release_rx) = oneshot::channel();
        let task = tokio::spawn(release_rx);
        let start = tokio::time::Instant::now();
        let deadline = start + Duration::from_secs(2);
        let result = dispatch_with_deadline(task, deadline, parent.clone()).await;
        assert!(matches!(result, Err(DispatchWaitError::Deadline)));
        assert!(parent.is_killed());
        assert_eq!(tokio::time::Instant::now(), deadline);
        release_tx.send(()).unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn graceful_stop_preserves_inflight_results() {
        let parent = Context::new(()).context();
        let task_context = parent.clone();
        let (result_tx, result_rx) = oneshot::channel();
        let task =
            tokio::spawn(async move { run_until_killed(task_context.as_ref(), result_rx).await });
        parent.stop_generating();
        tokio::task::yield_now().await;
        assert!(!task.is_finished());
        result_tx.send(42).unwrap();
        assert_eq!(task.await.unwrap().unwrap().unwrap(), 42);
    }

    #[tokio::test(start_paused = true)]
    async fn expired_deadline_kills_and_drains_local_work() {
        let parent = Context::new(()).context();
        let worker_context = parent.clone();
        let admission = Arc::new(Semaphore::new(1));
        let permit = admission.clone().acquire_owned().await.unwrap();
        let task = tokio::spawn(async move {
            let _permit = permit;
            run_until_killed(worker_context.as_ref(), pending::<()>()).await
        });
        let result =
            dispatch_with_deadline(task, tokio::time::Instant::now(), parent.clone()).await;
        assert!(matches!(result, Err(DispatchWaitError::Deadline)));
        assert!(parent.is_killed());
        let permit = tokio::time::timeout(Duration::from_secs(1), admission.acquire_owned())
            .await
            .unwrap()
            .unwrap();
        drop(permit);
    }

    #[tokio::test(start_paused = true)]
    async fn overdue_cleanup_is_logged_without_a_metric_and_keeps_admission() {
        let _interest_guard = crate::test_utils::tracing_interest_guard();
        let log = tempfile::NamedTempFile::new().unwrap();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(Arc::new(log.reopen().unwrap()))
            .finish();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        let registry = prometheus::Registry::new();
        super::super::metrics::Metrics::new()
            .register(&registry)
            .unwrap();
        assert!(
            !registry
                .gather()
                .iter()
                .any(|family| { family.name().ends_with("decision_cleanup_overruns_total") })
        );
        let parent = Context::new(()).context();
        let admission = Arc::new(Semaphore::new(1));
        let permit = admission.clone().acquire_owned().await.unwrap();
        let (release_tx, release_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let _permit = permit;
            let _ = release_rx.await;
        });
        let result = dispatch_with_deadline(task, tokio::time::Instant::now(), parent)
            .instrument(tracing::info_span!(
                "http-request",
                request_id = "cleanup-request"
            ))
            .await;
        assert!(matches!(result, Err(DispatchWaitError::Deadline)));
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        let output = std::fs::read_to_string(log.path()).unwrap();
        assert!(output.contains("exceeded cancellation grace"), "{output}");
        assert!(
            output.contains("request_id=\"cleanup-request\""),
            "{output}"
        );
        assert_eq!(admission.available_permits(), 0);
        release_tx.send(()).unwrap();
        let permit = tokio::time::timeout(Duration::from_secs(1), admission.acquire_owned())
            .await
            .unwrap()
            .unwrap();
        drop(permit);
    }

    #[tokio::test]
    async fn finished_dispatch_preserves_output_without_killing_context() {
        let parent = Context::new(()).context();
        let result = dispatch_with_deadline(
            tokio::spawn(async { 42 }),
            tokio::time::Instant::now() + Duration::from_secs(1),
            parent.clone(),
        )
        .await
        .unwrap();
        assert_eq!(result, 42);
        assert!(!parent.is_killed());
    }

    #[tokio::test]
    async fn failed_dispatch_kills_context() {
        let parent = Context::new(()).context();
        let task = tokio::spawn(pending::<()>());
        task.abort();
        let result = dispatch_with_deadline(
            task,
            tokio::time::Instant::now() + Duration::from_secs(1),
            parent.clone(),
        )
        .await;
        match result {
            Err(DispatchWaitError::Task(cause)) => assert!(cause.is_cancelled()),
            _ => panic!("cancelled task should report its join error"),
        }
        assert!(parent.is_killed());
    }

    #[tokio::test]
    async fn parent_kill_interrupts_pending_branch_work() {
        let parent = Context::new(());
        let context = parent.context();
        let task_context = context.clone();
        let task =
            tokio::spawn(
                async move { run_until_killed(task_context.as_ref(), pending::<()>()).await },
            );
        context.kill();
        assert!(task.await.unwrap().is_none());
    }

    #[tokio::test]
    async fn blocking_preflight_owns_admission_until_it_finishes() {
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let task = tokio::spawn(spawn_blocking_with_permit(permit, move || {
            started_tx.send(()).unwrap();
            release_rx.blocking_recv().unwrap();
            Ok::<_, ()>(())
        }));
        started_rx.await.unwrap();
        assert!(semaphore.clone().try_acquire_owned().is_err());
        release_tx.send(()).unwrap();
        let (_, returned_permit) = task.await.unwrap().unwrap().unwrap();
        drop(returned_permit);
        assert!(semaphore.try_acquire_owned().is_ok());
    }

    #[tokio::test]
    async fn abandoned_preflight_keeps_admission_until_blocking_work_finishes() {
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let task = tokio::spawn(spawn_blocking_with_permit(permit, move || {
            started_tx.send(()).unwrap();
            release_rx.blocking_recv().unwrap();
            Ok::<_, ()>(())
        }));
        started_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(semaphore.clone().try_acquire_owned().is_err());
        release_tx.send(()).unwrap();
        let permit = tokio::time::timeout(Duration::from_secs(2), semaphore.acquire_owned())
            .await
            .unwrap()
            .unwrap();
        drop(permit);
    }
}
