// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_runtime::engine::AsyncEngineContext;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AdmissionError {
    RequestTooLarge,
    CapacityExhausted,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Consumed by decision execution integration")
)]
pub(crate) fn acquire_admission(
    admission: Arc<Semaphore>,
    branches: u32,
    limit: usize,
) -> Result<OwnedSemaphorePermit, AdmissionError> {
    if branches as usize > limit {
        return Err(AdmissionError::RequestTooLarge);
    }
    admission
        .try_acquire_many_owned(branches)
        .map_err(|_| AdmissionError::CapacityExhausted)
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Consumed by decision execution integration")
)]
pub(crate) async fn run_until_killed<T>(
    context: &dyn AsyncEngineContext,
    operation: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::pin!(operation);
    tokio::select! {
        biased;
        result = &mut operation => Some(result),
        _ = context.killed() => None,
    }
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Consumed by decision execution integration")
)]
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
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Inspected by the decision response adapter")
    )]
    Task(tokio::task::JoinError),
    Deadline,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Consumed by decision execution integration")
)]
pub(crate) async fn dispatch_with_deadline<T>(
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
            // Local draining does not acknowledge physical GPU settlement.
            let _ = task.await;
            Err(DispatchWaitError::Deadline)
        }
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

    #[tokio::test]
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
        assert_eq!(admission.available_permits(), 1);
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
