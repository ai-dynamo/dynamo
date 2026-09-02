// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, sync::Mutex, time::Duration};

// `tokio::time::Instant` rather than the std type so the budget is measured on
// the same clock as the `tokio::time::timeout` that consumes it.
use tokio::time::Instant;

use dynamo_runtime::{
    error::{DynamoError, ErrorType},
    pipeline::{AsyncEngineContext, Error},
};

use crate::protocols::common::timing::RequestPhase;

/// How long a disconnected decode request may keep router state alive while it
/// finishes reaching its worker. On expiry we abandon the operation and release
/// everything it held, which is what every phase did before the decode carve-out.
const CLEANUP_DISPATCH_TIMEOUT: Duration = Duration::from_secs(120);

/// One budget for a disconnected request's whole cleanup route.
///
/// The decode leg passes through several wrapped stages in sequence — worker
/// selection, routing-decision recording, dispatch — and conditional
/// disaggregation can select more than once. Giving each stage its own timeout
/// would let total retention reach a multiple of [`CLEANUP_DISPATCH_TIMEOUT`],
/// so the stages share this budget instead.
///
/// The clock starts the first time a stage actually observes a stopped context,
/// which means a live request never arms it and keeps today's semantics exactly.
#[derive(Debug, Default)]
pub(super) struct CleanupBudget {
    started: Mutex<Option<Instant>>,
}

impl CleanupBudget {
    /// Time left in the budget, starting the clock on first use.
    fn remaining(&self) -> Duration {
        let mut started = self.started.lock().unwrap();
        let start = *started.get_or_insert_with(Instant::now);
        CLEANUP_DISPATCH_TIMEOUT.saturating_sub(start.elapsed())
    }
}

/// Whether a disconnected client's request still reaches its worker.
///
/// Decode must proceed: remote prefill stages KV blocks for one specific decode
/// worker, and only that worker's KV-transfer-complete path frees them. Every
/// other phase keeps cancelling as soon as the client goes away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DispatchCancellation {
    /// Abandon the operation when the context is already stopped.
    CancelWhenStopped,
    /// Run the operation to completion even when the context is stopped.
    DispatchWhenStopped,
}

impl DispatchCancellation {
    pub(super) fn for_phase(phase: RequestPhase) -> Self {
        match phase {
            RequestPhase::Decode => Self::DispatchWhenStopped,
            RequestPhase::Prefill | RequestPhase::Aggregated => Self::CancelWhenStopped,
        }
    }
}

/// Await a routing stage under the cancellation policy its phase requires.
///
/// `DispatchWhenStopped` is bounded by the shared [`CleanupBudget`] rather than
/// waiting forever, because the stages this wraps have no deadline of their own.
/// A live request never arms the budget.
///
/// `stage` names the routing stage for the log emitted when the budget runs out,
/// so an exhausted cleanup is distinguishable from an ordinary client disconnect.
pub(super) async fn await_with_phase_policy<T>(
    context: &dyn AsyncEngineContext,
    phase: RequestPhase,
    stage: &'static str,
    budget: &CleanupBudget,
    operation: impl Future<Output = T>,
) -> Result<T, Error> {
    match DispatchCancellation::for_phase(phase) {
        DispatchCancellation::CancelWhenStopped => cancel_on_stop(context, operation).await,
        DispatchCancellation::DispatchWhenStopped => {
            tokio::pin!(operation);
            tokio::select! {
                biased;

                result = &mut operation => Ok(result),
                _ = context.stopped() => {
                    match tokio::time::timeout(budget.remaining(), &mut operation).await {
                        Ok(result) => Ok(result),
                        Err(_) => Err(cleanup_budget_exhausted(context.id(), stage)),
                    }
                }
            }
        }
    }
}

/// The decode leg ran out of cleanup budget before reaching its worker, so the
/// KV blocks remote prefill staged for that worker are not released and will sit
/// until they expire. Reported separately from an ordinary client disconnect,
/// which is not a leak, so the two are distinguishable in logs.
fn cleanup_budget_exhausted(context_id: &str, stage: &'static str) -> Error {
    tracing::warn!(
        request_id = %context_id,
        stage,
        budget_secs = CLEANUP_DISPATCH_TIMEOUT.as_secs(),
        "decode cleanup budget exhausted before the worker was reached; staged KV \
         blocks will not be released until they expire"
    );
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(format!(
            "Request {context_id} exhausted its decode cleanup budget at {stage}"
        ))
        .build()
        .into()
}

pub(super) fn cancelled_error(context_id: &str) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(format!("Request {context_id} was cancelled"))
        .build()
        .into()
}

pub(super) async fn cancel_on_stop<T>(
    context: &dyn AsyncEngineContext,
    operation: impl Future<Output = T>,
) -> Result<T, Error> {
    tokio::pin!(operation);
    tokio::select! {
        biased;

        // Preserve a simultaneously completed ownership-bearing result so its
        // normal cleanup path runs instead of treating it as an unseen result.
        result = &mut operation => Ok(result),
        _ = context.stopped() => Err(cancelled_error(context.id())),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context, Poll},
        time::Duration,
    };

    use dynamo_runtime::{
        error::{DynamoError, ErrorType},
        pipeline::{AsyncEngineContext, context::Controller},
    };

    use super::{CLEANUP_DISPATCH_TIMEOUT, CleanupBudget, cancel_on_stop};

    struct PendingUntilDropped(Arc<AtomicBool>);

    impl Future for PendingUntilDropped {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for PendingUntilDropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// The budget is shared across a request's stages rather than restarting at
    /// each one, so several stages cannot together retain a stopped request for a
    /// multiple of the constant.
    #[tokio::test(start_paused = true)]
    async fn cleanup_budget_is_shared_across_stages() {
        let budget = CleanupBudget::default();

        // First stage starts the clock with the whole budget available.
        assert_eq!(budget.remaining(), CLEANUP_DISPATCH_TIMEOUT);

        tokio::time::advance(Duration::from_secs(90)).await;

        // A later stage inherits what is left, it does not get a fresh budget.
        assert_eq!(
            budget.remaining(),
            CLEANUP_DISPATCH_TIMEOUT - Duration::from_secs(90)
        );

        tokio::time::advance(Duration::from_secs(60)).await;

        // Past the deadline the remaining budget saturates at zero rather than
        // wrapping, so a late stage gets no extension.
        assert_eq!(budget.remaining(), Duration::ZERO);
    }

    #[tokio::test]
    async fn drops_pending_operation_when_context_stops() {
        let context = Controller::new("cancelled-request".to_string());
        context.stop();
        let dropped = Arc::new(AtomicBool::new(false));

        let error = cancel_on_stop(&context, PendingUntilDropped(dropped.clone()))
            .await
            .unwrap_err();

        let error = error
            .downcast_ref::<DynamoError>()
            .expect("cancellation should return DynamoError");
        assert_eq!(error.error_type(), ErrorType::Cancelled);
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn ready_operation_wins_if_context_is_already_stopped() {
        let context = Controller::new("completed-request".to_string());
        context.stop();

        let result = cancel_on_stop(&context, std::future::ready(42))
            .await
            .unwrap();

        assert_eq!(result, 42);
    }
}
