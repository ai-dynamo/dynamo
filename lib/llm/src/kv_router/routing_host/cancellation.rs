// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, pin::Pin, sync::Mutex, time::Duration};

// `tokio::time::Instant` rather than the std type so the budget is measured on
// the same clock as the `tokio::time::timeout` that consumes it.
use tokio::time::Instant;

use dynamo_runtime::{
    error::{DynamoError, ErrorType},
    pipeline::{AsyncEngineContext, Error},
};

use crate::{preprocessor::PreprocessedRequest, protocols::common::timing::RequestPhase};

/// Bound disconnected requests while they deliver an owned upstream handoff.
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
    pub(super) fn remaining(&self) -> Duration {
        let mut started = self.started.lock().unwrap();
        let start = *started.get_or_insert_with(Instant::now);
        CLEANUP_DISPATCH_TIMEOUT.saturating_sub(start.elapsed())
    }
}

/// Upstream resources whose receipt must survive caller cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StagedTransfer {
    /// Blocks are staged on a prefill worker for one specific decode worker.
    Kv,
    /// A completed encoder output must reach its prefill or combined consumer.
    Encoder,
    /// Nothing is staged, so abandoning the request releases everything it holds.
    Absent,
}

impl StagedTransfer {
    pub(super) fn for_request(request: &PreprocessedRequest) -> Self {
        if request.staged_kv_cleanup {
            Self::Kv
        } else if request.encoder_result.is_some() && request.prefill_result.is_none() {
            Self::Encoder
        } else {
            Self::Absent
        }
    }
}

/// Whether a disconnected client's request still reaches its worker.
///
/// Staged KV must reach decode; encoder output must reach its consumer.
/// Requests without a handoff cancel normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DispatchCancellation {
    /// Abandon the operation when the context is already stopped.
    CancelWhenStopped,
    /// Run the operation to completion even when the context is stopped.
    DispatchWhenStopped,
}

impl DispatchCancellation {
    pub(super) fn for_request(phase: RequestPhase, staged_transfer: StagedTransfer) -> Self {
        match (phase, staged_transfer) {
            (_, StagedTransfer::Encoder) | (RequestPhase::Decode, StagedTransfer::Kv) => {
                Self::DispatchWhenStopped
            }
            (RequestPhase::Decode, StagedTransfer::Absent)
            | (RequestPhase::Prefill, _)
            | (RequestPhase::Aggregated, _) => Self::CancelWhenStopped,
        }
    }
}

/// Await a routing stage under the cancellation policy the request requires.
///
/// `DispatchWhenStopped` is bounded by the shared [`CleanupBudget`] rather than
/// waiting forever, because the stages this wraps have no deadline of their own.
/// A live request never arms the budget.
///
/// `stage` names the routing stage for the log emitted when the budget runs out,
/// so an exhausted cleanup is distinguishable from an ordinary client disconnect.
pub(super) async fn await_with_cleanup_policy<T>(
    context: &dyn AsyncEngineContext,
    phase: RequestPhase,
    staged_transfer: StagedTransfer,
    stage: &'static str,
    budget: &CleanupBudget,
    operation: impl Future<Output = T>,
) -> Result<T, Error> {
    // Pinned once here and lent to both arms, so the wrapper's state holds the
    // operation a single time rather than once per arm.
    tokio::pin!(operation);
    match DispatchCancellation::for_request(phase, staged_transfer) {
        DispatchCancellation::CancelWhenStopped => cancel_on_stop(context, operation).await,
        DispatchCancellation::DispatchWhenStopped => {
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

/// Report failed handoff cleanup as `Cancelled` to prevent migration from
/// retrying an abandoned request on another worker.
fn cleanup_budget_exhausted(context_id: &str, stage: &'static str) -> Error {
    tracing::warn!(
        request_id = %context_id,
        stage,
        budget_secs = CLEANUP_DISPATCH_TIMEOUT.as_secs(),
        "handoff cleanup budget exhausted before the worker was reached; producer cleanup may be required"
    );
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(format!(
            "Request {context_id} exhausted its handoff cleanup budget at {stage}"
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
    mut operation: Pin<&mut impl Future<Output = T>>,
) -> Result<T, Error> {
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

    use super::{
        CLEANUP_DISPATCH_TIMEOUT, CleanupBudget, StagedTransfer, await_with_cleanup_policy,
        cancel_on_stop,
    };
    use crate::protocols::common::timing::RequestPhase;

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

    /// A budget decays with the clock and floors at zero, so a late stage gets
    /// what is left and never a negative or wrapped extension.
    ///
    /// That the stages of a real route share one budget is covered end to end by
    /// `conditional_route_stages_share_one_cleanup_budget` in the parent module.
    #[tokio::test(start_paused = true)]
    async fn cleanup_budget_decays_and_saturates_at_zero() {
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

        // Through the wrapper, which owns the operation: it is gone once the
        // cancellation error is returned.
        let error = await_with_cleanup_policy(
            &context,
            RequestPhase::Aggregated,
            StagedTransfer::Absent,
            "test",
            &CleanupBudget::default(),
            PendingUntilDropped(dropped.clone()),
        )
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

        let ready = std::future::ready(42);
        tokio::pin!(ready);
        let result = cancel_on_stop(&context, ready).await.unwrap();

        assert_eq!(result, 42);
    }
}
