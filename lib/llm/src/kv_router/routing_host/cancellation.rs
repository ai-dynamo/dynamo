// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, time::Duration};

use dynamo_runtime::{
    error::{DynamoError, ErrorType},
    pipeline::{AsyncEngineContext, Error},
};

use crate::protocols::common::timing::RequestPhase;

/// How long a disconnected decode request may keep router state alive while it
/// finishes reaching its worker. On expiry we fall back to the pre-DYN-4143
/// behaviour: abandon the operation and release everything it held.
const CLEANUP_DISPATCH_TIMEOUT: Duration = Duration::from_secs(120);

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
/// `DispatchWhenStopped` is bounded: a stopped context starts a
/// [`CLEANUP_DISPATCH_TIMEOUT`] deadline rather than waiting forever, because the
/// stages this wraps have no deadline of their own. A live request never arms the
/// deadline and keeps today's semantics exactly.
pub(super) async fn await_with_phase_policy<T>(
    context: &dyn AsyncEngineContext,
    phase: RequestPhase,
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
                    tokio::time::timeout(CLEANUP_DISPATCH_TIMEOUT, &mut operation)
                        .await
                        .map_err(|_| cancelled_error(context.id()))
                }
            }
        }
    }
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
    };

    use dynamo_runtime::{
        error::{DynamoError, ErrorType},
        pipeline::{AsyncEngineContext, context::Controller},
    };

    use super::cancel_on_stop;

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
