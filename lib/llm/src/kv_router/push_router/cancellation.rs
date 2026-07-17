// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;

use dynamo_runtime::{
    error::{DynamoError, ErrorType},
    pipeline::{AsyncEngineContext, Error},
};

pub(super) fn cancelled_error(context_id: &str) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(format!("Request {context_id} was cancelled"))
        .build()
        .into()
}

pub(super) async fn cancel_on_stop<T>(
    context: &dyn AsyncEngineContext,
    context_id: &str,
    operation: impl Future<Output = T>,
) -> Result<T, Error> {
    tokio::pin!(operation);
    tokio::select! {
        biased;

        // Preserve a simultaneously completed ownership-bearing result so its
        // normal cleanup path runs instead of treating it as an unseen result.
        result = &mut operation => Ok(result),
        _ = context.stopped() => Err(cancelled_error(context_id)),
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

        let error = cancel_on_stop(&context, context.id(), PendingUntilDropped(dropped.clone()))
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

        let result = cancel_on_stop(&context, context.id(), std::future::ready(42))
            .await
            .unwrap();

        assert_eq!(result, 42);
    }
}
