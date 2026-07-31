// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::oneshot;

type CompletionStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;
type DoneFuture = Pin<Box<dyn std::future::Future<Output = ()> + Send>>;

struct PassThroughWithCompletion<S> {
    inner: Option<S>,
    done_tx: Option<oneshot::Sender<()>>,
}

impl<S> PassThroughWithCompletion<S> {
    fn new(inner: S, done_tx: oneshot::Sender<()>) -> Self {
        Self {
            inner: Some(inner),
            done_tx: Some(done_tx),
        }
    }

    fn drop_inner(&mut self) {
        drop(self.inner.take());
    }

    fn notify_done(&mut self) {
        if let Some(done_tx) = self.done_tx.take() {
            let _ = done_tx.send(());
        }
    }
}

impl<S> Drop for PassThroughWithCompletion<S> {
    fn drop(&mut self) {
        self.drop_inner();
        self.notify_done();
    }
}

impl<S, T> Stream for PassThroughWithCompletion<S>
where
    S: Stream<Item = T> + Unpin,
{
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Some(inner) = self.inner.as_mut() else {
            return Poll::Ready(None);
        };
        match Pin::new(inner).poll_next(cx) {
            Poll::Ready(None) => {
                self.drop_inner();
                self.notify_done();
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

/// Return a pass-through stream and a future that resolves when the stream ends or is dropped.
pub fn notify_on_completion<S, T>(stream: S) -> (CompletionStream<T>, DoneFuture)
where
    S: Stream<Item = T> + Unpin + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = oneshot::channel::<()>();
    let passthrough = PassThroughWithCompletion::new(stream, tx);
    (
        Box::pin(passthrough),
        Box::pin(async move {
            let _ = rx.await;
        }),
    )
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::task::{Context, Poll};

    use futures::{StreamExt, stream};

    use super::notify_on_completion;

    struct DropMarker {
        dropped: Arc<AtomicBool>,
    }

    impl Drop for DropMarker {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    impl futures::Stream for DropMarker {
        type Item = ();

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    #[tokio::test]
    async fn notifies_when_stream_is_dropped_before_exhaustion() {
        let (wrapped, done) = notify_on_completion(stream::iter([1_u8]));
        drop(wrapped);
        done.await;
    }

    #[tokio::test]
    async fn notifies_when_stream_is_exhausted() {
        let (mut wrapped, done) = notify_on_completion(stream::iter([1_u8]));
        assert_eq!(wrapped.next().await, Some(1));
        assert_eq!(wrapped.next().await, None);
        done.await;
    }

    #[tokio::test]
    async fn drops_inner_stream_before_notifying() {
        let dropped = Arc::new(AtomicBool::new(false));
        let (wrapped, done) = notify_on_completion(DropMarker {
            dropped: dropped.clone(),
        });

        drop(wrapped);
        done.await;

        assert!(dropped.load(Ordering::Acquire));
    }
}
