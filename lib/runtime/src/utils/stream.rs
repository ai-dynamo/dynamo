// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::stream::{Stream, StreamExt};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use tokio::time::{self, Duration, Instant, Sleep, sleep_until};

pub struct DeadlineStream<S> {
    stream: S,
    sleep: Pin<Box<Sleep>>,
}

impl<S: Stream + Unpin> Stream for DeadlineStream<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Check if we've passed the deadline
        if Pin::new(&mut self.sleep).poll(cx).is_ready() {
            // The deadline expired; end the stream now
            return Poll::Ready(None);
        }

        // Otherwise, poll the underlying stream
        let val = self.as_mut().stream.poll_next_unpin(cx);
        // Log the poll result and return it
        match &val {
            Poll::Ready(Some(_)) => tracing::trace!("DeadlineStream: received item"),
            Poll::Ready(None) => tracing::trace!("DeadlineStream: underlying stream ended"),
            Poll::Pending => tracing::trace!("DeadlineStream: waiting for next item"),
        }
        val
    }
}

pub fn until_deadline<S: Stream + Unpin>(stream: S, deadline: Instant) -> DeadlineStream<S> {
    DeadlineStream {
        stream,
        // Set an async task that sleeps until deadline and wakes up to cancel the stream
        sleep: Box::pin(sleep_until(deadline)),
    }
}

#[cfg(test)]
mod tests {
    use futures::stream::{self, Stream, StreamExt};
    use tokio::pin;

    use super::*;

    // Helper function to run the deadline stream test with given parameters
    async fn run_deadline_test(sleep_times_ms: Vec<u64>, deadline_ms: u64) -> Vec<u64> {
        let stream = stream::iter(sleep_times_ms);
        let stream = stream.then(|x| {
            let sleep = time::sleep(Duration::from_millis(x));
            async move {
                sleep.await;
                x
            }
        });

        let deadline = Instant::now() + Duration::from_millis(deadline_ms);
        let mut result = Vec::new();

        pin!(stream);
        let mut stream = until_deadline(stream, deadline);

        while let Some(x) = stream.next().await {
            result.push(x);
        }

        result
    }

    #[tokio::test]
    async fn test_deadline_exceeded() {
        // The sum of the sleep times should exceed the deadline
        let sleep_times_ms = vec![100, 100, 200, 50];
        let deadline_ms = 300;

        let result = run_deadline_test(sleep_times_ms, deadline_ms).await;
        // Since deadline is exceeded, only the items before deadline should be returned
        assert_eq!(result, vec![100, 100]);
    }

    #[tokio::test]
    async fn test_complete_before_deadline() {
        // The sum of the sleep times should be less than the deadline
        let sleep_times_ms = vec![100, 50, 50];
        let deadline_ms = 300;

        let result = run_deadline_test(sleep_times_ms, deadline_ms).await;
        // Since deadline is not exceeded, all items should be returned from stream
        assert_eq!(result, vec![100, 50, 50]);
    }
}
