// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The list-and-watch stream every directory here serves: one `Added` per
//! current item, then a `Resync` with the same set, then changes. A
//! subscriber that falls behind gets a fresh `Resync`.

use crate::pool_discovery::{Snapshot, WatchEvent, WatchStream};
use tokio::sync::broadcast;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Events a slow subscriber may fall behind by before it gets a `Resync`.
pub(super) const EVENT_BUFFER: usize = 1024;

/// `subscribe_now` must subscribe and take its snapshot under the lock that
/// senders hold while they send, so the stream misses no event and sees none
/// twice. The snapshot is `None` until the source has a complete view; the
/// stream then starts at the first `Resync` it receives.
pub(super) fn watch_stream<T, K, F>(
    subscribe_now: F,
    cancel: CancellationToken,
    stop: CancellationToken,
) -> WatchStream<T, K>
where
    T: Clone + Send + 'static,
    K: Clone + Send + 'static,
    F: Fn() -> (broadcast::Receiver<WatchEvent<T, K>>, Option<Snapshot<T>>) + Send + 'static,
{
    Box::pin(async_stream::stream! {
        let (mut receiver, snapshot) = subscribe_now();
        let mut started = snapshot.is_some();
        if let Some(snapshot) = snapshot {
            for event in initial_events(snapshot) {
                yield Ok(event);
            }
        }
        loop {
            let received = tokio::select! {
                biased;
                () = cancel.cancelled() => break,
                () = stop.cancelled() => break,
                received = receiver.recv() => received,
            };
            match received {
                // The first complete view arrived after this subscription.
                Ok(WatchEvent::Resync(snapshot)) if !started => {
                    started = true;
                    for event in initial_events(snapshot) {
                        yield Ok(event);
                    }
                }
                Ok(event) => {
                    if started {
                        yield Ok(event);
                    }
                }
                Err(broadcast::error::RecvError::Lagged(missed)) => {
                    tracing::warn!(missed, "discovery watch fell behind; resyncing");
                    let (fresh, snapshot) = subscribe_now();
                    receiver = fresh;
                    if let Some(snapshot) = snapshot {
                        started = true;
                        yield Ok(WatchEvent::Resync(snapshot));
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    })
}

/// Sleeps until `at`, or forever without one: a timer branch in `select!`
/// that is off until it is armed.
pub(super) async fn wake_at(at: Option<Instant>) {
    match at {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending().await,
    }
}

fn initial_events<T: Clone, K>(snapshot: Snapshot<T>) -> Vec<WatchEvent<T, K>> {
    let mut events: Vec<WatchEvent<T, K>> = snapshot
        .items
        .iter()
        .cloned()
        .map(WatchEvent::Added)
        .collect();
    events.push(WatchEvent::Resync(snapshot));
    events
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use futures::StreamExt;

    use super::*;
    use crate::pool_discovery::Completeness;

    #[tokio::test]
    async fn a_subscriber_that_falls_behind_gets_a_fresh_resync() {
        let (events, _) = broadcast::channel::<WatchEvent<u32, u32>>(2);
        let items = Arc::new(Mutex::new(vec![1]));
        let subscribe_now = {
            let events = events.clone();
            let items = Arc::clone(&items);
            move || {
                let snapshot = Snapshot {
                    items: items.lock().unwrap().clone(),
                    completeness: Completeness::Exact,
                };
                (events.subscribe(), Some(snapshot))
            }
        };
        let mut stream = watch_stream(
            subscribe_now,
            CancellationToken::new(),
            CancellationToken::new(),
        );
        assert!(matches!(
            stream.next().await,
            Some(Ok(WatchEvent::Added(1)))
        ));
        assert!(matches!(
            stream.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));

        // Three changes overflow the two-event buffer.
        for item in 2..5 {
            items.lock().unwrap().push(item);
            events.send(WatchEvent::Added(item)).unwrap();
        }
        let Some(Ok(WatchEvent::Resync(snapshot))) = stream.next().await else {
            panic!("expected a fresh Resync");
        };
        assert_eq!(snapshot.items, [1, 2, 3, 4]);
    }
}
