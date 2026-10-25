// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The list-and-watch stream every directory here serves: one `Added` per
//! current item, then a `Resync` with the same set, then changes. A
//! subscriber that falls behind gets a fresh `Resync`.

use crate::pool_discovery::WatchEvent;
use crate::pool_discovery::watch::{Snapshot, WatchStream};
use tokio::sync::broadcast;
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
