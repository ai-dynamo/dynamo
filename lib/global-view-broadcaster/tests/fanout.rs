// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_global_view_broadcaster::{Frame, PoolViewBroadcaster};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

type Stream = mpsc::Receiver<Frame<&'static str, u64>>;

async fn next(stream: &mut Stream) -> Frame<&'static str, u64> {
    timeout(Duration::from_secs(2), stream.recv())
        .await
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn broadcasts_to_all_routers_and_recovers_a_disconnected_replica() {
    let broadcaster = PoolViewBroadcaster::new();
    broadcaster.upsert("blue", 10);

    let (directory_tx, directory_rx) = tokio::sync::watch::channel(HashSet::from(["a", "b"]));
    let (connections_tx, mut connections_rx) = mpsc::unbounded_channel();
    let running = broadcaster.clone();
    let task = tokio::spawn(async move {
        running
            .run(directory_rx, move |router| {
                let connections_tx = connections_tx.clone();
                async move {
                    let (sink, stream) = mpsc::channel(8);
                    connections_tx
                        .send((router, stream))
                        .map_err(|_| "connection observer closed")?;
                    Ok::<_, &'static str>(sink)
                }
            })
            .await;
    });

    let mut streams = HashMap::new();
    for _ in 0..2 {
        let (router, mut stream) = timeout(Duration::from_secs(2), connections_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            next(&mut stream).await,
            Frame::Snapshot {
                revision: 1,
                pools: HashMap::from([("blue", 10)]),
            }
        );
        streams.insert(router, stream);
    }
    assert_eq!(streams.len(), 2);

    broadcaster.upsert("blue", 11);
    for stream in streams.values_mut() {
        assert_eq!(
            next(stream).await,
            Frame::Upsert {
                revision: 2,
                pool: "blue",
                view: 11,
            }
        );
    }

    directory_tx.send(HashSet::from(["a", "b", "c"])).unwrap();
    let (router, mut new_stream) = timeout(Duration::from_secs(2), connections_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(router, "c");
    assert_eq!(
        next(&mut new_stream).await,
        Frame::Snapshot {
            revision: 2,
            pools: HashMap::from([("blue", 11)]),
        }
    );
    streams.insert(router, new_stream);

    drop(streams.remove("a"));
    let (router, mut recovered) = timeout(Duration::from_secs(2), connections_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(router, "a");
    assert_eq!(
        next(&mut recovered).await,
        Frame::Snapshot {
            revision: 2,
            pools: HashMap::from([("blue", 11)]),
        }
    );
    streams.insert(router, recovered);

    directory_tx.send(HashSet::from(["a", "c"])).unwrap();
    let mut removed = streams.remove("b").unwrap();
    assert_eq!(
        timeout(Duration::from_secs(2), removed.recv())
            .await
            .unwrap(),
        None
    );

    broadcaster.remove(&"blue");
    for stream in streams.values_mut() {
        assert_eq!(
            next(stream).await,
            Frame::Remove {
                revision: 3,
                pool: "blue",
            }
        );
    }

    broadcaster.replace_snapshot(HashMap::from([("green", 20)]));
    for stream in streams.values_mut() {
        assert_eq!(
            next(stream).await,
            Frame::Snapshot {
                revision: 4,
                pools: HashMap::from([("green", 20)]),
            }
        );
    }

    drop(directory_tx);
    timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn update_during_connection_is_included_in_the_initial_snapshot() {
    let broadcaster = PoolViewBroadcaster::new();
    broadcaster.upsert("blue", 10);
    let (directory_tx, directory_rx) = tokio::sync::watch::channel(HashSet::from(["a"]));
    let (connecting_tx, connecting_rx) = oneshot::channel();
    let (resume_tx, resume_rx) = oneshot::channel();
    let (stream_tx, mut stream_rx) = mpsc::unbounded_channel();
    let connecting_tx = std::sync::Mutex::new(Some(connecting_tx));
    let resume_rx = std::sync::Mutex::new(Some(resume_rx));
    let running = broadcaster.clone();
    let task = tokio::spawn(async move {
        running
            .run(directory_rx, move |_| {
                let connecting = connecting_tx.lock().unwrap().take().unwrap();
                let resume = resume_rx.lock().unwrap().take().unwrap();
                let stream_tx = stream_tx.clone();
                async move {
                    connecting.send(()).unwrap();
                    resume.await.unwrap();
                    let (sink, stream) = mpsc::channel(8);
                    stream_tx.send(stream).unwrap();
                    Ok::<_, &'static str>(sink)
                }
            })
            .await;
    });
    timeout(Duration::from_secs(2), connecting_rx)
        .await
        .unwrap()
        .unwrap();
    for view in 11..=110 {
        broadcaster.upsert("blue", view);
    }
    resume_tx.send(()).unwrap();
    let mut stream = timeout(Duration::from_secs(2), stream_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        next(&mut stream).await,
        Frame::Snapshot {
            revision: 101,
            pools: HashMap::from([("blue", 110)]),
        }
    );
    assert!(
        timeout(Duration::from_millis(200), stream.recv())
            .await
            .is_err()
    );
    broadcaster.upsert("blue", 111);
    assert_eq!(
        next(&mut stream).await,
        Frame::Upsert {
            revision: 102,
            pool: "blue",
            view: 111,
        }
    );
    drop(directory_tx);
    timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn retries_a_failed_router_attachment() {
    let broadcaster = PoolViewBroadcaster::new();
    broadcaster.upsert("blue", 10);
    let (directory_tx, directory_rx) = tokio::sync::watch::channel(HashSet::from(["a"]));
    let (connections_tx, mut connections_rx) = mpsc::unbounded_channel();
    let attempts = Arc::new(AtomicUsize::new(0));
    let attach_attempts = Arc::clone(&attempts);
    let running = broadcaster.clone();
    let task = tokio::spawn(async move {
        running
            .run(directory_rx, move |_| {
                let connections_tx = connections_tx.clone();
                let attempts = Arc::clone(&attach_attempts);
                async move {
                    if attempts.fetch_add(1, Ordering::Relaxed) == 0 {
                        return Err("router unavailable");
                    }
                    let (sink, stream) = mpsc::channel(8);
                    connections_tx
                        .send(stream)
                        .map_err(|_| "connection observer closed")?;
                    Ok(sink)
                }
            })
            .await;
    });

    let mut stream = timeout(Duration::from_secs(2), connections_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(attempts.load(Ordering::Relaxed), 2);
    assert_eq!(
        next(&mut stream).await,
        Frame::Snapshot {
            revision: 1,
            pools: HashMap::from([("blue", 10)]),
        }
    );

    drop(directory_tx);
    timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn lagging_router_receives_a_fresh_snapshot_on_the_same_connection() {
    let broadcaster = PoolViewBroadcaster::new();
    let (directory_tx, directory_rx) = tokio::sync::watch::channel(HashSet::from(["a"]));
    let (connections_tx, mut connections_rx) = mpsc::unbounded_channel();
    let running = broadcaster.clone();
    let task = tokio::spawn(async move {
        running
            .run(directory_rx, move |router| {
                let connections_tx = connections_tx.clone();
                async move {
                    let (sink, stream) = mpsc::channel(128);
                    connections_tx
                        .send((router, stream))
                        .map_err(|_| "connection observer closed")?;
                    Ok::<_, &'static str>(sink)
                }
            })
            .await;
    });
    let (_, mut first_stream) = timeout(Duration::from_secs(2), connections_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        next(&mut first_stream).await,
        Frame::Snapshot {
            revision: 0,
            pools: HashMap::new(),
        }
    );
    for revision in 1..=100 {
        broadcaster.upsert("blue", revision);
    }
    assert_eq!(
        next(&mut first_stream).await,
        Frame::Snapshot {
            revision: 100,
            pools: HashMap::from([("blue", 100)]),
        }
    );
    assert!(
        timeout(Duration::from_millis(200), connections_rx.recv())
            .await
            .is_err()
    );
    broadcaster.upsert("blue", 101);
    assert_eq!(
        next(&mut first_stream).await,
        Frame::Upsert {
            revision: 101,
            pool: "blue",
            view: 101,
        }
    );
    drop(directory_tx);
    timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn stalled_router_resnapshots_without_draining_its_old_channel() {
    let broadcaster = PoolViewBroadcaster::new();
    let (directory_tx, directory_rx) = tokio::sync::watch::channel(HashSet::from(["a"]));
    let (connections_tx, mut connections_rx) = mpsc::unbounded_channel();
    let running = broadcaster.clone();
    let task = tokio::spawn(async move {
        running
            .run(directory_rx, move |router| {
                let connections_tx = connections_tx.clone();
                async move {
                    let (sink, stream) = mpsc::channel(1);
                    connections_tx
                        .send((router, stream))
                        .map_err(|_| "connection observer closed")?;
                    Ok::<_, &'static str>(sink)
                }
            })
            .await;
    });

    let (_, mut stalled) = timeout(Duration::from_secs(2), connections_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(next(&mut stalled).await, Frame::Snapshot { .. }));
    broadcaster.upsert("blue", 1);
    assert!(matches!(next(&mut stalled).await, Frame::Upsert { .. }));

    broadcaster.upsert("blue", 2);
    timeout(Duration::from_secs(2), async {
        while stalled.capacity() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    broadcaster.upsert("blue", 3);
    tokio::time::sleep(Duration::from_millis(50)).await;
    for revision in 4..=100 {
        broadcaster.upsert("blue", revision);
    }

    let (router, mut recovered) = timeout(Duration::from_secs(3), connections_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(router, "a");
    assert_eq!(
        next(&mut recovered).await,
        Frame::Snapshot {
            revision: 100,
            pools: HashMap::from([("blue", 100)]),
        }
    );

    drop(directory_tx);
    timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
}

struct TransportStopped {
    id: usize,
    events: mpsc::UnboundedSender<(usize, &'static str)>,
}
impl Drop for TransportStopped {
    fn drop(&mut self) {
        let _ = self.events.send((self.id, "stopped"));
    }
}

async fn stalled_connection(
    id: usize,
    events: mpsc::UnboundedSender<(usize, &'static str)>,
) -> dynamo_global_view_broadcaster::RouterConnection<&'static str, u64> {
    let (sender, stream) = mpsc::channel(1);
    let (ready, started) = oneshot::channel();
    let transport = tokio::spawn(async move {
        let _closed = TransportStopped {
            id,
            events: events.clone(),
        };
        events.send((id, "started")).unwrap();
        ready.send(()).unwrap();
        // Model an RPC held by a live peer that never drains its request body.
        std::future::pending::<()>().await;
        drop(stream);
    });
    let connection =
        dynamo_global_view_broadcaster::RouterConnection::with_transport(sender, transport);
    timeout(Duration::from_secs(2), started)
        .await
        .unwrap()
        .unwrap();
    connection
}

#[tokio::test]
async fn directory_removal_aborts_a_transport_that_never_drains_frames() {
    let broadcaster = PoolViewBroadcaster::new();
    broadcaster.upsert("blue", 1);
    let (directory, routers) = tokio::sync::watch::channel(HashSet::from(["a"]));
    let (events, mut observed) = mpsc::unbounded_channel();
    let running = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(async move {
        broadcaster
            .run(routers, move |_| {
                let events = events.clone();
                async move { Ok::<_, &'static str>(stalled_connection(0, events).await) }
            })
            .await;
    }));
    assert_eq!(
        timeout(Duration::from_secs(2), observed.recv())
            .await
            .unwrap()
            .unwrap(),
        (0, "started")
    );
    directory.send_replace(HashSet::new());
    assert_eq!(
        timeout(Duration::from_secs(2), observed.recv())
            .await
            .unwrap()
            .unwrap(),
        (0, "stopped")
    );
    drop(running);
}

#[tokio::test]
async fn backpressure_aborts_the_old_transport_before_reconnecting() {
    let broadcaster = PoolViewBroadcaster::new();
    broadcaster.upsert("blue", 1);
    let (_directory, routers) = tokio::sync::watch::channel(HashSet::from(["a"]));
    let (events, mut observed) = mpsc::unbounded_channel();
    let next = Arc::new(AtomicUsize::new(0));
    let fanout = broadcaster.clone();
    let running = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(async move {
        fanout
            .run(routers, move |_| {
                let id = next.fetch_add(1, Ordering::SeqCst);
                let events = events.clone();
                async move { Ok::<_, &'static str>(stalled_connection(id, events).await) }
            })
            .await;
    }));
    assert_eq!(
        timeout(Duration::from_secs(2), observed.recv())
            .await
            .unwrap()
            .unwrap(),
        (0, "started")
    );
    broadcaster.upsert("blue", 2);
    broadcaster.upsert("blue", 3);
    assert_eq!(
        timeout(Duration::from_secs(3), observed.recv())
            .await
            .unwrap()
            .unwrap(),
        (0, "stopped")
    );
    assert_eq!(
        timeout(Duration::from_secs(2), observed.recv())
            .await
            .unwrap()
            .unwrap(),
        (1, "started")
    );
    drop(running);
    assert_eq!(
        timeout(Duration::from_secs(2), observed.recv())
            .await
            .unwrap()
            .unwrap(),
        (1, "stopped")
    );
}
