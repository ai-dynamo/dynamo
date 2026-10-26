// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fans out a current keyed view to every router in a directory.
//!
//! Callers supply the view, router identities, and per-router frame channels. This crate does not
//! define a network transport.

use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::{broadcast, mpsc, watch};
use tokio_util::task::AbortOnDropHandle;

const UPDATE_BUFFER: usize = 64;
const RECONNECT_DELAY: Duration = Duration::from_millis(100);
const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(5);
const SEND_TIMEOUT: Duration = Duration::from_secs(1);

/// A complete view on connection, followed by revisioned changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frame<P: Eq + Hash, V> {
    Snapshot { revision: u64, pools: HashMap<P, V> },
    Upsert { revision: u64, pool: P, view: V },
    Remove { revision: u64, pool: P },
}

impl<P: Eq + Hash, V> Frame<P, V> {
    /// This sequence resets when the broadcaster starts, so receivers must
    /// pair it with the relay incarnation.
    pub fn revision(&self) -> u64 {
        match self {
            Self::Snapshot { revision, .. }
            | Self::Upsert { revision, .. }
            | Self::Remove { revision, .. } => *revision,
        }
    }
}

struct State<P, V> {
    revision: u64,
    pools: HashMap<P, V>,
}

struct Inner<P: Eq + Hash, V> {
    state: RwLock<State<P, V>>,
    updates: broadcast::Sender<Frame<P, V>>,
}

pub struct GlobalViewBroadcaster<P: Eq + Hash, V> {
    inner: Arc<Inner<P, V>>,
}

impl<P: Eq + Hash, V> Clone for GlobalViewBroadcaster<P, V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<P, V> Default for GlobalViewBroadcaster<P, V>
where
    P: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<P, V> GlobalViewBroadcaster<P, V>
where
    P: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new() -> Self {
        let (updates, _) = broadcast::channel(UPDATE_BUFFER);
        Self {
            inner: Arc::new(Inner {
                state: RwLock::new(State {
                    revision: 0,
                    pools: HashMap::new(),
                }),
                updates,
            }),
        }
    }

    /// Replace all pool views. Omitted pools are withdrawn.
    pub fn replace_snapshot(&self, pools: HashMap<P, V>) {
        let mut state = self.inner.state.write();
        state.revision += 1;
        state.pools = pools;
        if self.inner.updates.receiver_count() != 0 {
            let _ = self.inner.updates.send(Frame::Snapshot {
                revision: state.revision,
                pools: state.pools.clone(),
            });
        }
    }

    pub fn upsert(&self, pool: P, view: V) {
        let mut state = self.inner.state.write();
        state.revision += 1;
        if self.inner.updates.receiver_count() == 0 {
            state.pools.insert(pool, view);
            return;
        }
        state.pools.insert(pool.clone(), view.clone());
        let _ = self.inner.updates.send(Frame::Upsert {
            revision: state.revision,
            pool,
            view,
        });
    }

    pub fn remove(&self, pool: &P) {
        let mut state = self.inner.state.write();
        let Some((pool, _)) = state.pools.remove_entry(pool) else {
            return;
        };
        state.revision += 1;
        if self.inner.updates.receiver_count() != 0 {
            let _ = self.inner.updates.send(Frame::Remove {
                revision: state.revision,
                pool,
            });
        }
    }

    /// Open one outbound connection from the relay to each router in the directory.
    /// Closing the directory stops every task.
    pub async fn run<R, C, F, E>(&self, mut directory: watch::Receiver<HashSet<R>>, attach: C)
    where
        R: Clone + Debug + Eq + Hash + Send + Sync + 'static,
        C: Fn(R) -> F + Send + Sync + 'static,
        F: Future<Output = Result<mpsc::Sender<Frame<P, V>>, E>> + Send + 'static,
        E: Display + Send + 'static,
    {
        let attach = Arc::new(attach);
        let mut tasks: HashMap<R, AbortOnDropHandle<()>> = HashMap::new();
        loop {
            // Release the watch borrow before changing connection tasks.
            let routers = directory.borrow_and_update().clone();
            tasks.retain(|router, _| routers.contains(router));
            for router in routers {
                if tasks.contains_key(&router) {
                    continue;
                }
                let broadcaster = self.clone();
                let attacher = Arc::clone(&attach);
                let task_router = router.clone();
                tasks.insert(
                    router,
                    AbortOnDropHandle::new(tokio::spawn(async move {
                        broadcaster.forward_to_router(task_router, attacher).await;
                    })),
                );
            }
            if directory.changed().await.is_err() {
                break;
            }
        }
    }

    async fn forward_to_router<R, C, F, E>(&self, router: R, attach: Arc<C>)
    where
        R: Clone + Debug,
        C: Fn(R) -> F,
        F: Future<Output = Result<mpsc::Sender<Frame<P, V>>, E>>,
        E: Display,
    {
        let mut retry_delay = RECONNECT_DELAY;
        loop {
            let sink = match attach(router.clone()).await {
                Ok(sink) => {
                    retry_delay = RECONNECT_DELAY;
                    sink
                }
                Err(error) => {
                    tracing::warn!(?router, error = %error, "Global view attach failed");
                    tokio::time::sleep(retry_delay).await;
                    retry_delay = (retry_delay * 2).min(MAX_RECONNECT_DELAY);
                    continue;
                }
            };
            let (mut updates, snapshot) = self.snapshot_and_subscribe();
            if sink.send_timeout(snapshot, SEND_TIMEOUT).await.is_err() {
                tokio::time::sleep(RECONNECT_DELAY).await;
                continue;
            }
            loop {
                tokio::select! {
                    _ = sink.closed() => break,
                    update = updates.recv() => match update {
                        Ok(frame) => {
                            if sink.send_timeout(frame, SEND_TIMEOUT).await.is_err() {
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(?router, skipped, "Global view router lagged; reconnecting for snapshot");
                            break;
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
            tokio::time::sleep(RECONNECT_DELAY).await;
        }
    }

    fn snapshot_and_subscribe(&self) -> (broadcast::Receiver<Frame<P, V>>, Frame<P, V>) {
        let state = self.inner.state.read();
        let updates = self.inner.updates.subscribe();
        let snapshot = Frame::Snapshot {
            revision: state.revision,
            pools: state.pools.clone(),
        };
        (updates, snapshot)
    }
}
