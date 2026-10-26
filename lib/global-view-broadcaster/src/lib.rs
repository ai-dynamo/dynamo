// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fans out a current keyed view to every router in a directory.
//!
//! Callers supply the view, router identities, and per-router frame channels. This crate does not
//! define a network transport.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::JoinHandle;

const UPDATE_BUFFER: usize = 64;
const RECONNECT_DELAY: Duration = Duration::from_millis(100);
const SEND_TIMEOUT: Duration = Duration::from_secs(1);

/// A complete view on connection, followed by revisioned changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frame<P: Eq + Hash, V> {
    Snapshot { revision: u64, pools: HashMap<P, V> },
    Upsert { revision: u64, pool: P, view: V },
    Remove { revision: u64, pool: P },
}

impl<P: Eq + Hash, V> Frame<P, V> {
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

/// Stores a keyed view and fans out snapshots and updates to router channels.
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

    /// Maintain one connection task for each router in the current directory.
    /// Closing the directory stops every task.
    pub async fn run<R, C, F, E>(&self, mut directory: watch::Receiver<HashSet<R>>, attach: C)
    where
        R: Clone + Eq + Hash + Send + Sync + 'static,
        C: Fn(R) -> F + Send + Sync + 'static,
        F: Future<Output = Result<mpsc::Sender<Frame<P, V>>, E>> + Send + 'static,
        E: Send + 'static,
    {
        let attach = Arc::new(attach);
        let mut tasks = RouterTasks {
            handles: HashMap::new(),
        };
        loop {
            let routers = directory.borrow_and_update().clone();
            tasks.handles.retain(|router, task| {
                if routers.contains(router) {
                    true
                } else {
                    task.abort();
                    false
                }
            });
            for router in routers {
                if tasks.handles.contains_key(&router) {
                    continue;
                }
                let broadcaster = self.clone();
                let attacher = Arc::clone(&attach);
                let task_router = router.clone();
                let task = tokio::spawn(async move {
                    broadcaster.forward_to_router(task_router, attacher).await;
                });
                tasks.handles.insert(router, task);
            }
            if directory.changed().await.is_err() {
                break;
            }
        }
    }

    async fn forward_to_router<R, C, F, E>(&self, router: R, attach: Arc<C>)
    where
        R: Clone,
        C: Fn(R) -> F,
        F: Future<Output = Result<mpsc::Sender<Frame<P, V>>, E>>,
    {
        loop {
            let mut updates = self.inner.updates.subscribe();
            let Ok(sink) = attach(router.clone()).await else {
                tokio::time::sleep(RECONNECT_DELAY).await;
                continue;
            };
            let (revision, pools) = {
                let state = self.inner.state.read();
                (state.revision, state.pools.clone())
            };
            if sink
                .send_timeout(Frame::Snapshot { revision, pools }, SEND_TIMEOUT)
                .await
                .is_err()
            {
                tokio::time::sleep(RECONNECT_DELAY).await;
                continue;
            }
            let mut seen = revision;
            loop {
                tokio::select! {
                    _ = sink.closed() => break,
                    update = updates.recv() => match update {
                        Ok(frame) if frame.revision() > seen => {
                            let revision = frame.revision();
                            if sink.send_timeout(frame, SEND_TIMEOUT).await.is_err() {
                                break;
                            }
                            seen = revision;
                        }
                        Ok(_) => {}
                        Err(_) => break, // Resnapshot after lag or stream closure.
                    }
                }
            }
            tokio::time::sleep(RECONNECT_DELAY).await;
        }
    }
}

struct RouterTasks<R> {
    handles: HashMap<R, JoinHandle<()>>,
}

impl<R> Drop for RouterTasks<R> {
    fn drop(&mut self) {
        for task in self.handles.values() {
            task.abort();
        }
    }
}
