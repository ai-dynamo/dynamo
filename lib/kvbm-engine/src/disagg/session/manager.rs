// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`SessionManager`] — keeps opened [`Session`]s alive and evicts them
//! when their lifecycle ends.
//!
//! `VeloSessionFactory` returns an `Arc<dyn Session>` and immediately forgets
//! it — the caller owns the only handle. A handler that opens a session and
//! returns must therefore park the session somewhere, or it tears down before
//! the peer can attach. `SessionManager` is that home: it holds the
//! `Arc<dyn Session>` in a map keyed by [`SessionId`] and spawns a per-session
//! watcher that removes the entry when the session detaches, fails, or a
//! watchdog timeout elapses.
//!
//! This mirrors the connector coordinator's `spawn_lifecycle_watcher` pattern
//! (`kvbm-connector/.../disagg/lifecycle.rs`) but stands alone; unifying the
//! two is a flagged follow-up.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use futures::StreamExt;
use tokio::runtime::Handle;

use super::{LifecycleEvent, Session, SessionId};

/// Default watchdog: evict a session that has neither detached nor failed
/// within this window. Guards against callers that open a session but whose
/// peer never attaches.
pub const DEFAULT_SESSION_WATCHDOG: Duration = Duration::from_secs(30);

/// Tracks live [`Session`]s by [`SessionId`] and auto-evicts them on
/// lifecycle termination.
pub struct SessionManager {
    sessions: DashMap<SessionId, Arc<dyn Session>>,
    runtime: Handle,
    watchdog: Duration,
}

impl SessionManager {
    /// Create a manager that spawns watcher tasks on `runtime` and evicts
    /// un-terminated sessions after `watchdog`.
    pub fn new(runtime: Handle, watchdog: Duration) -> Arc<Self> {
        Arc::new(Self {
            sessions: DashMap::new(),
            runtime,
            watchdog,
        })
    }

    /// Convenience constructor using [`DEFAULT_SESSION_WATCHDOG`].
    pub fn with_default_watchdog(runtime: Handle) -> Arc<Self> {
        Self::new(runtime, DEFAULT_SESSION_WATCHDOG)
    }

    /// Park a session: insert it into the map and spawn a watcher that
    /// evicts it on `Detached` / `Failed` / watchdog timeout.
    pub fn register(self: &Arc<Self>, session: Arc<dyn Session>) {
        let session_id = session.session_id();
        self.sessions.insert(session_id, Arc::clone(&session));
        self.spawn_watcher(session_id, session);
    }

    /// Look up a live session by id.
    pub fn get(&self, session_id: &SessionId) -> Option<Arc<dyn Session>> {
        self.sessions.get(session_id).map(|e| Arc::clone(&*e))
    }

    /// Remove (and return) a session explicitly. Normally the watcher does
    /// this; callers may use it for early teardown.
    pub fn remove(&self, session_id: &SessionId) -> Option<Arc<dyn Session>> {
        self.sessions.remove(session_id).map(|(_, s)| s)
    }

    /// Number of sessions currently parked.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether any sessions are parked.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    fn spawn_watcher(self: &Arc<Self>, session_id: SessionId, session: Arc<dyn Session>) {
        let manager = Arc::clone(self);
        let watchdog = self.watchdog;
        self.runtime.spawn(async move {
            // Hold `session` for the watcher's lifetime so the map entry is
            // not the only thing keeping it alive while we watch.
            let mut lifecycle = session.lifecycle();
            let outcome: String = loop {
                match tokio::time::timeout(watchdog, lifecycle.next()).await {
                    Ok(Some(LifecycleEvent::Attached { .. })) => continue,
                    Ok(Some(LifecycleEvent::Detached { reason })) => {
                        break format!("detached ({reason:?})");
                    }
                    Ok(Some(LifecycleEvent::Failed { reason })) => {
                        break format!("failed ({reason})");
                    }
                    Ok(None) => break "lifecycle stream ended".to_string(),
                    Err(_) => break "watchdog timeout".to_string(),
                }
            };
            manager.sessions.remove(&session_id);
            tracing::info!(%session_id, outcome, "SessionManager evicted session");
            drop(session);
        });
    }
}
