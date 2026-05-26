// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sticky session routing with pluggable affinity storage.
//!
//! Provides router-side session affinity so that all requests within
//! a multi-turn session are routed to the same `(worker, dp_rank)`. The
//! affinity store is trait-based: the default [`InMemoryAffinityStore`] uses a
//! `DashMap` with a background reaper, but implementations backed by
//! Redis, etcd, or NATS KV can be swapped in for multi-router deployments.
//!
//! Affinity is bound at `(worker, dp_rank)` granularity so that multi-DP-rank
//! engines (e.g. SGLang DEP) keep a conversation pinned to a single DP rank,
//! which is where its prefix stays warm in the rank-local radix cache. This is
//! purely a routing-layer decision -- no RPC is sent to the worker.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_kv_router::protocols::WorkerWithDpRank;

use crate::preprocessor::PreprocessedRequest;

/// Interval between sweeps of the background reaper that removes expired entries.
const REAPER_INTERVAL: Duration = Duration::from_secs(30);

type ExpiryHandler = Arc<dyn Fn(String, u64) + Send + Sync>;

/// Trait for session affinity storage backends.
pub trait AffinityStore: Send + Sync {
    /// Look up the `(worker, dp_rank)` for a session. Returns `None` if unknown
    /// or expired. Implementations should refresh the TTL on hit.
    fn get(&self, session_id: &str) -> Option<WorkerWithDpRank>;

    /// Look up the `(worker, dp_rank)` for a session without refreshing TTL.
    fn peek(&self, session_id: &str) -> Option<WorkerWithDpRank>;

    /// Bind a session to a `(worker, dp_rank)` with the given TTL.
    fn put(&self, session_id: &str, worker: WorkerWithDpRank, ttl: Duration);

    /// Remove a session binding.
    fn remove(&self, session_id: &str);
}

/// In-memory affinity entry with sliding-window TTL.
struct AffinityEntry {
    worker: WorkerWithDpRank,
    ttl: Duration,
    expires_at: Instant,
}

/// Default in-memory affinity store backed by `DashMap`.
///
/// A background tokio task sweeps expired entries every [`REAPER_INTERVAL`].
#[derive(Clone)]
pub struct InMemoryAffinityStore {
    map: Arc<DashMap<String, AffinityEntry>>,
    on_expire: Option<ExpiryHandler>,
}

impl Default for InMemoryAffinityStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryAffinityStore {
    pub fn new() -> Self {
        Self::new_with_on_expire(None)
    }

    pub fn new_with_on_expire(on_expire: Option<ExpiryHandler>) -> Self {
        let map = Arc::new(DashMap::new());

        let store = InMemoryAffinityStore { map, on_expire };

        let reaper_store = store.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(REAPER_INTERVAL);
            loop {
                interval.tick().await;
                reaper_store.reap_expired(Instant::now());
            }
        });

        store
    }

    fn reap_expired(&self, now: Instant) {
        let on_expire = self.on_expire.clone();
        self.map.retain(|session_id, entry: &mut AffinityEntry| {
            let alive = entry.expires_at > now;
            if !alive {
                tracing::debug!(%session_id, "Session affinity expired, removing");
                if let Some(handler) = &on_expire {
                    handler(session_id.clone(), entry.worker.worker_id);
                }
            }
            alive
        });
    }

    fn lookup(&self, session_id: &str, refresh: bool) -> Option<WorkerWithDpRank> {
        let now = Instant::now();
        let mut entry = self.map.get_mut(session_id)?;
        if entry.expires_at <= now {
            let worker = entry.worker;
            let expires_at = entry.expires_at;
            drop(entry);
            self.remove_expired_if_current(session_id, worker, expires_at);
            return None;
        }

        let worker = entry.worker;
        if refresh {
            entry.expires_at = now + entry.ttl;
        }
        tracing::info!(
            %session_id,
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            refreshed = refresh,
            "Sticky session hit"
        );
        Some(worker)
    }

    fn remove_expired_if_current(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        expires_at: Instant,
    ) {
        let removed = self.map.remove_if(session_id, |_, entry| {
            entry.worker == worker
                && entry.expires_at == expires_at
                && entry.expires_at <= Instant::now()
        });
        if removed.is_none() {
            return;
        }

        tracing::debug!(%session_id, "Session affinity expired during resolve");
        if let Some(handler) = &self.on_expire {
            handler(session_id.to_owned(), worker.worker_id);
        }
    }
}

impl AffinityStore for InMemoryAffinityStore {
    fn get(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.lookup(session_id, true)
    }

    fn peek(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.lookup(session_id, false)
    }

    fn put(&self, session_id: &str, worker: WorkerWithDpRank, ttl: Duration) {
        self.map.insert(
            session_id.to_owned(),
            AffinityEntry {
                worker,
                ttl,
                expires_at: Instant::now() + ttl,
            },
        );
    }

    fn remove(&self, session_id: &str) {
        self.map.remove(session_id);
    }
}

/// Routes requests to workers based on session affinity.
///
/// Wraps an [`AffinityStore`] and provides request-level helpers
/// that extract session IDs from [`PreprocessedRequest`] routing hints.
pub struct StickySessionRouter {
    store: Box<dyn AffinityStore>,
}

impl StickySessionRouter {
    pub fn new(store: impl AffinityStore + 'static) -> Self {
        tracing::debug!("StickySessionRouter initialized");
        StickySessionRouter {
            store: Box::new(store),
        }
    }

    /// Resolve a request's session to a pinned `(worker, dp_rank)`.
    ///
    /// Looks up `session_control.session_id` from the request's routing hints.
    /// Returns `None` if no session control is present or the session is unknown/expired.
    pub fn resolve(&self, request: &PreprocessedRequest) -> Option<WorkerWithDpRank> {
        let routing = request.routing.as_ref()?;
        let session_id = routing
            .session_control
            .as_ref()
            .map(|sc| sc.session_id.as_str())?;
        self.store.get(session_id)
    }

    /// Resolve a request's session without refreshing the sticky TTL.
    pub fn peek(&self, request: &PreprocessedRequest) -> Option<WorkerWithDpRank> {
        let routing = request.routing.as_ref()?;
        let session_id = routing
            .session_control
            .as_ref()
            .map(|sc| sc.session_id.as_str())?;
        self.store.peek(session_id)
    }

    /// Resolve a session id directly and refresh its sticky TTL.
    pub fn resolve_session(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.store.get(session_id)
    }

    /// Resolve a session id directly without refreshing its sticky TTL.
    pub fn peek_session(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.store.peek(session_id)
    }

    /// Bind a session to a `(worker, dp_rank)` with the given TTL.
    pub fn bind(&self, session_id: &str, worker: WorkerWithDpRank, ttl: Duration) {
        tracing::info!(%session_id, worker_id = worker.worker_id, dp_rank = worker.dp_rank, ttl_secs = ttl.as_secs(), "Binding session affinity");
        self.store.put(session_id, worker, ttl);
    }

    /// Remove a session binding.
    pub fn unbind(&self, session_id: &str) {
        tracing::info!(%session_id, "Removing session affinity");
        self.store.remove(session_id);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::protocols::common::preprocessor::{PreprocessedRequest, RoutingHints};
    use crate::protocols::openai::nvext::SessionControl;

    fn worker(worker_id: u64, dp_rank: u32) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, dp_rank)
    }

    fn make_request(session_id: Option<&str>) -> PreprocessedRequest {
        let routing = session_id.map(|id| RoutingHints {
            session_control: Some(SessionControl {
                session_id: id.to_owned(),
                action: None,
                timeout: 300,
            }),
            ..Default::default()
        });
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .routing(routing)
            .build()
            .unwrap()
    }

    #[test]
    fn resolve_returns_none_for_unknown_session() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        let req = make_request(Some("unknown-session"));
        assert!(router.resolve(&req).is_none());
    }

    #[test]
    fn resolve_returns_none_when_no_session_id() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        let req = make_request(None);
        assert!(router.resolve(&req).is_none());
    }

    #[test]
    fn bind_then_resolve_returns_worker_and_dp_rank() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        router.bind("sess-1", worker(42, 3), Duration::from_secs(300));

        let req = make_request(Some("sess-1"));
        assert_eq!(router.resolve(&req), Some(worker(42, 3)));
    }

    #[test]
    fn peek_returns_worker_without_refreshing_ttl() {
        let map = Arc::new(DashMap::new());
        let ttl = Duration::from_secs(60);
        let expires_at = Instant::now() + Duration::from_secs(5);
        map.insert(
            "sess-peek".to_owned(),
            AffinityEntry {
                worker: worker(7, 2),
                ttl,
                expires_at,
            },
        );
        let store = InMemoryAffinityStore {
            map: map.clone(),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);

        let req = make_request(Some("sess-peek"));
        assert_eq!(router.peek(&req), Some(worker(7, 2)));

        let entry = map.get("sess-peek").unwrap();
        assert_eq!(entry.expires_at, expires_at);
    }

    #[test]
    fn bind_overwrites_worker_rank_and_ttl() {
        let map = Arc::new(DashMap::new());
        let store = InMemoryAffinityStore {
            map: map.clone(),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        router.bind("sess-1", worker(1, 0), Duration::from_secs(10));
        router.bind("sess-1", worker(2, 3), Duration::from_secs(90));

        let req = make_request(Some("sess-1"));
        assert_eq!(router.peek(&req), Some(worker(2, 3)));

        let entry = map.get("sess-1").unwrap();
        assert_eq!(entry.worker, worker(2, 3));
        assert_eq!(entry.ttl, Duration::from_secs(90));
        assert!(entry.expires_at > Instant::now() + Duration::from_secs(80));
    }

    #[test]
    fn unbind_removes_affinity() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        router.bind("sess-1", worker(42, 1), Duration::from_secs(300));
        router.unbind("sess-1");

        let req = make_request(Some("sess-1"));
        assert!(router.resolve(&req).is_none());
    }

    #[test]
    fn expired_entry_returns_none() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        // Insert with zero TTL so it's already expired
        store.map.insert(
            "sess-expired".to_owned(),
            AffinityEntry {
                worker: worker(99, 0),
                ttl: Duration::from_secs(0),
                expires_at: Instant::now() - Duration::from_secs(1),
            },
        );
        let router = StickySessionRouter::new(store);

        let req = make_request(Some("sess-expired"));
        assert!(router.resolve(&req).is_none());
        // Entry should be cleaned up
        assert!(router.store.peek("sess-expired").is_none());
    }

    #[test]
    fn resolve_refreshes_ttl() {
        let map = Arc::new(DashMap::new());
        let ttl = Duration::from_secs(60);
        map.insert(
            "sess-refresh".to_owned(),
            AffinityEntry {
                worker: worker(7, 2),
                ttl,
                // Expires in 5 seconds (simulating time passing since bind)
                expires_at: Instant::now() + Duration::from_secs(5),
            },
        );
        let store = InMemoryAffinityStore {
            map: map.clone(),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);

        let req = make_request(Some("sess-refresh"));
        assert_eq!(router.resolve(&req), Some(worker(7, 2)));

        // After resolve, expires_at should be refreshed to now + ttl (60s),
        // so it should be at least 50s from now (not the original 5s).
        let entry = map.get("sess-refresh").unwrap();
        let remaining = entry.expires_at.duration_since(Instant::now());
        assert!(
            remaining > Duration::from_secs(50),
            "TTL should have been refreshed, but remaining={remaining:?}"
        );
    }

    #[test]
    fn expired_entry_triggers_close_callback_on_resolve() {
        let expired_sessions = Arc::new(Mutex::new(Vec::new()));
        let on_expire = {
            let expired_sessions = expired_sessions.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                expired_sessions
                    .lock()
                    .unwrap()
                    .push((session_id, worker_id));
            })
        };
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: Some(on_expire),
        };
        store.map.insert(
            "sess-expired".to_owned(),
            AffinityEntry {
                worker: worker(99, 0),
                ttl: Duration::from_secs(0),
                expires_at: Instant::now() - Duration::from_secs(1),
            },
        );
        let router = StickySessionRouter::new(store);

        let req = make_request(Some("sess-expired"));
        assert!(router.resolve(&req).is_none());
        assert_eq!(
            expired_sessions.lock().unwrap().as_slice(),
            &[("sess-expired".to_string(), 99)]
        );
    }

    #[test]
    fn expired_lookup_does_not_remove_newer_binding() {
        let expired_sessions = Arc::new(Mutex::new(Vec::new()));
        let on_expire = {
            let expired_sessions = expired_sessions.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                expired_sessions
                    .lock()
                    .unwrap()
                    .push((session_id, worker_id));
            })
        };
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: Some(on_expire),
        };
        store.map.insert(
            "sess-race".to_owned(),
            AffinityEntry {
                worker: worker(1, 0),
                ttl: Duration::from_secs(1),
                expires_at: Instant::now() - Duration::from_secs(1),
            },
        );

        let stale = store.map.get("sess-race").unwrap();
        let stale_worker = stale.worker;
        let stale_expires_at = stale.expires_at;
        drop(stale);

        store.put("sess-race", worker(2, 1), Duration::from_secs(300));
        store.remove_expired_if_current("sess-race", stale_worker, stale_expires_at);

        assert_eq!(store.peek("sess-race"), Some(worker(2, 1)));
        assert!(expired_sessions.lock().unwrap().is_empty());
    }

    #[test]
    fn reaper_triggers_close_callback_for_expired_entry() {
        let expired_sessions = Arc::new(Mutex::new(Vec::new()));
        let on_expire = {
            let expired_sessions = expired_sessions.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                expired_sessions
                    .lock()
                    .unwrap()
                    .push((session_id, worker_id));
            })
        };
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: Some(on_expire),
        };
        store.map.insert(
            "sess-reaped".to_owned(),
            AffinityEntry {
                worker: worker(17, 0),
                ttl: Duration::from_secs(30),
                expires_at: Instant::now() - Duration::from_secs(1),
            },
        );

        store.reap_expired(Instant::now());

        assert!(store.map.get("sess-reaped").is_none());
        assert_eq!(
            expired_sessions.lock().unwrap().as_slice(),
            &[("sess-reaped".to_string(), 17)]
        );
    }
}
