// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sticky session routing with pluggable affinity storage.
//!
//! Provides router-side session affinity so that all requests within
//! a multi-turn session are routed to the same worker. The affinity
//! store is trait-based: the default [`InMemoryAffinityStore`] uses a
//! `DashMap` with a background reaper, but implementations backed by
//! Redis, etcd, or NATS KV can be swapped in for multi-router deployments.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::preprocessor::PreprocessedRequest;

/// Interval between sweeps of the background reaper that removes expired entries.
const REAPER_INTERVAL: Duration = Duration::from_secs(30);

type ExpiryHandler = Arc<dyn Fn(String, u64) + Send + Sync>;

/// Trait for session affinity storage backends.
pub trait AffinityStore: Send + Sync {
    /// Look up the worker for a session. Returns `None` if unknown or expired.
    /// Implementations should refresh the TTL on hit.
    fn get(&self, session_id: &str) -> Option<u64>;

    /// Bind a session to a worker with the given TTL.
    fn put(&self, session_id: &str, worker_id: u64, ttl: Duration);

    /// Remove a session binding.
    fn remove(&self, session_id: &str);
}

/// In-memory affinity entry with sliding-window TTL.
struct AffinityEntry {
    worker_id: u64,
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
                    handler(session_id.clone(), entry.worker_id);
                }
            }
            alive
        });
    }
}

impl AffinityStore for InMemoryAffinityStore {
    fn get(&self, session_id: &str) -> Option<u64> {
        let mut entry = self.map.get_mut(session_id)?;
        if entry.expires_at <= Instant::now() {
            let worker_id = entry.worker_id;
            drop(entry);
            self.map.remove(session_id);
            tracing::debug!(%session_id, "Session affinity expired during resolve");
            if let Some(handler) = &self.on_expire {
                handler(session_id.to_owned(), worker_id);
            }
            return None;
        }
        // Refresh TTL on access (sliding window)
        entry.expires_at = Instant::now() + entry.ttl;
        let worker_id = entry.worker_id;
        tracing::info!(%session_id, worker_id, "Sticky session hit");
        Some(worker_id)
    }

    fn put(&self, session_id: &str, worker_id: u64, ttl: Duration) {
        self.map.insert(
            session_id.to_owned(),
            AffinityEntry {
                worker_id,
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

    /// Resolve a request's session to a pinned worker.
    ///
    /// Looks up `session_control.session_id` from the request's routing hints.
    /// Returns `None` if no session control is present or the session is unknown/expired.
    pub fn resolve(&self, request: &PreprocessedRequest) -> Option<u64> {
        let routing = request.routing.as_ref()?;
        let session_id = routing
            .session_control
            .as_ref()
            .map(|sc| sc.session_id.as_str())?;
        self.store.get(session_id)
    }

    /// Bind a session to a worker with the given TTL.
    pub fn bind(&self, session_id: &str, worker_id: u64, ttl: Duration) {
        tracing::info!(%session_id, worker_id, ttl_secs = ttl.as_secs(), "Binding session affinity");
        self.store.put(session_id, worker_id, ttl);
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
    fn bind_then_resolve_returns_worker() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        router.bind("sess-1", 42, Duration::from_secs(300));

        let req = make_request(Some("sess-1"));
        assert_eq!(router.resolve(&req), Some(42));
    }

    #[test]
    fn unbind_removes_affinity() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        router.bind("sess-1", 42, Duration::from_secs(300));
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
                worker_id: 99,
                ttl: Duration::from_secs(0),
                expires_at: Instant::now() - Duration::from_secs(1),
            },
        );
        let router = StickySessionRouter::new(store);

        let req = make_request(Some("sess-expired"));
        assert!(router.resolve(&req).is_none());
        // Entry should be cleaned up
        assert!(router.store.get("sess-expired").is_none());
    }

    #[test]
    fn resolve_refreshes_ttl() {
        let map = Arc::new(DashMap::new());
        let ttl = Duration::from_secs(60);
        map.insert(
            "sess-refresh".to_owned(),
            AffinityEntry {
                worker_id: 7,
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
        assert_eq!(router.resolve(&req), Some(7));

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
                worker_id: 99,
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
                worker_id: 17,
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
