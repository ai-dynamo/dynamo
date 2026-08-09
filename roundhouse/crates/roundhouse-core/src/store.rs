// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable session state.
//!
//! The trait is deliberately small. Conversation items and the routing ledger
//! are *projections* of the event log rather than separately stored
//! collections, so there is exactly one write path and no way for the log and
//! the materialized state to disagree after a crash. A backend therefore only
//! has to provide an append-only log plus a lease.
//!
//! Two implementations are expected: the [`MemoryStore`] here (tests, single
//! process) and a Redis Streams backend in `roundhouse-store-redis`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::event::{SessionEvent, SessionEventKind};
use crate::ids::SessionId;
use crate::now_ms;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("session `{0}` not found")]
    SessionNotFound(SessionId),
    #[error("lease for session `{session_id}` is not held by `{node_id}`")]
    LeaseLost {
        session_id: SessionId,
        node_id: String,
    },
    #[error("backend failure: {0}")]
    Backend(#[from] anyhow::Error),
}

/// Proof that a node is the single writer for a session.
///
/// Every mutating call takes one. A node whose lease has expired — because it
/// stalled, or was partitioned, or died and came back — fails its next append
/// rather than writing behind the successor that took over. This is the only
/// thing standing between a failover and a split-brain log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    pub session_id: SessionId,
    pub node_id: String,
    pub expires_at_ms: u64,
}

impl Lease {
    pub fn is_expired_at(&self, now: u64) -> bool {
        now >= self.expires_at_ms
    }
}

#[async_trait]
pub trait SessionStore: Send + Sync + 'static {
    /// Create a session. Returns `false` if it already existed.
    async fn create_session(
        &self,
        session_id: &SessionId,
        model_policy: &str,
    ) -> Result<bool, StoreError>;

    /// Claim single-writer ownership, or `None` if another live node holds it.
    async fn acquire_lease(
        &self,
        session_id: &SessionId,
        node_id: &str,
        ttl_ms: u64,
    ) -> Result<Option<Lease>, StoreError>;

    /// Extend a held lease. `None` means it was lost and must be re-acquired.
    async fn renew_lease(&self, lease: &Lease, ttl_ms: u64) -> Result<Option<Lease>, StoreError>;

    async fn release_lease(&self, lease: &Lease) -> Result<(), StoreError>;

    /// Append events, assigning contiguous sequence numbers.
    ///
    /// Fails with [`StoreError::LeaseLost`] if `lease` is no longer valid, so a
    /// stalled writer cannot interleave with its successor.
    async fn append_events(
        &self,
        lease: &Lease,
        kinds: Vec<SessionEventKind>,
    ) -> Result<Vec<SessionEvent>, StoreError>;

    /// Read events with `seq > after_seq`, oldest first.
    async fn read_events(
        &self,
        session_id: &SessionId,
        after_seq: u64,
        limit: usize,
    ) -> Result<Vec<SessionEvent>, StoreError>;

    /// Highest assigned sequence number, or 0 for an empty session.
    async fn last_seq(&self, session_id: &SessionId) -> Result<u64, StoreError>;
}

// ---------------------------------------------------------------------------
// In-memory implementation
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SessionRecord {
    model_policy: String,
    events: Vec<SessionEvent>,
    lease: Option<Lease>,
}

/// Non-durable [`SessionStore`] for tests and single-process runs.
///
/// Lease semantics are modelled faithfully — including expiry and takeover —
/// so failover logic can be tested without standing up Redis.
#[derive(Default, Clone)]
pub struct MemoryStore {
    sessions: Arc<RwLock<HashMap<SessionId, SessionRecord>>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Force-expire a session's lease. Test hook for simulating a dead owner
    /// without waiting out a TTL.
    pub async fn expire_lease_now(&self, session_id: &SessionId) {
        if let Some(record) = self.sessions.write().await.get_mut(session_id)
            && let Some(lease) = record.lease.as_mut()
        {
            lease.expires_at_ms = 0;
        }
    }
}

#[async_trait]
impl SessionStore for MemoryStore {
    async fn create_session(
        &self,
        session_id: &SessionId,
        model_policy: &str,
    ) -> Result<bool, StoreError> {
        let mut sessions = self.sessions.write().await;
        if sessions.contains_key(session_id) {
            return Ok(false);
        }
        sessions.insert(
            session_id.clone(),
            SessionRecord {
                model_policy: model_policy.to_string(),
                ..Default::default()
            },
        );
        Ok(true)
    }

    async fn acquire_lease(
        &self,
        session_id: &SessionId,
        node_id: &str,
        ttl_ms: u64,
    ) -> Result<Option<Lease>, StoreError> {
        let mut sessions = self.sessions.write().await;
        let record = sessions
            .get_mut(session_id)
            .ok_or_else(|| StoreError::SessionNotFound(session_id.clone()))?;

        let now = now_ms();
        // A live lease held by someone else blocks the claim; our own lease is
        // simply re-taken, which makes acquisition idempotent for a node that
        // is recovering rather than competing.
        if let Some(existing) = &record.lease
            && !existing.is_expired_at(now)
            && existing.node_id != node_id
        {
            return Ok(None);
        }

        let lease = Lease {
            session_id: session_id.clone(),
            node_id: node_id.to_string(),
            expires_at_ms: now + ttl_ms,
        };
        record.lease = Some(lease.clone());
        Ok(Some(lease))
    }

    async fn renew_lease(&self, lease: &Lease, ttl_ms: u64) -> Result<Option<Lease>, StoreError> {
        let mut sessions = self.sessions.write().await;
        let record = sessions
            .get_mut(&lease.session_id)
            .ok_or_else(|| StoreError::SessionNotFound(lease.session_id.clone()))?;

        match &record.lease {
            Some(current) if current.node_id == lease.node_id && !current.is_expired_at(now_ms()) => {
                let renewed = Lease {
                    expires_at_ms: now_ms() + ttl_ms,
                    ..lease.clone()
                };
                record.lease = Some(renewed.clone());
                Ok(Some(renewed))
            }
            _ => Ok(None),
        }
    }

    async fn release_lease(&self, lease: &Lease) -> Result<(), StoreError> {
        let mut sessions = self.sessions.write().await;
        if let Some(record) = sessions.get_mut(&lease.session_id)
            && record
                .lease
                .as_ref()
                .is_some_and(|current| current.node_id == lease.node_id)
        {
            record.lease = None;
        }
        Ok(())
    }

    async fn append_events(
        &self,
        lease: &Lease,
        kinds: Vec<SessionEventKind>,
    ) -> Result<Vec<SessionEvent>, StoreError> {
        let mut sessions = self.sessions.write().await;
        let record = sessions
            .get_mut(&lease.session_id)
            .ok_or_else(|| StoreError::SessionNotFound(lease.session_id.clone()))?;

        let held = record
            .lease
            .as_ref()
            .is_some_and(|current| current.node_id == lease.node_id && !current.is_expired_at(now_ms()));
        if !held {
            return Err(StoreError::LeaseLost {
                session_id: lease.session_id.clone(),
                node_id: lease.node_id.clone(),
            });
        }

        let at_ms = now_ms();
        let mut appended = Vec::with_capacity(kinds.len());
        for kind in kinds {
            let seq = record.events.len() as u64 + 1;
            let event = SessionEvent {
                seq,
                session_id: lease.session_id.clone(),
                at_ms,
                kind,
            };
            record.events.push(event.clone());
            appended.push(event);
        }
        Ok(appended)
    }

    async fn read_events(
        &self,
        session_id: &SessionId,
        after_seq: u64,
        limit: usize,
    ) -> Result<Vec<SessionEvent>, StoreError> {
        let sessions = self.sessions.read().await;
        let record = sessions
            .get(session_id)
            .ok_or_else(|| StoreError::SessionNotFound(session_id.clone()))?;
        Ok(record
            .events
            .iter()
            .filter(|event| event.seq > after_seq)
            .take(limit)
            .cloned()
            .collect())
    }

    async fn last_seq(&self, session_id: &SessionId) -> Result<u64, StoreError> {
        let sessions = self.sessions.read().await;
        let record = sessions
            .get(session_id)
            .ok_or_else(|| StoreError::SessionNotFound(session_id.clone()))?;
        Ok(record.events.last().map_or(0, |event| event.seq))
    }
}

impl MemoryStore {
    /// Test accessor for the policy recorded at creation.
    pub async fn model_policy(&self, session_id: &SessionId) -> Option<String> {
        self.sessions
            .read()
            .await
            .get(session_id)
            .map(|record| record.model_policy.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn store_with_session() -> (MemoryStore, SessionId) {
        let store = MemoryStore::new();
        let sid = SessionId::generate();
        assert!(store.create_session(&sid, "affinity").await.unwrap());
        (store, sid)
    }

    #[tokio::test]
    async fn create_is_idempotent_and_reports_existing() {
        let (store, sid) = store_with_session().await;
        assert!(!store.create_session(&sid, "affinity").await.unwrap());
    }

    #[tokio::test]
    async fn a_live_lease_blocks_another_node() {
        let (store, sid) = store_with_session().await;
        assert!(store.acquire_lease(&sid, "node-a", 5_000).await.unwrap().is_some());
        assert!(store.acquire_lease(&sid, "node-b", 5_000).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn an_expired_lease_is_takeable_and_the_loser_cannot_append() {
        let (store, sid) = store_with_session().await;
        let stale = store.acquire_lease(&sid, "node-a", 5_000).await.unwrap().unwrap();

        store.expire_lease_now(&sid).await;
        let fresh = store.acquire_lease(&sid, "node-b", 5_000).await.unwrap().unwrap();

        // The successor writes; the displaced owner is fenced out.
        store
            .append_events(&fresh, vec![SessionEventKind::Error { message: "ok".into() }])
            .await
            .unwrap();
        let err = store
            .append_events(&stale, vec![SessionEventKind::Error { message: "no".into() }])
            .await
            .unwrap_err();
        assert!(matches!(err, StoreError::LeaseLost { .. }));
    }

    #[tokio::test]
    async fn sequence_numbers_are_contiguous_and_replay_is_gapless() {
        let (store, sid) = store_with_session().await;
        let lease = store.acquire_lease(&sid, "node-a", 5_000).await.unwrap().unwrap();

        let first = store
            .append_events(
                &lease,
                vec![
                    SessionEventKind::SessionCreated { model_policy: "affinity".into() },
                    SessionEventKind::Error { message: "one".into() },
                ],
            )
            .await
            .unwrap();
        assert_eq!(first.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![1, 2]);

        let second = store
            .append_events(&lease, vec![SessionEventKind::Error { message: "two".into() }])
            .await
            .unwrap();
        assert_eq!(second[0].seq, 3);
        assert_eq!(store.last_seq(&sid).await.unwrap(), 3);

        // Resume from seq 1 yields exactly the tail, with no gap or repeat.
        let tail = store.read_events(&sid, 1, 100).await.unwrap();
        assert_eq!(tail.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![2, 3]);
    }

    #[tokio::test]
    async fn renew_fails_once_the_lease_was_taken_over() {
        let (store, sid) = store_with_session().await;
        let lease = store.acquire_lease(&sid, "node-a", 5_000).await.unwrap().unwrap();
        assert!(store.renew_lease(&lease, 5_000).await.unwrap().is_some());

        store.expire_lease_now(&sid).await;
        store.acquire_lease(&sid, "node-b", 5_000).await.unwrap().unwrap();
        assert!(store.renew_lease(&lease, 5_000).await.unwrap().is_none());
    }
}
