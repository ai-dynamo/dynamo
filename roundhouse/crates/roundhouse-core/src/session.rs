// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The session state machine.
//!
//! A [`Session`] is a lease plus a projection. Everything it knows — the
//! conversation, the routing ledger, which turns are already done — is derived
//! by replaying the event log, so a successor process that acquires the lease
//! reconstructs identical state without any handoff from the process it
//! replaced.

use std::collections::HashMap;
use std::sync::Arc;

use crate::event::{IncompleteReason, SessionEvent, SessionEventKind, Usage};
use crate::ids::{ResponseId, SessionId, TurnId};
use crate::item::Item;
use crate::routing::{CacheLedger, DecisionRecord};
use crate::store::{Lease, SessionStore, StoreError};

/// How many events to pull per replay batch.
const REPLAY_BATCH: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("session `{0}` is owned by another node")]
    NotOwner(SessionId),
    #[error("response `{0}` is not open")]
    ResponseNotOpen(ResponseId),
}

/// The result of admitting a turn.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnAdmission {
    /// A new turn was opened.
    Started(ResponseId),
    /// The turn id was already completed; the existing response is replayed
    /// rather than a second one being generated.
    Deduplicated(ResponseId),
}

impl TurnAdmission {
    pub fn response_id(&self) -> &ResponseId {
        match self {
            TurnAdmission::Started(id) | TurnAdmission::Deduplicated(id) => id,
        }
    }
}

/// State derived from the event log.
#[derive(Default)]
pub struct SessionState {
    pub items: Vec<Item>,
    pub ledger: CacheLedger,
    /// Number of turns started so far; the index the next turn will use.
    pub turn_index: u64,
    /// Turn ids that reached a terminal state, and what they produced.
    completed_turns: HashMap<TurnId, ResponseId>,
    /// Turn ids currently in flight.
    open_turns: HashMap<TurnId, ResponseId>,
    pub last_seq: u64,
}

impl SessionState {
    /// Fold one event into the projection.
    ///
    /// The exhaustive match is deliberate: a new event kind should not silently
    /// fail to update derived state.
    fn apply(&mut self, event: &SessionEvent) {
        self.last_seq = event.seq;
        match &event.kind {
            SessionEventKind::ItemAppended { item } => self.items.push(item.clone()),
            SessionEventKind::TurnStarted {
                turn_id,
                response_id,
            } => {
                self.turn_index += 1;
                self.open_turns.insert(turn_id.clone(), response_id.clone());
            }
            SessionEventKind::Routed { decision, .. } => {
                self.ledger
                    .record(&decision.chosen, event.at_ms, decision.isl_tokens);
            }
            SessionEventKind::ResponseCompleted { response_id, .. }
            | SessionEventKind::ResponseIncomplete { response_id, .. } => {
                // A turn is only settled once its response terminates, which is
                // what makes a re-sent turn after a mid-generation crash
                // restartable rather than deduplicated into a partial answer.
                if let Some(turn_id) = self
                    .open_turns
                    .iter()
                    .find(|(_, open)| *open == response_id)
                    .map(|(turn_id, _)| turn_id.clone())
                {
                    self.open_turns.remove(&turn_id);
                    if matches!(event.kind, SessionEventKind::ResponseCompleted { .. }) {
                        self.completed_turns.insert(turn_id, response_id.clone());
                    }
                }
            }
            SessionEventKind::SessionCreated { .. }
            | SessionEventKind::OutputTextDelta { .. }
            | SessionEventKind::TurnDeduplicated { .. }
            | SessionEventKind::Error { .. } => {}
        }
    }

    pub fn completed_response_for(&self, turn_id: &TurnId) -> Option<&ResponseId> {
        self.completed_turns.get(turn_id)
    }
}

/// A leased, projected session.
pub struct Session<S: SessionStore> {
    store: Arc<S>,
    session_id: SessionId,
    lease: Lease,
    state: SessionState,
}

impl<S: SessionStore> Session<S> {
    /// Claim a session and rebuild its state from the log.
    ///
    /// `ledger` carries the cache-model and pricing configuration; replayed
    /// dispatches are recorded into it, so what comes back reflects both static
    /// configuration and session history.
    pub async fn open(
        store: Arc<S>,
        session_id: SessionId,
        node_id: &str,
        lease_ttl_ms: u64,
        ledger: CacheLedger,
    ) -> Result<Self, SessionError> {
        let lease = store
            .acquire_lease(&session_id, node_id, lease_ttl_ms)
            .await?
            .ok_or_else(|| SessionError::NotOwner(session_id.clone()))?;

        let mut state = SessionState {
            ledger,
            ..Default::default()
        };

        // Replay in batches so a long session does not need the whole log
        // resident at once.
        let mut cursor = 0u64;
        loop {
            let batch = store.read_events(&session_id, cursor, REPLAY_BATCH).await?;
            if batch.is_empty() {
                break;
            }
            for event in &batch {
                state.apply(event);
            }
            cursor = batch.last().map_or(cursor, |event| event.seq);
        }

        Ok(Self {
            store,
            session_id,
            lease,
            state,
        })
    }

    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub fn state(&self) -> &SessionState {
        &self.state
    }

    pub fn ledger(&self) -> &CacheLedger {
        &self.state.ledger
    }

    /// Turn index the next admitted turn will receive.
    pub fn turn_index(&self) -> u64 {
        self.state.turn_index
    }

    pub fn last_seq(&self) -> u64 {
        self.state.last_seq
    }

    /// Extend the lease. `false` means ownership was lost and this session
    /// handle must be discarded.
    pub async fn renew(&mut self, ttl_ms: u64) -> Result<bool, SessionError> {
        match self.store.renew_lease(&self.lease, ttl_ms).await? {
            Some(renewed) => {
                self.lease = renewed;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub async fn release(self) -> Result<(), SessionError> {
        Ok(self.store.release_lease(&self.lease).await?)
    }

    /// Append events and fold them into the local projection.
    ///
    /// The projection is updated from what the store actually assigned rather
    /// than from what was submitted, so in-memory state can never claim a
    /// sequence number the log does not have.
    async fn commit(
        &mut self,
        kinds: Vec<SessionEventKind>,
    ) -> Result<Vec<SessionEvent>, SessionError> {
        let events = self.store.append_events(&self.lease, kinds).await?;
        for event in &events {
            self.state.apply(event);
        }
        Ok(events)
    }

    /// Record the session-created event. Safe to call once, at creation.
    pub async fn record_created(&mut self, model_policy: &str) -> Result<(), SessionError> {
        self.commit(vec![SessionEventKind::SessionCreated {
            model_policy: model_policy.to_string(),
        }])
        .await?;
        Ok(())
    }

    /// Admit a turn, appending its input items.
    ///
    /// A `turn_id` that already completed short-circuits: the prior response is
    /// returned and no items are appended, which is what makes a client retry
    /// after a dropped connection safe.
    pub async fn begin_turn(
        &mut self,
        turn_id: TurnId,
        input: Vec<Item>,
    ) -> Result<TurnAdmission, SessionError> {
        if let Some(existing) = self.state.completed_response_for(&turn_id).cloned() {
            self.commit(vec![SessionEventKind::TurnDeduplicated {
                turn_id,
                response_id: existing.clone(),
            }])
            .await?;
            return Ok(TurnAdmission::Deduplicated(existing));
        }

        let response_id = ResponseId::generate();
        let mut kinds = Vec::with_capacity(input.len() + 1);
        kinds.push(SessionEventKind::TurnStarted {
            turn_id,
            response_id: response_id.clone(),
        });
        kinds.extend(
            input
                .into_iter()
                .map(|item| SessionEventKind::ItemAppended { item }),
        );
        self.commit(kinds).await?;
        Ok(TurnAdmission::Started(response_id))
    }

    /// Record the routing choice before any execution begins.
    pub async fn record_routing(
        &mut self,
        response_id: &ResponseId,
        decision: DecisionRecord,
    ) -> Result<(), SessionError> {
        self.commit(vec![SessionEventKind::Routed {
            response_id: response_id.clone(),
            decision,
        }])
        .await?;
        Ok(())
    }

    pub async fn append_output(
        &mut self,
        response_id: &ResponseId,
        text: impl Into<String>,
    ) -> Result<SessionEvent, SessionError> {
        let mut events = self
            .commit(vec![SessionEventKind::OutputTextDelta {
                response_id: response_id.clone(),
                text: text.into(),
            }])
            .await?;
        events
            .pop()
            .ok_or_else(|| SessionError::ResponseNotOpen(response_id.clone()))
    }

    /// Close a response successfully, committing the assistant item.
    pub async fn complete(
        &mut self,
        response_id: &ResponseId,
        text: impl Into<String>,
        usage: Usage,
    ) -> Result<(), SessionError> {
        self.commit(vec![
            SessionEventKind::ItemAppended {
                item: Item::assistant_text(text, response_id.clone()),
            },
            SessionEventKind::ResponseCompleted {
                response_id: response_id.clone(),
                usage,
            },
        ])
        .await?;
        Ok(())
    }

    /// Close a response without a complete answer.
    ///
    /// The partial text is committed as an assistant item so the successor can
    /// resume from it. That partial is also, conveniently, a guaranteed cache
    /// hit on the target that produced it.
    pub async fn mark_incomplete(
        &mut self,
        response_id: &ResponseId,
        partial: impl Into<String>,
        reason: IncompleteReason,
        usage: Usage,
    ) -> Result<(), SessionError> {
        let partial = partial.into();
        let mut kinds = Vec::with_capacity(2);
        if !partial.is_empty() {
            kinds.push(SessionEventKind::ItemAppended {
                item: Item::assistant_text(partial, response_id.clone()),
            });
        }
        kinds.push(SessionEventKind::ResponseIncomplete {
            response_id: response_id.clone(),
            reason,
            usage,
        });
        self.commit(kinds).await?;
        Ok(())
    }

    /// Events after `after_seq`. Backs both SSE resumption and reconnect replay.
    pub async fn events_since(
        &self,
        after_seq: u64,
        limit: usize,
    ) -> Result<Vec<SessionEvent>, SessionError> {
        Ok(self
            .store
            .read_events(&self.session_id, after_seq, limit)
            .await?)
    }

    /// Events belonging to one response, for the Responses API view.
    pub async fn response_events(
        &self,
        response_id: &ResponseId,
        after_seq: u64,
        limit: usize,
    ) -> Result<Vec<SessionEvent>, SessionError> {
        Ok(self
            .store
            .read_events(&self.session_id, after_seq, limit)
            .await?
            .into_iter()
            .filter(|event| event.response_id() == Some(response_id))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::{Candidate, Target};
    use crate::store::MemoryStore;

    const TTL: u64 = 30_000;

    async fn new_session(store: Arc<MemoryStore>, node: &str) -> (SessionId, Session<MemoryStore>) {
        let sid = SessionId::generate();
        store.create_session(&sid, "affinity").await.unwrap();
        let session = Session::open(store, sid.clone(), node, TTL, CacheLedger::new())
            .await
            .unwrap();
        (sid, session)
    }

    fn decision_for(target: Target, isl: u64) -> DecisionRecord {
        DecisionRecord {
            chosen: target,
            rationale: "test".into(),
            policy: "affinity".into(),
            isl_tokens: isl,
            expected_prefill_tokens: isl as f64,
            expected_cost_usd: 0.0,
            considered: Vec::<Candidate>::new(),
        }
    }

    #[tokio::test]
    async fn a_turn_appends_items_and_advances_the_turn_index() {
        let store = Arc::new(MemoryStore::new());
        let (_, mut session) = new_session(store, "node-a").await;
        assert_eq!(session.turn_index(), 0);

        let admission = session
            .begin_turn(TurnId::new("t1"), vec![Item::user_text("hello")])
            .await
            .unwrap();
        assert!(matches!(admission, TurnAdmission::Started(_)));
        assert_eq!(session.turn_index(), 1);
        assert_eq!(session.state().items.len(), 1);
    }

    #[tokio::test]
    async fn replaying_a_completed_turn_id_does_not_generate_twice() {
        let store = Arc::new(MemoryStore::new());
        let (_, mut session) = new_session(store, "node-a").await;

        let first = session
            .begin_turn(TurnId::new("t1"), vec![Item::user_text("hello")])
            .await
            .unwrap();
        let response_id = first.response_id().clone();
        session
            .complete(&response_id, "hi there", Usage::default())
            .await
            .unwrap();

        let retry = session
            .begin_turn(TurnId::new("t1"), vec![Item::user_text("hello")])
            .await
            .unwrap();
        assert_eq!(retry, TurnAdmission::Deduplicated(response_id));
        // The retry must not have appended the user item a second time.
        assert_eq!(
            session
                .state()
                .items
                .iter()
                .filter(|item| item.render().contains("hello"))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn an_interrupted_turn_is_retryable_rather_than_deduplicated() {
        let store = Arc::new(MemoryStore::new());
        let (_, mut session) = new_session(store, "node-a").await;

        let admission = session
            .begin_turn(TurnId::new("t1"), vec![Item::user_text("hello")])
            .await
            .unwrap();
        let response_id = admission.response_id().clone();
        session
            .mark_incomplete(
                &response_id,
                "partial",
                IncompleteReason::OwnerLost,
                Usage::default(),
            )
            .await
            .unwrap();

        // The turn never completed, so re-sending it must start fresh.
        let retry = session
            .begin_turn(TurnId::new("t1"), vec![Item::user_text("hello")])
            .await
            .unwrap();
        assert!(matches!(retry, TurnAdmission::Started(_)));
        assert_ne!(retry.response_id(), &response_id);
    }

    #[tokio::test]
    async fn a_successor_node_reconstructs_identical_state_from_the_log() {
        let store = Arc::new(MemoryStore::new());
        let (sid, mut session) = new_session(store.clone(), "node-a").await;

        let admission = session
            .begin_turn(TurnId::new("t1"), vec![Item::user_text("first question")])
            .await
            .unwrap();
        let response_id = admission.response_id().clone();
        let target = Target::Local {
            worker_id: 7,
            dp_rank: 0,
            model: "llama".into(),
        };
        session
            .record_routing(&response_id, decision_for(target.clone(), 4_096))
            .await
            .unwrap();
        session
            .append_output(&response_id, "part one ")
            .await
            .unwrap();
        session
            .complete(&response_id, "part one and two", Usage::default())
            .await
            .unwrap();

        // Owner dies; a second node takes over.
        let seq_before = session.last_seq();
        let items_before = session.state().items.clone();
        drop(session);
        store.expire_lease_now(&sid).await;

        let successor = Session::open(store, sid, "node-b", TTL, CacheLedger::new())
            .await
            .unwrap();
        assert_eq!(successor.last_seq(), seq_before);
        assert_eq!(successor.state().items, items_before);
        assert_eq!(successor.turn_index(), 1);
        // The routing ledger is part of the projection, so the successor knows
        // worker 7 is warm without being told.
        assert!(successor.ledger().state_for(&target).is_some());
    }

    #[tokio::test]
    async fn a_displaced_owner_cannot_keep_writing() {
        let store = Arc::new(MemoryStore::new());
        let (sid, mut displaced) = new_session(store.clone(), "node-a").await;

        store.expire_lease_now(&sid).await;
        let _successor = Session::open(store, sid, "node-b", TTL, CacheLedger::new())
            .await
            .unwrap();

        let result = displaced
            .begin_turn(TurnId::new("t1"), vec![Item::user_text("hello")])
            .await;
        assert!(matches!(
            result,
            Err(SessionError::Store(StoreError::LeaseLost { .. }))
        ));
    }

    #[tokio::test]
    async fn resumption_from_a_sequence_number_is_gapless() {
        let store = Arc::new(MemoryStore::new());
        let (_, mut session) = new_session(store, "node-a").await;

        let admission = session
            .begin_turn(TurnId::new("t1"), vec![Item::user_text("q")])
            .await
            .unwrap();
        let response_id = admission.response_id().clone();
        let checkpoint = session.last_seq();

        for chunk in ["a", "b", "c"] {
            session.append_output(&response_id, chunk).await.unwrap();
        }

        let replayed = session.events_since(checkpoint, 100).await.unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(
            replayed.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![checkpoint + 1, checkpoint + 2, checkpoint + 3]
        );

        // Projecting to one response drops the session-level events.
        let scoped = session.response_events(&response_id, 0, 100).await.unwrap();
        assert!(scoped.iter().all(|e| e.response_id() == Some(&response_id)));
    }
}
