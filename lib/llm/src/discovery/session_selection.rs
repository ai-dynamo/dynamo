// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session-affinity preference applied before WorkerSet selection.
//!
//! `Model::select_worker_set_with` picks a WorkerSet by worker-count weight and only then
//! lets that set's router look the session up. During a rolling update that moves an
//! established session onto the new WorkerSet. This module keeps a known, unexpired `Hard`
//! binding on its WorkerSet while the bound worker and the set remain eligible, and it
//! coordinates the first requests of a session on one frontend so they cannot create
//! independent bindings in different WorkerSets.
//!
//! REMOVAL CONDITION: this is a temporary check in front of the existing selector, not a
//! second routing-policy framework. When normal worker routing can honor a session binding
//! before committing to a WorkerSet, delete this module, the `SessionBindingLookup`
//! plumbing in `WorkerSet` and `PreprocessedRouting`, the `*_for_session` accessors on
//! `ModelManager`, and the guard handling in the HTTP handlers. Removal needs no data
//! migration and no worker, client, or policy API change: every lasting binding lives in
//! the routers, and this module owns only transient initialization coordination.

use std::{collections::HashMap, sync::Arc};

use dynamo_runtime::{
    engine::AsyncEngineContext,
    error::{DynamoError, ErrorType},
    pipeline::{AsyncEngineContextProvider, Context, Error},
};
use tokio::sync::OwnedMutexGuard;

use crate::{
    protocols::common::extensions::{
        AGENT_CONTEXT_CONTEXT_KEY, AgentContext, SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId,
    },
    session_affinity::{invalid_argument, subagent_group_binding_id},
};

/// The session facts WorkerSet selection needs: the effective binding key, whether the
/// request may initialize a binding, and the request context that cancels a wait.
pub(crate) struct SessionSelection {
    key: Arc<SessionAffinityId>,
    query_only: bool,
    context: Arc<dyn AsyncEngineContext>,
}

impl SessionSelection {
    /// Read the session identity a handler attached to `request`.
    ///
    /// Returns `None` when the request carries no session identity, or when it pins a
    /// serving worker or DP rank explicitly (`has_explicit_target`). Explicit targets keep
    /// the existing selection and validation path: an affinity preference must never
    /// replace or weaken them, and the grouping rule already excludes them.
    pub(crate) fn from_context<T: Send + Sync + 'static>(
        request: &Context<T>,
        has_explicit_target: bool,
        query_only: bool,
    ) -> Result<Option<Self>, Error> {
        if has_explicit_target {
            return Ok(None);
        }
        let Some(session_id) = request
            .get_optional::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .map_err(|message| {
                invalid_argument(format!("invalid session affinity context: {message}"))
            })?
        else {
            return Ok(None);
        };
        let parent_session_id = request
            .get_optional::<AgentContext>(AGENT_CONTEXT_CONTEXT_KEY)
            .map_err(|message| invalid_argument(format!("invalid agent context: {message}")))?;
        let key = subagent_group_binding_id(
            parent_session_id
                .as_ref()
                .and_then(|context| context.parent_session_id.as_deref()),
            None,
        )
        .unwrap_or(session_id);
        Ok(Some(Self {
            key,
            query_only,
            context: request.context(),
        }))
    }

    /// The key the selected router will bind under.
    pub(crate) fn key(&self) -> &SessionAffinityId {
        &self.key
    }

    /// Query-only requests may read bindings but never initialize or refresh them.
    pub(crate) fn query_only(&self) -> bool {
        self.query_only
    }

    pub(crate) fn context(&self) -> &dyn AsyncEngineContext {
        self.context.as_ref()
    }
}

/// Coordinates the first requests of a session within one model scope.
///
/// A read-only binding lookup sees nothing while a router is still initializing a session,
/// so two first requests on one frontend could pick different WorkerSets and bind twice. A
/// miss takes the per-key guard before WorkerSet selection and holds it until the selected
/// pipeline returns its response stream; the router commits its binding before that, so the
/// next request's lookup hits. Entries exist only while a guard is held or awaited, and
/// different sessions never wait on each other.
///
/// Shared by every `Model::snapshot()` copy and alias of a model, so a discovery update
/// cannot create an independent guard for a session that is already initializing.
#[derive(Default)]
pub(crate) struct SessionInitCoordination {
    /// Held only for map reads and writes, never across an `.await`.
    entries: parking_lot::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl SessionInitCoordination {
    /// Take the guard for `key`, abandoning the wait when the request is stopped or killed.
    pub(crate) async fn acquire(
        self: &Arc<Self>,
        key: &SessionAffinityId,
        context: &dyn AsyncEngineContext,
    ) -> Result<SessionInitGuard, Error> {
        let entry = self
            .entries
            .lock()
            .entry(key.as_str().to_string())
            .or_default()
            .clone();
        let acquired = tokio::select! {
            biased;
            _ = context.stopped() => None,
            _ = context.killed() => None,
            guard = entry.lock_owned() => Some(guard),
        };
        match acquired {
            Some(guard) => Ok(SessionInitGuard {
                coordination: self.clone(),
                key: key.as_str().to_string(),
                guard: Some(guard),
            }),
            None => {
                // The abandoned lock future released its reference above; drop the entry if
                // nobody else holds or awaits it.
                self.release(key.as_str());
                Err(cancelled(context.id()))
            }
        }
    }

    /// Remove `key` once nothing holds or awaits its guard. A waiter keeps its own
    /// reference to the entry, so it can never be removed from under one.
    fn release(&self, key: &str) {
        let mut entries = self.entries.lock();
        if entries
            .get(key)
            .is_some_and(|entry| Arc::strong_count(entry) == 1)
        {
            entries.remove(key);
        }
    }

    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.entries.lock().len()
    }
}

/// Holds a session's initialization slot from WorkerSet selection until the selected
/// pipeline returns its response stream (or fails). Drop it at that boundary; do not retain
/// it for the stream's lifetime.
pub(crate) struct SessionInitGuard {
    coordination: Arc<SessionInitCoordination>,
    key: String,
    guard: Option<OwnedMutexGuard<()>>,
}

impl Drop for SessionInitGuard {
    fn drop(&mut self) {
        drop(self.guard.take());
        self.coordination.release(&self.key);
    }
}

fn cancelled(context_id: &str) -> Error {
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(format!(
            "request {context_id} was cancelled while waiting for session worker set selection"
        ))
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_runtime::pipeline::context::Controller;

    use super::*;
    use crate::protocols::common::extensions::AgentContextBuilder;

    fn session(key: &str) -> SessionAffinityId {
        SessionAffinityId::new(key)
    }

    #[test]
    fn from_context_uses_parent_group_key_and_bypasses_explicit_targets() {
        let mut request = Context::new(());
        request.insert(SESSION_AFFINITY_CONTEXT_KEY, session("child-1"));
        request.insert(
            AGENT_CONTEXT_CONTEXT_KEY,
            AgentContextBuilder::default()
                .session_id("child-1".to_string())
                .parent_session_id("parent-1".to_string())
                .build()
                .unwrap(),
        );
        let selection = SessionSelection::from_context(&request, false, true)
            .unwrap()
            .expect("session identity present");
        assert_eq!(
            selection.key().as_str(),
            crate::session_affinity::subagent_group_affinity_id("parent-1")
        );
        assert!(selection.query_only());

        assert!(
            SessionSelection::from_context(&request, true, false)
                .unwrap()
                .is_none(),
            "an explicit serving target keeps the existing selection path"
        );

        let mut own_session = Context::new(());
        own_session.insert(SESSION_AFFINITY_CONTEXT_KEY, session("parent-1"));
        let selection = SessionSelection::from_context(&own_session, false, false)
            .unwrap()
            .unwrap();
        assert_eq!(selection.key().as_str(), "parent-1");
        assert!(!selection.query_only());

        assert!(
            SessionSelection::from_context(&Context::new(()), false, false)
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn guard_serializes_one_key_and_cleans_up_when_released() {
        let coordination = Arc::new(SessionInitCoordination::default());
        let controller = Controller::default();

        let first = coordination
            .acquire(&session("s1"), &controller)
            .await
            .unwrap();
        // Another session proceeds independently while `s1` is held.
        let other = coordination
            .acquire(&session("s2"), &controller)
            .await
            .unwrap();
        assert_eq!(coordination.entry_count(), 2);
        drop(other);
        assert_eq!(coordination.entry_count(), 1);

        let waiter = {
            let coordination = coordination.clone();
            tokio::spawn(async move {
                let controller = Controller::default();
                coordination.acquire(&session("s1"), &controller).await
            })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !waiter.is_finished(),
            "second request must wait on the first"
        );
        // Releasing while a waiter exists must not remove the entry out from under it.
        drop(first);
        let second = waiter.await.unwrap().unwrap();
        assert_eq!(coordination.entry_count(), 1);
        drop(second);
        assert_eq!(coordination.entry_count(), 0);
    }

    #[tokio::test]
    async fn cancelled_wait_fails_without_leaking_or_blocking_the_holder() {
        let coordination = Arc::new(SessionInitCoordination::default());
        let holder_context = Controller::default();
        let held = coordination
            .acquire(&session("s1"), &holder_context)
            .await
            .unwrap();

        let cancelled_context = Controller::default();
        let waiter = {
            let coordination = coordination.clone();
            let context = Arc::new(cancelled_context);
            let handle = context.clone();
            (
                tokio::spawn(
                    async move { coordination.acquire(&session("s1"), handle.as_ref()).await },
                ),
                context,
            )
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        waiter.1.stop();
        let error = match waiter.0.await.unwrap() {
            Ok(_) => panic!("stopped request must abandon the wait"),
            Err(error) => error,
        };
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[ErrorType::Cancelled],
            &[]
        ));
        // The holder's entry survives the cancelled waiter and disappears with the holder.
        assert_eq!(coordination.entry_count(), 1);
        drop(held);
        assert_eq!(coordination.entry_count(), 0);

        // A request stopped before it waits never takes the slot.
        let stopped = Controller::default();
        stopped.stop();
        assert!(
            coordination
                .acquire(&session("s3"), &stopped)
                .await
                .is_err()
        );
        assert_eq!(coordination.entry_count(), 0);
    }
}
