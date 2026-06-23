// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use dashmap::{DashMap, mapref::entry::Entry};
use dynamo_runtime::{
    component::Component,
    engine::{AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream},
    error::{DynamoError, ErrorType},
    pipeline::{Error, ManyOut, ResponseStream},
};
use futures::Stream;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::{
    BoundState, ClosingState, LlmResponse, OpeningState, SESSION_TIMEOUT_FALLBACK_BUFFER,
    SessionBinding, SessionKind, SessionState, SessionTarget,
    lifecycle::{EventSessionLifecycle, SessionLifecycleBackend},
};
use crate::protocols::common::extensions::{SessionAction, SessionControl};

pub(super) const REAPER_INTERVAL: Duration = Duration::from_secs(30);

pub(super) struct SessionCoordinatorInner {
    pub(super) sessions: DashMap<String, SessionState>,
    lifecycle: Arc<dyn SessionLifecycleBackend>,
    next_revision: AtomicU64,
    cancel: CancellationToken,
    #[cfg(test)]
    test_hooks: Option<SessionTestHooks>,
}

#[cfg(test)]
#[derive(Clone)]
pub(super) struct SessionTestHooks {
    pub(super) retry_armed: Arc<tokio::sync::Semaphore>,
    pub(super) reaper_armed: Arc<tokio::sync::Semaphore>,
    pub(super) coalesce_waiting: Arc<tokio::sync::Semaphore>,
    pub(super) retry_stopped: Arc<tokio::sync::Semaphore>,
    pub(super) reaper_stopped: Arc<tokio::sync::Semaphore>,
}

#[cfg(test)]
struct TestTaskStop(Arc<tokio::sync::Semaphore>);

#[cfg(test)]
impl Drop for TestTaskStop {
    fn drop(&mut self) {
        self.0.add_permits(1);
    }
}

impl Drop for SessionCoordinatorInner {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[derive(Clone)]
pub struct SessionCoordinator {
    pub(super) inner: Arc<SessionCoordinatorInner>,
}

impl SessionCoordinator {
    pub fn new(component: Component) -> Self {
        Self::with_lifecycle(Arc::new(EventSessionLifecycle::new(component)))
    }

    pub fn with_lifecycle(lifecycle: Arc<dyn SessionLifecycleBackend>) -> Self {
        let inner = Arc::new(SessionCoordinatorInner {
            sessions: DashMap::new(),
            lifecycle,
            next_revision: AtomicU64::new(1),
            cancel: CancellationToken::new(),
            #[cfg(test)]
            test_hooks: None,
        });
        Self::spawn_reaper(&inner);
        Self { inner }
    }

    #[cfg(test)]
    pub(super) fn with_lifecycle_and_hooks(
        lifecycle: Arc<dyn SessionLifecycleBackend>,
        test_hooks: SessionTestHooks,
    ) -> Self {
        let inner = Arc::new(SessionCoordinatorInner {
            sessions: DashMap::new(),
            lifecycle,
            next_revision: AtomicU64::new(1),
            cancel: CancellationToken::new(),
            test_hooks: Some(test_hooks),
        });
        Self::spawn_reaper(&inner);
        Self { inner }
    }

    fn spawn_reaper(inner: &Arc<SessionCoordinatorInner>) {
        let weak = Arc::downgrade(inner);
        let cancel = inner.cancel.clone();
        tokio::spawn(async move {
            #[cfg(test)]
            let _stopped = weak
                .upgrade()
                .and_then(|inner| inner.test_hooks.clone())
                .map(|hooks| TestTaskStop(hooks.reaper_stopped));
            let mut interval =
                tokio::time::interval_at(Instant::now() + REAPER_INTERVAL, REAPER_INTERVAL);
            #[cfg(test)]
            if let Some(hooks) = weak.upgrade().and_then(|inner| inner.test_hooks.clone()) {
                hooks.reaper_armed.add_permits(1);
            }
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = interval.tick() => {
                        let Some(inner) = weak.upgrade() else {
                            return;
                        };
                        SessionCoordinator::reap_expired(&inner);
                    }
                }
            }
        });
    }

    fn reap_expired(inner: &Arc<SessionCoordinatorInner>) {
        let now = Instant::now();
        let mut closing = Vec::new();
        inner.sessions.retain(|session_id, state| match state {
            SessionState::Bound(bound)
                if bound.active_leases == 0 && bound.binding.expires_at <= now =>
            {
                if bound.binding.kind == SessionKind::RouterOnly {
                    return false;
                }
                let binding = bound.binding.clone();
                let revision = bound.revision;
                *state = SessionState::Closing(ClosingState {
                    revision,
                    remove_at: now + binding.timeout + SESSION_TIMEOUT_FALLBACK_BUFFER,
                    binding,
                    retry_started: false,
                });
                closing.push(session_id.clone());
                true
            }
            _ => true,
        });
        for session_id in closing {
            Self::ensure_close_task(inner, session_id);
        }
    }

    pub async fn begin(
        &self,
        control: &SessionControl,
        explicit_target: Option<SessionTarget>,
        direct: bool,
    ) -> Result<SessionOperation, Error> {
        validate_control(control, explicit_target, direct)?;
        let requested_kind = match control.action {
            Some(SessionAction::Open) => Some(SessionKind::EngineBacked),
            Some(SessionAction::Bind) => Some(SessionKind::RouterOnly),
            _ => None,
        };
        let timeout = Duration::from_secs(control.timeout);

        loop {
            let mut wait = None;
            let now = Instant::now();
            let operation = match self.inner.sessions.entry(control.session_id.clone()) {
                Entry::Vacant(entry) => {
                    let Some(kind) = requested_kind else {
                        return Err(invalid_argument(format!(
                            "session {} must be opened or bound before continuation or close",
                            control.session_id
                        )));
                    };
                    let revision = self.inner.next_revision.fetch_add(1, Ordering::Relaxed);
                    entry.insert(SessionState::Opening(OpeningState {
                        revision,
                        kind,
                        requested_target: explicit_target,
                        target: None,
                        timeout,
                        notify: Arc::new(tokio::sync::Notify::new()),
                    }));
                    Some(SessionOperation::opening(
                        Arc::downgrade(&self.inner),
                        control.session_id.clone(),
                        revision,
                        kind,
                    ))
                }
                Entry::Occupied(mut entry) => {
                    let expired = match entry.get() {
                        SessionState::Bound(bound)
                            if bound.active_leases == 0 && bound.binding.expires_at <= now =>
                        {
                            Some((bound.revision, bound.binding.clone()))
                        }
                        _ => None,
                    };
                    if let Some((revision, binding)) = expired {
                        if binding.kind == SessionKind::EngineBacked {
                            entry.insert(SessionState::Closing(ClosingState {
                                revision,
                                remove_at: now + binding.timeout + SESSION_TIMEOUT_FALLBACK_BUFFER,
                                binding,
                                retry_started: false,
                            }));
                            drop(entry);
                            Self::ensure_close_task(&self.inner, control.session_id.clone());
                            return Err(invalid_argument(format!(
                                "session {} expired and is closing",
                                control.session_id
                            )));
                        }

                        let Some(kind) = requested_kind else {
                            entry.remove();
                            return Err(invalid_argument(format!(
                                "session {} must be opened or bound before continuation or close",
                                control.session_id
                            )));
                        };
                        let revision = self.inner.next_revision.fetch_add(1, Ordering::Relaxed);
                        entry.insert(SessionState::Opening(OpeningState {
                            revision,
                            kind,
                            requested_target: explicit_target,
                            target: None,
                            timeout,
                            notify: Arc::new(tokio::sync::Notify::new()),
                        }));
                        return Ok(SessionOperation::opening(
                            Arc::downgrade(&self.inner),
                            control.session_id.clone(),
                            revision,
                            kind,
                        ));
                    }

                    let state = entry.get_mut();
                    match state {
                        SessionState::Opening(opening) => {
                            let Some(kind) = requested_kind else {
                                return Err(invalid_argument(format!(
                                    "session {} is opening; only a matching open or bind may coalesce",
                                    control.session_id
                                )));
                            };
                            ensure_opening_matches(opening, kind, explicit_target, timeout)?;
                            tracing::warn!(session_id = %control.session_id, "Coalescing duplicate session setup");
                            #[cfg(test)]
                            if let Some(hooks) = self.inner.test_hooks.as_ref() {
                                hooks.coalesce_waiting.add_permits(1);
                            }
                            wait = Some(opening.notify.clone().notified_owned());
                            None
                        }
                        SessionState::Bound(bound) => {
                            if let Some(kind) = requested_kind {
                                ensure_binding_matches(
                                    &bound.binding,
                                    kind,
                                    explicit_target,
                                    timeout,
                                )?;
                                tracing::warn!(session_id = %control.session_id, "Ignoring duplicate session setup");
                            } else if let Some(target) = explicit_target
                                && !target_matches(bound.binding.target, target)
                            {
                                return Err(target_mismatch(
                                    &control.session_id,
                                    bound.binding.target,
                                    target,
                                ));
                            }

                            if matches!(control.action, Some(SessionAction::Close)) {
                                if bound.active_leases != 0 {
                                    return Err(invalid_argument(format!(
                                        "session {} cannot close while {} request lease(s) are active",
                                        control.session_id, bound.active_leases
                                    )));
                                }
                                let prior = bound.clone();
                                let revision = bound.revision;
                                let binding = bound.binding.clone();
                                *state = SessionState::Closing(ClosingState {
                                    revision,
                                    remove_at: now
                                        + binding.timeout
                                        + SESSION_TIMEOUT_FALLBACK_BUFFER,
                                    binding: binding.clone(),
                                    retry_started: false,
                                });
                                Some(SessionOperation::closing(
                                    Arc::downgrade(&self.inner),
                                    control.session_id.clone(),
                                    revision,
                                    binding.target,
                                    prior,
                                ))
                            } else {
                                bound.active_leases += 1;
                                Some(SessionOperation::lease(
                                    Arc::downgrade(&self.inner),
                                    control.session_id.clone(),
                                    bound.revision,
                                    bound.binding.target,
                                ))
                            }
                        }
                        SessionState::Closing(_) => {
                            return Err(invalid_argument(format!(
                                "session {} is closing and is not routable",
                                control.session_id
                            )));
                        }
                    }
                }
            };
            if let Some(operation) = operation {
                return Ok(operation);
            }
            if let Some(wait) = wait {
                wait.await;
            }
        }
    }

    pub fn query_target(
        &self,
        control: &SessionControl,
        explicit_target: Option<SessionTarget>,
    ) -> Result<Option<SessionTarget>, Error> {
        validate_control(control, explicit_target, false)?;
        let requested_kind = match control.action {
            Some(SessionAction::Open) => Some(SessionKind::EngineBacked),
            Some(SessionAction::Bind) => Some(SessionKind::RouterOnly),
            _ => None,
        };
        let timeout = Duration::from_secs(control.timeout);
        match self.inner.sessions.get(&control.session_id).as_deref() {
            None if requested_kind.is_some() => Ok(None),
            None => Err(invalid_argument(format!(
                "session {} must be opened or bound before continuation or close",
                control.session_id
            ))),
            Some(SessionState::Opening(opening)) => {
                let Some(kind) = requested_kind else {
                    return Err(invalid_argument(format!(
                        "session {} is opening and is not yet routable",
                        control.session_id
                    )));
                };
                ensure_opening_matches(opening, kind, explicit_target, timeout)?;
                Err(invalid_argument(format!(
                    "session {} is opening and is not yet routable",
                    control.session_id
                )))
            }
            Some(SessionState::Bound(bound)) => {
                if bound.binding.expires_at <= Instant::now() && bound.active_leases == 0 {
                    return Err(invalid_argument(format!(
                        "session {} has expired",
                        control.session_id
                    )));
                }
                if let Some(kind) = requested_kind {
                    ensure_binding_matches(&bound.binding, kind, explicit_target, timeout)?;
                } else if let Some(target) = explicit_target
                    && !target_matches(bound.binding.target, target)
                {
                    return Err(target_mismatch(
                        &control.session_id,
                        bound.binding.target,
                        target,
                    ));
                }
                if matches!(control.action, Some(SessionAction::Close)) && bound.active_leases != 0
                {
                    return Err(invalid_argument(format!(
                        "session {} cannot close while {} request lease(s) are active",
                        control.session_id, bound.active_leases
                    )));
                }
                Ok(Some(bound.binding.target))
            }
            Some(SessionState::Closing(_)) => Err(invalid_argument(format!(
                "session {} is closing and is not routable",
                control.session_id
            ))),
        }
    }

    fn ensure_close_task(inner: &Arc<SessionCoordinatorInner>, session_id: String) {
        {
            let Some(mut state) = inner.sessions.get_mut(&session_id) else {
                return;
            };
            let SessionState::Closing(closing) = state.value_mut() else {
                return;
            };
            if closing.retry_started {
                return;
            }
            closing.retry_started = true;
        }

        let weak = Arc::downgrade(inner);
        let cancel = inner.cancel.clone();
        tokio::spawn(async move {
            #[cfg(test)]
            let _stopped = weak
                .upgrade()
                .and_then(|inner| inner.test_hooks.clone())
                .map(|hooks| TestTaskStop(hooks.retry_stopped));
            let mut backoff = Duration::from_millis(100);
            loop {
                let Some(inner) = weak.upgrade() else {
                    return;
                };
                let Some((revision, binding, remove_at)) = ({
                    match inner.sessions.get(&session_id).as_deref() {
                        Some(SessionState::Closing(closing)) => {
                            Some((closing.revision, closing.binding.clone(), closing.remove_at))
                        }
                        _ => None,
                    }
                }) else {
                    return;
                };

                if Instant::now() >= remove_at {
                    remove_closing(&inner, &session_id, revision);
                    return;
                }

                let lifecycle = inner.lifecycle.clone();
                drop(inner);
                let close_result = tokio::select! {
                    _ = cancel.cancelled() => return,
                    result = lifecycle.close(
                        &session_id,
                        binding.target,
                        "session-lifecycle-retry",
                    ) => result,
                };
                if close_result.is_ok() {
                    if let Some(inner) = weak.upgrade() {
                        remove_closing(&inner, &session_id, revision);
                    }
                    return;
                }

                #[cfg(test)]
                if let Some(hooks) = weak.upgrade().and_then(|inner| inner.test_hooks.clone()) {
                    hooks.retry_armed.add_permits(1);
                }

                let remaining = remove_at.saturating_duration_since(Instant::now());
                let sleep_for = backoff.min(remaining);
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(sleep_for) => {}
                }
                backoff = (backoff * 2).min(Duration::from_secs(5));
            }
        });
    }

    fn complete_stream_close(
        inner: &Arc<SessionCoordinatorInner>,
        session_id: String,
        revision: u64,
    ) {
        let kind = {
            let Some(state) = inner.sessions.get(&session_id) else {
                return;
            };
            let SessionState::Closing(closing) = state.value() else {
                return;
            };
            if closing.revision != revision {
                return;
            }
            closing.binding.kind
        };
        if kind == SessionKind::RouterOnly {
            remove_closing(inner, &session_id, revision);
            return;
        }
        Self::ensure_close_task(inner, session_id);
    }
}

fn remove_closing(inner: &SessionCoordinatorInner, session_id: &str, revision: u64) {
    inner.sessions.remove_if(
        session_id,
        |_, state| matches!(state, SessionState::Closing(closing) if closing.revision == revision),
    );
}

fn ensure_opening_matches(
    opening: &OpeningState,
    kind: SessionKind,
    target: Option<SessionTarget>,
    timeout: Duration,
) -> Result<(), Error> {
    if opening.kind != kind || opening.timeout != timeout {
        return Err(invalid_argument("conflicting duplicate session setup"));
    }
    let expected = opening.target.or(opening.requested_target);
    if let (Some(expected), Some(actual)) = (expected, target)
        && !target_matches(expected, actual)
    {
        return Err(invalid_argument(format!(
            "conflicting duplicate session target: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn ensure_binding_matches(
    binding: &SessionBinding,
    kind: SessionKind,
    target: Option<SessionTarget>,
    timeout: Duration,
) -> Result<(), Error> {
    if binding.kind != kind || binding.timeout != timeout {
        return Err(invalid_argument("conflicting duplicate session setup"));
    }
    if let Some(target) = target
        && !target_matches(binding.target, target)
    {
        return Err(target_mismatch("duplicate setup", binding.target, target));
    }
    Ok(())
}

fn validate_control(
    control: &SessionControl,
    explicit_target: Option<SessionTarget>,
    direct: bool,
) -> Result<(), Error> {
    if control.timeout == 0 {
        return Err(invalid_argument(
            "session timeout must be greater than zero",
        ));
    }
    if direct && explicit_target.is_none() {
        return Err(invalid_argument(
            "worker ID is required for every session request in Direct routing mode",
        ));
    }
    Ok(())
}

fn target_mismatch(session_id: &str, expected: SessionTarget, actual: SessionTarget) -> Error {
    invalid_argument(format!(
        "session {session_id} target conflict: expected {expected:?}, got {actual:?}"
    ))
}

fn target_matches(expected: SessionTarget, requested: SessionTarget) -> bool {
    expected.worker_id == requested.worker_id
        && requested
            .dp_rank
            .is_none_or(|rank| expected.dp_rank == Some(rank))
}

pub(super) fn invalid_argument(message: impl Into<String>) -> Error {
    let message = message.into();
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message)
        .build()
        .into()
}

enum OperationKind {
    Opening {
        kind: SessionKind,
        target: Option<SessionTarget>,
        backend_opened: bool,
    },
    Lease {
        target: SessionTarget,
    },
    Closing {
        target: SessionTarget,
        prior: BoundState,
    },
}

pub struct SessionOperation {
    coordinator: Weak<SessionCoordinatorInner>,
    session_id: String,
    revision: u64,
    kind: Option<OperationKind>,
}

impl SessionOperation {
    fn opening(
        coordinator: Weak<SessionCoordinatorInner>,
        session_id: String,
        revision: u64,
        kind: SessionKind,
    ) -> Self {
        Self {
            coordinator,
            session_id,
            revision,
            kind: Some(OperationKind::Opening {
                kind,
                target: None,
                backend_opened: false,
            }),
        }
    }

    fn lease(
        coordinator: Weak<SessionCoordinatorInner>,
        session_id: String,
        revision: u64,
        target: SessionTarget,
    ) -> Self {
        Self {
            coordinator,
            session_id,
            revision,
            kind: Some(OperationKind::Lease { target }),
        }
    }

    fn closing(
        coordinator: Weak<SessionCoordinatorInner>,
        session_id: String,
        revision: u64,
        target: SessionTarget,
        prior: BoundState,
    ) -> Self {
        Self {
            coordinator,
            session_id,
            revision,
            kind: Some(OperationKind::Closing { target, prior }),
        }
    }

    pub fn target(&self) -> Option<SessionTarget> {
        match self.kind.as_ref()? {
            OperationKind::Opening { target, .. } => *target,
            OperationKind::Lease { target } | OperationKind::Closing { target, .. } => {
                Some(*target)
            }
        }
    }

    pub async fn selected(&mut self, target: SessionTarget, context_id: &str) -> Result<(), Error> {
        let Some(inner) = self.coordinator.upgrade() else {
            return Err(anyhow::anyhow!("session coordinator dropped"));
        };
        let Some(kind) = self.kind.as_mut() else {
            return Err(anyhow::anyhow!("session operation already completed"));
        };
        match kind {
            OperationKind::Opening {
                kind,
                target: selected,
                backend_opened,
            } => {
                let timeout = {
                    let Some(mut state) = inner.sessions.get_mut(&self.session_id) else {
                        return Err(invalid_argument(format!(
                            "session {} is no longer opening",
                            self.session_id
                        )));
                    };
                    let SessionState::Opening(opening) = state.value_mut() else {
                        return Err(invalid_argument(format!(
                            "session {} is no longer opening",
                            self.session_id
                        )));
                    };
                    if opening.revision != self.revision {
                        return Err(invalid_argument(format!(
                            "session {} opening reservation changed",
                            self.session_id
                        )));
                    }
                    if let Some(requested) = opening.requested_target
                        && !target_matches(target, requested)
                    {
                        return Err(target_mismatch(&self.session_id, requested, target));
                    }
                    opening.target = Some(target);
                    *selected = Some(target);
                    opening.timeout
                };

                if *kind == SessionKind::RouterOnly {
                    return Ok(());
                }

                match inner
                    .lifecycle
                    .open(&self.session_id, timeout, target, context_id)
                    .await
                {
                    Ok(()) => {
                        *backend_opened = true;
                        Ok(())
                    }
                    Err(error) => {
                        abort_opening(&inner, &self.session_id, self.revision);
                        self.kind = None;
                        Err(anyhow::anyhow!(error))
                    }
                }
            }
            OperationKind::Lease { target: expected }
            | OperationKind::Closing {
                target: expected, ..
            } => {
                if *expected != target {
                    return Err(target_mismatch(&self.session_id, *expected, target));
                }
                Ok(())
            }
        }
    }

    pub fn into_stream(
        mut self,
        stream: ManyOut<LlmResponse>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        let Some(inner) = self.coordinator.upgrade() else {
            return Err(anyhow::anyhow!("session coordinator dropped"));
        };
        let Some(kind) = self.kind.take() else {
            return Err(anyhow::anyhow!("session operation already completed"));
        };
        let completion = match kind {
            OperationKind::Opening { kind, target, .. } => {
                let Some(target) = target else {
                    abort_opening(&inner, &self.session_id, self.revision);
                    return Err(anyhow::anyhow!("session target not selected"));
                };
                let notify = {
                    let Some(mut state) = inner.sessions.get_mut(&self.session_id) else {
                        return Err(invalid_argument(format!(
                            "session {} is no longer opening",
                            self.session_id
                        )));
                    };
                    let SessionState::Opening(opening) = state.value_mut() else {
                        return Err(invalid_argument(format!(
                            "session {} is no longer opening",
                            self.session_id
                        )));
                    };
                    if opening.revision != self.revision {
                        return Err(invalid_argument(format!(
                            "session {} opening reservation changed",
                            self.session_id
                        )));
                    }
                    let notify = opening.notify.clone();
                    let timeout = opening.timeout;
                    *state = SessionState::Bound(BoundState {
                        revision: self.revision,
                        binding: SessionBinding {
                            target,
                            kind,
                            timeout,
                            expires_at: Instant::now() + timeout,
                        },
                        active_leases: 1,
                    });
                    notify
                };
                notify.notify_waiters();
                StreamCompletion::Lease
            }
            OperationKind::Lease { .. } => StreamCompletion::Lease,
            OperationKind::Closing { .. } => StreamCompletion::Close,
        };
        let context = stream.context();
        Ok(ResponseStream::new(
            Box::pin(SessionTrackedStream {
                inner: stream,
                coordinator: Arc::downgrade(&inner),
                session_id: self.session_id.clone(),
                revision: self.revision,
                completion: Some(completion),
            }),
            context,
        ))
    }
}

impl Drop for SessionOperation {
    fn drop(&mut self) {
        let Some(kind) = self.kind.take() else {
            return;
        };
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        match kind {
            OperationKind::Opening {
                target,
                backend_opened,
                ..
            } => {
                if backend_opened {
                    let Some(target) = target else {
                        abort_opening(&inner, &self.session_id, self.revision);
                        return;
                    };
                    opening_to_closing(&inner, &self.session_id, self.revision, target);
                    SessionCoordinator::ensure_close_task(&inner, self.session_id.clone());
                } else {
                    abort_opening(&inner, &self.session_id, self.revision);
                }
            }
            OperationKind::Lease { .. } => {
                release_lease(&inner, &self.session_id, self.revision, false);
            }
            OperationKind::Closing { prior, .. } => {
                let Some(mut state) = inner.sessions.get_mut(&self.session_id) else {
                    return;
                };
                if matches!(
                    state.value(),
                    SessionState::Closing(closing) if closing.revision == self.revision
                ) {
                    *state = SessionState::Bound(prior);
                }
            }
        }
    }
}

fn abort_opening(inner: &SessionCoordinatorInner, session_id: &str, revision: u64) {
    let removed = inner.sessions.remove_if(
        session_id,
        |_, state| matches!(state, SessionState::Opening(opening) if opening.revision == revision),
    );
    let Some((_, SessionState::Opening(opening))) = removed else {
        return;
    };
    opening.notify.notify_waiters();
}

fn opening_to_closing(
    inner: &SessionCoordinatorInner,
    session_id: &str,
    revision: u64,
    target: SessionTarget,
) {
    let notify = {
        let Some(mut state) = inner.sessions.get_mut(session_id) else {
            return;
        };
        let SessionState::Opening(opening) = state.value_mut() else {
            return;
        };
        if opening.revision != revision {
            return;
        }
        let notify = opening.notify.clone();
        let timeout = opening.timeout;
        *state = SessionState::Closing(ClosingState {
            revision,
            binding: SessionBinding {
                target,
                kind: SessionKind::EngineBacked,
                timeout,
                expires_at: Instant::now(),
            },
            remove_at: Instant::now() + timeout + SESSION_TIMEOUT_FALLBACK_BUFFER,
            retry_started: false,
        });
        notify
    };
    notify.notify_waiters();
}

fn release_lease(inner: &SessionCoordinatorInner, session_id: &str, revision: u64, refresh: bool) {
    let Some(mut state) = inner.sessions.get_mut(session_id) else {
        return;
    };
    let SessionState::Bound(bound) = state.value_mut() else {
        return;
    };
    if bound.revision != revision || bound.active_leases == 0 {
        return;
    }
    bound.active_leases -= 1;
    if refresh {
        bound.binding.expires_at = Instant::now() + bound.binding.timeout;
    }
}

enum StreamCompletion {
    Lease,
    Close,
}

struct SessionTrackedStream {
    inner: ManyOut<LlmResponse>,
    coordinator: Weak<SessionCoordinatorInner>,
    session_id: String,
    revision: u64,
    completion: Option<StreamCompletion>,
}

impl std::fmt::Debug for SessionTrackedStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionTrackedStream")
            .field("session_id", &self.session_id)
            .field("revision", &self.revision)
            .finish()
    }
}

impl SessionTrackedStream {
    fn complete(&mut self) {
        let Some(completion) = self.completion.take() else {
            return;
        };
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        match completion {
            StreamCompletion::Lease => {
                release_lease(&inner, &self.session_id, self.revision, true);
            }
            StreamCompletion::Close => {
                SessionCoordinator::complete_stream_close(
                    &inner,
                    self.session_id.clone(),
                    self.revision,
                );
            }
        }
    }
}

impl Drop for SessionTrackedStream {
    fn drop(&mut self) {
        self.complete();
    }
}

impl Stream for SessionTrackedStream {
    type Item = LlmResponse;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.inner.as_mut().poll_next(cx);
        if matches!(poll, Poll::Ready(None)) {
            self.complete();
        }
        poll
    }
}

impl AsyncEngineContextProvider for SessionTrackedStream {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.inner.context()
    }
}

impl AsyncEngineStream<LlmResponse> for SessionTrackedStream {}
