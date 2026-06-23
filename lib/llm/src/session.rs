// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing-neutral session affinity and backend lifecycle coordination.

use std::{
    collections::HashMap,
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use async_trait::async_trait;
use dynamo_runtime::{
    component::Component,
    engine::{AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream},
    error::{DynamoError, ErrorType},
    pipeline::{
        AsyncEngine, Error, ManyOut, PushRouter, ResponseStream, RouterMode, SingleIn,
        async_trait as pipeline_async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use tokio::{sync::Mutex as AsyncMutex, time::Instant};
use tokio_util::sync::CancellationToken;

use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SessionAction, SessionControl},
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL},
    },
};

const REAPER_INTERVAL: Duration = Duration::from_secs(30);
const SESSION_TIMEOUT_FALLBACK_BUFFER: Duration = Duration::from_secs(30);
const DEFAULT_SESSION_CAPACITY: u64 = 65_536;

type LlmResponse = Annotated<LLMEngineOutput>;
type EventPlaneClient = PushRouter<serde_json::Value, Annotated<serde_json::Value>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionTarget {
    pub worker_id: u64,
    pub dp_rank: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SessionKind {
    RouterOnly,
    EngineBacked,
}

#[derive(Clone, Debug)]
struct SessionBinding {
    target: SessionTarget,
    kind: SessionKind,
    timeout: Duration,
    expires_at: Instant,
}

#[derive(Clone, Debug)]
struct OpeningState {
    revision: u64,
    kind: SessionKind,
    requested_target: Option<SessionTarget>,
    target: Option<SessionTarget>,
    timeout: Duration,
    notify: Arc<tokio::sync::Notify>,
}

#[derive(Clone, Debug)]
struct BoundState {
    revision: u64,
    binding: SessionBinding,
    active_leases: usize,
}

#[derive(Clone, Debug)]
struct ClosingState {
    revision: u64,
    binding: SessionBinding,
    remove_at: Instant,
    retry_started: bool,
}

#[derive(Clone, Debug)]
enum SessionState {
    Opening(OpeningState),
    Bound(BoundState),
    Closing(ClosingState),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenOutcome {
    Created,
    AlreadyExists,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleErrorKind {
    Definitive,
    Ambiguous,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct LifecycleError {
    kind: LifecycleErrorKind,
    message: String,
}

impl LifecycleError {
    fn definitive(message: impl Into<String>) -> Self {
        Self {
            kind: LifecycleErrorKind::Definitive,
            message: message.into(),
        }
    }

    fn ambiguous(message: impl Into<String>) -> Self {
        Self {
            kind: LifecycleErrorKind::Ambiguous,
            message: message.into(),
        }
    }
}

#[async_trait]
pub trait SessionLifecycleBackend: Send + Sync {
    async fn open(
        &self,
        session_id: &str,
        timeout: Duration,
        target: SessionTarget,
        context_id: &str,
    ) -> Result<OpenOutcome, LifecycleError>;

    async fn close(
        &self,
        session_id: &str,
        target: SessionTarget,
        context_id: &str,
    ) -> Result<(), LifecycleError>;
}

pub struct EventSessionLifecycle {
    component: Component,
    client: AsyncMutex<Option<EventPlaneClient>>,
}

impl EventSessionLifecycle {
    pub fn new(component: Component) -> Self {
        Self {
            component,
            client: AsyncMutex::new(None),
        }
    }

    async fn client(&self) -> Result<EventPlaneClient, LifecycleError> {
        let mut cached = self.client.lock().await;
        if let Some(client) = cached.as_ref() {
            return Ok(client.clone());
        }

        let client = self
            .component
            .endpoint("session_control")
            .client()
            .await
            .map_err(|error| {
                LifecycleError::definitive(format!(
                    "failed to create session-control client: {error}"
                ))
            })?;
        tokio::time::timeout(Duration::from_secs(5), client.wait_for_instances())
            .await
            .map_err(|_| {
                LifecycleError::definitive(
                    "no session-control endpoint registered within five seconds",
                )
            })?
            .map_err(|error| {
                LifecycleError::definitive(format!(
                    "failed waiting for a session-control endpoint: {error}"
                ))
            })?;
        let router = EventPlaneClient::from_client_no_fault_detection(client, RouterMode::KV)
            .await
            .map_err(|error| {
                LifecycleError::definitive(format!(
                    "failed to create session-control router: {error}"
                ))
            })?;
        *cached = Some(router.clone());
        Ok(router)
    }

    async fn send(
        &self,
        request: serde_json::Value,
        session_id: &str,
        target: SessionTarget,
        context_id: &str,
        action: &str,
    ) -> Result<Annotated<serde_json::Value>, LifecycleError> {
        let client = self.client().await?;
        let mut stream = client
            .dispatch_exact(SingleIn::new(request), target.worker_id)
            .await
            .map_err(|error| {
                LifecycleError::ambiguous(format!(
                    "{action} RPC failed for session {session_id}: {error}"
                ))
            })?;
        let response = stream.next().await.ok_or_else(|| {
            LifecycleError::ambiguous(format!(
                "{action} returned no response for session {session_id}"
            ))
        })?;
        while stream.next().await.is_some() {}
        tracing::info!(
            request_id = %context_id,
            worker_id = target.worker_id,
            %session_id,
            ?response,
            "{action} response"
        );
        Ok(response)
    }

    fn response_body<'a>(
        response: &'a Annotated<serde_json::Value>,
        session_id: &str,
        action: &str,
    ) -> Result<&'a serde_json::Value, LifecycleError> {
        if response.is_error() {
            return Err(LifecycleError::definitive(format!(
                "{action} returned an annotated error for session {session_id}"
            )));
        }
        let body = response.data.as_ref().ok_or_else(|| {
            LifecycleError::ambiguous(format!(
                "{action} returned no response body for session {session_id}"
            ))
        })?;
        match body.get("status").and_then(serde_json::Value::as_str) {
            Some("ok") => Ok(body),
            Some(status) => Err(LifecycleError::definitive(format!(
                "{action} failed for session {session_id}: status={status}, message={}",
                body.get("message")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown error")
            ))),
            None => Err(LifecycleError::ambiguous(format!(
                "{action} returned a malformed response for session {session_id}"
            ))),
        }
    }
}

#[async_trait]
impl SessionLifecycleBackend for EventSessionLifecycle {
    async fn open(
        &self,
        session_id: &str,
        timeout: Duration,
        target: SessionTarget,
        context_id: &str,
    ) -> Result<OpenOutcome, LifecycleError> {
        let worker_timeout = timeout.saturating_add(SESSION_TIMEOUT_FALLBACK_BUFFER);
        let request = serde_json::json!({
            "action": "open_session",
            "session_id": session_id,
            "timeout": worker_timeout.as_secs(),
            "capacity_of_str_len": DEFAULT_SESSION_CAPACITY,
        });
        let response = self
            .send(request, session_id, target, context_id, "open_session")
            .await?;
        let body = Self::response_body(&response, session_id, "open_session")?;
        match body.get("created").and_then(serde_json::Value::as_bool) {
            Some(true) => Ok(OpenOutcome::Created),
            Some(false) => Ok(OpenOutcome::AlreadyExists),
            None => Err(LifecycleError::ambiguous(format!(
                "open_session response for session {session_id} is missing created"
            ))),
        }
    }

    async fn close(
        &self,
        session_id: &str,
        target: SessionTarget,
        context_id: &str,
    ) -> Result<(), LifecycleError> {
        let request = serde_json::json!({
            "action": "close_session",
            "session_id": session_id,
        });
        let response = self
            .send(request, session_id, target, context_id, "close_session")
            .await?;
        Self::response_body(&response, session_id, "close_session")?;
        Ok(())
    }
}

struct SessionCoordinatorInner {
    sessions: Mutex<HashMap<String, SessionState>>,
    lifecycle: Arc<dyn SessionLifecycleBackend>,
    next_revision: AtomicU64,
    cancel: CancellationToken,
    #[cfg(test)]
    test_hooks: Option<SessionTestHooks>,
}

#[cfg(test)]
#[derive(Clone)]
struct SessionTestHooks {
    retry_armed: Arc<tokio::sync::Semaphore>,
    reaper_armed: Arc<tokio::sync::Semaphore>,
    coalesce_waiting: Arc<tokio::sync::Semaphore>,
    retry_stopped: Arc<tokio::sync::Semaphore>,
    reaper_stopped: Arc<tokio::sync::Semaphore>,
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
    inner: Arc<SessionCoordinatorInner>,
}

impl SessionCoordinator {
    pub fn new(component: Component) -> Self {
        Self::with_lifecycle(Arc::new(EventSessionLifecycle::new(component)))
    }

    pub fn with_lifecycle(lifecycle: Arc<dyn SessionLifecycleBackend>) -> Self {
        let inner = Arc::new(SessionCoordinatorInner {
            sessions: Mutex::new(HashMap::new()),
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
    fn with_lifecycle_and_hooks(
        lifecycle: Arc<dyn SessionLifecycleBackend>,
        test_hooks: SessionTestHooks,
    ) -> Self {
        let inner = Arc::new(SessionCoordinatorInner {
            sessions: Mutex::new(HashMap::new()),
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
        {
            let mut sessions = inner.sessions.lock();
            sessions.retain(|session_id, state| match state {
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
        }
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
            let mut close_expired = false;
            let mut wait = None;
            let operation = {
                let now = Instant::now();
                let mut sessions = self.inner.sessions.lock();
                if let Some(SessionState::Bound(bound)) = sessions.get(&control.session_id)
                    && bound.active_leases == 0
                    && bound.binding.expires_at <= now
                {
                    if bound.binding.kind == SessionKind::RouterOnly {
                        sessions.remove(&control.session_id);
                    } else {
                        let binding = bound.binding.clone();
                        let revision = bound.revision;
                        sessions.insert(
                            control.session_id.clone(),
                            SessionState::Closing(ClosingState {
                                revision,
                                remove_at: now + binding.timeout + SESSION_TIMEOUT_FALLBACK_BUFFER,
                                binding,
                                retry_started: false,
                            }),
                        );
                        close_expired = true;
                    }
                }

                match sessions.get_mut(&control.session_id) {
                    None => {
                        let Some(kind) = requested_kind else {
                            return Err(invalid_argument(format!(
                                "session {} must be opened or bound before continuation or close",
                                control.session_id
                            )));
                        };
                        let revision = self.inner.next_revision.fetch_add(1, Ordering::Relaxed);
                        let notify = Arc::new(tokio::sync::Notify::new());
                        sessions.insert(
                            control.session_id.clone(),
                            SessionState::Opening(OpeningState {
                                revision,
                                kind,
                                requested_target: explicit_target,
                                target: None,
                                timeout,
                                notify,
                            }),
                        );
                        Some(SessionOperation::opening(
                            Arc::downgrade(&self.inner),
                            control.session_id.clone(),
                            revision,
                            kind,
                        ))
                    }
                    Some(SessionState::Opening(opening)) => {
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
                    Some(SessionState::Bound(bound)) => {
                        if let Some(kind) = requested_kind {
                            ensure_binding_matches(&bound.binding, kind, explicit_target, timeout)?;
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
                            *sessions
                                .get_mut(&control.session_id)
                                .expect("bound session") = SessionState::Closing(ClosingState {
                                revision,
                                remove_at: now + binding.timeout + SESSION_TIMEOUT_FALLBACK_BUFFER,
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
                    Some(SessionState::Closing(_)) => {
                        return Err(invalid_argument(format!(
                            "session {} is closing and is not routable",
                            control.session_id
                        )));
                    }
                }
            };

            if close_expired {
                Self::ensure_close_task(&self.inner, control.session_id.clone());
                return Err(invalid_argument(format!(
                    "session {} expired and is closing",
                    control.session_id
                )));
            }
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
        let sessions = self.inner.sessions.lock();
        match sessions.get(&control.session_id) {
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
            let mut sessions = inner.sessions.lock();
            let Some(SessionState::Closing(closing)) = sessions.get_mut(&session_id) else {
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
                    let sessions = inner.sessions.lock();
                    match sessions.get(&session_id) {
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
        let engine_backed = {
            let mut sessions = inner.sessions.lock();
            let Some(SessionState::Closing(closing)) = sessions.get(&session_id) else {
                return;
            };
            if closing.revision != revision {
                return;
            }
            if closing.binding.kind == SessionKind::RouterOnly {
                sessions.remove(&session_id);
                return;
            }
            true
        };
        if engine_backed {
            Self::ensure_close_task(inner, session_id);
        }
    }
}

fn remove_closing(inner: &SessionCoordinatorInner, session_id: &str, revision: u64) {
    inner.sessions.lock().retain(|id, state| {
        id != session_id
            || !matches!(state, SessionState::Closing(closing) if closing.revision == revision)
    });
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

fn invalid_argument(message: impl Into<String>) -> Error {
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
                {
                    let mut sessions = inner.sessions.lock();
                    let Some(SessionState::Opening(opening)) = sessions.get_mut(&self.session_id)
                    else {
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
                }

                if *kind == SessionKind::RouterOnly {
                    return Ok(());
                }

                match inner
                    .lifecycle
                    .open(
                        &self.session_id,
                        opening_timeout(&inner, &self.session_id, self.revision)?,
                        target,
                        context_id,
                    )
                    .await
                {
                    Ok(OpenOutcome::Created) => {
                        *backend_opened = true;
                        Ok(())
                    }
                    Ok(OpenOutcome::AlreadyExists) => {
                        abort_opening(&inner, &self.session_id, self.revision);
                        self.kind = None;
                        Err(invalid_argument(format!(
                            "backend session {} already exists",
                            self.session_id
                        )))
                    }
                    Err(error) => {
                        if error.kind == LifecycleErrorKind::Ambiguous {
                            opening_to_closing(&inner, &self.session_id, self.revision, target);
                            SessionCoordinator::ensure_close_task(&inner, self.session_id.clone());
                        } else {
                            abort_opening(&inner, &self.session_id, self.revision);
                        }
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
                    let mut sessions = inner.sessions.lock();
                    let Some(SessionState::Opening(opening)) = sessions.get(&self.session_id)
                    else {
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
                    sessions.insert(
                        self.session_id.clone(),
                        SessionState::Bound(BoundState {
                            revision: self.revision,
                            binding: SessionBinding {
                                target,
                                kind,
                                timeout,
                                expires_at: Instant::now() + timeout,
                            },
                            active_leases: 1,
                        }),
                    );
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
                let mut sessions = inner.sessions.lock();
                if matches!(
                    sessions.get(&self.session_id),
                    Some(SessionState::Closing(closing)) if closing.revision == self.revision
                ) {
                    sessions.insert(self.session_id.clone(), SessionState::Bound(prior));
                }
            }
        }
    }
}

fn opening_timeout(
    inner: &SessionCoordinatorInner,
    session_id: &str,
    revision: u64,
) -> Result<Duration, Error> {
    let sessions = inner.sessions.lock();
    match sessions.get(session_id) {
        Some(SessionState::Opening(opening)) if opening.revision == revision => Ok(opening.timeout),
        _ => Err(invalid_argument(format!(
            "session {session_id} opening reservation changed"
        ))),
    }
}

fn abort_opening(inner: &SessionCoordinatorInner, session_id: &str, revision: u64) {
    let notify = {
        let mut sessions = inner.sessions.lock();
        let notify = match sessions.get(session_id) {
            Some(SessionState::Opening(opening)) if opening.revision == revision => {
                Some(opening.notify.clone())
            }
            _ => None,
        };
        if notify.is_some() {
            sessions.remove(session_id);
        }
        notify
    };
    if let Some(notify) = notify {
        notify.notify_waiters();
    }
}

fn opening_to_closing(
    inner: &SessionCoordinatorInner,
    session_id: &str,
    revision: u64,
    target: SessionTarget,
) {
    let notify = {
        let mut sessions = inner.sessions.lock();
        let Some(SessionState::Opening(opening)) = sessions.get(session_id) else {
            return;
        };
        if opening.revision != revision {
            return;
        }
        let notify = opening.notify.clone();
        let timeout = opening.timeout;
        sessions.insert(
            session_id.to_string(),
            SessionState::Closing(ClosingState {
                revision,
                binding: SessionBinding {
                    target,
                    kind: SessionKind::EngineBacked,
                    timeout,
                    expires_at: Instant::now(),
                },
                remove_at: Instant::now() + timeout + SESSION_TIMEOUT_FALLBACK_BUFFER,
                retry_started: false,
            }),
        );
        notify
    };
    notify.notify_waiters();
}

fn release_lease(inner: &SessionCoordinatorInner, session_id: &str, revision: u64, refresh: bool) {
    let mut sessions = inner.sessions.lock();
    let Some(SessionState::Bound(bound)) = sessions.get_mut(session_id) else {
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

pub struct SessionPushRouter {
    inner: PushRouter<PreprocessedRequest, LlmResponse>,
    coordinator: SessionCoordinator,
}

impl SessionPushRouter {
    pub fn new(inner: PushRouter<PreprocessedRequest, LlmResponse>) -> Self {
        let component = inner.component().clone();
        Self {
            inner,
            coordinator: SessionCoordinator::new(component),
        }
    }

    fn phase(request: &PreprocessedRequest) -> RequestPhase {
        request
            .tracker
            .as_ref()
            .map(|tracker| tracker.phase())
            .unwrap_or(RequestPhase::Aggregated)
    }

    fn record_target(request: &PreprocessedRequest, target: SessionTarget) {
        let Some(tracker) = request.tracker.as_ref() else {
            return;
        };
        let worker_type = if tracker.phase() == RequestPhase::Prefill {
            WORKER_TYPE_PREFILL
        } else {
            WORKER_TYPE_DECODE
        };
        tracker.record_worker(target.worker_id, target.dp_rank, worker_type);
    }

    pub fn peek_next_worker(&self) -> Option<u64> {
        self.inner.peek_next_worker()
    }

    pub async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<LlmResponse>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, u64, Option<u32>) -> Result<M, Error>,
    {
        let control = request
            .routing
            .as_ref()
            .and_then(|routing| routing.session_control.clone());
        if control.is_none() {
            let pinned = request
                .routing
                .as_ref()
                .and_then(|routing| routing.prefill_worker_id);
            return self
                .inner
                .select_and_dispatch_exact(request, pinned, |request, worker_id| {
                    prepare(request, worker_id, None)
                })
                .await;
        }

        let control = control.expect("checked above");
        let explicit = explicit_target(&request, RequestPhase::Prefill)?;
        let operation = self
            .coordinator
            .begin(&control, explicit, self.inner.is_direct_routing())
            .await?;
        let pinned = operation
            .target()
            .or(explicit)
            .map(|target| target.worker_id);
        let context_id = request.context().id().to_string();
        let rank = operation
            .target()
            .or(explicit)
            .and_then(|target| target.dp_rank);
        let ((operation, metadata), stream) = self
            .inner
            .select_and_dispatch_exact_async(
                request,
                pinned,
                move |worker_id| async move {
                    let mut operation = operation;
                    operation
                        .selected(
                            SessionTarget {
                                worker_id,
                                dp_rank: rank,
                            },
                            &context_id,
                        )
                        .await?;
                    Ok(operation)
                },
                move |request, worker_id, operation| {
                    let target = SessionTarget {
                        worker_id,
                        dp_rank: rank,
                    };
                    Self::record_target(request, target);
                    Ok((operation, prepare(request, worker_id, rank)?))
                },
            )
            .await?;
        Ok((metadata, operation.into_stream(stream)?))
    }
}

#[pipeline_async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<LlmResponse>, Error> for SessionPushRouter {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        let control = request
            .routing
            .as_ref()
            .and_then(|routing| routing.session_control.clone());
        if control.is_none() {
            if self.inner.is_direct_routing() {
                let worker_id = request
                    .routing
                    .as_ref()
                    .and_then(|routing| routing.decode_worker_id.or(routing.backend_instance_id))
                    .ok_or_else(|| anyhow::anyhow!("worker ID required in Direct routing mode"))?;
                return self.inner.direct(request, worker_id).await;
            }
            return self.inner.generate(request).await;
        }

        let control = control.expect("checked above");
        let phase = Self::phase(&request);
        let explicit = explicit_target(&request, phase)?;
        let operation = self
            .coordinator
            .begin(&control, explicit, self.inner.is_direct_routing())
            .await?;
        let selected_target = operation.target().or(explicit);
        let pinned = selected_target.map(|target| target.worker_id);
        let rank = selected_target.and_then(|target| target.dp_rank);
        let context_id = request.context().id().to_string();
        let (operation, stream) = self
            .inner
            .select_and_dispatch_exact_async(
                request,
                pinned,
                move |worker_id| async move {
                    let mut operation = operation;
                    operation
                        .selected(
                            SessionTarget {
                                worker_id,
                                dp_rank: rank,
                            },
                            &context_id,
                        )
                        .await?;
                    Ok(operation)
                },
                move |request, worker_id, operation| {
                    Self::record_target(
                        request,
                        SessionTarget {
                            worker_id,
                            dp_rank: rank,
                        },
                    );
                    Ok(operation)
                },
            )
            .await?;
        operation.into_stream(stream)
    }
}

pub fn explicit_target(
    request: &PreprocessedRequest,
    phase: RequestPhase,
) -> Result<Option<SessionTarget>, Error> {
    let Some(routing) = request.routing.as_ref() else {
        return Ok(None);
    };
    let (worker_id, dp_rank) = match phase {
        RequestPhase::Prefill => (
            routing.prefill_worker_id.or(routing.backend_instance_id),
            routing.prefill_dp_rank.or(routing.dp_rank),
        ),
        RequestPhase::Decode => (
            routing.decode_worker_id.or(routing.backend_instance_id),
            routing.dp_rank,
        ),
        RequestPhase::Aggregated => (routing.backend_instance_id, routing.dp_rank),
    };
    if worker_id.is_none() && dp_rank.is_some() {
        return Err(invalid_argument(
            "DP rank requires an explicit worker for session routing",
        ));
    }
    Ok(worker_id.map(|worker_id| SessionTarget { worker_id, dp_rank }))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::atomic::{AtomicBool, AtomicUsize},
    };

    use dynamo_runtime::pipeline::context::Controller;
    use futures::{FutureExt, stream};
    use tokio::sync::Semaphore;

    use super::*;
    use crate::protocols::common::preprocessor::RoutingHints;

    #[derive(Clone, Copy)]
    enum FakeOpen {
        Created,
        Existing,
        Ambiguous,
    }

    struct FakeLifecycle {
        open_result: Mutex<FakeOpen>,
        close_results: Mutex<VecDeque<bool>>,
        close_default: AtomicBool,
        block_open: AtomicBool,
        open_calls: AtomicUsize,
        close_calls: AtomicUsize,
        open_started: Arc<Semaphore>,
        open_release: Arc<Semaphore>,
        close_started: Arc<Semaphore>,
    }

    impl FakeLifecycle {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                open_result: Mutex::new(FakeOpen::Created),
                close_results: Mutex::new(VecDeque::new()),
                close_default: AtomicBool::new(true),
                block_open: AtomicBool::new(false),
                open_calls: AtomicUsize::new(0),
                close_calls: AtomicUsize::new(0),
                open_started: Arc::new(Semaphore::new(0)),
                open_release: Arc::new(Semaphore::new(0)),
                close_started: Arc::new(Semaphore::new(0)),
            })
        }

        fn close_sequence(&self, results: impl IntoIterator<Item = bool>) {
            *self.close_results.lock() = results.into_iter().collect();
        }
    }

    #[async_trait]
    impl SessionLifecycleBackend for FakeLifecycle {
        async fn open(
            &self,
            _session_id: &str,
            _timeout: Duration,
            _target: SessionTarget,
            _context_id: &str,
        ) -> Result<OpenOutcome, LifecycleError> {
            self.open_calls.fetch_add(1, Ordering::Relaxed);
            if self.block_open.load(Ordering::Relaxed) {
                self.open_started.add_permits(1);
                self.open_release
                    .acquire()
                    .await
                    .expect("open release semaphore")
                    .forget();
            }
            match *self.open_result.lock() {
                FakeOpen::Created => Ok(OpenOutcome::Created),
                FakeOpen::Existing => Ok(OpenOutcome::AlreadyExists),
                FakeOpen::Ambiguous => Err(LifecycleError::ambiguous("ambiguous open")),
            }
        }

        async fn close(
            &self,
            _session_id: &str,
            _target: SessionTarget,
            _context_id: &str,
        ) -> Result<(), LifecycleError> {
            self.close_calls.fetch_add(1, Ordering::Relaxed);
            self.close_started.add_permits(1);
            let succeeds = self
                .close_results
                .lock()
                .pop_front()
                .unwrap_or_else(|| self.close_default.load(Ordering::Relaxed));
            if succeeds {
                Ok(())
            } else {
                Err(LifecycleError::ambiguous("close failed"))
            }
        }
    }

    fn hooks() -> SessionTestHooks {
        SessionTestHooks {
            retry_armed: Arc::new(Semaphore::new(0)),
            reaper_armed: Arc::new(Semaphore::new(0)),
            coalesce_waiting: Arc::new(Semaphore::new(0)),
            retry_stopped: Arc::new(Semaphore::new(0)),
            reaper_stopped: Arc::new(Semaphore::new(0)),
        }
    }

    fn coordinator(lifecycle: Arc<FakeLifecycle>, hooks: SessionTestHooks) -> SessionCoordinator {
        SessionCoordinator::with_lifecycle_and_hooks(lifecycle, hooks)
    }

    fn control(action: Option<SessionAction>) -> SessionControl {
        SessionControl {
            session_id: "session-1".to_string(),
            action,
            timeout: 10,
        }
    }

    fn target(worker_id: u64) -> SessionTarget {
        SessionTarget {
            worker_id,
            dp_rank: Some(0),
        }
    }

    fn binding(kind: SessionKind) -> SessionBinding {
        SessionBinding {
            target: target(1),
            kind,
            timeout: Duration::from_secs(10),
            expires_at: Instant::now() + Duration::from_secs(10),
        }
    }

    fn empty_stream() -> ManyOut<LlmResponse> {
        ResponseStream::new(Box::pin(stream::empty()), Arc::new(Controller::default()))
    }

    #[derive(Clone, Copy)]
    enum SeedState {
        Absent,
        Opening(SessionKind),
        Bound(SessionKind, usize),
        Closing,
    }

    fn seed(coordinator: &SessionCoordinator, state: SeedState) {
        let state = match state {
            SeedState::Absent => return,
            SeedState::Opening(kind) => SessionState::Opening(OpeningState {
                revision: 1,
                kind,
                requested_target: None,
                target: Some(target(1)),
                timeout: Duration::from_secs(10),
                notify: Arc::new(tokio::sync::Notify::new()),
            }),
            SeedState::Bound(kind, active_leases) => SessionState::Bound(BoundState {
                revision: 1,
                binding: binding(kind),
                active_leases,
            }),
            SeedState::Closing => SessionState::Closing(ClosingState {
                revision: 1,
                binding: binding(SessionKind::EngineBacked),
                remove_at: Instant::now() + Duration::from_secs(40),
                retry_started: true,
            }),
        };
        coordinator
            .inner
            .sessions
            .lock()
            .insert("session-1".to_string(), state);
    }

    #[derive(Clone, Copy)]
    enum Expected {
        Ready,
        Error,
        Waiting,
    }

    #[tokio::test]
    async fn transition_contract_table() {
        struct Case {
            name: &'static str,
            state: SeedState,
            action: Option<SessionAction>,
            expected: Expected,
        }

        let cases = [
            Case {
                name: "absent open",
                state: SeedState::Absent,
                action: Some(SessionAction::Open),
                expected: Expected::Ready,
            },
            Case {
                name: "absent bind",
                state: SeedState::Absent,
                action: Some(SessionAction::Bind),
                expected: Expected::Ready,
            },
            Case {
                name: "absent continue",
                state: SeedState::Absent,
                action: None,
                expected: Expected::Error,
            },
            Case {
                name: "absent close",
                state: SeedState::Absent,
                action: Some(SessionAction::Close),
                expected: Expected::Error,
            },
            Case {
                name: "opening matching open",
                state: SeedState::Opening(SessionKind::EngineBacked),
                action: Some(SessionAction::Open),
                expected: Expected::Waiting,
            },
            Case {
                name: "opening conflicting bind",
                state: SeedState::Opening(SessionKind::EngineBacked),
                action: Some(SessionAction::Bind),
                expected: Expected::Error,
            },
            Case {
                name: "opening continue",
                state: SeedState::Opening(SessionKind::EngineBacked),
                action: None,
                expected: Expected::Error,
            },
            Case {
                name: "opening close",
                state: SeedState::Opening(SessionKind::EngineBacked),
                action: Some(SessionAction::Close),
                expected: Expected::Error,
            },
            Case {
                name: "bound continue",
                state: SeedState::Bound(SessionKind::EngineBacked, 0),
                action: None,
                expected: Expected::Ready,
            },
            Case {
                name: "bound matching open",
                state: SeedState::Bound(SessionKind::EngineBacked, 0),
                action: Some(SessionAction::Open),
                expected: Expected::Ready,
            },
            Case {
                name: "bound conflicting bind",
                state: SeedState::Bound(SessionKind::EngineBacked, 0),
                action: Some(SessionAction::Bind),
                expected: Expected::Error,
            },
            Case {
                name: "bound close",
                state: SeedState::Bound(SessionKind::EngineBacked, 0),
                action: Some(SessionAction::Close),
                expected: Expected::Ready,
            },
            Case {
                name: "bound close with lease",
                state: SeedState::Bound(SessionKind::EngineBacked, 1),
                action: Some(SessionAction::Close),
                expected: Expected::Error,
            },
            Case {
                name: "closing continue",
                state: SeedState::Closing,
                action: None,
                expected: Expected::Error,
            },
            Case {
                name: "closing open",
                state: SeedState::Closing,
                action: Some(SessionAction::Open),
                expected: Expected::Error,
            },
            Case {
                name: "closing bind",
                state: SeedState::Closing,
                action: Some(SessionAction::Bind),
                expected: Expected::Error,
            },
            Case {
                name: "closing duplicate close",
                state: SeedState::Closing,
                action: Some(SessionAction::Close),
                expected: Expected::Error,
            },
        ];

        for case in cases {
            let lifecycle = FakeLifecycle::new();
            let coordinator = coordinator(lifecycle, hooks());
            seed(&coordinator, case.state);
            let request = control(case.action);
            let mut future = Box::pin(coordinator.begin(&request, None, false));
            match case.expected {
                Expected::Ready => assert!(
                    future.as_mut().now_or_never().unwrap().is_ok(),
                    "{}",
                    case.name
                ),
                Expected::Error => assert!(
                    future.as_mut().now_or_never().unwrap().is_err(),
                    "{}",
                    case.name
                ),
                Expected::Waiting => {
                    assert!(future.as_mut().now_or_never().is_none(), "{}", case.name)
                }
            }
        }
    }

    #[tokio::test]
    async fn duplicate_ensure_contract_is_parameterized() {
        struct Case {
            name: &'static str,
            kind: SessionKind,
            action: SessionAction,
            requested: Option<SessionTarget>,
            timeout: u64,
            succeeds: bool,
        }
        let cases = [
            Case {
                name: "matching open",
                kind: SessionKind::EngineBacked,
                action: SessionAction::Open,
                requested: Some(target(1)),
                timeout: 10,
                succeeds: true,
            },
            Case {
                name: "matching open without rank",
                kind: SessionKind::EngineBacked,
                action: SessionAction::Open,
                requested: Some(SessionTarget {
                    worker_id: 1,
                    dp_rank: None,
                }),
                timeout: 10,
                succeeds: true,
            },
            Case {
                name: "open target conflict",
                kind: SessionKind::EngineBacked,
                action: SessionAction::Open,
                requested: Some(target(2)),
                timeout: 10,
                succeeds: false,
            },
            Case {
                name: "open timeout conflict",
                kind: SessionKind::EngineBacked,
                action: SessionAction::Open,
                requested: Some(target(1)),
                timeout: 11,
                succeeds: false,
            },
            Case {
                name: "matching bind",
                kind: SessionKind::RouterOnly,
                action: SessionAction::Bind,
                requested: Some(target(1)),
                timeout: 10,
                succeeds: true,
            },
            Case {
                name: "bind kind conflict",
                kind: SessionKind::EngineBacked,
                action: SessionAction::Bind,
                requested: Some(target(1)),
                timeout: 10,
                succeeds: false,
            },
        ];

        for case in cases {
            let coordinator = coordinator(FakeLifecycle::new(), hooks());
            seed(&coordinator, SeedState::Bound(case.kind, 0));
            let request = SessionControl {
                session_id: "session-1".to_string(),
                action: Some(case.action),
                timeout: case.timeout,
            };
            assert_eq!(
                coordinator
                    .begin(&request, case.requested, false)
                    .await
                    .is_ok(),
                case.succeeds,
                "{}",
                case.name
            );
        }
    }

    #[tokio::test]
    async fn concurrent_open_coalesces_backend_setup() {
        let lifecycle = FakeLifecycle::new();
        lifecycle.block_open.store(true, Ordering::Relaxed);
        let hooks = hooks();
        let coordinator = coordinator(lifecycle.clone(), hooks.clone());
        let first_coordinator = coordinator.clone();
        let first = tokio::spawn(async move {
            let mut operation = first_coordinator
                .begin(&control(Some(SessionAction::Open)), None, false)
                .await
                .unwrap();
            operation.selected(target(1), "first").await.unwrap();
            operation.into_stream(empty_stream()).unwrap()
        });

        lifecycle.open_started.acquire().await.unwrap().forget();
        let second_coordinator = coordinator.clone();
        let second = tokio::spawn(async move {
            second_coordinator
                .begin(&control(Some(SessionAction::Open)), None, false)
                .await
                .unwrap()
        });
        hooks.coalesce_waiting.acquire().await.unwrap().forget();
        lifecycle.open_release.add_permits(1);

        let first_stream = first.await.unwrap();
        let second_operation = second.await.unwrap();
        assert_eq!(lifecycle.open_calls.load(Ordering::Relaxed), 1);
        drop(first_stream);
        drop(second_operation);
    }

    #[tokio::test(start_paused = true)]
    async fn active_lease_prevents_expiry() {
        let hooks = hooks();
        let coordinator = coordinator(FakeLifecycle::new(), hooks.clone());
        coordinator.inner.sessions.lock().insert(
            "session-1".to_string(),
            SessionState::Bound(BoundState {
                revision: 1,
                binding: SessionBinding {
                    expires_at: Instant::now() + Duration::from_secs(1),
                    ..binding(SessionKind::RouterOnly)
                },
                active_leases: 1,
            }),
        );
        hooks.reaper_armed.acquire().await.unwrap().forget();
        tokio::time::advance(REAPER_INTERVAL + Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        assert!(matches!(
            coordinator.inner.sessions.lock().get("session-1"),
            Some(SessionState::Bound(bound)) if bound.active_leases == 1
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn stream_cleanup_releases_lease_after_eof_error_or_drop() {
        enum Completion {
            Eof,
            Error,
            Drop,
        }

        for completion in [Completion::Eof, Completion::Error, Completion::Drop] {
            let coordinator = coordinator(FakeLifecycle::new(), hooks());
            let mut initial_binding = binding(SessionKind::RouterOnly);
            initial_binding.expires_at = Instant::now() + Duration::from_secs(1);
            coordinator.inner.sessions.lock().insert(
                "session-1".to_string(),
                SessionState::Bound(BoundState {
                    revision: 1,
                    binding: initial_binding,
                    active_leases: 0,
                }),
            );
            let mut operation = coordinator
                .begin(&control(None), None, false)
                .await
                .unwrap();
            operation.selected(target(1), "continue").await.unwrap();
            let inner = match completion {
                Completion::Error => ResponseStream::new(
                    Box::pin(stream::iter(vec![Annotated::from_error("stream failed")])),
                    Arc::new(Controller::default()),
                ),
                Completion::Eof | Completion::Drop => empty_stream(),
            };
            let mut stream = operation.into_stream(inner).unwrap();
            match completion {
                Completion::Eof => assert!(stream.next().await.is_none()),
                Completion::Error => {
                    assert!(stream.next().await.unwrap().is_error());
                    assert!(stream.next().await.is_none());
                }
                Completion::Drop => drop(stream),
            }

            let sessions = coordinator.inner.sessions.lock();
            let Some(SessionState::Bound(bound)) = sessions.get("session-1") else {
                panic!("session should remain bound after continuation cleanup");
            };
            assert_eq!(bound.active_leases, 0);
            assert!(bound.binding.expires_at >= Instant::now() + Duration::from_secs(10));
        }
    }

    #[tokio::test]
    async fn pre_dispatch_drop_rolls_back_opening_state() {
        let lifecycle = FakeLifecycle::new();
        let coordinator = coordinator(lifecycle.clone(), hooks());
        let operation = coordinator
            .begin(&control(Some(SessionAction::Bind)), None, false)
            .await
            .unwrap();
        drop(operation);
        assert!(coordinator.inner.sessions.lock().is_empty());

        let mut operation = coordinator
            .begin(&control(Some(SessionAction::Open)), None, false)
            .await
            .unwrap();
        operation.selected(target(1), "open").await.unwrap();
        drop(operation);
        lifecycle.close_started.acquire().await.unwrap().forget();
        tokio::task::yield_now().await;
        assert!(coordinator.inner.sessions.lock().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn engine_expiry_enters_closing() {
        let lifecycle = FakeLifecycle::new();
        lifecycle.close_default.store(false, Ordering::Relaxed);
        let hooks = hooks();
        let coordinator = coordinator(lifecycle.clone(), hooks.clone());
        coordinator.inner.sessions.lock().insert(
            "session-1".to_string(),
            SessionState::Bound(BoundState {
                revision: 1,
                binding: SessionBinding {
                    expires_at: Instant::now() + Duration::from_secs(1),
                    ..binding(SessionKind::EngineBacked)
                },
                active_leases: 0,
            }),
        );
        hooks.reaper_armed.acquire().await.unwrap().forget();
        tokio::time::advance(REAPER_INTERVAL).await;
        lifecycle.close_started.acquire().await.unwrap().forget();
        hooks.retry_armed.acquire().await.unwrap().forget();
        assert!(matches!(
            coordinator.inner.sessions.lock().get("session-1"),
            Some(SessionState::Closing(_))
        ));
        drop(coordinator);
        hooks.reaper_stopped.acquire().await.unwrap().forget();
        hooks.retry_stopped.acquire().await.unwrap().forget();
    }

    #[tokio::test]
    async fn validation_contract_table() {
        enum Case {
            ZeroTimeout,
            DirectWithoutWorker,
            RankWithoutWorker,
            TargetMismatch,
        }
        for case in [
            Case::ZeroTimeout,
            Case::DirectWithoutWorker,
            Case::RankWithoutWorker,
            Case::TargetMismatch,
        ] {
            let coordinator = coordinator(FakeLifecycle::new(), hooks());
            let result = match case {
                Case::ZeroTimeout => {
                    let mut request = control(Some(SessionAction::Bind));
                    request.timeout = 0;
                    coordinator.begin(&request, None, false).await.map(|_| ())
                }
                Case::DirectWithoutWorker => coordinator
                    .begin(&control(Some(SessionAction::Bind)), None, true)
                    .await
                    .map(|_| ()),
                Case::RankWithoutWorker => {
                    let request = PreprocessedRequest::builder()
                        .model("model".to_string())
                        .token_ids(vec![1])
                        .stop_conditions(Default::default())
                        .sampling_options(Default::default())
                        .output_options(Default::default())
                        .routing(Some(RoutingHints {
                            session_control: Some(control(Some(SessionAction::Bind))),
                            dp_rank: Some(1),
                            ..Default::default()
                        }))
                        .build()
                        .unwrap();
                    explicit_target(&request, RequestPhase::Aggregated).map(|_| ())
                }
                Case::TargetMismatch => {
                    seed(&coordinator, SeedState::Bound(SessionKind::RouterOnly, 0));
                    coordinator
                        .begin(
                            &control(None),
                            Some(SessionTarget {
                                worker_id: 2,
                                dp_rank: Some(0),
                            }),
                            false,
                        )
                        .await
                        .map(|_| ())
                }
            };
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn query_only_validation_does_not_mutate_session_or_call_lifecycle() {
        let lifecycle = FakeLifecycle::new();
        let coordinator = coordinator(lifecycle.clone(), hooks());

        assert_eq!(
            coordinator
                .query_target(&control(Some(SessionAction::Open)), None)
                .unwrap(),
            None
        );
        assert!(coordinator.inner.sessions.lock().is_empty());

        seed(&coordinator, SeedState::Bound(SessionKind::EngineBacked, 0));
        let (expires_at, active_leases) = {
            let sessions = coordinator.inner.sessions.lock();
            let Some(SessionState::Bound(bound)) = sessions.get("session-1") else {
                panic!("expected bound session");
            };
            (bound.binding.expires_at, bound.active_leases)
        };
        assert_eq!(
            coordinator.query_target(&control(None), None).unwrap(),
            Some(target(1))
        );
        assert_eq!(
            coordinator
                .query_target(&control(Some(SessionAction::Close)), None)
                .unwrap(),
            Some(target(1))
        );

        let sessions = coordinator.inner.sessions.lock();
        let Some(SessionState::Bound(bound)) = sessions.get("session-1") else {
            panic!("query-only request changed session state");
        };
        assert_eq!(bound.binding.expires_at, expires_at);
        assert_eq!(bound.active_leases, active_leases);
        assert_eq!(lifecycle.open_calls.load(Ordering::Relaxed), 0);
        assert_eq!(lifecycle.close_calls.load(Ordering::Relaxed), 0);
    }

    async fn close_stream(coordinator: &SessionCoordinator) -> ManyOut<LlmResponse> {
        seed(coordinator, SeedState::Bound(SessionKind::EngineBacked, 0));
        let mut operation = coordinator
            .begin(&control(Some(SessionAction::Close)), None, false)
            .await
            .unwrap();
        operation.selected(target(1), "close").await.unwrap();
        operation.into_stream(empty_stream()).unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn close_retry_keeps_tombstone_until_confirmed_success() {
        let lifecycle = FakeLifecycle::new();
        lifecycle.close_sequence([false, true]);
        let hooks = hooks();
        let coordinator = coordinator(lifecycle.clone(), hooks.clone());
        hooks.reaper_armed.acquire().await.unwrap().forget();
        let stream = close_stream(&coordinator).await;
        drop(stream);
        lifecycle.close_started.acquire().await.unwrap().forget();
        hooks.retry_armed.acquire().await.unwrap().forget();

        assert!(matches!(
            coordinator.inner.sessions.lock().get("session-1"),
            Some(SessionState::Closing(_))
        ));
        assert!(
            coordinator
                .begin(&control(None), None, false)
                .await
                .is_err()
        );

        tokio::time::advance(Duration::from_secs(1)).await;
        lifecycle.close_started.acquire().await.unwrap().forget();
        tokio::task::yield_now().await;
        assert!(coordinator.inner.sessions.lock().get("session-1").is_none());
        hooks.retry_stopped.acquire().await.unwrap().forget();
        drop(coordinator);
        hooks.reaper_stopped.acquire().await.unwrap().forget();
    }

    #[tokio::test(start_paused = true)]
    async fn always_failing_close_expires_tombstone() {
        let lifecycle = FakeLifecycle::new();
        lifecycle.close_default.store(false, Ordering::Relaxed);
        let hooks = hooks();
        let coordinator = coordinator(lifecycle.clone(), hooks.clone());
        hooks.reaper_armed.acquire().await.unwrap().forget();
        let stream = close_stream(&coordinator).await;
        drop(stream);
        lifecycle.close_started.acquire().await.unwrap().forget();
        hooks.retry_armed.acquire().await.unwrap().forget();

        let mut observed_calls = lifecycle.close_calls.load(Ordering::Relaxed);
        for _ in 0..45 {
            tokio::time::advance(Duration::from_secs(1)).await;
            tokio::task::yield_now().await;
            if coordinator.inner.sessions.lock().get("session-1").is_none() {
                break;
            }
            let calls = lifecycle.close_calls.load(Ordering::Relaxed);
            if calls > observed_calls {
                hooks.retry_armed.acquire().await.unwrap().forget();
                observed_calls = calls;
            }
        }
        assert!(coordinator.inner.sessions.lock().get("session-1").is_none());
        hooks.retry_stopped.acquire().await.unwrap().forget();
        drop(coordinator);
        hooks.reaper_stopped.acquire().await.unwrap().forget();
    }

    #[tokio::test(start_paused = true)]
    async fn ambiguous_open_starts_compensating_close() {
        let lifecycle = FakeLifecycle::new();
        *lifecycle.open_result.lock() = FakeOpen::Ambiguous;
        let hooks = hooks();
        let coordinator = coordinator(lifecycle.clone(), hooks.clone());
        hooks.reaper_armed.acquire().await.unwrap().forget();
        let mut operation = coordinator
            .begin(&control(Some(SessionAction::Open)), None, false)
            .await
            .unwrap();
        assert!(operation.selected(target(1), "open").await.is_err());
        lifecycle.close_started.acquire().await.unwrap().forget();
        tokio::task::yield_now().await;
        assert!(coordinator.inner.sessions.lock().get("session-1").is_none());
        hooks.retry_stopped.acquire().await.unwrap().forget();
        drop(coordinator);
        hooks.reaper_stopped.acquire().await.unwrap().forget();
    }

    #[tokio::test]
    async fn already_existing_open_is_rejected_without_attaching() {
        let lifecycle = FakeLifecycle::new();
        *lifecycle.open_result.lock() = FakeOpen::Existing;
        let coordinator = coordinator(lifecycle, hooks());
        let mut operation = coordinator
            .begin(&control(Some(SessionAction::Open)), None, false)
            .await
            .unwrap();
        assert!(operation.selected(target(1), "open").await.is_err());
        assert!(coordinator.inner.sessions.lock().get("session-1").is_none());
    }
}
