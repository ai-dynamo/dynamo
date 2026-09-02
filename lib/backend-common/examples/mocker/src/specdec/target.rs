// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    BackendError, DraftCleanupOutcomeV1, DynamoError,
    EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY, Endpoint, EndpointId, EngineConfig,
    ExternalDraftBinding, ExternalSpeculationLifecycleV1, GenerateContext, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, MetricsBindings, MetricsCtx,
    ModelRegistration, PreprocessedRequest, RouterHintEnvelope, SpeculativeDecodingRouterHintV1,
    WorkerConfig, WorkerRole, chunk, usage,
};
use futures::stream::BoxStream;
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore, oneshot};
use uuid::Uuid;

use super::config::{parse_target, worker_config};
use super::metrics::SpecdecMetrics;
use super::protocol::{DraftIdentity, FailureState, MAX_OUTPUT_TOKENS, Start};
use super::transport::{
    DraftClient, DraftClientConfig, DraftProposal, DraftSession, TransportError, TransportErrorKind,
};
use super::{DP_RANK, PROTOCOL, backend_error};

const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 8;
const KV_TRANSFER_PARAMS_KEY: &str = "kv_transfer_params";
const ROUTER_HINT_KEY: &str = "router_hint";

enum DraftExchange {
    Proposal {
        started_at: tokio::time::Instant,
        proposal: DraftProposal,
    },
    Cancelled {
        started_at: Option<tokio::time::Instant>,
        cleanup: Option<DraftCleanupOutcomeV1>,
    },
    Failure {
        started_at: Option<tokio::time::Instant>,
        phase: DraftFailurePhase,
        error: TransportError,
        cleanup: Option<DraftCleanupOutcomeV1>,
    },
    InjectedFailure {
        started_at: tokio::time::Instant,
        cleanup: CleanupResolution,
    },
}

struct CleanupResolution {
    outcome: DraftCleanupOutcomeV1,
    error: Option<TransportError>,
}

#[derive(Clone, Copy)]
enum DraftFailurePhase {
    Start,
    Proposal,
    Cleanup,
}

struct ClientPoolState {
    shutdown: bool,
    identity: Option<DraftIdentity>,
    idle: Vec<DraftClient>,
    active: HashMap<Uuid, DraftClient>,
}

struct ClientPool {
    config: DraftClientConfig,
    admission_permits: Arc<Semaphore>,
    connection_permits: Arc<Semaphore>,
    metrics: Arc<SpecdecMetrics>,
    state: Mutex<ClientPoolState>,
    active_operations: AtomicUsize,
    operations_done: Notify,
}

struct PoolOperation {
    pool: Arc<ClientPool>,
}

struct ClientLease {
    pool: Arc<ClientPool>,
    request_id: Uuid,
    key: DraftIdentity,
    client: Option<DraftClient>,
    admission_permit: Option<OwnedSemaphorePermit>,
    connection_permit: Option<OwnedSemaphorePermit>,
    operation: Option<PoolOperation>,
}

#[derive(Clone, Copy)]
enum LeaseDisposition {
    Recycle,
    Quarantine,
}

impl ClientPool {
    #[cfg(test)]
    fn new(config: DraftClientConfig, max_connections: usize) -> Self {
        Self::new_with_metrics(config, max_connections, Arc::new(SpecdecMetrics::default()))
    }

    fn new_with_metrics(
        mut config: DraftClientConfig,
        max_connections: usize,
        metrics: Arc<SpecdecMetrics>,
    ) -> Self {
        let max_sessions = config.max_sessions;
        config.max_sessions = 1;
        Self {
            config,
            admission_permits: Arc::new(Semaphore::new(max_sessions)),
            connection_permits: Arc::new(Semaphore::new(max_connections)),
            metrics,
            state: Mutex::new(ClientPoolState {
                shutdown: false,
                identity: None,
                idle: Vec::new(),
                active: HashMap::new(),
            }),
            active_operations: AtomicUsize::new(0),
            operations_done: Notify::new(),
        }
    }

    async fn start(
        self: &Arc<Self>,
        expected: DraftIdentity,
        request_id: Uuid,
        start: Start,
    ) -> Result<(ClientLease, DraftSession), TransportError> {
        let admission_permit =
            self.admission_permits
                .clone()
                .try_acquire_owned()
                .map_err(|error| match error {
                    tokio::sync::TryAcquireError::Closed => pool_closed_error(),
                    tokio::sync::TryAcquireError::NoPermits => TransportError::new(
                        TransportErrorKind::Backpressure,
                        super::protocol::FailureState::NotStarted,
                        "draft target session limit is reached",
                    ),
                })?;
        let connection_permit = tokio::time::timeout(
            self.config.start_timeout,
            self.connection_permits.clone().acquire_owned(),
        )
        .await
        .map_err(|_| {
            TransportError::new(
                TransportErrorKind::Timeout,
                super::protocol::FailureState::NotStarted,
                "draft connection pool admission timed out",
            )
        })?
        .map_err(|_| {
            TransportError::new(
                TransportErrorKind::Closed,
                super::protocol::FailureState::NotStarted,
                "draft connection pool is closed",
            )
        })?;
        let operation = self.begin_operation().await?;
        let (existing, stale) = self.take_idle(&expected).await?;
        for client in stale {
            let _ = client.shutdown().await;
        }
        let client = if let Some(client) = existing {
            client
        } else {
            DraftClient::connect_with_metrics(
                expected.clone(),
                self.config.clone(),
                self.metrics.clone(),
            )
            .await?
        };
        if let Err(error) = self.activate(request_id, client.clone()).await {
            let _ = client.shutdown().await;
            return Err(error);
        }
        let lease = ClientLease {
            pool: self.clone(),
            request_id,
            key: expected,
            client: Some(client.clone()),
            admission_permit: Some(admission_permit),
            connection_permit: Some(connection_permit),
            operation: Some(operation),
        };
        match client.start(request_id, start).await {
            Ok(session) => Ok((lease, session)),
            Err(error) => {
                if error.state == super::protocol::FailureState::NotStarted {
                    lease.recycle().await;
                } else {
                    lease.quarantine().await;
                }
                Err(error)
            }
        }
    }

    async fn begin_operation(self: &Arc<Self>) -> Result<PoolOperation, TransportError> {
        let state = self.state.lock().await;
        if state.shutdown {
            return Err(pool_closed_error());
        }
        self.active_operations.fetch_add(1, Ordering::AcqRel);
        Ok(PoolOperation { pool: self.clone() })
    }

    async fn take_idle(
        &self,
        expected: &DraftIdentity,
    ) -> Result<(Option<DraftClient>, Vec<DraftClient>), TransportError> {
        let mut state = self.state.lock().await;
        if state.shutdown {
            return Err(pool_closed_error());
        }
        let mut stale = Vec::new();
        if state.identity.as_ref() != Some(expected) {
            stale = std::mem::take(&mut state.idle);
            state.identity = Some(expected.clone());
        }
        let existing = loop {
            match state.idle.pop() {
                Some(client) if client.is_closed() => stale.push(client),
                client => break client,
            }
        };
        Ok((existing, stale))
    }

    async fn activate(&self, request_id: Uuid, client: DraftClient) -> Result<(), TransportError> {
        let mut state = self.state.lock().await;
        if state.shutdown {
            return Err(pool_closed_error());
        }
        match state.active.entry(request_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(client);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(TransportError::new(
                    TransportErrorKind::Protocol,
                    super::protocol::FailureState::NotStarted,
                    "draft request ID is already active in the target pool",
                ));
            }
        }
        Ok(())
    }

    async fn recycle(&self, request_id: Uuid, key: &DraftIdentity, client: DraftClient) {
        let should_recycle = {
            let mut state = self.state.lock().await;
            state.active.remove(&request_id);
            if !state.shutdown && state.identity.as_ref() == Some(key) && !client.is_closed() {
                state.idle.push(client.clone());
                true
            } else {
                false
            }
        };
        if !should_recycle {
            let _ = client.shutdown().await;
        }
    }

    async fn retire(&self, request_id: Uuid, client: DraftClient) {
        let _ = client.shutdown().await;
        self.state.lock().await.active.remove(&request_id);
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        self.admission_permits.close();
        self.connection_permits.close();
        let clients = {
            let mut state = self.state.lock().await;
            state.shutdown = true;
            let mut clients = std::mem::take(&mut state.idle);
            clients.extend(state.active.values().cloned());
            clients
        };
        for client in &clients {
            client.close();
        }
        let mut result = Ok(());
        for client in clients {
            if let Err(error) = client.shutdown().await {
                result = Err(error);
            }
        }
        loop {
            let operations_done = self.operations_done.notified();
            if self.active_operations.load(Ordering::Acquire) == 0 {
                break;
            }
            operations_done.await;
        }
        let mut state = self.state.lock().await;
        state.idle.clear();
        state.active.clear();
        result
    }
}

impl Drop for PoolOperation {
    fn drop(&mut self) {
        if self.pool.active_operations.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.pool.operations_done.notify_waiters();
        }
    }
}

impl ClientLease {
    #[cfg(test)]
    fn client(&self) -> &DraftClient {
        self.client.as_ref().expect("client lease is active")
    }

    async fn recycle(mut self) {
        let _ = self.spawn_finalizer(LeaseDisposition::Recycle).await;
    }

    async fn quarantine(mut self) {
        let _ = self.spawn_finalizer(LeaseDisposition::Quarantine).await;
    }

    fn spawn_finalizer(&mut self, disposition: LeaseDisposition) -> oneshot::Receiver<()> {
        let client = self.client.take().expect("client lease is active");
        if matches!(disposition, LeaseDisposition::Quarantine) {
            client.close();
        }
        let pool = self.pool.clone();
        let request_id = self.request_id;
        let key = self.key.clone();
        let admission_permit = self.admission_permit.take();
        let connection_permit = self.connection_permit.take();
        let operation = self.operation.take();
        let (finished, receiver) = oneshot::channel();
        tokio::spawn(async move {
            match disposition {
                LeaseDisposition::Recycle => pool.recycle(request_id, &key, client).await,
                LeaseDisposition::Quarantine => pool.retire(request_id, client).await,
            }
            drop(operation);
            drop(connection_permit);
            drop(admission_permit);
            let _ = finished.send(());
        });
        receiver
    }
}

impl Drop for ClientLease {
    fn drop(&mut self) {
        if self.client.is_some() {
            std::mem::drop(self.spawn_finalizer(LeaseDisposition::Quarantine));
        }
    }
}

async fn resolve_cleanup(
    mut session: DraftSession,
    lease: ClientLease,
    orphan_cleanup_timeout: Duration,
) -> CleanupResolution {
    match session.cleanup().await {
        Ok(()) => {
            lease.recycle().await;
            CleanupResolution {
                outcome: DraftCleanupOutcomeV1::Acknowledged,
                error: None,
            }
        }
        Err(error) => {
            lease.quarantine().await;
            tokio::time::sleep(orphan_cleanup_timeout).await;
            CleanupResolution {
                outcome: DraftCleanupOutcomeV1::CleanupBoundElapsed,
                error: Some(error),
            }
        }
    }
}

fn with_cleanup_lifecycle(
    mut output: LLMEngineOutput,
    cleanup: DraftCleanupOutcomeV1,
) -> LLMEngineOutput {
    let lifecycle = ExternalSpeculationLifecycleV1 {
        schema_version: ExternalSpeculationLifecycleV1::SCHEMA_VERSION,
        draft_cleanup: cleanup,
    };
    let mut engine_data = output
        .engine_data
        .take()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    engine_data.insert(
        EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY.to_string(),
        serde_json::to_value(lifecycle).expect("lifecycle serialization is infallible"),
    );
    output.engine_data = Some(serde_json::Value::Object(engine_data));
    output
}

fn pool_closed_error() -> TransportError {
    TransportError::new(
        TransportErrorKind::Closed,
        super::protocol::FailureState::NotStarted,
        "draft connection pool is closed",
    )
}

pub struct TargetEngine {
    model_name: String,
    context_length: u32,
    draft_endpoint: EndpointId,
    metrics: Arc<SpecdecMetrics>,
    clients: Arc<ClientPool>,
    target_prefill: Duration,
    target_token_interval: Duration,
    accepted_proposal_tokens: u32,
    fail_after_draft_start_prompt_token: Option<u32>,
}

impl TargetEngine {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = parse_target(argv)?;
        let metrics = Arc::new(SpecdecMetrics::default());
        let engine = Self {
            model_name: args.model_name.clone(),
            context_length: args.context_length,
            draft_endpoint: EndpointId::from(args.draft_endpoint.as_str()),
            metrics: metrics.clone(),
            clients: Arc::new(ClientPool::new_with_metrics(
                DraftClientConfig {
                    transport_hwm: args.transport_hwm,
                    outbound_capacity: args.transport_queue_capacity,
                    session_capacity: args.session_queue_capacity,
                    max_sessions: args.max_inflight_sessions,
                    handshake_timeout: Duration::from_millis(args.handshake_timeout_ms),
                    start_timeout: Duration::from_millis(args.start_timeout_ms),
                    inactivity_timeout: Duration::from_millis(args.inactivity_timeout_ms),
                    cleanup_timeout: Duration::from_millis(args.cleanup_timeout_ms),
                },
                args.draft_connection_pool_size,
                metrics,
            )),
            target_prefill: Duration::from_millis(args.target_prefill_ms),
            target_token_interval: Duration::from_millis(args.target_token_interval_ms),
            accepted_proposal_tokens: args.accepted_proposal_tokens,
            fail_after_draft_start_prompt_token: args.fail_after_draft_start_prompt_token,
        };
        let config = worker_config(args.common, args.model_name, args.model_path);
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for TargetEngine {
    async fn start(&self, worker_id: u64) -> Result<EngineConfig, DynamoError> {
        tracing::info!(
            worker_id,
            dp_rank = DP_RANK,
            model = %self.model_name,
            "mock speculative target started"
        );
        Ok(EngineConfig {
            model: self.model_name.clone(),
            served_model_name: Some(self.model_name.clone()),
            llm: Some(LlmRegistration {
                context_length: Some(self.context_length),
                data_parallel_size: Some(1),
                data_parallel_start_rank: Some(DP_RANK),
                ..LlmRegistration::default()
            }),
            ..EngineConfig::default()
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let hint = speculative_hint(&request)?;
        if hint.draft_endpoint != self.draft_endpoint || hint.transport.protocol != PROTOCOL {
            return Err(backend_error(
                BackendError::InvalidArgument,
                "speculative router hint does not match the target binding",
            ));
        }
        let expected = DraftIdentity {
            endpoint: hint.draft_endpoint,
            worker: hint.draft,
            draft_incarnation_id: hint.draft_incarnation_id,
            protocol: hint.transport.protocol,
            address: hint.transport.address,
            orphan_cleanup_timeout_ms: hint.transport.orphan_cleanup_timeout_ms,
        };
        expected.validate().map_err(|_| {
            backend_error(
                BackendError::InvalidArgument,
                "speculative router hint has an invalid draft identity",
            )
        })?;
        let draft_worker_id = expected.worker.worker_id;
        let draft_dp_rank = expected.worker.dp_rank;
        let draft_incarnation = expected.draft_incarnation_id;
        let requested_tokens = request
            .stop_conditions
            .max_tokens
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
        if requested_tokens > MAX_OUTPUT_TOKENS {
            return Err(backend_error(
                BackendError::InvalidArgument,
                format!(
                    "requested max_tokens {requested_tokens} exceeds mock protocol limit {MAX_OUTPUT_TOKENS}"
                ),
            ));
        }
        let prompt_length = request.token_ids.len() as u32;
        if requested_tokens == 0 {
            return Ok(Box::pin(async_stream::stream! {
                let terminal = if ctx.is_stopped() {
                    LLMEngineOutput::cancelled()
                } else {
                    LLMEngineOutput::length()
                };
                yield Ok(terminal.with_usage(usage(prompt_length, 0)));
            }));
        }
        let request_id = request_uuid(ctx.id());
        let fail_after_draft_start = self
            .fail_after_draft_start_prompt_token
            .is_some_and(|token| request.token_ids.contains(&token));
        let start = Start {
            prompt_token_ids: request.token_ids,
            max_output_tokens: requested_tokens,
        };
        let clients = self.clients.clone();
        let target_prefill = self.target_prefill;
        let target_token_interval = self.target_token_interval;
        let accepted_proposal_tokens = self.accepted_proposal_tokens;
        let orphan_cleanup_timeout =
            Duration::from_millis(u64::from(expected.orphan_cleanup_timeout_ms));

        Ok(Box::pin(async_stream::stream! {
            let setup = async {
                let target_prefill = async {
                    tokio::time::sleep(target_prefill).await;
                    tokio::time::Instant::now()
                };
                let draft_exchange = async {
                    let (lease, mut session) = match clients
                        .start(expected, request_id, start)
                        .await
                    {
                        Ok(started) => started,
                        Err(error) => {
                            let cleanup = if error.state == FailureState::NotStarted {
                                None
                            } else {
                                tokio::time::sleep(orphan_cleanup_timeout).await;
                                Some(DraftCleanupOutcomeV1::CleanupBoundElapsed)
                            };
                            if ctx.is_stopped() {
                                return DraftExchange::Cancelled {
                                    started_at: None,
                                    cleanup,
                                };
                            }
                            return DraftExchange::Failure {
                                started_at: None,
                                phase: DraftFailurePhase::Start,
                                error,
                                cleanup,
                            };
                        }
                    };
                    let started_at = tokio::time::Instant::now();
                    if ctx.is_stopped() {
                        let cleanup = resolve_cleanup(
                            session,
                            lease,
                            orphan_cleanup_timeout,
                        )
                        .await;
                        return DraftExchange::Cancelled {
                            started_at: Some(started_at),
                            cleanup: Some(cleanup.outcome),
                        };
                    }
                    if fail_after_draft_start {
                        let cleanup = resolve_cleanup(
                            session,
                            lease,
                            orphan_cleanup_timeout,
                        )
                        .await;
                        return DraftExchange::InjectedFailure {
                            started_at,
                            cleanup,
                        };
                    }
                    let proposal = tokio::select! {
                        biased;
                        _ = ctx.stopped() => {
                            let cleanup = resolve_cleanup(
                                session,
                                lease,
                                orphan_cleanup_timeout,
                            )
                            .await;
                            return DraftExchange::Cancelled {
                                started_at: Some(started_at),
                                cleanup: Some(cleanup.outcome),
                            };
                        }
                        proposal = session.collect() => proposal,
                    };
                    let proposal = match proposal {
                        Ok(proposal) => proposal,
                        Err(error) => {
                            let cleanup = resolve_cleanup(
                                session,
                                lease,
                                orphan_cleanup_timeout,
                            )
                            .await;
                            return DraftExchange::Failure {
                                started_at: Some(started_at),
                                phase: DraftFailurePhase::Proposal,
                                error,
                                cleanup: Some(cleanup.outcome),
                            };
                        }
                    };
                    let cleanup = resolve_cleanup(
                        session,
                        lease,
                        orphan_cleanup_timeout,
                    )
                    .await;
                    if let Some(error) = cleanup.error {
                        return DraftExchange::Failure {
                            started_at: Some(started_at),
                            phase: DraftFailurePhase::Cleanup,
                            error,
                            cleanup: Some(cleanup.outcome),
                        };
                    }
                    DraftExchange::Proposal {
                        started_at,
                        proposal,
                    }
                };
                let (target_prefill, draft_exchange) = tokio::join!(
                    target_prefill,
                    draft_exchange,
                );
                (target_prefill, draft_exchange)
            };
            let (prefill_completed_at, draft_exchange) = setup.await;
            let draft_started_at = match &draft_exchange {
                DraftExchange::Proposal { started_at, .. }
                | DraftExchange::InjectedFailure { started_at, .. } => Some(*started_at),
                DraftExchange::Cancelled { started_at, .. }
                | DraftExchange::Failure { started_at, .. } => *started_at,
            };
            if draft_started_at.is_some_and(|started_at| started_at <= prefill_completed_at) {
                tracing::info!(
                    worker_id = draft_worker_id,
                    dp_rank = draft_dp_rank,
                    draft_incarnation,
                    %request_id,
                    "mock speculative draft START preceded target prefill completion"
                );
            } else if draft_started_at.is_some() {
                tracing::warn!(
                    worker_id = draft_worker_id,
                    dp_rank = draft_dp_rank,
                    draft_incarnation,
                    %request_id,
                    "mock speculative draft START followed target prefill completion"
                );
            }
            let proposal = match draft_exchange {
                DraftExchange::Cancelled { cleanup, .. } => {
                    tracing::info!(
                        worker_id = draft_worker_id,
                        dp_rank = draft_dp_rank,
                        draft_incarnation,
                        %request_id,
                        ?cleanup,
                        "mock speculative target cancelled during setup"
                    );
                    let terminal = LLMEngineOutput::cancelled()
                        .with_usage(usage(prompt_length, 0));
                    let terminal = match cleanup {
                        Some(outcome) => with_cleanup_lifecycle(terminal, outcome),
                        None => terminal,
                    };
                    yield Ok(terminal);
                    return;
                }
                DraftExchange::Failure {
                    phase,
                    error,
                    cleanup,
                    ..
                } => {
                    let message = match phase {
                        DraftFailurePhase::Start => "mock speculative draft START failed",
                        DraftFailurePhase::Proposal => "mock speculative draft proposal failed",
                        DraftFailurePhase::Cleanup => "mock speculative draft cleanup failed",
                    };
                    tracing::warn!(
                        worker_id = draft_worker_id,
                        dp_rank = draft_dp_rank,
                        draft_incarnation,
                        %request_id,
                        %error,
                        ?cleanup,
                        phase = message,
                        "mock speculative draft exchange failed"
                    );
                    if let Some(outcome) = cleanup {
                        let terminal = LLMEngineOutput::error(format!("{message}: {error}"))
                            .with_usage(usage(prompt_length, 0));
                        yield Ok(with_cleanup_lifecycle(terminal, outcome));
                    } else {
                        yield Err(transport_dynamo_error(error));
                    }
                    return;
                }
                DraftExchange::InjectedFailure { cleanup, .. } => {
                    match &cleanup.error {
                        None => tracing::warn!(
                            worker_id = draft_worker_id,
                            dp_rank = draft_dp_rank,
                            draft_incarnation,
                            %request_id,
                            draft_cleanup = "acknowledged",
                            "mock speculative target injected failure after draft START"
                        ),
                        Some(error) => tracing::warn!(
                            worker_id = draft_worker_id,
                            dp_rank = draft_dp_rank,
                            draft_incarnation,
                            %request_id,
                            %error,
                            draft_cleanup = "cleanup_bound_elapsed",
                            "mock speculative target injected failure after draft START"
                        ),
                    }
                    let terminal = LLMEngineOutput::error(
                        "injected mock target failure after draft START".to_string(),
                    )
                    .with_usage(usage(prompt_length, 0));
                    yield Ok(with_cleanup_lifecycle(terminal, cleanup.outcome));
                    return;
                }
                DraftExchange::Proposal { proposal, .. } => proposal,
            };
            let accepted = proposal
                .token_ids
                .len()
                .min(accepted_proposal_tokens as usize);
            tracing::info!(
                worker_id = draft_worker_id,
                dp_rank = draft_dp_rank,
                draft_incarnation,
                %request_id,
                proposal_digest = %proposal.proposal_digest,
                proposal_tokens = proposal.token_ids.len(),
                accepted_tokens = accepted,
                draft_cleanup = "acknowledged",
                "mock speculative target consumed draft proposal"
            );
            let mut emitted = 0_u32;
            for token_id in proposal.token_ids.into_iter().take(accepted) {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        let terminal = LLMEngineOutput::cancelled().with_usage(usage(
                            prompt_length, emitted,
                        ));
                        yield Ok(with_cleanup_lifecycle(
                            terminal,
                            DraftCleanupOutcomeV1::Acknowledged,
                        ));
                        return;
                    }
                    _ = tokio::time::sleep(target_token_interval) => {}
                }
                yield Ok(chunk::token(token_id));
                emitted += 1;
            }

            if ctx.is_stopped() {
                let terminal = LLMEngineOutput::cancelled().with_usage(usage(
                    prompt_length, emitted,
                ));
                yield Ok(with_cleanup_lifecycle(
                    terminal,
                    DraftCleanupOutcomeV1::Acknowledged,
                ));
                return;
            }

            let mut engine_data = serde_json::Map::new();
            engine_data.insert(
                "mock_specdec".to_string(),
                serde_json::json!({
                    "request_id": request_id,
                    "proposal_digest": proposal.proposal_digest,
                }),
            );
            let mut terminal = LLMEngineOutput::length().with_usage(usage(
                prompt_length,
                emitted,
            ));
            terminal.engine_data = Some(serde_json::Value::Object(engine_data));
            yield Ok(with_cleanup_lifecycle(
                terminal,
                DraftCleanupOutcomeV1::Acknowledged,
            ));
        }))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.clients
            .shutdown()
            .await
            .map_err(transport_dynamo_error)
    }

    async fn setup_metrics(&self, ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        self.metrics.register(ctx);
        Ok(MetricsBindings::default())
    }

    async fn model_registration(
        &self,
        _endpoint: &Endpoint,
    ) -> Result<ModelRegistration, DynamoError> {
        Ok(ModelRegistration {
            worker_role: WorkerRole::SpeculativeTarget(ExternalDraftBinding {
                endpoint: self.draft_endpoint.clone(),
                protocol: PROTOCOL.to_string(),
                router_hint_schema_version: ExternalDraftBinding::ROUTER_HINT_SCHEMA_VERSION,
            }),
            external_draft_transports: Default::default(),
        })
    }
}

fn speculative_hint(
    request: &PreprocessedRequest,
) -> Result<SpeculativeDecodingRouterHintV1, DynamoError> {
    let value = request
        .extra_args
        .as_ref()
        .and_then(|extra| extra.get(KV_TRANSFER_PARAMS_KEY))
        .and_then(|params| params.get(ROUTER_HINT_KEY))
        .cloned()
        .ok_or_else(|| {
            backend_error(
                BackendError::InvalidArgument,
                "speculative target request is missing the authoritative router hint",
            )
        })?;
    let envelope: RouterHintEnvelope = serde_json::from_value(value).map_err(|_| {
        backend_error(
            BackendError::InvalidArgument,
            "speculative target request has an invalid router hint",
        )
    })?;
    envelope.speculative_decoding.ok_or_else(|| {
        backend_error(
            BackendError::InvalidArgument,
            "speculative target request has no speculative router selection",
        )
    })
}

fn request_uuid(request_id: &str) -> Uuid {
    Uuid::parse_str(request_id).unwrap_or_else(|_| {
        let digest = blake3::hash(request_id.as_bytes());
        let mut bytes = [0_u8; 16];
        bytes.copy_from_slice(&digest.as_bytes()[..16]);
        Uuid::from_bytes(bytes)
    })
}

fn transport_dynamo_error(error: TransportError) -> DynamoError {
    let kind = match error.kind {
        TransportErrorKind::Configuration
        | TransportErrorKind::Protocol
        | TransportErrorKind::Identity => BackendError::InvalidArgument,
        TransportErrorKind::Connect => BackendError::CannotConnect,
        TransportErrorKind::Timeout => BackendError::ResponseTimeout,
        TransportErrorKind::Closed => BackendError::Disconnected,
        TransportErrorKind::Backpressure | TransportErrorKind::Queue => BackendError::Unknown,
        TransportErrorKind::Task => BackendError::EngineShutdown,
    };
    backend_error(kind, format!("mock draft transport: {error}"))
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use dynamo_backend_common::{
        FinishReason, OutputOptions, SamplingOptions, StopConditions, WorkerWithDpRank,
    };
    use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
    use futures::StreamExt;

    use super::*;
    use crate::specdec::queue::{SchedulerConfig, TokenMode};
    use crate::specdec::transport::{DraftServer, DraftServerConfig};

    fn unused_address() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
        let address = listener.local_addr().expect("read loopback port");
        drop(listener);
        format!("tcp://{address}")
    }

    fn target_request(max_tokens: u32, identity: &DraftIdentity) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock-specdec-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions {
                max_tokens: Some(max_tokens),
                ..StopConditions::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .extra_args(Some(serde_json::json!({
                "kv_transfer_params": {
                    "router_hint": {
                        "speculative_decoding": {
                            "schema_version": 1,
                            "draft_endpoint": identity.endpoint,
                            "draft": identity.worker,
                            "draft_incarnation_id": identity.draft_incarnation_id,
                            "transport": {
                                "protocol": identity.protocol,
                                "address": identity.address,
                                "orphan_cleanup_timeout_ms": identity.orphan_cleanup_timeout_ms,
                            }
                        }
                    }
                }
            })))
            .build()
            .unwrap()
    }

    fn target_for(identity: &DraftIdentity) -> TargetEngine {
        let metrics = Arc::new(SpecdecMetrics::default());
        TargetEngine {
            model_name: "mock-specdec-model".to_string(),
            context_length: MAX_OUTPUT_TOKENS + 10,
            draft_endpoint: identity.endpoint.clone(),
            clients: Arc::new(ClientPool::new_with_metrics(
                DraftClientConfig {
                    cleanup_timeout: Duration::from_secs(2),
                    inactivity_timeout: Duration::from_secs(30),
                    session_capacity: (MAX_OUTPUT_TOKENS as usize) + 2,
                    transport_hwm: (MAX_OUTPUT_TOKENS as i32) + 2,
                    ..DraftClientConfig::default()
                },
                1,
                metrics.clone(),
            )),
            metrics,
            target_prefill: Duration::ZERO,
            target_token_interval: Duration::ZERO,
            accepted_proposal_tokens: MAX_OUTPUT_TOKENS,
            fail_after_draft_start_prompt_token: None,
        }
    }

    fn generate_context(context: &Context<()>) -> GenerateContext {
        GenerateContext::new(context.context(), None)
    }

    fn cleanup_outcome(output: &LLMEngineOutput) -> Option<DraftCleanupOutcomeV1> {
        let lifecycle = output
            .engine_data
            .as_ref()?
            .get(EXTERNAL_SPECULATION_LIFECYCLE_ENGINE_DATA_KEY)?
            .clone();
        serde_json::from_value::<ExternalSpeculationLifecycleV1>(lifecycle)
            .ok()
            .map(|lifecycle| lifecycle.draft_cleanup)
    }

    #[tokio::test]
    async fn target_rejects_above_limit_and_defines_limit_and_zero_behavior() {
        let address = unused_address();
        let identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 500,
        };
        let server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::ZERO,
                token_interval: Duration::ZERO,
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            identity.clone(),
        )
        .await
        .unwrap();
        let target = target_for(&identity);
        target.start(19).await.unwrap();

        let context = Context::new(());
        let above_limit = target
            .generate(
                target_request(MAX_OUTPUT_TOKENS + 1, &identity),
                generate_context(&context),
            )
            .await;
        let Err(above_limit) = above_limit else {
            panic!("above-limit request must be rejected before START");
        };
        assert_eq!(
            above_limit.error_type(),
            dynamo_backend_common::ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert_eq!(server.metrics_snapshot().starts_accepted, 0);

        let mut zero = target
            .generate(target_request(0, &identity), generate_context(&context))
            .await
            .unwrap();
        let zero_terminal = zero.next().await.unwrap().unwrap();
        assert!(matches!(
            zero_terminal.finish_reason,
            Some(FinishReason::Length)
        ));
        assert!(zero.next().await.is_none());
        assert_eq!(server.metrics_snapshot().starts_accepted, 0);

        let cancelled_zero_context = Context::new(());
        let cancelled_zero_control = cancelled_zero_context.context();
        let mut cancelled_zero = target
            .generate(
                target_request(0, &identity),
                GenerateContext::new(cancelled_zero_control.clone(), None),
            )
            .await
            .unwrap();
        cancelled_zero_control.stop_generating();
        let cancelled_zero_terminal = cancelled_zero.next().await.unwrap().unwrap();
        assert!(matches!(
            cancelled_zero_terminal.finish_reason,
            Some(FinishReason::Cancelled)
        ));
        assert!(cancelled_zero.next().await.is_none());
        assert_eq!(server.metrics_snapshot().starts_accepted, 0);

        let limit_context = Context::new(());
        let limit_control = limit_context.context();
        let mut at_limit = target
            .generate(
                target_request(MAX_OUTPUT_TOKENS, &identity),
                GenerateContext::new(limit_control.clone(), None),
            )
            .await
            .unwrap();
        let first_output = tokio::spawn(async move { at_limit.next().await });
        tokio::time::timeout(Duration::from_secs(2), async {
            while server.metrics_snapshot().starts_accepted != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("at-limit request did not reach draft START");
        limit_control.stop_generating();
        let limit_terminal = first_output.await.unwrap().unwrap().unwrap();
        assert!(matches!(
            limit_terminal.finish_reason,
            Some(FinishReason::Cancelled)
        ));
        assert_eq!(
            cleanup_outcome(&limit_terminal),
            Some(DraftCleanupOutcomeV1::Acknowledged)
        );

        target.cleanup().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn post_start_cancellation_waits_for_cleanup_proof() {
        let address = unused_address();
        let identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 500,
        };
        let server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::from_secs(10),
                token_interval: Duration::ZERO,
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            identity.clone(),
        )
        .await
        .unwrap();
        let target = target_for(&identity);
        target.start(19).await.unwrap();
        let context = Context::new(());
        let control = context.context();
        let mut stream = target
            .generate(
                target_request(8, &identity),
                GenerateContext::new(control.clone(), None),
            )
            .await
            .unwrap();
        let terminal = tokio::spawn(async move { stream.next().await });
        tokio::time::timeout(Duration::from_secs(2), async {
            while server.metrics_snapshot().starts_accepted != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("request did not reach draft START_ACK");

        control.stop_generating();
        let terminal = tokio::time::timeout(Duration::from_secs(2), terminal)
            .await
            .expect("cancelled request did not finish after cleanup")
            .unwrap()
            .unwrap()
            .unwrap();

        assert!(matches!(
            terminal.finish_reason,
            Some(FinishReason::Cancelled)
        ));
        assert_eq!(
            cleanup_outcome(&terminal),
            Some(DraftCleanupOutcomeV1::Acknowledged)
        );
        assert_eq!(server.active_sessions(), 0);

        target.cleanup().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn injected_post_start_error_carries_acknowledged_cleanup() {
        let address = unused_address();
        let identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 500,
        };
        let server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::ZERO,
                token_interval: Duration::ZERO,
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            identity.clone(),
        )
        .await
        .unwrap();
        let mut target = target_for(&identity);
        target.fail_after_draft_start_prompt_token = Some(3);
        target.start(19).await.unwrap();
        let context = Context::new(());
        let mut stream = target
            .generate(target_request(8, &identity), generate_context(&context))
            .await
            .unwrap();

        let terminal = stream.next().await.unwrap().unwrap();

        assert!(matches!(
            terminal.finish_reason,
            Some(FinishReason::Error(_))
        ));
        assert_eq!(
            cleanup_outcome(&terminal),
            Some(DraftCleanupOutcomeV1::Acknowledged)
        );
        assert!(stream.next().await.is_none());
        assert_eq!(server.active_sessions(), 0);

        target.cleanup().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn cancellation_after_final_token_yields_cancelled_not_length() {
        let address = unused_address();
        let identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 500,
        };
        let server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::ZERO,
                token_interval: Duration::ZERO,
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            identity.clone(),
        )
        .await
        .unwrap();
        let target = target_for(&identity);
        target.start(19).await.unwrap();
        let context = Context::new(());
        let control = context.context();
        let mut stream = target
            .generate(
                target_request(1, &identity),
                GenerateContext::new(control.clone(), None),
            )
            .await
            .unwrap();

        let token = stream.next().await.unwrap().unwrap();
        assert_eq!(token.token_ids.len(), 1);
        control.stop_generating();
        let terminal = stream.next().await.unwrap().unwrap();
        assert!(matches!(
            terminal.finish_reason,
            Some(FinishReason::Cancelled)
        ));
        assert!(stream.next().await.is_none());

        target.cleanup().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn acknowledged_cleanup_recycles_the_connection() {
        let address = unused_address();
        let identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 500,
        };
        let server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::from_millis(1),
                token_interval: Duration::from_millis(1),
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            identity.clone(),
        )
        .await
        .unwrap();
        let pool = Arc::new(ClientPool::new(DraftClientConfig::default(), 8));
        let (first_lease, mut first_session) = pool
            .start(
                identity.clone(),
                Uuid::from_u128(701),
                Start {
                    prompt_token_ids: vec![1],
                    max_output_tokens: 1,
                },
            )
            .await
            .unwrap();
        let first_connection = first_lease.client().clone();
        first_session.collect().await.unwrap();
        first_session.cleanup().await.unwrap();
        first_lease.recycle().await;

        let (second_lease, mut second_session) = pool
            .start(
                identity,
                Uuid::from_u128(702),
                Start {
                    prompt_token_ids: vec![2],
                    max_output_tokens: 1,
                },
            )
            .await
            .unwrap();
        assert!(first_connection.same_connection(second_lease.client()));
        second_session.collect().await.unwrap();
        second_session.cleanup().await.unwrap();
        second_lease.recycle().await;

        let metrics = server.metrics_snapshot();
        assert_eq!(metrics.starts_accepted, 2);
        assert_eq!(metrics.starts_rejected, 0);
        assert_eq!(metrics.proposals, 2);
        assert_eq!(metrics.completions, 2);
        assert_eq!(metrics.cleanup_acknowledgements, 2);
        assert_eq!(metrics.active_sessions, 0);
        assert_eq!(metrics.queue_depth, 0);

        pool.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn replacement_incarnation_rejects_stale_identity_and_replaces_the_pool() {
        let address = unused_address();
        let old_identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 500,
        };
        let old_server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address.clone(),
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::from_millis(1),
                token_interval: Duration::from_millis(1),
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            old_identity.clone(),
        )
        .await
        .unwrap();
        let pool = Arc::new(ClientPool::new(DraftClientConfig::default(), 8));
        let old_request = Uuid::from_u128(801);
        let (old_lease, mut old_session) = pool
            .start(
                old_identity.clone(),
                old_request,
                Start {
                    prompt_token_ids: vec![1],
                    max_output_tokens: 1,
                },
            )
            .await
            .unwrap();
        let old_connection = old_lease.client().clone();
        old_session.collect().await.unwrap();
        old_session.cleanup().await.unwrap();
        old_lease.recycle().await;
        old_server.shutdown().await.unwrap();

        let replacement_identity = DraftIdentity {
            draft_incarnation_id: 24,
            ..old_identity.clone()
        };
        let replacement_server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::from_millis(1),
                token_interval: Duration::from_millis(1),
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            replacement_identity.clone(),
        )
        .await
        .unwrap();
        let replacement_request = Uuid::from_u128(802);
        let (replacement_lease, mut replacement_session) = pool
            .start(
                replacement_identity.clone(),
                replacement_request,
                Start {
                    prompt_token_ids: vec![2],
                    max_output_tokens: 1,
                },
            )
            .await
            .unwrap();
        assert!(!old_connection.same_connection(replacement_lease.client()));
        replacement_session.collect().await.unwrap();
        replacement_session.cleanup().await.unwrap();
        replacement_lease.recycle().await;

        let stale = match DraftClient::connect(old_identity, DraftClientConfig::default()).await {
            Ok(_) => panic!("stale draft identity unexpectedly connected"),
            Err(error) => error,
        };
        assert_eq!(stale.kind, TransportErrorKind::Identity);

        pool.shutdown().await.unwrap();
        replacement_server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn cancelled_explicit_finalization_remains_pool_owned() {
        let address = unused_address();
        let identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 100,
        };
        let server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::from_millis(1),
                token_interval: Duration::from_millis(1),
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            identity.clone(),
        )
        .await
        .unwrap();
        let pool = Arc::new(ClientPool::new(DraftClientConfig::default(), 1));

        let (recycle_lease, mut recycle_session) = pool
            .start(
                identity.clone(),
                Uuid::from_u128(851),
                Start {
                    prompt_token_ids: vec![1],
                    max_output_tokens: 1,
                },
            )
            .await
            .unwrap();
        recycle_session.collect().await.unwrap();
        recycle_session.cleanup().await.unwrap();
        let state = pool.state.lock().await;
        let mut recycling = Box::pin(recycle_lease.recycle());
        assert!(futures::poll!(recycling.as_mut()).is_pending());
        drop(recycling);
        assert_eq!(pool.active_operations.load(Ordering::Acquire), 1);
        assert_eq!(pool.connection_permits.available_permits(), 0);
        drop(state);
        tokio::time::timeout(Duration::from_secs(1), async {
            while pool.active_operations.load(Ordering::Acquire) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let state = pool.state.lock().await;
        assert!(state.active.is_empty());
        assert_eq!(state.idle.len(), 1);
        drop(state);
        assert_eq!(pool.connection_permits.available_permits(), 1);

        let (quarantine_lease, quarantine_session) = pool
            .start(
                identity,
                Uuid::from_u128(852),
                Start {
                    prompt_token_ids: vec![2],
                    max_output_tokens: 1,
                },
            )
            .await
            .unwrap();
        let quarantine_client = quarantine_lease.client().clone();
        drop(quarantine_session);
        let state = pool.state.lock().await;
        let mut quarantining = Box::pin(quarantine_lease.quarantine());
        assert!(futures::poll!(quarantining.as_mut()).is_pending());
        drop(quarantining);
        assert!(quarantine_client.is_closed());
        assert_eq!(pool.active_operations.load(Ordering::Acquire), 1);
        assert_eq!(pool.connection_permits.available_permits(), 0);
        drop(state);
        tokio::time::timeout(Duration::from_secs(1), async {
            while pool.active_operations.load(Ordering::Acquire) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let state = pool.state.lock().await;
        assert!(state.active.is_empty());
        assert!(state.idle.is_empty());
        drop(state);
        assert_eq!(pool.connection_permits.available_permits(), 1);

        tokio::time::timeout(Duration::from_secs(1), async {
            while server.active_sessions() != 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        pool.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_owns_active_clients_and_rejects_connection_waiters() {
        let address = unused_address();
        let identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 100,
        };
        let server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::from_secs(10),
                token_interval: Duration::from_millis(1),
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            identity.clone(),
        )
        .await
        .unwrap();
        let pool = Arc::new(ClientPool::new(
            DraftClientConfig {
                start_timeout: Duration::from_secs(5),
                ..DraftClientConfig::default()
            },
            1,
        ));
        let (active_lease, active_session) = pool
            .start(
                identity.clone(),
                Uuid::from_u128(901),
                Start {
                    prompt_token_ids: vec![1],
                    max_output_tokens: 1,
                },
            )
            .await
            .unwrap();
        let active_client = active_lease.client().clone();
        assert_eq!(server.active_sessions(), 1);

        let waiting_pool = pool.clone();
        let waiting_identity = identity.clone();
        let waiter = tokio::spawn(async move {
            waiting_pool
                .start(
                    waiting_identity,
                    Uuid::from_u128(902),
                    Start {
                        prompt_token_ids: vec![2],
                        max_output_tokens: 1,
                    },
                )
                .await
        });
        tokio::task::yield_now().await;
        let shutdown_pool = pool.clone();
        let shutdown = tokio::spawn(async move { shutdown_pool.shutdown().await });

        tokio::time::timeout(Duration::from_secs(1), async {
            while !active_client.is_closed() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let waiter_error = match waiter.await.unwrap() {
            Ok(_) => panic!("connection waiter unexpectedly started after pool shutdown"),
            Err(error) => error,
        };
        assert_eq!(waiter_error.kind, TransportErrorKind::Closed);

        drop(active_session);
        drop(active_lease);
        tokio::time::timeout(Duration::from_secs(2), shutdown)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let post_shutdown_error = match pool
            .start(
                identity,
                Uuid::from_u128(903),
                Start {
                    prompt_token_ids: vec![3],
                    max_output_tokens: 1,
                },
            )
            .await
        {
            Ok(_) => panic!("shutdown pool unexpectedly accepted a new request"),
            Err(error) => error,
        };
        assert_eq!(post_shutdown_error.kind, TransportErrorKind::Closed);
        let state = pool.state.lock().await;
        assert!(state.idle.is_empty());
        assert!(state.active.is_empty());
        drop(state);

        tokio::time::timeout(Duration::from_secs(2), async {
            while server.active_sessions() != 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let metrics = server.metrics_snapshot();
        assert_eq!(metrics.orphaned_sessions, 0);
        assert!(metrics.orphaned_sessions_reaped >= 1);
        server.shutdown().await.unwrap();
    }
}
