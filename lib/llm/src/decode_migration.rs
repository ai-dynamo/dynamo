// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-local decode-to-decode migration.
//!
//! Destination discovery and reservation are fail-open while the source is still
//! decoding. Once source quiescence starts, the first-pass protocol is deliberately
//! one-way: failures abort the destination, cancel the detached source request, and
//! terminate the client stream with an error rather than attempting rollback.

use std::{collections::HashSet, sync::Arc, time::Duration};

use anyhow::{Context as AnyhowContext, Result, anyhow, bail};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerId, WorkerWithDpRank};
#[cfg(test)]
use dynamo_runtime::pipeline::ServiceEngine;
use dynamo_runtime::{
    component::{Client, Component},
    engine::AsyncEngineContextProvider,
    logging::get_distributed_tracing_context,
    pipeline::{
        Context, ManyOut, Operator, PipelineOperator, PushRouter, ResponseStream,
        ServerStreamingEngine, SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    kv_router::{FindBestMatchOutcome, KvRouter},
    protocols::{
        TokenIdType,
        common::{
            llm_backend::LLMEngineOutput,
            preprocessor::{
                BootstrapInfo, DecodeMigrationConstraints, DecodeMigrationPolicy,
                DecodeMigrationRequestState, DecodeMigrationTrigger, PreprocessedRequest,
            },
        },
    },
};

#[cfg(test)]
type BackendEngine =
    ServiceEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>>;
type DirectBackend = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;
type ControlClient = PushRouter<Value, Annotated<Value>>;

const CONTROL_TIMEOUT: Duration = Duration::from_secs(10);
const QUIESCE_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, Copy)]
enum ControlEndpoint {
    Prepare,
    Sync,
    Finalize,
}

#[async_trait]
trait MigrationBackend: Send + Sync {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        worker: WorkerWithDpRank,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>>;
}

struct DirectMigrationBackend {
    router: DirectBackend,
}

#[async_trait]
impl MigrationBackend for DirectMigrationBackend {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        worker: WorkerWithDpRank,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        self.router
            .direct(request, worker.worker_id)
            .await
            .map_err(Into::into)
    }
}

#[cfg(test)]
struct EngineMigrationBackend {
    engine: BackendEngine,
}

#[cfg(test)]
#[async_trait]
impl MigrationBackend for EngineMigrationBackend {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        _worker: WorkerWithDpRank,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        self.engine.generate(request).await
    }
}

#[async_trait]
trait MigrationControl: Send + Sync {
    async fn call(
        &self,
        endpoint: ControlEndpoint,
        instance_id: u64,
        request: Value,
    ) -> Result<ControlResponse>;
}

#[async_trait]
trait MigrationSelector: Send + Sync {
    async fn select_source(
        &self,
        request: &PreprocessedRequest,
        constraints: &DecodeMigrationConstraints,
    ) -> Result<WorkerWithDpRank>;

    async fn select_destination(
        &self,
        request: &PreprocessedRequest,
        tokens: &[TokenIdType],
        constraints: &DecodeMigrationConstraints,
        source: WorkerWithDpRank,
    ) -> Result<WorkerWithDpRank>;
}

#[derive(Clone)]
struct ControlClients {
    prepare: ControlClient,
    sync: ControlClient,
    finalize: ControlClient,
}

impl ControlClients {
    async fn new(component: Component) -> Result<Self> {
        Ok(Self {
            prepare: control_client(&component, "migration_prepare").await?,
            sync: control_client(&component, "migration_sync").await?,
            finalize: control_client(&component, "migration_finalize").await?,
        })
    }
}

#[async_trait]
impl MigrationControl for ControlClients {
    async fn call(
        &self,
        endpoint: ControlEndpoint,
        instance_id: u64,
        request: Value,
    ) -> Result<ControlResponse> {
        let client = match endpoint {
            ControlEndpoint::Prepare => &self.prepare,
            ControlEndpoint::Sync => &self.sync,
            ControlEndpoint::Finalize => &self.finalize,
        };
        call_control_client(client, instance_id, request).await
    }
}

async fn control_client(component: &Component, endpoint: &str) -> Result<ControlClient> {
    let client = component.endpoint(endpoint).client().await?;
    PushRouter::from_client(client, Default::default()).await
}

async fn call_control_client(
    client: &ControlClient,
    instance_id: u64,
    request: Value,
) -> Result<ControlResponse> {
    let mut stream = tokio::time::timeout(
        CONTROL_TIMEOUT,
        client.direct(SingleIn::new(request), instance_id),
    )
    .await
    .context("migration control request timed out")??;
    let first = tokio::time::timeout(CONTROL_TIMEOUT, stream.next())
        .await
        .context("migration control response timed out")?
        .context("migration control endpoint returned an empty stream")?;
    let first = first.ok().map_err(|error| anyhow!(error))?;
    let data = first
        .data
        .context("migration control response did not contain data")?;
    while stream.next().await.is_some() {}
    serde_json::from_value(data).context("invalid migration control response")
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ControlResponse {
    #[serde(default)]
    success: bool,
    #[serde(default)]
    status: String,
    error: Option<String>,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
    bootstrap_room: Option<u64>,
    source_dp_rank: Option<u32>,
    destination_dp_rank: Option<u32>,
    prompt_len: Option<usize>,
    committed_len: Option<usize>,
    logical_len: Option<usize>,
    unforwarded_committed_output_ids: Option<Vec<TokenIdType>>,
}

struct PreparedDestination {
    worker: WorkerWithDpRank,
    source_host: String,
    source_port: u16,
    source_dp_rank: u32,
    bootstrap_room: u64,
    cleanup: MigrationCleanup,
}

async fn prepare_destination(
    selector: Arc<dyn MigrationSelector>,
    control: Arc<dyn MigrationControl>,
    request: PreprocessedRequest,
    constraints: DecodeMigrationConstraints,
    source_worker: WorkerWithDpRank,
    rid: String,
    migration_id: String,
    source_trigger: DecodeMigrationTrigger,
) -> Result<PreparedDestination> {
    let setup_started = std::time::Instant::now();
    let destination_worker = selector
        .select_destination(&request, &request.token_ids, &constraints, source_worker)
        .await?;
    tracing::info!(
        request_id = %rid,
        %migration_id,
        source_worker = source_worker.worker_id,
        destination_worker = destination_worker.worker_id,
        destination_dp_rank = destination_worker.dp_rank,
        "decode migration destination selected"
    );

    let source_description = control
        .call(
            ControlEndpoint::Sync,
            source_worker.worker_id,
            json!({
                "rid": rid,
                "migration_id": migration_id,
                "phase": "describe",
                "source_dp_rank": source_worker.dp_rank,
            }),
        )
        .await
        .context("source describe failed")?;
    if !source_description.success {
        bail!(
            "source describe declined with status {}: {}",
            source_description.status,
            source_description
                .error
                .as_deref()
                .unwrap_or("unknown error")
        );
    }
    let source_host = source_description
        .bootstrap_host
        .context("source describe omitted bootstrap_host")?;
    let source_port = source_description
        .bootstrap_port
        .context("source describe omitted bootstrap_port")?;
    let source_dp_rank = source_description
        .source_dp_rank
        .unwrap_or(source_worker.dp_rank);
    if source_dp_rank != source_worker.dp_rank {
        bail!(
            "source describe returned DP rank {source_dp_rank}, expected {}",
            source_worker.dp_rank
        );
    }

    let mut cleanup = MigrationCleanup::new(
        control.clone(),
        rid.clone(),
        migration_id.clone(),
        source_worker.worker_id,
        destination_worker.worker_id,
        source_dp_rank,
        destination_worker.dp_rank,
    );
    let reserved_logical_len = match request.stop_conditions.max_tokens {
        Some(max_output_tokens) => request
            .token_ids
            .len()
            .saturating_add(max_output_tokens as usize),
        None => match &source_trigger {
            DecodeMigrationTrigger::SequenceLength { tokens } => *tokens as usize,
            DecodeMigrationTrigger::TokenId { .. } => {
                bail!("token-ID migration requires a bounded max_tokens")
            }
        },
    };
    if reserved_logical_len <= request.token_ids.len() {
        bail!("decode migration boundary must extend beyond the prompt");
    }
    let reserve_tokens = reserved_logical_len;
    let reserve = control
        .call(
            ControlEndpoint::Prepare,
            destination_worker.worker_id,
            json!({
                "rid": rid,
                "migration_id": migration_id,
                "source": {
                    "bootstrap_host": source_host,
                    "bootstrap_port": source_port,
                    "dp_rank": source_dp_rank,
                },
                "reserve_tokens": reserve_tokens,
                "destination_dp_rank": destination_worker.dp_rank,
            }),
        )
        .await
        .context("destination reservation failed")?;
    if !reserve.success {
        bail!(
            "destination reservation declined with status {}: {}",
            reserve.status,
            reserve.error.as_deref().unwrap_or("unknown error")
        );
    }
    let destination_dp_rank = reserve
        .destination_dp_rank
        .context("destination reservation omitted destination_dp_rank")?;
    if destination_dp_rank != destination_worker.dp_rank {
        bail!(
            "destination reserved DP rank {destination_dp_rank}, expected {}",
            destination_worker.dp_rank
        );
    }
    let bootstrap_room = reserve
        .bootstrap_room
        .context("destination reservation omitted bootstrap_room")?;

    // Stream delivery can coalesce several generated tokens around a trigger.
    // Reserve the bounded request maximum so an acknowledged trigger overshoot
    // can still bind exactly; SGLang releases the unused page-aligned tail.
    let reserved_committed_len = reserved_logical_len - 1;
    let reserved_output_tokens = reserved_committed_len - request.token_ids.len();
    let mut reserved_input_ids = request.token_ids.clone();
    reserved_input_ids.resize(reserved_committed_len, 0);
    let mut reserved_request = build_destination_request(
        &request,
        BootstrapInfo {
            bootstrap_host: source_host.clone(),
            bootstrap_port: source_port,
            bootstrap_room,
        },
        DecodeMigrationRequestState {
            rid: rid.clone(),
            migration_id: migration_id.clone(),
            source_dp_rank,
            is_destination: true,
        },
        reserved_output_tokens,
        destination_worker,
    );
    reserved_request.token_ids = reserved_input_ids;
    reserved_request.decode_migration = None;
    let reserved_request = serde_json::to_value(reserved_request)
        .context("failed to serialize reserved destination request")?;
    let source_prepare = json!({
        "rid": &rid,
        "migration_id": &migration_id,
        "phase": "prepare",
        "bootstrap_room": bootstrap_room,
        "source_dp_rank": source_dp_rank,
    });
    let destination_bootstrap = json!({
        "rid": &rid,
        "migration_id": &migration_id,
        "destination_request": reserved_request,
        "destination_dp_rank": destination_worker.dp_rank,
    });
    // Both operations depend on the reservation room but not on each other.
    // Starting them together keeps setup ahead of short or heavily batched
    // source stages. A response loss may mean the source was prepared, so the
    // cleanup path must cancel it from this point onward.
    cleanup.mark_source_prepared();
    let (source_ready, started) = tokio::join!(
        control.call(
            ControlEndpoint::Sync,
            source_worker.worker_id,
            source_prepare,
        ),
        control.call(
            ControlEndpoint::Prepare,
            destination_worker.worker_id,
            destination_bootstrap,
        )
    );
    let source_ready = source_ready.context("source migration prepare failed")?;
    let started = started.context("destination early bootstrap failed")?;
    if !source_ready.success || source_ready.status != "ready" {
        bail!(
            "source migration prepare declined with status {}: {}",
            source_ready.status,
            source_ready.error.as_deref().unwrap_or("unknown error")
        );
    }
    if !started.success || !matches!(started.status.as_str(), "bootstrapping" | "ready") {
        bail!(
            "destination early bootstrap declined with status {}: {}",
            started.status,
            started.error.as_deref().unwrap_or("unknown error")
        );
    }
    tracing::info!(
        request_id = %rid,
        %migration_id,
        setup_latency_ms = setup_started.elapsed().as_secs_f64() * 1000.0,
        "decode migration setup prepared"
    );

    Ok(PreparedDestination {
        worker: destination_worker,
        source_host,
        source_port,
        source_dp_rank,
        bootstrap_room,
        cleanup,
    })
}

struct KvMigrationSelector {
    chooser: Arc<KvRouter>,
    generate_client: Client,
}

impl KvMigrationSelector {
    async fn select(
        &self,
        request: &PreprocessedRequest,
        tokens: &[TokenIdType],
        policy_constraints: &DecodeMigrationConstraints,
        pinned: Option<WorkerWithDpRank>,
        allowed: Option<HashSet<WorkerId>>,
    ) -> Result<WorkerWithDpRank> {
        let routing = request.routing.as_ref();
        let mut constraints = routing
            .and_then(|r| r.routing_constraints.clone())
            .unwrap_or_default();
        merge_constraints(&mut constraints, policy_constraints);

        let outcome = self
            .chooser
            .find_best_match_details(
                None,
                tokens,
                None,
                request.router_config_override.as_ref(),
                false,
                false,
                routing.and_then(|r| r.lora_name.clone()),
                routing.and_then(|r| r.priority_jump).unwrap_or_default(),
                routing
                    .and_then(|r| r.expected_output_tokens)
                    .or(request.stop_conditions.max_tokens),
                pinned,
                allowed,
                constraints,
            )
            .await?;
        match outcome {
            FindBestMatchOutcome::Routed { worker, .. } => Ok(worker),
            FindBestMatchOutcome::Backpressure { reason, .. } => {
                bail!("decode migration selection backpressured: {reason:?}")
            }
        }
    }
}

#[async_trait]
impl MigrationSelector for KvMigrationSelector {
    async fn select_source(
        &self,
        request: &PreprocessedRequest,
        constraints: &DecodeMigrationConstraints,
    ) -> Result<WorkerWithDpRank> {
        let pinned = request.routing.as_ref().and_then(|routing| {
            routing
                .decode_worker_id
                .or(routing.backend_instance_id)
                .map(|worker_id| WorkerWithDpRank::new(worker_id, routing.dp_rank.unwrap_or(0)))
        });
        let allowed = request
            .routing
            .as_ref()
            .and_then(|routing| routing.allowed_worker_ids.clone());
        self.select(request, &request.token_ids, constraints, pinned, allowed)
            .await
    }

    async fn select_destination(
        &self,
        request: &PreprocessedRequest,
        tokens: &[TokenIdType],
        constraints: &DecodeMigrationConstraints,
        source: WorkerWithDpRank,
    ) -> Result<WorkerWithDpRank> {
        let mut allowed: HashSet<_> = self
            .generate_client
            .instance_ids()
            .into_iter()
            .filter(|worker| *worker != source.worker_id)
            .collect();
        if let Some(request_allowed) = request
            .routing
            .as_ref()
            .and_then(|routing| routing.allowed_worker_ids.as_ref())
        {
            allowed.retain(|worker| request_allowed.contains(worker));
        }
        if allowed.is_empty() {
            bail!("no alternate decode migration destination is available");
        }
        self.select(request, tokens, constraints, None, Some(allowed))
            .await
    }
}

/// Pipeline operator that handles opt-in decode migration requests.
pub struct DecodeMigration {
    enabled: bool,
    selector: Option<Arc<dyn MigrationSelector>>,
    control: Option<Arc<dyn MigrationControl>>,
    destination_backend: Arc<dyn MigrationBackend>,
}

impl DecodeMigration {
    pub async fn new(
        enabled: bool,
        generate_client: Client,
        chooser: Option<Arc<KvRouter>>,
    ) -> Result<Arc<Self>> {
        if enabled && chooser.is_none() {
            bail!("decode migration requires KV routing");
        }
        let destination_backend = Arc::new(DirectMigrationBackend {
            router: DirectBackend::from_client(generate_client.clone(), Default::default()).await?,
        }) as Arc<dyn MigrationBackend>;
        let selector = chooser.map(|chooser| {
            Arc::new(KvMigrationSelector {
                chooser,
                generate_client: generate_client.clone(),
            }) as Arc<dyn MigrationSelector>
        });
        let control = if enabled {
            Some(
                Arc::new(ControlClients::new(generate_client.endpoint.component().clone()).await?)
                    as Arc<dyn MigrationControl>,
            )
        } else {
            None
        };
        Ok(Arc::new(Self {
            enabled,
            selector,
            control,
            destination_backend,
        }))
    }

    #[cfg(test)]
    fn with_dependencies(
        selector: Arc<dyn MigrationSelector>,
        control: Arc<dyn MigrationControl>,
        backend_engine: BackendEngine,
    ) -> Arc<Self> {
        Arc::new(Self {
            enabled: true,
            selector: Some(selector),
            control: Some(control),
            destination_backend: Arc::new(EngineMigrationBackend {
                engine: backend_engine,
            }),
        })
    }

    pub(crate) fn into_operator(
        self: &Arc<Self>,
    ) -> Arc<
        PipelineOperator<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
        >,
    > {
        Operator::into_operator(self)
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for DecodeMigration
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let Some(policy) = request.decode_migration.clone() else {
            let (request, context) = request.into_parts();
            let stream_context = context.context();
            let stream = next.generate(context.map(|_| request)).await?;
            return Ok(ResponseStream::new(Box::pin(stream), stream_context));
        };
        if !self.enabled {
            bail!(
                "request contains nvext.decode_migration but the frontend was not started with --enable-decode-migration"
            );
        }
        validate_request(&request, &policy)?;

        let (mut source_request, context) = request.into_parts();
        let stream_context = context.context();
        let metadata = context.metadata().clone();
        let rid = stream_context.id().to_string();
        let engine_rid = get_distributed_tracing_context()
            .map(|context| context.trace_id)
            .unwrap_or_else(|| rid.clone());
        let migration_uuid = uuid::Uuid::new_v4();
        let migration_id = migration_uuid.to_string();
        ensure_migration_sampling_seed(&mut source_request, migration_uuid);

        let selector = self
            .selector
            .clone()
            .context("migration selector unavailable")?;
        let source_worker = selector
            .select_source(&source_request, &policy.source)
            .await?;
        tracing::info!(
            request_id = %rid,
            source_worker = source_worker.worker_id,
            source_dp_rank = source_worker.dp_rank,
            "decode migration source selected"
        );
        pin_request(&mut source_request, source_worker);
        source_request.decode_migration = None;
        let original_request = source_request.clone();
        source_request.decode_migration_state = Some(DecodeMigrationRequestState {
            rid: engine_rid.clone(),
            migration_id: migration_id.clone(),
            source_dp_rank: source_worker.dp_rank,
            is_destination: false,
        });

        let mut source_stream = next.generate(context.map(|_| source_request)).await?;
        let control = self
            .control
            .clone()
            .context("migration control unavailable")?;
        let destination_backend = self.destination_backend.clone();
        let policy_for_stream = policy.clone();
        let parent_context = stream_context.clone();
        let setup_selector = selector.clone();
        let setup_control = control.clone();
        let setup_request = original_request.clone();
        let setup_constraints = policy.destination.clone();
        let setup_rid = engine_rid.clone();
        let setup_migration_id = migration_id.clone();
        let source_trigger = policy.trigger.clone();
        let output = async_stream::stream! {
            let mut setup = Box::pin(prepare_destination(
                setup_selector,
                setup_control,
                setup_request,
                setup_constraints,
                source_worker,
                setup_rid,
                setup_migration_id,
                source_trigger,
            ));
            let mut setup_result = None;
            let mut forwarded_tokens = 0usize;
            let mut attempted = false;

            'request_stream: loop {
                let source_item = if setup_result.is_none() && !attempted {
                    tokio::select! {
                        result = &mut setup => {
                            setup_result = Some(result);
                            continue;
                        }
                        item = source_stream.next() => item,
                    }
                } else {
                    source_stream.next().await
                };
                let Some(source_item) = source_item else {
                    break;
                };
                if parent_context.is_stopped() || parent_context.is_killed() {
                    break;
                }

                let mut trigger_now = false;
                if let Some(data) = source_item.data.as_ref() {
                    forwarded_tokens += data.token_ids.len();
                    let source_finished = data.finish_reason.is_some();
                    trigger_now = !attempted
                        && !source_finished
                        && trigger_matches(
                            &policy_for_stream.trigger,
                            original_request.token_ids.len(),
                            forwarded_tokens,
                            &data.token_ids,
                        );
                }

                let should_emit = source_item.data.as_ref().is_none_or(|data| {
                    !data.token_ids.is_empty() || data.finish_reason.is_some()
                });
                if should_emit {
                    yield source_item;
                }
                if !trigger_now {
                    continue;
                }
                attempted = true;
                tracing::info!(
                    request_id = %rid,
                    forwarded_tokens,
                    sequence_length = original_request.token_ids.len() + forwarded_tokens,
                    "decode migration trigger reached"
                );

                let prepared = match setup_result.take() {
                    Some(result) => result,
                    None => setup.as_mut().await,
                };
                let PreparedDestination {
                    worker: destination_worker,
                    source_host,
                    source_port,
                    source_dp_rank,
                    bootstrap_room,
                    mut cleanup,
                } = match prepared {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        tracing::warn!(request_id = %rid, "destination setup failed: {error:#}");
                        continue;
                    }
                };

                if parent_context.is_stopped() || parent_context.is_killed() {
                    return;
                }

                // Dynamo owns trigger detection. SGLang only receives a generic
                // request to quiesce at the latest output frontier Dynamo observed.
                let quiesce_deadline = tokio::time::Instant::now() + CONTROL_TIMEOUT;
                let quiesce_started = tokio::time::Instant::now();
                let quiesced = loop {
                    let response = control.call(
                        ControlEndpoint::Sync,
                        source_worker.worker_id,
                        json!({
                            "rid": engine_rid,
                            "migration_id": migration_id,
                            "phase": "quiesce",
                            "bootstrap_room": bootstrap_room,
                            "output_tokens_seen": forwarded_tokens,
                            "source_dp_rank": source_dp_rank,
                        }),
                    ).await;
                    match response {
                        Ok(response) if response.success && response.status == "quiescing" => {
                            if tokio::time::Instant::now() >= quiesce_deadline {
                                cleanup.terminate().await;
                                yield Annotated::from_error(
                                    "source did not reach a safe quiesce boundary",
                                );
                                return;
                            }
                            tokio::time::sleep(QUIESCE_POLL_INTERVAL).await;
                        }
                        Ok(response) if matches!(response.status.as_str(), "finished" | "not_found") => {
                            let _ = control.call(
                                ControlEndpoint::Finalize,
                                destination_worker.worker_id,
                                json!({
                                    "rid": engine_rid,
                                    "migration_id": migration_id,
                                    "side": "destination",
                                    "action": "abort",
                                    "destination_dp_rank": destination_worker.dp_rank,
                                }),
                            ).await;
                            cleanup.disarm();
                            continue 'request_stream;
                        }
                        Ok(response) if response.success && response.status == "quiesced" => {
                            cleanup.mark_source_quiesced();
                            break response;
                        }
                        Ok(response) => {
                            tracing::warn!(request_id = %rid, status = %response.status, error = ?response.error, "source quiesce declined; terminating request");
                            cleanup.terminate().await;
                            yield Annotated::from_error(format!(
                                "source quiesce declined with status {}",
                                response.status
                            ));
                            return;
                        }
                        Err(error) => {
                            tracing::warn!(request_id = %rid, "source quiesce failed; terminating request: {error:#}");
                            cleanup.terminate().await;
                            yield Annotated::from_error(format!("source quiesce failed: {error:#}"));
                            return;
                        }
                    }
                };
                let prompt_len = quiesced.prompt_len.unwrap_or(original_request.token_ids.len());
                let Some(committed_len) = quiesced.committed_len else {
                    cleanup.terminate().await;
                    yield Annotated::from_error("source quiesce omitted committed_len");
                    return;
                };
                let Some(logical_len) = quiesced.logical_len else {
                    cleanup.terminate().await;
                    yield Annotated::from_error("source quiesce omitted logical_len");
                    return;
                };
                let pending_tokens = logical_len.saturating_sub(committed_len);
                if pending_tokens != 1 {
                    cleanup.terminate().await;
                    yield Annotated::from_error(format!(
                        "source quiesce returned {pending_tokens} pending tokens; expected one"
                    ));
                    return;
                }
                let unforwarded = quiesced
                    .unforwarded_committed_output_ids
                    .clone()
                    .unwrap_or_default();
                let committed_output_tokens = committed_len.saturating_sub(prompt_len);
                let mut duplicate_tokens = (forwarded_tokens + unforwarded.len())
                    .saturating_sub(committed_output_tokens);
                tracing::info!(
                    request_id = %rid,
                    %migration_id,
                    quiesce_latency_ms = quiesce_started.elapsed().as_secs_f64() * 1000.0,
                    committed_len,
                    logical_len,
                    unforwarded_tokens = unforwarded.len(),
                    duplicate_tokens,
                    "decode migration source quiesced"
                );

                let mut destination_request = build_destination_request(
                    &original_request,
                    BootstrapInfo {
                        bootstrap_host: source_host,
                        bootstrap_port: source_port,
                        bootstrap_room,
                    },
                    DecodeMigrationRequestState {
                        rid: engine_rid.clone(),
                        migration_id: migration_id.clone(),
                        source_dp_rank,
                        is_destination: true,
                    },
                    committed_output_tokens,
                    destination_worker,
                );
                destination_request.decode_migration = None;
                // The destination was bootstrapped before the trigger. The source
                // now sends the exact token frontier in the same auxiliary transfer
                // as KV, and transfer completion binds the destination request. This
                // avoids a second scheduler control round trip on the handoff path.
                duplicate_tokens = duplicate_tokens.saturating_sub(pending_tokens);

                if !unforwarded.is_empty() {
                    yield Annotated::from_data(LLMEngineOutput {
                        token_ids: unforwarded.clone(),
                        index: Some(0),
                        ..Default::default()
                    });
                }

                let destination_context = Context::with_id_and_metadata(
                    destination_request,
                    destination_transport_id(&rid, &migration_id),
                    metadata.clone(),
                );
                parent_context.link_child(destination_context.context());
                let mut destination_stream = match destination_backend
                    .generate(destination_context, destination_worker)
                    .await
                {
                    Ok(stream) => stream,
                    Err(error) => {
                        tracing::warn!(request_id = %rid, "destination dispatch failed; terminating request: {error:#}");
                        cleanup.terminate().await;
                        yield Annotated::from_error(format!("destination dispatch failed: {error:#}"));
                        return;
                    }
                };

                let mut destination_ready = false;
                let mut destination_failed = false;
                while let Some(destination_item) = destination_stream.next().await {
                    if parent_context.is_stopped() || parent_context.is_killed() {
                        return;
                    }
                    if !destination_item.is_ok() {
                        tracing::warn!(request_id = %rid, "destination stream returned an error before migration completed");
                        destination_failed = true;
                        break;
                    }
                    if destination_item.data.is_none() {
                        if destination_ready {
                            yield destination_item;
                        }
                        continue;
                    }
                    let (mut trimmed, dropped) = match trim_annotated_prefix(destination_item, duplicate_tokens) {
                        Ok(value) => value,
                        Err(error) => {
                            tracing::warn!(request_id = %rid, "destination prefix reconciliation failed: {error:#}");
                            destination_failed = true;
                            break;
                        }
                    };
                    duplicate_tokens -= dropped;
                    normalize_migrated_usage(
                        &mut trimmed,
                        original_request.token_ids.len(),
                        committed_output_tokens,
                    );

                    if !destination_ready {
                        // A decode result proves that SGLang has admitted the
                        // transferred KV and started the destination request.
                        // NIXL completion owns source cleanup. Once destination
                        // decode starts, no control-plane work remains.
                        cleanup.disarm();
                        destination_ready = true;
                        tracing::info!(
                            request_id = %rid,
                            %migration_id,
                            source_worker = source_worker.worker_id,
                            destination_worker = destination_worker.worker_id,
                            "decode migration destination handoff started"
                        );
                    }

                    let should_emit = trimmed.data.as_ref().is_none_or(|data| {
                        !data.token_ids.is_empty() || data.finish_reason.is_some()
                    });
                    if should_emit {
                        yield trimmed;
                    }
                }

                if destination_ready && !destination_failed {
                    return;
                }
                if destination_ready {
                    let _ = control.call(
                        ControlEndpoint::Finalize,
                        destination_worker.worker_id,
                        json!({
                            "rid": engine_rid,
                            "migration_id": migration_id,
                            "side": "destination",
                            "action": "abort",
                            "destination_dp_rank": destination_worker.dp_rank,
                        }),
                    ).await;
                    yield Annotated::from_error("destination stream failed after source migration committed");
                    return;
                }

                cleanup.terminate().await;
                yield Annotated::from_error("destination migration failed before commit");
                return;
            }
        };

        Ok(ResponseStream::new(Box::pin(output), stream_context))
    }
}

fn validate_request(request: &PreprocessedRequest, policy: &DecodeMigrationPolicy) -> Result<()> {
    if matches!(
        policy.trigger,
        DecodeMigrationTrigger::SequenceLength { tokens: 0 }
    ) {
        bail!("decode migration sequence_length trigger must be greater than zero");
    }
    if request.sampling_options.n.unwrap_or(1) != 1
        || request.sampling_options.best_of.unwrap_or(1) != 1
        || request.sampling_options.use_beam_search.unwrap_or(false)
    {
        bail!("decode migration currently supports exactly one non-beam output sequence");
    }
    if request.sampling_options.guided_decoding.is_some() {
        bail!("decode migration does not yet support guided decoding state transfer");
    }
    Ok(())
}

fn ensure_migration_sampling_seed(request: &mut PreprocessedRequest, migration_id: uuid::Uuid) {
    if request.sampling_options.seed.is_none() {
        request.sampling_options.seed = Some((migration_id.as_u128() & i64::MAX as u128) as i64);
    }
}

fn merge_constraints(target: &mut RoutingConstraints, extra: &DecodeMigrationConstraints) {
    target
        .required_taints
        .extend(extra.required_taints.iter().cloned());
    for (taint, weight) in &extra.preferred_taints {
        *target.preferred_taints.entry(taint.clone()).or_default() += weight;
    }
}

fn pin_request(request: &mut PreprocessedRequest, worker: WorkerWithDpRank) {
    let routing = request.routing_mut();
    routing.backend_instance_id = Some(worker.worker_id);
    routing.decode_worker_id = Some(worker.worker_id);
    routing.dp_rank = Some(worker.dp_rank);
}

fn destination_transport_id(rid: &str, migration_id: &str) -> String {
    format!("{rid}:decode-migration:{migration_id}")
}

fn trigger_matches(
    trigger: &DecodeMigrationTrigger,
    prompt_tokens: usize,
    forwarded_tokens: usize,
    chunk_tokens: &[TokenIdType],
) -> bool {
    match trigger {
        DecodeMigrationTrigger::TokenId { token_id } => chunk_tokens.contains(token_id),
        DecodeMigrationTrigger::SequenceLength { tokens } => {
            prompt_tokens.saturating_add(forwarded_tokens) >= *tokens as usize
        }
    }
}

fn build_destination_request(
    original: &PreprocessedRequest,
    bootstrap_info: BootstrapInfo,
    state: DecodeMigrationRequestState,
    committed_output_tokens: usize,
    worker: WorkerWithDpRank,
) -> PreprocessedRequest {
    let mut request = original.clone();
    request.prompt_embeds = None;
    request.multi_modal_data = None;
    request.mm_routing_info = None;
    request.prefill_result = None;
    request.bootstrap_info = Some(bootstrap_info);
    request.decode_migration_state = Some(state);
    request.migration_link = None;
    if let Some(max_tokens) = request.stop_conditions.max_tokens.as_mut() {
        *max_tokens = max_tokens
            .saturating_sub(committed_output_tokens as u32)
            .max(1);
    }
    if let Some(min_tokens) = request.stop_conditions.min_tokens.as_mut() {
        *min_tokens = min_tokens.saturating_sub(committed_output_tokens as u32);
    }
    pin_request(&mut request, worker);
    request
}

fn trim_annotated_prefix(
    mut item: Annotated<LLMEngineOutput>,
    count: usize,
) -> Result<(Annotated<LLMEngineOutput>, usize)> {
    if count == 0 {
        return Ok((item, 0));
    }
    let Some(output) = item.data.as_mut() else {
        return Ok((item, 0));
    };
    let dropped = count.min(output.token_ids.len());
    if dropped == 0 {
        return Ok((item, 0));
    }
    if output.text.as_deref().is_some_and(|text| !text.is_empty()) {
        bail!(
            "cannot trim already-forwarded token positions from engine-detokenized output; use Dynamo token postprocessing"
        );
    }
    output.token_ids.drain(..dropped);
    if let Some(tokens) = output.tokens.as_mut() {
        tokens.drain(..dropped.min(tokens.len()));
    }
    if let Some(log_probs) = output.log_probs.as_mut() {
        log_probs.drain(..dropped.min(log_probs.len()));
    }
    if let Some(top_logprobs) = output.top_logprobs.as_mut() {
        top_logprobs.drain(..dropped.min(top_logprobs.len()));
    }
    Ok((item, dropped))
}

fn normalize_migrated_usage(
    item: &mut Annotated<LLMEngineOutput>,
    original_prompt_tokens: usize,
    committed_output_tokens: usize,
) {
    let Some(usage) = item
        .data
        .as_mut()
        .and_then(|output| output.completion_usage.as_mut())
    else {
        return;
    };
    let prompt_tokens = u32::try_from(original_prompt_tokens).unwrap_or(u32::MAX);
    let committed_output_tokens = u32::try_from(committed_output_tokens).unwrap_or(u32::MAX);
    usage.prompt_tokens = prompt_tokens;
    usage.completion_tokens = usage
        .completion_tokens
        .saturating_add(committed_output_tokens);
    usage.total_tokens = prompt_tokens.saturating_add(usage.completion_tokens);
    usage.prompt_tokens_details = None;
    usage.completion_tokens_details = None;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceCleanupPhase {
    NotPrepared,
    Prepared,
    Quiesced,
}

struct MigrationCleanupPlan {
    control: Arc<dyn MigrationControl>,
    rid: String,
    migration_id: String,
    source_worker: u64,
    destination_worker: u64,
    source_dp_rank: u32,
    destination_dp_rank: u32,
    source_phase: SourceCleanupPhase,
}

impl MigrationCleanupPlan {
    async fn run(self) {
        let _ = self
            .control
            .call(
                ControlEndpoint::Finalize,
                self.destination_worker,
                json!({
                    "rid": self.rid,
                    "migration_id": self.migration_id,
                    "side": "destination",
                    "action": "abort",
                    "destination_dp_rank": self.destination_dp_rank,
                }),
            )
            .await;
        if self.source_phase == SourceCleanupPhase::Prepared {
            let _ = self
                .control
                .call(
                    ControlEndpoint::Finalize,
                    self.source_worker,
                    json!({
                        "rid": self.rid,
                        "migration_id": self.migration_id,
                        "action": "cancel",
                        "source_dp_rank": self.source_dp_rank,
                    }),
                )
                .await;
        }
    }
}

struct MigrationCleanup {
    plan: Option<MigrationCleanupPlan>,
}

impl MigrationCleanup {
    fn new(
        control: Arc<dyn MigrationControl>,
        rid: String,
        migration_id: String,
        source_worker: u64,
        destination_worker: u64,
        source_dp_rank: u32,
        destination_dp_rank: u32,
    ) -> Self {
        Self {
            plan: Some(MigrationCleanupPlan {
                control,
                rid,
                migration_id,
                source_worker,
                destination_worker,
                source_dp_rank,
                destination_dp_rank,
                source_phase: SourceCleanupPhase::NotPrepared,
            }),
        }
    }

    fn disarm(&mut self) {
        self.plan = None;
    }

    fn mark_source_prepared(&mut self) {
        if let Some(plan) = self.plan.as_mut() {
            plan.source_phase = SourceCleanupPhase::Prepared;
        }
    }

    fn mark_source_quiesced(&mut self) {
        if let Some(plan) = self.plan.as_mut() {
            plan.source_phase = SourceCleanupPhase::Quiesced;
        }
    }

    async fn terminate(&mut self) {
        if let Some(plan) = self.plan.take() {
            plan.run().await;
        }
    }
}

impl Drop for MigrationCleanup {
    fn drop(&mut self) {
        if let Some(plan) = self.plan.take() {
            tokio::spawn(plan.run());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::llm_backend::TokenType;
    use dynamo_runtime::pipeline::AsyncEngine;

    const SOURCE_DP_RANK: u32 = 3;
    const DESTINATION_DP_RANK: u32 = 1;

    fn assert_control_rank(request: &Value, field: &str, expected: u32) {
        assert_eq!(
            request.get(field).and_then(Value::as_u64),
            Some(u64::from(expected)),
            "control request omitted or misrouted {field}: {request}"
        );
    }

    #[test]
    fn single_trigger_is_structurally_tagged() {
        let policy: DecodeMigrationPolicy = serde_json::from_value(json!({
            "source": {"required_taints": ["decode/fast"]},
            "destination": {"required_taints": ["decode/slow"]},
            "trigger": {"type": "token_id", "token_id": 42}
        }))
        .unwrap();
        assert!(matches!(
            policy.trigger,
            DecodeMigrationTrigger::TokenId { token_id: 42 }
        ));
        assert!(
            serde_json::from_value::<DecodeMigrationPolicy>(json!({
                "source": {},
                "destination": {},
                "trigger": {"token_id": 42, "sequence_length": 100}
            }))
            .is_err()
        );
    }

    #[test]
    fn sequence_trigger_counts_coalesced_stream_chunks() {
        let trigger = DecodeMigrationTrigger::SequenceLength { tokens: 8 };
        assert!(!trigger_matches(&trigger, 3, 4, &[10, 11, 12, 13]));
        assert!(trigger_matches(&trigger, 3, 5, &[14]));
    }

    #[test]
    fn token_trigger_matches_any_position_in_stream_interval() {
        let trigger = DecodeMigrationTrigger::TokenId { token_id: 7 };
        assert!(trigger_matches(&trigger, 10, 4, &[4, 7, 9, 10]));
    }

    #[test]
    fn token_trigger_fires_on_boundary_token_without_waiting() {
        let trigger = DecodeMigrationTrigger::TokenId { token_id: 7 };
        assert!(trigger_matches(&trigger, 10, 1, &[7]));
        assert!(!trigger_matches(&trigger, 10, 2, &[8]));
    }

    #[tokio::test]
    async fn concurrent_requests_can_migrate_from_the_same_source() {
        let harness = integration_harness(Scenario::SuccessCoalesced);
        let (first, second) =
            tokio::join!(collect_tokens(&harness, 6), collect_tokens(&harness, 6),);

        for (tokens, errors) in [first, second] {
            assert!(errors.is_empty(), "{errors:?}");
            assert_eq!(tokens, vec![10, 11, 12, 13, 14]);
        }
        assert_eq!(
            harness
                .state
                .events()
                .iter()
                .filter(|event| event.as_str() == "source:quiesce")
                .count(),
            2
        );
    }

    #[test]
    fn prefix_trim_keeps_token_aligned_fields() {
        let output = LLMEngineOutput {
            token_ids: vec![1, 2, 3],
            tokens: Some(vec![
                Some("a".into()) as TokenType,
                Some("b".into()),
                Some("c".into()),
            ]),
            log_probs: Some(vec![-1.0, -2.0, -3.0]),
            top_logprobs: Some(vec![vec![], vec![], vec![]]),
            ..Default::default()
        };
        let (trimmed, dropped) = trim_annotated_prefix(Annotated::from_data(output), 2).unwrap();
        let output = trimmed.data.unwrap();
        assert_eq!(dropped, 2);
        assert_eq!(output.token_ids, vec![3]);
        assert_eq!(output.tokens.unwrap(), vec![Some("c".into())]);
        assert_eq!(output.log_probs.unwrap(), vec![-3.0]);
        assert_eq!(output.top_logprobs.unwrap().len(), 1);
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Scenario {
        SuccessCoalesced,
        RollbackSyntheticTail,
        QuiesceTransportFailure,
        CancelDuringReserve,
        CancelDuringQuiesce,
        DestinationReserveTransportFailure,
        DestinationBootstrapFailure,
        DestinationDispatchFailure,
        EmptyFirstDestinationChunk,
        FinishAtTrigger,
        FinishDuringQuiesce,
        CancelAfterArm,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum WorkerRole {
        Source,
        Destination,
    }

    #[derive(Default)]
    struct ScenarioState {
        events: std::sync::Mutex<Vec<String>>,
        destination_generate_calls: std::sync::atomic::AtomicUsize,
    }

    impl ScenarioState {
        fn record(&self, event: impl Into<String>) {
            self.events.lock().unwrap().push(event.into());
        }

        fn events(&self) -> Vec<String> {
            self.events.lock().unwrap().clone()
        }
    }

    async fn wait_for_event(state: &ScenarioState, expected: &str) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if state.events().iter().any(|event| event == expected) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    struct MockGenerateEngine {
        role: WorkerRole,
        scenario: Scenario,
        state: Arc<ScenarioState>,
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            anyhow::Error,
        > for MockGenerateEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            let (request, context) = request.into_parts();
            let output = |ids: Vec<u32>, finish_reason| {
                Annotated::from_data(LLMEngineOutput {
                    token_ids: ids,
                    finish_reason,
                    index: Some(0),
                    ..Default::default()
                })
            };
            let responses = match self.role {
                WorkerRole::Source => {
                    let migration_state = request
                        .decode_migration_state
                        .as_ref()
                        .expect("source request must carry its engine migration identity");
                    assert_eq!(migration_state.rid, context.id());
                    assert!(!migration_state.migration_id.is_empty());
                    assert_eq!(migration_state.source_dp_rank, SOURCE_DP_RANK);
                    assert_eq!(
                        request.routing.as_ref().and_then(|routing| routing.dp_rank),
                        Some(SOURCE_DP_RANK)
                    );
                    self.state.record("source:generate");
                    match self.scenario {
                        Scenario::SuccessCoalesced | Scenario::EmptyFirstDestinationChunk => vec![
                            output(vec![10, 11, 12], None),
                            output(vec![99], Some(crate::protocols::common::FinishReason::Stop)),
                        ],
                        Scenario::RollbackSyntheticTail
                        | Scenario::QuiesceTransportFailure
                        | Scenario::CancelDuringReserve
                        | Scenario::CancelDuringQuiesce
                        | Scenario::DestinationReserveTransportFailure
                        | Scenario::DestinationBootstrapFailure
                        | Scenario::DestinationDispatchFailure
                        | Scenario::CancelAfterArm => vec![
                            output(vec![10], None),
                            output(
                                vec![11, 12, 13],
                                Some(crate::protocols::common::FinishReason::Stop),
                            ),
                        ],
                        Scenario::FinishAtTrigger => vec![output(
                            vec![10, 11],
                            Some(crate::protocols::common::FinishReason::Stop),
                        )],
                        Scenario::FinishDuringQuiesce => vec![
                            output(vec![10], None),
                            output(vec![11], Some(crate::protocols::common::FinishReason::Stop)),
                        ],
                    }
                }
                WorkerRole::Destination => {
                    self.state
                        .destination_generate_calls
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    self.state.record("destination:generate");
                    let migration_state = request
                        .decode_migration_state
                        .as_ref()
                        .expect("destination request must retain the source engine RID");
                    assert_eq!(migration_state.source_dp_rank, SOURCE_DP_RANK);
                    assert_eq!(
                        request.routing.as_ref().and_then(|routing| routing.dp_rank),
                        Some(DESTINATION_DP_RANK)
                    );
                    assert_ne!(migration_state.rid, context.id());
                    assert_eq!(
                        context.id(),
                        destination_transport_id(
                            &migration_state.rid,
                            &migration_state.migration_id
                        )
                    );
                    if self.scenario == Scenario::DestinationDispatchFailure {
                        return Err(anyhow!("injected destination dispatch failure"));
                    }
                    match self.scenario {
                        Scenario::RollbackSyntheticTail => {
                            vec![Annotated::from_error("injected destination failure")]
                        }
                        Scenario::SuccessCoalesced => vec![
                            output(vec![13], None),
                            output(vec![14], Some(crate::protocols::common::FinishReason::Stop)),
                        ],
                        Scenario::EmptyFirstDestinationChunk => vec![
                            output(vec![], None),
                            output(vec![13], Some(crate::protocols::common::FinishReason::Stop)),
                        ],
                        Scenario::CancelAfterArm => vec![output(vec![13], None)],
                        Scenario::QuiesceTransportFailure
                        | Scenario::CancelDuringReserve
                        | Scenario::CancelDuringQuiesce
                        | Scenario::DestinationReserveTransportFailure
                        | Scenario::DestinationBootstrapFailure
                        | Scenario::FinishAtTrigger
                        | Scenario::FinishDuringQuiesce => {
                            panic!("destination generation must not start in this scenario")
                        }
                        Scenario::DestinationDispatchFailure => unreachable!(),
                    }
                }
            };
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter(responses)),
                context.context(),
            ))
        }
    }

    struct ContextChangingEngine;

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            anyhow::Error,
        > for ContextChangingEngine
    {
        async fn generate(
            &self,
            _request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            let response_context = Context::with_id_and_metadata(
                (),
                "downstream-context".to_string(),
                Default::default(),
            );
            let response = Annotated::from_data(LLMEngineOutput {
                token_ids: vec![17],
                index: Some(0),
                ..Default::default()
            });
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter([response])),
                response_context.context(),
            ))
        }
    }

    struct MockControl {
        scenario: Scenario,
        state: Arc<ScenarioState>,
    }

    #[async_trait]
    impl MigrationControl for MockControl {
        async fn call(
            &self,
            endpoint: ControlEndpoint,
            instance_id: u64,
            request: Value,
        ) -> Result<ControlResponse> {
            let phase = request.get("phase").and_then(Value::as_str);
            let action = request.get("action").and_then(Value::as_str);
            let side = request.get("side").and_then(Value::as_str);
            let response = if phase == Some("describe") {
                assert!(matches!(endpoint, ControlEndpoint::Sync));
                assert_eq!(instance_id, 1);
                assert_control_rank(&request, "source_dp_rank", SOURCE_DP_RANK);
                self.state.record("source:describe");
                json!({
                    "success": true,
                    "status": "described",
                    "bootstrap_host": "127.0.0.1",
                    "bootstrap_port": 9876,
                    "source_dp_rank": SOURCE_DP_RANK,
                })
            } else if phase == Some("quiesce") {
                assert!(matches!(endpoint, ControlEndpoint::Sync));
                assert_eq!(instance_id, 1);
                assert_control_rank(&request, "source_dp_rank", SOURCE_DP_RANK);
                self.state.record("source:quiesce");
                if self.scenario == Scenario::QuiesceTransportFailure {
                    return Err(anyhow!("injected source quiesce transport failure"));
                }
                if self.scenario == Scenario::CancelDuringQuiesce {
                    futures::future::pending::<()>().await;
                    unreachable!();
                }
                if self.scenario == Scenario::FinishDuringQuiesce {
                    json!({"success": false, "status": "finished"})
                } else if matches!(
                    self.scenario,
                    Scenario::SuccessCoalesced | Scenario::EmptyFirstDestinationChunk
                ) {
                    json!({
                        "success": true,
                        "status": "quiesced",
                        "source_dp_rank": SOURCE_DP_RANK,
                        "prompt_len": 3,
                        "committed_len": 5,
                        "logical_len": 6,
                        "unforwarded_committed_output_ids": [],
                    })
                } else {
                    json!({
                        "success": true,
                        "status": "quiesced",
                        "source_dp_rank": SOURCE_DP_RANK,
                        "prompt_len": 3,
                        "committed_len": 5,
                        "logical_len": 6,
                        "unforwarded_committed_output_ids": [11],
                    })
                }
            } else if phase == Some("prepare") {
                assert!(matches!(endpoint, ControlEndpoint::Sync));
                assert_eq!(instance_id, 1);
                assert_control_rank(&request, "source_dp_rank", SOURCE_DP_RANK);
                assert!(
                    request
                        .get("bootstrap_room")
                        .and_then(Value::as_u64)
                        .is_some()
                );
                self.state.record("source:prepare");
                json!({"success": true, "status": "ready"})
            } else if request.get("source").is_some() {
                assert!(matches!(endpoint, ControlEndpoint::Prepare));
                assert_eq!(instance_id, 2);
                assert_control_rank(&request, "destination_dp_rank", DESTINATION_DP_RANK);
                assert_eq!(request["source"]["dp_rank"], SOURCE_DP_RANK);
                self.state.record("destination:reserve");
                if self.scenario == Scenario::CancelDuringReserve {
                    futures::future::pending::<()>().await;
                    unreachable!();
                }
                if self.scenario == Scenario::DestinationReserveTransportFailure {
                    return Err(anyhow!("injected destination reservation response loss"));
                }
                json!({
                    "success": true,
                    "status": "reserved",
                    "bootstrap_room": 777,
                    "destination_dp_rank": DESTINATION_DP_RANK,
                })
            } else if request.get("destination_request").is_some() {
                assert!(matches!(endpoint, ControlEndpoint::Prepare));
                assert_eq!(instance_id, 2);
                assert_control_rank(&request, "destination_dp_rank", DESTINATION_DP_RANK);
                assert_eq!(
                    request["destination_request"]["token_ids"]
                        .as_array()
                        .unwrap()
                        .len(),
                    12,
                );
                self.state.record("destination:bootstrap");
                if self.scenario == Scenario::DestinationBootstrapFailure {
                    json!({
                        "success": false,
                        "status": "error",
                        "error": "injected destination bootstrap failure",
                    })
                } else {
                    json!({"success": true, "status": "bootstrapping"})
                }
            } else if side == Some("destination") {
                assert!(matches!(endpoint, ControlEndpoint::Finalize));
                assert_eq!(instance_id, 2);
                assert_control_rank(&request, "destination_dp_rank", DESTINATION_DP_RANK);
                let action = action.unwrap();
                self.state.record(format!("destination:{action}"));
                json!({"success": true, "status": action})
            } else if let Some(action) = action {
                assert!(matches!(endpoint, ControlEndpoint::Finalize));
                assert_eq!(instance_id, 1);
                assert_control_rank(&request, "source_dp_rank", SOURCE_DP_RANK);
                self.state.record(format!("source:{action}"));
                assert_eq!(action, "cancel");
                json!({"success": true, "status": action})
            } else {
                panic!("unexpected control request: {request}")
            };
            serde_json::from_value(response).map_err(Into::into)
        }
    }

    struct MockSelector;

    #[async_trait]
    impl MigrationSelector for MockSelector {
        async fn select_source(
            &self,
            _request: &PreprocessedRequest,
            constraints: &DecodeMigrationConstraints,
        ) -> Result<WorkerWithDpRank> {
            assert_eq!(constraints.required_taints, vec!["test/fast"]);
            Ok(WorkerWithDpRank::new(1, SOURCE_DP_RANK))
        }

        async fn select_destination(
            &self,
            _request: &PreprocessedRequest,
            _tokens: &[TokenIdType],
            constraints: &DecodeMigrationConstraints,
            source: WorkerWithDpRank,
        ) -> Result<WorkerWithDpRank> {
            assert_eq!(source, WorkerWithDpRank::new(1, SOURCE_DP_RANK));
            assert_eq!(constraints.required_taints, vec!["test/slow"]);
            Ok(WorkerWithDpRank::new(2, DESTINATION_DP_RANK))
        }
    }

    struct IntegrationHarness {
        operator: Arc<DecodeMigration>,
        source: BackendEngine,
        state: Arc<ScenarioState>,
    }

    fn integration_harness(scenario: Scenario) -> IntegrationHarness {
        let state = Arc::new(ScenarioState::default());
        let source: BackendEngine = Arc::new(MockGenerateEngine {
            role: WorkerRole::Source,
            scenario,
            state: state.clone(),
        });
        let destination: BackendEngine = Arc::new(MockGenerateEngine {
            role: WorkerRole::Destination,
            scenario,
            state: state.clone(),
        });
        let operator = DecodeMigration::with_dependencies(
            Arc::new(MockSelector),
            Arc::new(MockControl {
                scenario,
                state: state.clone(),
            }),
            destination,
        );
        IntegrationHarness {
            operator,
            source,
            state,
        }
    }

    fn plain_request(id: &str) -> Context<PreprocessedRequest> {
        let request = PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(crate::protocols::common::StopConditions {
                max_tokens: Some(10),
                ..Default::default()
            })
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        Context::with_id_and_metadata(request, id.to_string(), Default::default())
    }

    fn migration_request(trigger_tokens: u32) -> Context<PreprocessedRequest> {
        let request = PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(crate::protocols::common::StopConditions {
                max_tokens: Some(10),
                ..Default::default()
            })
            .sampling_options(Default::default())
            .output_options(Default::default())
            .decode_migration(Some(DecodeMigrationPolicy {
                source: DecodeMigrationConstraints {
                    required_taints: vec!["test/fast".into()],
                    ..Default::default()
                },
                destination: DecodeMigrationConstraints {
                    required_taints: vec!["test/slow".into()],
                    ..Default::default()
                },
                trigger: DecodeMigrationTrigger::SequenceLength {
                    tokens: trigger_tokens,
                },
            }))
            .build()
            .unwrap();
        Context::with_id_and_metadata(
            request,
            uuid::Uuid::new_v4().to_string(),
            Default::default(),
        )
    }

    async fn collect_tokens(
        harness: &IntegrationHarness,
        trigger_tokens: u32,
    ) -> (Vec<u32>, Vec<String>) {
        let mut stream = harness
            .operator
            .generate(migration_request(trigger_tokens), harness.source.clone())
            .await
            .unwrap();
        let mut tokens = Vec::new();
        let mut errors = Vec::new();
        while let Some(item) = stream.next().await {
            if let Some(data) = item.data {
                tokens.extend(data.token_ids);
            }
            if let Some(error) = item.error {
                errors.push(error.to_string());
            }
        }
        (tokens, errors)
    }

    #[tokio::test]
    async fn pass_through_preserves_upstream_context_and_data() {
        let harness = integration_harness(Scenario::SuccessCoalesced);
        let mut stream = harness
            .operator
            .generate(
                plain_request("upstream-context"),
                Arc::new(ContextChangingEngine),
            )
            .await
            .unwrap();

        assert_eq!(stream.context().id(), "upstream-context");
        assert_eq!(
            stream.next().await.unwrap().data.unwrap().token_ids,
            vec![17]
        );
        assert!(stream.next().await.is_none());
        assert!(harness.state.events().is_empty());
    }

    #[tokio::test]
    async fn integration_migrates_coalesced_chunks_without_duplicates() {
        let harness = integration_harness(Scenario::SuccessCoalesced);
        let (tokens, errors) = collect_tokens(&harness, 6).await;
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(tokens, vec![10, 11, 12, 13, 14]);
        let events = harness.state.events();
        assert!(events.contains(&"source:quiesce".to_string()));
        assert!(!events.contains(&"source:cancel".to_string()));
    }

    #[tokio::test]
    async fn integration_skips_a_fully_duplicate_first_destination_chunk() {
        let harness = integration_harness(Scenario::EmptyFirstDestinationChunk);
        let (tokens, errors) = collect_tokens(&harness, 6).await;
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(tokens, vec![10, 11, 12, 13]);
    }

    #[tokio::test]
    async fn integration_destination_failure_terminates_quiesced_request() {
        let harness = integration_harness(Scenario::RollbackSyntheticTail);
        let (tokens, errors) = collect_tokens(&harness, 4).await;
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(tokens, vec![10, 11]);
        let events = harness.state.events();
        assert!(events.contains(&"destination:abort".to_string()));
        assert!(!events.contains(&"source:cancel".to_string()));
    }

    async fn drop_stream_while_control_is_pending(scenario: Scenario) -> Vec<String> {
        let harness = integration_harness(scenario);
        let mut stream = harness
            .operator
            .generate(migration_request(4), harness.source.clone())
            .await
            .unwrap();
        assert_eq!(
            stream.next().await.unwrap().data.unwrap().token_ids,
            vec![10]
        );
        {
            let mut pending_item = Box::pin(stream.next());
            assert!(
                tokio::time::timeout(Duration::from_millis(50), &mut pending_item)
                    .await
                    .is_err()
            );
        }
        drop(stream);
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if harness
                    .state
                    .events()
                    .contains(&"destination:abort".to_string())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        harness.state.events()
    }

    #[tokio::test]
    async fn integration_cancel_during_reserve_aborts_destination() {
        let events = drop_stream_while_control_is_pending(Scenario::CancelDuringReserve).await;
        assert!(events.contains(&"destination:reserve".to_string()));
        assert!(events.contains(&"destination:abort".to_string()));
        assert!(!events.contains(&"source:quiesce".to_string()));
        assert!(!events.contains(&"source:cancel".to_string()));
    }

    #[tokio::test]
    async fn integration_cancel_during_quiesce_cancels_both_sides() {
        let events = drop_stream_while_control_is_pending(Scenario::CancelDuringQuiesce).await;
        assert!(events.contains(&"source:quiesce".to_string()));
        assert!(events.contains(&"destination:abort".to_string()));
        assert!(events.contains(&"source:cancel".to_string()));
    }

    #[tokio::test]
    async fn integration_reserve_response_loss_aborts_destination() {
        let harness = integration_harness(Scenario::DestinationReserveTransportFailure);
        let (tokens, errors) = collect_tokens(&harness, 4).await;
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(tokens, vec![10, 11, 12, 13]);
        wait_for_event(&harness.state, "destination:abort").await;
        let events = harness.state.events();
        assert!(events.contains(&"destination:reserve".to_string()));
        assert!(events.contains(&"destination:abort".to_string()));
        assert!(!events.contains(&"source:quiesce".to_string()));
        assert_eq!(
            harness
                .state
                .destination_generate_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn integration_quiesce_transport_failure_terminates_request() {
        let harness = integration_harness(Scenario::QuiesceTransportFailure);
        let (tokens, errors) = collect_tokens(&harness, 4).await;
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(tokens, vec![10]);
        let events = harness.state.events();
        assert!(events.contains(&"destination:abort".to_string()));
        assert!(events.contains(&"source:cancel".to_string()));
        assert_eq!(
            harness
                .state
                .destination_generate_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn integration_destination_bootstrap_failure_keeps_source_running() {
        let harness = integration_harness(Scenario::DestinationBootstrapFailure);
        let (tokens, errors) = collect_tokens(&harness, 4).await;
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(tokens, vec![10, 11, 12, 13]);
        wait_for_event(&harness.state, "destination:abort").await;
        let events = harness.state.events();
        assert!(events.contains(&"destination:abort".to_string()));
        assert!(events.contains(&"source:cancel".to_string()));
        assert_eq!(
            harness
                .state
                .destination_generate_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn integration_destination_dispatch_failure_terminates_request() {
        let harness = integration_harness(Scenario::DestinationDispatchFailure);
        let (tokens, errors) = collect_tokens(&harness, 4).await;
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(tokens, vec![10, 11]);
        let events = harness.state.events();
        assert!(events.contains(&"destination:abort".to_string()));
        assert!(!events.contains(&"source:cancel".to_string()));
        assert_eq!(
            harness
                .state
                .destination_generate_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn integration_finished_trigger_chunk_never_starts_migration() {
        let harness = integration_harness(Scenario::FinishAtTrigger);
        let (tokens, errors) = collect_tokens(&harness, 5).await;
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(tokens, vec![10, 11]);
        let mut events = harness.state.events();
        if events.contains(&"destination:reserve".to_string()) {
            wait_for_event(&harness.state, "destination:abort").await;
            events = harness.state.events();
            assert!(events.contains(&"destination:abort".to_string()));
        }
        assert!(events.contains(&"source:generate".to_string()));
        assert!(!events.contains(&"source:quiesce".to_string()));
        assert_eq!(
            harness
                .state
                .destination_generate_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn integration_source_finishing_during_quiesce_aborts_reservation() {
        let harness = integration_harness(Scenario::FinishDuringQuiesce);
        let (tokens, errors) = collect_tokens(&harness, 4).await;
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(tokens, vec![10, 11]);
        let events = harness.state.events();
        assert!(events.contains(&"destination:abort".to_string()));
        assert_eq!(
            harness
                .state
                .destination_generate_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn integration_stream_drop_aborts_destination_after_source_quiesce() {
        let harness = integration_harness(Scenario::CancelAfterArm);
        let mut stream = harness
            .operator
            .generate(migration_request(4), harness.source.clone())
            .await
            .unwrap();
        assert_eq!(
            stream.next().await.unwrap().data.unwrap().token_ids,
            vec![10]
        );
        assert_eq!(
            stream.next().await.unwrap().data.unwrap().token_ids,
            vec![11]
        );
        drop(stream);
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let events = harness.state.events();
                if events.contains(&"destination:abort".to_string()) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(
            !harness
                .state
                .events()
                .contains(&"source:cancel".to_string())
        );
    }

    #[test]
    fn migrated_request_decrements_remaining_generation() {
        let mut request = PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(crate::protocols::common::StopConditions {
                max_tokens: Some(10),
                min_tokens: Some(4),
                ..Default::default()
            })
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        request.decode_migration = Some(DecodeMigrationPolicy {
            source: Default::default(),
            destination: Default::default(),
            trigger: DecodeMigrationTrigger::SequenceLength { tokens: 5 },
        });
        let migrated = build_destination_request(
            &request,
            BootstrapInfo {
                bootstrap_host: "127.0.0.1".into(),
                bootstrap_port: 1234,
                bootstrap_room: 99,
            },
            DecodeMigrationRequestState {
                rid: "rid".into(),
                migration_id: "migration".into(),
                source_dp_rank: 0,
                is_destination: true,
            },
            3,
            WorkerWithDpRank::new(8, 0),
        );
        assert_eq!(migrated.stop_conditions.max_tokens, Some(7));
        assert_eq!(migrated.stop_conditions.min_tokens, Some(1));
        assert_eq!(migrated.routing.unwrap().backend_instance_id, Some(8));
    }

    #[test]
    fn migration_assigns_a_stable_sampling_seed_when_missing() {
        let mut request = PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        let migration_id = uuid::Uuid::from_u128(123456789);

        ensure_migration_sampling_seed(&mut request, migration_id);
        let assigned = request.sampling_options.seed;
        ensure_migration_sampling_seed(&mut request, uuid::Uuid::from_u128(987654321));

        assert_eq!(assigned, Some(123456789));
        assert_eq!(request.sampling_options.seed, assigned);
    }

    #[test]
    fn migration_preserves_an_explicit_sampling_seed() {
        let mut request = PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(crate::protocols::common::SamplingOptions {
                seed: Some(42),
                ..Default::default()
            })
            .output_options(Default::default())
            .build()
            .unwrap();

        ensure_migration_sampling_seed(&mut request, uuid::Uuid::new_v4());

        assert_eq!(request.sampling_options.seed, Some(42));
    }
}
