// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use clap::ValueEnum;
use dashmap::DashMap;
use dynamo_mocker::common::protocols::{EngineType, MockEngineArgs, WorkerType};
use dynamo_mocker::live::{LiveEngine, LiveRequest};
use dynamo_mocker::scheduler::MockerMetrics;
use dynamo_trtllm_sidecar::proto as pb;
use futures::Stream;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::{Request, Response, Status};
use uuid::Uuid;

use request::PreparedRequest;

#[path = "server_handoff.rs"]
mod handoff;
#[path = "server_request.rs"]
mod request;

const DP_RANK: u32 = 0;
const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 256;
/// Ceiling on the pump's per-request buffer. Without it the buffer is the
/// client's own `max_tokens`, so idle streams could pin arbitrary memory.
const MAX_PUMP_BUFFER: usize = 256;
/// `ServerInfo.schema_revision` is documented as "zero is invalid".
const SCHEMA_REVISION: u32 = 1;
type BoxedStatusResult<T> = Result<T, Box<Status>>;

/// Wire-level role exposed by one mock server process.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ServerMode {
    Aggregated,
    Prefill,
    Decode,
}

impl fmt::Display for ServerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Aggregated => "aggregated",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        })
    }
}

#[derive(Clone, Debug)]
pub struct MockerServerConfig {
    pub model: String,
    pub mode: ServerMode,
    pub seed: u64,
    /// Surfaced as `ModelInfo.max_context_length`. The TensorRT-LLM sidecar
    /// refuses to start without a positive value, and derives a default
    /// `max_tokens` from it when a request omits one.
    pub context_length: u32,
    pub max_concurrent_requests: usize,
    pub kv_host: String,
    pub kv_port: u16,
}

impl Default for MockerServerConfig {
    fn default() -> Self {
        Self {
            model: "mocker-model".to_string(),
            mode: ServerMode::Aggregated,
            seed: 42,
            context_length: 32_768,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
            kv_host: "127.0.0.1".to_string(),
            kv_port: 5600,
        }
    }
}

struct InFlight {
    uuid: Uuid,
    /// The session id the *client* knows this request by: its own on a decode
    /// request, this server's otherwise. Abort-by-session matches on it.
    session_id: String,
    aborted: Arc<AtomicBool>,
}

/// Removes the in-flight entry however the response stream ends -- terminal
/// event, error, or the client dropping it. Leaking an entry would make Abort
/// report a dead request as live and wedge its id against reuse.
struct InFlightGuard {
    inflight: Arc<DashMap<String, InFlight>>,
    request_id: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.inflight.remove(&self.request_id);
    }
}

/// Mocker-backed TensorRT-LLM OpenEngine services.
#[derive(Clone)]
pub struct TrtllmMockerService {
    config: Arc<MockerServerConfig>,
    model_info: Arc<pb::ModelInfo>,
    server_info: Arc<pb::ServerInfo>,
    engine: LiveEngine,
    request_permits: Arc<Semaphore>,
    inflight: Arc<DashMap<String, InFlight>>,
    /// Every `GenerateRequest` the server accepted, so a test can assert what
    /// the client actually put on the wire rather than only what came back.
    received: Arc<Mutex<Vec<pb::GenerateRequest>>>,
}

impl TrtllmMockerService {
    pub fn new(config: MockerServerConfig, engine_args: MockEngineArgs) -> anyhow::Result<Self> {
        // Normalizing first is what applies the TensorRT-LLM rules: block-size
        // floor and default, the max_model_len rejection, and the capacity
        // scheduler policy check.
        let engine_args = engine_args.normalized()?;
        anyhow::ensure!(
            engine_args.engine_type == EngineType::Trtllm,
            "Mocker engine_type must be trtllm"
        );
        anyhow::ensure!(engine_args.dp_size == 1, "Mocker dp_size must be 1");
        anyhow::ensure!(
            engine_args.worker_type == WorkerType::Aggregated,
            "Mocker worker_type must be aggregated; use the server mode for the emulated wire role"
        );
        anyhow::ensure!(!config.model.trim().is_empty(), "model must be non-empty");
        anyhow::ensure!(
            config.context_length > 0,
            "context_length must be greater than 0"
        );
        anyhow::ensure!(
            config.max_concurrent_requests > 0,
            "max_concurrent_requests must be greater than 0"
        );
        anyhow::ensure!(
            config.mode == ServerMode::Aggregated || config.kv_port != 0,
            "kv_port must be non-zero in prefill and decode modes"
        );

        let max_concurrent_requests = config.max_concurrent_requests;
        let model_info = pb::ModelInfo {
            model_id: config.model.clone(),
            served_model_name: config.model.clone(),
            served_model_aliases: Vec::new(),
            max_context_length: Some(config.context_length),
            max_output_tokens: Some(request::MAX_NEW_TOKENS),
            tokenizer_modes: Vec::new(),
            supports_text_input: Some(false),
            supports_token_ids_input: Some(true),
            generation: Some(pb::GenerationCapabilities {
                prompt_logprobs: Some(pb::LogprobCapabilities {
                    supported: Some(true),
                    candidate_selection_modes: candidate_modes(),
                    max_top_n: Some(request::MAX_CANDIDATES as u32),
                }),
                output_logprobs: Some(pb::LogprobCapabilities {
                    supported: Some(true),
                    candidate_selection_modes: candidate_modes(),
                    max_top_n: Some(request::MAX_CANDIDATES as u32),
                }),
                guided_decoding: None,
                max_num_sequences: Some(1),
                supports_priority: Some(false),
                supports_stop_in_output: Some(false),
                supports_cache_salt: Some(false),
                supports_prefix_cache_bypass: Some(false),
            }),
            supports_lora: Some(false),
            supports_multimodal: Some(false),
            reasoning_parser: String::new(),
            tool_call_parser: String::new(),
            extra: None,
        };
        let server_info = pb::ServerInfo {
            engine_name: "tensorrt_llm".to_string(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            engine_role: match config.mode {
                ServerMode::Aggregated => pb::EngineRole::Aggregated,
                ServerMode::Prefill => pb::EngineRole::Prefill,
                ServerMode::Decode => pb::EngineRole::Decode,
            } as i32,
            instance_id: format!("dynamo-trtllm-mocker-{}", config.mode),
            supported_models: vec![config.model.clone()],
            parallelism: Some(pb::ParallelismInfo {
                tensor_parallel_size: Some(1),
                pipeline_parallel_size: Some(1),
                data_parallel_size: Some(engine_args.dp_size),
                data_parallel_rank: Some(DP_RANK),
                data_parallel_start_rank: Some(DP_RANK),
                decode_context_parallel_size: Some(1),
            }),
            kv_connector: Some(pb::KvConnectorInfo {
                enabled: Some(config.mode != ServerMode::Aggregated),
                transfer_backend: handoff::TRANSFER_BACKEND.to_string(),
                local_endpoints: vec![pb::KvEndpoint {
                    host: config.kv_host.clone(),
                    port: u32::from(config.kv_port),
                    protocol: handoff::KV_PROTOCOL.to_string(),
                }],
                supported_protocols: vec![handoff::KV_PROTOCOL.to_string()],
                supports_remote_prefill: Some(true),
                supports_decode_pull: Some(false),
                supports_abort_cleanup: Some(true),
                schema_version: Some(SCHEMA_REVISION),
            }),
            schema_revision: SCHEMA_REVISION,
            minimum_client_revision: SCHEMA_REVISION,
            schema_release: String::new(),
            capacity: Some(pb::DeploymentCapacity {
                kv_block_size: Some(
                    u32::try_from(engine_args.block_size)
                        .map_err(|_| anyhow::anyhow!("block_size exceeds the Control API range"))?,
                ),
                total_kv_blocks: Some(u64::try_from(engine_args.num_gpu_blocks).map_err(|_| {
                    anyhow::anyhow!("num_gpu_blocks exceeds the Control API range")
                })?),
                max_running_requests: engine_args
                    .max_num_seqs
                    .map(u64::try_from)
                    .transpose()
                    .map_err(|_| anyhow::anyhow!("max_num_seqs exceeds the Control API range"))?,
                max_batched_tokens: engine_args
                    .max_num_batched_tokens
                    .map(u64::try_from)
                    .transpose()
                    .map_err(|_| {
                        anyhow::anyhow!("max_num_batched_tokens exceeds the Control API range")
                    })?,
                max_loras: None,
            }),
            extra: None,
        };

        Ok(Self {
            config: Arc::new(config),
            model_info: Arc::new(model_info),
            server_info: Arc::new(server_info),
            engine: LiveEngine::start(engine_args, DP_RANK)?,
            request_permits: Arc::new(Semaphore::new(max_concurrent_requests)),
            inflight: Arc::new(DashMap::new()),
            received: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn config(&self) -> &MockerServerConfig {
        &self.config
    }

    pub fn active_request_count(&self) -> usize {
        self.engine.active_request_count()
    }

    pub fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics> {
        self.engine.metrics_receiver()
    }

    /// Requests the server accepted, in arrival order.
    pub fn received_requests(&self) -> Vec<pb::GenerateRequest> {
        self.received
            .lock()
            .expect("received lock poisoned")
            .clone()
    }

    /// Stop the simulated engine and report any scheduler failure it collected.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        self.engine.shutdown().await
    }

    async fn start_generation(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<
        (
            PreparedRequest,
            LiveRequest,
            OwnedSemaphorePermit,
            InFlightGuard,
            Arc<AtomicBool>,
        ),
        Status,
    > {
        // Rejecting an overload before parsing keeps the cheap path cheap.
        let permit = self
            .request_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("Mocker concurrent request limit reached"))?;
        let request = request.into_inner();
        self.received
            .lock()
            .expect("received lock poisoned")
            .push(request.clone());
        let prepared = PreparedRequest::new(request, &self.config).map_err(|status| *status)?;

        // Claim the id before submitting: LiveEngine would otherwise reject the
        // duplicate with an anyhow that surfaces as an opaque INTERNAL.
        let aborted = Arc::new(AtomicBool::new(false));
        let entry = InFlight {
            uuid: prepared.uuid,
            session_id: prepared.client_session_id.clone(),
            aborted: Arc::clone(&aborted),
        };
        match self.inflight.entry(prepared.request_id.clone()) {
            dashmap::mapref::entry::Entry::Occupied(_) => {
                return Err(Status::already_exists(format!(
                    "request_id '{}' is already in flight",
                    prepared.request_id
                )));
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                vacant.insert(entry);
            }
        }
        let guard = InFlightGuard {
            inflight: Arc::clone(&self.inflight),
            request_id: prepared.request_id.clone(),
        };

        let live = self
            .engine
            .submit(prepared.direct_request())
            .await
            .map_err(|error| {
                Status::internal(format!("Mocker request submission failed: {error}"))
            })?;
        Ok((prepared, live, permit, guard, aborted))
    }

    async fn abort_uuid(&self, request_id: &str) -> Result<pb::AbortStatus, Status> {
        // The DashMap guard is a temporary of this `let`, so it is released
        // before the `.await` below. Do not restructure this into an `if let`
        // that spans the await: a live shard guard would block every task that
        // touches the same shard, including InFlightGuard::drop.
        let Some((uuid, aborted)) = self
            .inflight
            .get(request_id)
            .map(|entry| (entry.uuid, Arc::clone(&entry.aborted)))
        else {
            return Ok(pb::AbortStatus::AlreadyFinished);
        };
        // `cancel` reports false once the scheduler has already released the
        // request, which happens when it finishes -- not when the client has
        // read the last token. Reporting ABORTED there would contradict the
        // terminal event the stream is about to emit.
        let cancelled = self
            .engine
            .cancel(uuid)
            .await
            .map_err(|error| Status::internal(format!("Mocker abort failed: {error}")))?;
        if !cancelled {
            return Ok(pb::AbortStatus::AlreadyFinished);
        }
        aborted.store(true, Ordering::SeqCst);
        Ok(pb::AbortStatus::Aborted)
    }
}

fn candidate_modes() -> Vec<i32> {
    vec![
        pb::CandidateTokenSelectionMode::TopN as i32,
        pb::CandidateTokenSelectionMode::TokenIds as i32,
        pb::CandidateTokenSelectionMode::All as i32,
    ]
}

/// The only place a `GenerateResponse` is built, so no call site can emit one
/// with an empty `event` oneof -- which the sidecar rejects outright.
fn response(request_id: &str, event: pb::generate_response::Event) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event: Some(event),
        usage: None,
    }
}

fn engine_error(
    request_id: &str,
    code: pb::ErrorCode,
    message: &str,
    retryable: bool,
) -> pb::GenerateResponse {
    response(
        request_id,
        pb::generate_response::Event::Error(pb::EngineError {
            code: code as i32,
            message: message.to_string(),
            retryable,
        }),
    )
}

#[tonic::async_trait]
impl pb::inference_server::Inference for TrtllmMockerService {
    type GenerateStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send + 'static>>;

    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let (prepared, mut live, permit, guard, aborted) = self.start_generation(request).await?;
        let config = Arc::clone(&self.config);

        // Decouple LiveEngine's small fixed per-request buffer from client and
        // transport pacing. A pump drains the engine promptly into a buffer
        // bounded by this request's own token budget, so a bursty producer
        // racing ahead of a slow gRPC consumer no longer trips LiveEngine's
        // slow-consumer shedding. Dropping the client stream still cancels
        // unfinished scheduler work.
        let (signal_tx, mut signal_rx) = tokio::sync::mpsc::channel(
            prepared
                .max_output_tokens
                .saturating_add(1)
                .min(MAX_PUMP_BUFFER),
        );
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = signal_tx.closed() => break,
                    signal = live.recv() => {
                        let Some(signal) = signal else { break };
                        let completed = signal.completed;
                        if signal_tx.send(signal).await.is_err() || completed {
                            break;
                        }
                    }
                }
            }
        });

        let stream = async_stream::try_stream! {
            let _permit = permit;
            let _guard = guard;
            let request_id = prepared.request_id.clone();

            if let Some(prompt) = prepared.prompt_output() {
                yield response(&request_id, pb::generate_response::Event::Prompt(prompt));
            }

            let mut generated = 0usize;
            let mut cached_tokens = None;
            while let Some(signal) = signal_rx.recv().await {
                if signal.rejected {
                    // An accepted request fails in-band and the RPC still closes
                    // OK; a non-OK status is reserved for validation and
                    // transport failures.
                    yield engine_error(
                        &request_id,
                        pb::ErrorCode::Overloaded,
                        "request exceeds the simulated KV-cache capacity",
                        true,
                    );
                    return;
                }
                cached_tokens = cached_tokens.or(signal.cached_tokens);
                let Some(token_id) = signal.token_id else {
                    // Accepted requests report failure in-band, per the same
                    // contract as the capacity rejection above.
                    yield engine_error(
                        &request_id,
                        pb::ErrorCode::Internal,
                        "Mocker output signal is missing a token ID",
                        false,
                    );
                    return;
                };
                generated += 1;
                yield response(
                    &request_id,
                    pb::generate_response::Event::Token(prepared.token_output(token_id)),
                );

                if signal.completed {
                    if config.mode == ServerMode::Prefill {
                        // PrefillReady is the terminal event for a context
                        // request; a `finished` after it reads as "request
                        // complete" and the decode leg never runs.
                        yield response(
                            &request_id,
                            pb::generate_response::Event::PrefillReady(
                                prepared.prefill_ready(&config),
                            ),
                        );
                    } else {
                        // An abort may have landed while the engine was running
                        // ahead of the client; reporting LENGTH here would
                        // contradict the ABORTED the abort caller was given.
                        let reason = if aborted.load(Ordering::SeqCst) {
                            pb::FinishReason::Cancelled
                        } else {
                            pb::FinishReason::Length
                        };
                        yield prepared.finished(reason, generated, cached_tokens);
                    }
                    return;
                }
            }

            // The stream must never end without a terminal event: the sidecar
            // fails the request outright if it does.
            if aborted.load(Ordering::SeqCst) {
                yield prepared.finished(pb::FinishReason::Cancelled, generated, cached_tokens);
            } else {
                yield engine_error(
                    &request_id,
                    pb::ErrorCode::Internal,
                    "Mocker output channel closed before a terminal response",
                    false,
                );
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for TrtllmMockerService {
    async fn get_server_info(
        &self,
        _request: Request<pb::GetServerInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        Ok(Response::new((*self.server_info).clone()))
    }

    async fn get_model_info(
        &self,
        request: Request<pb::GetModelInfoRequest>,
    ) -> Result<Response<pb::ModelInfo>, Status> {
        let requested = request.into_inner().model;
        if !requested.is_empty() && requested != self.config.model {
            return Err(Status::not_found(format!(
                "model '{requested}' is not served; this server serves '{}'",
                self.config.model
            )));
        }
        Ok(Response::new((*self.model_info).clone()))
    }

    async fn get_load(
        &self,
        _request: Request<pb::GetLoadRequest>,
    ) -> Result<Response<pb::LoadInfo>, Status> {
        let metrics = self.engine.metrics_receiver().borrow().clone();
        Ok(Response::new(pb::LoadInfo {
            instance_id: self.server_info.instance_id.clone(),
            timestamp_unix_nanos: None,
            running_requests: Some(metrics.running_requests as u32),
            queued_requests: Some(metrics.waiting_requests as u32),
            active_kv_sessions: (self.config.mode != ServerMode::Aggregated)
                .then(|| self.inflight.len() as u32),
            used_kv_blocks: Some(metrics.active_decode_blocks),
            total_kv_blocks: Some(metrics.total_blocks),
            running_tokens: None,
            waiting_tokens: None,
            prefill_batch_size: None,
            decode_batch_size: None,
            ranks: Vec::new(),
            attributes: None,
        }))
    }

    async fn health(
        &self,
        request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        if request.into_inner().include_inference_probe {
            return unsupported("Health with an inference probe");
        }
        let check = |name: &str| pb::HealthCheck {
            name: name.to_string(),
            state: pb::HealthState::Ready as i32,
            message: String::new(),
        };
        Ok(Response::new(pb::HealthResponse {
            state: pb::HealthState::Ready as i32,
            checks: vec![check("grpc"), check("scheduler"), check("model")],
        }))
    }

    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        // ABORT_STATUS_UNSPECIFIED is a protocol error to the sidecar, so every
        // path below returns ABORTED or ALREADY_FINISHED.
        let status = match request.into_inner().target {
            Some(pb::abort_request::Target::RequestId(request_id)) => {
                self.abort_uuid(&request_id).await?
            }
            Some(pb::abort_request::Target::KvSession(session)) => {
                let request_id = self.inflight.iter().find_map(|entry| {
                    (entry.session_id == session.session_id).then(|| entry.key().clone())
                });
                match request_id {
                    Some(request_id) => self.abort_uuid(&request_id).await?,
                    None => pb::AbortStatus::AlreadyFinished,
                }
            }
            Some(pb::abort_request::Target::AllRequests(_)) => {
                // Collect first: holding shard guards across the awaits below
                // would block every task touching the same shard.
                let request_ids: Vec<String> = self
                    .inflight
                    .iter()
                    .map(|entry| entry.key().clone())
                    .collect();
                let mut aborted = pb::AbortStatus::AlreadyFinished;
                let mut failures = Vec::new();
                for request_id in request_ids {
                    // Keep going on failure: stopping here would leave the
                    // sweep half-applied with no way to tell how far it got.
                    match self.abort_uuid(&request_id).await {
                        Ok(pb::AbortStatus::Aborted) => aborted = pb::AbortStatus::Aborted,
                        Ok(_) => {}
                        Err(error) => failures.push(format!("{request_id}: {error}")),
                    }
                }
                if !failures.is_empty() {
                    return Err(Status::internal(format!(
                        "Mocker aborted what it could; {} request(s) failed: {}",
                        failures.len(),
                        failures.join(", ")
                    )));
                }
                aborted
            }
            None => return Err(Status::invalid_argument("Abort requires a target")),
        };
        Ok(Response::new(pb::AbortResponse {
            status: status as i32,
            message: String::new(),
        }))
    }

    async fn load_lora(
        &self,
        _request: Request<pb::LoadLoraRequest>,
    ) -> Result<Response<pb::LoadLoraResponse>, Status> {
        unsupported("LoadLora")
    }

    async fn unload_lora(
        &self,
        _request: Request<pb::UnloadLoraRequest>,
    ) -> Result<Response<pb::UnloadLoraResponse>, Status> {
        unsupported("UnloadLora")
    }

    async fn list_loras(
        &self,
        _request: Request<pb::ListLorasRequest>,
    ) -> Result<Response<pb::ListLorasResponse>, Status> {
        unsupported("ListLoras")
    }

    // KV events stay UNIMPLEMENTED on purpose: the real TensorRT-LLM OpenEngine
    // server does not implement them, and a mocker that did would let a test
    // pass here and fail against a real engine.
    async fn get_kv_event_sources(
        &self,
        _request: Request<pb::GetKvEventSourcesRequest>,
    ) -> Result<Response<pb::GetKvEventSourcesResponse>, Status> {
        unsupported("GetKvEventSources")
    }

    type SubscribeKvEventsStream =
        Pin<Box<dyn Stream<Item = Result<pb::SubscribeKvEventsResponse, Status>> + Send + 'static>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        unsupported("SubscribeKvEvents")
    }
}

#[allow(clippy::result_large_err)]
fn unsupported<T>(rpc: &str) -> Result<Response<T>, Status> {
    Err(Status::unimplemented(format!(
        "{rpc} is not implemented by the TensorRT-LLM OpenEngine server"
    )))
}

#[cfg(test)]
#[path = "server_tests.rs"]
mod tests;
