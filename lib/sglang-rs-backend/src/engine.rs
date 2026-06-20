// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo SGLang backend for Shepherd Model Gateway's scheduler gRPC service. Forwards
//! `PreprocessedRequest` to SMG's `Generate` RPC and streams tokens back.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, DisaggregationMode, DynamoError, EngineConfig, ErrorType,
    FinishReason, GenerateContext, KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt,
    LlmRegistration, PreprocessedRequest, WorkerConfig, chunk, usage,
};
use futures::stream::BoxStream;
use rand::Rng;
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;
use tonic::transport::{Channel, Endpoint};

use crate::args::Args;
use crate::proto::scheduler::{
    AbortRequest, DisaggregatedParams, GenerateRequest, GetModelInfoRequest, GetServerInfoRequest,
    HealthCheckRequest, TokenizedInput, generate_response,
    sglang_scheduler_client::SglangSchedulerClient,
};
use crate::sampling::{build_sampling_params, parse_finish_reason};
use crate::server_info::{KvEventsConfig, offset_endpoint_port, parse_server_info};

/// Initial gRPC dial timeout. The port binds early — before the scheduler
/// finishes loading weights — so the connect itself is fast; this just
/// guards against a misconfigured endpoint.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Cap on the exponential backoff between HealthCheck polls during startup.
const HEALTH_BACKOFF_MAX: Duration = Duration::from_secs(5);
/// Emit an info-level "still waiting" log once per this many HealthCheck
/// failures so ops can see the backend is alive and waiting on SGLang.
const HEALTH_LOG_EVERY_N_ATTEMPTS: u32 = 20;
/// Spread concurrent Generate streams across multiple HTTP/2 connections.
/// A single tonic Channel can trip local grpc/h2 stream pressure at high
/// concurrency before SGLang itself is saturated.
const DEFAULT_GRPC_CHANNELS: usize = 16;
const GRPC_CHANNELS_ENV: &str = "DYN_SGLANG_GRPC_CHANNELS";
/// Enable deterministic remapping of Dynamo's 63-bit bootstrap rooms into
/// SMG v1.5's signed 32-bit `bootstrap_room` field.
const SMG_BOOTSTRAP_ROOM_ENV: &str = "DYN_SMG_BOOTSTRAP_ROOM";
const SMG_BOOTSTRAP_ROOM_MAX: u64 = i32::MAX as u64;

pub struct SglangBackend {
    grpc_endpoint: String,
    /// Discovered from SGLang's `GetServerInfo` in `start()`; read by
    /// `generate()` to dispatch prefill vs decode.
    disaggregation_mode: OnceCell<DisaggregationMode>,
    bootstrap: OnceCell<(String, u16)>,
    clients: OnceCell<Vec<SglangSchedulerClient<Channel>>>,
    next_client: AtomicUsize,
    grpc_channel_count: usize,
    /// Unset → KV routing off; populated in `start()` from `GetServerInfo`.
    kv_events: OnceCell<KvEventsConfig>,
    /// SGLang's `dp_size`; 1 when DP attention is off (the common case).
    dp_size: OnceCell<u32>,
    smg_bootstrap_room_compat: bool,
}

impl SglangBackend {
    /// Parse from an explicit argv (the PyO3 entry point).
    pub fn from_argv(argv: Vec<String>) -> Result<(Self, WorkerConfig), DynamoError> {
        Self::build(<Args as clap::Parser>::try_parse_from(argv))
    }

    fn build(parsed: Result<Args, clap::Error>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = parsed.map_err(|e| invalid_arg(e.to_string()))?;
        let engine = SglangBackend {
            grpc_endpoint: args.sglang_grpc_endpoint,
            disaggregation_mode: OnceCell::new(),
            bootstrap: OnceCell::new(),
            clients: OnceCell::new(),
            next_client: AtomicUsize::new(0),
            grpc_channel_count: grpc_channel_count_from_env(),
            kv_events: OnceCell::new(),
            dp_size: OnceCell::new(),
            smg_bootstrap_room_compat: truthy_env(SMG_BOOTSTRAP_ROOM_ENV),
        };
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            model_name: args.model_name.unwrap_or_default(),
            served_model_name: args.served_model_name,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            disaggregation_mode: args.common.disaggregation_mode,
            ..Default::default()
        };
        Ok((engine, config))
    }

    fn mode(&self) -> DisaggregationMode {
        self.disaggregation_mode.get().copied().unwrap_or_default()
    }

    async fn generate_decode(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let mut client = self.client_or_err()?;
        let prompt_len = request.token_ids.len() as u32;
        let disaggregated_params = request
            .bootstrap_info
            .as_ref()
            .map(|bi| {
                self.to_smg_bootstrap_room(bi.bootstrap_room)
                    .map(|bootstrap_room| DisaggregatedParams {
                        bootstrap_host: bi.bootstrap_host.clone(),
                        bootstrap_port: i32::from(bi.bootstrap_port),
                        bootstrap_room,
                    })
            })
            .transpose()?;
        let grpc_req = GenerateRequest {
            request_id: ctx.id().to_string(),
            tokenized: Some(TokenizedInput {
                original_text: String::new(),
                input_ids: request.token_ids.clone(),
            }),
            sampling_params: Some(build_sampling_params(
                &request.sampling_options,
                &request.stop_conditions,
                &request.output_options,
            )),
            stream: true,
            disaggregated_params,
            data_parallel_rank: request_dp_rank(&request, false),
            ..Default::default()
        };
        let mut stream = client
            .generate(grpc_req)
            .await
            .map_err(|e| backend_error(format!("Generate RPC: {e}")))?
            .into_inner();

        Ok(Box::pin(async_stream::stream! {
            let mut completion_tokens: u32 = 0;
            let mut saw_chunk = false;
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield Ok(LLMEngineOutput::cancelled()
                            .with_usage(usage(prompt_len, completion_tokens)));
                        break;
                    }
                    maybe_chunk = stream.next() => {
                        let Some(chunk_res) = maybe_chunk else {
                            yield Ok(LLMEngineOutput::error("stream ended without terminal".into()));
                            break;
                        };
                        let resp = match chunk_res {
                            Ok(r) => r,
                            Err(s) => {
                                yield Ok(LLMEngineOutput::error(format!("gRPC stream error: {s}")));
                                break;
                            }
                        };
                        match resp.response {
                            Some(generate_response::Response::Chunk(chunk_resp)) => {
                                saw_chunk = true;
                                let chunk_len = chunk_resp.token_ids.len() as u32;
                                completion_tokens = if chunk_resp.completion_tokens > 0 {
                                    chunk_resp.completion_tokens
                                } else {
                                    completion_tokens.saturating_add(chunk_len)
                                };
                                for tid in chunk_resp.token_ids {
                                    yield Ok(chunk::token(tid));
                                }
                            }
                            Some(generate_response::Response::Complete(complete)) => {
                                if !saw_chunk && !complete.output_ids.is_empty() {
                                    completion_tokens = completion_tokens.saturating_add(complete.output_ids.len() as u32);
                                    for tid in complete.output_ids.iter().copied() {
                                        yield Ok(chunk::token(tid));
                                    }
                                }
                                let total = if complete.completion_tokens > 0 {
                                    complete.completion_tokens
                                } else {
                                    completion_tokens
                                };
                                let terminal = match parse_finish_reason(&complete.finish_reason) {
                                    FinishReason::Length => LLMEngineOutput::length(),
                                    FinishReason::Cancelled => LLMEngineOutput::cancelled(),
                                    FinishReason::Error(msg) => LLMEngineOutput::error(msg),
                                    _ => LLMEngineOutput::stop(),
                                };
                                yield Ok(terminal.with_usage(usage(prompt_len, total)));
                                break;
                            }
                            None => {
                                yield Ok(LLMEngineOutput::error("Generate response missing chunk/complete".into()));
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    /// Mirrors `components/src/dynamo/sglang/request_handlers/llm/prefill_handler.py`:
    /// open the Generate RPC once, yield ONE non-terminal bootstrap chunk
    /// synchronously, then drain the gRPC stream in-band (inside the same
    /// async_stream) until SMG returns a terminal `complete` response — only
    /// then close the outer stream with a terminal Stop chunk.
    ///
    /// The drain must stay in-band (no `tokio::spawn`) because closing the
    /// outer stream early — i.e. before sglang acknowledges prefill done —
    /// makes the frontend treat the stream as incomplete and retry the
    /// prefill on the same worker, which then collides with the still-active
    /// gRPC request (sglang rejects "Duplicate active gRPC request id").
    async fn generate_prefill(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let mut client = self.client_or_err()?;
        let (bootstrap_host, bootstrap_port) = self
            .bootstrap
            .get()
            .ok_or_else(|| backend_error("prefill: bootstrap not initialised"))?
            .clone();
        // Honour a router-supplied room (carries dp_rank), else mint a fresh room.
        let raw_bootstrap_room = request
            .bootstrap_info
            .as_ref()
            .map(|bi| bi.bootstrap_room)
            .unwrap_or_else(|| self.generate_bootstrap_room());
        let bootstrap_room = self.to_smg_bootstrap_room(raw_bootstrap_room)?;
        let bootstrap_room_u64 = bootstrap_room as u64;

        let grpc_req = GenerateRequest {
            request_id: ctx.id().to_string(),
            tokenized: Some(TokenizedInput {
                original_text: String::new(),
                input_ids: request.token_ids.clone(),
            }),
            sampling_params: Some(build_sampling_params(
                &request.sampling_options,
                &request.stop_conditions,
                &request.output_options,
            )),
            stream: true,
            disaggregated_params: Some(DisaggregatedParams {
                bootstrap_host: bootstrap_host.clone(),
                bootstrap_port: i32::from(bootstrap_port),
                bootstrap_room,
            }),
            data_parallel_rank: request_dp_rank(&request, true),
            ..Default::default()
        };
        let bootstrap_chunk = LLMEngineOutput {
            disaggregated_params: Some(serde_json::json!({
                "bootstrap_host": bootstrap_host,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": bootstrap_room_u64,
            })),
            ..Default::default()
        };

        Ok(Box::pin(async_stream::stream! {
            // Open the gRPC stream BEFORE yielding bootstrap so a failed RPC
            // surfaces as an error chunk on the outer stream rather than a
            // closed stream after the frontend has already extracted the
            // bootstrap triple.
            let mut stream = match client.generate(grpc_req).await {
                Ok(r) => r.into_inner(),
                Err(e) => {
                    yield Ok(LLMEngineOutput::error(format!("prefill Generate RPC: {e}")));
                    return;
                }
            };
            yield Ok(bootstrap_chunk);

            // Drain the stream until SMG returns `complete`. SGLang prefill
            // blocks until decode pulls KV, so this is what holds the outer
            // stream open.
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield Ok(LLMEngineOutput::cancelled());
                        return;
                    }
                    maybe_chunk = stream.next() => match maybe_chunk {
                        Some(Ok(r)) => match r.response {
                            Some(generate_response::Response::Complete(complete)) => {
                                match parse_finish_reason(&complete.finish_reason) {
                                    FinishReason::Cancelled => yield Ok(LLMEngineOutput::cancelled()),
                                    FinishReason::Error(msg) => yield Ok(LLMEngineOutput::error(msg)),
                                    _ => yield Ok(LLMEngineOutput::stop()),
                                }
                                return;
                            }
                            // Non-terminal chunks from prefill carry no tokens we
                            // need to forward (the bootstrap triple was already
                            // emitted); discard them.
                            Some(generate_response::Response::Chunk(_)) => continue,
                            None => {
                                yield Ok(LLMEngineOutput::error("prefill response missing chunk/complete".to_string()));
                                return;
                            }
                        },
                        Some(Err(s)) => {
                            yield Ok(LLMEngineOutput::error(format!("prefill stream error: {s}")));
                            return;
                        }
                        None => {
                            yield Ok(LLMEngineOutput::error("prefill stream closed without terminal".to_string()));
                            return;
                        }
                    },
                }
            }
        }))
    }

    fn client_or_err(&self) -> Result<SglangSchedulerClient<Channel>, DynamoError> {
        let clients = self
            .clients
            .get()
            .ok_or_else(|| backend_error("generate called before start"))?;
        let idx = self.next_client.fetch_add(1, Ordering::Relaxed) % clients.len();
        Ok(clients[idx].clone())
    }

    fn generate_bootstrap_room(&self) -> u64 {
        if self.smg_bootstrap_room_compat {
            rand::rng().random_range(1..=SMG_BOOTSTRAP_ROOM_MAX)
        } else {
            rand::rng().random_range(1..i64::MAX) as u64
        }
    }

    fn to_smg_bootstrap_room(&self, room: u64) -> Result<i32, DynamoError> {
        let dp_size = self.dp_size.get().copied().unwrap_or(1);
        map_smg_bootstrap_room(room, dp_size, self.smg_bootstrap_room_compat)
    }
}

#[async_trait]
impl LLMEngine for SglangBackend {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        let mut client = connect_client(&self.grpc_endpoint).await?;

        // Wait for SGLang's scheduler to finish loading weights. We don't cap
        // this with a wall-clock — model load time is workload-dependent (small
        // models load in seconds; multi-100B models from cold HF cache can take
        // tens of minutes). Lifecycle ownership belongs to the operator: k8s
        // liveness probes / SIGTERM / `ctx.stopped()` will cancel this loop.
        //
        // What still fails fast:
        //  - any non-transient gRPC Status (Unimplemented / InvalidArgument /
        //    PermissionDenied etc.) — wrong proto or misconfigured endpoint.
        //  - the underlying tonic Channel dying — sglang process gone.
        // What we retry forever (with backoff):
        //  - `Status::Unavailable` and transport-level transient errors —
        //    sglang still starting / momentarily restarting.
        //  - `Ok { healthy: false }` — sglang scheduler reports "Starting".
        let mut backoff = Duration::from_millis(500);
        let mut attempts: u32 = 0;
        loop {
            attempts = attempts.saturating_add(1);
            match client.health_check(HealthCheckRequest {}).await {
                Ok(resp) => {
                    if resp.into_inner().healthy {
                        break;
                    }
                    if attempts.is_multiple_of(HEALTH_LOG_EVERY_N_ATTEMPTS) {
                        tracing::info!(
                            endpoint = %self.grpc_endpoint,
                            attempts,
                            "waiting for SGLang scheduler to report healthy"
                        );
                    }
                }
                Err(status) if is_transient_health_error(&status) => {
                    if attempts.is_multiple_of(HEALTH_LOG_EVERY_N_ATTEMPTS) {
                        tracing::info!(
                            endpoint = %self.grpc_endpoint,
                            attempts,
                            code = ?status.code(),
                            "waiting for SGLang gRPC server (transient: {})",
                            status.message()
                        );
                    }
                }
                Err(status) => {
                    return Err(backend_error(format!("HealthCheck: {status}")));
                }
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(HEALTH_BACKOFF_MAX);
        }

        let model_info = client
            .get_model_info(GetModelInfoRequest {})
            .await
            .map_err(|e| backend_error(format!("GetModelInfo: {e}")))?
            .into_inner();
        let server_info = client
            .get_server_info(GetServerInfoRequest {})
            .await
            .map_err(|e| backend_error(format!("GetServerInfo: {e}")))?
            .into_inner();
        let info = parse_server_info(&server_info);
        let mode = info.disaggregation_mode.unwrap_or_default();
        let _ = self.disaggregation_mode.set(mode);

        let model_path = if !model_info.model_path.is_empty() {
            model_info.model_path.clone()
        } else {
            info.model_path.clone().unwrap_or_default()
        };
        if model_path.is_empty() {
            return Err(backend_error("SGLang reported empty model_path"));
        }
        let served_model_name = if !model_info.served_model_name.is_empty() {
            model_info.served_model_name.clone()
        } else {
            info.served_model_name.unwrap_or_else(|| model_path.clone())
        };

        if mode.is_prefill() {
            let port = info.bootstrap_port.ok_or_else(|| {
                backend_error("GetServerInfo.disaggregation_bootstrap_port missing")
            })?;
            let host = advertised_bootstrap_host(info.bootstrap_host, local_bootstrap_host());
            let _ = self.bootstrap.set((host, port));
        }

        let mut clients = Vec::with_capacity(self.grpc_channel_count);
        clients.push(client);
        for _ in 1..self.grpc_channel_count {
            clients.push(connect_client(&self.grpc_endpoint).await?);
        }
        self.clients
            .set(clients)
            .map_err(|_| backend_error("clients already set"))?;

        let total_kv_blocks = match (info.max_total_num_tokens, info.page_size) {
            (Some(t), Some(p)) if p > 0 => Some(t.div_ceil(p as u64)),
            _ => None,
        };

        let dp_size = info.dp_size.unwrap_or(1);
        let _ = self.dp_size.set(dp_size);
        if let Some(kv) = info.kv_events.clone() {
            let _ = self.kv_events.set(kv);
        }

        tracing::info!(
            endpoint = %self.grpc_endpoint,
            model = %model_path,
            disagg_mode = %mode,
            kv_events_enabled = info.kv_events.is_some(),
            dp_size,
            grpc_channels = self.grpc_channel_count,
            smg_bootstrap_room_compat = self.smg_bootstrap_room_compat,
            "sglang backend connected"
        );

        let context_length = u32::try_from(model_info.max_context_length)
            .ok()
            .filter(|&n| n > 0);

        let (bootstrap_host, bootstrap_port) = self.bootstrap.get().cloned().unzip();
        Ok(EngineConfig {
            model: model_path,
            served_model_name: Some(served_model_name),
            llm: Some(LlmRegistration {
                context_length,
                kv_cache_block_size: info.page_size,
                total_kv_blocks,
                max_num_seqs: info.max_running_requests,
                max_num_batched_tokens: info.max_prefill_tokens.or(info.max_total_num_tokens),
                data_parallel_size: info.dp_size,
                data_parallel_start_rank: None,
                bootstrap_host,
                bootstrap_port,
            }),
            ..Default::default()
        })
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        Ok(Some(build_health_check_payload()))
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let ctx = ctx.inner_arc();
        if self.mode().is_prefill() {
            self.generate_prefill(request, ctx).await
        } else {
            self.generate_decode(request, ctx).await
        }
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(clients) = self.clients.get() else {
            return;
        };
        let idx = self.next_client.fetch_add(1, Ordering::Relaxed) % clients.len();
        let mut client = clients[idx].clone();
        let req = AbortRequest {
            request_id: ctx.id().to_string(),
            reason: "client_cancelled".to_string(),
        };
        if let Err(e) = client.abort(req).await {
            tracing::warn!(request_id = ctx.id(), error = %e, "abort RPC failed");
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        Ok(())
    }

    /// One `KvEventSource::Zmq` per DP rank, subscribing to SGLang's
    /// scheduler ZMQ PUB. Mirrors `dynamo.sglang`'s `init_kv_event_publish`:
    /// same `port + dp_rank` offset, same loopback rewrite. Empty when
    /// SGLang was launched without `--kv-events-config`.
    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        let Some(kv) = self.kv_events.get() else {
            return Ok(Vec::new());
        };
        let dp_size = self.dp_size.get().copied().unwrap_or(1);

        let mut sources = Vec::with_capacity(dp_size as usize);
        for dp_rank in 0..dp_size {
            let Some(endpoint) = offset_endpoint_port(&kv.endpoint, dp_rank) else {
                tracing::warn!(base = %kv.endpoint, dp_rank, "offset_endpoint_port failed");
                continue;
            };
            let endpoint = endpoint.replacen("*", "127.0.0.1", 1);
            tracing::debug!(%endpoint, dp_rank, "kv_events subscriber");
            sources.push(KvEventSource::Zmq {
                endpoint,
                topic: kv.topic.clone(),
                dp_rank,
            });
        }
        Ok(sources)
    }
}

/// Canary `PreprocessedRequest` shape sent by the runtime's health-check
/// manager (when `DYN_HEALTH_CHECK_ENABLED=true`) through the normal
/// `generate()` path. Mirrors `components/src/dynamo/sglang/health_check.py`'s
/// `SglangHealthCheckPayload`: one input token (BOS fallback), one output
/// token, greedy, no EOS shortcut. We don't query SGLang for the real BOS
/// token id — SGLang's scheduler accepts any valid token for a 1-step
/// generate, and token `1` is the BOS for every Qwen/Llama/DeepSeek family
/// we care about.
fn build_health_check_payload() -> serde_json::Value {
    serde_json::json!({
        "token_ids": [1],
        "stop_conditions": {"max_tokens": 1, "ignore_eos": false},
        "sampling_options": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
        "eos_token_ids": [],
        "annotations": [],
    })
}

/// Classify a `tonic::Status` from the startup `HealthCheck` loop.
///
/// Transient (keep waiting): `Unavailable` covers tonic's "channel not
/// connected yet / peer reset / temporarily refused" cases. SGLang's gRPC
/// server may bind its port early but reject calls until the scheduler is
/// up, which surfaces as Unavailable.
///
/// Fatal (give up): everything else is either a proto mismatch or a real
/// engine failure that retrying won't fix.
fn is_transient_health_error(status: &tonic::Status) -> bool {
    matches!(status.code(), tonic::Code::Unavailable)
}

async fn connect_client(endpoint: &str) -> Result<SglangSchedulerClient<Channel>, DynamoError> {
    let channel = Endpoint::from_shared(endpoint.to_string())
        .map_err(|e| invalid_arg(format!("invalid grpc endpoint: {e}")))?
        .connect_timeout(CONNECT_TIMEOUT)
        .connect()
        .await
        .map_err(|e| backend_error(format!("connect {endpoint}: {e}")))?;
    Ok(SglangSchedulerClient::new(channel))
}

fn grpc_channel_count_from_env() -> usize {
    std::env::var(GRPC_CHANNELS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(DEFAULT_GRPC_CHANNELS)
}

fn request_dp_rank(request: &PreprocessedRequest, prefill: bool) -> i32 {
    let rank = request.routing.as_ref().and_then(|routing| {
        if prefill {
            routing.prefill_dp_rank.or(routing.dp_rank)
        } else {
            routing.dp_rank
        }
    });
    match rank {
        Some(rank) => i32::try_from(rank).unwrap_or_else(|_| {
            tracing::warn!(rank, "dp_rank exceeds SMG int32 range; forwarding rank 0");
            0
        }),
        None => 0,
    }
}

fn map_smg_bootstrap_room(
    room: u64,
    dp_size: u32,
    compat_enabled: bool,
) -> Result<i32, DynamoError> {
    if room <= SMG_BOOTSTRAP_ROOM_MAX {
        return Ok(room as i32);
    }
    if !compat_enabled {
        return Err(invalid_arg(format!(
            "bootstrap_room {room} exceeds SMG int32 range; set {SMG_BOOTSTRAP_ROOM_ENV}=1 \
             to remap Dynamo rooms for SMG v1.5"
        )));
    }

    let mapped = remap_bootstrap_room_preserving_dp_rank(room, dp_size)?;
    Ok(mapped as i32)
}

fn remap_bootstrap_room_preserving_dp_rank(room: u64, dp_size: u32) -> Result<u64, DynamoError> {
    if room == 0 {
        return Ok(0);
    }

    let dp_size = u64::from(dp_size.max(1));
    if dp_size > SMG_BOOTSTRAP_ROOM_MAX {
        return Err(invalid_arg(format!(
            "dp_size {dp_size} exceeds SMG bootstrap room range"
        )));
    }

    let rank = room % dp_size;
    let slots = ((SMG_BOOTSTRAP_ROOM_MAX - rank) / dp_size) + 1;
    let quotient = (room / dp_size) % slots;
    let mapped = quotient * dp_size + rank;
    debug_assert!(mapped <= SMG_BOOTSTRAP_ROOM_MAX);
    debug_assert_eq!(mapped % dp_size, room % dp_size);
    Ok(mapped)
}

fn truthy_env(name: &str) -> bool {
    let Ok(value) = std::env::var(name) else {
        return false;
    };
    matches!(
        value.to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn local_bootstrap_host() -> Option<String> {
    std::env::var("SGLANG_HOST_IP")
        .ok()
        .or_else(|| std::env::var("HOST_IP").ok())
        .and_then(|host| normalize_host(host.trim()))
        .or_else(|| local_ip_address::local_ip().ok().map(format_ip_host))
        .or_else(|| local_ip_address::local_ipv6().ok().map(format_ip_host))
}

fn advertised_bootstrap_host(
    reported_host: Option<String>,
    detected_local_host: Option<String>,
) -> String {
    let detected = detected_local_host.and_then(|host| normalize_host(host.trim()));
    if let Some(reported) = reported_host.and_then(|host| normalize_host(host.trim())) {
        if !is_non_routable_host(&reported) {
            return reported;
        }
        if let Some(local) = detected.filter(|host| !is_non_routable_host(host)) {
            tracing::warn!(
                reported_bootstrap_host = %reported,
                advertised_bootstrap_host = %local,
                "SGLang reported a non-routable bootstrap host; advertising detected local host"
            );
            return local;
        }
        return reported;
    }

    detected.unwrap_or_else(|| "127.0.0.1".to_string())
}

fn normalize_host(host: &str) -> Option<String> {
    let host = host.trim();
    if host.is_empty() {
        return None;
    }
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    match host.parse() {
        Ok(ip) => Some(format_ip_host(ip)),
        Err(_) => Some(host.to_string()),
    }
}

fn format_ip_host(ip: std::net::IpAddr) -> String {
    match ip {
        std::net::IpAddr::V4(_) => ip.to_string(),
        std::net::IpAddr::V6(_) => format!("[{ip}]"),
    }
}

fn is_non_routable_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_unspecified() || ip.is_loopback(),
        Err(_) => false,
    }
}

fn invalid_arg(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(msg)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smg_bootstrap_room_accepts_int32_range_without_compat() {
        assert_eq!(map_smg_bootstrap_room(0, 1, false).unwrap(), 0);
        assert_eq!(
            map_smg_bootstrap_room(SMG_BOOTSTRAP_ROOM_MAX, 1, false).unwrap(),
            i32::MAX
        );
    }

    #[test]
    fn smg_bootstrap_room_requires_compat_for_large_rooms() {
        assert!(map_smg_bootstrap_room(SMG_BOOTSTRAP_ROOM_MAX + 1, 1, false).is_err());
    }

    #[test]
    fn smg_bootstrap_room_compat_preserves_dp_rank_modulo() {
        let room = i64::MAX as u64 - 17;
        let dp_size = 8;
        let mapped = map_smg_bootstrap_room(room, dp_size, true).unwrap() as u64;
        assert!(mapped <= SMG_BOOTSTRAP_ROOM_MAX);
        assert_eq!(mapped % u64::from(dp_size), room % u64::from(dp_size));
    }

    #[test]
    fn backend_args_set_worker_model_name_for_mdc() {
        let (_engine, config) = SglangBackend::from_argv(vec![
            "dynamo-sglang-rs-backend".to_string(),
            "--model-name".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
        ])
        .unwrap();

        assert_eq!(config.model_name, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn backend_args_set_served_model_name_for_registration() {
        let (_engine, config) = SglangBackend::from_argv(vec![
            "dynamo-sglang-rs-backend".to_string(),
            "--model-name".to_string(),
            "/models/qwen".to_string(),
            "--served-model-name".to_string(),
            "qwen-public".to_string(),
        ])
        .unwrap();

        assert_eq!(config.model_name, "/models/qwen");
        assert_eq!(config.served_model_name.as_deref(), Some("qwen-public"));
    }

    #[test]
    fn backend_args_set_worker_disaggregation_mode() {
        let (_engine, config) = SglangBackend::from_argv(vec![
            "dynamo-sglang-rs-backend".to_string(),
            "--disaggregation-mode".to_string(),
            "prefill".to_string(),
        ])
        .unwrap();

        assert_eq!(config.disaggregation_mode, DisaggregationMode::Prefill);

        let (_engine, config) = SglangBackend::from_argv(vec![
            "dynamo-sglang-rs-backend".to_string(),
            "--disaggregation-mode".to_string(),
            "decode".to_string(),
        ])
        .unwrap();

        assert_eq!(config.disaggregation_mode, DisaggregationMode::Decode);
    }

    #[test]
    fn bootstrap_host_prefers_reported_routable_host() {
        assert_eq!(
            advertised_bootstrap_host(Some("10.0.0.8".to_string()), Some("10.0.0.9".to_string())),
            "10.0.0.8"
        );
    }

    #[test]
    fn bootstrap_host_replaces_unspecified_or_loopback_with_local_host() {
        for reported in ["0.0.0.0", "127.0.0.1", "localhost", "::", "[::1]"] {
            assert_eq!(
                advertised_bootstrap_host(Some(reported.to_string()), Some("10.0.0.9".to_string())),
                "10.0.0.9"
            );
        }
    }

    #[test]
    fn bootstrap_host_uses_detected_host_when_sglang_reports_none() {
        assert_eq!(
            advertised_bootstrap_host(None, Some("10.0.0.9".to_string())),
            "10.0.0.9"
        );
    }

    #[test]
    fn bootstrap_host_formats_ipv6_with_brackets() {
        assert_eq!(
            advertised_bootstrap_host(Some("2001:db8::1".to_string()), None),
            "[2001:db8::1]"
        );
        assert_eq!(
            advertised_bootstrap_host(None, Some("2001:db8::2".to_string())),
            "[2001:db8::2]"
        );
    }
}

fn backend_error(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::EngineShutdown))
        .message(msg)
        .build()
}
