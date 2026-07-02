// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridge to SGLang's native gRPC server (`sglang.runtime.v1`). Forwards
//! `PreprocessedRequest` to SGLang's `Generate` RPC and streams tokens back.

use std::sync::Arc;
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
use crate::proto::v1::{
    AbortRequest, DisaggregatedParams, GenerateRequest, GetModelInfoRequest, GetServerInfoRequest,
    HealthCheckRequest, sglang_service_client::SglangServiceClient,
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
/// failures so ops can see the bridge is alive and waiting on SGLang.
const HEALTH_LOG_EVERY_N_ATTEMPTS: u32 = 20;

pub struct SglangBridge {
    grpc_endpoint: String,
    /// Discovered from SGLang's `GetServerInfo` in `start()`; read by
    /// `generate()` to dispatch prefill vs decode.
    disaggregation_mode: OnceCell<DisaggregationMode>,
    bootstrap: OnceCell<(String, u16)>,
    client: OnceCell<SglangServiceClient<Channel>>,
    /// Unset → KV routing off; populated in `start()` from `GetServerInfo`.
    kv_events: OnceCell<KvEventsConfig>,
    /// SGLang's `dp_size`; 1 when DP attention is off (the common case).
    dp_size: OnceCell<u32>,
}

impl SglangBridge {
    /// Parse from process args (the binary entry point).
    pub fn from_env() -> Result<(Self, WorkerConfig), DynamoError> {
        Self::build(<Args as clap::Parser>::try_parse())
    }

    /// Parse from an explicit argv (the PyO3 entry point).
    pub fn from_argv(argv: Vec<String>) -> Result<(Self, WorkerConfig), DynamoError> {
        Self::build(<Args as clap::Parser>::try_parse_from(argv))
    }

    fn build(parsed: Result<Args, clap::Error>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = parsed.map_err(|e| invalid_arg(e.to_string()))?;
        let engine = SglangBridge {
            grpc_endpoint: args.sglang_grpc_endpoint,
            disaggregation_mode: OnceCell::new(),
            bootstrap: OnceCell::new(),
            client: OnceCell::new(),
            kv_events: OnceCell::new(),
            dp_size: OnceCell::new(),
        };
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            disaggregation_mode: args.common.disaggregation_mode,
            route_to_encoder: args.common.route_to_encoder,
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
        let grpc_req = GenerateRequest {
            input_ids: request.token_ids.iter().map(|&t| t as i32).collect(),
            sampling_params: Some(build_sampling_params(
                &request.sampling_options,
                &request.stop_conditions,
            )),
            stream: Some(true),
            rid: Some(ctx.id().to_string()),
            disaggregated_params: request
                .bootstrap_info
                .as_ref()
                .map(|bi| DisaggregatedParams {
                    bootstrap_host: bi.bootstrap_host.clone(),
                    bootstrap_port: bi.bootstrap_port.into(),
                    bootstrap_room: bi.bootstrap_room as i64,
                }),
            ..Default::default()
        };
        let mut stream = client
            .generate(grpc_req)
            .await
            .map_err(|e| backend_error(format!("Generate RPC: {e}")))?
            .into_inner();

        Ok(Box::pin(async_stream::stream! {
            let mut completion_tokens: u32 = 0;
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
                        // Terminal chunks may carry the final token(s); flush before the terminal.
                        if !resp.output_ids.is_empty() {
                            completion_tokens = completion_tokens.saturating_add(resp.output_ids.len() as u32);
                            for tid in resp.output_ids {
                                yield Ok(chunk::token(tid as u32));
                            }
                        }
                        if !resp.finished {
                            continue;
                        }
                        let finish = parse_finish_reason(
                            resp.meta_info.get("finish_reason").map(String::as_str).unwrap_or(""),
                        );
                        let total = resp.meta_info.get("completion_tokens")
                            .and_then(|s| s.parse::<u32>().ok())
                            .unwrap_or(completion_tokens);
                        let terminal = match finish {
                            FinishReason::Length => LLMEngineOutput::length(),
                            FinishReason::Cancelled => LLMEngineOutput::cancelled(),
                            FinishReason::Error(msg) => LLMEngineOutput::error(msg),
                            _ => LLMEngineOutput::stop(),
                        };
                        yield Ok(terminal.with_usage(usage(prompt_len, total)));
                        break;
                    }
                }
            }
        }))
    }

    /// Mirrors `components/src/dynamo/sglang/request_handlers/llm/prefill_handler.py`:
    /// open the Generate RPC once, yield ONE non-terminal bootstrap chunk
    /// synchronously, then drain the gRPC stream in-band (inside the same
    /// async_stream) until SGLang signals `finished` — only then close the
    /// outer stream with a terminal Stop chunk.
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
        // Honour a router-supplied room (carries dp_rank), else mint a fresh 63-bit room.
        let bootstrap_room: i64 = request
            .bootstrap_info
            .as_ref()
            .map(|bi| bi.bootstrap_room as i64)
            .unwrap_or_else(|| rand::rng().random_range(1..i64::MAX));

        let grpc_req = GenerateRequest {
            input_ids: request.token_ids.iter().map(|&t| t as i32).collect(),
            sampling_params: Some(build_sampling_params(
                &request.sampling_options,
                &request.stop_conditions,
            )),
            stream: Some(true),
            rid: Some(ctx.id().to_string()),
            disaggregated_params: Some(DisaggregatedParams {
                bootstrap_host: bootstrap_host.clone(),
                bootstrap_port: bootstrap_port.into(),
                bootstrap_room,
            }),
            ..Default::default()
        };
        let bootstrap_chunk = LLMEngineOutput {
            disaggregated_params: Some(serde_json::json!({
                "bootstrap_host": bootstrap_host,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": bootstrap_room,
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

            // Drain the stream until SGLang signals `finished`. SGLang prefill
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
                        Some(Ok(r)) if r.finished => {
                            yield Ok(LLMEngineOutput::stop());
                            return;
                        }
                        // Non-terminal chunks from prefill carry no tokens we
                        // need to forward (the bootstrap triple was already
                        // emitted); discard them.
                        Some(Ok(_)) => continue,
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

    fn client_or_err(&self) -> Result<SglangServiceClient<Channel>, DynamoError> {
        self.client
            .get()
            .ok_or_else(|| backend_error("generate called before start"))
            .cloned()
    }
}

#[async_trait]
impl LLMEngine for SglangBridge {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        let channel = Endpoint::from_shared(self.grpc_endpoint.clone())
            .map_err(|e| invalid_arg(format!("invalid grpc endpoint: {e}")))?
            .connect_timeout(CONNECT_TIMEOUT)
            .connect()
            .await
            .map_err(|e| backend_error(format!("connect {}: {e}", self.grpc_endpoint)))?;
        let mut client = SglangServiceClient::new(channel);

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
        let info = parse_server_info(&server_info.json_info);
        let mode = info.disaggregation_mode.unwrap_or_default();
        let _ = self.disaggregation_mode.set(mode);

        let model_path = if !model_info.model_path.is_empty() {
            model_info.model_path
        } else {
            info.model_path.clone().unwrap_or_default()
        };
        if model_path.is_empty() {
            return Err(backend_error("SGLang reported empty model_path"));
        }
        let served_model_name = info.served_model_name.unwrap_or_else(|| model_path.clone());

        if mode.is_prefill() {
            let port = info.bootstrap_port.ok_or_else(|| {
                backend_error("GetServerInfo.disaggregation_bootstrap_port missing")
            })?;
            let host = info
                .bootstrap_host
                .unwrap_or_else(|| "127.0.0.1".to_string());
            let _ = self.bootstrap.set((host, port));
        }

        self.client
            .set(client)
            .map_err(|_| backend_error("client already set"))?;

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
            "sglang_bridge connected"
        );

        let context_length = serde_json::from_str::<serde_json::Value>(&model_info.json_info)
            .ok()
            .and_then(|v| {
                v.get("max_context_length")
                    .or_else(|| v.get("context_length"))
                    .and_then(|n| n.as_u64())
            })
            .filter(|&n| n > 0 && n <= u32::MAX as u64)
            .map(|n| n as u32);

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
        let Some(client) = self.client.get() else {
            return;
        };
        let req = AbortRequest {
            rid: ctx.id().to_string(),
            abort_all: false,
        };
        if let Err(e) = client.clone().abort(req).await {
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

fn invalid_arg(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(msg)
        .build()
}

fn backend_error(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::EngineShutdown))
        .message(msg)
        .build()
}
