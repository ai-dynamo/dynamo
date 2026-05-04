// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sidecar bridge backend: registers as a Dynamo worker and forwards each
//! request to a SGLang `sglang.grpc.scheduler` gRPC server (the schema in
//! upstream main, activated via `python -m sglang.launch_server --grpc-mode`).
//!
//! Token deltas are incremental in this schema (see `GenerateStreamChunk` —
//! `token_ids` is "incremental chunk" per the proto comment), and finish_reason
//! is a typed string in `GenerateComplete`. Disagg PD bootstrap fields are
//! first-class via `DisaggregatedParams`. KV events are available as a separate
//! server-streaming RPC (`SubscribeKvEvents`); not wired up in this POC.
//!
//! ## Disaggregated serving
//!
//! Mirrors the Python `dynamo.sglang` flow (see `prefill_handler.py`):
//!
//! - **Prefill** worker (`--disaggregation-mode prefill`): registers with
//!   `endpoint_types = "prefill"`. On `start()` it pulls
//!   `disaggregation_bootstrap_port` and `dist_init_addr` from SGLang's
//!   `GetServerInfo.server_args`. On each `generate()` it generates a 31-bit
//!   random `bootstrap_room`, yields one `LLMEngineOutput` chunk with
//!   `disaggregated_params` populated (the frontend captures this and forwards
//!   to the decode worker), then issues the gRPC `Generate` with
//!   `DisaggregatedParams` set and silently drains the server stream — the
//!   prefill output is consumed via NIXL/Mooncake KV transfer, not via gRPC.
//!
//! - **Decode** worker (`--disaggregation-mode decode`, the default agg path
//!   also): pulls `request.bootstrap_info` if present and forwards it via the
//!   proto's `DisaggregatedParams`. Streams generated tokens normally.
//!
//! Rooms are 31-bit so they round-trip safely through the proto's `int32`
//! field. The frontend stores `bootstrap_room` as `u64` but the same low bits
//! reach both prefill and decode legs, so the rendezvous matches.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, CommonArgs, DynamoError, EngineConfig, ErrorType,
    FinishReason, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest,
    WorkerConfig, chunk, usage,
};
use futures::stream::BoxStream;
use rand::Rng;
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;
use tonic::transport::{Channel, Endpoint};

use crate::proto::scheduler::{
    AbortRequest, DisaggregatedParams, GenerateRequest, GetModelInfoRequest, GetServerInfoRequest,
    HealthCheckRequest, SamplingParams, TokenizedInput,
    generate_response::Response as GenResponse, sglang_scheduler_client::SglangSchedulerClient,
};

#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DisaggMode {
    /// Aggregated serving — single worker handles prefill + decode (default).
    #[default]
    None,
    /// Prefill-only worker. Registers with `endpoint_types=prefill` and yields
    /// bootstrap info as the first chunk per request.
    Prefill,
    /// Decode-only worker. Forwards inbound `bootstrap_info` to SGLang as
    /// `DisaggregatedParams`.
    Decode,
}

#[derive(clap::Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Dynamo sidecar bridge to upstream SGLang's gRPC scheduler (sglang.grpc.scheduler)."
)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    /// gRPC endpoint of the SGLang server. SGLang in `--grpc-mode` listens on
    /// the `--port` value (default 30000) over gRPC instead of HTTP.
    #[arg(long, env = "SGLANG_GRPC_ENDPOINT", default_value = "http://127.0.0.1:30000")]
    pub sglang_grpc_endpoint: String,

    /// HF repo or local path. Used by Dynamo to load tokenizer + chat template.
    #[arg(long)]
    pub model_path: String,

    /// Friendly model name advertised to clients. Defaults to `model_path`.
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Connect timeout for the initial gRPC dial, seconds.
    #[arg(long, default_value_t = 30)]
    pub connect_timeout_secs: u64,

    /// Disaggregation mode. `prefill` registers as a prefill-only worker and
    /// yields bootstrap info on each request; `decode` forwards inbound
    /// bootstrap info to SGLang; `none` (default) is aggregated serving.
    /// Should match the SGLang server's `--disaggregation-mode`.
    #[arg(long, value_enum, default_value_t = DisaggMode::None,
          env = "DYN_DISAGGREGATION_MODE")]
    pub disaggregation_mode: DisaggMode,

    /// Bootstrap host advertised to decode workers. Falls back to
    /// `dist_init_addr` from SGLang's `GetServerInfo`, then to `127.0.0.1`.
    /// In multi-host deployments, set this to the prefill Pod's reachable
    /// address.
    #[arg(long, env = "DYN_BOOTSTRAP_HOST")]
    pub bootstrap_host: Option<String>,
}

pub struct SglangBridge {
    grpc_endpoint: String,
    served_model_name: String,
    connect_timeout_secs: u64,
    disaggregation_mode: DisaggMode,
    bootstrap_host_override: Option<String>,
    /// Resolved on `start()` from SGLang's `GetServerInfo` (prefill workers only).
    bootstrap_host: OnceCell<String>,
    bootstrap_port: OnceCell<u16>,
    client: OnceCell<SglangSchedulerClient<Channel>>,
}

impl SglangBridge {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(a) => <Args as clap::Parser>::try_parse_from(a),
            None => <Args as clap::Parser>::try_parse(),
        }
        .map_err(|e| invalid_arg(e.to_string()))?;

        let served = args
            .served_model_name
            .clone()
            .unwrap_or_else(|| args.model_path.clone());

        // Prefill workers register with endpoint_types=prefill so the
        // frontend's prefill/decode routers can distinguish them. Decode
        // workers use whatever the operator passed (default chat,completions).
        let endpoint_types = match args.disaggregation_mode {
            DisaggMode::Prefill => "prefill".to_string(),
            DisaggMode::Decode | DisaggMode::None => args.common.endpoint_types,
        };

        let engine = SglangBridge {
            grpc_endpoint: args.sglang_grpc_endpoint,
            served_model_name: served.clone(),
            connect_timeout_secs: args.connect_timeout_secs,
            disaggregation_mode: args.disaggregation_mode,
            bootstrap_host_override: args.bootstrap_host,
            bootstrap_host: OnceCell::new(),
            bootstrap_port: OnceCell::new(),
            client: OnceCell::new(),
        };
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            model_name: args.model_path,
            served_model_name: Some(served),
            ..Default::default()
        };
        Ok((engine, config))
    }
}

fn build_sampling_params(req: &PreprocessedRequest) -> SamplingParams {
    SamplingParams {
        temperature: req.sampling_options.temperature.unwrap_or(0.0),
        top_p: req.sampling_options.top_p.unwrap_or(0.0),
        top_k: req.sampling_options.top_k.unwrap_or(0),
        min_p: req.sampling_options.min_p.unwrap_or(0.0),
        frequency_penalty: req.sampling_options.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.sampling_options.presence_penalty.unwrap_or(0.0),
        repetition_penalty: req.sampling_options.repetition_penalty.unwrap_or(1.0),
        max_new_tokens: req.stop_conditions.max_tokens,
        stop: req.stop_conditions.stop.clone().unwrap_or_default(),
        stop_token_ids: req
            .stop_conditions
            .stop_token_ids_hidden
            .clone()
            .unwrap_or_default(),
        skip_special_tokens: false,
        spaces_between_special_tokens: false,
        n: req.sampling_options.n.map(|v| v as u32).unwrap_or(1),
        min_new_tokens: req.stop_conditions.min_tokens.unwrap_or(0),
        ignore_eos: req.stop_conditions.ignore_eos.unwrap_or(false),
        no_stop_trim: false,
        stream_interval: None,
        logit_bias: Default::default(),
        custom_params: None,
        constraint: None,
    }
}

fn parse_finish_reason(raw: &str) -> FinishReason {
    // Legacy proto: GenerateComplete.finish_reason is an OpenAI-style string.
    match raw {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "abort" | "cancelled" => FinishReason::Cancelled,
        "" => FinishReason::Stop, // empty defaults to stop
        other => FinishReason::Error(format!("unknown sglang finish_reason: {other}")),
    }
}

/// Pull `disaggregation_bootstrap_port` and `dist_init_addr` out of SGLang's
/// `GetServerInfoResponse.server_args` Struct. SGLang ships ServerArgs as a
/// `google.protobuf.Struct` over the wire so every numeric field arrives as
/// `Kind::NumberValue(f64)` — we narrow back to i32 here.
fn extract_bootstrap_from_server_args(
    server_args: Option<&prost_types::Struct>,
) -> (Option<u16>, Option<String>) {
    use prost_types::value::Kind;
    let Some(s) = server_args else {
        return (None, None);
    };
    let port = s
        .fields
        .get("disaggregation_bootstrap_port")
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| match k {
            Kind::NumberValue(n) => Some(*n as u16),
            _ => None,
        });
    let host = s
        .fields
        .get("dist_init_addr")
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| match k {
            Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
        // SGLang stores dist_init_addr as "host:port"; bootstrap uses the host
        // half only. Strip the trailing :port if present.
        .map(|addr| addr.rsplit_once(':').map(|(h, _)| h.to_string()).unwrap_or(addr));
    (port, host)
}

#[async_trait]
impl LLMEngine for SglangBridge {
    async fn start(&self) -> Result<EngineConfig, DynamoError> {
        let endpoint = Endpoint::from_shared(self.grpc_endpoint.clone())
            .map_err(|e| invalid_arg(format!("invalid grpc endpoint: {e}")))?
            .connect_timeout(Duration::from_secs(self.connect_timeout_secs));
        let channel = endpoint
            .connect()
            .await
            .map_err(|e| backend_error(format!("connect {}: {e}", self.grpc_endpoint)))?;
        let mut client = SglangSchedulerClient::new(channel);

        let health = client
            .health_check(HealthCheckRequest {})
            .await
            .map_err(|e| backend_error(format!("HealthCheck: {e}")))?
            .into_inner();
        if !health.healthy {
            return Err(backend_error(format!(
                "SGLang reported unhealthy: {}",
                health.message
            )));
        }

        let info = client
            .get_model_info(GetModelInfoRequest {})
            .await
            .map_err(|e| backend_error(format!("GetModelInfo: {e}")))?
            .into_inner();

        let context_length = if info.max_context_length > 0 {
            Some(info.max_context_length as u32)
        } else {
            None
        };

        // Prefill workers need bootstrap host/port to advertise to decode
        // workers via the per-request bootstrap chunk. Pull from
        // GetServerInfo.server_args once at startup.
        if matches!(self.disaggregation_mode, DisaggMode::Prefill) {
            let server_info = client
                .get_server_info(GetServerInfoRequest {})
                .await
                .map_err(|e| backend_error(format!("GetServerInfo: {e}")))?
                .into_inner();
            let (port, host_from_args) =
                extract_bootstrap_from_server_args(server_info.server_args.as_ref());
            let port = port.ok_or_else(|| {
                backend_error(
                    "GetServerInfo.server_args.disaggregation_bootstrap_port missing — \
                     is SGLang launched with `--disaggregation-mode prefill`?",
                )
            })?;
            let host = self
                .bootstrap_host_override
                .clone()
                .or(host_from_args)
                .unwrap_or_else(|| "127.0.0.1".to_string());
            tracing::info!(
                bootstrap_host = %host,
                bootstrap_port = port,
                "sglang_bridge prefill mode: bootstrap rendezvous"
            );
            let _ = self.bootstrap_host.set(host);
            let _ = self.bootstrap_port.set(port);
        }

        self.client
            .set(client)
            .map_err(|_| backend_error("client already set"))?;

        tracing::info!(
            endpoint = %self.grpc_endpoint,
            model = %self.served_model_name,
            context_length = ?context_length,
            sglang_model_path = %info.model_path,
            disagg_mode = ?self.disaggregation_mode,
            "sglang_bridge: connected"
        );

        Ok(EngineConfig {
            model: self.served_model_name.clone(),
            served_model_name: Some(self.served_model_name.clone()),
            context_length,
            // KV-cache hints intentionally omitted for now: KvRouter falls back
            // to round-robin. Wire `SubscribeKvEvents` in a follow-up to enable
            // KV-aware routing.
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, LLMEngineOutput>, DynamoError> {
        match self.disaggregation_mode {
            DisaggMode::Prefill => self.generate_prefill(request, ctx).await,
            DisaggMode::Decode | DisaggMode::None => self.generate_decode(request, ctx).await,
        }
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(client) = self.client.get() else { return };
        let mut client = client.clone();
        let req = AbortRequest {
            request_id: ctx.id().to_string(),
            reason: "client cancelled".to_string(),
        };
        if let Err(e) = client.abort(req).await {
            tracing::warn!(request_id = ctx.id(), err = %e, "sglang_bridge abort RPC failed");
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        Ok(())
    }
}

impl SglangBridge {
    /// Decode and aggregated paths share this shape: forward request to
    /// SGLang, stream `GenerateStreamChunk` tokens, emit a typed terminal on
    /// `GenerateComplete`. Decode workers additionally fill
    /// `DisaggregatedParams` from inbound `request.bootstrap_info` so SGLang
    /// pulls KV from the prefill rendezvous.
    async fn generate_decode(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, LLMEngineOutput>, DynamoError> {
        let mut client = self
            .client
            .get()
            .ok_or_else(|| backend_error("generate called before start"))?
            .clone();

        let prompt_len = request.token_ids.len() as u32;
        let input_ids: Vec<u32> = request.token_ids.clone();
        let sampling = build_sampling_params(&request);

        // Decode side: pull bootstrap_info forwarded from the prefill leg.
        let disagg = request.bootstrap_info.as_ref().map(|bi| DisaggregatedParams {
            bootstrap_host: bi.bootstrap_host.clone(),
            bootstrap_port: bi.bootstrap_port as i32,
            bootstrap_room: bi.bootstrap_room as i32,
        });

        let grpc_req = GenerateRequest {
            request_id: ctx.id().to_string(),
            tokenized: Some(TokenizedInput {
                original_text: String::new(),
                input_ids,
            }),
            mm_inputs: None,
            sampling_params: Some(sampling),
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: vec![],
            return_hidden_states: false,
            disaggregated_params: disagg,
            custom_logit_processor: String::new(),
            timestamp: None,
            log_metrics: false,
            input_embeds: vec![],
            lora_id: String::new(),
            data_parallel_rank: -1,
            stream: true,
        };

        let response = client
            .generate(grpc_req)
            .await
            .map_err(|e| backend_error(format!("Generate RPC: {e}")))?;
        let mut stream = response.into_inner();

        Ok(Box::pin(async_stream::stream! {
            // GenerateStreamChunk.token_ids is the *incremental chunk* per
            // proto comment — no slicing needed (vs Phase-1 cumulative shape).
            let mut completion_tokens: u32 = 0;
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield LLMEngineOutput::cancelled()
                            .with_usage(usage(prompt_len, completion_tokens));
                        break;
                    }
                    maybe_chunk = stream.next() => {
                        let Some(chunk_res) = maybe_chunk else {
                            yield LLMEngineOutput::error(
                                "sglang_bridge: stream ended without terminal".to_string()
                            );
                            break;
                        };
                        let resp = match chunk_res {
                            Ok(r) => r,
                            Err(status) => {
                                yield LLMEngineOutput::error(
                                    format!("sglang_bridge: gRPC stream error: {status}")
                                );
                                break;
                            }
                        };

                        match resp.response {
                            Some(GenResponse::Chunk(stream_chunk)) => {
                                completion_tokens = stream_chunk.completion_tokens;
                                for tid in stream_chunk.token_ids {
                                    yield chunk::token(tid);
                                }
                            }
                            Some(GenResponse::Complete(complete)) => {
                                let finish = parse_finish_reason(&complete.finish_reason);
                                let mut terminal = match finish {
                                    FinishReason::Length => LLMEngineOutput::length(),
                                    FinishReason::Cancelled => LLMEngineOutput::cancelled(),
                                    FinishReason::Error(msg) => LLMEngineOutput::error(msg),
                                    _ => LLMEngineOutput::stop(),
                                };
                                terminal.token_ids = vec![];
                                let total = complete.completion_tokens.max(completion_tokens);
                                terminal = terminal.with_usage(usage(prompt_len, total));
                                yield terminal;
                                break;
                            }
                            Some(GenResponse::Error(err)) => {
                                yield LLMEngineOutput::error(format!(
                                    "sglang_bridge: {} ({})",
                                    err.message, err.http_status_code
                                ));
                                break;
                            }
                            None => {
                                yield LLMEngineOutput::error(
                                    "sglang_bridge: empty oneof response".to_string()
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    /// Prefill path. Generates a 31-bit `bootstrap_room`, yields one chunk
    /// carrying `disaggregated_params: {host, port, room}` for the frontend's
    /// prefill router to capture and forward to the decode worker. Then
    /// issues `Generate` to SGLang (which writes KV to the rendezvous) and
    /// drains the response stream silently — prefill output is consumed via
    /// NIXL/Mooncake on the decode side, not via gRPC.
    async fn generate_prefill(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, LLMEngineOutput>, DynamoError> {
        let mut client = self
            .client
            .get()
            .ok_or_else(|| backend_error("generate called before start"))?
            .clone();

        let bootstrap_host = self
            .bootstrap_host
            .get()
            .ok_or_else(|| backend_error("prefill: bootstrap_host not initialised"))?
            .clone();
        let bootstrap_port = *self
            .bootstrap_port
            .get()
            .ok_or_else(|| backend_error("prefill: bootstrap_port not initialised"))?;

        // Honor a router-provided room if the frontend already chose one;
        // otherwise generate a fresh 31-bit room (positive i32 for proto wire,
        // round-trips via u64 in PreprocessedRequest.bootstrap_info).
        let bootstrap_room: i32 = request
            .bootstrap_info
            .as_ref()
            .map(|bi| bi.bootstrap_room as i32)
            .unwrap_or_else(|| rand::thread_rng().gen_range(1..i32::MAX));

        let prompt_len = request.token_ids.len() as u32;
        let input_ids: Vec<u32> = request.token_ids.clone();
        let sampling = build_sampling_params(&request);

        let disagg = DisaggregatedParams {
            bootstrap_host: bootstrap_host.clone(),
            bootstrap_port: bootstrap_port as i32,
            bootstrap_room,
        };

        let grpc_req = GenerateRequest {
            request_id: ctx.id().to_string(),
            tokenized: Some(TokenizedInput {
                original_text: String::new(),
                input_ids,
            }),
            mm_inputs: None,
            sampling_params: Some(sampling),
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: vec![],
            return_hidden_states: false,
            disaggregated_params: Some(disagg),
            custom_logit_processor: String::new(),
            timestamp: None,
            log_metrics: false,
            input_embeds: vec![],
            lora_id: String::new(),
            data_parallel_rank: -1,
            stream: true,
        };

        let response = client
            .generate(grpc_req)
            .await
            .map_err(|e| backend_error(format!("Generate RPC: {e}")))?;
        let mut stream = response.into_inner();

        // First chunk: bootstrap info for the frontend to forward to decode.
        let mut bootstrap_chunk = LLMEngineOutput::default();
        bootstrap_chunk.disaggregated_params = Some(serde_json::json!({
            "bootstrap_host": bootstrap_host,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room as u64,
        }));

        Ok(Box::pin(async_stream::stream! {
            yield bootstrap_chunk;

            // Drain SGLang's stream silently. Prefill tokens are consumed by
            // the decode worker via NIXL/Mooncake, not by us. We only watch
            // for errors and the terminal. Honor cancellation along the way.
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield LLMEngineOutput::cancelled()
                            .with_usage(usage(prompt_len, 0));
                        break;
                    }
                    maybe_chunk = stream.next() => {
                        let Some(chunk_res) = maybe_chunk else {
                            // Stream closed without Complete — treat as stop.
                            yield LLMEngineOutput::stop().with_usage(usage(prompt_len, 0));
                            break;
                        };
                        match chunk_res {
                            Ok(resp) => match resp.response {
                                Some(GenResponse::Chunk(_)) => {
                                    // Drop — decode worker reads from KV rendezvous.
                                }
                                Some(GenResponse::Complete(_)) => {
                                    yield LLMEngineOutput::stop().with_usage(usage(prompt_len, 0));
                                    break;
                                }
                                Some(GenResponse::Error(err)) => {
                                    yield LLMEngineOutput::error(format!(
                                        "sglang_bridge prefill: {} ({})",
                                        err.message, err.http_status_code
                                    ));
                                    break;
                                }
                                None => {
                                    yield LLMEngineOutput::error(
                                        "sglang_bridge prefill: empty oneof response".to_string()
                                    );
                                    break;
                                }
                            },
                            Err(status) => {
                                yield LLMEngineOutput::error(
                                    format!("sglang_bridge prefill: gRPC stream error: {status}")
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }
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
