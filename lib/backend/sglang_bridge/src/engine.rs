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
//!   `GetServerInfo.server_args`. On each `generate()` it generates a 63-bit
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
//! Rooms are 63-bit. The proto field is `int64` (patched smg-grpc-proto;
//! see lib/backend/sglang_bridge/proto/sglang_scheduler.proto). The frontend
//! stores `bootstrap_room` as `u64` and dynamo's PrefillRouter encodes
//! `dp_rank` into the upper bits via `compute_bootstrap_room`, so the full
//! width must survive the wire for dp-aware NIXL rendezvous.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, CommonArgs, DisaggregatedEndpoint, DynamoError, EngineConfig,
    ErrorType, FinishReason, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest,
    WorkerConfig, chunk, usage,
};
use futures::stream::BoxStream;
use rand::Rng;
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;
use tonic::transport::{Channel, Endpoint};

use crate::proto::v1::{
    AbortRequest, DisaggregatedParams, GenerateRequest, GetModelInfoRequest, GetServerInfoRequest,
    HealthCheckRequest, SamplingParams,
    sglang_service_client::SglangServiceClient,
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
    name = "dynamo-sglang-bridge",
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
    client: OnceCell<SglangServiceClient<Channel>>,
}

impl SglangBridge {
    /// Programmatic constructor for in-process callers (B1). Skips CLI
    /// parsing; the launcher provides the values directly.
    pub fn new(
        grpc_endpoint: String,
        served_model_name: String,
        connect_timeout_secs: u64,
        disaggregation_mode: DisaggMode,
        bootstrap_host_override: Option<String>,
    ) -> Self {
        Self {
            grpc_endpoint,
            served_model_name,
            connect_timeout_secs,
            disaggregation_mode,
            bootstrap_host_override,
            bootstrap_host: OnceCell::new(),
            bootstrap_port: OnceCell::new(),
            client: OnceCell::new(),
        }
    }

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

        // Prefill workers register under component=prefill so they live on a
        // separate discovery endpoint from decode workers. Without this both
        // bridges register on `{ns}.backend.generate` and the frontend's
        // prefill PushRouter routes prefill traffic to either bridge at
        // random. Mirrors `dynamo.sglang` (components/.../args.py:265:
        // `dyn://{ns}.prefill.generate`).
        let component = match args.disaggregation_mode {
            DisaggMode::Prefill => "prefill".to_string(),
            DisaggMode::Decode | DisaggMode::None => args.common.component,
        };

        let engine = SglangBridge::new(
            args.sglang_grpc_endpoint,
            served.clone(),
            args.connect_timeout_secs,
            args.disaggregation_mode,
            args.bootstrap_host,
        );
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component,
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
    let (json_schema, regex) = build_structured_constraint(req);
    SamplingParams {
        temperature: req.sampling_options.temperature,
        top_p: req.sampling_options.top_p,
        top_k: req.sampling_options.top_k,
        min_p: req.sampling_options.min_p,
        frequency_penalty: req.sampling_options.frequency_penalty,
        presence_penalty: req.sampling_options.presence_penalty,
        repetition_penalty: req.sampling_options.repetition_penalty,
        max_new_tokens: req.stop_conditions.max_tokens.map(|v| v as i32),
        min_new_tokens: req.stop_conditions.min_tokens.map(|v| v as i32),
        stop: req.stop_conditions.stop.clone().unwrap_or_default(),
        stop_token_ids: req
            .stop_conditions
            .stop_token_ids_hidden
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|t| t as i32)
            .collect(),
        ignore_eos: req.stop_conditions.ignore_eos,
        n: req.sampling_options.n.map(|v| v as i32),
        json_schema,
        regex,
    }
}

/// Map Dynamo's `GuidedDecodingOptions` onto the v1 SamplingParams structured
/// generation fields. v1 has `json_schema: optional string` and
/// `regex: optional string` as separate fields (no oneof, no EBNF). Returns
/// `(json_schema, regex)` — caller wires both into SamplingParams.
///
/// Priority when multiple guided-decoding fields are set: json > regex >
/// choice (collapsed to anchored regex alternation) > grammar (sent as regex
/// fallback — v1 has no EBNF field; backend may reject if it's not regex).
///
/// `backend` and `whitespace_pattern` on `GuidedDecodingOptions` are dropped
/// — v1 exposes neither; SGLang picks its grammar backend via server args.
fn build_structured_constraint(req: &PreprocessedRequest) -> (Option<String>, Option<String>) {
    let Some(g) = req.sampling_options.guided_decoding.as_ref() else {
        return (None, None);
    };
    if let Some(schema) = &g.json {
        match serde_json::to_string(schema) {
            Ok(s) => return (Some(s), None),
            Err(e) => {
                tracing::warn!(error = %e, "guided_decoding.json serialize failed; dropping constraint");
            }
        }
    }
    if let Some(regex) = &g.regex {
        return (None, Some(regex.clone()));
    }
    if let Some(choices) = &g.choice
        && !choices.is_empty()
    {
        let alt: Vec<String> = choices.iter().map(|c| regex_escape(c)).collect();
        let pattern = format!("^({})$", alt.join("|"));
        return (None, Some(pattern));
    }
    if let Some(grammar) = &g.grammar {
        // v1 has no EBNF field; surface grammar as regex on the wire and let
        // the backend reject if it can't interpret it. Log so operators see.
        tracing::warn!("guided_decoding.grammar set but v1 proto has no EBNF field; forwarding as regex");
        return (None, Some(grammar.clone()));
    }
    (None, None)
}

/// Escape regex metacharacters. The `regex` crate has `regex::escape`, but
/// pulling it in just for this is heavier than open-coding it.
fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn parse_finish_reason(raw: &str) -> FinishReason {
    // v1 stores finish_reason in GenerateResponse.meta_info["finish_reason"].
    // SGLang emits OpenAI-style strings.
    match raw {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "abort" | "cancelled" => FinishReason::Cancelled,
        "" => FinishReason::Stop, // empty defaults to stop
        other => FinishReason::Error(format!("unknown sglang finish_reason: {other}")),
    }
}

/// Pull `disaggregation_bootstrap_port` and `dist_init_addr` from the JSON
/// blob in `GetServerInfoResponse.json_info`. v1 emits the full ServerArgs
/// dict as a JSON string rather than a typed `google.protobuf.Struct`.
fn extract_bootstrap_from_server_info(json_info: &str) -> (Option<u16>, Option<String>) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json_info) else {
        tracing::warn!("GetServerInfo.json_info is not valid JSON");
        return (None, None);
    };
    // ServerArgs may be nested under "server_args" or at the top level —
    // accept either layout for forward-compat.
    let root = v.get("server_args").unwrap_or(&v);
    let port = root
        .get("disaggregation_bootstrap_port")
        .and_then(|n| n.as_u64())
        .map(|n| n as u16);
    let host = root
        .get("dist_init_addr")
        .and_then(|s| s.as_str())
        // SGLang stores dist_init_addr as "host:port"; bootstrap uses the host
        // half only. Strip the trailing :port if present.
        .map(|addr| {
            addr.rsplit_once(':')
                .map(|(h, _)| h.to_string())
                .unwrap_or_else(|| addr.to_string())
        });
    (port, host)
}

/// Pull `max_context_length` (or compatible) from the JSON blob in
/// `GetModelInfoResponse.json_info`. v1 collapses the per-field model card
/// into a JSON string.
fn extract_context_length_from_model_info(json_info: &str) -> Option<u32> {
    let v: serde_json::Value = serde_json::from_str(json_info).ok()?;
    // SGLang's get_internal_model_info usually exposes max_context_length;
    // some builds expose context_length only. Try both.
    let n = v
        .get("max_context_length")
        .or_else(|| v.get("context_length"))
        .and_then(|n| n.as_u64())?;
    if n == 0 || n > u32::MAX as u64 {
        None
    } else {
        Some(n as u32)
    }
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
        let mut client = SglangServiceClient::new(channel);

        // The native gRPC server binds the listening socket as soon as the
        // server task starts, but the scheduler subprocess takes 30-60s to
        // load the model + warm up. HealthCheck returns healthy=false until
        // the scheduler is reachable. Poll for up to ~120s.
        let mut backoff = Duration::from_millis(500);
        loop {
            let health = client
                .health_check(HealthCheckRequest {})
                .await
                .map_err(|e| backend_error(format!("HealthCheck: {e}")))?
                .into_inner();
            if health.healthy {
                break;
            }
            if backoff > Duration::from_secs(120) {
                return Err(backend_error(
                    "SGLang HealthCheck returned healthy=false for 120s — \
                     scheduler likely failed to load. Check SGLang logs.",
                ));
            }
            tracing::debug!(
                ?backoff,
                "sglang_bridge: HealthCheck reports unhealthy, retrying"
            );
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(5));
        }

        let info = client
            .get_model_info(GetModelInfoRequest {})
            .await
            .map_err(|e| backend_error(format!("GetModelInfo: {e}")))?
            .into_inner();

        let context_length = extract_context_length_from_model_info(&info.json_info);

        // Prefill workers need bootstrap host/port to advertise to decode
        // workers. Pull from GetServerInfo.json_info once at startup.
        if matches!(self.disaggregation_mode, DisaggMode::Prefill) {
            let server_info = client
                .get_server_info(GetServerInfoRequest {})
                .await
                .map_err(|e| backend_error(format!("GetServerInfo: {e}")))?
                .into_inner();
            let (port, host_from_args) =
                extract_bootstrap_from_server_info(&server_info.json_info);
            let port = port.ok_or_else(|| {
                backend_error(
                    "GetServerInfo.json_info.disaggregation_bootstrap_port missing — \
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

        // Advertise our bootstrap rendezvous to the frontend's PrefillRouter
        // via ModelRuntimeConfig. Without this, the router can't find the
        // prefill worker's bootstrap_host/port and falls into the
        // "original prefill path" that forwards prefill output via the
        // wire instead of letting SGLang do NIXL KV transfer directly.
        let disaggregated_endpoint = matches!(self.disaggregation_mode, DisaggMode::Prefill)
            .then(|| DisaggregatedEndpoint {
                bootstrap_host: self.bootstrap_host.get().cloned(),
                bootstrap_port: self.bootstrap_port.get().copied(),
            });

        Ok(EngineConfig {
            model: self.served_model_name.clone(),
            served_model_name: Some(self.served_model_name.clone()),
            context_length,
            disaggregated_endpoint,
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
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        match self.disaggregation_mode {
            DisaggMode::Prefill => self.generate_prefill(request, ctx).await,
            DisaggMode::Decode | DisaggMode::None => self.generate_decode(request, ctx).await,
        }
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(client) = self.client.get() else { return };
        let mut client = client.clone();
        let req = AbortRequest {
            rid: ctx.id().to_string(),
            abort_all: false,
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
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        tracing::info!(
            request_id = ctx.id(),
            disagg_mode = ?self.disaggregation_mode,
            has_bootstrap_info = request.bootstrap_info.is_some(),
            "sglang_bridge decode: generate_decode ENTRY"
        );
        let mut client = self
            .client
            .get()
            .ok_or_else(|| backend_error("generate called before start"))?
            .clone();

        let prompt_len = request.token_ids.len() as u32;
        let input_ids: Vec<i32> = request.token_ids.iter().map(|&t| t as i32).collect();
        let sampling = build_sampling_params(&request);

        // Decode side: pull bootstrap_info forwarded from the prefill leg.
        // Forward the full 63-bit room — dynamo's PrefillRouter encodes
        // dp_rank into the upper bits via `compute_bootstrap_room`, so
        // truncating would clobber dp-aware rendezvous on the decode side.
        let disagg = request.bootstrap_info.as_ref().map(|bi| DisaggregatedParams {
            bootstrap_host: bi.bootstrap_host.clone(),
            bootstrap_port: bi.bootstrap_port as i32,
            bootstrap_room: bi.bootstrap_room as i64,
        });

        let grpc_req = GenerateRequest {
            input_ids,
            sampling_params: Some(sampling),
            stream: Some(true),
            return_logprob: Some(false),
            top_logprobs_num: None,
            logprob_start_len: None,
            rid: Some(ctx.id().to_string()),
            lora_path: None,
            routing_key: None,
            routed_dp_rank: None,
            trace_headers: Default::default(),
            disaggregated_params: disagg,
        };

        let response = client
            .generate(grpc_req)
            .await
            .map_err(|e| backend_error(format!("Generate RPC: {e}")))?;
        let mut stream = response.into_inner();

        Ok(Box::pin(async_stream::stream! {
            // v1 GenerateResponse: output_ids (incremental chunk per stream),
            // meta_info (string→string map; finish_reason is here), finished
            // (bool — true on the terminal chunk). No oneof.
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
                            yield Ok(LLMEngineOutput::error(
                                "sglang_bridge: stream ended without terminal".to_string()
                            ));
                            break;
                        };
                        let resp = match chunk_res {
                            Ok(r) => r,
                            Err(status) => {
                                yield Ok(LLMEngineOutput::error(
                                    format!("sglang_bridge: gRPC stream error: {status}")
                                ));
                                break;
                            }
                        };

                        // Non-terminal token chunks: yield each token.
                        if !resp.finished {
                            completion_tokens = completion_tokens.saturating_add(resp.output_ids.len() as u32);
                            for tid in resp.output_ids {
                                yield Ok(chunk::token(tid as u32));
                            }
                            continue;
                        }

                        // Terminal chunk. SGLang sometimes emits the final
                        // token(s) on the same message as finished=true; flush
                        // them before the terminal.
                        if !resp.output_ids.is_empty() {
                            completion_tokens = completion_tokens.saturating_add(resp.output_ids.len() as u32);
                            for tid in resp.output_ids {
                                yield Ok(chunk::token(tid as u32));
                            }
                        }

                        let raw_reason = resp.meta_info.get("finish_reason").map(String::as_str).unwrap_or("");
                        let finish = parse_finish_reason(raw_reason);
                        let total = resp
                            .meta_info
                            .get("completion_tokens")
                            .and_then(|s| s.parse::<u32>().ok())
                            .unwrap_or(completion_tokens);
                        let mut terminal = match finish {
                            FinishReason::Length => LLMEngineOutput::length(),
                            FinishReason::Cancelled => LLMEngineOutput::cancelled(),
                            FinishReason::Error(msg) => LLMEngineOutput::error(msg),
                            _ => LLMEngineOutput::stop(),
                        };
                        terminal.token_ids = vec![];
                        terminal = terminal.with_usage(usage(prompt_len, total));
                        yield Ok(terminal);
                        break;
                    }
                }
            }
        }))
    }

    /// Prefill path. Generates a 63-bit `bootstrap_room`, yields one chunk
    /// carrying `disaggregated_params: {host, port, room}` for the frontend's
    /// prefill router to capture and forward to the decode worker. Then
    /// issues `Generate` to SGLang (which writes KV to the rendezvous) and
    /// drains the response stream silently — prefill output is consumed via
    /// NIXL/Mooncake on the decode side, not via gRPC.
    async fn generate_prefill(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        tracing::info!(
            request_id = ctx.id(),
            "sglang_bridge prefill: generate_prefill ENTRY"
        );
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
        // otherwise generate a fresh 63-bit room. The frontend's
        // PrefillRouter::compute_bootstrap_room encodes dp_rank into the
        // upper bits, so we keep the full width on the wire (proto field is
        // int64; see lib/backend/sglang_bridge/proto/sglang_scheduler.proto).
        let bootstrap_room: i64 = request
            .bootstrap_info
            .as_ref()
            .map(|bi| bi.bootstrap_room as i64)
            .unwrap_or_else(|| rand::thread_rng().gen_range(1..i64::MAX));

        let prompt_len = request.token_ids.len() as u32;
        let input_ids: Vec<i32> = request.token_ids.iter().map(|&t| t as i32).collect();
        let sampling = build_sampling_params(&request);

        let disagg = DisaggregatedParams {
            bootstrap_host: bootstrap_host.clone(),
            bootstrap_port: bootstrap_port as i32,
            bootstrap_room,
        };

        let grpc_req = GenerateRequest {
            input_ids,
            sampling_params: Some(sampling),
            stream: Some(true),
            return_logprob: Some(false),
            top_logprobs_num: None,
            logprob_start_len: None,
            rid: Some(ctx.id().to_string()),
            lora_path: None,
            routing_key: None,
            routed_dp_rank: None,
            trace_headers: Default::default(),
            disaggregated_params: Some(disagg),
        };

        // First chunk: bootstrap info for the frontend to forward to decode.
        let mut bootstrap_chunk = LLMEngineOutput::default();
        bootstrap_chunk.disaggregated_params = Some(serde_json::json!({
            "bootstrap_host": bootstrap_host,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }));
        tracing::debug!(
            request_id = ctx.id(),
            bootstrap_host = %bootstrap_host,
            bootstrap_port = bootstrap_port,
            bootstrap_room = bootstrap_room,
            "sglang_bridge prefill: yielding bootstrap chunk"
        );

        // Run the gRPC `Generate` call in a background task and decouple it
        // from the response stream we hand back to Dynamo's frontend. The
        // `--grpc-mode` servicer doesn't yield its first gRPC response until
        // the engine produces output, and SGLang prefill in disagg mode does
        // not produce output until the decode worker reads KV via NIXL — so
        // awaiting `client.generate(...).await` here would deadlock with the
        // frontend's prefill router (which drains our stream before
        // dispatching to decode). Instead the bridge stream emits bootstrap
        // + a synthetic terminal so the prefill router unblocks immediately,
        // and the spawned task keeps the gRPC call alive in parallel until
        // SGLang closes it (KV transfer complete) or the request is
        // cancelled.
        let request_id = ctx.id().to_string();
        let ctx_for_bg = ctx.clone();
        tokio::spawn(async move {
            tracing::debug!(
                request_id = %request_id,
                "sglang_bridge prefill bg: issuing gRPC Generate"
            );
            let response = match client.generate(grpc_req).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(
                        request_id = %request_id,
                        err = %e,
                        "sglang_bridge prefill bg: Generate RPC failed"
                    );
                    return;
                }
            };
            tracing::debug!(
                request_id = %request_id,
                "sglang_bridge prefill bg: gRPC stream opened, draining"
            );
            let mut stream = response.into_inner();
            loop {
                tokio::select! {
                    biased;
                    _ = ctx_for_bg.stopped() => {
                        tracing::debug!(request_id = %request_id, "sglang_bridge prefill bg: cancellation observed");
                        return;
                    }
                    maybe_chunk = stream.next() => {
                        match maybe_chunk {
                            Some(Ok(resp)) => {
                                // v1: drop output_ids — decode worker reads
                                // tokens from KV rendezvous. `finished=true`
                                // on the terminal chunk signals SGLang is
                                // done writing KV and we can release.
                                if resp.finished {
                                    tracing::debug!(request_id = %request_id, "sglang_bridge prefill bg: SGLang finished");
                                    return;
                                }
                            }
                            Some(Err(status)) => {
                                tracing::warn!(request_id = %request_id, %status, "sglang_bridge prefill bg: gRPC stream error");
                                return;
                            }
                            None => {
                                tracing::debug!(request_id = %request_id, "sglang_bridge prefill bg: stream closed");
                                return;
                            }
                        }
                    }
                }
            }
        });

        // Suppress unused warning when we drop the synthetic stop chunk's usage helper.
        let _ = prompt_len;

        Ok(Box::pin(async_stream::stream! {
            // The bootstrap chunk delivers disaggregated_params to the
            // frontend's prefill_router. We then end the stream naturally
            // (no terminal finish_reason) so the prefill_router's drain
            // loop exits cleanly without short-circuiting the rest of the
            // chat completion (which otherwise treats a Stop terminal here
            // as the *whole request*'s end and skips decode dispatch).
            //
            // The actual prefill compute + KV transfer continues in the
            // spawned background task; the decode worker fetches KV via
            // NIXL using the bootstrap_room we just delivered.
            yield Ok(bootstrap_chunk);
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
