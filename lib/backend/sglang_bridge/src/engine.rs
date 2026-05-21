// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridge to SGLang's native gRPC server (`sglang.runtime.v1`, enabled by
//! `sglang.launch_server --enable-grpc`). Forwards `PreprocessedRequest`
//! to SGLang's `Generate` RPC and streams tokens back.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, CommonArgs, DisaggregationMode, DynamoError, EngineConfig,
    ErrorType, FinishReason, GenerateContext, LLMEngine, LLMEngineOutput, LLMEngineOutputExt,
    PreprocessedRequest, WorkerConfig, chunk, usage,
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

#[derive(clap::Parser, Debug)]
#[command(
    name = "dynamo-sglang-bridge",
    about = "Dynamo sidecar bridge to SGLang's native gRPC server (sglang.runtime.v1)."
)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    /// gRPC endpoint of the SGLang server.
    #[arg(long, env = "SGLANG_GRPC_ENDPOINT", default_value = "http://127.0.0.1:30000")]
    pub sglang_grpc_endpoint: String,

    /// Operator override for the public-facing model name. When unset, the
    /// bridge advertises whatever SGLang reports in `GetServerInfo`
    /// (its own `--served-model-name` flag, defaulted to model_path).
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Connect timeout for the initial gRPC dial, seconds.
    #[arg(long, default_value_t = 30)]
    pub connect_timeout_secs: u64,

    /// Bootstrap host advertised to decode workers. Falls back to
    /// `dist_init_addr` from SGLang's `GetServerInfo`, then to `127.0.0.1`.
    /// Set this to the prefill Pod's reachable address in multi-host
    /// deployments.
    #[arg(long, env = "DYN_BOOTSTRAP_HOST")]
    pub bootstrap_host: Option<String>,
}

pub struct SglangBridge {
    grpc_endpoint: String,
    connect_timeout_secs: u64,
    disaggregation_mode: DisaggregationMode,
    bootstrap_host_override: Option<String>,
    served_model_name_override: Option<String>,
    /// All resolved on `start()` from SGLang's `GetModelInfo` + `GetServerInfo`.
    model_path: OnceCell<String>,
    served_model_name: OnceCell<String>,
    bootstrap_host: OnceCell<String>,
    bootstrap_port: OnceCell<u16>,
    client: OnceCell<SglangServiceClient<Channel>>,
}

impl SglangBridge {
    pub(crate) fn new(
        grpc_endpoint: String,
        connect_timeout_secs: u64,
        disaggregation_mode: DisaggregationMode,
        bootstrap_host_override: Option<String>,
        served_model_name_override: Option<String>,
    ) -> Self {
        Self {
            grpc_endpoint,
            connect_timeout_secs,
            disaggregation_mode,
            bootstrap_host_override,
            served_model_name_override,
            model_path: OnceCell::new(),
            served_model_name: OnceCell::new(),
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

        let mode = args.common.disaggregation_mode;

        // Prefill workers register under component=prefill so the prefill
        // router targets them on a separate discovery endpoint from decode.
        let endpoint_types = if mode.is_prefill() {
            "prefill".to_string()
        } else {
            args.common.endpoint_types
        };
        let component = if mode.is_prefill() {
            "prefill".to_string()
        } else {
            args.common.component
        };

        let engine = SglangBridge::new(
            args.sglang_grpc_endpoint,
            args.connect_timeout_secs,
            mode,
            args.bootstrap_host,
            args.served_model_name,
        );
        // model_name + served_model_name are left empty; the bridge populates
        // them on `start()` from SGLang's GetModelInfo + GetServerInfo. The
        // Worker falls back to EngineConfig.model when WorkerConfig.model_name
        // is empty.
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component,
            endpoint: args.common.endpoint,
            endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
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

/// Map Dynamo's `GuidedDecodingOptions` onto v1 `SamplingParams.{json_schema,
/// regex}`. Priority: json > regex > choice (anchored alternation) > grammar
/// (forwarded as regex; v1 has no EBNF field).
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
        return (None, Some(format!("^({})$", alt.join("|"))));
    }
    if let Some(grammar) = &g.grammar {
        tracing::warn!("guided_decoding.grammar set but v1 proto has no EBNF field; forwarding as regex");
        return (None, Some(grammar.clone()));
    }
    (None, None)
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(
            c,
            '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | '\\'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn parse_finish_reason(raw: &str) -> FinishReason {
    match raw {
        "stop" | "" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "abort" => FinishReason::Cancelled,
        other => FinishReason::Error(format!("unknown sglang finish_reason: {other}")),
    }
}

fn extract_context_length_from_model_info(json_info: &str) -> Option<u32> {
    let v: serde_json::Value = serde_json::from_str(json_info).ok()?;
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

/// Fields the bridge reads out of SGLang's `GetServerInfo.json_info`. Mirrors
/// the subset that `components/src/dynamo/sglang/register.py:_get_runtime_config`
/// pulls off `ServerArgs` + `scheduler_info` so the MDC populated here matches
/// the one the in-process worker would have published.
#[derive(Debug, Default)]
struct SglangServerInfo {
    /// `ServerArgs.model_path` — HF id or local dir SGLang loaded from.
    model_path: Option<String>,
    /// `ServerArgs.served_model_name` — public model name. SGLang defaults
    /// this to model_path when the operator doesn't override.
    served_model_name: Option<String>,
    /// `ServerArgs.page_size`.
    page_size: Option<u32>,
    /// `ServerArgs.max_running_requests`.
    max_running_requests: Option<u64>,
    /// `ServerArgs.max_prefill_tokens`.
    max_prefill_tokens: Option<u64>,
    /// `ServerArgs.dp_size`.
    dp_size: Option<u32>,
    /// `scheduler_infos[0].max_total_num_tokens`. Used to derive total_kv_blocks
    /// and as a fallback for max_num_batched_tokens (mirrors the Python path).
    max_total_num_tokens: Option<u64>,
    /// `ServerArgs.disaggregation_bootstrap_port` (prefill workers only).
    bootstrap_port: Option<u16>,
    /// Host half of `ServerArgs.dist_init_addr` (prefill workers only).
    bootstrap_host: Option<String>,
}

/// Parse the subset of `GetServerInfo.json_info` fields the bridge cares about.
/// ServerArgs may be nested under `server_args` or live at the top level
/// depending on SGLang version.
fn parse_server_info(json_info: &str) -> SglangServerInfo {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json_info) else {
        tracing::warn!("GetServerInfo.json_info is not valid JSON");
        return SglangServerInfo::default();
    };
    let args = v.get("server_args").unwrap_or(&v);
    SglangServerInfo {
        model_path: args
            .get("model_path")
            .and_then(|s| s.as_str())
            .map(str::to_string),
        served_model_name: args
            .get("served_model_name")
            .and_then(|s| s.as_str())
            .map(str::to_string),
        page_size: args.get("page_size").and_then(|n| n.as_u64()).map(|n| n as u32),
        max_running_requests: args.get("max_running_requests").and_then(|n| n.as_u64()),
        max_prefill_tokens: args.get("max_prefill_tokens").and_then(|n| n.as_u64()),
        dp_size: args.get("dp_size").and_then(|n| n.as_u64()).map(|n| n as u32),
        // scheduler_info lives at the top level alongside server_args.
        max_total_num_tokens: v.get("max_total_num_tokens").and_then(|n| n.as_u64()),
        bootstrap_port: args
            .get("disaggregation_bootstrap_port")
            .and_then(|n| n.as_u64())
            .map(|n| n as u16),
        bootstrap_host: args
            .get("dist_init_addr")
            .and_then(|s| s.as_str())
            .map(|addr| {
                addr.rsplit_once(':')
                    .map(|(h, _)| h.to_string())
                    .unwrap_or_else(|| addr.to_string())
            }),
    }
}

#[async_trait]
impl LLMEngine for SglangBridge {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        let endpoint = Endpoint::from_shared(self.grpc_endpoint.clone())
            .map_err(|e| invalid_arg(format!("invalid grpc endpoint: {e}")))?
            .connect_timeout(Duration::from_secs(self.connect_timeout_secs));
        let channel = endpoint
            .connect()
            .await
            .map_err(|e| backend_error(format!("connect {}: {e}", self.grpc_endpoint)))?;
        let mut client = SglangServiceClient::new(channel);

        // gRPC port binds before the scheduler is loaded (~30-60s).
        // HealthCheck returns healthy=false until ready; poll up to 120s.
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
                    "SGLang HealthCheck returned healthy=false for 120s",
                ));
            }
            tracing::debug!(?backoff, "sglang_bridge: HealthCheck unhealthy, retrying");
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(5));
        }

        let model_info = client
            .get_model_info(GetModelInfoRequest {})
            .await
            .map_err(|e| backend_error(format!("GetModelInfo: {e}")))?
            .into_inner();
        let context_length = extract_context_length_from_model_info(&model_info.json_info);

        let server_info = client
            .get_server_info(GetServerInfoRequest {})
            .await
            .map_err(|e| backend_error(format!("GetServerInfo: {e}")))?
            .into_inner();
        let info = parse_server_info(&server_info.json_info);

        // `model_path` is the tokenizer source the frontend will load. Prefer
        // GetModelInfo.model_path (always present) over GetServerInfo's copy.
        let model_path = if !model_info.model_path.is_empty() {
            model_info.model_path.clone()
        } else {
            info.model_path.clone().unwrap_or_default()
        };
        if model_path.is_empty() {
            return Err(backend_error(
                "SGLang reported empty model_path in GetModelInfo / GetServerInfo",
            ));
        }
        let served_model_name = self
            .served_model_name_override
            .clone()
            .or(info.served_model_name)
            .unwrap_or_else(|| model_path.clone());
        let _ = self.model_path.set(model_path.clone());
        let _ = self.served_model_name.set(served_model_name.clone());

        if self.disaggregation_mode.is_prefill() {
            let port = info.bootstrap_port.ok_or_else(|| {
                backend_error(
                    "GetServerInfo.json_info.disaggregation_bootstrap_port missing — \
                     is SGLang launched with `--disaggregation-mode prefill`?",
                )
            })?;
            let host = self
                .bootstrap_host_override
                .clone()
                .or(info.bootstrap_host)
                .unwrap_or_else(|| "127.0.0.1".to_string());
            tracing::info!(
                bootstrap_host = %host,
                bootstrap_port = port,
                "sglang_bridge prefill: bootstrap rendezvous resolved"
            );
            let _ = self.bootstrap_host.set(host);
            let _ = self.bootstrap_port.set(port);
        }

        self.client
            .set(client)
            .map_err(|_| backend_error("client already set"))?;

        // Mirror dynamo.sglang's MDC: if max_prefill_tokens is unset, fall
        // back to scheduler's max_total_num_tokens so the planner always has
        // a prefill load signal.
        let max_num_batched_tokens = info.max_prefill_tokens.or(info.max_total_num_tokens);
        // total_kv_blocks = ceil(max_total_num_tokens / page_size).
        let total_kv_blocks = match (info.max_total_num_tokens, info.page_size) {
            (Some(t), Some(p)) if p > 0 => Some((t + p as u64 - 1) / p as u64),
            _ => None,
        };

        tracing::info!(
            endpoint = %self.grpc_endpoint,
            model = %model_path,
            served_as = %served_model_name,
            context_length = ?context_length,
            disagg_mode = %self.disaggregation_mode,
            "sglang_bridge: connected"
        );

        Ok(EngineConfig {
            model: model_path,
            served_model_name: Some(served_model_name),
            context_length,
            kv_cache_block_size: info.page_size,
            total_kv_blocks,
            max_num_seqs: info.max_running_requests,
            max_num_batched_tokens,
            data_parallel_size: info.dp_size,
            bootstrap_host: self.bootstrap_host.get().cloned(),
            bootstrap_port: self.bootstrap_port.get().copied(),
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let ctx = ctx.inner_arc();
        if self.disaggregation_mode.is_prefill() {
            self.generate_prefill(request, ctx).await
        } else {
            self.generate_decode(request, ctx).await
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
            tracing::warn!(request_id = ctx.id(), error = %e, "sglang_bridge abort RPC failed");
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        Ok(())
    }
}

impl SglangBridge {
    /// Decode and aggregated paths: forward request to SGLang, stream tokens
    /// back, emit a typed terminal on the `finished=true` chunk. Decode
    /// workers also forward `request.bootstrap_info` so SGLang pulls KV from
    /// the prefill rendezvous.
    async fn generate_decode(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        tracing::debug!(
            request_id = ctx.id(),
            has_bootstrap_info = request.bootstrap_info.is_some(),
            "sglang_bridge: generate_decode"
        );
        let mut client = self
            .client
            .get()
            .ok_or_else(|| backend_error("generate called before start"))?
            .clone();

        let prompt_len = request.token_ids.len() as u32;
        let input_ids: Vec<i32> = request.token_ids.iter().map(|&t| t as i32).collect();
        let sampling = build_sampling_params(&request);

        // Forward the full 63-bit room — the frontend's PrefillRouter encodes
        // dp_rank into the upper bits via `compute_bootstrap_room`.
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

                        if !resp.finished {
                            completion_tokens = completion_tokens.saturating_add(resp.output_ids.len() as u32);
                            for tid in resp.output_ids {
                                yield Ok(chunk::token(tid as u32));
                            }
                            continue;
                        }

                        // Terminal chunk may carry the final token(s) alongside
                        // finished=true; flush before the terminal.
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
    /// carrying `disaggregated_params` for the frontend's prefill router to
    /// forward to decode. The actual `Generate` RPC runs in a background
    /// task so the prefill router unblocks immediately — SGLang prefill
    /// doesn't emit gRPC output until decode pulls KV via NIXL/Mooncake, so
    /// awaiting it here would deadlock.
    async fn generate_prefill(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        tracing::debug!(request_id = ctx.id(), "sglang_bridge: generate_prefill");
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

        // Honour a router-provided room (carries dp_rank in upper bits),
        // otherwise mint a fresh 63-bit room.
        let bootstrap_room: i64 = request
            .bootstrap_info
            .as_ref()
            .map(|bi| bi.bootstrap_room as i64)
            .unwrap_or_else(|| rand::rng().random_range(1..i64::MAX));

        let input_ids: Vec<i32> = request.token_ids.iter().map(|&t| t as i32).collect();
        let sampling = build_sampling_params(&request);

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
            disaggregated_params: Some(DisaggregatedParams {
                bootstrap_host: bootstrap_host.clone(),
                bootstrap_port: bootstrap_port as i32,
                bootstrap_room,
            }),
        };

        let mut bootstrap_chunk = LLMEngineOutput::default();
        bootstrap_chunk.disaggregated_params = Some(serde_json::json!({
            "bootstrap_host": bootstrap_host,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }));

        let request_id = ctx.id().to_string();
        let ctx_bg = ctx.clone();
        tokio::spawn(async move {
            let response = match client.generate(grpc_req).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(
                        request_id = %request_id,
                        error = %e,
                        "sglang_bridge prefill: Generate RPC failed"
                    );
                    return;
                }
            };
            let mut stream = response.into_inner();
            loop {
                tokio::select! {
                    biased;
                    _ = ctx_bg.stopped() => return,
                    maybe_chunk = stream.next() => match maybe_chunk {
                        Some(Ok(resp)) if resp.finished => return,
                        Some(Ok(_)) => continue,
                        Some(Err(status)) => {
                            tracing::warn!(
                                request_id = %request_id,
                                %status,
                                "sglang_bridge prefill: gRPC stream error"
                            );
                            return;
                        }
                        None => return,
                    },
                }
            }
        });

        // Yield only the bootstrap chunk (no terminal). The prefill router
        // forwards `disaggregated_params` to decode; the spawned task above
        // keeps the gRPC call alive until SGLang closes it.
        Ok(Box::pin(async_stream::stream! {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_finish_reasons() {
        assert!(matches!(parse_finish_reason("stop"), FinishReason::Stop));
        assert!(matches!(parse_finish_reason(""), FinishReason::Stop));
        assert!(matches!(
            parse_finish_reason("length"),
            FinishReason::Length
        ));
        assert!(matches!(
            parse_finish_reason("abort"),
            FinishReason::Cancelled
        ));
        match parse_finish_reason("???") {
            FinishReason::Error(msg) => assert!(msg.contains("???")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn regex_escape_handles_metas_and_passthrough() {
        assert_eq!(regex_escape("foo"), "foo");
        assert_eq!(regex_escape("a.b"), "a\\.b");
        assert_eq!(regex_escape("(x|y)"), "\\(x\\|y\\)");
        assert_eq!(regex_escape("\\"), "\\\\");
    }

    #[test]
    fn parses_server_info_both_layouts_and_fields() {
        // Nested under `server_args`, with a full set of fields + scheduler info.
        let info = parse_server_info(
            r#"{
              "server_args": {
                "model_path": "Qwen/Qwen3-0.6B",
                "served_model_name": "my-model",
                "page_size": 16,
                "max_running_requests": 256,
                "max_prefill_tokens": 16384,
                "dp_size": 2,
                "disaggregation_bootstrap_port": 8998,
                "dist_init_addr": "10.0.0.1:5555"
              },
              "max_total_num_tokens": 1048576
            }"#,
        );
        assert_eq!(info.model_path.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(info.served_model_name.as_deref(), Some("my-model"));
        assert_eq!(info.page_size, Some(16));
        assert_eq!(info.max_running_requests, Some(256));
        assert_eq!(info.max_prefill_tokens, Some(16384));
        assert_eq!(info.dp_size, Some(2));
        assert_eq!(info.max_total_num_tokens, Some(1048576));
        assert_eq!(info.bootstrap_port, Some(8998));
        assert_eq!(info.bootstrap_host.as_deref(), Some("10.0.0.1"));

        // Top-level layout + bare host in dist_init_addr.
        let info = parse_server_info(
            r#"{"model_path": "X", "disaggregation_bootstrap_port": 9000, "dist_init_addr": "node1"}"#,
        );
        assert_eq!(info.model_path.as_deref(), Some("X"));
        assert_eq!(info.bootstrap_port, Some(9000));
        assert_eq!(info.bootstrap_host.as_deref(), Some("node1"));

        // Bad JSON → all-None, no panic.
        let info = parse_server_info("not json");
        assert!(info.model_path.is_none() && info.bootstrap_port.is_none());
    }

    #[test]
    fn extracts_context_length_with_fallback_and_bounds() {
        assert_eq!(
            extract_context_length_from_model_info(r#"{"max_context_length": 4096}"#),
            Some(4096)
        );
        // Falls back to context_length when max_context_length missing.
        assert_eq!(
            extract_context_length_from_model_info(r#"{"context_length": 8192}"#),
            Some(8192)
        );
        // Rejects zero and out-of-range.
        assert_eq!(
            extract_context_length_from_model_info(r#"{"max_context_length": 0}"#),
            None
        );
        assert_eq!(
            extract_context_length_from_model_info(r#"{"max_context_length": 99999999999}"#),
            None
        );
    }

}
