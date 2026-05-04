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

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, CommonArgs, DynamoError, EngineConfig, ErrorType,
    FinishReason, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest,
    WorkerConfig, chunk, usage,
};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;
use tonic::transport::{Channel, Endpoint};

use crate::proto::scheduler::{
    AbortRequest, DisaggregatedParams, GenerateRequest, GetModelInfoRequest, HealthCheckRequest,
    SamplingParams, TokenizedInput, generate_response::Response as GenResponse,
    sglang_scheduler_client::SglangSchedulerClient,
};

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
}

pub struct SglangBridge {
    grpc_endpoint: String,
    served_model_name: String,
    connect_timeout_secs: u64,
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
        let engine = SglangBridge {
            grpc_endpoint: args.sglang_grpc_endpoint,
            served_model_name: served.clone(),
            connect_timeout_secs: args.connect_timeout_secs,
            client: OnceCell::new(),
        };
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
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

        self.client
            .set(client)
            .map_err(|_| backend_error("client already set"))?;

        tracing::info!(
            endpoint = %self.grpc_endpoint,
            model = %self.served_model_name,
            context_length = ?context_length,
            sglang_model_path = %info.model_path,
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
        let mut client = self
            .client
            .get()
            .ok_or_else(|| backend_error("generate called before start"))?
            .clone();

        let prompt_len = request.token_ids.len() as u32;
        let input_ids: Vec<u32> = request.token_ids.clone();
        let sampling = build_sampling_params(&request);

        // Disagg PD: forward bootstrap_info if present. The legacy schema
        // carries this first-class, unlike the Phase-1 proto.
        let disagg = request.bootstrap_info.as_ref().and_then(|bi| {
            serde_json::to_value(bi).ok().and_then(|v| {
                let host = v.get("bootstrap_host")?.as_str()?.to_string();
                let port = v.get("bootstrap_port")?.as_i64()? as i32;
                let room = v.get("bootstrap_room")?.as_i64()? as i32;
                Some(DisaggregatedParams {
                    bootstrap_host: host,
                    bootstrap_port: port,
                    bootstrap_room: room,
                })
            })
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
                                // GenerateComplete.output_ids is the cumulative
                                // final list. We want only what hasn't been
                                // streamed yet via Chunk events. Since streaming
                                // chunks already emitted everything, the
                                // terminal carries no new tokens.
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
