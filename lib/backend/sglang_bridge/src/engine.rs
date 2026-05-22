// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridge to SGLang's native gRPC server (`sglang.runtime.v1`). Forwards
//! `PreprocessedRequest` to SGLang's `Generate` RPC and streams tokens back.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, DisaggregationMode, DynamoError, EngineConfig, ErrorType,
    FinishReason, GenerateContext, LLMEngine, LLMEngineOutput, LLMEngineOutputExt,
    PreprocessedRequest, WorkerConfig, chunk, usage,
};
use futures::stream::BoxStream;
use rand::Rng;
use tokio::sync::OnceCell;
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;
use tonic::transport::{Channel, Endpoint};

use crate::args::Args;
use crate::health;
use crate::proto::v1::{
    AbortRequest, DisaggregatedParams, GenerateRequest, GetModelInfoRequest, GetServerInfoRequest,
    HealthCheckRequest, sglang_service_client::SglangServiceClient,
};
use crate::sampling::{build_sampling_params, parse_finish_reason};
use crate::server_info::parse_server_info;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const SCHEDULER_READY_TIMEOUT: Duration = Duration::from_secs(120);

pub struct SglangBridge {
    grpc_endpoint: String,
    /// Discovered from SGLang's `GetServerInfo` in `start()`; read by
    /// `generate()` to dispatch prefill vs decode.
    disaggregation_mode: OnceCell<DisaggregationMode>,
    bootstrap: OnceCell<(String, u16)>,
    client: OnceCell<SglangServiceClient<Channel>>,
    health_check_task: OnceCell<JoinHandle<()>>,
}

impl SglangBridge {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(a) => <Args as clap::Parser>::try_parse_from(a),
            None => <Args as clap::Parser>::try_parse(),
        }
        .map_err(|e| invalid_arg(e.to_string()))?;

        let engine = SglangBridge {
            grpc_endpoint: args.sglang_grpc_endpoint,
            disaggregation_mode: OnceCell::new(),
            bootstrap: OnceCell::new(),
            client: OnceCell::new(),
            health_check_task: OnceCell::new(),
        };
        // disaggregation_mode is discovered by `start()` from SGLang and
        // returned on EngineConfig; Worker reads the engine's override.
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            ..Default::default()
        };
        Ok((engine, config))
    }

    fn mode(&self) -> DisaggregationMode {
        self.disaggregation_mode.get().copied().unwrap_or_default()
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

        let mut backoff = Duration::from_millis(500);
        loop {
            if client
                .health_check(HealthCheckRequest {})
                .await
                .map_err(|e| backend_error(format!("HealthCheck: {e}")))?
                .into_inner()
                .healthy
            {
                break;
            }
            if backoff > SCHEDULER_READY_TIMEOUT {
                return Err(backend_error("SGLang HealthCheck unhealthy for 120s"));
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(5));
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
            let host = info.bootstrap_host.unwrap_or_else(|| "127.0.0.1".to_string());
            let _ = self.bootstrap.set((host, port));
        }

        self.client
            .set(client.clone())
            .map_err(|_| backend_error("client already set"))?;
        self.health_check_task
            .set(tokio::spawn(health::run_loop(client)))
            .map_err(|_| backend_error("start called twice"))?;

        let total_kv_blocks = match (info.max_total_num_tokens, info.page_size) {
            (Some(t), Some(p)) if p > 0 => Some(t.div_ceil(p as u64)),
            _ => None,
        };

        tracing::info!(
            endpoint = %self.grpc_endpoint,
            model = %model_path,
            disagg_mode = %mode,
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
            disaggregation_mode_override: Some(mode),
            context_length,
            kv_cache_block_size: info.page_size,
            total_kv_blocks,
            max_num_seqs: info.max_running_requests,
            max_num_batched_tokens: info.max_prefill_tokens.or(info.max_total_num_tokens),
            data_parallel_size: info.dp_size,
            bootstrap_host,
            bootstrap_port,
            ..Default::default()
        })
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
        let Some(client) = self.client.get() else { return };
        let req = AbortRequest { rid: ctx.id().to_string(), abort_all: false };
        if let Err(e) = client.clone().abort(req).await {
            tracing::warn!(request_id = ctx.id(), error = %e, "abort RPC failed");
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        if let Some(task) = self.health_check_task.get() {
            task.abort();
        }
        Ok(())
    }
}

impl SglangBridge {
    async fn generate_decode(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let mut client = self.client_or_err()?;
        let prompt_len = request.token_ids.len() as u32;
        let grpc_req = GenerateRequest {
            input_ids: request.token_ids.iter().map(|&t| t as i32).collect(),
            sampling_params: Some(build_sampling_params(&request)),
            stream: Some(true),
            rid: Some(ctx.id().to_string()),
            disaggregated_params: request.bootstrap_info.as_ref().map(|bi| DisaggregatedParams {
                bootstrap_host: bi.bootstrap_host.clone(),
                bootstrap_port: bi.bootstrap_port as i32,
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

    /// Yields one bootstrap chunk synchronously; the RPC drains in the
    /// background since SGLang prefill blocks until decode pulls KV.
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
            sampling_params: Some(build_sampling_params(&request)),
            stream: Some(true),
            rid: Some(ctx.id().to_string()),
            disaggregated_params: Some(DisaggregatedParams {
                bootstrap_host: bootstrap_host.clone(),
                bootstrap_port: bootstrap_port as i32,
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

        tokio::spawn(async move {
            let mut stream = match client.generate(grpc_req).await {
                Ok(r) => r.into_inner(),
                Err(e) => {
                    tracing::warn!(request_id = ctx.id(), error = %e, "prefill Generate failed");
                    return;
                }
            };
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => return,
                    maybe_chunk = stream.next() => match maybe_chunk {
                        Some(Ok(r)) if r.finished => return,
                        Some(Ok(_)) => continue,
                        Some(Err(s)) => {
                            tracing::warn!(request_id = ctx.id(), status = %s, "prefill stream error");
                            return;
                        }
                        None => return,
                    },
                }
            }
        });

        Ok(Box::pin(async_stream::stream! { yield Ok(bootstrap_chunk); }))
    }

    fn client_or_err(&self) -> Result<SglangServiceClient<Channel>, DynamoError> {
        self.client
            .get()
            .ok_or_else(|| backend_error("generate called before start"))
            .cloned()
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
