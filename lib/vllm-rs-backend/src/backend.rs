// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-based native vLLM backend using the backend-common [`LLMEngine`] contract.

use std::ffi::OsString;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use clap::Parser;
use dynamo_backend_common::{
    AsyncEngineContext, CommonArgs, DisaggregationMode, DynamoError, EngineConfig, GenerateContext,
    LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, MetricsBindings, MetricsCtx,
    ModelInput, PreprocessedRequest, SnapshotPublisher, WorkerConfig, usage,
};
use futures::{StreamExt, stream::BoxStream};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, TransportMode};
use vllm_llm::Llm;
use vllm_managed_engine::ManagedEngineHandle;
use vllm_managed_engine::cli::{ManagedEngineArgs, repartition_managed_engine_args};
use vllm_metrics::{EngineLabels, F64Gauge, METRICS as VLLM_METRICS, U64Counter};

use crate::control;
use crate::convert::{lower_request, map_output};
use crate::error::{backend_unknown, cannot_connect, clap_error, engine_shutdown, invalid_arg};

#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Dynamo vLLM backend based on Rust vLLM engine-core client."
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Model identifier or local model directory passed to vLLM.
    #[arg(value_name = "MODEL")]
    model: String,

    /// Public-facing model name advertised to Dynamo clients.
    #[arg(long)]
    served_model_name: Option<String>,

    /// Maximum time to wait for the managed vLLM engine to report ready.
    #[arg(
        long = "engine-ready-timeout-secs",
        env = "VLLM_ENGINE_READY_TIMEOUT_S",
        default_value_t = default_engine_ready_timeout_secs()
    )]
    engine_ready_timeout_secs: u64,

    /// Managed Python headless-engine arguments.
    #[command(flatten)]
    managed_engine: ManagedEngineArgs,

    /// Extra engine arguments forwarded to vLLM and advertised to Dynamo.
    #[command(flatten)]
    extra: ExtraEngineArgs,
}

#[derive(Clone, Debug, Default, clap::Args)]
struct ExtraEngineArgs {
    /// KV cache block size in tokens, forwarded to vLLM and advertised to Dynamo.
    #[arg(long = "block-size", value_parser = clap::value_parser!(u32).range(1..))]
    block_size: Option<u32>,

    /// Maximum number of concurrent sequences, forwarded to vLLM and advertised to Dynamo.
    #[arg(long = "max-num-seqs", value_parser = clap::value_parser!(u64).range(1..))]
    max_num_seqs: Option<u64>,

    /// Maximum number of batched tokens, forwarded to vLLM and advertised to Dynamo.
    #[arg(long = "max-num-batched-tokens", value_parser = clap::value_parser!(u64).range(1..))]
    max_num_batched_tokens: Option<u64>,
}

fn default_engine_ready_timeout_secs() -> u64 {
    600
}

fn engine_max_num_seqs(max_num_seqs: Option<u64>, data_parallel_size: u32) -> Option<u64> {
    max_num_seqs.map(|capacity| capacity.saturating_mul(u64::from(data_parallel_size.max(1))))
}

/// Dynamo backend implementation backed by a managed Python vLLM engine-core.
///
/// The backend consumes tokenized [`PreprocessedRequest`] values produced by
/// Dynamo preprocessing and streams token-level [`LLMEngineOutput`] values back
/// through the backend-common worker runtime.
pub struct VllmBackend {
    model: String,
    served_model_name: Option<String>,
    disaggregation_mode: DisaggregationMode,
    engine_ready_timeout_secs: u64,
    managed_engine: ManagedEngineArgs,
    extra: ExtraEngineArgs,
    reasoning_parser: Option<String>,
    cancel: CancellationToken,
    inner: RwLock<Option<Inner>>,
}

struct Inner {
    engine_handle: ManagedEngineHandle,
    llm: Llm,
    max_model_len: u32,
}

impl VllmBackend {
    fn new(
        model: String,
        served_model_name: Option<String>,
        disaggregation_mode: DisaggregationMode,
        engine_ready_timeout_secs: u64,
        managed_engine: ManagedEngineArgs,
        extra: ExtraEngineArgs,
        reasoning_parser: Option<String>,
    ) -> Self {
        Self {
            model,
            served_model_name,
            disaggregation_mode,
            engine_ready_timeout_secs,
            managed_engine,
            extra,
            reasoning_parser,
            cancel: CancellationToken::new(),
            inner: RwLock::new(None),
        }
    }

    /// Builds a backend and worker registration config from CLI-style arguments.
    ///
    /// When `argv` is `None`, arguments are read from the current process. The
    /// parser repartitions managed-engine Python flags before normal clap
    /// parsing so vLLM engine-core flags can be forwarded unchanged.
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let raw_args: Vec<OsString> = match argv {
            Some(a) => a.into_iter().map(Into::into).collect(),
            None => std::env::args_os().collect(),
        };
        let repartitioned_args =
            repartition_managed_engine_args::<Args>(&raw_args, None).map_err(clap_error)?;
        let args =
            Args::try_parse_from(repartitioned_args).map_err(|e| invalid_arg(e.to_string()))?;

        let disaggregation_mode = args.common.disaggregation_mode;

        let (tool_call_parser, reasoning_parser) = if disaggregation_mode.is_prefill() {
            (None, None)
        } else {
            (
                args.common.dyn_tool_call_parser,
                args.common.dyn_reasoning_parser,
            )
        };
        let engine = Self::new(
            args.model.clone(),
            args.served_model_name.clone(),
            disaggregation_mode,
            args.engine_ready_timeout_secs,
            args.managed_engine,
            args.extra,
            reasoning_parser.clone(),
        );
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            disaggregation_mode,
            model_name: args.model,
            served_model_name: args.served_model_name,
            model_input: ModelInput::Tokens,
            tool_call_parser,
            reasoning_parser,
            exclude_tools_when_tool_choice_none: args.common.exclude_tools_when_tool_choice_none,
            ..Default::default()
        };
        Ok((engine, config))
    }

    fn build_engine_config(
        &self,
        context_length: u32,
        total_kv_blocks: Option<u64>,
        data_parallel_size: u32,
    ) -> EngineConfig {
        EngineConfig {
            model: self.model.clone(),
            served_model_name: Some(
                self.served_model_name
                    .clone()
                    .unwrap_or_else(|| self.model.clone()),
            ),
            runtime_data: Default::default(),
            llm: Some(LlmRegistration {
                context_length: Some(context_length),
                kv_cache_block_size: self.extra.block_size,
                total_kv_blocks,
                max_num_seqs: self.extra.max_num_seqs,
                engine_max_num_seqs: engine_max_num_seqs(
                    self.extra.max_num_seqs,
                    data_parallel_size,
                ),
                max_num_batched_tokens: self.extra.max_num_batched_tokens,
                data_parallel_size: Some(data_parallel_size),
                // TODO: currently vLLM's Rust engine-core client only supports local data-parallel engines
                data_parallel_start_rank: Some(0),
                bootstrap_host: None,
                bootstrap_port: None,
            }),
        }
    }
}

#[async_trait]
impl LLMEngine for VllmBackend {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        let mut inner = self.inner.write().await;
        if inner.is_some() {
            return Err(engine_shutdown("vLLM backend has already been started"));
        }

        // Fail fast on an unsupported role: the Encode disaggregation mode is a
        // multimodal encoder upstream of P/D/Agg, but this backend is text-only
        // (multimodal payloads are rejected in `validate_request`). Reject at
        // startup so the misconfiguration can't register as a healthy worker and
        // only surface as a per-request error in `apply_disaggregation_mode`.
        if self.disaggregation_mode.is_encode() {
            return Err(invalid_arg(
                "encode disaggregation mode is not supported by the native vLLM backend",
            ));
        }

        // TODO: currently vLLM's Rust engine-core client only supports local data-parallel engines
        if !self.managed_engine.frontend_local_only() {
            return Err(invalid_arg(
                "remote or partially local data-parallel managed engines are not supported yet",
            ));
        }

        let handshake_port = self
            .managed_engine
            .resolve_handshake_port()
            .map_err(|e| cannot_connect(format!("failed to resolve handshake port: {e:#}")))?;

        let managed_config = {
            // vLLM 0.25 into_config takes 9 positional args; name them so a future
            // value/signature change can't silently land in the wrong slot.
            let max_model_len: Option<u32> = None; // let vLLM auto-fit from KV profiling
            let max_logprobs: Option<i32> = None;
            let profiler_config: Option<String> = None;
            let reasoning_parser: Option<&str> = self.reasoning_parser.as_deref();
            let language_model_only = false;
            let disable_log_stats = false; // keep stats on (mirrors with_log_stats(true) below)
            let shutdown_timeout: u64 = 0; // NOTE: 0 = abort in-flight requests on shutdown
            let mut config = self.managed_engine.clone().into_config(
                self.model.clone(),
                max_model_len,
                max_logprobs,
                profiler_config,
                reasoning_parser,
                language_model_only,
                disable_log_stats,
                shutdown_timeout,
                handshake_port,
            );
            self.extra.append_python_args(&mut config.python_args);
            config
        };

        let handshake_address = managed_config.handshake_address();
        let advertised_host = managed_config.handshake_host.clone();
        let engine_count = managed_config.data_parallel_size;
        let data_parallel_size = u32::try_from(engine_count).map_err(|_| {
            invalid_arg(format!(
                "data_parallel_size must fit in u32; got {engine_count}"
            ))
        })?;
        let ready_timeout = Duration::from_secs(self.engine_ready_timeout_secs);

        info!(
            %handshake_address,
            engine_count,
            ?ready_timeout,
            "starting managed vLLM engine"
        );
        let engine_handle = ManagedEngineHandle::spawn(managed_config)
            .await
            .map_err(|e| cannot_connect(format!("failed to spawn managed vLLM engine: {e:#}")))?;

        let client_config = EngineCoreClientConfig {
            transport_mode: TransportMode::HandshakeOwner {
                handshake_address,
                advertised_host,
                engine_count,
                ready_timeout,
                local_input_address: None,
                local_output_address: None,
            },
            coordinator_mode: None,
            model_name: self.model.clone(),
            client_index: 0,
        };

        let client = match EngineCoreClient::connect(client_config).await {
            Ok(client) => client,
            Err(error) => {
                let _ = engine_handle.shutdown(Duration::from_secs(0)).await;
                return Err(cannot_connect(format!(
                    "failed to connect to managed vLLM engine-core: {error}"
                )));
            }
        };

        let context_length = client.max_model_len();
        let total_kv_blocks = match client.total_num_gpu_blocks() {
            0 => None,
            blocks => Some(blocks),
        };
        let llm = Llm::new(client)
            .with_request_id_randomization(false)
            .with_log_stats(true);

        *inner = Some(Inner {
            engine_handle,
            llm,
            max_model_len: context_length,
        });

        info!(
            model = %self.model,
            engine_count,
            context_length = ?context_length,
            total_kv_blocks = ?total_kv_blocks,
            "vLLM backend started"
        );

        Ok(self.build_engine_config(context_length, total_kv_blocks, data_parallel_size))
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let request_id = ctx.id().to_string();
        let prompt_tokens = request.token_ids.len() as u32;

        let mut output_stream = {
            let inner = self.inner.read().await;
            let inner = inner
                .as_ref()
                .ok_or_else(|| engine_shutdown("vLLM backend has not been started"))?;
            let generate_request = lower_request(
                request_id,
                request,
                inner.max_model_len,
                self.disaggregation_mode,
            )?;

            inner
                .llm
                .generate(generate_request)
                .await
                .map_err(|e| backend_unknown(format!("failed to submit vLLM request: {e}")))?
        };

        Ok(Box::pin(async_stream::try_stream! {
            let mut completion_tokens = 0_u32;
            loop {
                let next = tokio::select! {
                    _ = ctx.stopped() => {
                        debug!(request_id = %ctx.id(), "vLLM backend request cancelled");
                        yield LLMEngineOutput::cancelled()
                            .with_usage(usage(prompt_tokens, completion_tokens));
                        break;
                    }
                    next = output_stream.next() => next,
                };

                let next = next.ok_or_else(|| {
                    backend_unknown(
                        "vLLM backend stream ended before a terminal output".to_string(),
                    )
                })?;

                let output = next.map_err(|error| {
                    backend_unknown(format!("vLLM backend stream failed: {error}"))
                })?;
                completion_tokens = completion_tokens
                    .saturating_add(output.token_ids.len() as u32);
                let finished = output.finished();
                yield map_output(output, prompt_tokens, completion_tokens)?;
                if finished {
                    break;
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let request_id = ctx.id().to_string();
        let inner = self.inner.read().await;
        let Some(inner) = inner.as_ref() else {
            debug!(%request_id, "vLLM backend abort skipped because engine is not started");
            return;
        };

        // Since randomized request IDs are disabled, we can directly abort by using the request ID
        // from the context.
        if let Err(error) = inner
            .llm
            .engine_core_client()
            .abort(std::slice::from_ref(&request_id))
            .await
        {
            warn!(%request_id, %error, "failed to abort vLLM request");
        } else {
            debug!(%request_id, "aborted vLLM request");
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        let Some(Inner {
            engine_handle, llm, ..
        }) = self.inner.write().await.take()
        else {
            return Ok(());
        };
        info!(model = %self.model, "shutting down vLLM backend");

        // Cancel the background metrics snapshot loop.
        self.cancel.cancel();

        // Shutdown the engine and the engine client.
        let llm_result = llm.shutdown().await;
        let engine_result = engine_handle.shutdown(Duration::from_secs(0)).await;

        let mut shutdown_errors = Vec::new();
        if let Err(error) = llm_result {
            shutdown_errors.push(format!(
                "failed to shut down vLLM engine-core client: {error}"
            ));
        }
        if let Err(error) = engine_result {
            shutdown_errors.push(format!(
                "failed to shut down managed vLLM engine: {error:#}"
            ));
        }
        if !shutdown_errors.is_empty() {
            return Err(engine_shutdown(shutdown_errors.join("; ")));
        }

        info!(model = %self.model, "vLLM backend cleanup complete");
        Ok(())
    }

    async fn setup_metrics(&self, ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        let inner = self.inner.read().await;
        let inner = inner
            .as_ref()
            .ok_or_else(|| engine_shutdown("vLLM backend has not been started"))?;

        ctx.metrics.add_expfmt_callback(Arc::new(|| {
            VLLM_METRICS
                .render()
                .map_err(|error| anyhow::anyhow!("failed to render vLLM metrics: {error}"))
        }));

        let client = inner.llm.engine_core_client();
        let model_name = client.model_name().to_string();
        let ranks = client
            .ready_responses()
            .into_iter()
            .enumerate()
            .map(|(engine_index, ready)| {
                MetricsRank::new(engine_index, model_name.clone(), ready.num_gpu_blocks)
            })
            .collect::<Vec<_>>();
        let dp_ranks = ranks.iter().map(|rank| rank.dp_rank).collect::<Vec<_>>();
        let cancel = self.cancel.clone();

        Ok(MetricsBindings {
            dp_ranks,
            on_publisher_ready: Some(Box::new(move |publisher| {
                spawn_metrics_snapshot_loop(publisher, ranks, cancel);
                Ok(())
            })),
        })
    }

    async fn health_check_payload(&self) -> Result<Option<serde_json::Value>, DynamoError> {
        if self.disaggregation_mode.is_decode() {
            // Decode probes need a real prefill handoff payload, which the
            // runtime health canary cannot synthesize locally.
            return Ok(None);
        }

        Ok(Some(serde_json::json!({
            "token_ids": [1],
            "stop_conditions": {"max_tokens": 1, "ignore_eos": true},
            "sampling_options": {"temperature": 0.0},
        })))
    }

    async fn supported_controls(&self) -> Result<Vec<String>, DynamoError> {
        Ok(control::supported_controls())
    }

    async fn engine_control(
        &self,
        control: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        if !control::is_supported(&control) {
            return Ok(control::unsupported_response(control));
        }
        let inner = self.inner.read().await;
        let Some(inner) = inner.as_ref() else {
            return Ok(control::error_response("engine is not initialized"));
        };
        control::engine_control(inner.llm.engine_core_client(), control, body).await
    }
}

#[derive(Clone)]
struct MetricsRank {
    dp_rank: u32,
    total_kv_blocks: u64,
    usage: F64Gauge,
    queries: U64Counter,
    hits: U64Counter,
}

impl MetricsRank {
    fn new(engine_index: usize, model_name: String, total_kv_blocks: u64) -> Self {
        let metrics = &VLLM_METRICS.scheduler;
        let labels = EngineLabels {
            model_name,
            engine: engine_index as u32,
        };
        Self {
            // TODO: currently vLLM's Rust engine-core client only supports local data-parallel engines,
            // so we use the engine index as the DP rank here.
            dp_rank: engine_index as u32,
            total_kv_blocks,
            usage: metrics.kv_cache_usage.get_or_create_owned(&labels),
            queries: metrics.prefix_cache_queries.get_or_create_owned(&labels),
            hits: metrics.prefix_cache_hits.get_or_create_owned(&labels),
        }
    }

    fn publish(&self, publisher: &SnapshotPublisher) {
        let usage = self.usage.get();
        let queries = self.queries.get();
        let hits = self.hits.get();
        let kv_cache_hit_rate = (queries > 0).then_some((hits as f64 / queries as f64) as f32);
        let kv_used_blocks = (self.total_kv_blocks as f64 * usage).round() as u64;

        publisher.publish(
            self.dp_rank,
            dynamo_backend_common::ComponentSnapshot {
                kv_used_blocks,
                kv_total_blocks: self.total_kv_blocks,
                gpu_cache_usage: usage as f32,
                kv_cache_hit_rate,
                dp_rank: self.dp_rank,
            },
        );
    }
}

fn spawn_metrics_snapshot_loop(
    publisher: Arc<SnapshotPublisher>,
    ranks: Vec<MetricsRank>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = ticker.tick() => {
                    for rank in &ranks {
                        rank.publish(&publisher);
                    }
                }
            }
        }
    });
}

impl ExtraEngineArgs {
    fn append_python_args(&self, python_args: &mut Vec<String>) {
        if let Some(block_size) = self.block_size {
            python_args.push("--block-size".to_string());
            python_args.push(block_size.to_string());
        }
        if let Some(max_num_seqs) = self.max_num_seqs {
            python_args.push("--max-num-seqs".to_string());
            python_args.push(max_num_seqs.to_string());
        }
        if let Some(max_num_batched_tokens) = self.max_num_batched_tokens {
            python_args.push("--max-num-batched-tokens".to_string());
            python_args.push(max_num_batched_tokens.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::ModelInput;

    use super::{VllmBackend, engine_max_num_seqs};

    #[test]
    fn local_capacity_multiplies_explicit_per_rank_limit_by_dp_size() {
        assert_eq!(engine_max_num_seqs(Some(128), 2), Some(256));
        assert_eq!(engine_max_num_seqs(None, 2), None);
        assert_eq!(engine_max_num_seqs(Some(u64::MAX), 2), Some(u64::MAX));
    }

    #[test]
    fn engine_config_reports_normalized_local_capacity() {
        let (engine, _config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-rs-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
            "--data-parallel-size".to_string(),
            "2".to_string(),
            "--max-num-seqs".to_string(),
            "128".to_string(),
        ]))
        .unwrap();
        let data_parallel_size = u32::try_from(engine.managed_engine.data_parallel_size).unwrap();

        let config = engine.build_engine_config(4096, Some(1024), data_parallel_size);
        let llm = config.llm.expect("vLLM backend must register LLM capacity");

        assert_eq!(llm.max_num_seqs, Some(128));
        assert_eq!(llm.data_parallel_size, Some(2));
        assert_eq!(llm.engine_max_num_seqs, Some(256));
    }

    #[test]
    fn from_args_auto_forwards_python_flags_without_separator() {
        let (engine, config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-rs-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
            "--namespace".to_string(),
            "ns".to_string(),
            "--served-model-name".to_string(),
            "served-qwen".to_string(),
            "--dyn-tool-call-parser".to_string(),
            "hermes".to_string(),
            "--dyn-reasoning-parser".to_string(),
            "qwen3".to_string(),
            "--exclude-tools-when-tool-choice-none=false".to_string(),
            "--dtype".to_string(),
            "float16".to_string(),
            "--data-parallel-size".to_string(),
            "2".to_string(),
            "--block-size".to_string(),
            "32".to_string(),
            "--max-num-seqs".to_string(),
            "128".to_string(),
            "--max-num-batched-tokens".to_string(),
            "4096".to_string(),
        ]))
        .unwrap();

        assert_eq!(config.namespace, "ns");
        assert_eq!(config.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(config.served_model_name.as_deref(), Some("served-qwen"));
        assert_eq!(engine.served_model_name.as_deref(), Some("served-qwen"));
        assert_eq!(config.model_input, ModelInput::Tokens);
        assert_eq!(config.tool_call_parser.as_deref(), Some("hermes"));
        assert_eq!(config.reasoning_parser.as_deref(), Some("qwen3"));
        assert_eq!(engine.reasoning_parser.as_deref(), Some("qwen3"));
        assert!(!config.exclude_tools_when_tool_choice_none);
        assert_eq!(engine.managed_engine.data_parallel_size, 2);
        assert_eq!(
            engine.managed_engine.python_args,
            vec!["--dtype", "float16"]
        );
        assert_eq!(engine.engine_ready_timeout_secs, 600);
        assert_eq!(engine.extra.block_size, Some(32));
        assert_eq!(engine.extra.max_num_seqs, Some(128));
        assert_eq!(engine.extra.max_num_batched_tokens, Some(4096));
        assert!(!engine.disaggregation_mode.is_decode());
    }

    #[test]
    fn engine_ready_timeout_arg_is_parsed_on_rust_side() {
        let (engine, _config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-rs-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
            "--engine-ready-timeout-secs".to_string(),
            "42".to_string(),
        ]))
        .unwrap();

        assert_eq!(engine.engine_ready_timeout_secs, 42);
        assert!(engine.managed_engine.python_args.is_empty());
    }

    #[test]
    fn extra_args_are_forwarded_to_python() {
        let (engine, _config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-rs-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
            "--block-size".to_string(),
            "32".to_string(),
            "--max-num-seqs".to_string(),
            "128".to_string(),
            "--max-num-batched-tokens".to_string(),
            "4096".to_string(),
        ]))
        .unwrap();
        let mut python_args = Vec::new();

        engine.extra.append_python_args(&mut python_args);

        assert_eq!(
            python_args,
            vec![
                "--block-size",
                "32",
                "--max-num-seqs",
                "128",
                "--max-num-batched-tokens",
                "4096"
            ]
        );
    }

    #[tokio::test]
    async fn health_check_payload_matches_python_token_canary_shape() {
        let (engine, _config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-rs-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
        ]))
        .unwrap();

        let payload = dynamo_backend_common::LLMEngine::health_check_payload(&engine)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(payload["token_ids"], serde_json::json!([1]));
        assert_eq!(payload["stop_conditions"]["max_tokens"], 1);
        assert_eq!(payload["stop_conditions"]["ignore_eos"], true);
        assert_eq!(payload["sampling_options"]["temperature"], 0.0);
    }

    #[tokio::test]
    async fn decode_health_check_is_disabled_until_pd_wiring_exists() {
        let (engine, _config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-rs-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
            "--disaggregation-mode".to_string(),
            "decode".to_string(),
        ]))
        .unwrap();

        let payload = dynamo_backend_common::LLMEngine::health_check_payload(&engine)
            .await
            .unwrap();

        assert!(payload.is_none());
    }

    #[tokio::test]
    #[ignore = "requires a configured Python vLLM engine and model"]
    async fn vllm_rs_backend_passes_conformance() {
        let model = std::env::var("DYNAMO_VLLM_BACKEND_CONFORMANCE_MODEL")
            .expect("set DYNAMO_VLLM_BACKEND_CONFORMANCE_MODEL to run this test");
        dynamo_backend_common::testing::run_conformance(move || {
            let (engine, _config) = VllmBackend::from_args(Some(vec![
                "dynamo-vllm-rs-backend".to_string(),
                model.clone(),
            ]))
            .unwrap();
            engine
        })
        .await
        .expect("conformance");
    }
}
