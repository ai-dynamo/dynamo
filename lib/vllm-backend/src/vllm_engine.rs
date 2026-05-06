// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-based native vLLM backend using the backend-common [`LLMEngine`] contract.

use std::ffi::OsString;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use clap::Parser;
use dynamo_backend_common::ModelInput;
use dynamo_backend_common::{
    AsyncEngineContext, CommonArgs, DynamoError, EngineConfig, LLMEngine, LLMEngineOutput,
    LLMEngineOutputExt, PreprocessedRequest, WorkerConfig, usage,
};
use futures::{StreamExt, stream::BoxStream};
use tokio::sync::RwLock;
use tracing::{debug, info};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, TransportMode};
use vllm_llm::Llm;
use vllm_managed_engine::ManagedEngineHandle;
use vllm_managed_engine::cli::{ManagedEngineArgs, repartition_managed_engine_args};

use crate::convert::{lower_request, map_output};
use crate::error::{backend_unknown, cannot_connect, clap_error, engine_shutdown, invalid_arg};

#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Dynamo vLLM backend based on Rust frontend and engine-core client."
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

/// Dynamo backend implementation backed by a managed Python vLLM engine-core.
///
/// The backend consumes tokenized [`PreprocessedRequest`] values produced by
/// Dynamo preprocessing and streams token-level [`LLMEngineOutput`] values back
/// through the backend-common worker runtime.
pub struct VllmBackend {
    model: String,
    managed_engine: ManagedEngineArgs,
    extra: ExtraEngineArgs,
    inner: RwLock<Option<Inner>>,
}

struct Inner {
    engine_handle: ManagedEngineHandle,
    llm: Llm,
}

impl VllmBackend {
    fn new(model: String, managed_engine: ManagedEngineArgs, extra: ExtraEngineArgs) -> Self {
        Self {
            model,
            managed_engine,
            extra,
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

        let engine = Self::new(args.model.clone(), args.managed_engine, args.extra.clone());
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            model_name: args.model,
            served_model_name: args.served_model_name,
            model_input: ModelInput::Tokens,
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for VllmBackend {
    async fn start(&self) -> Result<EngineConfig, DynamoError> {
        let mut inner = self.inner.write().await;
        if inner.is_some() {
            return Err(engine_shutdown("vLLM backend has already been started"));
        }

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
            let mut config =
                self.managed_engine
                    .clone()
                    .into_config(self.model.clone(), None, handshake_port);
            self.extra.append_python_args(&mut config.python_args);
            config
        };

        let handshake_address = managed_config.handshake_address();
        let advertised_host = managed_config.handshake_host.clone();
        let engine_count = managed_config.data_parallel_size;

        info!(
            %handshake_address,
            engine_count,
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
                ready_timeout: Duration::from_secs(30),
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
        let llm = Llm::new(client);

        *inner = Some(Inner { engine_handle, llm });

        info!(
            model = %self.model,
            engine_count,
            context_length = ?context_length,
            total_kv_blocks = ?total_kv_blocks,
            "vLLM backend started"
        );

        Ok(EngineConfig {
            model: self.model.clone(),
            served_model_name: Some(self.model.clone()),
            context_length,
            kv_cache_block_size: self.extra.block_size,
            total_kv_blocks,
            max_num_seqs: self.extra.max_num_seqs,
            max_num_batched_tokens: self.extra.max_num_batched_tokens,
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, LLMEngineOutput>, DynamoError> {
        let request_id = ctx.id().to_string();
        let prompt_tokens = request.token_ids.len() as u32;

        let mut output_stream = {
            let inner = self.inner.read().await;
            let inner = inner
                .as_ref()
                .ok_or_else(|| engine_shutdown("vLLM backend has not been started"))?;
            let max_model_len = inner.llm.engine_core_client().max_model_len();
            let generate_request = lower_request(request_id, request, max_model_len)?;

            inner
                .llm
                .generate(generate_request)
                .await
                .map_err(|e| backend_unknown(format!("failed to submit vLLM request: {e}")))?
        };

        Ok(Box::pin(async_stream::stream! {
            let mut completion_tokens = 0_u32;
            loop {
                tokio::select! {
                    _ = ctx.stopped() => {
                        debug!(request_id = %ctx.id(), "vLLM backend request cancelled");
                        yield LLMEngineOutput::cancelled()
                            .with_usage(usage(prompt_tokens, completion_tokens));
                        break;
                    }
                    next = output_stream.next() => {
                        let Some(next) = next else {
                            yield LLMEngineOutput::error(
                                "vLLM backend stream ended before a terminal output".to_string()
                            );
                            break;
                        };

                        match next {
                            Ok(output) => {
                                completion_tokens = completion_tokens
                                    .saturating_add(output.token_ids.len() as u32);
                                let finished = output.finished();
                                let mapped = map_output(output, prompt_tokens, completion_tokens);
                                yield mapped;
                                if finished {
                                    break;
                                }
                            }
                            Err(error) => {
                                yield LLMEngineOutput::error(
                                    format!("vLLM backend stream failed: {error}")
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        let Some(Inner { engine_handle, llm }) = self.inner.write().await.take() else {
            return Ok(());
        };

        info!(model = %self.model, "shutting down vLLM backend");

        let llm_result = llm.shutdown().await;
        let engine_result = engine_handle.shutdown(Duration::from_secs(0)).await;

        if let Err(error) = llm_result {
            return Err(engine_shutdown(format!(
                "failed to shut down vLLM engine-core client: {error}"
            )));
        }
        if let Err(error) = engine_result {
            return Err(engine_shutdown(format!(
                "failed to shut down managed vLLM engine: {error:#}"
            )));
        }
        info!(model = %self.model, "vLLM backend cleanup complete");
        Ok(())
    }
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

    use super::VllmBackend;

    #[test]
    fn from_args_auto_forwards_python_flags_without_separator() {
        let (engine, config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
            "--namespace".to_string(),
            "ns".to_string(),
            "--served-model-name".to_string(),
            "served-qwen".to_string(),
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
        assert_eq!(config.model_input, ModelInput::Tokens);
        assert_eq!(engine.managed_engine.data_parallel_size, 2);
        assert_eq!(
            engine.managed_engine.python_args,
            vec!["--dtype", "float16"]
        );
        assert_eq!(engine.extra.block_size, Some(32));
        assert_eq!(engine.extra.max_num_seqs, Some(128));
        assert_eq!(engine.extra.max_num_batched_tokens, Some(4096));
    }

    #[test]
    fn extra_args_are_forwarded_to_python() {
        let (engine, _config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-backend".to_string(),
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
    #[ignore = "requires a configured Python vLLM engine and model"]
    async fn vllm_backend_passes_conformance() {
        let model = std::env::var("DYNAMO_VLLM_BACKEND_CONFORMANCE_MODEL")
            .expect("set DYNAMO_VLLM_BACKEND_CONFORMANCE_MODEL to run this test");
        let (engine, _config) =
            VllmBackend::from_args(Some(vec!["dynamo-vllm-backend".to_string(), model])).unwrap();

        dynamo_backend_common::testing::run_conformance(engine)
            .await
            .expect("conformance");
    }
}
