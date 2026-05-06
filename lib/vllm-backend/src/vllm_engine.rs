// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM backend skeleton using the backend-common [`LLMEngine`] contract.

use std::ffi::OsString;
use std::sync::Arc;

use async_trait::async_trait;
use clap::Parser;
use dynamo_backend_common::{
    AsyncEngineContext, BackendError, CommonArgs, DynamoError, EngineConfig, ErrorType, LLMEngine,
    LLMEngineOutput, PreprocessedRequest, WorkerConfig,
};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use vllm_llm::Llm;
use vllm_managed_engine::ManagedEngineHandle;
use vllm_managed_engine::cli::{ManagedEngineArgs, repartition_managed_engine_args};

#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Dynamo vLLM backend — serves vLLM engine-core through the backend-common LLMEngine trait."
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Model identifier or local model directory passed to vLLM.
    #[arg(value_name = "MODEL")]
    model: String,

    /// Managed Python headless-engine arguments.
    #[command(flatten)]
    managed_engine: ManagedEngineArgs,
}

pub struct VllmBackend {
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    managed_engine: ManagedEngineArgs,
    #[allow(dead_code)]
    engine_handle: OnceCell<ManagedEngineHandle>,
    #[allow(dead_code)]
    llm: OnceCell<Llm>,
}

impl VllmBackend {
    fn new(model: String, managed_engine: ManagedEngineArgs) -> Self {
        Self {
            model,
            managed_engine,
            engine_handle: OnceCell::new(),
            llm: OnceCell::new(),
        }
    }

    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let raw_args: Vec<OsString> = match argv {
            Some(a) => a.into_iter().map(Into::into).collect(),
            None => std::env::args_os().collect(),
        };
        let repartitioned_args =
            repartition_managed_engine_args::<Args>(&raw_args, None).map_err(clap_error)?;
        let args =
            Args::try_parse_from(repartitioned_args).map_err(|e| invalid_arg(e.to_string()))?;

        let engine = Self::new(args.model.clone(), args.managed_engine);
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            model_name: args.model,
            served_model_name: None,
            ..Default::default()
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for VllmBackend {
    async fn start(&self) -> Result<EngineConfig, DynamoError> {
        todo!("start the managed vLLM engine and connect vllm_llm::Llm")
    }

    async fn generate(
        &self,
        _request: PreprocessedRequest,
        _ctx: Arc<dyn AsyncEngineContext>,
    ) -> Result<BoxStream<'static, LLMEngineOutput>, DynamoError> {
        todo!("translate PreprocessedRequest into vllm_llm::GenerateRequest and stream outputs")
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        todo!("shut down vllm_llm::Llm and the managed vLLM engine")
    }
}

fn invalid_arg(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(msg)
        .build()
}

fn clap_error(error: clap::Error) -> DynamoError {
    invalid_arg(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::VllmBackend;

    #[test]
    fn from_args_auto_forwards_python_flags_without_separator() {
        let (engine, config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
            "--namespace".to_string(),
            "ns".to_string(),
            "--dtype".to_string(),
            "float16".to_string(),
            "--data-parallel-size".to_string(),
            "2".to_string(),
        ]))
        .unwrap();

        assert_eq!(config.namespace, "ns");
        assert_eq!(config.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(engine.managed_engine.data_parallel_size, 2);
        assert_eq!(
            engine.managed_engine.python_args,
            vec!["--dtype", "float16"]
        );
    }
}
