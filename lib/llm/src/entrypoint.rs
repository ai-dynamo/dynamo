// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The entrypoint module provides tools to build a Dynamo runner.
//! - Create an EngineConfig of the engine (potentially auto-discovered) to execute
//! - Connect it to an Input

pub mod input;
pub use input::{PreprocessedRouting, build_preprocessed_routing, http::HttpFrontend};

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use dynamo_runtime::discovery::ModelCardInstanceId;
use dynamo_scheduler::host::PrefillLoadEstimator;

use crate::{
    admission_gate::WorkerAdmissionGate, backend::ExecutionContext, engines::StreamingEngine,
    types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine,
};
use dynamo_model_card::{local_model::LocalModel, model_card::ModelDeploymentCard};

/// Callback type for chat engine factory (async)
pub type PrefillRoutedEngine = dynamo_runtime::pipeline::ServiceEngine<
    dynamo_runtime::pipeline::SingleIn<crate::protocols::common::preprocessor::PreprocessedRequest>,
    dynamo_runtime::pipeline::ManyOut<
        crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>,
    >,
>;

pub type ChatEngineFactoryCallback = Arc<
    dyn Fn(
            ModelCardInstanceId,
            ModelDeploymentCard,
            PrefillRoutedEngine,
        ) -> Pin<
            Box<dyn Future<Output = anyhow::Result<OpenAIChatCompletionsStreamingEngine>> + Send>,
        > + Send
        + Sync,
>;

#[derive(Clone)]
pub enum EngineConfig {
    /// Remote networked engines that we discover via etcd
    Dynamic {
        model: Box<LocalModel>,
        chat_engine_factory: Option<ChatEngineFactoryCallback>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    },

    /// A Text engine receives text, does it's own tokenization and prompt formatting.
    InProcessText {
        engine: Arc<dyn StreamingEngine>,
        model: Box<LocalModel>,
    },

    /// A token engine receives preprocessed requests and emits raw engine output.
    /// Distributed endpoints forward that output; standalone HTTP, gRPC, and
    /// interactive inputs apply local pre/post-processing.
    InProcessTokens {
        engine: ExecutionContext,
        model: Box<LocalModel>,
        is_prefill: bool,
        is_decode: bool,
        /// The admission gate of the worker this config is: every endpoint
        /// serving it, through this config or a clone, admits through it.
        admission_gate: WorkerAdmissionGate,
    },
}

impl EngineConfig {
    pub fn local_model(&self) -> &LocalModel {
        use EngineConfig::*;
        match self {
            Dynamic { model, .. } => model,
            InProcessText { model, .. } => model,
            InProcessTokens { model, .. } => model,
        }
    }

    pub fn chat_engine_factory(&self) -> Option<&ChatEngineFactoryCallback> {
        match self {
            EngineConfig::Dynamic {
                chat_engine_factory,
                ..
            } => chat_engine_factory.as_ref(),
            _ => None,
        }
    }
}
