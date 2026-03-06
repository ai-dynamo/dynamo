// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The entrypoint module provides tools to build a Dynamo runner.
//! - Create an EngineConfig of the engine (potentially auto-discovered) to execute
//! - Connect it to an Input

pub mod input;
pub use input::{build_routed_pipeline, build_routed_pipeline_with_preprocessor};

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use dynamo_runtime::{discovery::ModelCardInstanceId, pipeline::RouterMode};

use crate::{
    backend::ExecutionContext, discovery::LoadThresholdConfig, engines::StreamingEngine,
    kv_router::KvRouterConfig, local_model::LocalModel, model_card::ModelDeploymentCard,
    types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine,
};

/// Callback type for chat engine factory (async)
pub type ChatEngineFactoryCallback = Arc<
    dyn Fn(
            ModelCardInstanceId,
            ModelDeploymentCard,
        ) -> Pin<
            Box<dyn Future<Output = anyhow::Result<OpenAIChatCompletionsStreamingEngine>> + Send>,
        > + Send
        + Sync,
>;

#[derive(Debug, Clone, Default)]
pub struct RouterConfig {
    pub router_mode: RouterMode,
    pub kv_router_config: KvRouterConfig,
    /// Load threshold configuration for busy detection
    pub load_threshold_config: LoadThresholdConfig,
    pub enforce_disagg: bool,
}

impl RouterConfig {
    pub fn new(router_mode: RouterMode, kv_router_config: KvRouterConfig) -> Self {
        Self {
            router_mode,
            kv_router_config,
            load_threshold_config: LoadThresholdConfig::default(),
            enforce_disagg: false,
        }
    }

    pub fn with_load_threshold_config(mut self, config: LoadThresholdConfig) -> Self {
        self.load_threshold_config = config;
        self
    }

    pub fn with_enforce_disagg(mut self, enforce_disagg: bool) -> Self {
        self.enforce_disagg = enforce_disagg;
        self
    }
}

#[derive(Clone)]
pub enum EngineConfig {
    /// Remote networked engines that we discover via etcd
    Dynamic {
        model: Box<LocalModel>,
        chat_engine_factory: Option<ChatEngineFactoryCallback>,
    },

    /// A Text engine receives text, does it's own tokenization and prompt formatting.
    InProcessText {
        engine: Arc<dyn StreamingEngine>,
        model: Box<LocalModel>,
    },

    /// A Tokens engine receives tokens, expects to be wrapped with pre/post processors that handle tokenization.
    InProcessTokens {
        engine: ExecutionContext,
        model: Box<LocalModel>,
        is_prefill: bool,
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
