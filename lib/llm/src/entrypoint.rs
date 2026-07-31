// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The entrypoint module provides tools to build a Dynamo runner.
//! - Create an EngineConfig of the engine (potentially auto-discovered) to execute
//! - Connect it to an Input

pub mod input;
pub use input::{PreprocessedRouting, build_preprocessed_routing};

use std::pin::Pin;
use std::sync::Arc;
use std::{env, future::Future};

use dynamo_kv_router::{PrefillLoadEstimator, config::KvRouterConfig};
use dynamo_runtime::{
    config::environment_names::router::DYN_ENCODER_CUDA_TO_CPU_RATIO,
    discovery::ModelCardInstanceId,
    pipeline::{RouterMode, resolve_non_cpu_to_cpu_ratio},
};
use serde::{Deserialize, Serialize};

use crate::{
    backend::ExecutionContext, discovery::LoadThresholdConfig, engines::StreamingEngine,
    local_model::LocalModel, model_card::ModelDeploymentCard,
    types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine,
};

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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterConfig {
    pub router_mode: RouterMode,
    pub kv_router_config: KvRouterConfig,
    /// Load threshold configuration for overload detection
    pub load_threshold_config: LoadThresholdConfig,
    /// Deprecated compatibility field. Routing and readiness ignore this value.
    #[serde(default)]
    pub enforce_disagg: bool,
    #[serde(default)]
    pub session_affinity_ttl_secs: Option<u64>,
    /// Optional non-CPU-to-CPU capacity ratio for device-aware weighted routing.
    /// When unset or below 1, the legacy environment variable fallback is used.
    #[serde(default)]
    pub non_cpu_to_cpu_ratio: Option<usize>,
}

impl RouterConfig {
    pub fn new(router_mode: RouterMode, kv_router_config: KvRouterConfig) -> Self {
        Self {
            router_mode,
            kv_router_config,
            load_threshold_config: LoadThresholdConfig::default(),
            enforce_disagg: false,
            session_affinity_ttl_secs: None,
            non_cpu_to_cpu_ratio: None,
        }
    }

    pub fn with_load_threshold_config(mut self, config: LoadThresholdConfig) -> Self {
        self.load_threshold_config = config;
        self
    }

    #[deprecated(
        note = "enforce_disagg is ignored; topology and readiness come from registered worker types"
    )]
    pub fn with_enforce_disagg(mut self, enforce_disagg: bool) -> Self {
        self.enforce_disagg = enforce_disagg;
        self
    }

    pub fn with_session_affinity_ttl_secs(mut self, ttl_secs: u64) -> Self {
        self.session_affinity_ttl_secs = Some(ttl_secs);
        self
    }

    pub fn with_non_cpu_to_cpu_ratio(mut self, ratio: usize) -> Self {
        self.non_cpu_to_cpu_ratio = Some(ratio);
        self
    }

    pub fn resolved_non_cpu_to_cpu_ratio(&self) -> usize {
        resolve_non_cpu_to_cpu_ratio(
            self.non_cpu_to_cpu_ratio,
            env::var(DYN_ENCODER_CUDA_TO_CPU_RATIO).ok().as_deref(),
        )
    }
}

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

    /// A Tokens engine receives tokens, expects to be wrapped with pre/post processors that handle tokenization.
    InProcessTokens {
        engine: ExecutionContext,
        model: Box<LocalModel>,
        is_prefill: bool,
        is_decode: bool,
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
