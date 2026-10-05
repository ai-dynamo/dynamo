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

use dynamo_kv_router::{PrefillLoadEstimator, config::KvRouterConfig};
use dynamo_runtime::{discovery::ModelCardInstanceId, pipeline::RouterMode};
use serde::{Deserialize, Serialize};

use crate::{
    backend::ExecutionContext, discovery::LoadThresholdConfig, engines::StreamingEngine,
    local_model::LocalModel, model_card::ModelDeploymentCard,
    session_affinity::SessionAffinityMode,
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

/// Identity declared by the frontend registering a chat processor, never by a
/// worker card or inferred from a callback's name. This is not a capability grant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChatProcessorIdentity {
    #[default]
    Custom,
    Vllm,
    Sglang,
}

impl ChatProcessorIdentity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Custom => "custom",
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
        }
    }
}

impl std::str::FromStr for ChatProcessorIdentity {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "custom" => Ok(Self::Custom),
            "vllm" => Ok(Self::Vllm),
            "sglang" => Ok(Self::Sglang),
            _ => Err("chat_engine_factory_identity must be custom, vllm, or sglang"),
        }
    }
}

/// Keep the registration's identity attached to the callback across pipeline
/// construction. Custom callbacks must not inherit a built-in processor profile.
#[derive(Clone)]
pub struct ChatEngineFactory {
    callback: ChatEngineFactoryCallback,
    identity: ChatProcessorIdentity,
}

impl ChatEngineFactory {
    pub fn new(callback: ChatEngineFactoryCallback, identity: ChatProcessorIdentity) -> Self {
        Self { callback, identity }
    }

    pub fn identity(&self) -> ChatProcessorIdentity {
        self.identity
    }

    pub async fn create(
        &self,
        instance: ModelCardInstanceId,
        card: ModelDeploymentCard,
        routed_engine: PrefillRoutedEngine,
    ) -> anyhow::Result<OpenAIChatCompletionsStreamingEngine> {
        (self.callback)(instance, card, routed_engine).await
    }
}

impl From<ChatEngineFactoryCallback> for ChatEngineFactory {
    fn from(callback: ChatEngineFactoryCallback) -> Self {
        Self::new(callback, ChatProcessorIdentity::Custom)
    }
}

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
    #[serde(default)]
    pub session_affinity_mode: SessionAffinityMode,
}

impl RouterConfig {
    pub fn new(router_mode: RouterMode, kv_router_config: KvRouterConfig) -> Self {
        Self {
            router_mode,
            kv_router_config,
            load_threshold_config: LoadThresholdConfig::default(),
            enforce_disagg: false,
            session_affinity_ttl_secs: None,
            session_affinity_mode: SessionAffinityMode::Hard,
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
}

#[derive(Clone)]
pub enum EngineConfig {
    /// Remote networked engines that we discover via etcd
    Dynamic {
        model: Box<LocalModel>,
        chat_engine_factory: Option<ChatEngineFactory>,
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

    pub fn chat_engine_factory(&self) -> Option<&ChatEngineFactory> {
        match self {
            EngineConfig::Dynamic {
                chat_engine_factory,
                ..
            } => chat_engine_factory.as_ref(),
            _ => None,
        }
    }
}
