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
    /// Non-CPU-to-CPU capacity ratio for device-aware weighted routing.
    #[serde(default)]
    pub encoder_cuda_to_cpu_ratio: Option<usize>,
}

impl RouterConfig {
    pub fn new(router_mode: RouterMode, kv_router_config: KvRouterConfig) -> Self {
        Self {
            router_mode,
            kv_router_config,
            load_threshold_config: LoadThresholdConfig::default(),
            enforce_disagg: false,
            session_affinity_ttl_secs: None,
            encoder_cuda_to_cpu_ratio: None,
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

    pub fn with_encoder_cuda_to_cpu_ratio(mut self, ratio: usize) -> Self {
        self.encoder_cuda_to_cpu_ratio = Some(ratio);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_leaves_encoder_ratio_unset() {
        let config = RouterConfig::new(RouterMode::Random, KvRouterConfig::default());
        assert_eq!(config.encoder_cuda_to_cpu_ratio, None);
    }

    #[test]
    fn builder_sets_encoder_ratio() {
        let config = RouterConfig::new(RouterMode::Random, KvRouterConfig::default())
            .with_encoder_cuda_to_cpu_ratio(4);
        assert_eq!(config.encoder_cuda_to_cpu_ratio, Some(4));
    }

    #[test]
    fn encoder_ratio_defaults_to_none_when_absent_from_json() {
        let mut value = serde_json::to_value(RouterConfig::default()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("encoder_cuda_to_cpu_ratio")
            .unwrap();
        let config: RouterConfig = serde_json::from_value(value).unwrap();
        assert_eq!(config.encoder_cuda_to_cpu_ratio, None);
    }

    #[test]
    fn encoder_ratio_round_trips_through_json() {
        let config = RouterConfig::default().with_encoder_cuda_to_cpu_ratio(3);
        let text = serde_json::to_string(&config).unwrap();
        let parsed: RouterConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed.encoder_cuda_to_cpu_ratio, Some(3));
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
