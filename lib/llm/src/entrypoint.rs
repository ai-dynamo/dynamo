// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The entrypoint module provides tools to build a Dynamo runner.
//! - Create an EngineConfig of the engine (potentially auto-discovered) to execute
//! - Connect it to an Input

pub mod input;
pub use input::{build_routed_pipeline, build_routed_pipeline_with_preprocessor};

use std::sync::Arc;

use dynamo_runtime::pipeline::RouterMode;

use crate::{
    backend::ExecutionContext, engines::StreamingEngine, kv_router::KvRouterConfig,
    local_model::LocalModel,
};

#[derive(Debug, Clone, Default)]
pub struct RouterConfig {
    pub router_mode: RouterMode,
    pub kv_router_config: KvRouterConfig,
    /// Threshold for active decode blocks utilization (0.0-1.0)
    pub active_decode_blocks_threshold: Option<f64>,
    /// Threshold for active prefill tokens utilization (literal token count)
    pub active_prefill_tokens_threshold: Option<u64>,
    pub enforce_disagg: bool,
}

impl RouterConfig {
    pub fn new(router_mode: RouterMode, kv_router_config: KvRouterConfig) -> Self {
        Self {
            router_mode,
            kv_router_config,
            active_decode_blocks_threshold: None,
            active_prefill_tokens_threshold: None,
            enforce_disagg: false,
        }
    }

    pub fn with_active_decode_blocks_threshold(mut self, threshold: Option<f64>) -> Self {
        self.active_decode_blocks_threshold = threshold;
        self
    }

    pub fn with_active_prefill_tokens_threshold(mut self, threshold: Option<u64>) -> Self {
        self.active_prefill_tokens_threshold = threshold;
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
    Dynamic(Box<LocalModel>),

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
    fn local_model(&self) -> &LocalModel {
        use EngineConfig::*;
        match self {
            Dynamic(lm) => lm,
            InProcessText { model, .. } => model,
            InProcessTokens { model, .. } => model,
        }
    }
}
