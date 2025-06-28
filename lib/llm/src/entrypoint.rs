// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The entrypoint module provides tools to build a Dynamo runner.
//! - Create an EngineConfig of the engine (potentially auto-discovered) to execute
//! - Connect it to an Input

pub mod input;

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
}

impl RouterConfig {
    pub fn new(router_mode: RouterMode, kv_router_config: KvRouterConfig) -> Self {
        Self {
            router_mode,
            kv_router_config,
        }
    }
}

pub enum EngineConfig {
    /// Remote networked engines
    Dynamic(Box<LocalModel>),

    /// A Full service engine does it's own tokenization and prompt formatting.
    StaticFull {
        engine: Arc<dyn StreamingEngine>,
        model: Box<LocalModel>,
    },

    /// A core engine expects to be wrapped with pre/post processors that handle tokenization.
    StaticCore {
        engine: ExecutionContext,
        model: Box<LocalModel>,
    },
}

impl EngineConfig {
    fn local_model(&self) -> &LocalModel {
        use EngineConfig::*;
        match self {
            Dynamic(lm) => lm,
            StaticFull { model, .. } => model,
            StaticCore { model, .. } => model,
        }
    }
}
