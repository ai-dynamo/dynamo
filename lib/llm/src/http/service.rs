// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! HTTP Service for Nova LLM
//!
//! The primary purpose of this crate is to service the nova-llm-protocols via OpenAI compatible HTTP endpoints. This component
//! is meant to be a gateway/ingress into the Nova LLM Distributed Runtime.
//!
//! In order to create a common pattern, the HttpService forwards the incoming OAI Chat Request or OAI Completion Request to the
//! to a model-specific engines.  The engines can be attached and detached dynamically using the [`ModelManager`].
//!
//! Note: All requests, whether the client requests `stream=true` or `stream=false`, are propagated downstream as `stream=true`.
//! This enables use to handle only 1 pattern of request-response in the downstream services. Non-streaming user requests are
//! aggregated by the HttpService and returned as a single response.
//!
//! TODO(): Add support for model-specific metadata and status. Status will allow us to return a 503 when the model is supposed
//! to be ready, but there is a problem with the model.
//!
//! The [`service_v2::HttpService`] can be further extended to host any [`axum::Router`] using the [`service_v2::HttpServiceConfigBuilder`].

mod openai;

pub mod discovery;
pub mod error;
pub mod metrics;
pub mod service_v2;

// #[cfg(feature = "py3")]
// pub mod py3;

pub use async_trait::async_trait;
pub use axum;
pub use error::ServiceHttpError;
pub use metrics::Metrics;

use crate::types::openai::{
    chat_completions::OpenAIChatCompletionsStreamingEngine,
    completions::OpenAICompletionsStreamingEngine,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Clone)]
pub struct ModelManager {
    state: Arc<DeploymentState>,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    pub fn new() -> Self {
        let state = Arc::new(DeploymentState::new());
        Self { state }
    }

    pub fn state(&self) -> Arc<DeploymentState> {
        self.state.clone()
    }

    pub fn has_model_any(&self, model: &str) -> bool {
        self.state
            .chat_completion_engines
            .lock()
            .unwrap()
            .contains(model)
            || self
                .state
                .completion_engines
                .lock()
                .unwrap()
                .contains(model)
    }

    pub fn list_chat_completions_models(&self) -> Vec<String> {
        self.state.chat_completion_engines.lock().unwrap().list()
    }

    pub fn list_completions_models(&self) -> Vec<String> {
        self.state.completion_engines.lock().unwrap().list()
    }

    pub fn add_completions_model(
        &self,
        model: &str,
        engine: OpenAICompletionsStreamingEngine,
    ) -> Result<(), ServiceHttpError> {
        let mut clients = self.state.completion_engines.lock().unwrap();
        clients.add(model, engine)
    }

    pub fn add_chat_completions_model(
        &self,
        model: &str,
        engine: OpenAIChatCompletionsStreamingEngine,
    ) -> Result<(), ServiceHttpError> {
        let mut clients = self.state.chat_completion_engines.lock().unwrap();
        clients.add(model, engine)
    }

    pub fn remove_completions_model(&self, model: &str) -> Result<(), ServiceHttpError> {
        let mut clients = self.state.completion_engines.lock().unwrap();
        clients.remove(model)
    }

    pub fn remove_chat_completions_model(&self, model: &str) -> Result<(), ServiceHttpError> {
        let mut clients = self.state.chat_completion_engines.lock().unwrap();
        clients.remove(model)
    }

    /// Get the Prometheus [`Metrics`] object which tracks request counts and inflight requests
    pub fn metrics(&self) -> Arc<Metrics> {
        self.state.metrics.clone()
    }
}

struct ModelEngines<E> {
    /// Optional default model name
    default: Option<String>,
    engines: HashMap<String, E>,
}

impl<E> Default for ModelEngines<E> {
    fn default() -> Self {
        Self {
            default: None,
            engines: HashMap::new(),
        }
    }
}

impl<E> ModelEngines<E> {
    #[allow(dead_code)]
    fn set_default(&mut self, model: &str) {
        self.default = Some(model.to_string());
    }

    #[allow(dead_code)]
    fn clear_default(&mut self) {
        self.default = None;
    }

    fn add(&mut self, model: &str, engine: E) -> Result<(), ServiceHttpError> {
        if self.engines.contains_key(model) {
            return Err(ServiceHttpError::ModelAlreadyExists(model.to_string()));
        }
        self.engines.insert(model.to_string(), engine);
        Ok(())
    }

    fn remove(&mut self, model: &str) -> Result<(), ServiceHttpError> {
        if self.engines.remove(model).is_none() {
            return Err(ServiceHttpError::ModelNotFound(model.to_string()));
        }
        Ok(())
    }

    fn get(&self, model: &str) -> Option<&E> {
        self.engines.get(model)
    }

    fn contains(&self, model: &str) -> bool {
        self.engines.contains_key(model)
    }

    fn list(&self) -> Vec<String> {
        self.engines.keys().map(|k| k.to_owned()).collect()
    }
}

/// The DeploymentState is a global state that is shared across all the workers
/// this provides set of known clients to Engines
pub struct DeploymentState {
    completion_engines: Arc<Mutex<ModelEngines<OpenAICompletionsStreamingEngine>>>,
    chat_completion_engines: Arc<Mutex<ModelEngines<OpenAIChatCompletionsStreamingEngine>>>,
    metrics: Arc<Metrics>,
    sse_keep_alive: Option<Duration>,
}

impl DeploymentState {
    fn new() -> Self {
        Self {
            completion_engines: Arc::new(Mutex::new(ModelEngines::default())),
            chat_completion_engines: Arc::new(Mutex::new(ModelEngines::default())),
            metrics: Arc::new(Metrics::default()),
            sse_keep_alive: None,
        }
    }

    fn get_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAICompletionsStreamingEngine, ServiceHttpError> {
        self.completion_engines
            .lock()
            .unwrap()
            .get(model)
            .cloned()
            .ok_or(ServiceHttpError::ModelNotFound(model.to_string()))
    }

    fn get_chat_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIChatCompletionsStreamingEngine, ServiceHttpError> {
        self.chat_completion_engines
            .lock()
            .unwrap()
            .get(model)
            .cloned()
            .ok_or(ServiceHttpError::ModelNotFound(model.to_string()))
    }
}

/// Documentation for a route
#[derive(Debug, Clone)]
pub struct RouteDoc {
    method: axum::http::Method,
    path: String,
}

impl std::fmt::Display for RouteDoc {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {}", self.method, self.path)
    }
}

impl RouteDoc {
    pub fn new<T: Into<String>>(method: axum::http::Method, path: T) -> Self {
        RouteDoc {
            method,
            path: path.into(),
        }
    }
}
