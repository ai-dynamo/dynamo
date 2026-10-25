// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use dynamo_global_router::PoolSnapshot;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_type::ModelType;
use dynamo_llm::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;
use dynamo_llm::types::openai::completions::OpenAICompletionsStreamingEngine;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::forward::PoolForwardingEngine;

/// A model registered by hand needs a card checksum. Every model here is the
/// same forwarding engine, so they share one.
const FORWARDING_CHECKSUM: &str = "global-ingress-forwarding";

/// Registers the forwarding engine under every model the pool catalog
/// serves, and enables the HTTP endpoints those models need.
///
/// The Global Frontend is the only writer of its model manager.
pub struct ModelSync {
    service: HttpService,
    engine: Arc<PoolForwardingEngine>,
    served: BTreeSet<String>,
    /// Whether the endpoints were last enabled or disabled. `None` until the
    /// first snapshot, so that one always sets them.
    endpoints_enabled: Option<bool>,
}

impl ModelSync {
    pub fn new(service: HttpService, engine: Arc<PoolForwardingEngine>) -> Self {
        Self {
            service,
            engine,
            served: BTreeSet::new(),
            endpoints_enabled: None,
        }
    }

    /// Serve exactly the models in `snapshot`.
    pub fn apply(&mut self, snapshot: &PoolSnapshot) {
        let wanted = snapshot.models();
        let manager = self.service.model_manager();

        let gone = self
            .served
            .iter()
            .filter(|model| !wanted.contains(model.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        for model in gone {
            manager.remove_model(&model);
            self.served.remove(&model);
            tracing::info!(model, "no pool serves the model any more");
        }

        for model in wanted {
            if self.served.contains(model) {
                continue;
            }
            match self.register(model) {
                Ok(()) => {
                    self.served.insert(model.to_owned());
                    tracing::info!(model, "serving model");
                }
                Err(error) => {
                    // Drop a half-registered model so it is never served
                    // through only some of its APIs.
                    manager.remove_model(model);
                    tracing::error!(model, %error, "cannot serve model");
                }
            }
        }

        let enabled = !self.served.is_empty();
        if self.endpoints_enabled == Some(enabled) {
            return;
        }
        self.endpoints_enabled = Some(enabled);
        let endpoints = (ModelType::Chat | ModelType::Completions)
            .as_endpoint_types_with_anthropic(self.service.anthropic_api_enabled());
        for endpoint in endpoints {
            if let Err(error) = self.service.enable_model_endpoint(endpoint, enabled) {
                tracing::error!(?endpoint, %error, "cannot update endpoint");
            }
        }
    }

    /// Apply every snapshot the catalog publishes until `cancel` fires or the
    /// catalog is dropped. Models stay registered after either.
    pub async fn follow(
        mut self,
        mut pools: watch::Receiver<Arc<PoolSnapshot>>,
        cancel: CancellationToken,
    ) {
        loop {
            let snapshot = pools.borrow_and_update().clone();
            self.apply(&snapshot);
            tokio::select! {
                _ = cancel.cancelled() => return,
                changed = pools.changed() => {
                    if changed.is_err() {
                        return;
                    }
                }
            }
        }
    }

    fn register(&self, model: &str) -> anyhow::Result<()> {
        let manager = self.service.model_manager();
        let chat: OpenAIChatCompletionsStreamingEngine = self.engine.clone();
        manager.add_chat_completions_model(model, FORWARDING_CHECKSUM, chat)?;
        let completions: OpenAICompletionsStreamingEngine = self.engine.clone();
        manager.add_completions_model(model, FORWARDING_CHECKSUM, completions)?;
        Ok(())
    }
}
