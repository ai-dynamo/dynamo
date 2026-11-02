// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::model_card::decision_profile::DecisionKinds;

use crate::{
    discovery::{DecisionResources, ScoreFactory, TokenizerLoad, WorkerSet},
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    tokenizers::Tokenizer,
};
use dynamo_runtime::{component::Client, pipeline::RouterMode};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub(super) struct DecisionPlan {
    advertised: DecisionKinds,
    resources: Option<DecisionResources>,
}

impl DecisionPlan {
    #[cfg(test)]
    pub(super) fn new(
        card: &ModelDeploymentCard,
        enabled: bool,
        mode: RouterMode,
        teardown: Arc<super::super::allocator::AllocatorTrimOnDrop>,
    ) -> Self {
        Self::with_kinds(
            card,
            enabled,
            mode,
            DecisionKinds::advertised(card),
            teardown,
        )
    }

    pub(super) fn with_kinds(
        card: &ModelDeploymentCard,
        enabled: bool,
        mode: RouterMode,
        kinds: DecisionKinds,
        teardown: Arc<super::super::allocator::AllocatorTrimOnDrop>,
    ) -> Self {
        Self {
            advertised: kinds,
            resources: DecisionResources::planned(card, mode, enabled, teardown),
        }
    }

    pub(super) fn needs_load_context(&self) -> bool {
        self.resources
            .as_ref()
            .is_some_and(|resources| resources.capabilities().intent(self.advertised).any())
    }

    pub(super) fn prepare_resources(
        &mut self,
        tokenizer: Option<&Tokenizer>,
        routing: Option<&crate::entrypoint::input::PreprocessedRouting>,
        load: Option<Arc<crate::kv_router::RoutingLoadContext>>,
    ) -> Option<&DecisionResources> {
        let resources = self.resources.take()?;
        let score = routing
            .filter(|_| resources.capabilities().token_possible)
            .map(|routing| {
                ScoreFactory::new(
                    routing.score_engine(),
                    tokenizer
                        .cloned()
                        .map_or(TokenizerLoad::Card, TokenizerLoad::Loaded),
                )
            });
        self.resources = Some(resources.with_inputs(score, load));
        let resources = self.resources.as_ref()?;
        if resources.capabilities().intent(self.advertised).token {
            // Optional scoring failure must leave ordinary surfaces available.
            let _ = resources.prepare_token();
        }
        Some(resources)
    }

    pub(super) fn chat_preprocessor(
        &self,
        card: &ModelDeploymentCard,
        tokenizer: Tokenizer,
        cancellation: &CancellationToken,
    ) -> anyhow::Result<Arc<OpenAIPreprocessor>> {
        match self
            .resources
            .as_ref()
            .and_then(DecisionResources::prepared_token)
        {
            Some(token) => Ok(token.preprocessor.clone()),
            None => super::worker_set_chat_preprocessor(card, tokenizer, cancellation),
        }
    }

    pub(super) async fn routing_dispatch(
        &self,
        client: &Client,
        chooser: Option<&Arc<crate::kv_router::KvRouter>>,
    ) -> anyhow::Result<Option<crate::kv_router::LlmDispatch>> {
        let Some(capabilities) = self.resources.as_ref().map(DecisionResources::capabilities)
        else {
            return Ok(None);
        };
        if !capabilities.token_possible {
            return Ok(None);
        }
        let client = crate::entrypoint::input::routing_client(client, capabilities.mode, chooser)?;
        Ok(Some(
            super::super::DecisionDispatch::for_client(
                &client,
                super::super::GuardPolicy::Optional,
            )
            .await?,
        ))
    }

    pub(super) fn check_ordinary_surfaces(
        &self,
        card: &ModelDeploymentCard,
        workers: &WorkerSet,
    ) -> anyhow::Result<()> {
        let native_read = self
            .resources
            .as_ref()
            .is_some_and(|resources| resources.capabilities().intent(self.advertised).read);
        if workers.has_any_serving_engine() || native_read {
            if card.model_type.supports_chat() && workers.chat_engine.is_none() {
                tracing::warn!(
                    model_name = card.name(),
                    surface = "chat/completions",
                    "Advertised surface is unavailable: no supported Rust tokenizer or chat_engine_factory"
                );
            }
            if card.model_type.supports_completions() && workers.completions_engine.is_none() {
                tracing::warn!(
                    model_name = card.name(),
                    surface = "completions",
                    "Advertised surface is unavailable: no Rust tokenizer available for this model"
                );
            }
            return Ok(());
        }
        if card.model_type.supports_chat() {
            anyhow::bail!(
                "Model has no supported Rust tokenizer and no chat_engine_factory. \
                     Use --dyn-chat-processor vllm/sglang or provide a supported \
                     tokenizer file (tokenizer.json, tiktoken.model, or *.tiktoken)."
            );
        }
        anyhow::bail!(
            "Model '{}' requires frontend tokenization/preprocessing (ModelInput::Tokens) \
                 but no serving engine could be built. Provide a working tokenizer config or \
                 perform tokenization in the backend (ModelInput::Text).",
            card.name()
        );
    }

    pub(super) fn attach(self, worker_set: &mut WorkerSet, fleet: super::super::DecisionFleet) {
        worker_set.decision = self
            .resources
            .and_then(DecisionResources::into_shared)
            .map(|resources| super::super::DecisionSurface { resources, fleet });
    }
}
