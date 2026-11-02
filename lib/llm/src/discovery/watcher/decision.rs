// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context as _;
use dynamo_runtime::pipeline::{RouterMode, network::egress::push_router::PushRouter};
use tokio_util::sync::CancellationToken;

use crate::{
    discovery::{DecisionEngine, DecisionRoute, TokenExecutor, WorkerSet},
    local_model::runtime_config::{DecisionProfileKind, NativeExecutorKind, TokenBackend},
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::{
        Annotated,
        decision_head::{DecisionHeadRequest, DecisionHeadResponse},
        decision_read::{DecisionReadRequest, DecisionReadResponse},
    },
    tokenizers::Tokenizer,
    types::openai::generate::GenerateStreamingEngine,
};

pub(super) enum DecisionPlan {
    Disabled,
    Token(TokenBackend),
    Native {
        kind: NativeExecutorKind,
        mode: RouterMode,
    },
    UnsupportedRouting,
}

impl DecisionPlan {
    pub(super) fn new(card: &ModelDeploymentCard, enabled: bool, mode: RouterMode) -> Self {
        let Some(profile) = enabled.then(|| card.decision_execution_profile()).flatten() else {
            return Self::Disabled;
        };
        match profile.kind() {
            DecisionProfileKind::Token(backend) => Self::Token(backend),
            DecisionProfileKind::Native(kind) if Self::supports_native_routing(mode) => {
                Self::Native { kind, mode }
            }
            DecisionProfileKind::Native(_) => {
                tracing::warn!(
                    model_name = card.name(),
                    router_mode = ?mode,
                    "Skipping optional decision engine: whole-request decisions have neither KV tokens nor explicit worker IDs; keeping other model surfaces active"
                );
                Self::UnsupportedRouting
            }
        }
    }

    pub(super) fn supports_native_routing(mode: RouterMode) -> bool {
        matches!(
            mode,
            RouterMode::RoundRobin
                | RouterMode::Random
                | RouterMode::PowerOfTwoChoices
                | RouterMode::LeastLoaded
                | RouterMode::DeviceAwareWeighted
        )
    }

    pub(super) fn needs_routing(&self) -> bool {
        matches!(self, Self::Token(_) | Self::Native { .. })
    }

    pub(super) fn needs_internal_generate(&self) -> bool {
        matches!(self, Self::Token(_))
    }

    pub(super) fn has_native_route(&self) -> bool {
        matches!(self, Self::Native { .. })
    }

    pub(super) fn preprocessor(
        &self,
        card: &ModelDeploymentCard,
        tokenizer: Option<&Tokenizer>,
        cancellation: &CancellationToken,
    ) -> Option<Arc<OpenAIPreprocessor>> {
        if !self.needs_internal_generate() {
            return None;
        }
        let Some(tokenizer) = tokenizer else {
            tracing::warn!(
                model = card.name(),
                "System One preprocessor is unavailable: model has no supported Rust tokenizer"
            );
            return None;
        };
        match super::worker_set_chat_preprocessor(card, tokenizer.clone(), cancellation) {
            Ok(preprocessor) => Some(preprocessor),
            Err(error) => {
                tracing::warn!(
                    model = card.name(),
                    error = %format!("{error:#}"),
                    "System One preprocessor is unavailable"
                );
                None
            }
        }
    }

    pub(super) async fn attach(
        self,
        worker_set: &mut WorkerSet,
        token_execution: Option<(GenerateStreamingEngine, Arc<OpenAIPreprocessor>)>,
    ) -> anyhow::Result<()> {
        let route = match self {
            Self::Disabled => None,
            Self::UnsupportedRouting => Some(DecisionRoute::UnsupportedRouting),
            Self::Token(backend) => token_execution.map(|(engine, preprocessor)| {
                DecisionRoute::Available(DecisionEngine::Token(TokenExecutor::new(
                    backend,
                    engine,
                    preprocessor,
                )))
            }),
            Self::Native { kind, mode } => {
                let client = worker_set
                    .load_context()
                    .context("native decision load context")?
                    .client()
                    .clone();
                let engine = match kind {
                    NativeExecutorKind::Head => {
                        let router = PushRouter::<
                            DecisionHeadRequest,
                            Annotated<DecisionHeadResponse>,
                        >::from_client_with_monitor(
                            client, mode, None
                        )
                        .await?;
                        DecisionEngine::Head(Arc::new(router))
                    }
                    NativeExecutorKind::Read => {
                        let router = PushRouter::<
                            DecisionReadRequest,
                            Annotated<DecisionReadResponse>,
                        >::from_client_with_monitor(
                            client, mode, None
                        )
                        .await?;
                        DecisionEngine::Read(Arc::new(router))
                    }
                };
                Some(DecisionRoute::Available(engine))
            }
        };
        worker_set.decision = route;
        Ok(())
    }
}
