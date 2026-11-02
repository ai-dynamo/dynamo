// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{SystemOneSelectionError, WorkerSet};
use crate::{
    local_model::runtime_config::{
        DecisionExecutionProfile, DecisionProfileKind, NativeExecutorKind, TokenBackend,
    },
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::{
        decision_head::DecisionHeadStreamingEngine, decision_read::DecisionReadStreamingEngine,
    },
    types::openai::generate::GenerateStreamingEngine,
};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct TokenExecutor {
    pub(super) engine: GenerateStreamingEngine,
    pub(super) preprocessor: Arc<OpenAIPreprocessor>,
    pub(super) backend: TokenBackend,
}

impl TokenExecutor {
    pub(crate) fn new(
        backend: TokenBackend,
        engine: GenerateStreamingEngine,
        preprocessor: Arc<OpenAIPreprocessor>,
    ) -> Self {
        Self {
            engine,
            preprocessor,
            backend,
        }
    }
    pub(crate) fn engine(&self) -> &GenerateStreamingEngine {
        &self.engine
    }
    pub(crate) fn preprocessor(&self) -> &Arc<OpenAIPreprocessor> {
        &self.preprocessor
    }
    pub(crate) fn backend(&self) -> TokenBackend {
        self.backend
    }
}

#[derive(Clone)]
pub(crate) enum DecisionEngine {
    Token(TokenExecutor),
    Head(DecisionHeadStreamingEngine),
    Read(DecisionReadStreamingEngine),
}

#[derive(Clone)]
pub(crate) enum DecisionRoute {
    Available(DecisionEngine),
    UnsupportedRouting,
}

impl DecisionRoute {
    pub(crate) fn available(&self) -> Option<&DecisionEngine> {
        match self {
            Self::Available(engine) => Some(engine),
            Self::UnsupportedRouting => None,
        }
    }

    pub(crate) fn available_for(
        &self,
        profile: DecisionExecutionProfile,
    ) -> Option<&DecisionEngine> {
        let engine = self.available()?;
        let expected = match engine {
            DecisionEngine::Token(executor) => DecisionProfileKind::Token(executor.backend),
            DecisionEngine::Head(_) => DecisionProfileKind::Native(NativeExecutorKind::Head),
            DecisionEngine::Read(_) => DecisionProfileKind::Native(NativeExecutorKind::Read),
        };
        (profile.kind() == expected).then_some(engine)
    }
}

#[derive(Clone)]
pub(crate) struct DecisionContext {
    pub(crate) canonical_model: String,
    pub(crate) card: ModelDeploymentCard,
    pub(crate) admitted_ids: Option<tokio::sync::watch::Receiver<Vec<u64>>>,
    pub(crate) lifecycle_cancellation: Option<tokio_util::sync::CancellationToken>,
    profile: DecisionExecutionProfile,
    max_candidates: usize,
}

impl DecisionContext {
    pub(crate) fn profile(&self) -> DecisionExecutionProfile {
        self.profile
    }
    pub(crate) fn validate_dispatch(
        &self,
        pinned_worker: Option<u64>,
    ) -> Result<(), SystemOneSelectionError> {
        if self
            .lifecycle_cancellation
            .as_ref()
            .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
        {
            return Err(SystemOneSelectionError::Unavailable);
        }
        let Some(admitted) = &self.admitted_ids else {
            return Ok(());
        };
        if admitted.has_changed().is_err() {
            return Err(SystemOneSelectionError::Unavailable);
        }
        let workers = admitted.borrow();
        if workers.is_empty() || pinned_worker.is_some_and(|worker| !workers.contains(&worker)) {
            return Err(SystemOneSelectionError::Unavailable);
        }
        Ok(())
    }
    pub(crate) fn decision_capabilities(&self) -> dynamo_decisions::Capabilities {
        dynamo_decisions::Capabilities {
            max_questions: self.profile.limits().0,
            max_candidates: self.max_candidates,
            supports_predicate: true,
            supports_choice: true,
            supports_score: true,
            measured_cache_reads: true,
            prompt_format_version: 1,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SystemOneExecutionSelection {
    pub(crate) context: DecisionContext,
    engine: DecisionEngine,
}

impl SystemOneExecutionSelection {
    pub(crate) fn from_worker_set(canonical_model: String, worker_set: &WorkerSet) -> Option<Self> {
        let profile = worker_set.card().decision_execution_profile()?;
        Self::build(canonical_model, profile, worker_set)
    }
    #[cfg(test)]
    pub(crate) fn for_test(
        canonical_model: String,
        profile: DecisionExecutionProfile,
        worker_set: &WorkerSet,
    ) -> Option<Self> {
        Self::build(canonical_model, profile, worker_set)
    }
    fn build(
        canonical_model: String,
        profile: DecisionExecutionProfile,
        worker_set: &WorkerSet,
    ) -> Option<Self> {
        let engine = worker_set
            .decision
            .as_ref()?
            .available_for(profile)?
            .clone();
        let max_candidates = worker_set.card().decision_max_candidates(profile)?;
        Some(Self {
            context: DecisionContext {
                canonical_model,
                profile,
                max_candidates,
                card: worker_set.card().clone(),
                admitted_ids: worker_set.decision_admissions().cloned(),
                lifecycle_cancellation: worker_set.lifecycle_cancellation(),
            },
            engine,
        })
    }
    #[cfg(test)]
    pub(crate) fn engine(&self) -> &DecisionEngine {
        &self.engine
    }
    pub(crate) fn into_parts(self) -> (DecisionContext, DecisionEngine) {
        (self.context, self.engine)
    }
}
