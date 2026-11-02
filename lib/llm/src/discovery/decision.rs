// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{DecisionResources, SystemOneSelectionError, TokenResources, WorkerSet};
#[cfg(test)]
use crate::decision_profile::DecisionProfileKind;
use crate::{
    decision_profile::{DecisionExecutionProfile, TokenBackend},
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::{
        decision_head::DecisionHeadStreamingEngine, decision_read::DecisionReadStreamingEngine,
    },
    types::openai::generate::GenerateStreamingEngine,
};
use std::sync::Arc;

#[cfg(test)]
#[path = "decision_dispatch_tests.rs"]
mod dispatch_tests;
#[cfg(test)]
#[path = "decision/test_support.rs"]
pub(crate) mod test_support;

pub(crate) type DecisionQualification = (DecisionExecutionProfile, usize);

#[derive(Clone, Debug)]
pub(crate) struct DecisionAdmission {
    pub(crate) qualification: DecisionQualification,
    pub(crate) ids: Vec<u64>,
    epoch: Arc<()>,
}

#[derive(Clone, Debug)]
pub(crate) enum DecisionAvailability {
    Unadvertised,
    Pending,
    Failed,
    Qualified(Arc<DecisionAdmission>),
}

impl DecisionAvailability {
    pub(crate) fn qualified(qualification: DecisionQualification, mut ids: Vec<u64>) -> Self {
        ids.sort_unstable();
        ids.dedup();
        Self::Qualified(Arc::new(DecisionAdmission {
            qualification,
            ids,
            epoch: Arc::new(()),
        }))
    }

    pub(crate) fn admission(&self) -> Option<&Arc<DecisionAdmission>> {
        match self {
            Self::Qualified(admission) => Some(admission),
            Self::Unadvertised | Self::Pending | Self::Failed => None,
        }
    }

    pub(crate) fn publish(sender: &tokio::sync::watch::Sender<Self>, mut next: Self) {
        sender.send_if_modified(|current| {
            if match (&*current, &next) {
                (Self::Qualified(previous), Self::Qualified(next)) => {
                    previous.qualification == next.qualification && previous.ids == next.ids
                }
                (Self::Unadvertised, Self::Unadvertised)
                | (Self::Pending, Self::Pending)
                | (Self::Failed, Self::Failed) => true,
                _ => false,
            } {
                return false;
            }
            if let (Self::Qualified(previous), Self::Qualified(next)) = (&*current, &mut next)
                && previous.qualification == next.qualification
                && !previous.ids.is_empty()
                && !next.ids.is_empty()
            {
                Arc::make_mut(next).epoch = previous.epoch.clone();
            }
            *current = next;
            true
        });
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DecisionFleet {
    pub(crate) state: tokio::sync::watch::Receiver<DecisionAvailability>,
}

impl DecisionFleet {
    pub(crate) fn current(&self) -> Option<tokio::sync::watch::Ref<'_, DecisionAvailability>> {
        self.state.has_changed().ok()?;
        Some(self.state.borrow())
    }
}

pub(crate) struct DecisionSurface {
    pub(crate) resources: Arc<DecisionResources>,
    pub(crate) fleet: DecisionFleet,
}

#[derive(Clone)]
pub(crate) struct TokenExecutor {
    resources: Arc<TokenResources>,
    backend: TokenBackend,
}

impl TokenExecutor {
    pub(crate) fn new(backend: TokenBackend, resources: Arc<TokenResources>) -> Self {
        Self { resources, backend }
    }
    pub(crate) fn engine(&self) -> &GenerateStreamingEngine {
        &self.resources.engine
    }
    pub(crate) fn preprocessor(&self) -> &Arc<OpenAIPreprocessor> {
        &self.resources.preprocessor
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
pub(crate) struct DecisionContext {
    pub(crate) canonical_model: String,
    pub(crate) card: ModelDeploymentCard,
    fence: Arc<DispatchFence>,
    profile: DecisionExecutionProfile,
    max_candidates: usize,
}

pub(super) struct DispatchFence {
    fleet: DecisionFleet,
    admission: Arc<DecisionAdmission>,
    admitted_ids: tokio::sync::watch::Receiver<Vec<u64>>,
    cancellation: tokio_util::sync::CancellationToken,
}

impl DispatchFence {
    pub(super) fn validate(&self, worker: Option<u64>) -> Result<(), SystemOneSelectionError> {
        {
            let state = self
                .fleet
                .current()
                .ok_or(SystemOneSelectionError::Unavailable)?;
            let current = state
                .admission()
                .ok_or(SystemOneSelectionError::Unavailable)?;
            if self.admission.qualification != current.qualification
                || !Arc::ptr_eq(&self.admission.epoch, &current.epoch)
                || current.ids.is_empty()
                || worker.is_some_and(|worker| !current.ids.contains(&worker))
            {
                return Err(SystemOneSelectionError::Unavailable);
            }
        }
        if self.cancellation.is_cancelled() {
            return Err(SystemOneSelectionError::Unavailable);
        }
        let admitted = &self.admitted_ids;
        if admitted.has_changed().is_err() {
            return Err(SystemOneSelectionError::Unavailable);
        }
        let workers = admitted.borrow();
        if workers.is_empty() || worker.is_some_and(|worker| !workers.contains(&worker)) {
            return Err(SystemOneSelectionError::Unavailable);
        }
        Ok(())
    }
}

impl DecisionContext {
    pub(crate) fn profile(&self) -> DecisionExecutionProfile {
        self.profile
    }
    pub(crate) fn validate_dispatch(
        &self,
        pinned_worker: Option<u64>,
    ) -> Result<(), SystemOneSelectionError> {
        self.fence.validate(pinned_worker)
    }

    pub(crate) fn attach_dispatch_guard<T: dynamo_runtime::engine::Data>(
        &self,
        request: &mut dynamo_runtime::pipeline::Context<T>,
    ) -> super::decision_dispatch::DispatchReceipt {
        let receipt = super::decision_dispatch::DispatchReceipt::default();
        super::decision_dispatch::attach_dispatch_guard(
            request,
            self.fence.clone(),
            receipt.clone(),
        );
        receipt
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
pub(crate) struct SystemOneExecutionPlan {
    pub(crate) context: DecisionContext,
    resources: Arc<DecisionResources>,
}

impl SystemOneExecutionPlan {
    pub(crate) fn from_worker_set(canonical_model: String, worker_set: &WorkerSet) -> Option<Self> {
        let surface = worker_set.decision.as_ref()?;
        let state = surface.fleet.current()?;
        let super::DecisionState::Qualified {
            profile,
            resources,
            capacity: max_candidates,
            ..
        } = worker_set.decision_state_for(Some(&state))
        else {
            return None;
        };
        let admission = state.admission()?.clone();
        Some(Self {
            context: DecisionContext {
                canonical_model,
                profile,
                max_candidates,
                card: worker_set.card().clone(),
                fence: Arc::new(DispatchFence {
                    admitted_ids: worker_set.decision_admissions()?.clone(),
                    cancellation: worker_set.lifecycle_cancellation()?.clone(),
                    fleet: surface.fleet.clone(),
                    admission,
                }),
            },
            resources: resources.clone(),
        })
    }

    pub(crate) async fn prepare_native(&self) -> anyhow::Result<()> {
        self.context.validate_dispatch(None)?;
        self.resources
            .prepare_native(self.context.profile.kind())
            .await
    }

    // Token construction runs in admitted blocking preflight, never on the async executor.
    pub(crate) fn prepare(self) -> anyhow::Result<SystemOneExecutionSelection> {
        self.context.validate_dispatch(None)?;
        let engine = self.resources.prepare(self.context.profile.kind())?;
        self.context.validate_dispatch(None)?;
        Ok(SystemOneExecutionSelection {
            engine,
            context: self.context,
        })
    }
}

#[derive(Clone)]
pub(crate) struct SystemOneExecutionSelection {
    pub(crate) context: DecisionContext,
    pub(crate) engine: DecisionEngine,
}

impl SystemOneExecutionSelection {
    pub(crate) fn into_parts(self) -> (DecisionContext, DecisionEngine) {
        (self.context, self.engine)
    }
}
