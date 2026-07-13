// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::OwnedSemaphorePermit;
use tracing::Instrument;

use dynamo_runtime::{
    pipeline::{ManyOut, SingleIn},
    protocols::annotated::Annotated,
};

use super::{PrefillCompletion, PrefillError, PrefillRouter};
use crate::{
    kv_router::KvPushRouter,
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::RequestTracker,
    },
    session_affinity::{AffinityTarget, SessionAffinityPushRouter},
};

#[derive(Clone)]
pub(super) enum InnerPrefillRouter {
    KvRouter(Arc<KvPushRouter>),
    SimpleRouter(Arc<SessionAffinityPushRouter>),
}

impl InnerPrefillRouter {
    pub(super) async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>)>
    where
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M>,
    {
        match self {
            InnerPrefillRouter::KvRouter(router) => {
                router.select_and_dispatch_prefill(request, prepare).await
            }
            InnerPrefillRouter::SimpleRouter(router) => {
                router.select_and_dispatch_prefill(request, prepare).await
            }
        }
    }
}

impl PrefillRouter {
    pub(super) fn spawn_prefill_task(
        &self,
        prefill_stream: ManyOut<Annotated<LLMEngineOutput>>,
        tracker: Option<Arc<RequestTracker>>,
        phase_transition_permit: OwnedSemaphorePermit,
        expected_choices: Option<u32>,
    ) -> tokio::task::JoinHandle<Result<PrefillCompletion, PrefillError>> {
        let span = tracing::Span::current();
        tokio::spawn(
            async move {
                drop(phase_transition_permit);
                let result =
                    Self::consume_prefill_stream(prefill_stream, tracker, expected_choices).await;
                match &result {
                    Ok(_) => tracing::debug!("Prefill background task completed"),
                    Err(error) => tracing::warn!("Prefill background task error: {error:?}"),
                };
                result
            }
            .instrument(span),
        )
    }
}
