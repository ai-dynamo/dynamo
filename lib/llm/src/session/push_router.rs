// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{
    engine::AsyncEngineContextProvider,
    pipeline::{
        AsyncEngine, Error, ManyOut, PushRouter, SingleIn, async_trait as pipeline_async_trait,
    },
};

use super::{
    LlmResponse, SessionTarget,
    coordinator::{SessionCoordinator, invalid_argument},
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::timing::{RequestPhase, WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL},
};

pub struct SessionPushRouter {
    inner: PushRouter<PreprocessedRequest, LlmResponse>,
    coordinator: SessionCoordinator,
}

impl SessionPushRouter {
    pub fn new(inner: PushRouter<PreprocessedRequest, LlmResponse>) -> Self {
        let component = inner.component().clone();
        Self {
            inner,
            coordinator: SessionCoordinator::new(component),
        }
    }

    fn phase(request: &PreprocessedRequest) -> RequestPhase {
        request
            .tracker
            .as_ref()
            .map(|tracker| tracker.phase())
            .unwrap_or(RequestPhase::Aggregated)
    }

    fn record_target(request: &PreprocessedRequest, target: SessionTarget) {
        let Some(tracker) = request.tracker.as_ref() else {
            return;
        };
        let worker_type = if tracker.phase() == RequestPhase::Prefill {
            WORKER_TYPE_PREFILL
        } else {
            WORKER_TYPE_DECODE
        };
        tracker.record_worker(target.worker_id, target.dp_rank, worker_type);
    }

    pub fn peek_next_worker(&self) -> Option<u64> {
        self.inner.peek_next_worker()
    }

    pub async fn select_and_dispatch_prefill<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<LlmResponse>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, u64, Option<u32>) -> Result<M, Error>,
    {
        let control = request
            .routing
            .as_ref()
            .and_then(|routing| routing.session_control.clone());
        if control.is_none() {
            let explicit = explicit_target(&request, RequestPhase::Prefill)?;
            let pinned = explicit.map(|target| target.worker_id);
            let rank = explicit.and_then(|target| target.dp_rank);
            return self
                .inner
                .select_and_dispatch_exact(request, pinned, move |request, worker_id| {
                    prepare(request, worker_id, rank)
                })
                .await;
        }

        let control = control.expect("checked above");
        let explicit = explicit_target(&request, RequestPhase::Prefill)?;
        let operation = self
            .coordinator
            .begin(&control, explicit, self.inner.is_direct_routing())
            .await?;
        let pinned = operation
            .target()
            .or(explicit)
            .map(|target| target.worker_id);
        let context_id = request.context().id().to_string();
        let rank = operation
            .target()
            .or(explicit)
            .and_then(|target| target.dp_rank);
        let ((operation, metadata), stream) = self
            .inner
            .select_and_dispatch_exact_async(
                request,
                pinned,
                move |worker_id| async move {
                    let mut operation = operation;
                    operation
                        .selected(
                            SessionTarget {
                                worker_id,
                                dp_rank: rank,
                            },
                            &context_id,
                        )
                        .await?;
                    Ok(operation)
                },
                move |request, worker_id, operation| {
                    let target = SessionTarget {
                        worker_id,
                        dp_rank: rank,
                    };
                    Self::record_target(request, target);
                    Ok((operation, prepare(request, worker_id, rank)?))
                },
            )
            .await?;
        Ok((metadata, operation.into_stream(stream)?))
    }
}

#[pipeline_async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<LlmResponse>, Error> for SessionPushRouter {
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        let control = request
            .routing
            .as_ref()
            .and_then(|routing| routing.session_control.clone());
        if control.is_none() {
            if self.inner.is_direct_routing() {
                let target = explicit_target(&request, RequestPhase::Decode)?
                    .ok_or_else(|| anyhow::anyhow!("worker ID required in Direct routing mode"))?;
                let rank = target.dp_rank;
                let (_, stream) = self
                    .inner
                    .select_and_dispatch_exact(
                        request,
                        Some(target.worker_id),
                        move |request, worker_id| {
                            Self::record_target(
                                request,
                                SessionTarget {
                                    worker_id,
                                    dp_rank: rank,
                                },
                            );
                            Ok(())
                        },
                    )
                    .await?;
                return Ok(stream);
            }
            return self.inner.generate(request).await;
        }

        let control = control.expect("checked above");
        let phase = Self::phase(&request);
        let explicit = explicit_target(&request, phase)?;
        let operation = self
            .coordinator
            .begin(&control, explicit, self.inner.is_direct_routing())
            .await?;
        let selected_target = operation.target().or(explicit);
        let pinned = selected_target.map(|target| target.worker_id);
        let rank = selected_target.and_then(|target| target.dp_rank);
        let context_id = request.context().id().to_string();
        let (operation, stream) = self
            .inner
            .select_and_dispatch_exact_async(
                request,
                pinned,
                move |worker_id| async move {
                    let mut operation = operation;
                    operation
                        .selected(
                            SessionTarget {
                                worker_id,
                                dp_rank: rank,
                            },
                            &context_id,
                        )
                        .await?;
                    Ok(operation)
                },
                move |request, worker_id, operation| {
                    Self::record_target(
                        request,
                        SessionTarget {
                            worker_id,
                            dp_rank: rank,
                        },
                    );
                    Ok(operation)
                },
            )
            .await?;
        operation.into_stream(stream)
    }
}

pub fn explicit_target(
    request: &PreprocessedRequest,
    phase: RequestPhase,
) -> Result<Option<SessionTarget>, Error> {
    let Some(routing) = request.routing.as_ref() else {
        return Ok(None);
    };
    let (worker_id, dp_rank) = match phase {
        RequestPhase::Prefill => (
            routing.prefill_worker_id.or(routing.backend_instance_id),
            routing.prefill_dp_rank.or(routing.dp_rank),
        ),
        RequestPhase::Decode => (
            routing.decode_worker_id.or(routing.backend_instance_id),
            routing.dp_rank,
        ),
        RequestPhase::Aggregated => (routing.backend_instance_id, routing.dp_rank),
    };
    if worker_id.is_none() && dp_rank.is_some() {
        return Err(invalid_argument("DP rank requires an explicit worker"));
    }
    Ok(worker_id.map(|worker_id| SessionTarget { worker_id, dp_rank }))
}
