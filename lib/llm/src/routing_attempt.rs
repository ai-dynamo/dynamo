// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One host-owned orchestration path for an LLM routing attempt.
//!
//! Worker choice remains backend-owned. This module owns the shared intent,
//! affinity, dispatch, and error-observation sequence around that choice.

use dynamo_runtime::{
    engine::AsyncEngineContextProvider,
    error::{ErrorType, match_error_chain},
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{Error, ManyOut, ResponseStream, SingleIn},
    protocols::annotated::Annotated,
};
use futures::stream;

use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RoutingData},
    },
    session_affinity::{
        AffinityAcquire, AffinityCoordinator, AffinityTarget, LlmResponse, affinity_id,
        explicit_target, invalid_argument,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SelectionIntent {
    Advisory,
    Committed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttemptKind {
    Generate,
    Prefill,
}

pub(crate) trait AttemptBackend: Send + Sync {
    type Attempt: Send;

    fn affinity(&self) -> Option<&AffinityCoordinator>;

    fn direct(&self) -> bool;

    async fn select(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        intent: SelectionIntent,
        pinned_target: Option<AffinityTarget>,
    ) -> Result<Self::Attempt, Error>;

    fn observe_advisory(&self, request: &SingleIn<PreprocessedRequest>, attempt: &Self::Attempt);

    async fn dispatch<M, F>(
        &self,
        request: SingleIn<PreprocessedRequest>,
        attempt: Self::Attempt,
        kind: AttemptKind,
        prepare: F,
    ) -> Result<(M, AffinityTarget, ManyOut<LlmResponse>), Error>
    where
        M: Send,
        F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error> + Send;
}

fn phase(request: &PreprocessedRequest) -> RequestPhase {
    request
        .tracker
        .as_ref()
        .map(|tracker| tracker.phase())
        .unwrap_or(RequestPhase::Aggregated)
}

fn is_cancelled(error: &Error) -> bool {
    match_error_chain(error.as_ref(), &[ErrorType::Cancelled], &[])
}

fn invalidate_on_non_cancellation(operation: &mut Option<AffinityAcquire>, error: &Error) {
    if is_cancelled(error) {
        return;
    }
    if let Some(operation) = operation.take() {
        operation.invalidate();
    }
}

fn direct_target<B: AttemptBackend>(
    backend: &B,
    explicit: Option<AffinityTarget>,
    phase: RequestPhase,
) -> Result<Option<AffinityTarget>, Error> {
    if !backend.direct() {
        return Ok(explicit);
    }
    explicit.map(Some).ok_or_else(|| {
        invalid_argument(format!(
            "worker ID required for {phase} request in Direct routing mode"
        ))
    })
}

async fn select_with_affinity<B: AttemptBackend>(
    backend: &B,
    request: &SingleIn<PreprocessedRequest>,
    phase: RequestPhase,
    intent: SelectionIntent,
) -> Result<(B::Attempt, Option<AffinityAcquire>), Error> {
    let explicit = direct_target(backend, explicit_target(request, phase)?, phase)?;
    let Some(affinity) = backend.affinity() else {
        return Ok((
            backend.select(request, phase, intent, explicit).await?,
            None,
        ));
    };
    let Some(session_id) = affinity_id(request)? else {
        return Ok((
            backend.select(request, phase, intent, explicit).await?,
            None,
        ));
    };

    if intent == SelectionIntent::Advisory {
        let target = affinity.query_target(&session_id, explicit)?.or(explicit);
        return Ok((backend.select(request, phase, intent, target).await?, None));
    }

    let request_context = request.context();
    let operation = affinity
        .acquire_with_context(&session_id, explicit, request_context.as_ref())
        .await?;
    let target = operation.target().or(explicit);
    match backend.select(request, phase, intent, target).await {
        Ok(attempt) => Ok((attempt, Some(operation))),
        Err(error) if is_cancelled(&error) => Err(error),
        Err(_) if operation.target().is_some() && explicit.is_none() => {
            operation.invalidate();
            let retry = affinity
                .acquire_with_context(&session_id, None, request_context.as_ref())
                .await?;
            let retry_target = retry.target();
            match backend.select(request, phase, intent, retry_target).await {
                Ok(attempt) => Ok((attempt, Some(retry))),
                Err(error) => {
                    retry.invalidate();
                    Err(error)
                }
            }
        }
        Err(error) => {
            operation.invalidate();
            Err(error)
        }
    }
}

fn advisory_response<B: AttemptBackend>(
    backend: &B,
    request: SingleIn<PreprocessedRequest>,
    attempt: &B::Attempt,
) -> ManyOut<LlmResponse> {
    backend.observe_advisory(&request, attempt);
    let stream_context = request.context().clone();
    let worker_id = request
        .tracker
        .as_ref()
        .and_then(|tracker| tracker.get_worker_info());
    let output = LLMEngineOutput {
        routing_data: Some(RoutingData {
            worker_id,
            token_ids: Some(request.token_ids.clone()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let response = Annotated::from_data(output);
    ResponseStream::new(Box::pin(stream::iter([response])), stream_context)
}

pub(crate) async fn generate<B: AttemptBackend>(
    backend: &B,
    request: SingleIn<PreprocessedRequest>,
) -> Result<ManyOut<LlmResponse>, Error> {
    let phase = phase(&request);
    let intent = if request.get_annotation_value("query_instance_id").is_some() {
        SelectionIntent::Advisory
    } else {
        SelectionIntent::Committed
    };
    let phase_label = phase.to_string();
    let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
    let (attempt, mut operation) = select_with_affinity(backend, &request, phase, intent).await?;
    if intent == SelectionIntent::Advisory {
        return Ok(advisory_response(backend, request, &attempt));
    }

    drop(route_guard);
    let dispatch = backend
        .dispatch(request, attempt, AttemptKind::Generate, |_, _| Ok(()))
        .await;
    let ((), target, stream) = match dispatch {
        Ok(result) => result,
        Err(error) => {
            invalidate_on_non_cancellation(&mut operation, &error);
            return Err(error);
        }
    };
    match operation {
        Some(operation) => operation.into_stream(target, stream),
        None => Ok(stream),
    }
}

pub(crate) async fn select_and_dispatch_prefill<B, M, F>(
    backend: &B,
    request: SingleIn<PreprocessedRequest>,
    prepare: F,
) -> Result<(M, ManyOut<LlmResponse>), Error>
where
    B: AttemptBackend,
    M: Send,
    F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error> + Send,
{
    let phase = RequestPhase::Prefill;
    let phase_label = phase.to_string();
    let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
    let (attempt, mut operation) =
        select_with_affinity(backend, &request, phase, SelectionIntent::Committed).await?;
    drop(route_guard);
    let dispatch = backend
        .dispatch(request, attempt, AttemptKind::Prefill, prepare)
        .await;
    let (metadata, target, stream) = match dispatch {
        Ok(result) => result,
        Err(error) => {
            invalidate_on_non_cancellation(&mut operation, &error);
            return Err(error);
        }
    };
    let stream = match operation {
        Some(operation) => operation.into_stream(target, stream)?,
        None => stream,
    };
    Ok((metadata, stream))
}
