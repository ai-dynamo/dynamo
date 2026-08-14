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
    /// Return whether `query_instance_id` requests are advisory for this backend.
    fn supports_advisory(&self) -> bool {
        false
    }

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

    async fn begin_dispatch(
        &self,
        request: &mut SingleIn<PreprocessedRequest>,
        attempt: &mut Self::Attempt,
        kind: AttemptKind,
    ) -> Result<AffinityTarget, Error>;

    fn after_prepare(
        &self,
        request: &mut SingleIn<PreprocessedRequest>,
        attempt: &mut Self::Attempt,
    );

    async fn dispatch_prepared(
        &self,
        request: SingleIn<PreprocessedRequest>,
        attempt: &mut Self::Attempt,
        kind: AttemptKind,
    ) -> Result<ManyOut<LlmResponse>, Error>;

    async fn abort(&self, attempt: &mut Self::Attempt);

    fn finish_dispatch(
        &self,
        attempt: Self::Attempt,
        target: AffinityTarget,
        stream: ManyOut<LlmResponse>,
    ) -> ManyOut<LlmResponse>;
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

pub(crate) async fn dispatch_attempt<B, M, F>(
    backend: &B,
    mut request: SingleIn<PreprocessedRequest>,
    mut attempt: B::Attempt,
    kind: AttemptKind,
    prepare: F,
) -> Result<(M, AffinityTarget, ManyOut<LlmResponse>), Error>
where
    B: AttemptBackend,
    M: Send,
    F: FnOnce(&mut PreprocessedRequest, AffinityTarget) -> Result<M, Error> + Send,
{
    let target = match backend
        .begin_dispatch(&mut request, &mut attempt, kind)
        .await
    {
        Ok(target) => target,
        Err(error) => {
            backend.abort(&mut attempt).await;
            return Err(error);
        }
    };
    let metadata = match prepare(&mut request, target) {
        Ok(metadata) => metadata,
        Err(error) => {
            backend.abort(&mut attempt).await;
            return Err(error);
        }
    };
    backend.after_prepare(&mut request, &mut attempt);
    let stream = match backend.dispatch_prepared(request, &mut attempt, kind).await {
        Ok(stream) => stream,
        Err(error) => {
            backend.abort(&mut attempt).await;
            return Err(error);
        }
    };
    Ok((
        metadata,
        target,
        backend.finish_dispatch(attempt, target, stream),
    ))
}

pub(crate) async fn generate<B: AttemptBackend>(
    backend: &B,
    request: SingleIn<PreprocessedRequest>,
) -> Result<ManyOut<LlmResponse>, Error> {
    let phase = phase(&request);
    let intent = if backend.supports_advisory()
        && request.get_annotation_value("query_instance_id").is_some()
    {
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
    let dispatch = dispatch_attempt(backend, request, attempt, AttemptKind::Generate, |_, _| {
        Ok(())
    })
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
    let dispatch = dispatch_attempt(backend, request, attempt, AttemptKind::Prefill, prepare).await;
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use dynamo_runtime::pipeline::Context;
    use futures::StreamExt;

    use super::*;

    struct TestBackend {
        advisory: bool,
        intents: Mutex<Vec<SelectionIntent>>,
    }

    impl TestBackend {
        fn new(advisory: bool) -> Self {
            Self {
                advisory,
                intents: Mutex::new(Vec::new()),
            }
        }
    }

    impl AttemptBackend for TestBackend {
        type Attempt = ();

        fn supports_advisory(&self) -> bool {
            self.advisory
        }

        fn affinity(&self) -> Option<&AffinityCoordinator> {
            None
        }

        fn direct(&self) -> bool {
            false
        }

        async fn select(
            &self,
            _request: &SingleIn<PreprocessedRequest>,
            _phase: RequestPhase,
            intent: SelectionIntent,
            _pinned_target: Option<AffinityTarget>,
        ) -> Result<Self::Attempt, Error> {
            self.intents.lock().unwrap().push(intent);
            Ok(())
        }

        fn observe_advisory(
            &self,
            _request: &SingleIn<PreprocessedRequest>,
            _attempt: &Self::Attempt,
        ) {
        }

        async fn begin_dispatch(
            &self,
            _request: &mut SingleIn<PreprocessedRequest>,
            _attempt: &mut Self::Attempt,
            _kind: AttemptKind,
        ) -> Result<AffinityTarget, Error> {
            Ok(AffinityTarget::new(1, None))
        }

        fn after_prepare(
            &self,
            _request: &mut SingleIn<PreprocessedRequest>,
            _attempt: &mut Self::Attempt,
        ) {
        }

        async fn dispatch_prepared(
            &self,
            request: SingleIn<PreprocessedRequest>,
            _attempt: &mut Self::Attempt,
            _kind: AttemptKind,
        ) -> Result<ManyOut<LlmResponse>, Error> {
            let context = request.context().clone();
            let output = Annotated::from_data(LLMEngineOutput {
                token_ids: vec![9],
                ..Default::default()
            });
            Ok(ResponseStream::new(
                Box::pin(stream::iter([output])),
                context,
            ))
        }

        async fn abort(&self, _attempt: &mut Self::Attempt) {}

        fn finish_dispatch(
            &self,
            _attempt: Self::Attempt,
            _target: AffinityTarget,
            stream: ManyOut<LlmResponse>,
        ) -> ManyOut<LlmResponse> {
            stream
        }
    }

    fn query_request() -> SingleIn<PreprocessedRequest> {
        Context::new(
            PreprocessedRequest::builder()
                .model("test".to_string())
                .token_ids(vec![1, 2, 3])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .annotations(vec!["query_instance_id:true".to_string()])
                .build()
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn query_annotation_dispatches_when_backend_does_not_support_advisory() {
        let backend = TestBackend::new(false);
        let mut response = generate(&backend, query_request()).await.unwrap();

        assert_eq!(
            response.next().await.unwrap().data.unwrap().token_ids,
            vec![9]
        );
        assert_eq!(
            backend.intents.into_inner().unwrap(),
            vec![SelectionIntent::Committed]
        );
    }

    #[tokio::test]
    async fn query_annotation_remains_advisory_for_kv_style_backend() {
        let backend = TestBackend::new(true);
        let mut response = generate(&backend, query_request()).await.unwrap();

        let output = response.next().await.unwrap().data.unwrap();
        assert!(output.routing_data.is_some());
        assert_eq!(
            backend.intents.into_inner().unwrap(),
            vec![SelectionIntent::Advisory]
        );
    }
}
