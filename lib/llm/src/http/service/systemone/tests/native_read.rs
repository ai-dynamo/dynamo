// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    discovery::DecisionExecutor,
    local_model::runtime_config::DecisionExecutionProfile,
    protocols::{
        decision_head::HeadUsage,
        decision_read::{DecisionReadRequest, DecisionReadResponse, ReadOutcome},
    },
};

struct ReadEngine {
    malformed: bool,
    unknown_cache: bool,
    calls: Mutex<Vec<DecisionReadRequest>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<DecisionReadRequest>, ManyOut<Annotated<DecisionReadResponse>>, Error>
    for ReadEngine
{
    async fn generate(
        &self,
        request: SingleIn<DecisionReadRequest>,
    ) -> Result<ManyOut<Annotated<DecisionReadResponse>>, Error> {
        let context = request.context();
        let (wire, _) = request.into_parts();
        self.calls.lock().unwrap().push(wire.clone());
        let wire = wire.decision_read;
        let response = DecisionReadResponse {
            protocol_version: 1,
            profile: wire.profile,
            outcomes: wire
                .questions
                .iter()
                .map(|q| ReadOutcome {
                    ordinal: q.ordinal,
                    probabilities: vec![0.25, 0.75],
                    label_mass: if self.malformed { 2.0 } else { 0.5 },
                })
                .collect(),
            usage: HeadUsage {
                input_tokens: 17,
                output_tokens: wire.questions.len() as u64 * 2,
                cached_tokens: if self.unknown_cache { None } else { Some(0) },
            },
        };
        Ok(ResponseStream::new(
            Box::pin(stream::once(async { Annotated::from_data(response) })),
            context,
        ))
    }
}

#[tokio::test]
async fn native_read_uses_public_projection_with_real_mass_and_output_usage() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        let engine = Arc::new(ReadEngine {
            malformed: false,
            unknown_cache: false,
            calls: Mutex::new(vec![]),
        });
        let mut selection = selection(TestEngine::new(EngineBehavior::PendingStream));
        selection.profile = DecisionExecutionProfile::VllmDiffusionV1;
        selection.executor = DecisionExecutor::NativeRead {
            engine: engine.clone(),
        };
        let mut request = canonical_request(2).as_ref().clone();
        request.dialect = dialect;
        let prepared = super::super::head::prepare(&request, &selection, 100).unwrap();
        let admission = Arc::new(Semaphore::new(2));
        let response = super::super::head::dispatch(
            selection,
            prepared,
            "native-read".into(),
            Context::new(()).context(),
            admission.clone().acquire_many_owned(2).await.unwrap(),
            Arc::new(super::super::Metrics::new()),
            Arc::new(request),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-request-id"], "native-read");
        let body = response_body(response).await;
        assert_eq!(body["usage"]["input_tokens"], 17);
        assert_eq!(body["usage"]["output_tokens"], 4);
        if dialect == dynamo_decisions::Dialect::OpenAi {
            assert_eq!(body["answers"].as_array().unwrap().len(), 2);
            assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 0);
        } else {
            assert_eq!(body["answers"].as_object().unwrap().len(), 2);
        }
        assert_eq!(engine.calls.lock().unwrap().len(), 1);
        assert_eq!(admission.available_permits(), 2);
    }
}

#[tokio::test]
async fn native_read_invalid_mass_is_not_a_normal_answer() {
    let engine = Arc::new(ReadEngine {
        malformed: true,
        unknown_cache: false,
        calls: Mutex::new(vec![]),
    });
    let mut selection = selection(TestEngine::new(EngineBehavior::PendingStream));
    selection.profile = DecisionExecutionProfile::VllmDiffusionV1;
    selection.executor = DecisionExecutor::NativeRead { engine };
    let request = canonical_request(1);
    let prepared = super::super::head::prepare(&request, &selection, 100).unwrap();
    let parent = Context::new(()).context();
    let admission = Arc::new(Semaphore::new(1));
    let response = super::super::head::dispatch(
        selection,
        prepared,
        "bad-native".into(),
        parent.clone(),
        admission.clone().acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        request,
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(response_body(response).await.get("answers").is_none());
    assert!(parent.is_killed());
    assert_eq!(admission.available_permits(), 1);
}

#[tokio::test]
async fn native_read_unknown_cache_is_not_invented_for_openai() {
    let engine = Arc::new(ReadEngine {
        malformed: false,
        unknown_cache: true,
        calls: Mutex::new(vec![]),
    });
    let mut selected = selection(TestEngine::new(EngineBehavior::PendingStream));
    selected.profile = DecisionExecutionProfile::VllmDiffusionV1;
    selected.executor = DecisionExecutor::NativeRead { engine };
    let mut request = canonical_request(1).as_ref().clone();
    request.dialect = dynamo_decisions::Dialect::OpenAi;
    let request = Arc::new(request);
    let prepared = super::super::head::prepare(&request, &selected, 100).unwrap();
    let admission = Arc::new(Semaphore::new(1));
    let response = super::super::head::dispatch(
        selected,
        prepared,
        "unknown-cache".into(),
        Context::new(()).context(),
        admission.acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        request,
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(response_body(response).await.get("answers").is_none());
}

struct PendingReadEngine {
    started: Notify,
    context: Mutex<Option<Arc<dyn AsyncEngineContext>>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<DecisionReadRequest>, ManyOut<Annotated<DecisionReadResponse>>, Error>
    for PendingReadEngine
{
    async fn generate(
        &self,
        request: SingleIn<DecisionReadRequest>,
    ) -> Result<ManyOut<Annotated<DecisionReadResponse>>, Error> {
        let context = request.context();
        self.context.lock().unwrap().replace(context.clone());
        self.started.notify_one();
        Ok(ResponseStream::new(Box::pin(stream::pending()), context))
    }
}

#[tokio::test]
async fn native_read_deadline_kills_linked_child_and_releases_admission() {
    let engine = Arc::new(PendingReadEngine {
        started: Notify::new(),
        context: Mutex::new(None),
    });
    let mut selected = selection(TestEngine::new(EngineBehavior::PendingStream));
    selected.profile = DecisionExecutionProfile::VllmLiquidV1;
    selected.executor = DecisionExecutor::NativeRead {
        engine: engine.clone(),
    };
    let request = canonical_request(1);
    let prepared = super::super::head::prepare(&request, &selected, 100).unwrap();
    let admission = Arc::new(Semaphore::new(1));
    let parent = Context::new(()).context();
    let task = tokio::spawn(super::super::head::dispatch(
        selected,
        prepared,
        "deadline-read".into(),
        parent.clone(),
        admission.clone().acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        request,
    ));
    tokio::time::timeout(Duration::from_secs(2), engine.started.notified())
        .await
        .unwrap();
    let response = super::super::response::dispatch_with_deadline(
        task,
        tokio::time::Instant::now() + Duration::from_millis(10),
        parent.clone(),
        dynamo_decisions::Dialect::OpenAi,
    )
    .await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert!(parent.is_killed());
    assert!(engine.context.lock().unwrap().as_ref().unwrap().is_killed());
    assert_eq!(admission.available_permits(), 1);
}
