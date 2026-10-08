// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::pending,
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use axum::{body::to_bytes, http::StatusCode, response::Response};
use base64::Engine as _;
use dynamo_runtime::{
    engine::{AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider},
    pipeline::{Context, Error, ManyOut, ResponseStream, SingleIn},
};
use futures::stream;
use serde_json::{Value, json};
use tokio::sync::{Notify, Semaphore};

#[tokio::test]
async fn dialect_errors_preserve_request_ids_and_validation_envelopes() {
    use dynamo_decisions::{DecisionError, Dialect};
    for dialect in [Dialect::OpenAi, Dialect::SglangNative, Dialect::Jev] {
        let mut response =
            super::decision_error(DecisionError::validation(dialect, "invalid questions"));
        super::attach_request_ids(&mut response, "fixture-request");
        assert_eq!(response.headers()["x-request-id"], "fixture-request");
        assert_eq!(
            response.headers()["x-typesafe-request-id"],
            "fixture-request"
        );
        assert_eq!(
            response.status().as_u16(),
            if dialect == Dialect::Jev { 422 } else { 400 }
        );
        let body = response_body(response).await;
        match dialect {
            Dialect::OpenAi => assert_eq!(body["error"]["type"], "invalid_request_error"),
            Dialect::SglangNative => assert_eq!(body["object"], "error"),
            Dialect::Jev => assert!(body["detail"].is_array()),
        }
    }
}

#[test]
fn missing_cache_measurement_is_not_zero() {
    assert_eq!(
        super::measured_cached_tokens(&json!({"meta_info":{}}), 4).unwrap(),
        None
    );
    assert_eq!(
        super::measured_cached_tokens(&json!({"meta_info":{"cached_tokens":0}}), 4).unwrap(),
        Some(0)
    );
    assert!(super::measured_cached_tokens(&json!({"meta_info":{"cached_tokens":5}}), 4).is_err());
    assert!(super::measured_cached_tokens(&json!({"meta_info":{"cached_tokens":-1}}), 4).is_err());
}

#[tokio::test]
async fn deadline_kills_stalled_dispatch_and_releases_local_admission() {
    let engine = TestEngine::new(EngineBehavior::PendingStream);
    let parent = Context::new(()).context();
    let admission = Arc::new(Semaphore::new(1));
    let permit = admission.clone().acquire_owned().await.unwrap();
    let task = tokio::spawn(dispatch(
        selection(engine.clone()),
        vec![prepared_noul_branch(0)],
        "deadline".into(),
        parent.clone(),
        permit,
        Arc::new(super::super::metrics::Metrics::new()),
        canonical_request(1),
    ));
    tokio::time::timeout(Duration::from_secs(2), engine.started.notified())
        .await
        .unwrap();
    let response = super::response::dispatch_with_deadline(
        task,
        tokio::time::Instant::now() + Duration::from_millis(20),
        parent.clone(),
        dynamo_decisions::Dialect::Jev,
    )
    .await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert!(parent.is_killed());
    assert_eq!(admission.available_permits(), 1);
    assert!(
        engine
            .observations
            .lock()
            .unwrap()
            .iter()
            .all(|o| o.request_context.is_killed())
    );
}

use super::{
    Placement, PreparedBranch, StartedBranch, add_input_tokens, dispatch, finish_branch,
    pin_to_placement, resolve_dp_rank,
};
use crate::{
    discovery::SystemOneExecutionSelection,
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::{
        Annotated,
        common::{
            llm_backend::LLMEngineOutput,
            preprocessor::{PreprocessedRequest, RoutingHints},
            timing::RequestTracker,
        },
        sglang::generate::SglangGenerateRequest,
        systemone::build_native_score_request,
    },
};

#[test]
fn derives_only_a_proven_single_data_parallel_rank() {
    assert_eq!(resolve_dp_rank(None, 7, 1), Some(7));
    assert_eq!(resolve_dp_rank(None, 7, 2), None);
    assert_eq!(resolve_dp_rank(Some(9), 7, 2), Some(9));
}

#[test]
fn cumulative_input_limit_is_checked_without_overflow() {
    assert_eq!(add_input_tokens(7, 5, 12), Ok(12));
    assert!(add_input_tokens(7, 6, 12).is_err());
    assert!(add_input_tokens(usize::MAX, 1, usize::MAX).is_err());
}

#[test]
fn impossible_admission_is_not_retryable_but_busy_capacity_is() {
    let admission = Arc::new(Semaphore::new(2));
    let response =
        super::acquire_admission(admission.clone(), 3, 2, dynamo_decisions::Dialect::Jev)
            .unwrap_err();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert!(!response.headers().contains_key("retry-after"));
    let permit =
        super::acquire_admission(admission.clone(), 2, 2, dynamo_decisions::Dialect::Jev).unwrap();
    let response =
        super::acquire_admission(admission.clone(), 1, 2, dynamo_decisions::Dialect::Jev)
            .unwrap_err();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "1");
    drop(permit);
    assert!(super::acquire_admission(admission, 1, 2, dynamo_decisions::Dialect::Jev).is_ok());
}

#[test]
fn candidate_tokenization_work_bounds_prompt_size() {
    assert_eq!(super::encoding_prompt_limit(120, 10), 10);
    assert_eq!(super::encoding_prompt_limit(11, 10), 0);
    assert_eq!(super::encoding_prompt_limit(120, 2), 30);
}

#[test]
fn placement_pin_sets_both_worker_and_rank() {
    let native: SglangGenerateRequest =
        serde_json::from_value(build_native_score_request(&[1], &[2], "test-salt").unwrap())
            .unwrap();
    let mut preprocessed = super::super::sglang_generate::preprocessed_request(
        native,
        "test-model",
        None,
        "request-1",
    )
    .unwrap();
    pin_to_placement(
        &mut preprocessed,
        Placement {
            worker_id: 41,
            dp_rank: 3,
        },
    );
    let routing = preprocessed.routing.unwrap();
    assert_eq!(routing.backend_instance_id, Some(41));
    assert_eq!(routing.dp_rank, Some(3));
}

fn prepared_noul_branch(index: usize) -> PreparedBranch {
    let prompt_ids = vec![1, 2];
    let label_ids = vec![17 + index as u32 * 100, 4 + index as u32 * 100];
    let native = serde_json::from_value(
        build_native_score_request(&prompt_ids, &label_ids, "validation").unwrap(),
    )
    .unwrap();
    PreparedBranch {
        index,
        temperature: 1.0,
        prompt_ids,
        label_ids,
        native: Some(native),
    }
}

fn terminal_output(label_ids: &[u32]) -> Annotated<LLMEngineOutput> {
    Annotated::from_data(LLMEngineOutput {
        engine_data: Some(json!({
            "sglang_response": {
                "output_ids": [],
                "meta_info": {
                    "finish_reason": {"type": "length"},
                    "prompt_tokens": 999,
                    "completion_tokens": 0,
                    "output_token_ids_logprobs": [[
                        [-0.3, label_ids[0], null],
                        [-1.4, label_ids[1], null]
                    ]]
                }
            }
        })),
        ..Default::default()
    })
}

#[tokio::test]
async fn finish_branch_parses_terminal_scores_and_tracker_placement() {
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_worker(41, Some(3), "decode");
    let stream = ResponseStream::new(
        Box::pin(stream::iter([terminal_output(&[17, 4])])),
        Context::new(()).context(),
    );
    let parent = Context::new(());
    let result = finish_branch(
        StartedBranch {
            prepared: prepared_noul_branch(0),
            tracker,
            stream,
        },
        0,
        4,
        parent.context(),
    )
    .await;
    let Ok(result) = result else {
        panic!("valid terminal branch should succeed")
    };
    assert_eq!(result.placement.worker_id, 41);
    assert_eq!(result.placement.dp_rank, 3);
    assert_eq!(result.input_tokens, 2);
    assert!(matches!(
        result.answer,
        dynamo_decisions::QuestionOutcome::Answer(_)
    ));
}

#[tokio::test]
async fn finish_branch_stops_when_parent_is_cancelled() {
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_worker(41, Some(0), "decode");
    let stream = ResponseStream::new(
        Box::pin(stream::pending::<Annotated<LLMEngineOutput>>()),
        Context::new(()).context(),
    );
    let parent = Context::new(());
    let parent_context = parent.context();
    parent_context.kill();
    let result = finish_branch(
        StartedBranch {
            prepared: prepared_noul_branch(0),
            tracker,
            stream,
        },
        0,
        1,
        parent_context,
    )
    .await;
    let Err(error) = result else {
        panic!("cancelled parent should stop branch completion")
    };
    assert_eq!(error.status.as_u16(), 499);
}

#[derive(Clone, Copy)]
enum EngineBehavior {
    Success,
    PlacementMismatch,
    MalformedWithPendingSibling,
    PendingGenerate,
    PendingStream,
    PendingSiblings,
    RejectGenerate,
    RejectStream,
}

struct ObservedBranch {
    index: usize,
    routing: RoutingHints,
    native: Value,
    request_context: Arc<dyn AsyncEngineContext>,
    stream_context: Arc<dyn AsyncEngineContext>,
}

struct TestEngine {
    behavior: EngineBehavior,
    observations: Mutex<Vec<ObservedBranch>>,
    started: Notify,
    sibling_started: Notify,
}

impl TestEngine {
    fn new(behavior: EngineBehavior) -> Arc<Self> {
        Arc::new(Self {
            behavior,
            observations: Mutex::new(Vec::new()),
            started: Notify::new(),
            sibling_started: Notify::new(),
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for TestEngine
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let index: usize = request.id().rsplit('-').next().unwrap().parse().unwrap();
        let native = request.extra_args.as_ref().unwrap()["sglang_tito"].clone();
        let label_ids: Vec<u32> =
            serde_json::from_value(native["token_ids_logprob"].clone()).unwrap();
        let stream_context = Context::new(()).context();
        self.observations.lock().unwrap().push(ObservedBranch {
            index,
            routing: request.routing.clone().unwrap(),
            native,
            request_context: request.context(),
            stream_context: stream_context.clone(),
        });
        let worker_id = if matches!(self.behavior, EngineBehavior::PlacementMismatch) && index == 1
        {
            99
        } else {
            41
        };
        request
            .tracker
            .as_ref()
            .unwrap()
            .record_worker(worker_id, Some(3), "decode");
        self.started.notify_one();
        if matches!(
            self.behavior,
            EngineBehavior::RejectGenerate | EngineBehavior::RejectStream
        ) {
            let error = dynamo_runtime::error::DynamoError::builder()
                .class(dynamo_runtime::error::ErrorClass::CapacityExhausted)
                .message("private worker detail")
                .build();
            if matches!(self.behavior, EngineBehavior::RejectGenerate) {
                return Err(error.into());
            }
            return Ok(ResponseStream::new(
                Box::pin(stream::iter([Annotated {
                    data: None,
                    id: None,
                    event: Some("error".to_string()),
                    comment: None,
                    error: Some(error),
                }])),
                stream_context,
            ));
        }
        if matches!(self.behavior, EngineBehavior::PendingGenerate) {
            return pending().await;
        }
        if matches!(self.behavior, EngineBehavior::PendingStream)
            || (matches!(self.behavior, EngineBehavior::PendingSiblings) && index > 0)
        {
            return Ok(ResponseStream::new(
                Box::pin(stream::pending()),
                stream_context,
            ));
        }
        if matches!(self.behavior, EngineBehavior::MalformedWithPendingSibling) {
            if index == 2 {
                self.sibling_started.notify_one();
                return Ok(ResponseStream::new(
                    Box::pin(stream::pending()),
                    stream_context,
                ));
            }
            if index == 1 {
                self.sibling_started.notified().await;
                return Ok(ResponseStream::new(
                    Box::pin(stream::iter([Annotated::from_data(
                        LLMEngineOutput::default(),
                    )])),
                    stream_context,
                ));
            }
        }
        Ok(ResponseStream::new(
            Box::pin(stream::iter([terminal_output(&label_ids)])),
            stream_context,
        ))
    }
}

fn selection(engine: Arc<TestEngine>) -> SystemOneExecutionSelection {
    let mut card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    card.set_name("test-model");
    card.runtime_config.data_parallel_size = 4;
    let preprocessor = OpenAIPreprocessor::new(card.clone()).unwrap();
    SystemOneExecutionSelection {
        canonical_model: "test-model".to_string(),
        engine,
        card,
        preprocessor,
        admitted_ids: None,
    }
}

async fn response_body(response: Response) -> Value {
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

fn canonical_request(branch_count: usize) -> Arc<dynamo_decisions::CanonicalRequest> {
    let questions: serde_json::Map<String, Value> = (0..branch_count)
        .map(|index| {
            (
                format!("question-{index}"),
                json!({"type":"noul", "instructions":"urgent?"}),
            )
        })
        .collect();
    Arc::new(
        dynamo_decisions::parse_request(
            &serde_json::to_vec(
                &json!({"model":"test-model", "state":"test", "questions":questions}),
            )
            .unwrap(),
            dynamo_decisions::Route::SystemOne,
        )
        .unwrap(),
    )
}

async fn dispatch_request(
    engine: Arc<TestEngine>,
    branch_count: usize,
    request_id: &str,
) -> (Response, Arc<dyn AsyncEngineContext>, Arc<Semaphore>) {
    let parent = Context::new(()).context();
    let semaphore = Arc::new(Semaphore::new(branch_count));
    let permit = semaphore
        .clone()
        .acquire_many_owned(branch_count as u32)
        .await
        .unwrap();
    let response = dispatch(
        selection(engine),
        (0..branch_count).map(prepared_noul_branch).collect(),
        request_id.to_string(),
        parent.clone(),
        permit,
        Arc::new(super::super::metrics::Metrics::new()),
        canonical_request(branch_count),
    )
    .await;
    (response, parent, semaphore)
}

#[tokio::test]
async fn dispatch_limits_active_siblings_and_cancels_queued_work() {
    let engine = TestEngine::new(EngineBehavior::PendingSiblings);
    let parent = Context::new(()).context();
    let admission = Arc::new(Semaphore::new(20));
    let permit = admission.clone().acquire_many_owned(20).await.unwrap();
    let task = tokio::spawn(dispatch(
        selection(engine.clone()),
        (0..20).map(prepared_noul_branch).collect(),
        "bounded-fanout".to_string(),
        parent.clone(),
        permit,
        Arc::new(super::super::metrics::Metrics::new()),
        canonical_request(20),
    ));
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if engine.observations.lock().unwrap().len() >= 5 {
                break;
            }
            engine.started.notified().await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(engine.observations.lock().unwrap().len(), 5);
    parent.kill();
    let response = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.status().as_u16(), 499);
    assert_eq!(engine.observations.lock().unwrap().len(), 5);
    assert_eq!(admission.available_permits(), 20);
}

#[tokio::test]
async fn dispatch_pins_siblings_and_isolates_request_salts() {
    let engine = TestEngine::new(EngineBehavior::Success);
    let (response, parent, semaphore) = dispatch_request(engine.clone(), 3, "request-one").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-dynamo-systemone-version"], "1");
    let body = response_body(response).await;
    assert_eq!(
        body["usage"],
        json!({"input_tokens": 6, "output_tokens": 0})
    );
    assert_eq!(
        body["answers"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["question-0", "question-1", "question-2"]
    );
    assert!(!parent.is_killed());
    assert_eq!(semaphore.available_permits(), 3);
    let salt = {
        let observations = engine.observations.lock().unwrap();
        assert_eq!(observations.len(), 3);
        let salt = observations[0].native["cache_salt"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(&salt)
                .unwrap()
                .len(),
            16
        );
        for observation in observations.iter() {
            assert_eq!(
                observation.routing.cache_namespace.as_deref(),
                Some(salt.as_str())
            );
            assert_eq!(observation.native["cache_salt"], salt);
            assert_eq!(
                observation.native["token_ids_logprob"],
                json!([
                    17 + observation.index as u32 * 100,
                    4 + observation.index as u32 * 100
                ])
            );
            assert_eq!(observation.routing.expected_output_tokens, Some(0));
            if observation.index == 0 {
                assert_eq!(observation.routing.backend_instance_id, None);
                assert_eq!(observation.routing.dp_rank, None);
            } else {
                assert_eq!(observation.routing.backend_instance_id, Some(41));
                assert_eq!(observation.routing.dp_rank, Some(3));
            }
        }
        salt
    };
    let (response, _, _) = dispatch_request(engine.clone(), 1, "request-two").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_ne!(
        engine.observations.lock().unwrap()[3].native["cache_salt"],
        salt
    );
}

#[tokio::test]
async fn sibling_failure_is_atomic_and_cancels_pending_peer() {
    let engine = TestEngine::new(EngineBehavior::MalformedWithPendingSibling);
    let (response, parent, semaphore) = tokio::time::timeout(
        Duration::from_secs(2),
        dispatch_request(engine.clone(), 3, "request"),
    )
    .await
    .expect("sibling failure must terminate without draining the pending peer");
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(response_body(response).await.get("answers").is_none());
    assert!(parent.is_killed());
    assert_eq!(semaphore.available_permits(), 3);
    let observations = engine.observations.lock().unwrap();
    let pending_peer = observations
        .iter()
        .find(|observation| observation.index == 2)
        .unwrap();
    assert!(pending_peer.request_context.is_killed());
    assert!(pending_peer.stream_context.is_killed());
}

#[tokio::test]
async fn placement_mismatch_fails_with_unavailable_without_answers() {
    let engine = TestEngine::new(EngineBehavior::PlacementMismatch);
    let (response, parent, semaphore) = dispatch_request(engine, 2, "request").await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response_body(response).await;
    assert!(body.get("answers").is_none());
    assert_eq!(body["detail"], "SGLang scoring worker is unavailable");
    assert!(parent.is_killed());
    assert_eq!(semaphore.available_permits(), 2);
}

#[tokio::test]
async fn backend_overload_preserves_retry_status_and_releases_admission() {
    for behavior in [EngineBehavior::RejectGenerate, EngineBehavior::RejectStream] {
        let (response, parent, semaphore) =
            dispatch_request(TestEngine::new(behavior), 2, "request").await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "1");
        let body = response_body(response).await;
        assert!(body.get("answers").is_none());
        assert!(!body.to_string().contains("private worker detail"));
        assert!(parent.is_killed());
        assert_eq!(semaphore.available_permits(), 2);
    }
}

async fn assert_cancellation(behavior: EngineBehavior) {
    let engine = TestEngine::new(behavior);
    let parent = Context::new(()).context();
    let semaphore = Arc::new(Semaphore::new(2));
    let permit = semaphore.clone().acquire_many_owned(2).await.unwrap();
    let task = tokio::spawn(dispatch(
        selection(engine.clone()),
        vec![prepared_noul_branch(0), prepared_noul_branch(1)],
        "request".to_string(),
        parent.clone(),
        permit,
        Arc::new(super::super::metrics::Metrics::new()),
        canonical_request(2),
    ));
    tokio::time::timeout(Duration::from_secs(2), engine.started.notified())
        .await
        .unwrap();
    parent.kill();
    let response = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.status().as_u16(), 499);
    assert!(response_body(response).await.get("answers").is_none());
    assert_eq!(semaphore.available_permits(), 2);
    let observations = engine.observations.lock().unwrap();
    assert_eq!(observations.len(), 1);
    assert!(observations[0].request_context.is_killed());
    if matches!(behavior, EngineBehavior::PendingStream) {
        assert!(observations[0].stream_context.is_killed());
    }
}

#[tokio::test]
async fn parent_cancel_interrupts_actual_generate_and_releases_admission() {
    assert_cancellation(EngineBehavior::PendingGenerate).await;
}

#[tokio::test]
async fn parent_cancel_interrupts_actual_stream_and_releases_admission() {
    assert_cancellation(EngineBehavior::PendingStream).await;
}
