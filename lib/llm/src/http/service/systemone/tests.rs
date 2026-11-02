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
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use tokio::sync::{Notify, Semaphore};

#[path = "tests/backend_failure.rs"]
mod backend_failure;
#[path = "tests/native_abort.rs"]
mod native_abort;
#[path = "tests/preflight.rs"]
mod preflight;

#[path = "tests/vllm.rs"]
mod vllm;

#[path = "tests/http.rs"]
mod http;

#[path = "tests/metrics.rs"]
mod metrics;

use super::{
    Placement, PreparedBranch, ScoringBranch, StartedBranch, add_input_tokens, bind_native,
    pin_to_placement, resolve_dp_rank,
};
use crate::{
    discovery::{
        DecisionEngine, DecisionRoute, SystemOneExecutionSelection, TokenExecutor, WorkerSet,
    },
    local_model::runtime_config::{DecisionExecutionProfile as Profile, TokenBackend},
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::{
        Annotated,
        common::{
            llm_backend::LLMEngineOutput,
            preprocessor::{PreprocessedRequest, RoutingHints},
            timing::RequestTracker,
        },
        systemone::build_native_score_request,
    },
};

#[test]
fn cumulative_input_limit_is_checked_without_overflow() {
    assert_eq!(add_input_tokens(7, 5, 12), Ok(12));
    assert_eq!(add_input_tokens(65_536, 1, 131_072), Ok(65_537));
    assert!(add_input_tokens(7, 6, 12).is_err());
    assert!(add_input_tokens(usize::MAX, 1, usize::MAX).is_err());
}

#[test]
fn placement_pin_sets_both_worker_and_rank() {
    crate::generate_format::test_formats::install();
    let native = build_native_score_request(&[1], &[2], "test-salt").unwrap();
    let mut preprocessed = bind_native("sglang", "test-model", 16, &native, "request-1")
        .unwrap()
        .request;
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
    prepared_noul_branch_with_salt(index, "fixture")
}

fn prepared_noul_branch_with_salt(index: usize, cache_salt: &str) -> PreparedBranch {
    let prompt_ids = vec![1, 2];
    let label_ids = vec![17 + index as u32 * 100, 4 + index as u32 * 100];
    let native = build_native_score_request(&prompt_ids, &label_ids, cache_salt).unwrap();
    PreparedBranch {
        scoring: ScoringBranch {
            index,
            temperature: 1.0,
            prompt_len: prompt_ids.len(),
            label_ids,
        },
        wire: super::PreparedScoringRequest::Sglang(Box::new(native)),
    }
}

fn started_branch(
    prepared: PreparedBranch,
    tracker: Arc<RequestTracker>,
    stream: ManyOut<Annotated<LLMEngineOutput>>,
) -> StartedBranch {
    crate::generate_format::test_formats::install();
    let exchange = prepared
        .wire
        .compile("test-model", 16, "branch-0")
        .unwrap()
        .exchange;
    StartedBranch {
        prepared: prepared.scoring,
        tracker,
        stream,
        exchange,
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
    let result = super::token_scorer::finish_branch(
        TokenBackend::Sglang,
        started_branch(prepared_noul_branch(0), tracker, stream),
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

#[test]
fn sglang_scoring_preprocessing_preserves_exact_tracker_isl() {
    let selected = selection(TestEngine::new(EngineBehavior::Success));
    let request =
        super::token_scorer::PreparedScoringRequest::sglang(vec![1, 2, 3], &[17, 4], "isolated")
            .unwrap();
    let preprocessed = request
        .compile(
            &selected.context.canonical_model,
            selected.context.card.kv_cache_block_size,
            "isl-regression",
        )
        .unwrap()
        .request;
    let tracker = preprocessed
        .tracker
        .as_ref()
        .expect("canonical preprocessor tracker");
    assert_eq!(preprocessed.token_ids.as_slice(), &[1, 2, 3]);
    assert_eq!(tracker.isl_tokens(), Some(3));
    assert_eq!(tracker.cached_tokens(), None);
    assert_eq!(
        preprocessed
            .routing
            .as_ref()
            .unwrap()
            .expected_output_tokens,
        Some(0)
    );
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
    let result = super::token_scorer::finish_branch(
        TokenBackend::Sglang,
        started_branch(prepared_noul_branch(0), tracker, stream),
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

#[tokio::test]
async fn cancelled_stream_completion_is_not_a_malformed_response() {
    let parent = Context::new(()).context();
    let stream_parent = parent.clone();
    let stream = stream::poll_fn(move |_| {
        stream_parent.kill();
        std::task::Poll::Ready(None::<Annotated<LLMEngineOutput>>)
    });
    let result = super::token_scorer::finish_branch(
        TokenBackend::Sglang,
        started_branch(
            prepared_noul_branch(0),
            Arc::new(RequestTracker::new()),
            ResponseStream::new(Box::pin(stream), Context::new(()).context()),
        ),
        0,
        1,
        parent,
    )
    .await;
    let Err(error) = result else {
        panic!("cancelled stream must not produce an answer")
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
    RejectRequest {
        class: dynamo_runtime::error::ErrorClass,
        branch: usize,
        in_stream: bool,
    },
    TransportFailure,
    RemoveWorkerDuringFanout,
    NativeAbort {
        status: Option<u16>,
        branch: usize,
    },
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
    admissions: Option<tokio::sync::watch::Sender<Vec<u64>>>,
    frame: fn(&PreprocessedRequest) -> (Value, Annotated<LLMEngineOutput>),
}

impl TestEngine {
    fn new(behavior: EngineBehavior) -> Arc<Self> {
        Self::with_frame(behavior, sglang_frame)
    }
    fn with_frame(
        behavior: EngineBehavior,
        frame: fn(&PreprocessedRequest) -> (Value, Annotated<LLMEngineOutput>),
    ) -> Arc<Self> {
        Arc::new(Self {
            behavior,
            observations: Mutex::new(Vec::new()),
            started: Notify::new(),
            sibling_started: Notify::new(),
            admissions: None,
            frame,
        })
    }
}

fn sglang_frame(request: &PreprocessedRequest) -> (Value, Annotated<LLMEngineOutput>) {
    let native = request.extra_args.as_ref().unwrap()["sglang_tito"].clone();
    let labels: Vec<u32> = serde_json::from_value(native["token_ids_logprob"].clone()).unwrap();
    let terminal = terminal_output(&labels);
    (native, terminal)
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
        let (native, terminal) = (self.frame)(&request);
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
        if let EngineBehavior::RejectRequest {
            class,
            branch,
            in_stream,
        } = self.behavior
            && index == branch
        {
            let error = dynamo_runtime::error::DynamoError::builder()
                .class(class)
                .message("private deterministic worker detail")
                .build();
            if !in_stream {
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
        if matches!(self.behavior, EngineBehavior::TransportFailure) {
            let transport = std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "private transport detail",
            );
            let error = dynamo_runtime::error::DynamoError::from(
                &transport as &(dyn std::error::Error + 'static),
            );
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
        if matches!(self.behavior, EngineBehavior::RemoveWorkerDuringFanout) && index == 1 {
            self.admissions.as_ref().unwrap().send_replace(vec![99]);
        }
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
        if let EngineBehavior::NativeAbort { status, branch } = self.behavior {
            if index == branch + 1 {
                self.sibling_started.notify_one();
                return Ok(ResponseStream::new(
                    Box::pin(stream::pending()),
                    stream_context,
                ));
            }
            if index == branch {
                if branch > 0 {
                    self.sibling_started.notified().await;
                }
                let abort = Annotated::from_data(LLMEngineOutput {
                    engine_data: Some(json!({"sglang_response": {
                        "output_ids": [],
                        "meta_info": {"finish_reason": {
                            "type": "abort",
                            "status_code": status,
                            "message": "private worker detail"
                        }}
                    }})),
                    ..Default::default()
                });
                if branch == 0 {
                    return Ok(ResponseStream::new(
                        Box::pin(stream::iter([abort])),
                        stream_context,
                    ));
                }
                return Ok(ResponseStream::new(
                    Box::pin(stream::iter([abort]).chain(stream::pending())),
                    stream_context,
                ));
            }
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
            Box::pin(stream::iter([terminal])),
            stream_context,
        ))
    }
}

fn selection(engine: Arc<TestEngine>) -> SystemOneExecutionSelection {
    crate::generate_format::test_formats::install();
    let mut card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    card.set_name("test-model");
    card.runtime_config.data_parallel_size = 4;
    let preprocessor = OpenAIPreprocessor::new(card.clone()).unwrap();
    token_selection(card, engine, preprocessor, Profile::SglangSerialV1)
}

fn token_selection(
    card: ModelDeploymentCard,
    engine: crate::types::openai::generate::GenerateStreamingEngine,
    preprocessor: Arc<OpenAIPreprocessor>,
    profile: Profile,
) -> SystemOneExecutionSelection {
    let mut workers = WorkerSet::new("test-workers".into(), card.mdcsum().into(), card);
    let crate::local_model::runtime_config::DecisionProfileKind::Token(backend) = profile.kind()
    else {
        panic!("token fixture requires a token profile")
    };
    workers.decision = Some(DecisionRoute::Available(DecisionEngine::Token(
        TokenExecutor::new(backend, engine, preprocessor),
    )));
    SystemOneExecutionSelection::for_test("test-model".into(), profile, &workers).unwrap()
}

fn token_executor(selection: &SystemOneExecutionSelection) -> &TokenExecutor {
    let DecisionEngine::Token(executor) = selection.engine() else {
        panic!("expected token executor")
    };
    executor
}

fn prepare_branches(
    request: &dynamo_decisions::CanonicalRequest,
    selection: &SystemOneExecutionSelection,
    limit: usize,
) -> Result<Vec<PreparedBranch>, (StatusCode, String)> {
    super::scoring::prepare_branches(
        request,
        &selection.context,
        token_executor(selection),
        limit,
        &super::fresh_cache_salt(),
    )
}

async fn dispatch(
    selection: SystemOneExecutionSelection,
    branches: Vec<PreparedBranch>,
    request_id: String,
    parent: Arc<dyn AsyncEngineContext>,
    permit: tokio::sync::OwnedSemaphorePermit,
    metrics: Arc<super::Metrics>,
    request: Arc<dynamo_decisions::CanonicalRequest>,
) -> Response {
    super::execution::dispatch(
        super::execution::PreparedExecution::token(selection, branches).unwrap(),
        request_id,
        parent,
        permit,
        metrics,
        "test-model".into(),
        request,
    )
    .await
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
    let cache_salt = super::fresh_cache_salt();
    let response = dispatch(
        selection(engine),
        (0..branch_count)
            .map(|index| prepared_noul_branch_with_salt(index, &cache_salt))
            .collect(),
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
    assert_eq!(body["detail"], "decision scoring worker is unavailable");
    assert!(parent.is_killed());
    assert_eq!(semaphore.available_permits(), 2);
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
