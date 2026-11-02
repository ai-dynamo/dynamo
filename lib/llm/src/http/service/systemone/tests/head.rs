// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    discovery::{DecisionEngine, Model, WorkerSet},
    model_type::{ModelInput, ModelType},
    protocols::{
        decision_head::{DecisionHeadRequest, DecisionHeadResponse, HeadOutcome},
        decision_native::NativeUsage,
    },
    worker_type::WorkerType,
};

#[derive(Clone, Copy)]
enum Behavior {
    Success,
    Malformed,
    Reordered,
    Error,
    InvalidRequest,
    Unavailable,
    Pending,
    Empty,
    Duplicate,
    CacheDisabled,
    UnknownCache,
}

struct HeadEngine {
    behavior: Behavior,
    calls: Mutex<Vec<DecisionHeadRequest>>,
    contexts: Mutex<Vec<Arc<dyn AsyncEngineContext>>>,
    started: Notify,
}

impl HeadEngine {
    fn new(behavior: Behavior) -> Arc<Self> {
        Arc::new(Self {
            behavior,
            calls: Mutex::new(Vec::new()),
            contexts: Mutex::new(Vec::new()),
            started: Notify::new(),
        })
    }
}

#[tokio::test]
async fn native_heads_reject_single_level_scores_before_worker_dispatch() {
    for profile in [
        Profile::VllmD3V1,
        Profile::VllmClefFlashV1,
        Profile::VllmClefOmniV1,
    ] {
        let engine = HeadEngine::new(Behavior::Success);
        let request = single_level_score_request();
        let fixture = head_selection_for_test(engine.clone(), profile).await;
        let error = super::super::execution::prepare(&request, fixture.selection.clone(), 100)
            .err()
            .expect("native profile requires at least two score levels");
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(error.1.contains("at least two score levels"));
        assert!(engine.calls.lock().unwrap().is_empty());
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<DecisionHeadRequest>, ManyOut<Annotated<DecisionHeadResponse>>, Error>
    for HeadEngine
{
    async fn generate(
        &self,
        request: SingleIn<DecisionHeadRequest>,
    ) -> Result<ManyOut<Annotated<DecisionHeadResponse>>, Error> {
        let context = request.context();
        assert_eq!(
            request
                .metadata()
                .get(crate::protocols::systemone::DECISION_DISPATCH_METADATA_KEY)
                .map(String::as_str),
            Some("head")
        );
        let (request, _) = request.into_parts();
        self.contexts.lock().unwrap().push(context.clone());
        self.calls.lock().unwrap().push(request.clone());
        self.started.notify_one();
        if matches!(
            self.behavior,
            Behavior::Error
                | Behavior::InvalidRequest
                | Behavior::Unavailable
                | Behavior::UnknownCache
        ) {
            return Err(dynamo_runtime::error::DynamoError::builder()
                .class(match self.behavior {
                    Behavior::InvalidRequest => dynamo_runtime::error::ErrorClass::InvalidRequest,
                    Behavior::Unavailable => dynamo_runtime::error::ErrorClass::WorkerUnavailable,
                    Behavior::UnknownCache => dynamo_runtime::error::ErrorClass::BackendProtocol,
                    _ => dynamo_runtime::error::ErrorClass::CapacityExhausted,
                })
                .message("private head details")
                .build()
                .into());
        }
        if matches!(self.behavior, Behavior::Pending) {
            return Ok(ResponseStream::new(Box::pin(stream::pending()), context));
        }
        let mut outcomes: Vec<_> = request
            .decision_head
            .questions
            .iter()
            .map(|q| HeadOutcome {
                ordinal: q.ordinal,
                probabilities: vec![0.25, 0.75],
            })
            .collect();
        if matches!(self.behavior, Behavior::Reordered) {
            outcomes.reverse();
        }
        if matches!(self.behavior, Behavior::Malformed) {
            outcomes[0].probabilities = vec![0.1, 0.2];
        }
        let response = DecisionHeadResponse {
            finish_reason: crate::protocols::decision_native::NativeFinishReason::Stop,
            protocol_version: 1,
            profile: request.decision_head.profile,
            outcomes,
            usage: NativeUsage {
                input_tokens: 13,
                output_tokens: 0,
                cached_tokens: match self.behavior {
                    Behavior::CacheDisabled => 0,
                    _ => 2,
                },
            },
        };
        let frames = match self.behavior {
            Behavior::Empty => vec![],
            Behavior::Duplicate => vec![
                Annotated::from_data(response.clone()),
                Annotated::from_data(response),
            ],
            _ => vec![Annotated::from_data(response)],
        };
        Ok(ResponseStream::new(Box::pin(stream::iter(frames)), context))
    }
}

async fn head_selection_for_test(
    engine: Arc<HeadEngine>,
    profile: Profile,
) -> PreparedSelectionFixture {
    let mut card = ModelDeploymentCard::with_name_only("test-model");
    card.model_input = ModelInput::Text;
    card.model_type = ModelType::Classify | ModelType::Pooling;
    card.worker_type = Some(WorkerType::Aggregated);
    card.runtime_config.data_parallel_size = 1;
    card.runtime_config
        .runtime_data
        .insert(profile.decision_capability().into(), true.into());
    card.runtime_config.runtime_data.insert(
        "vllm_native_decision_max_candidates".into(),
        profile.limits().1.into(),
    );
    let workers = WorkerSet::new("heads".into(), "fixture".into(), card);
    let resources = crate::discovery::test_support::native_resources(DecisionEngine::Head(engine));
    crate::discovery::test_support::forced_selection_for_test(
        "test-model".into(),
        profile,
        workers,
        resources,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn native_openai_messages_reach_the_head_executor_as_adapter_text() {
    let request = Arc::new(dynamo_decisions::parse_request(&serde_json::to_vec(&json!({
        "model":"test-model", "input":[
            {"role":"user","content":[{"type":"input_text","text":"first"},{"type":"input_text","text":"second"}]},
            {"role":"user","content":"third"}
        ], "questions":[{"type":"predicate","instructions":"True?"}]
    })).unwrap(), dynamo_decisions::Route::Decisions).unwrap());
    let engine = HeadEngine::new(Behavior::Success);
    let fixture = head_selection_for_test(engine.clone(), Profile::VllmD3V1).await;
    let prepared =
        super::super::execution::prepare(&request, fixture.selection.clone(), 100).unwrap();
    let response = super::super::execution::dispatch(
        prepared,
        "native-input".into(),
        Context::new(()).context(),
        Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        "test-model".into(),
        request,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        engine.calls.lock().unwrap()[0].decision_head.input,
        json!("first\nsecond\n\nthird")
    );
}

async fn dispatch_head(
    engine: Arc<HeadEngine>,
    dialect: dynamo_decisions::Dialect,
) -> (Response, Arc<dyn AsyncEngineContext>, Arc<Semaphore>) {
    let selection = head_selection_for_test(engine, Profile::VllmClefFlashV1).await;
    let mut request = canonical_request(2).as_ref().clone();
    request.dialect = dialect;
    let prepared =
        super::super::execution::prepare(&request, selection.selection.clone(), 100).unwrap();
    let admission = Arc::new(Semaphore::new(2));
    let permit = admission.clone().acquire_many_owned(2).await.unwrap();
    let parent = Context::new(()).context();
    let response = super::super::execution::dispatch(
        prepared,
        "head-request".into(),
        parent.clone(),
        permit,
        Arc::new(super::super::Metrics::new()),
        "test-model".into(),
        Arc::new(request),
    )
    .await;
    (response, parent, admission)
}

#[tokio::test]
async fn native_dispatch_uses_bounded_metric_labels() {
    let request = canonical_request(2);
    let fixture =
        head_selection_for_test(HeadEngine::new(Behavior::Success), Profile::VllmD3V1).await;
    let prepared =
        super::super::execution::prepare(&request, fixture.selection.clone(), 100).unwrap();
    let metrics = Arc::new(super::super::Metrics::new_with_prefix(Some(
        "bounded_native".into(),
    )));
    let registry = prometheus::Registry::new();
    metrics.register(&registry).unwrap();
    let manager = crate::discovery::ModelManager::new();
    let label = manager.metric_model_for("test-model");
    assert_ne!(label, "test-model");
    super::super::execution::observe_work(label, &prepared, &metrics);
    let response = super::super::execution::dispatch(
        prepared,
        "bounded".into(),
        Context::new(()).context(),
        Arc::new(Semaphore::new(2))
            .acquire_many_owned(2)
            .await
            .unwrap(),
        metrics,
        label.into(),
        request,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    super::metrics::assert_bounded_labels(
        &registry,
        "bounded_native_systemone_",
        label,
        &["questions", "admitted_branches_total"],
    );
    assert!(!registry.gather().iter().any(|family| matches!(
        family.name(),
        "bounded_native_systemone_input_tokens" | "bounded_native_systemone_candidate_ids"
    )));
}

#[tokio::test]
async fn native_selection_caches_capacity_and_rejects_adapter_views() {
    for profile in [
        Profile::VllmD3V1,
        Profile::VllmClefFlashV1,
        Profile::VllmClefOmniV1,
    ] {
        let prototype = head_selection_for_test(HeadEngine::new(Behavior::Success), profile).await;
        let mut card = prototype.context.card.clone();
        card.runtime_config
            .runtime_data
            .insert("vllm_native_decision_max_candidates".into(), 3.into());
        let mut workers = WorkerSet::new("capacity".into(), card.mdcsum().into(), card);
        let (_admissions, admitted_ids) = tokio::sync::watch::channel(vec![1]);
        workers.set_instance_watcher(admitted_ids);
        let _qualification = crate::discovery::test_support::attach_card_surface_for_test(
            &mut workers,
            crate::discovery::test_support::native_resources(prototype.engine.clone()),
        )
        .await;
        let mut selected = crate::discovery::test_support::prepared_selection_for_test(
            "test-model".into(),
            &workers,
        )
        .await
        .unwrap();
        let expected = selected
            .context
            .card
            .decision_max_candidates(profile)
            .unwrap();
        assert_eq!(expected, 3);
        assert_eq!(
            selected.context.decision_capabilities().max_candidates,
            expected
        );
        let qualified_card = selected.context.card.clone();
        selected
            .context
            .card
            .runtime_config
            .runtime_data
            .remove("vllm_native_decision_max_candidates");
        assert_eq!(
            selected.context.decision_capabilities().max_candidates,
            expected
        );
        let mut adapter_card = qualified_card.clone();
        adapter_card.lora = Some(crate::model_card::LoraInfo {
            name: "adapter".into(),
            max_gpu_lora_count: Some(4),
        });
        let mut workers = WorkerSet::new("native".into(), "fixture".into(), qualified_card);
        let (_adapter_admissions, admitted_ids) = tokio::sync::watch::channel(vec![1]);
        workers.set_instance_watcher(admitted_ids);
        let _adapter_qualification = crate::discovery::test_support::attach_card_surface_for_test(
            &mut workers,
            crate::discovery::test_support::native_resources(selected.engine.clone()),
        )
        .await;
        let adapter = workers.adapter_view(adapter_card);
        let model = Model::new("adapter".into());
        model.add_worker_set("native".into(), Arc::new(adapter));
        assert!(matches!(
            model.prepared_selection_for_test().await,
            Err(crate::discovery::SystemOneSelectionError::Unsupported)
        ));
    }
}

#[tokio::test]
async fn native_selection_rejects_a_worker_engine_of_the_wrong_shape() {
    use crate::model_type::{ModelInput, ModelType};

    let selected =
        head_selection_for_test(HeadEngine::new(Behavior::Success), Profile::VllmD3V1).await;
    let mut card = selected.context.card.clone();
    card.runtime_config
        .runtime_data
        .remove(Profile::VllmD3V1.decision_capability());
    card.runtime_config.runtime_data.insert(
        Profile::VllmLiquidV1.decision_capability().into(),
        true.into(),
    );
    card.runtime_config
        .runtime_data
        .insert("vllm_native_decision_max_candidates".into(), 26.into());
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat;
    assert_eq!(
        card.decision_execution_profile(),
        Some(Profile::VllmLiquidV1)
    );
    let mut workers = WorkerSet::new("mismatched".into(), card.mdcsum().into(), card);
    let (_admissions, admitted_ids) = tokio::sync::watch::channel(vec![1]);
    workers.set_instance_watcher(admitted_ids);
    let _qualification = crate::discovery::test_support::attach_card_surface_for_test(
        &mut workers,
        crate::discovery::test_support::native_resources(selected.engine.clone()),
    )
    .await;
    assert!(
        crate::discovery::test_support::prepared_selection_for_test("test-model".into(), &workers)
            .await
            .is_none()
    );
    let model = Model::new("test-model".into());
    model.add_worker_set("mismatched".into(), Arc::new(workers));
    assert!(!model.has_decision_engine());
    assert!(model.prepared_selection_for_test().await.is_err());
}

#[tokio::test]
async fn native_tokenization_budget_tracks_the_operator_input_limit() {
    for input_limit in [
        32,
        64,
        crate::protocols::systemone::MAX_DECISION_INPUT_TOKENS,
    ] {
        let engine = HeadEngine::new(Behavior::Success);
        let selection = head_selection_for_test(engine.clone(), Profile::VllmD3V1).await;
        let expected = super::super::tokenization_work_limit(
            input_limit,
            selection.context.decision_capabilities().max_candidates,
        )
        .unwrap()
        .min(crate::protocols::decision_native::MAX_DECISION_TOKENIZATION_WORK);
        let request = canonical_request(2);
        let prepared =
            super::super::execution::prepare(&request, selection.selection.clone(), input_limit)
                .unwrap();
        let response = super::super::execution::dispatch(
            prepared,
            "budget".into(),
            Context::new(()).context(),
            Arc::new(Semaphore::new(2))
                .acquire_many_owned(2)
                .await
                .unwrap(),
            Arc::new(super::super::Metrics::new()),
            "test-model".into(),
            request,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let calls = engine.calls.lock().unwrap();
        assert_eq!(calls[0].decision_head.max_tokenization_work, expected);
        assert_eq!(calls[0].decision_head.max_input_tokens, input_limit);
    }
}

#[tokio::test]
async fn head_dispatch_preserves_both_public_contracts_and_joint_usage() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        let engine = HeadEngine::new(Behavior::Success);
        let (response, parent, admission) = dispatch_head(engine.clone(), dialect).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-request-id"], "head-request");
        let body = response_body(response).await;
        assert_eq!(body["usage"]["input_tokens"], 13);
        assert_eq!(body["usage"]["output_tokens"], 0);
        if dialect == dynamo_decisions::Dialect::OpenAi {
            assert_eq!(body["answers"][0]["probability"], 0.25);
            assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 2);
        } else {
            assert_eq!(body["answers"]["question-0"]["noul"], 0.25);
        }
        assert!(!parent.is_killed());
        assert_eq!(admission.available_permits(), 2);
        let calls = engine.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].decision_head.questions.len(), 2);
        assert!(!calls[0].decision_head.cache_salt.is_empty());
    }
}

#[tokio::test]
async fn head_native_validation_preserves_dialect_specific_error_status() {
    for (dialect, status) in [
        (dynamo_decisions::Dialect::OpenAi, StatusCode::BAD_REQUEST),
        (
            dynamo_decisions::Dialect::Jev,
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
    ] {
        let (response, parent, admission) =
            dispatch_head(HeadEngine::new(Behavior::InvalidRequest), dialect).await;
        assert_eq!(response.status(), status);
        let body = response_body(response).await;
        assert!(body.get("answers").is_none());
        assert!(!body.to_string().contains("private head details"));
        assert!(parent.is_killed());
        assert_eq!(admission.available_permits(), 2);
    }
}

#[tokio::test]
async fn missing_head_adapter_is_unavailable_without_answers() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        let (response, parent, admission) =
            dispatch_head(HeadEngine::new(Behavior::Unavailable), dialect).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response_body(response).await.get("answers").is_none());
        assert!(parent.is_killed());
        assert_eq!(admission.available_permits(), 2);
    }
}

#[tokio::test]
async fn head_errors_malformed_and_out_of_order_results_are_not_answers() {
    for behavior in [
        Behavior::Malformed,
        Behavior::Reordered,
        Behavior::Empty,
        Behavior::Duplicate,
        Behavior::Error,
    ] {
        let (response, parent, admission) =
            dispatch_head(HeadEngine::new(behavior), dynamo_decisions::Dialect::OpenAi).await;
        assert_eq!(
            response.status(),
            if matches!(behavior, Behavior::Error) {
                super::super::super::error::overload_status_code()
            } else {
                StatusCode::BAD_GATEWAY
            }
        );
        if matches!(behavior, Behavior::Error) {
            assert_eq!(response.headers()["retry-after"], "1");
            assert!(
                response
                    .extensions()
                    .get::<super::super::DecisionCapacityRejection>()
                    .is_some()
            );
        }
        let body = response_body(response).await;
        assert!(body.get("answers").is_none());
        assert!(parent.is_killed());
        assert_eq!(admission.available_permits(), 2);
    }
}

#[tokio::test]
async fn malformed_native_results_have_executor_error_in_both_dialects() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        for behavior in [
            Behavior::Malformed,
            Behavior::Reordered,
            Behavior::Empty,
            Behavior::Duplicate,
        ] {
            let (response, _, _) = dispatch_head(HeadEngine::new(behavior), dialect).await;
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
            let body = response_body(response).await;
            let code = if dialect == dynamo_decisions::Dialect::OpenAi {
                &body["error"]["code"]
            } else {
                &body["code"]
            };
            assert_eq!(code, "invalid_executor_result");
            assert!(body.get("answers").is_none());
        }
    }
}

#[tokio::test]
async fn native_profiles_accept_core_validated_thinking_off_controls() {
    for constructor in [
        Profile::VllmD3V1,
        Profile::VllmClefFlashV1,
        Profile::VllmClefOmniV1,
    ] {
        let engine = HeadEngine::new(Behavior::Success);
        let selection = head_selection_for_test(engine.clone(), constructor).await;
        let mut request = canonical_request(1).as_ref().clone();
        request
            .chat_template_kwargs
            .insert("enable_thinking".into(), false.into());
        assert!(
            super::super::execution::prepare(&request, selection.selection.clone(), 100).is_ok()
        );
        assert!(engine.calls.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn head_cache_measurement_distinguishes_disabled_from_unknown() {
    for (behavior, expected) in [
        (Behavior::CacheDisabled, StatusCode::OK),
        (Behavior::UnknownCache, StatusCode::BAD_GATEWAY),
    ] {
        let engine = HeadEngine::new(behavior);
        let (response, _, _) =
            dispatch_head(engine.clone(), dynamo_decisions::Dialect::OpenAi).await;
        assert_eq!(response.status(), expected);
        let body = response_body(response).await;
        if expected == StatusCode::OK {
            assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 0);
        } else {
            assert!(body.get("answers").is_none());
        }
        assert_eq!(engine.calls.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn versioned_native_profile_supports_both_dialects_without_cache_configuration() {
    let engine = HeadEngine::new(Behavior::Success);
    let selection = head_selection_for_test(engine.clone(), Profile::VllmClefFlashV1).await;
    let mut request = canonical_request(2).as_ref().clone();
    request.dialect = dynamo_decisions::Dialect::OpenAi;
    request
        .validate_capabilities(&selection.context.decision_capabilities())
        .unwrap();
    request.dialect = dynamo_decisions::Dialect::Jev;
    request
        .validate_capabilities(&selection.context.decision_capabilities())
        .unwrap();
    assert!(engine.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn head_deadline_cancels_child_and_releases_question_weighted_admission() {
    let engine = HeadEngine::new(Behavior::Pending);
    let selection = head_selection_for_test(engine.clone(), Profile::VllmD3V1).await;
    let request = canonical_request(2);
    let prepared =
        super::super::execution::prepare(&request, selection.selection.clone(), 100).unwrap();
    let admission = Arc::new(Semaphore::new(2));
    let permit = admission.clone().acquire_many_owned(2).await.unwrap();
    let parent = Context::new(()).context();
    let task = tokio::spawn(super::super::execution::dispatch(
        prepared,
        "deadline".into(),
        parent.clone(),
        permit,
        Arc::new(super::super::Metrics::new()),
        "test-model".into(),
        request,
    ));
    tokio::time::timeout(Duration::from_secs(2), engine.started.notified())
        .await
        .unwrap();
    let response = super::super::response::dispatch_with_deadline(
        task,
        tokio::time::Instant::now() + Duration::from_millis(10),
        parent.clone(),
        dynamo_decisions::Dialect::Jev,
    )
    .await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert!(parent.is_killed());
    assert!(
        engine
            .contexts
            .lock()
            .unwrap()
            .iter()
            .all(|context| context.is_killed())
    );
    let released = tokio::time::timeout(
        Duration::from_secs(2),
        admission.clone().acquire_many_owned(2),
    )
    .await
    .expect("cancelled head dispatch must release admission")
    .unwrap();
    drop(released);
    assert_eq!(admission.available_permits(), 2);
}

#[tokio::test]
async fn head_discovery_needs_no_rust_tokenizer_and_fences_retired_admissions() {
    for constructor in [
        Profile::VllmD3V1,
        Profile::VllmClefFlashV1,
        Profile::VllmClefOmniV1,
    ] {
        let selected =
            head_selection_for_test(HeadEngine::new(Behavior::Success), constructor).await;
        let profile = selected.context.profile();
        assert!(!selected.context.card.has_tokenizer());
        let mut workers = WorkerSet::new(
            "heads".into(),
            selected.context.card.mdcsum().into(),
            selected.context.card.clone(),
        );
        let (sender, receiver) = tokio::sync::watch::channel(vec![7]);
        workers.set_instance_watcher(receiver);
        let _qualification = crate::discovery::test_support::attach_card_surface_for_test(
            &mut workers,
            crate::discovery::test_support::native_resources(selected.engine.clone()),
        )
        .await;
        let model = Model::new("test-model".into());
        model.add_worker_set("heads".into(), Arc::new(workers));
        let selection = model.prepared_selection_for_test().await.unwrap();
        assert!(matches!(selection.engine, DecisionEngine::Head(_)));
        assert_eq!(selection.context.profile(), profile);
        sender.send(vec![]).unwrap();
        assert!(selection.context.validate_dispatch(None).is_err());
    }
}

#[tokio::test]
async fn head_capabilities_fail_closed_without_partitioning_legacy_cards() {
    let base = head_selection_for_test(HeadEngine::new(Behavior::Success), Profile::VllmD3V1)
        .await
        .context
        .card
        .clone();
    let mut absent = base.clone();
    absent.runtime_config.runtime_data.clear();
    let mut disabled = absent.clone();
    disabled
        .runtime_config
        .runtime_data
        .insert("vllm_d3_decision_v1".into(), false.into());
    assert_eq!(absent.mdcsum(), disabled.mdcsum());
    assert_eq!(base.mdcsum(), absent.mdcsum());
    assert_eq!(base.decision_execution_profile(), Some(Profile::VllmD3V1));
    assert!(absent.decision_execution_profile().is_none());
    assert!(disabled.decision_execution_profile().is_none());
    let mut dp = base.clone();
    dp.runtime_config.data_parallel_size = 2;
    assert!(dp.decision_execution_profile().is_none());
    let mut ambiguous = base.clone();
    ambiguous
        .runtime_config
        .runtime_data
        .insert("vllm_clef_flash_joint_decision_v1".into(), true.into());
    assert!(ambiguous.decision_execution_profile().is_none());
}

#[tokio::test]
async fn native_candidate_capacity_is_checked_before_dispatch() {
    let engine = HeadEngine::new(Behavior::Success);
    let mut card = head_selection_for_test(engine.clone(), Profile::VllmD3V1)
        .await
        .context
        .card
        .clone();
    card.runtime_config
        .runtime_data
        .insert("vllm_native_decision_max_candidates".into(), 3.into());
    let mut workers = WorkerSet::new("heads".into(), card.mdcsum().into(), card);
    let (_admissions, admitted_ids) = tokio::sync::watch::channel(vec![1]);
    workers.set_instance_watcher(admitted_ids);
    let _qualification = crate::discovery::test_support::attach_card_surface_for_test(
        &mut workers,
        crate::discovery::test_support::native_resources(DecisionEngine::Head(engine.clone())),
    )
    .await;
    let service = super::super::service_v2::HttpService::builder()
        .build()
        .unwrap();
    let state = service.state_clone();
    assert!(
        state
            .manager()
            .add_worker_set("test-model", "heads", workers)
    );
    let request = dynamo_decisions::parse_request(
        &serde_json::to_vec(&json!({
            "model":"test-model", "input":"Choose a team", "questions":[{
                "type":"choice", "instructions":"Department", "choices":[
                    {"value":"billing"},{"value":"support"},{"value":"sales"},{"value":"other"}
                ]
            }]
        }))
        .unwrap(),
        dynamo_decisions::Route::Decisions,
    )
    .unwrap();
    let response = super::super::handle_request(state, "capacity".into(), request).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(response_body(response).await.get("answers").is_none());
    assert!(engine.calls.lock().unwrap().is_empty());
}
