// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::protocols::systemone::{WORKER_CANONICAL_V1, WorkerDecisionResult};

pub(super) fn prepare(
    request: &CanonicalRequest,
    model: &str,
    max_input_tokens: usize,
) -> Result<PreprocessedRequest, (StatusCode, String)> {
    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(Vec::<u32>::new())
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .extra_args(Some(serde_json::json!({WORKER_CANONICAL_V1: {
            "request": request, "max_input_tokens": max_input_tokens
        }})))
        .build()
        .map_err(|cause| (StatusCode::INTERNAL_SERVER_ERROR, cause.to_string()))
}

pub(super) async fn execute(
    selection: &SystemOneExecutionSelection,
    request: &CanonicalRequest,
    native: PreprocessedRequest,
    request_id: &str,
    parent: Arc<dyn AsyncEngineContext>,
    max_input_tokens: usize,
) -> Result<(Vec<QuestionOutcome>, Usage), DispatchError> {
    selection.validate_dispatch(None).map_err(|_| {
        DispatchError::unavailable("selected worker revision is no longer available")
    })?;
    let context = Context::with_id_and_metadata(native, request_id.to_string(), Default::default());
    let child = context.context();
    parent.link_child(child.clone());
    if parent.is_killed() {
        child.kill();
        return Err(DispatchError::cancelled());
    }
    let mut stream = run_until_killed(parent.as_ref(), selection.engine.generate(context))
        .await
        .ok_or_else(DispatchError::cancelled)?
        .map_err(|cause| backend_error::classify(cause.as_ref()))?;
    let child = stream.context();
    parent.link_child(child.clone());
    if parent.is_killed() {
        child.kill();
        return Err(DispatchError::cancelled());
    }
    let mut terminal = None;
    loop {
        let frame = run_until_killed(parent.as_ref(), stream.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        let Some(frame) = frame else { break };
        let Some(output) = frame.into_data().map_err(stream_error)? else {
            continue;
        };
        match &output.finish_reason {
            Some(FinishReason::Cancelled) => {
                return Err(DispatchError::unavailable(
                    "native scoring worker cancelled",
                ));
            }
            Some(FinishReason::Error(detail)) => {
                return Err(DispatchError::worker_failed(detail.clone()));
            }
            _ => {}
        }
        if terminal.is_some()
            || output.finish_reason != Some(FinishReason::Stop)
            || !output.token_ids.is_empty()
            || output.index.is_some_and(|i| i != 0)
        {
            return Err(DispatchError::malformed(
                "invalid canonical worker terminal frame",
            ));
        }
        let Some(serde_json::Value::Object(mut extra)) = output.extra_args.map(|value| *value)
        else {
            return Err(DispatchError::malformed(
                "missing canonical worker execution result",
            ));
        };
        if extra.len() != 1 {
            return Err(DispatchError::malformed(
                "unexpected canonical worker result fields",
            ));
        }
        terminal = Some(extra.remove(WORKER_CANONICAL_V1).ok_or_else(|| {
            DispatchError::malformed("unsupported canonical worker execution version")
        })?);
    }
    if parent.is_killed() {
        return Err(DispatchError::cancelled());
    }
    let result: WorkerDecisionResult = serde_json::from_value(
        terminal
            .ok_or_else(|| DispatchError::malformed("missing canonical worker terminal frame"))?,
    )
    .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    if result.outcomes.len() != request.questions.len()
        || result.usage.input_tokens > max_input_tokens as u64
        || result
            .usage
            .input_tokens
            .checked_add(result.usage.output_tokens)
            .is_none()
        || result
            .usage
            .cached_tokens
            .is_some_and(|cached| cached > result.usage.input_tokens)
        || result.usage.reasoning_tokens > result.usage.output_tokens
        || (selection.capabilities.measured_cache_reads && result.usage.cached_tokens.is_none())
    {
        return Err(DispatchError::malformed(
            "invalid canonical worker outcome count or usage",
        ));
    }
    let outcomes = result
        .outcomes
        .into_iter()
        .zip(&request.questions)
        .map(|(outcome, question)| {
            if outcome.ordinal != question.ordinal {
                return Err(DispatchError::malformed(
                    "canonical worker outcome order differs from request",
                ));
            }
            if let QuestionOutcome::Answer(distribution) = &outcome.outcome {
                distribution
                    .validate(question.candidates.len())
                    .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
            }
            Ok(outcome.outcome)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((outcomes, result.usage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use dynamo_runtime::{
        engine::AsyncEngine,
        pipeline::{Error, ResponseStream, SingleIn},
    };
    use serde_json::json;
    use std::sync::Mutex;
    use tower::ServiceExt;

    #[derive(Clone, Copy)]
    enum Reply {
        Good,
        Duplicate,
        Missing,
        InvalidDistribution,
        WrongOrdinal,
        InvalidUsage,
        MissingCache,
        ZeroOutput,
        Refusal,
        Cancel,
    }

    struct CanonicalWorker {
        reply: Reply,
        parent: Option<Arc<dyn AsyncEngineContext>>,
        stream_context: Mutex<Option<Arc<dyn AsyncEngineContext>>>,
    }

    impl CanonicalWorker {
        fn new(reply: Reply, parent: Option<Arc<dyn AsyncEngineContext>>) -> Arc<Self> {
            Arc::new(Self {
                reply,
                parent,
                stream_context: Mutex::new(None),
            })
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for CanonicalWorker
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            assert!(request.token_ids.is_empty());
            let envelope = &request.extra_args.as_ref().unwrap()[WORKER_CANONICAL_V1];
            assert!(envelope["max_input_tokens"].as_u64().unwrap() > 0);
            if let Some(parent) = &self.parent {
                let child = Context::new(()).context();
                *self.stream_context.lock().unwrap() = Some(child.clone());
                parent.kill();
                return Ok(ResponseStream::new(
                    Box::pin(futures::stream::pending()),
                    child,
                ));
            }
            let questions = envelope["request"]["questions"].as_array().unwrap();
            let outcomes: Vec<_> = questions.iter().map(|q| json!({
                "ordinal": if matches!(self.reply, Reply::WrongOrdinal) { 999 } else { q["ordinal"].as_u64().unwrap() },
                "outcome": if matches!(self.reply, Reply::Refusal) { json!({"Refusal":"policy"}) } else { json!({"Answer": {
                    "probabilities": if matches!(self.reply, Reply::InvalidDistribution) { vec![0.8,0.3] }
                        else if q["candidates"].as_array().unwrap().len() == 2 { vec![0.8,0.2] }
                        else { vec![1.0/q["candidates"].as_array().unwrap().len() as f64;q["candidates"].as_array().unwrap().len()] },
                    "label_mass": null
                }}) }
            })).collect();
            let output = LLMEngineOutput {
                finish_reason: Some(FinishReason::Stop),
                extra_args: Some(Box::new(json!({WORKER_CANONICAL_V1:{"outcomes":outcomes,
                    "usage":{"input_tokens":10,"output_tokens":if matches!(self.reply, Reply::ZeroOutput) {0} else {7},
                    "cached_tokens":if matches!(self.reply, Reply::MissingCache) { serde_json::Value::Null }
                        else if matches!(self.reply, Reply::InvalidUsage) {json!(11)} else {json!(2)},
                    "reasoning_tokens":if matches!(self.reply, Reply::ZeroOutput) {0} else {3}}}}))),
                ..Default::default()
            };
            let frames = match self.reply {
                Reply::Duplicate => vec![
                    Annotated::from_data(output.clone()),
                    Annotated::from_data(output),
                ],
                Reply::Missing => vec![],
                _ => vec![Annotated::from_data(output)],
            };
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter(frames)),
                request.context(),
            ))
        }
    }

    fn descriptor() -> serde_json::Value {
        json!({"max_questions":8,"max_candidates":64,"question_types":["Predicate","Choice","Score"],
            "measured_cache_reads":true,"prompt_format_version":1,"routing":"round_robin"})
    }

    fn register(
        state: &Arc<service_v2::State>,
        engine: Arc<CanonicalWorker>,
    ) -> tokio::sync::watch::Sender<Vec<u64>> {
        register_with(state, engine, "worker", descriptor())
    }

    fn register_with(
        state: &Arc<service_v2::State>,
        engine: Arc<CanonicalWorker>,
        namespace: &str,
        capabilities: serde_json::Value,
    ) -> tokio::sync::watch::Sender<Vec<u64>> {
        use crate::{
            discovery::WorkerSet,
            model_card::ModelDeploymentCard,
            model_type::{ModelInput, ModelType},
            worker_type::WorkerType,
        };
        let mut card = ModelDeploymentCard::load_from_disk(
            "tests/data/sample-models/mock-llama-3.1-8b-instruct",
            None,
        )
        .unwrap();
        card.set_name("test-model");
        card.model_input = ModelInput::Tokens;
        card.model_type = ModelType::empty();
        card.worker_type = Some(WorkerType::Aggregated);
        card.runtime_config.max_num_seqs = Some(8);
        card.runtime_config.data_parallel_size = 4;
        card.runtime_config
            .runtime_data
            .insert(WORKER_CANONICAL_V1.into(), capabilities);
        assert!(matches!(
            card.decision_profile(),
            Some((DecisionExecution::WorkerCanonicalV1, _))
        ));
        let mut workers = WorkerSet::new(namespace.into(), namespace.into(), card);
        workers.generate_engine = Some(engine);
        let (members, receiver) = tokio::sync::watch::channel(vec![41]);
        workers.set_instance_watcher(receiver);
        assert!(
            state
                .manager()
                .add_worker_set("test-model", namespace, workers)
        );
        state.manager().register_alias("public-model", "test-model");
        members
    }

    #[test]
    fn canonical_worker_readiness_uses_enabled_decision_routes() {
        for enabled in [false, true] {
            let mut api = crate::frontend_config::FrontendApiConfig::default();
            api.set_systemone_enabled(enabled);
            let service = service_v2::HttpService::builder()
                .frontend_api_config(api)
                .build()
                .unwrap();
            let state = service.state_clone();
            let _members = register(&state, CanonicalWorker::new(Reply::Good, None));
            let model = state.manager().get_model("test-model").unwrap();
            assert!(model.has_decision_engine());
            assert!(!model.has_chat_engine());
            assert!(!service.model_endpoint_enabled(crate::endpoint_type::EndpointType::Generate));
            assert_eq!(state.model_http_routes_ready(&model), enabled);
        }
    }

    #[tokio::test]
    async fn both_existing_routes_project_native_results_and_retire_workers() {
        let service = service_v2::HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let registry = prometheus::Registry::new();
        state.metrics_clone().register(&registry).unwrap();
        let members = register(&state, CanonicalWorker::new(Reply::Good, None));
        let (_, app) = super::super::router(state.clone(), None);
        let available = state.systemone_admission().available_permits();
        for (path, body) in [
            (
                "/v1/systemone",
                json!({"model":"public-model","state":"test","questions":{
                    "route":{"type":"choice","instructions":"Choose","criteria":{"a":"A","b":"B"}},
                    "urgent":{"type":"noul","instructions":"Urgent?"},
                    "score":{"type":"score","instructions":"Rate","criteria":["Low","High"]}
                }}),
            ),
            (
                "/v1/decisions",
                json!({"model":"public-model","input":"test","questions":[
                    {"type":"choice","instructions":"Choose","choices":[{"value":"a"},{"value":"b"}]},
                    {"type":"predicate","instructions":"Urgent?"},
                    {"type":"score","instructions":"Rate","levels":[{"label":"Low"},{"label":"High"}]}
                ]}),
            ),
        ] {
            let response = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                app.clone().oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("content-type", "application/json")
                        .header("x-request-id", "native-http-test")
                        .body(Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap(),
                ),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(
                uuid::Uuid::parse_str(response.headers()["x-request-id"].to_str().unwrap()).is_ok()
            );
            assert_eq!(
                response.headers()["x-request-id"],
                response.headers()["x-typesafe-request-id"]
            );
            let body: serde_json::Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                    .unwrap();
            assert_eq!(body["model"], "test-model");
            if path == "/v1/decisions" {
                assert_eq!(body["usage"]["output_tokens"], 7);
                assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 2);
                assert_eq!(
                    body["usage"]["output_tokens_details"]["reasoning_tokens"],
                    3
                );
                assert!((body["answers"][0]["confidence"].as_f64().unwrap() - 0.6).abs() < 1e-9);
                assert_eq!(body["answers"][1]["probability"], 0.8);
                assert!((body["answers"][2]["confidence"].as_f64().unwrap() - 0.6).abs() < 1e-9);
            } else {
                assert_eq!(body["answers"]["urgent"]["noul"], 0.8);
            }
            assert_eq!(state.systemone_admission().available_permits(), available);
        }
        let families = registry.gather();
        for (suffix, expected) in [
            ("questions", 6.0),
            ("input_tokens", 20.0),
            ("candidate_ids", 0.0),
            ("cached_tokens", 4.0),
        ] {
            let family = families
                .iter()
                .find(|family| family.name().ends_with(&format!("_systemone_{suffix}")))
                .unwrap();
            let histogram = family.get_metric()[0].get_histogram();
            assert_eq!(histogram.sample_count(), 2);
            assert_eq!(histogram.sample_sum(), expected);
        }
        let phases = families
            .iter()
            .find(|family| family.name().ends_with("_systemone_phase_seconds"))
            .unwrap();
        let first_branch = phases
            .get_metric()
            .iter()
            .find(|metric| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.name() == "phase" && label.value() == "first_branch")
            })
            .unwrap();
        assert_eq!(first_branch.get_histogram().sample_count(), 2);
        members.send_replace(vec![]);
        assert!(matches!(
            state
                .manager()
                .get_systemone_execution_selection("public-model", |caps| caps.max_candidates >= 2),
            Err(crate::discovery::SystemOneSelectionError::Unavailable)
        ));
    }

    fn canonical_request() -> CanonicalRequest {
        dynamo_decisions::parse_request(
            &serde_json::to_vec(&json!({"model":"public-model","input":"test", "questions":[
                {"type":"predicate","instructions":"Urgent?"}
            ]}))
            .unwrap(),
            dynamo_decisions::Route::Decisions,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn worker_capabilities_are_cohort_local_and_controls_are_forwarded() {
        let service = service_v2::HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let mut narrow = descriptor();
        narrow["max_candidates"] = json!(2);
        let _narrow = register_with(
            &state,
            CanonicalWorker::new(Reply::Good, None),
            "narrow",
            narrow,
        );
        let wide = register_with(
            &state,
            CanonicalWorker::new(Reply::Good, None),
            "wide",
            descriptor(),
        );
        let selection = state
            .manager()
            .get_systemone_execution_selection("public-model", |caps| caps.max_candidates >= 40)
            .unwrap();
        assert_eq!(selection.execution, DecisionExecution::WorkerCanonicalV1);
        assert_eq!(selection.capabilities.max_candidates, 64);
        assert!(selection.preprocessor.is_none());
        let mut request = dynamo_decisions::parse_request(&serde_json::to_vec(&json!({
            "model":"public-model","input":"test","questions":[{"type":"choice","instructions":"Choose",
            "choices":(0..40).map(|i| json!({"value":i.to_string()})).collect::<Vec<_>>()}]
        })).unwrap(),dynamo_decisions::Route::Decisions).unwrap();
        request.temperature = 2.0;
        request
            .chat_template_kwargs
            .insert("worker_policy".into(), json!(true));
        let prepared = prepare(&request, "test-model", 100).unwrap();
        assert_eq!(
            prepared.extra_args.as_ref().unwrap()[WORKER_CANONICAL_V1]["request"]["temperature"],
            2.0
        );
        assert_eq!(
            prepared.extra_args.as_ref().unwrap()[WORKER_CANONICAL_V1]["request"]["chat_template_kwargs"]
                ["worker_policy"],
            true
        );
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            super::super::handle_request(state.clone(), "generic-policy".into(), request),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(
            body["answers"][0]["probabilities"]
                .as_array()
                .unwrap()
                .len(),
            40
        );
        wide.send_replace(vec![]);
        assert!(matches!(
            state
                .manager()
                .get_systemone_execution_selection("public-model", |caps| caps.max_candidates
                    >= 40),
            Err(SystemOneSelectionError::Unavailable)
        ));
    }

    #[tokio::test]
    async fn generic_usage_and_refusals_preserve_public_contracts() {
        for (reply, path, expected) in [
            (Reply::ZeroOutput, "/v1/decisions", StatusCode::OK),
            (Reply::Refusal, "/v1/decisions", StatusCode::OK),
            (
                Reply::Refusal,
                "/v1/systemone",
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (Reply::MissingCache, "/v1/systemone", StatusCode::OK),
            (
                Reply::MissingCache,
                "/v1/decisions",
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let service = service_v2::HttpService::builder().build().unwrap();
            let state = service.state_clone();
            let mut capabilities = descriptor();
            if matches!(reply, Reply::MissingCache) {
                capabilities["measured_cache_reads"] = json!(false);
            }
            let _members = register_with(
                &state,
                CanonicalWorker::new(reply, None),
                "worker",
                capabilities,
            );
            let (_, app) = super::super::router(state.clone(), None);
            let body = if path == "/v1/decisions" {
                json!({"model":"public-model","input":"test","questions":[{"type":"predicate","instructions":"Question?"}]})
            } else {
                json!({"model":"public-model","state":"test","questions":{"q":{"type":"noul","instructions":"Question?"}}})
            };
            let response = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                app.oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("content-type", "application/json")
                        .body(Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap(),
                ),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(response.status(), expected);
            let body: serde_json::Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                    .unwrap();
            if matches!(reply, Reply::ZeroOutput) {
                assert_eq!(body["usage"]["output_tokens"], 0);
            }
            if matches!(reply, Reply::Refusal) && path == "/v1/decisions" {
                assert_eq!(body["answers"][0]["type"], "refusal");
            }
        }
    }

    #[tokio::test]
    async fn cohort_selection_checks_question_types_and_question_limits() {
        let mut predicate_only = descriptor();
        predicate_only["question_types"] = json!(["Predicate"]);
        let mut single_question = descriptor();
        single_question["max_questions"] = json!(1);
        for limited in [predicate_only, single_question] {
            let service = service_v2::HttpService::builder().build().unwrap();
            let state = service.state_clone();
            let _limited = register_with(
                &state,
                CanonicalWorker::new(Reply::Good, None),
                "limited",
                limited,
            );
            let capable = register_with(
                &state,
                CanonicalWorker::new(Reply::Good, None),
                "capable",
                descriptor(),
            );
            let question = json!({"type":"choice","instructions":"Choose", "choices":[{"value":"a"},{"value":"b"}]});
            let request = dynamo_decisions::parse_request(
                &serde_json::to_vec(&json!({"model":"public-model","input":"test", "questions":[question.clone(),question]})).unwrap(),
                dynamo_decisions::Route::Decisions,
            ).unwrap();
            for _ in 0..16 {
                let selection = state
                    .manager()
                    .get_systemone_execution_selection(&request.model, |caps| {
                        request.validate_capabilities(caps).is_ok()
                    })
                    .unwrap();
                assert!(selection.capabilities.supports_choice);
                assert_eq!(selection.capabilities.max_questions, 8);
            }
            capable.send_replace(vec![]);
            assert!(matches!(
                state
                    .manager()
                    .get_systemone_execution_selection(&request.model, |caps| request
                        .validate_capabilities(caps)
                        .is_ok(),),
                Err(SystemOneSelectionError::Unavailable)
            ));
        }
    }

    #[tokio::test]
    async fn malformed_results_are_rejected_after_terminal_collection() {
        for reply in [
            Reply::Duplicate,
            Reply::Missing,
            Reply::InvalidDistribution,
            Reply::WrongOrdinal,
            Reply::InvalidUsage,
            Reply::MissingCache,
        ] {
            let service = service_v2::HttpService::builder().build().unwrap();
            let state = service.state_clone();
            let _members = register(&state, CanonicalWorker::new(reply, None));
            let selection = state
                .manager()
                .get_systemone_execution_selection("public-model", |caps| caps.max_candidates >= 2)
                .unwrap();
            let request = canonical_request();
            let result = execute(
                &selection,
                &request,
                prepare(&request, "test-model", 100).unwrap(),
                "test",
                Context::new(()).context(),
                100,
            )
            .await;
            assert_eq!(
                result.err().unwrap().status,
                StatusCode::INTERNAL_SERVER_ERROR
            );
        }
    }

    #[tokio::test]
    async fn cancellation_is_replayed_to_a_late_linked_stream() {
        let parent = Context::new(()).context();
        let engine = CanonicalWorker::new(Reply::Cancel, Some(parent.clone()));
        let service = service_v2::HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let _members = register(&state, engine.clone());
        let selection = state
            .manager()
            .get_systemone_execution_selection("public-model", |caps| caps.max_candidates >= 2)
            .unwrap();
        let request = canonical_request();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            execute(
                &selection,
                &request,
                prepare(&request, "test-model", 100).unwrap(),
                "test",
                parent,
                100,
            ),
        )
        .await
        .unwrap();
        assert_eq!(result.err().unwrap().status.as_u16(), 499);
        assert!(
            engine
                .stream_context
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .is_killed()
        );
    }
}
