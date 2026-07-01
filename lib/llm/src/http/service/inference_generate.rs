// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP boundary and response assembly for engine-native token generation.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::response::IntoResponse;
use axum::{Json, Router, extract::State, http::HeaderMap, middleware, routing::post};
use dynamo_runtime::pipeline::AsyncEngineContextProvider;
use futures::StreamExt;
use tracing::Instrument;

use super::disconnect::create_connection_monitor;
use super::error::HttpError;
use super::metrics::{CancellationLabels, Endpoint, ErrorType};
use super::openai::{
    ErrorMessage, ErrorResponse, check_model_serving_ready, check_ready, context_from_headers,
    get_body_limit, get_or_create_request_id, smart_json_error_middleware,
};
use super::{RouteDoc, service_v2};
use crate::discovery::ModelManagerError;
use crate::protocols::common::FinishReason;
use crate::protocols::common::extensions::HEADER_DATA_PARALLEL_RANK_ALIAS;
use crate::protocols::common::llm_backend::{LLMEngineOutput, TopLogprob};
use crate::protocols::inference::generate::{
    GENERATE_DP_RANK_CONTEXT_KEY, GENERATE_PATH, GenerateChoiceLogprobs, GenerateProtocolError,
    GenerateRequest, GenerateResponse, GenerateResponseChoice, GenerateTokenLogprob,
    GenerateTopLogprob,
};

pub(super) fn router(state: Arc<service_v2::State>) -> (Vec<RouteDoc>, Router) {
    let doc = RouteDoc::new(axum::http::Method::POST, GENERATE_PATH);
    let router = Router::new()
        .route(GENERATE_PATH, post(handler))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

async fn handler(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(mut request): Json<GenerateRequest>,
) -> Result<axum::response::Response, ErrorResponse> {
    check_ready(&state)?;
    request.validate().map_err(protocol_error)?;
    if request.stream {
        return Err(ErrorMessage::not_implemented_error(
            "Streaming token generation is not enabled in this implementation slice",
        ));
    }

    let model = resolve_model(&state, request.model.as_deref())?;
    check_model_serving_ready(&state, &model)?;
    request.model = Some(model.clone());

    let request_id = resolve_request_id(&headers, request.request_id.as_deref());
    request.request_id = Some(request_id.clone());
    let require_logprobs = request.sampling_params.logprobs.is_some();
    let expected_choices = request.sampling_params.n.unwrap_or(1);

    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: Endpoint::InferenceGenerate.to_string(),
        request_type: "unary".to_string(),
    };
    let mut request = context_from_headers(request, request_id.clone(), &headers)?;
    if let Some(dp_rank) = data_parallel_rank_from_headers(&headers) {
        request.insert(GENERATE_DP_RANK_CONTEXT_KEY, dp_rank);
    }
    let context = request.context();
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(context, Some(state.metrics_clone()), cancellation_labels).await;

    let response = tokio::spawn(
        generate_unary(
            state,
            request,
            request_id,
            model,
            expected_choices,
            require_logprobs,
            stream_handle,
        )
        .in_current_span(),
    )
    .await
    .map_err(|error| {
        ErrorMessage::internal_server_error_with_details(
            "Failed to await token generation task",
            format!("{error:?}"),
        )
    })??;
    connection_handle.disarm();
    Ok(Json(response).into_response())
}

#[allow(clippy::too_many_arguments)]
async fn generate_unary(
    state: Arc<service_v2::State>,
    request: dynamo_runtime::pipeline::Context<GenerateRequest>,
    request_id: String,
    model: String,
    expected_choices: u32,
    require_logprobs: bool,
    _stream_handle: super::disconnect::ConnectionHandle,
) -> Result<GenerateResponse, ErrorResponse> {
    let mut inflight = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::InferenceGenerate,
        false,
        &request_id,
    );
    let engine = state
        .manager()
        .get_generate_engine(&model)
        .map_err(|error| {
            let response = ErrorMessage::from_model_error(&error);
            inflight.mark_error(match error {
                ModelManagerError::ModelUnavailable(_) => ErrorType::Unavailable,
                _ => ErrorType::NotFound,
            });
            response
        })?;
    let mut stream = engine.generate(request).await.map_err(|error| {
        let response = ErrorMessage::from_anyhow(error, "Failed to generate tokens");
        inflight.mark_error(ErrorType::Internal);
        response
    })?;
    let context = stream.context();
    let mut accumulator =
        GenerateAccumulator::new(request_id, expected_choices as usize, require_logprobs);

    while let Some(mut event) = stream.next().await {
        if let Some(error) = event.error.take() {
            inflight.mark_error(ErrorType::Internal);
            return Err(ErrorMessage::from_anyhow(
                anyhow::Error::new(error),
                "Token generation failed",
            ));
        }
        if event.event.as_deref() == Some("error") {
            let details = event
                .comment
                .as_ref()
                .map(|parts| parts.join(", "))
                .unwrap_or_else(|| "backend error".to_string());
            inflight.mark_error(ErrorType::Internal);
            return Err(ErrorMessage::internal_server_error_with_details(
                "Token generation failed",
                details,
            ));
        }
        if let Some(output) = event.data.take() {
            accumulator.push(output).map_err(|error| {
                inflight.mark_error(ErrorType::Internal);
                protocol_error(error)
            })?;
        }
    }

    let response = accumulator.finish().map_err(|error| {
        inflight.mark_error(ErrorType::Internal);
        protocol_error(error)
    })?;
    inflight.mark_ok();
    if context.is_killed() {
        inflight.mark_error(ErrorType::Cancelled);
    }
    Ok(response)
}

fn resolve_request_id(headers: &HeaderMap, body_request_id: Option<&str>) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| body_request_id.map(ToOwned::to_owned))
        .unwrap_or_else(|| get_or_create_request_id(headers))
}

fn data_parallel_rank_from_headers(headers: &HeaderMap) -> Option<u32> {
    headers
        .get(HEADER_DATA_PARALLEL_RANK_ALIAS)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok())
}

fn resolve_model(
    state: &service_v2::State,
    requested: Option<&str>,
) -> Result<String, ErrorResponse> {
    if let Some(model) = requested {
        return Ok(model.to_string());
    }
    let mut models = state.manager().list_generate_models();
    models.sort();
    match models.as_slice() {
        [model] => Ok(model.clone()),
        [] => Err(ErrorMessage::model_not_found()),
        _ => Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: "Validation: `model` is required when multiple models are served".into(),
        })),
    }
}

struct GenerateAccumulator {
    request_id: String,
    choices: BTreeMap<u32, ChoiceAccumulator>,
    prompt_logprobs: Option<
        Vec<
            Option<
                std::collections::HashMap<
                    u32,
                    crate::protocols::inference::generate::GenerateLogprob,
                >,
            >,
        >,
    >,
    kv_transfer_params: Option<serde_json::Value>,
    expected_choices: usize,
    require_logprobs: bool,
}

#[derive(Default)]
struct ChoiceAccumulator {
    token_ids: Vec<u32>,
    logprobs: Vec<GenerateTokenLogprob>,
    finish_reason: Option<String>,
    routed_experts: Option<String>,
}

impl GenerateAccumulator {
    fn new(request_id: String, expected_choices: usize, require_logprobs: bool) -> Self {
        Self {
            request_id,
            choices: BTreeMap::new(),
            prompt_logprobs: None,
            kv_transfer_params: None,
            expected_choices,
            require_logprobs,
        }
    }

    fn push(&mut self, output: LLMEngineOutput) -> Result<(), GenerateProtocolError> {
        let index = output.index.unwrap_or(0);
        let choice = self.choices.entry(index).or_default();
        if choice.finish_reason.is_some() {
            return Err(invalid_response(format!(
                "choice {index} emitted data after its terminal chunk"
            )));
        }

        if self.require_logprobs && !output.token_ids.is_empty() {
            let selected = output.log_probs.as_ref().ok_or_else(|| {
                invalid_response(format!(
                    "choice {index} omitted requested completion logprobs"
                ))
            })?;
            let candidates = output.top_logprobs.as_ref().ok_or_else(|| {
                invalid_response(format!("choice {index} omitted requested top logprobs"))
            })?;
            if selected.len() != output.token_ids.len()
                || candidates.len() != output.token_ids.len()
            {
                return Err(invalid_response(format!(
                    "choice {index} token/logprob counts are not aligned"
                )));
            }
            for ((token_id, logprob), top) in output.token_ids.iter().zip(selected).zip(candidates)
            {
                choice
                    .logprobs
                    .push(build_token_logprob(*token_id, *logprob, top));
            }
        }
        choice.token_ids.extend(output.token_ids);

        if let Some(metadata) = output.generate_metadata {
            if let Some(prompt_logprobs) = metadata.prompt_logprobs {
                if self
                    .prompt_logprobs
                    .as_ref()
                    .is_some_and(|current| current != &prompt_logprobs)
                {
                    return Err(invalid_response(
                        "workers returned inconsistent prompt logprobs",
                    ));
                }
                self.prompt_logprobs = Some(prompt_logprobs);
            }
            if let Some(kv_transfer_params) = metadata.kv_transfer_params {
                self.kv_transfer_params = Some(kv_transfer_params);
            }
            if let Some(routed_experts) = metadata.routed_experts {
                choice.routed_experts = Some(routed_experts);
            }
        }
        if let Some(reason) = output.finish_reason {
            choice.finish_reason = Some(public_finish_reason(reason)?);
        }
        Ok(())
    }

    fn finish(self) -> Result<GenerateResponse, GenerateProtocolError> {
        if self.choices.len() != self.expected_choices {
            return Err(invalid_response(format!(
                "expected {} choices, received {}",
                self.expected_choices,
                self.choices.len()
            )));
        }
        let choices = self
            .choices
            .into_iter()
            .map(|(index, choice)| {
                let finish_reason = choice.finish_reason.ok_or_else(|| {
                    invalid_response(format!("choice {index} has no terminal finish reason"))
                })?;
                if self.require_logprobs && choice.logprobs.len() != choice.token_ids.len() {
                    return Err(invalid_response(format!(
                        "choice {index} accumulated token/logprob counts are not aligned"
                    )));
                }
                Ok(GenerateResponseChoice {
                    index,
                    logprobs: self.require_logprobs.then_some(GenerateChoiceLogprobs {
                        content: Some(choice.logprobs),
                        refusal: None,
                    }),
                    finish_reason: Some(finish_reason),
                    token_ids: choice.token_ids,
                    routed_experts: choice.routed_experts,
                })
            })
            .collect::<Result<Vec<_>, GenerateProtocolError>>()?;
        Ok(GenerateResponse {
            request_id: self.request_id,
            choices,
            prompt_logprobs: self.prompt_logprobs,
            kv_transfer_params: self.kv_transfer_params,
        })
    }
}

fn build_token_logprob(token_id: u32, selected: f64, top: &[TopLogprob]) -> GenerateTokenLogprob {
    let selected_candidate = top.iter().find(|candidate| candidate.token_id == token_id);
    GenerateTokenLogprob {
        token: selected_candidate
            .and_then(|candidate| candidate.token.clone())
            .unwrap_or_else(|| format!("token_id:{token_id}")),
        logprob: selected as f32,
        bytes: selected_candidate.and_then(|candidate| candidate.bytes.clone()),
        top_logprobs: top
            .iter()
            .map(|candidate| GenerateTopLogprob {
                token: candidate
                    .token
                    .clone()
                    .unwrap_or_else(|| format!("token_id:{}", candidate.token_id)),
                logprob: candidate.logprob as f32,
                bytes: candidate.bytes.clone(),
            })
            .collect(),
    }
}

fn public_finish_reason(reason: FinishReason) -> Result<String, GenerateProtocolError> {
    match reason {
        FinishReason::EoS | FinishReason::Stop => Ok("stop".to_string()),
        FinishReason::Length => Ok("length".to_string()),
        FinishReason::ContentFilter => Ok("content_filter".to_string()),
        FinishReason::Cancelled => Ok("stop".to_string()),
        FinishReason::Error(message) => Err(invalid_response(format!(
            "backend generation failed: {message}"
        ))),
    }
}

fn protocol_error(error: GenerateProtocolError) -> ErrorResponse {
    match error {
        GenerateProtocolError::InvalidRequest(message) => {
            ErrorMessage::from_http_error(HttpError {
                code: 400,
                message: format!("Validation: {message}"),
            })
        }
        GenerateProtocolError::InvalidResponse(message) => {
            ErrorMessage::internal_server_error_with_details(
                "Token generation returned an invalid response",
                message,
            )
        }
    }
}

fn invalid_response(message: impl Into<String>) -> GenerateProtocolError {
    GenerateProtocolError::InvalidResponse(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::WorkerSet;
    use crate::model_card::ModelDeploymentCard;
    use crate::protocols::inference::generate::GenerateBackendMetadata;
    use crate::types::Annotated;
    use dynamo_runtime::engine::{AsyncEngine, ResponseStream};
    use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
    use futures::stream;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct CapturedGenerate {
        request: GenerateRequest,
        dp_rank: Option<u32>,
    }

    struct CaptureGenerateEngine {
        captured: Arc<Mutex<Option<CapturedGenerate>>>,
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<GenerateRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for CaptureGenerateEngine
    {
        async fn generate(
            &self,
            request: SingleIn<GenerateRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let dp_rank = request
                .get::<u32>(GENERATE_DP_RANK_CONTEXT_KEY)
                .ok()
                .map(|rank| *rank);
            self.captured.lock().unwrap().replace(CapturedGenerate {
                request: request.content().clone(),
                dp_rank,
            });
            let context = request.context();
            let response = Annotated::from_data(LLMEngineOutput {
                token_ids: vec![99],
                finish_reason: Some(FinishReason::Stop),
                index: Some(0),
                ..Default::default()
            });
            Ok(ResponseStream::new(
                Box::pin(stream::iter([response])),
                context,
            ))
        }
    }

    fn output(index: u32, token_id: u32, finished: bool) -> LLMEngineOutput {
        let top = TopLogprob {
            rank: 1,
            token_id,
            token: Some(format!("token_id:{token_id}")),
            logprob: -0.25,
            bytes: None,
        };
        LLMEngineOutput {
            token_ids: vec![token_id],
            log_probs: Some(vec![-0.25]),
            top_logprobs: Some(vec![vec![top]]),
            finish_reason: finished.then_some(FinishReason::Stop),
            index: Some(index),
            generate_metadata: finished.then_some(GenerateBackendMetadata::default()),
            ..Default::default()
        }
    }

    #[test]
    fn unary_accumulator_preserves_choice_order_and_alignment() {
        let mut accumulator = GenerateAccumulator::new("request-1".into(), 2, true);
        accumulator.push(output(1, 21, false)).unwrap();
        accumulator.push(output(0, 11, true)).unwrap();
        accumulator.push(output(1, 22, true)).unwrap();
        let response = accumulator.finish().unwrap();
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].token_ids, vec![21, 22]);
        assert_eq!(
            response.choices[1]
                .logprobs
                .as_ref()
                .unwrap()
                .content
                .as_ref()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn unary_accumulator_rejects_missing_requested_logprobs() {
        let mut accumulator = GenerateAccumulator::new("request-1".into(), 1, true);
        let mut broken = output(0, 11, true);
        broken.log_probs = None;
        assert!(accumulator.push(broken).is_err());
    }

    #[test]
    fn request_context_matches_vllm_header_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "header-id".parse().unwrap());
        headers.insert(HEADER_DATA_PARALLEL_RANK_ALIAS, "7".parse().unwrap());

        assert_eq!(resolve_request_id(&headers, Some("body-id")), "header-id");
        assert_eq!(data_parallel_rank_from_headers(&headers), Some(7));
    }

    #[test]
    fn malformed_data_parallel_rank_is_ignored_like_vllm() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_DATA_PARALLEL_RANK_ALIAS,
            "not-a-rank".parse().unwrap(),
        );
        assert_eq!(data_parallel_rank_from_headers(&headers), None);
    }

    #[tokio::test]
    async fn unary_http_boundary_preserves_token_native_request_and_private_context() {
        let service = service_v2::HttpService::builder()
            .enable_engine_apis(true)
            .build()
            .unwrap();
        let captured = Arc::new(Mutex::new(None));
        let mut worker_set = WorkerSet::new(
            "test".to_string(),
            "test-mdc".to_string(),
            ModelDeploymentCard::default(),
        );
        worker_set.generate_engine = Some(Arc::new(CaptureGenerateEngine {
            captured: captured.clone(),
        }));
        service
            .model_manager()
            .add_worker_set("test-model", "test", worker_set);

        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "header-request".parse().unwrap());
        headers.insert(HEADER_DATA_PARALLEL_RANK_ALIAS, "3".parse().unwrap());
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "request_id": "body-request",
            "token_ids": [11, 22, 33],
            "sampling_params": {"max_tokens": 4},
            "priority": -2
        }))
        .unwrap();

        let response = handler(State(service.state_clone()), headers, Json(request))
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response: GenerateResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.request_id, "header-request");
        assert_eq!(response.choices[0].token_ids, vec![99]);

        let captured = captured.lock().unwrap();
        let captured = captured.as_ref().unwrap();
        assert_eq!(captured.request.model.as_deref(), Some("test-model"));
        assert_eq!(
            captured.request.request_id.as_deref(),
            Some("header-request")
        );
        assert_eq!(captured.request.token_ids, vec![11, 22, 33]);
        assert_eq!(captured.request.priority, -2);
        assert_eq!(captured.dp_rank, Some(3));
    }
}
