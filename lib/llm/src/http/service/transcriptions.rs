// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Multipart, State, multipart::MultipartRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use base64::Engine as _;
use futures::StreamExt;

use super::{
    RouteDoc,
    metrics::{Endpoint, ErrorType, process_response_and_observe_metrics, request_was_rejected},
    openai::{
        ErrorMessage, ErrorResponse, check_model_serving_ready, check_ready, context_from_headers,
        error_is_invalid_argument, extract_error_type_from_response, get_body_limit,
        get_or_create_request_id,
    },
    service_v2,
};
use crate::protocols::openai::transcriptions::{
    NvAudioTranscriptionResponse, NvCreateAudioTranscriptionRequest,
};

#[derive(Default)]
struct TranscriptionForm {
    audio_b64: Option<String>,
    filename: Option<String>,
    model: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
    temperature: Option<f64>,
    timestamp_granularities: Vec<String>,
}

impl TranscriptionForm {
    async fn parse(mut multipart: Multipart) -> Result<Self, ErrorResponse> {
        let mut form = Self::default();
        while let Some(field) = multipart.next_field().await.map_err(|error| {
            ErrorMessage::bad_request(format!("Invalid multipart body: {error}"))
        })? {
            let name = field.name().unwrap_or_default().to_string();
            if name == "file" {
                form.filename = Some(field.file_name().unwrap_or("audio").to_string());
                let bytes = field.bytes().await.map_err(|error| {
                    ErrorMessage::bad_request(format!("Failed to read audio file: {error}"))
                })?;
                if bytes.is_empty() {
                    return Err(ErrorMessage::bad_request(
                        "The audio file must not be empty",
                    ));
                }
                form.audio_b64 = Some(base64::engine::general_purpose::STANDARD.encode(bytes));
                continue;
            }

            let value = field.text().await.map_err(|error| {
                ErrorMessage::bad_request(format!("Failed to read form field `{name}`: {error}"))
            })?;
            match name.as_str() {
                "model" => form.model = nonempty(value),
                "language" => form.language = nonempty(value),
                "prompt" => form.prompt = Some(value),
                "response_format" => form.response_format = nonempty(value),
                "temperature" => form.temperature = Some(parse_field(&name, &value)?),
                "timestamp_granularities" | "timestamp_granularities[]" => {
                    if !value.is_empty() {
                        form.timestamp_granularities.push(value);
                    }
                }
                "stream" if value.eq_ignore_ascii_case("true") || value == "1" => {
                    return Err(ErrorMessage::bad_request(
                        "Streaming transcriptions are not supported yet",
                    ));
                }
                _ => {}
            }
        }
        Ok(form)
    }

    fn into_request(self) -> Result<NvCreateAudioTranscriptionRequest, ErrorResponse> {
        let audio_b64 = self
            .audio_b64
            .ok_or_else(|| ErrorMessage::bad_request("Missing required `file` field"))?;
        let response_format = self.response_format.as_deref().unwrap_or("json");
        if !matches!(response_format, "json" | "verbose_json") {
            return Err(ErrorMessage::bad_request(format!(
                "Unsupported response_format `{response_format}`; expected `json` or `verbose_json`"
            )));
        }
        if self
            .timestamp_granularities
            .iter()
            .any(|value| !matches!(value.as_str(), "word" | "segment"))
        {
            return Err(ErrorMessage::bad_request(
                "timestamp_granularities must contain only `word` or `segment`",
            ));
        }

        Ok(NvCreateAudioTranscriptionRequest {
            audio_b64,
            filename: self.filename.unwrap_or_else(|| "audio".to_string()),
            model: self.model,
            language: self.language,
            prompt: self.prompt,
            response_format: Some(response_format.to_string()),
            temperature: self.temperature,
            timestamp_granularities: (!self.timestamp_granularities.is_empty())
                .then_some(self.timestamp_granularities),
        })
    }
}

fn nonempty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn parse_field<T>(name: &str, value: &str) -> Result<T, ErrorResponse>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value.parse().map_err(|error| {
        ErrorMessage::bad_request(format!("Invalid `{name}` value `{value}`: {error}"))
    })
}

async fn audio_transcription(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, ErrorResponse> {
    check_ready(&state)?;
    let multipart = multipart
        .map_err(|error| ErrorMessage::bad_request(format!("Invalid multipart body: {error}")))?;
    let mut request = TranscriptionForm::parse(multipart).await?.into_request()?;

    let model = match request.model.clone() {
        Some(model) => model,
        None => state
            .manager()
            .list_transcriptions_models()
            .into_iter()
            .find(|model| state.manager().is_model_ready_to_serve(model))
            .unwrap_or_default(),
    };
    request.model = Some(model.clone());
    check_model_serving_ready(&state, &model)?;

    let metric_model = state.manager().metric_model_for(&model).to_string();
    let request_id = get_or_create_request_id(&headers);
    let request = context_from_headers(request, request_id, &headers)?;
    let request_id = request.id().to_string();
    let mut inflight = state.metrics_clone().create_inflight_guard(
        &metric_model,
        Endpoint::Transcriptions,
        false,
        &request_id,
    );
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);
    let engine = state
        .manager()
        .get_transcriptions_engine(&model)
        .map_err(|error| {
            let response = ErrorMessage::from_model_error(&error);
            inflight.mark_error(extract_error_type_from_response(&response));
            response
        })?;
    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metric_model);

    let stream = engine.generate(request).await.map_err(|error| {
        let is_validation = error_is_invalid_argument(error.as_ref());
        if request_was_rejected(error.as_ref()) {
            state
                .metrics_clone()
                .inc_rejection(&metric_model, Endpoint::Transcriptions);
        }
        let response = ErrorMessage::from_anyhow(error, "Failed to transcribe audio");
        inflight.mark_error(if is_validation {
            ErrorType::Validation
        } else {
            extract_error_type_from_response(&response)
        });
        response
    })?;
    let mut http_queue_guard = Some(http_queue_guard);
    let stream = stream.inspect(move |response| {
        process_response_and_observe_metrics(
            response,
            &mut response_collector,
            &mut http_queue_guard,
        );
    });
    let response = NvAudioTranscriptionResponse::from_annotated_stream(stream)
        .await
        .map_err(|error| {
            tracing::error!(request_id, %error, "Failed to fold transcription response");
            let is_validation = error_is_invalid_argument(&error);
            let response = ErrorMessage::from_anyhow(
                anyhow::Error::new(error),
                "Failed to fold transcription response",
            );
            inflight.mark_error(if is_validation {
                ErrorType::Validation
            } else {
                extract_error_type_from_response(&response)
            });
            response
        })?;

    inflight.mark_ok();
    Ok((StatusCode::OK, Json(response)).into_response())
}

pub fn transcriptions_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/v1/audio/transcriptions".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(audio_transcription))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_json_response_formats() {
        let form = TranscriptionForm {
            audio_b64: Some("AQ==".to_string()),
            response_format: Some("srt".to_string()),
            ..Default::default()
        };
        assert_eq!(form.into_request().unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn builds_transport_neutral_request() {
        let form = TranscriptionForm {
            audio_b64: Some("YXVkaW8=".to_string()),
            filename: Some("sample.wav".to_string()),
            language: Some("en".to_string()),
            timestamp_granularities: vec!["word".to_string()],
            ..Default::default()
        };
        let request = form.into_request().unwrap();
        assert_eq!(request.filename, "sample.wav");
        assert_eq!(request.audio_b64, "YXVkaW8=");
        assert_eq!(request.response_format.as_deref(), Some("json"));
        assert_eq!(
            request.timestamp_granularities,
            Some(vec!["word".to_string()])
        );
    }
}
