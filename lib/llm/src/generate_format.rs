// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-native `/generate` formats.
//!
//! A [`GenerateFormat`] owns one `/generate` wire format end to end: it turns
//! a client body into the routed request, joins the worker's responses into
//! client responses, and continues migrated requests. The frontend reads no
//! request or response field.
//!
//! Formats are statically linked plugins. A host builds a registry through its
//! catalog and calls [`install`]; `DYN_GENERATE_FORMAT` selects
//! one by name.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use axum::extract::rejection::{BytesRejection, JsonRejection, MissingJsonContentType};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use dynamo_runtime::error::DynamoError;
use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::migration::RequestContinuation;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

mod error;
pub use error::{ErrorPhase, WorkerError};

/// Runtime-config key under which a worker advertises engine facts its
/// format needs, such as engine-side defaults. Absent on older workers.
pub const ENGINE_INFO_KEY: &str = "generate_engine_info";
/// Names the format the frontend serves.
pub const FORMAT_ENV: &str = "DYN_GENERATE_FORMAT";

/// One client request, as the frontend received it.
pub struct ClientRequest<'a> {
    pub body: &'a [u8],
    pub headers: &'a HeaderMap,
    /// Dynamo's request ID, for requests that carry none.
    pub default_request_id: &'a str,
}

impl ClientRequest<'_> {
    /// Parse a JSON format with the same media-type and data checks as Axum's extractor.
    /// Formats accepting another encoding can read `body` directly.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, JsonRejection> {
        json_content_type(self.headers)?;
        axum::Json::from_bytes(self.body).map(|axum::Json(value)| value)
    }
}

fn json_content_type(headers: &HeaderMap) -> Result<(), JsonRejection> {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<mime::Mime>().ok());
    if !content_type.is_some_and(|mime| {
        mime.type_() == mime::APPLICATION
            && (mime.subtype() == mime::JSON || mime.suffix() == Some(mime::JSON))
    }) {
        return Err(MissingJsonContentType::default().into());
    }
    Ok(())
}

/// Preserve JSON extraction precedence: reject media type before body-read errors.
pub fn json_body_error(headers: &HeaderMap, error: BytesRejection) -> JsonRejection {
    json_content_type(headers)
        .err()
        .unwrap_or_else(|| error.into())
}

/// A request the format rejects; the frontend sends `response` when supplied,
/// or with [`GenerateFormat::error_body`].
#[derive(Debug)]
pub struct ClientError {
    pub status: StatusCode,
    pub message: String,
    pub response: Option<Box<Response>>,
}

impl ClientError {
    /// A 400.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            response: None,
        }
    }
}

impl From<JsonRejection> for ClientError {
    fn from(error: JsonRejection) -> Self {
        // vLLM used Axum's extractor plus the OpenAI 422-to-400 middleware.
        if error.status() == StatusCode::UNPROCESSABLE_ENTITY {
            crate::http::service::metrics::record_failure(
                &DynamoError::builder()
                    .class(dynamo_runtime::error::ErrorClass::InvalidRequest)
                    .build(),
            );
            let message = error.body_text();
            return ClientError {
                status: StatusCode::BAD_REQUEST,
                response: Some(Box::new(
                    (
                        StatusCode::BAD_REQUEST,
                        axum::Json(json!({
                            "message": message, "type": "Bad Request", "code": 400,
                        })),
                    )
                        .into_response(),
                )),
                message,
            };
        }
        ClientError {
            status: error.status(),
            message: error.body_text(),
            response: Some(Box::new(error.into_response())),
        }
    }
}

/// The worker set chosen to serve a request.
pub struct Worker<'a> {
    pub model: &'a str,
    /// What it advertises under [`ENGINE_INFO_KEY`]; `Null` when nothing.
    pub engine_info: &'a Value,
    pub kv_cache_block_size: u32,
    pub lora_name: Option<&'a str>,
    pub tower_connector_lora_enabled: bool,
}

/// A started client request.
pub struct Opened {
    pub request_id: String,
    pub stream: bool,
    /// The model the client names; the frontend's only one when `None`.
    pub model: Option<String>,
    pub input_tokens: usize,
    /// Semantic validation runs after readiness; decoding errors fail in `open`.
    pub validation: Result<(), ClientError>,
    pub bind: Bind,
}

/// Build the request and exchange once the worker set is known. Consumed once.
pub type Bind = Box<dyn FnOnce(&Worker<'_>) -> Result<Bound, ClientError> + Send>;

/// A request ready for dispatch and its migration admission decision.
pub struct Bound {
    pub request: PreprocessedRequest,
    pub migration: Result<(), String>,
    pub exchange: Box<dyn GenerateExchange>,
}

/// One client request across every attempt. Dynamo calls these in stream
/// order and never concurrently; [`Self::migrate`] runs only after every
/// delivered output has passed through [`Self::chunk`].
pub trait GenerateExchange: Send {
    /// One item of the worker stream; `prefill` marks prefill's final output
    /// in P/D. Streaming passes each client chunk to `emit`, which serializes
    /// it before returning; unary buffers.
    fn chunk(
        &mut self,
        output: LLMEngineOutput,
        prefill: bool,
        emit: &mut dyn FnMut(&Value),
    ) -> Result<(), DynamoError>;

    /// Continue after a failed attempt, atomically: rewrite `request` for the
    /// next attempt, whose prompt Dynamo has already extended with the
    /// `delivered_tokens`, or return the output that ends the request; and
    /// rebase this exchange so later outputs continue the earlier ones.
    fn migrate(
        &mut self,
        delivered_tokens: usize,
        request: &mut PreprocessedRequest,
    ) -> Result<Option<LLMEngineOutput>, String>;

    /// Unary: the client response. Streaming: an optional final chunk.
    fn finish(&mut self) -> Result<Option<Value>, DynamoError>;
}

impl RequestContinuation for Mutex<Box<dyn GenerateExchange>> {
    fn next_attempt(
        &self,
        delivered_tokens: usize,
        request: &mut PreprocessedRequest,
    ) -> Result<Option<LLMEngineOutput>, String> {
        self.lock().migrate(delivered_tokens, request)
    }
}

/// One client-facing `/generate` wire format, linked into the frontend.
pub trait GenerateFormat: Send + Sync + 'static {
    fn name(&self) -> &str;

    /// Runtime capability a worker advertises when it serves this format.
    fn capability(&self) -> &'static str;

    /// Frontend: start one client request.
    fn open(&self, request: ClientRequest<'_>) -> Result<Opened, ClientError>;

    /// SGLang accepts POST and PUT; vLLM's native endpoint accepts only POST.
    fn supports_put(&self) -> bool {
        true
    }

    fn ambiguous_model_message(&self) -> &'static str {
        "multiple models are registered; specify `model` in the request"
    }

    /// Adapt shared frontend failures, such as readiness or header validation.
    /// Formats may retain `response` when their legacy API exposed it directly.
    fn frontend_error(&self, mut error: ClientError) -> ClientError {
        error.response = None;
        error
    }

    /// Preserve the format's HTTP body-extraction errors, including body limits.
    fn body_error(&self, error: BytesRejection, _headers: &HeaderMap) -> ClientError {
        ClientError {
            status: error.status(),
            message: error.body_text(),
            response: Some(Box::new(error.into_response())),
        }
    }

    /// Classify a worker error before dispatch completes or from its
    /// response stream. Formats own their existing phase-specific policy.
    fn worker_error(
        &self,
        error: &(dyn std::error::Error + 'static),
        _phase: ErrorPhase,
    ) -> WorkerError {
        WorkerError::dispatch(error, None)
    }

    /// The body of a failed response.
    fn error_body(&self, _status: StatusCode, message: &str) -> Value {
        json!({"error": {"message": message}})
    }
}

pub type SharedFormat = Arc<dyn GenerateFormat>;

/// Formats by name. Starts empty; the host catalog registers its formats.
/// Duplicate names are rejected.
#[derive(Clone, Default)]
pub struct Registry(BTreeMap<String, SharedFormat>);

impl Registry {
    pub fn register(&mut self, format: SharedFormat) -> anyhow::Result<()> {
        let name = format.name().to_string();
        anyhow::ensure!(
            !self.0.contains_key(&name),
            "generate format {name:?} is already registered"
        );
        self.0.insert(name, format);
        Ok(())
    }

    pub fn get(&self, name: &str) -> anyhow::Result<SharedFormat> {
        self.0.get(name).cloned().ok_or_else(|| {
            let linked: Vec<&str> = self.0.keys().map(String::as_str).collect();
            anyhow::anyhow!(
                "unknown generate format {name:?}; linked: {}",
                linked.join(", ")
            )
        })
    }
}

static INSTALLED: OnceLock<Registry> = OnceLock::new();

/// Install a binary's formats before it serves; `false` when already set.
pub fn install(registry: Registry) -> bool {
    INSTALLED.set(registry).is_ok()
}

/// The format [`FORMAT_ENV`] names, by default SGLang's native one.
pub fn selected() -> anyhow::Result<SharedFormat> {
    let name = std::env::var(FORMAT_ENV)
        .ok()
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "sglang".to_string());
    named(&name)
}

/// A linked format by name.
pub fn named(name: &str) -> anyhow::Result<SharedFormat> {
    INSTALLED
        .get()
        .ok_or_else(|| anyhow::anyhow!("generate format catalog is not installed"))?
        .get(name)
}

/// Formats for this crate's tests, which cannot link the real format crates.
#[cfg(test)]
pub(crate) mod test_formats {
    use super::*;
    use crate::local_model::runtime_config::{
        SGLANG_GENERATE_CAPABILITY, VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
    };
    use crate::protocols::common::preprocessor::RoutingHints;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    /// SGLang-shaped: the body minus `input_ids` is the worker envelope, and
    /// each worker response is a client chunk.
    struct Passthrough;

    struct PassthroughExchange;

    impl GenerateExchange for PassthroughExchange {
        fn chunk(
            &mut self,
            output: LLMEngineOutput,
            prefill: bool,
            emit: &mut dyn FnMut(&Value),
        ) -> Result<(), DynamoError> {
            match output
                .engine_data
                .and_then(|mut data| data.as_object_mut()?.remove("sglang_response"))
            {
                Some(response) if !prefill => emit(&response),
                Some(_) => {}
                None if prefill => {}
                None => {
                    return Err(DynamoError::builder()
                        .class(dynamo_runtime::error::ErrorClass::BackendProtocol)
                        .message("missing opaque SGLang response")
                        .build());
                }
            }
            Ok(())
        }

        fn finish(&mut self) -> Result<Option<Value>, DynamoError> {
            Ok(None)
        }
    }

    impl GenerateFormat for Passthrough {
        fn name(&self) -> &str {
            "sglang"
        }

        fn capability(&self) -> &'static str {
            SGLANG_GENERATE_CAPABILITY
        }

        fn open(&self, request: ClientRequest<'_>) -> Result<Opened, ClientError> {
            let mut body: serde_json::Map<String, Value> = request.json()?;
            let input_ids: Vec<u32> =
                serde_json::from_value(body.remove("input_ids").unwrap_or_default())
                    .map_err(|error| ClientError::new(error.to_string()))?;
            let request_id = body
                .get("rid")
                .and_then(Value::as_str)
                .unwrap_or(request.default_request_id)
                .to_string();
            body.insert("rid".to_string(), Value::String(request_id.clone()));
            let max_tokens = body["sampling_params"]["max_new_tokens"]
                .as_u64()
                .map(|tokens| tokens as u32);
            let cache_namespace = body
                .get("cache_salt")
                .and_then(Value::as_str)
                .filter(|salt| !salt.is_empty())
                .map(str::to_owned);
            Ok(Opened {
                request_id,
                stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
                model: None,
                input_tokens: input_ids.len(),
                validation: Ok(()),
                bind: Box::new(move |worker| {
                    let request = PreprocessedRequest::builder()
                        .model(worker.model.to_string())
                        .token_ids(input_ids)
                        .stop_conditions(StopConditions {
                            max_tokens,
                            ..Default::default()
                        })
                        .sampling_options(SamplingOptions::default())
                        .output_options(OutputOptions::default())
                        .routing(Some(RoutingHints {
                            cache_namespace,
                            ..Default::default()
                        }))
                        .extra_args(Some(json!({ "sglang_tito": body })))
                        .annotations(vec![])
                        .build()
                        .map_err(|error| ClientError::new(error.to_string()))?;
                    Ok(Bound {
                        request,
                        exchange: Box::new(PassthroughExchange),
                    })
                }),
            })
        }
    }

    /// Mounts a route; readiness tests never dispatch through it.
    struct Readiness;

    impl GenerateFormat for Readiness {
        fn name(&self) -> &str {
            "vllm"
        }

        fn capability(&self) -> &'static str {
            VLLM_INFERENCE_V1_GENERATE_CAPABILITY
        }

        fn open(&self, _: ClientRequest<'_>) -> Result<Opened, ClientError> {
            unreachable!("route readiness must not parse inference requests")
        }
    }

    pub(crate) fn install() {
        static INSTALL: std::sync::Once = std::sync::Once::new();
        INSTALL.call_once(|| {
            let mut registry = Registry::default();
            registry.register(Arc::new(Passthrough)).unwrap();
            registry.register(Arc::new(Readiness)).unwrap();
            assert!(super::install(registry));
        });
    }
}
