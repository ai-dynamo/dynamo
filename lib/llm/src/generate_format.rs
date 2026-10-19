// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-native `/generate` formats.
//!
//! A [`GenerateFormat`] owns one `/generate` wire format end to end: it turns
//! a client body into the routed request, joins the worker's responses into
//! client responses, and continues migrated requests. The frontend reads no
//! request or response field.
//!
//! Formats are statically linked plugins. Dynamo registers its own first; a
//! binary adds a catalog's with [`install`], and `DYN_GENERATE_FORMAT` selects
//! one by name.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use axum::http::{HeaderMap, StatusCode};
use dynamo_runtime::error::DynamoError;
use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::migration::RequestContinuation;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

#[cfg(any(test, feature = "generate-testkit"))]
pub mod testkit;

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

/// A request the format rejects; the frontend answers `status` with `body`,
/// or with [`GenerateFormat::error_body`].
#[derive(Debug)]
pub struct ClientError {
    pub status: StatusCode,
    pub message: String,
    pub body: Option<Value>,
}

impl ClientError {
    /// A 400.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            body: None,
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

    /// The body of a failed response.
    fn error_body(&self, _status: StatusCode, message: &str) -> Value {
        json!({"error": {"message": message}})
    }
}

pub type SharedFormat = Arc<dyn GenerateFormat>;

/// Formats by name. A later registration never replaces an earlier one, so a
/// catalog cannot shadow a built-in format.
#[derive(Clone)]
pub struct Registry(BTreeMap<String, SharedFormat>);

impl Default for Registry {
    /// The formats Dynamo ships.
    fn default() -> Self {
        let sglang: SharedFormat = Arc::new(crate::protocols::sglang::SglangFormat::native());
        let vllm: SharedFormat = Arc::new(crate::protocols::openai::vllm_generate::VllmFormat);
        Self(
            [sglang, vllm]
                .into_iter()
                .map(|f| (f.name().to_string(), f))
                .collect(),
        )
    }
}

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
        .unwrap_or_else(|| crate::protocols::sglang::NAME.to_string());
    named(&name)
}

/// A linked format by name.
pub fn named(name: &str) -> anyhow::Result<SharedFormat> {
    INSTALLED.get_or_init(Registry::default).get(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::sglang::{ClientCodec, Encoded, SglangFormat};

    struct Wrapped;
    impl ClientCodec for Wrapped {
        fn decode_request(&self, body: &[u8]) -> Result<Option<Value>, ClientError> {
            let value: Value = serde_json::from_slice(body)
                .map_err(|error| ClientError::new(error.to_string()))?;
            Ok(Some(value["request"].clone()))
        }
        fn encode_response(&self, response: &Value) -> Encoded {
            Encoded::Replace(json!({"response": response}))
        }
        fn error_body(&self, message: &str) -> Value {
            json!({"failure": message})
        }
    }

    #[test]
    fn catalog_format_owns_client_shape_and_reuses_native_exchange() {
        let mut registry = Registry::default();
        registry
            .register(Arc::new(SglangFormat::with_codec("wrapped", Wrapped)))
            .unwrap();
        assert!(registry.register(Arc::new(SglangFormat::native())).is_err());
        assert!(registry.get("missing").is_err());
        let format = registry.get("wrapped").unwrap();
        assert_eq!(format.capability(), crate::protocols::sglang::CAPABILITY);
        assert_eq!(
            format.error_body(StatusCode::BAD_REQUEST, "bad"),
            json!({"failure": "bad"})
        );
        let opened = format
            .open(ClientRequest {
                body: br#"{"request":{"input_ids":[1],"sampling_params":{"max_new_tokens":2}}}"#,
                headers: &HeaderMap::new(),
                default_request_id: "public",
            })
            .unwrap();
        let mut bound = (opened.bind)(&testkit::worker(&json!({}))).unwrap();
        assert_eq!(bound.request.token_ids.as_slice(), &[1]);
        assert!(bound.migration.is_ok());
        for (ids, finish) in [
            (json!([2]), Value::Null),
            (json!([3]), json!({"type":"length"})),
        ] {
            let output = crate::protocols::sglang::inspect(
                json!({"output_ids": ids, "meta_info": {
                    "id": "private", "finish_reason": finish, "custom_metadata": {"score": 7},
                }}),
                false,
            )
            .unwrap()
            .output;
            bound.exchange.chunk(output, false, &mut |_| {}).unwrap();
        }
        let response = bound.exchange.finish().unwrap().unwrap();
        assert_eq!(response["response"]["output_ids"], json!([2, 3]));
        assert_eq!(response["response"]["meta_info"]["id"], "public");
        assert_eq!(
            response["response"]["meta_info"]["custom_metadata"]["score"],
            7
        );
    }
}
