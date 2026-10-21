// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-native `/generate` formats.
//!
//! A [`GenerateFormat`] owns one `/generate` wire format end to end: it turns
//! a client body into the routed request and assembles the worker's responses
//! into unary or streamed client output. The frontend reads no request or
//! response field.
//!
//! Formats are statically linked plugins. A host builds a registry through its
//! catalog and calls [`install`]; `DYN_GENERATE_FORMAT` selects
//! one by name.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use axum::http::{HeaderMap, StatusCode};
use dynamo_runtime::error::DynamoError;
use serde_json::{Value, json};

use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

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

/// A request ready for dispatch and its response exchange.
pub struct Bound {
    pub request: PreprocessedRequest,
    pub exchange: Box<dyn GenerateExchange>,
}

/// One client request. Output callbacks run in stream order, never concurrently.
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

    /// Unary: the client response. Streaming: an optional final chunk.
    fn finish(&mut self) -> Result<Option<Value>, DynamoError>;
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
