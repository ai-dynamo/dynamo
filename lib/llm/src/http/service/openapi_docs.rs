// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAPI documentation generation and Swagger UI integration
//!
//! This module provides automatic OpenAPI specification generation from the HTTP service routes
//! and serves Swagger UI for interactive API documentation.
//!
//! ## Features
//!
//! - **OpenAPI Specification**: Automatically generates OpenAPI spec from defined routes
//! - **Swagger UI**: Interactive API documentation accessible via web browser
//! - **Dynamic Route Documentation**: Introspects registered routes and generates documentation
//!
//! ## Endpoints
//!
//! The module exposes two main endpoints:
//!
//! - `GET /openapi.json` - Returns the OpenAPI specification in JSON format
//! - `GET /docs` - Serves the Swagger UI interface for interactive API exploration
//!
//! ## Configuration
//!
//! The OpenAPI documentation endpoints use fixed paths:
//! - `/openapi.json` - The OpenAPI specification
//! - `/docs` - The Swagger UI documentation interface
//!
//! ## Example Usage
//!
//! The OpenAPI documentation is automatically integrated into the HTTP service.
//! Once the service is running, you can:
//!
//! 1. View the raw OpenAPI spec: `curl http://localhost:8000/openapi.json`
//! 2. Access Swagger UI: Open `http://localhost:8000/docs` in a web browser

use std::sync::Arc;

use axum::{Router, body::Bytes, http::StatusCode, response::IntoResponse};
use tokio::sync::OnceCell;
use utoipa::openapi::{PathItem, Paths, RefOr};
use utoipa::{OpenApi, ToSchema};

use crate::http::service::{OpenApiRouteKind, RouteDoc};
use crate::protocols::openai::{
    chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
        NvCreateChatCompletionStreamResponse,
    },
    completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    embeddings::NvCreateEmbeddingRequest,
    responses::NvCreateResponse,
};

/// Response serialization options that must match the service being documented.
/// Defaults match the frontend: canonical reasoning key and no named dispatches.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenApiOptions {
    pub reasoning_field: crate::reasoning_field::ReasoningField,
    pub streaming_tool_dispatch: bool,
    pub streaming_reasoning_dispatch: bool,
}

/// OpenAPI documentation structure
///
/// Registers native schemas; dependency import slots are resolved separately.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "NVIDIA Dynamo OpenAI Frontend",
        version = env!("CARGO_PKG_VERSION"),
        description = "OpenAI-compatible HTTP API for NVIDIA Dynamo.",
        license(name = "Apache-2.0"),
        contact(name = "NVIDIA Dynamo", url = "https://github.com/ai-dynamo/dynamo")
    ),
    servers(
        (url = "/", description = "Current server")
    ),
    components(
        schemas(
            crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest,
            crate::protocols::openai::chat_completions::NvCreateChatCompletionResponse,
            crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
            crate::protocols::openai::completions::NvCreateCompletionRequest,
            crate::protocols::openai::completions::NvCreateCompletionResponse,
            crate::protocols::openai::embeddings::NvCreateEmbeddingRequest,
            crate::protocols::openai::responses::NvCreateResponse,
            super::openai::ErrorMessage,
            super::openai::ToolCallDispatchPayload<'static>,
            super::openai::ReasoningDispatchPayload<'static>
        )
    )
)]
struct ApiDoc;

/// Generate the native document with the default `reasoning_content` response key.
///
/// Use when no response-key override is needed. The `route_docs` contract, returned
/// document, and errors are described by [`generate_openapi_spec_with_options`].
pub fn generate_openapi_spec(route_docs: &[RouteDoc]) -> Result<serde_json::Value, String> {
    generate_openapi_spec_with_options(route_docs, OpenApiOptions::default())
}

/// Generate a native OpenAPI document for export or embedding without starting HTTP.
///
/// # Arguments
///
/// * `route_docs` - Registered routes to describe, including their handler kinds.
///   Unsupported methods are skipped with a warning; methods on one path combine.
/// * `options` - Response property spelling and dispatch flags; match the service's
///   configuration, or explicitly enable both dispatches for a full-surface export.
///
/// # Returns
///
/// An owned OpenAPI 3.2 JSON document with native components, request-alias metadata,
/// and SSE event schemas. Dependency import slots remain unresolved; this is not a
/// fully composed contract. Return JSON rather than Utoipa 5's typed document because
/// its model cannot retain OpenAPI 3.2 fields. Do not deserialize this into that model.
///
/// # Errors
///
/// Returns a diagnostic if schema invariants or serialization fail. No partial
/// document is returned. HTTP callers must isolate this failure to documentation.
pub fn generate_openapi_spec_with_options(
    route_docs: &[RouteDoc],
    options: OpenApiOptions,
) -> Result<serde_json::Value, String> {
    let mut openapi = ApiDoc::openapi();
    // Like input aliases below, response projection is a compiled-schema invariant.
    configure_response_schemas(&mut openapi, options.reasoning_field)?;

    // Build paths from route documentation
    let mut paths = Paths::new();

    for route in route_docs {
        tracing::debug!("Adding route to OpenAPI spec: {route}");
        let method = route.method().as_str();
        let path = route.path();

        // Add operation based on method
        let operation = create_operation_for_route(route);

        // Create PathItem with the operation
        use utoipa::openapi::HttpMethod;
        let path_item = match method.to_uppercase().as_str() {
            "GET" => PathItem::new(HttpMethod::Get, operation),
            "POST" => PathItem::new(HttpMethod::Post, operation),
            "PUT" => PathItem::new(HttpMethod::Put, operation),
            "DELETE" => PathItem::new(HttpMethod::Delete, operation),
            "PATCH" => PathItem::new(HttpMethod::Patch, operation),
            "HEAD" => PathItem::new(HttpMethod::Head, operation),
            "OPTIONS" => PathItem::new(HttpMethod::Options, operation),
            _ => {
                tracing::warn!("Unknown HTTP method: {}", method);
                continue;
            }
        };

        // Merge into an existing PathItem so multiple methods on one path (the
        // built-in GET+POST /busy_threshold, or an extension GET on a built-in
        // path) all appear, instead of the last method overwriting the rest.
        match paths.paths.get_mut(path) {
            Some(existing) => existing.merge_operations(path_item),
            None => {
                paths.paths.insert(path.to_string(), path_item);
            }
        }
    }

    openapi.paths = paths;
    // Serde input aliases are not represented by ToSchema. Publish explicit
    // metadata for request-contract consumers, without changing deserialization
    // or claiming that aliases are response spellings. The fidelity tests guard
    // these declarations against the actual request deserializer. By definition,
    // x-dynamo-input-aliases rejects multiple spellings of one field together.
    // This is a compiled-schema invariant, not a check of a remote deployment.
    // Reject the document rather than publish incomplete alias metadata.
    // `http_export_describes_request_aliases` exercises this path in the
    // pre-merge Rust suite so schema changes must update the declarations too.
    annotate_request_aliases(&mut openapi)?;
    // Keep the dependency's Utoipa 5 derives, then cross the version boundary once.
    // A round trip back through its typed model would discard itemSchema and cannot
    // represent the 3.2 version. HTTP and file export share this final representation.
    let mut document = serde_json::to_value(openapi).map_err(|error| error.to_string())?;
    document["openapi"] = serde_json::json!("3.2.0");
    for route in route_docs {
        let Some((_, chunk, chat)) = completion_response_types(route) else {
            continue;
        };
        // JSON Pointer escaping preserves custom paths, including literal '~'.
        // Non-inserting lookup must never manufacture an unregistered operation.
        let path = route.path().replace('~', "~0").replace('/', "~1");
        let pointer = format!("/paths/{path}/post/responses/200/content/text~1event-stream");
        let media = document
            .pointer_mut(&pointer)
            .ok_or_else(|| format!("missing SSE response for {route}"))?;
        media["itemSchema"] = streaming_event_schema(&chunk, chat, options);
    }
    Ok(document)
}

/// Describe parsed data-bearing SSE events for a registered completion response.
///
/// `chunk` names a registered payload component; `chat` permits the named tool
/// and reasoning events enabled in `options`. The returned schema
/// constrains event envelopes and string-encoded JSON, not ordering or termination.
/// Consumers must explicitly validate `contentSchema`. Comment-only annotations and
/// keep-alives have no dispatched data event and are omitted after SSE parsing.
fn streaming_event_schema(chunk: &str, chat: bool, options: OpenApiOptions) -> serde_json::Value {
    use serde_json::json;

    // disconnect.rs emits this error envelope before [DONE]; the shared error type
    // also permits optional details used by unary errors, absent from stream errors.
    let mut events = vec![json!({
        "type": "object",
        "required": ["data"],
        "properties": {
            "event": {"type": "string", "const": "message"},
            "data": {
                "type": "string",
                "anyOf": [
                    {"const": "[DONE]"},
                    {"contentMediaType": "application/json", "contentSchema": {
                        "$ref": format!("#/components/schemas/{chunk}")
                    }},
                    {"contentMediaType": "application/json", "contentSchema": {
                        "type": "object",
                        "required": ["error"],
                        "properties": {"error": {"$ref": format!("#/components/schemas/{}", super::openai::ErrorMessage::name())}}
                    }}
                ]
            }
        }
    })];
    if chat {
        for (enabled, event, payload) in [
            (
                options.streaming_tool_dispatch,
                "tool_call_dispatch",
                super::openai::ToolCallDispatchPayload::name(),
            ),
            (
                options.streaming_reasoning_dispatch,
                "reasoning_dispatch",
                super::openai::ReasoningDispatchPayload::name(),
            ),
        ] {
            if !enabled {
                continue;
            }
            events.push(json!({
                "type": "object",
                "required": ["event", "data"],
                "properties": {
                    "event": {"type": "string", "const": event},
                    "data": {
                        "type": "string", "contentMediaType": "application/json",
                        "contentSchema": {"$ref": format!("#/components/schemas/{payload}")}
                    }
                }
            }));
        }
    }
    json!({"anyOf": events})
}

/// Add the Serde input-alias metadata that schema derivation does not supply.
///
/// `document` must contain the registered chat request and assistant-message schemas.
/// Mutates their field metadata in place; `Ok(())` means each declared alias was
/// attached to exactly one canonical field.
///
/// # Errors
///
/// Returns a diagnostic for missing components, missing or ambiguous fields, alias
/// collisions, or unsupported field schemas. Earlier annotations are not rolled back.
fn annotate_request_aliases(document: &mut utoipa::openapi::OpenApi) -> Result<(), String> {
    use utoipa::openapi::schema::Schema;

    /// Attach `alias` as an accepted input spelling for canonical `field` in `node`.
    ///
    /// Searches inline object/allOf fields, not references or arbitrary descendants;
    /// returns the number annotated, or zero if none matched. Mutates metadata in
    /// place. Collisions or unsupported field schemas return a diagnostic without
    /// rolling back annotations already applied.
    fn annotate(node: &mut RefOr<Schema>, field: &str, alias: &str) -> Result<usize, String> {
        match node {
            RefOr::T(Schema::Object(object)) => {
                if object.properties.contains_key(alias) {
                    return Err(format!("alias {alias} collides with an existing property"));
                }
                let Some(property) = object.properties.get_mut(field) else {
                    return Ok(0);
                };
                let extensions = match property {
                    RefOr::T(Schema::Object(schema)) => &mut schema.extensions,
                    RefOr::T(Schema::AnyOf(schema)) => &mut schema.extensions,
                    RefOr::T(Schema::OneOf(schema)) => &mut schema.extensions,
                    RefOr::T(Schema::AllOf(schema)) => &mut schema.extensions,
                    RefOr::T(Schema::Array(schema)) => &mut schema.extensions,
                    _ => return Err(format!("unsupported alias field schema for {field}")),
                };
                let extensions = extensions.get_or_insert_with(Default::default);
                extensions.insert("x-dynamo-input-aliases".into(), serde_json::json!([alias]));
                Ok(1)
            }
            // Search only this exact object's flattened fields, not arbitrary
            // descendants. Keep the typed schema intact: a whole-document JSON
            // round trip is not lossless for all current dependency schemas.
            RefOr::T(Schema::AllOf(schema)) => {
                schema.items.iter_mut().try_fold(0, |count, item| {
                    annotate(item, field, alias).map(|found| count + found)
                })
            }
            _ => Ok(0),
        }
    }

    let components = document.components.as_mut().ok_or("missing components")?;
    for (name, field, alias) in [
        (
            NvCreateChatCompletionRequest::name().as_ref(),
            "chat_template_args",
            "chat_template_kwargs",
        ),
        (
            "dynamo_protocols.chat.ChatCompletionRequestAssistantMessage",
            "reasoning_content",
            "reasoning",
        ),
    ] {
        let node = components
            .schemas
            .get_mut(name)
            .ok_or_else(|| format!("missing {name}"))?;
        if annotate(node, field, alias)? != 1 {
            return Err(format!("expected exactly one {field} at {name}"));
        }
    }
    Ok(())
}

/// Align canonical response schemas with the frontend's configured reasoning key.
///
/// # Arguments
///
/// * `openapi` - Document to mutate. Its unary message and stream-delta components
///   must be objects containing `reasoning_content` and no `reasoning` property.
/// * `reasoning_field` - Client-visible property spelling to use in both components.
///
/// # Returns
///
/// `Ok(())` when both property names and any matching required entries are updated.
/// Request schemas are unchanged. Apply this to canonical schemas, not an already
/// renamed document.
///
/// # Errors
///
/// Returns a diagnostic for missing components or violated input requirements.
/// Earlier component changes are not rolled back.
fn configure_response_schemas(
    openapi: &mut utoipa::openapi::OpenApi,
    reasoning_field: crate::reasoning_field::ReasoningField,
) -> Result<(), String> {
    use utoipa::openapi::schema::Schema;

    let components = openapi.components.as_mut().ok_or("missing components")?;
    for name in [
        "ChatCompletionResponseMessage",
        "ChatCompletionStreamResponseDelta",
    ] {
        let name = format!("dynamo_protocols.chat.{name}");
        let schema = components
            .schemas
            .get_mut(&name)
            .ok_or_else(|| format!("missing {name}"))?;
        let RefOr::T(Schema::Object(object)) = schema else {
            return Err(format!("expected object schema at {name}"));
        };
        if object.properties.contains_key("reasoning") {
            return Err(format!(
                "reasoning collides with an existing property at {name}"
            ));
        }
        let property = object
            .properties
            .remove("reasoning_content")
            .ok_or_else(|| format!("missing reasoning_content at {name}"))?;
        object
            .properties
            .insert(reasoning_field.as_str().to_owned(), property);
        for required in &mut object.required {
            if required == "reasoning_content" {
                *required = reasoning_field.as_str().to_owned();
            }
        }
    }
    Ok(())
}

/// Create an OpenAPI operation for a specific route
/// Uses the handler kind for schemas and prose; the registered URL remains the operation ID.
fn create_operation_for_route(route: &RouteDoc) -> utoipa::openapi::path::Operation {
    use utoipa::openapi::ResponseBuilder;
    use utoipa::openapi::path::OperationBuilder;

    let method = route.method().as_str();
    let path = route.path();
    let operation_id = format!(
        "{}_{}",
        method.to_lowercase(),
        path.replace('/', "_").trim_matches('_')
    );
    // Select the built-in prose by handler identity, even at a remapped URL.
    // Other routes retain their existing path-based descriptions.
    let description_path = match route.openapi_kind() {
        Some(OpenApiRouteKind::ChatCompletions) => "/v1/chat/completions",
        Some(OpenApiRouteKind::Completions) => "/v1/completions",
        Some(OpenApiRouteKind::Embeddings) => "/v1/embeddings",
        Some(OpenApiRouteKind::Responses) => "/v1/responses",
        None => path,
    };
    let summary = generate_summary_for_path(description_path);
    let description = generate_description_for_path(description_path);

    let mut operation = OperationBuilder::new()
        .operation_id(Some(operation_id))
        .summary(Some(summary))
        .description(Some(description));

    // Add request body for POST methods
    if method.to_uppercase() == "POST" {
        operation = add_request_body_for_route(operation, route.openapi_kind());
    }

    // Add responses
    operation = operation.response("200", success_response(route));

    operation = operation.response(
        "400",
        ResponseBuilder::new()
            .description("Bad request - invalid input")
            .build(),
    );

    operation = operation.response(
        "404",
        ResponseBuilder::new()
            .description("Model not found")
            .build(),
    );

    operation = operation.response(
        "503",
        ResponseBuilder::new()
            .description("Service unavailable")
            .build(),
    );

    operation = operation.response(
        "529",
        ResponseBuilder::new()
            .description("Service overloaded")
            .build(),
    );

    operation.build()
}

/// Describe success media types for chat/completion POSTs, or omit content otherwise.
///
/// Only registered chat/completion POST handlers get JSON and SSE declarations,
/// including at custom paths. The exporter adds itemSchema after serialization because Utoipa 5 cannot
/// represent it; this intermediate response must not be published on its own.
fn success_response(route: &RouteDoc) -> utoipa::openapi::Response {
    use utoipa::openapi::{ContentBuilder, Ref, ResponseBuilder};

    let response = ResponseBuilder::new().description("Successful response");
    let Some((unary, _, _)) = completion_response_types(route) else {
        return response.build();
    };
    response
        .description(
            "With stream=false, returns a JSON response. With stream=true, returns \
             server-sent events described by itemSchema: JSON chunks, structured \
             errors, and the [DONE] marker, plus optional named dispatch events \
             for chat. Parse SSE before validating the string data's contentSchema. \
             Comment-only annotations and keep-alives are not data events. Ordering, \
             termination, and disconnect behavior require separate lifecycle checks.",
        )
        .content(
            "application/json",
            ContentBuilder::new()
                .schema(Some(Ref::from_schema_name(unary)))
                .build(),
        )
        .content("text/event-stream", ContentBuilder::new().build())
        .build()
}

/// Select unary/chunk component names and chat status for a registered streaming POST.
/// All other methods and handler kinds return `None`; URLs do not imply contracts.
fn completion_response_types(
    route: &RouteDoc,
) -> Option<(
    std::borrow::Cow<'static, str>,
    std::borrow::Cow<'static, str>,
    bool,
)> {
    if route.method() != axum::http::Method::POST {
        return None;
    }
    match route.openapi_kind()? {
        OpenApiRouteKind::ChatCompletions => Some((
            NvCreateChatCompletionResponse::name(),
            NvCreateChatCompletionStreamResponse::name(),
            true,
        )),
        OpenApiRouteKind::Completions => Some((
            NvCreateCompletionResponse::name(),
            NvCreateCompletionResponse::name(),
            false,
        )),
        _ => None,
    }
}

/// Add a registered POST handler's request component, or a generic body for unknown handlers.
fn add_request_body_for_route(
    operation: utoipa::openapi::path::OperationBuilder,
    kind: Option<OpenApiRouteKind>,
) -> utoipa::openapi::path::OperationBuilder {
    use utoipa::openapi::ContentBuilder;
    use utoipa::openapi::request_body::RequestBodyBuilder;

    let (description, schema, example) = match kind {
        Some(OpenApiRouteKind::ChatCompletions) => (
            "Chat completion request with model, messages, and optional parameters",
            create_chat_completion_schema(),
            create_chat_completion_example(),
        ),
        Some(OpenApiRouteKind::Completions) => (
            "Text completion request with model, prompt, and optional parameters",
            create_completion_schema(),
            create_completion_example(),
        ),
        Some(OpenApiRouteKind::Embeddings) => (
            "Embedding request with model and input text",
            create_embedding_schema(),
            create_embedding_example(),
        ),
        Some(OpenApiRouteKind::Responses) => (
            "Response request with model and input",
            create_response_schema(),
            create_response_example(),
        ),
        _ => {
            return operation.request_body(Some(
                RequestBodyBuilder::new()
                    .description(Some("Request body"))
                    .required(Some(utoipa::openapi::Required::True))
                    .build(),
            ));
        }
    };

    operation.request_body(Some(
        RequestBodyBuilder::new()
            .description(Some(description))
            .content(
                "application/json",
                ContentBuilder::new()
                    .schema(Some(schema))
                    .example(Some(example))
                    .build(),
            )
            .required(Some(utoipa::openapi::Required::True))
            .build(),
    ))
}

/// Reference the registered chat request component without copying its schema.
fn create_chat_completion_schema() -> RefOr<utoipa::openapi::schema::Schema> {
    utoipa::openapi::Ref::from_schema_name(NvCreateChatCompletionRequest::name()).into()
}

/// Create example for chat completion request
fn create_chat_completion_example() -> serde_json::Value {
    serde_json::json!({
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello! Can you help me understand what this API does?"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": false
    })
}

/// Reference the registered completion request component without copying its schema.
fn create_completion_schema() -> RefOr<utoipa::openapi::schema::Schema> {
    utoipa::openapi::Ref::from_schema_name(NvCreateCompletionRequest::name()).into()
}

/// Create example for completion request
fn create_completion_example() -> serde_json::Value {
    serde_json::json!({
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Once upon a time",
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": false
    })
}

/// Reference the registered embedding request component without duplicating its schema.
fn create_embedding_schema() -> RefOr<utoipa::openapi::schema::Schema> {
    utoipa::openapi::Ref::from_schema_name(NvCreateEmbeddingRequest::name()).into()
}

/// Create example for embedding request
fn create_embedding_example() -> serde_json::Value {
    serde_json::json!({
        "model": "Qwen/Qwen3-Embedding-4B",
        "input": "The quick brown fox jumps over the lazy dog"
    })
}

/// Reference the registered Responses request component without duplicating its schema.
fn create_response_schema() -> RefOr<utoipa::openapi::schema::Schema> {
    utoipa::openapi::Ref::from_schema_name(NvCreateResponse::name()).into()
}

/// Create example for response request
fn create_response_example() -> serde_json::Value {
    serde_json::json!({
        "model": "Qwen/Qwen3-0.6B",
        "input": "What is the capital of France?"
    })
}

/// Generate a human-readable summary for a path
fn generate_summary_for_path(path: &str) -> String {
    match path {
        "/v1/chat/completions" => "Create chat completion".to_string(),
        "/v1/completions" => "Create text completion".to_string(),
        "/v1/embeddings" => "Create embeddings".to_string(),
        "/v1/responses" => "Create response".to_string(),
        "/v1/models" => "List available models".to_string(),
        "/health" => "Health check".to_string(),
        "/live" => "Liveness check".to_string(),
        "/metrics" => "Prometheus metrics".to_string(),
        "/openapi.json" => "OpenAPI specification".to_string(),
        "/docs" => "API documentation".to_string(),
        _ => format!("Endpoint: {}", path),
    }
}

/// Generate a detailed description for a path
fn generate_description_for_path(path: &str) -> String {
    match path {
        "/v1/chat/completions" => {
            "Creates a completion for a chat conversation. Supports both streaming and non-streaming modes. \
            Compatible with OpenAI's chat completions API."
                .to_string()
        }
        "/v1/completions" => {
            "Creates a completion for a given prompt. Supports both streaming and non-streaming modes. \
            Compatible with OpenAI's completions API."
                .to_string()
        }
        "/v1/embeddings" => {
            "Creates an embedding vector representing the input text. \
            Compatible with OpenAI's embeddings API."
                .to_string()
        }
        "/v1/responses" => {
            "Creates a response for a given input. Compatible with OpenAI's responses API."
                .to_string()
        }
        "/v1/models" => {
            "Lists the currently available models and provides basic information about each."
                .to_string()
        }
        "/health" => {
            "Returns the health status of the service. Used for readiness probes."
                .to_string()
        }
        "/live" => {
            "Returns the liveness status of the service. Used for liveness probes."
                .to_string()
        }
        "/metrics" => {
            "Returns Prometheus metrics for monitoring the service."
                .to_string()
        }
        "/openapi.json" => {
            "Returns the OpenAPI 3.2 specification for this API in JSON format."
                .to_string()
        }
        "/docs" => {
            "Interactive API documentation powered by Swagger UI."
                .to_string()
        }
        _ => format!("Endpoint for path: {}", path),
    }
}

/// Build the documentation routes to mount alongside a frontend's API routes.
///
/// # Arguments
///
/// * `route_docs` - API routes to describe; see [`generate_openapi_spec_with_options`].
/// * `options` - Response settings; must match the frontend's serializer and flags.
///
/// # Returns
///
/// A pair of metadata for GET `/openapi.json` and GET `/docs`, and an unserved router
/// providing those endpoints. The metadata does not include the input API routes;
/// nor is it automatically added to the generated document. The caller must mount
/// and serve the router; this function does not start a listener.
///
/// Generation is lazy and cached as serialized bytes per router/service. A schema
/// error is logged once and cached as HTTP 500 for `/openapi.json`; inference remains
/// available. Neither an invalid partial document nor internal diagnostics are served.
/// Thread-launch failures return HTTP 500 without initializing the cache, so a later
/// request can retry.
pub fn openapi_router(
    route_docs: Vec<RouteDoc>,
    options: OpenApiOptions,
) -> (Vec<RouteDoc>, Router) {
    let document = Arc::new(CachedOpenApi {
        routes: route_docs,
        options,
        bytes: OnceCell::new(),
    });
    let router = documentation_router(document);

    let docs = vec![
        RouteDoc::new(axum::http::Method::GET, "/openapi.json"),
        RouteDoc::new(axum::http::Method::GET, "/docs"),
    ];

    (docs, router)
}

/// Immutable service-specific inputs and the once-generated document or failure.
struct CachedOpenApi {
    routes: Vec<RouteDoc>,
    options: OpenApiOptions,
    bytes: OnceCell<Result<Bytes, String>>,
}

impl CachedOpenApi {
    /// Return shared serialized bytes, generating on an 8 MiB thread on first access.
    /// Await its channel without parking a blocking-pool thread. Cache generation
    /// failures, but allow later requests to retry after a thread-launch failure.
    async fn bytes(self: &Arc<Self>) -> Result<Bytes, String> {
        self.bytes_with_spawn(|generate| {
            std::thread::Builder::new()
                .name("openapi-schema".into())
                .stack_size(8 * 1024 * 1024)
                .spawn(generate)
                .map(|_| ())
        })
        .await
    }

    /// Initialize the cache with an injectable thread launcher for failure testing.
    /// `spawn` must run its task independently on success, or return an error without
    /// running it. Launch errors leave the cache empty; generation errors or a lost
    /// sender are cached. Successful calls return a cheap clone of the cached bytes.
    async fn bytes_with_spawn<F>(self: &Arc<Self>, spawn: F) -> Result<Bytes, String>
    where
        F: FnOnce(Box<dyn FnOnce() + Send + 'static>) -> std::io::Result<()>,
    {
        self.bytes
            .get_or_try_init(|| async {
                let document = Arc::clone(self);
                let (sender, receiver) = tokio::sync::oneshot::channel();
                spawn(Box::new(move || {
                    let result = generate_openapi_spec_with_options(&document.routes, document.options)
                        .and_then(|value| {
                            serde_json::to_vec(&value)
                                .map(Bytes::from)
                                .map_err(|error| error.to_string())
                        });
                    // A cancelled request can drop the receiver while generation finishes.
                    let _ = sender.send(result);
                }))?;
                let result = receiver.await.unwrap_or_else(|_| {
                    Err("OpenAPI generator exited without a result".to_owned())
                });
                if let Err(error) = &result {
                    tracing::error!(%error, "OpenAPI documentation unavailable; inference is unaffected");
                }
                Ok::<_, std::io::Error>(result)
            })
            .await
            .map_err(|error| {
                tracing::error!(%error, "OpenAPI generator could not start; a later request can retry");
                error.to_string()
            })?
            .clone()
    }
}

/// Mount the cached JSON handler and Swagger assets without eagerly generating schemas.
fn documentation_router(document: Arc<CachedOpenApi>) -> Router {
    use utoipa_swagger_ui::{Config, SwaggerUi};

    Router::new()
        .route("/openapi.json", axum::routing::get(serve_openapi))
        .with_state(document)
        .merge(SwaggerUi::new("/docs").config(Config::new(["/openapi.json"])))
}

/// Serve cached JSON bytes, or a generic 500 when generation failed; never expose diagnostics.
async fn serve_openapi(
    axum::extract::State(document): axum::extract::State<Arc<CachedOpenApi>>,
) -> axum::response::Response {
    match document.bytes().await {
        Ok(bytes) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "OpenAPI documentation unavailable",
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Partial and remapped registrations must not invent default or POST routes.
    #[test]
    fn registered_handlers_keep_contracts_at_custom_paths() {
        use super::super::{openai, service_v2::HttpService};
        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let registrations = [
            openai::chat_completions_router(state.clone(), None, Some("/custom/~chat".into())),
            openai::completions_router(state.clone(), Some("/custom/completions".into())),
            openai::embeddings_router(state.clone(), Some("/custom/embeddings".into())),
            openai::responses_router(state, None, Some("/custom/responses".into())),
        ];
        for ((routes, _), (request, stream)) in registrations.into_iter().zip([
            (
                NvCreateChatCompletionRequest::name(),
                Some(NvCreateChatCompletionStreamResponse::name()),
            ),
            (
                NvCreateCompletionRequest::name(),
                Some(NvCreateCompletionResponse::name()),
            ),
            (NvCreateEmbeddingRequest::name(), None),
            (NvCreateResponse::name(), None),
        ]) {
            let spec = generate_openapi_spec(&routes).unwrap();
            assert_eq!(spec["paths"].as_object().unwrap().len(), routes.len());
            let operation = &spec["paths"][routes[0].path()]["post"];
            let (summary, description) = match routes[0].openapi_kind().unwrap() {
                OpenApiRouteKind::ChatCompletions => {
                    ("Create chat completion", "chat conversation")
                }
                OpenApiRouteKind::Completions => ("Create text completion", "given prompt"),
                OpenApiRouteKind::Embeddings => ("Create embeddings", "embedding vector"),
                OpenApiRouteKind::Responses => ("Create response", "given input"),
            };
            assert_eq!(operation["summary"], summary);
            assert!(
                operation["description"]
                    .as_str()
                    .unwrap()
                    .contains(description)
            );
            assert_eq!(
                operation["requestBody"]["content"]["application/json"]["schema"]["$ref"],
                format!("#/components/schemas/{request}")
            );
            if let Some(stream) = stream {
                assert_eq!(
                    operation["responses"]["200"]["content"]["text/event-stream"]["itemSchema"]["anyOf"]
                        [0]["properties"]["data"]["anyOf"][1]["contentSchema"]["$ref"],
                    format!("#/components/schemas/{stream}")
                );
            }
        }
        for routes in [
            vec![],
            vec![RouteDoc::new(
                axum::http::Method::GET,
                "/v1/chat/completions",
            )],
            vec![RouteDoc::new(
                axum::http::Method::POST,
                "/v1/chat/completions",
            )],
        ] {
            let spec = generate_openapi_spec(&routes).unwrap();
            assert_eq!(spec["paths"].as_object().unwrap().len(), routes.len());
            assert!(
                spec.pointer("/paths/~1v1~1chat~1completions/post/responses/200/content")
                    .is_none()
            );
        }
    }

    /// Each dispatch flag independently controls the chat event union, never completions.
    #[test]
    fn dispatch_event_contracts_follow_configuration() {
        let routes = [RouteDoc::new(axum::http::Method::POST, "/chat")
            .with_openapi_kind(OpenApiRouteKind::ChatCompletions)];
        for tool in [false, true] {
            for reasoning in [false, true] {
                let spec = generate_openapi_spec_with_options(
                    &routes,
                    OpenApiOptions {
                        streaming_tool_dispatch: tool,
                        streaming_reasoning_dispatch: reasoning,
                        ..Default::default()
                    },
                )
                .unwrap();
                let events = spec["paths"]["/chat"]["post"]["responses"]["200"]["content"]
                    ["text/event-stream"]["itemSchema"]["anyOf"]
                    .as_array()
                    .unwrap();
                assert_eq!(events.len(), 1 + usize::from(tool) + usize::from(reasoning));
                for (event, enabled) in [
                    ("tool_call_dispatch", tool),
                    ("reasoning_dispatch", reasoning),
                ] {
                    assert_eq!(
                        events
                            .iter()
                            .any(|schema| schema["properties"]["event"]["const"] == event),
                        enabled
                    );
                }
            }
        }
    }

    /// Repeated concurrent HTTP requests reuse bytes; independent routers keep separate options.
    #[tokio::test]
    async fn openapi_http_cache_is_lazy_and_per_service() {
        for tool in [false, true] {
            let document = Arc::new(CachedOpenApi {
                routes: vec![
                    RouteDoc::new(axum::http::Method::POST, "/chat")
                        .with_openapi_kind(OpenApiRouteKind::ChatCompletions),
                ],
                options: OpenApiOptions {
                    streaming_tool_dispatch: tool,
                    ..Default::default()
                },
                bytes: OnceCell::new(),
            });
            let router = documentation_router(document.clone());
            assert!(
                document.bytes.get().is_none(),
                "router construction must be lazy"
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/openapi.json", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
            let client = reqwest::Client::builder().no_proxy().build().unwrap();
            let (first, second) = tokio::join!(client.get(&url).send(), client.get(&url).send());
            let first = first.unwrap();
            assert_eq!(first.status(), StatusCode::OK);
            assert_eq!(first.headers()["content-type"], "application/json");
            let first = first.bytes().await.unwrap();
            assert_eq!(first, second.unwrap().bytes().await.unwrap());
            assert_eq!(&first, document.bytes.get().unwrap().as_ref().unwrap());
            let cached = document.bytes().await.unwrap();
            assert_eq!(
                cached.as_ptr(),
                document.bytes().await.as_ref().unwrap().as_ptr()
            );
            let spec: serde_json::Value = serde_json::from_slice(&first).unwrap();
            assert_eq!(
                spec["paths"]["/chat"]["post"]["responses"]["200"]["content"]["text/event-stream"]
                    ["itemSchema"]["anyOf"]
                    .as_array()
                    .unwrap()
                    .len(),
                1 + usize::from(tool)
            );
            let initializer = client
                .get(url.replace("openapi.json", "docs/swagger-initializer.js"))
                .send()
                .await
                .unwrap();
            assert_eq!(initializer.status(), StatusCode::OK);
            assert!(initializer.text().await.unwrap().contains("/openapi.json"));
            server.abort();
            let _ = server.await;
        }
    }

    /// A failed launch must leave the document retryable, then reuse the successful bytes.
    #[tokio::test]
    async fn openapi_thread_launch_failure_is_retryable() {
        let document = Arc::new(CachedOpenApi {
            routes: vec![],
            options: OpenApiOptions::default(),
            bytes: OnceCell::new(),
        });
        let failed = document
            .bytes_with_spawn(|_| Err(std::io::Error::from(std::io::ErrorKind::WouldBlock)))
            .await;
        assert!(failed.is_err());
        assert!(document.bytes.get().is_none());
        let bytes = document.bytes().await.unwrap();
        let cached = document
            .bytes_with_spawn(|_| panic!("cached bytes must not start another thread"))
            .await
            .unwrap();
        assert_eq!(bytes.as_ptr(), cached.as_ptr());
    }

    /// Handler prose takes precedence over URL spelling; untyped routes keep their fallback.
    #[test]
    fn openapi_route_prose_follows_handler_identity() {
        let route = RouteDoc::new(axum::http::Method::POST, "/v1/embeddings")
            .with_openapi_kind(OpenApiRouteKind::ChatCompletions);
        let operation = create_operation_for_route(&route);
        assert_eq!(operation.summary.as_deref(), Some("Create chat completion"));
        assert!(operation.description.unwrap().contains("chat conversation"));
        assert_eq!(
            operation.operation_id.as_deref(),
            Some("post_v1_embeddings")
        );

        let health = create_operation_for_route(&RouteDoc::new(axum::http::Method::GET, "/health"));
        assert_eq!(health.summary.as_deref(), Some("Health check"));
        let custom = create_operation_for_route(&RouteDoc::new(axum::http::Method::GET, "/custom"));
        assert_eq!(custom.summary.as_deref(), Some("Endpoint: /custom"));
        assert_eq!(
            custom.description.as_deref(),
            Some("Endpoint for path: /custom")
        );
    }

    /// A generator that exits without sending a result must not hang or relaunch forever.
    #[tokio::test]
    async fn openapi_generator_exit_is_cached() {
        let document = Arc::new(CachedOpenApi {
            routes: vec![],
            options: OpenApiOptions::default(),
            bytes: OnceCell::new(),
        });
        let error = document
            .bytes_with_spawn(|task| {
                drop(task);
                Ok(())
            })
            .await
            .unwrap_err();
        assert_eq!(error, "OpenAPI generator exited without a result");
        assert_eq!(
            document
                .bytes_with_spawn(|_| panic!("cached error must not start another thread"))
                .await
                .unwrap_err(),
            error
        );
    }

    /// A cached generation error returns a generic 500 without disabling unrelated routes.
    #[tokio::test]
    async fn openapi_failure_is_confined_to_documentation() {
        let document = Arc::new(CachedOpenApi {
            routes: vec![],
            options: OpenApiOptions::default(),
            bytes: OnceCell::new_with(Some(Err("internal schema invariant diagnostic".into()))),
        });
        let router = documentation_router(document.clone())
            .route("/inference", axum::routing::get(|| async { "available" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        for _ in 0..2 {
            let response = client
                .get(format!("{url}/openapi.json"))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(
                response.text().await.unwrap(),
                "OpenAPI documentation unavailable"
            );
        }
        assert!(document.bytes().await.is_err());
        assert_eq!(
            client
                .get(format!("{url}/inference"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        server.abort();
        let _ = server.await;
    }

    #[test]
    fn stale_alias_metadata_is_not_silently_ignored() {
        let mut document = utoipa::openapi::OpenApi::default();
        assert!(annotate_request_aliases(&mut document).is_err());
    }

    #[test]
    fn stale_response_reasoning_schemas_are_not_silently_ignored() {
        use crate::reasoning_field::ReasoningField;
        use utoipa::openapi::{Ref, Schema};

        // Exercise both projections and both response types: dependency schema
        // drift must fail export instead of silently advertising the wrong key.
        for field in [ReasoningField::ReasoningContent, ReasoningField::Reasoning] {
            let mut missing_components = utoipa::openapi::OpenApi::default();
            assert!(configure_response_schemas(&mut missing_components, field).is_err());
            for name in [
                "ChatCompletionResponseMessage",
                "ChatCompletionStreamResponseDelta",
            ] {
                let name = format!("dynamo_protocols.chat.{name}");
                for change in [
                    "missing component",
                    "reference",
                    "missing property",
                    "collision",
                ] {
                    let mut document = ApiDoc::openapi();
                    let schemas = &mut document.components.as_mut().unwrap().schemas;
                    match change {
                        "missing component" => {
                            schemas.remove(&name);
                        }
                        "reference" => {
                            schemas.insert(name.clone(), Ref::from_schema_name("Other").into());
                        }
                        _ => {
                            let RefOr::T(Schema::Object(object)) = schemas.get_mut(&name).unwrap()
                            else {
                                panic!("expected response object at {name}");
                            };
                            let property = object.properties.remove("reasoning_content").unwrap();
                            if change == "collision" {
                                object
                                    .properties
                                    .insert("reasoning".to_owned(), property.clone());
                                object
                                    .properties
                                    .insert("reasoning_content".to_owned(), property);
                            }
                        }
                    }
                    let error = configure_response_schemas(&mut document, field).unwrap_err();
                    assert!(error.contains(&name), "{change}: {error}");
                }
            }
        }
    }

    #[test]
    fn test_generate_openapi_spec() {
        let routes = vec![
            RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions")
                .with_openapi_kind(OpenApiRouteKind::ChatCompletions),
            RouteDoc::new(axum::http::Method::GET, "/v1/models"),
        ];

        let spec = generate_openapi_spec(&routes).unwrap();

        // Verify basic structure
        assert!(!spec["info"]["title"].as_str().unwrap().is_empty());
        assert!(!spec["info"]["version"].as_str().unwrap().is_empty());
        assert_eq!(spec["openapi"], "3.2.0");

        // Verify paths were added
        assert!(spec["paths"].get("/v1/chat/completions").is_some());
        assert!(spec["paths"].get("/v1/models").is_some());
        assert_eq!(spec["paths"].as_object().unwrap().len(), 2);
        assert!(spec["paths"].get("/v1/completions").is_none());
    }

    // Two methods on one path (e.g. built-in GET+POST /busy_threshold, or an
    // extension GET on a built-in POST path) must both survive in the spec,
    // not overwrite each other under the path key.
    #[test]
    fn openapi_spec_keeps_all_methods_on_a_shared_path() {
        let routes = vec![
            RouteDoc::new(axum::http::Method::POST, "/busy_threshold"),
            RouteDoc::new(axum::http::Method::GET, "/busy_threshold"),
        ];
        let spec = generate_openapi_spec(&routes).unwrap();
        let item = &spec["paths"]["/busy_threshold"];
        assert!(item.get("get").is_some(), "GET operation missing: {item}");
        assert!(item.get("post").is_some(), "POST operation missing: {item}");
    }
}
