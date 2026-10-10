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

use axum::Router;
use utoipa::OpenApi;
use utoipa::openapi::{PathItem, Paths, RefOr};

use crate::http::service::RouteDoc;

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
            crate::protocols::openai::responses::NvCreateResponse
        )
    )
)]
struct ApiDoc;

/// Generate the native document with the default `reasoning_content` response key.
///
/// Use when no response-key override is needed. The `route_docs` contract, returned
/// document, and panics are described by [`generate_openapi_spec_with_reasoning_field`].
pub fn generate_openapi_spec(route_docs: &[RouteDoc]) -> utoipa::openapi::OpenApi {
    generate_openapi_spec_with_reasoning_field(
        route_docs,
        crate::reasoning_field::ReasoningField::DEFAULT,
    )
}

/// Generate a native OpenAPI document for export or embedding without starting HTTP.
///
/// # Arguments
///
/// * `route_docs` - Routes to describe. Invalid route strings and unsupported methods
///   are skipped with a warning; supported methods on the same path are combined.
/// * `reasoning_field` - Response property spelling to advertise; use the same value
///   as the frontend's response serialization configuration.
///
/// # Returns
///
/// An owned document for the supplied routes with native components and request-alias
/// metadata. Dependency import slots remain unresolved; this is not a fully composed contract.
///
/// # Panics
///
/// Panics if compiled schemas violate the alias or reasoning-projection invariants,
/// rather than publishing stale metadata.
pub fn generate_openapi_spec_with_reasoning_field(
    route_docs: &[RouteDoc],
    reasoning_field: crate::reasoning_field::ReasoningField,
) -> utoipa::openapi::OpenApi {
    let mut openapi = ApiDoc::openapi();
    // Like input aliases below, response projection is a compiled-schema invariant.
    configure_response_schemas(&mut openapi, reasoning_field)
        .expect("response reasoning schema locations changed");

    // Build paths from route documentation
    let mut paths = Paths::new();

    for route in route_docs {
        let path_str = route.to_string();
        tracing::debug!("Adding route to OpenAPI spec: {}", path_str);

        // Parse the route to extract method and path
        let parts: Vec<&str> = path_str.split_whitespace().collect();
        if parts.len() != 2 {
            tracing::warn!("Invalid route format: {}", path_str);
            continue;
        }

        let method = parts[0];
        let path = parts[1];

        // Add operation based on method
        let operation = create_operation_for_route(method, path);

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
    // Deliberately fail startup rather than publish incomplete alias metadata.
    // `http_export_describes_request_aliases` exercises this path in the
    // pre-merge Rust suite so schema changes must update the declarations too.
    annotate_request_aliases(&mut openapi).expect("request alias schema locations changed");
    openapi
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
            "NvCreateChatCompletionRequest",
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
fn create_operation_for_route(method: &str, path: &str) -> utoipa::openapi::path::Operation {
    use utoipa::openapi::ResponseBuilder;
    use utoipa::openapi::path::OperationBuilder;

    let operation_id = format!(
        "{}_{}",
        method.to_lowercase(),
        path.replace('/', "_").trim_matches('_')
    );
    let summary = generate_summary_for_path(path);
    let description = generate_description_for_path(path);

    let mut operation = OperationBuilder::new()
        .operation_id(Some(operation_id))
        .summary(Some(summary))
        .description(Some(description));

    // Add request body for POST methods
    if method.to_uppercase() == "POST" {
        operation = add_request_body_for_path(operation, path);
    }

    // Add responses
    operation = operation.response("200", success_response(method, path));

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

/// Describe success payloads for chat/completion POSTs, or omit content otherwise.
///
/// `method` and `path` select the response declaration: only POSTs to
/// `/v1/chat/completions` and `/v1/completions` get JSON and SSE schema references.
/// The SSE schema describes one successful JSON `data` payload, not the
/// transport framing, error events, annotations, or the literal `[DONE]` marker.
/// `x-dynamo-sse-data-schema: true` marks this payload-only convention for consumers.
fn success_response(method: &str, path: &str) -> utoipa::openapi::Response {
    use utoipa::openapi::{ContentBuilder, Ref, ResponseBuilder};

    let response = ResponseBuilder::new().description("Successful response");
    if !method.eq_ignore_ascii_case("POST") {
        return response.build();
    }
    let (unary, streaming) = match path {
        "/v1/chat/completions" => (
            "NvCreateChatCompletionResponse",
            "NvCreateChatCompletionStreamResponse",
        ),
        // The legacy completion API serializes the same type for both modes.
        "/v1/completions" => ("NvCreateCompletionResponse", "NvCreateCompletionResponse"),
        _ => return response.build(),
    };
    response
        .description(
            "With stream=false, returns a JSON response. With stream=true, returns \
             server-sent events; each successful JSON data payload follows the \
             text/event-stream schema. The literal data: [DONE] terminates the \
             stream and is not a JSON chunk. Error events and optional annotation \
             events are outside this successful-payload schema.",
        )
        .content(
            "application/json",
            ContentBuilder::new()
                .schema(Some(Ref::from_schema_name(unary)))
                .build(),
        )
        .content(
            "text/event-stream",
            ContentBuilder::new()
                .schema(Some(Ref::from_schema_name(streaming)))
                .extensions(Some(
                    [("x-dynamo-sse-data-schema", serde_json::json!(true))]
                        .into_iter()
                        .collect(),
                ))
                .build(),
        )
        .build()
}

/// Add request body schema for POST endpoints
fn add_request_body_for_path(
    operation: utoipa::openapi::path::OperationBuilder,
    path: &str,
) -> utoipa::openapi::path::OperationBuilder {
    use utoipa::openapi::ContentBuilder;
    use utoipa::openapi::request_body::RequestBodyBuilder;

    let (description, schema, example) = match path {
        "/v1/chat/completions" => (
            "Chat completion request with model, messages, and optional parameters",
            create_chat_completion_schema(),
            create_chat_completion_example(),
        ),
        "/v1/completions" => (
            "Text completion request with model, prompt, and optional parameters",
            create_completion_schema(),
            create_completion_example(),
        ),
        "/v1/embeddings" => (
            "Embedding request with model and input text",
            create_embedding_schema(),
            create_embedding_example(),
        ),
        "/v1/responses" => (
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
    utoipa::openapi::Ref::from_schema_name("NvCreateChatCompletionRequest").into()
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
    utoipa::openapi::Ref::from_schema_name("NvCreateCompletionRequest").into()
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

/// Create schema for embedding request
fn create_embedding_schema() -> RefOr<utoipa::openapi::schema::Schema> {
    <crate::protocols::openai::embeddings::NvCreateEmbeddingRequest as utoipa::PartialSchema>::schema()
}

/// Create example for embedding request
fn create_embedding_example() -> serde_json::Value {
    serde_json::json!({
        "model": "Qwen/Qwen3-Embedding-4B",
        "input": "The quick brown fox jumps over the lazy dog"
    })
}

/// Create schema for response request
fn create_response_schema() -> RefOr<utoipa::openapi::schema::Schema> {
    // Schema derived from NvCreateResponse type via ToSchema
    <crate::protocols::openai::responses::NvCreateResponse as utoipa::PartialSchema>::schema()
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
            "Returns the OpenAPI 3.0 specification for this API in JSON format."
                .to_string()
        }
        "/docs" => {
            "Interactive API documentation powered by Swagger UI."
                .to_string()
        }
        _ => format!("Endpoint for path: {}", path),
    }
}

/// Build the default-reasoning documentation routes without starting a listener.
///
/// Use when no response-key override is needed. `_path` is ignored; the `route_docs`
/// contract, return values, and panics match [`openapi_router_with_reasoning_field`].
pub fn openapi_router(route_docs: Vec<RouteDoc>, _path: Option<String>) -> (Vec<RouteDoc>, Router) {
    openapi_router_with_reasoning_field(route_docs, crate::reasoning_field::ReasoningField::DEFAULT)
}

/// Build the documentation routes to mount alongside a frontend's API routes.
///
/// # Arguments
///
/// * `route_docs` - API routes to describe, consumed to generate the document under
///   the rules of [`generate_openapi_spec_with_reasoning_field`].
/// * `reasoning_field` - Response-key spelling; must match the frontend's serializer.
///
/// # Returns
///
/// A pair of metadata for GET `/openapi.json` and GET `/docs`, and an unserved router
/// providing those endpoints. The metadata does not include the input API routes;
/// nor is it automatically added to the generated document. The caller must mount
/// and serve the router; this function does not start a listener.
///
/// # Panics
///
/// Panics on the schema invariants documented by [`generate_openapi_spec_with_reasoning_field`].
pub fn openapi_router_with_reasoning_field(
    route_docs: Vec<RouteDoc>,
    reasoning_field: crate::reasoning_field::ReasoningField,
) -> (Vec<RouteDoc>, Router) {
    use utoipa_swagger_ui::SwaggerUi;

    // Generate the OpenAPI spec from route docs
    let openapi_spec = generate_openapi_spec_with_reasoning_field(&route_docs, reasoning_field);

    // Note: SwaggerUi requires a static string for the URL path, so we ignore the custom path
    // parameter and always use "/openapi.json"
    let openapi_path = "/openapi.json";

    // Create Swagger UI with the OpenAPI spec
    // SwaggerUi automatically serves both the spec at /openapi.json and the UI at /docs
    let swagger_ui = SwaggerUi::new("/docs").url(openapi_path, openapi_spec);

    // SwaggerUi handles both routes internally, so we just merge it
    let router = Router::new().merge(swagger_ui);

    let docs = vec![
        RouteDoc::new(axum::http::Method::GET, openapi_path),
        RouteDoc::new(axum::http::Method::GET, "/docs"),
    ];

    (docs, router)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions"),
            RouteDoc::new(axum::http::Method::GET, "/v1/models"),
        ];

        let spec = generate_openapi_spec(&routes);

        // Verify basic structure
        assert!(!spec.info.title.is_empty());
        assert!(!spec.info.version.is_empty());

        // Verify paths were added
        assert!(spec.paths.paths.contains_key("/v1/chat/completions"));
        assert!(spec.paths.paths.contains_key("/v1/models"));
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
        let spec = generate_openapi_spec(&routes);
        let json = serde_json::to_value(&spec).unwrap();
        let item = &json["paths"]["/busy_threshold"];
        assert!(item.get("get").is_some(), "GET operation missing: {item}");
        assert!(item.get("post").is_some(), "POST operation missing: {item}");
    }
}
