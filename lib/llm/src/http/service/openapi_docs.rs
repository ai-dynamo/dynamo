// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAPI documentation generation and Swagger UI integration
//!
//! This module provides automatic OpenAPI specification generation from the HTTP service routes
//! and serves Swagger UI for interactive API documentation.
//!
//! ## Features
//!
//! - **OpenAPI 3.0 Specification**: Automatically generates OpenAPI spec from defined routes
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
/// This struct defines the complete OpenAPI specification for the Dynamo HTTP service.
/// It includes all the schemas, paths, and metadata needed to document the API.
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
    )
)]
struct ApiDoc;

/// Generate OpenAPI specification from route documentation
fn generate_openapi_spec(route_docs: &[RouteDoc]) -> utoipa::openapi::OpenApi {
    let mut openapi = ApiDoc::openapi();

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

        paths.paths.insert(path.to_string(), path_item);
    }

    openapi.paths = paths;
    openapi
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
    operation = operation.response(
        "200",
        ResponseBuilder::new()
            .description("Successful response")
            .build(),
    );

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

    operation.build()
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

/// Create schema for chat completion request
fn create_chat_completion_schema() -> RefOr<utoipa::openapi::schema::Schema> {
    use utoipa::openapi::schema::{ArrayBuilder, ObjectBuilder};

    RefOr::T(utoipa::openapi::schema::Schema::Object(
        ObjectBuilder::new()
            .property(
                "model",
                ObjectBuilder::new()
                    .description(Some("ID of the model to use"))
                    .build(),
            )
            .property(
                "messages",
                ArrayBuilder::new()
                    .description(Some("A list of messages comprising the conversation so far"))
                    .items(
                        ObjectBuilder::new()
                            .property(
                                "role",
                                ObjectBuilder::new()
                                    .description(Some("The role of the message author (system, user, assistant)"))
                                    .build(),
                            )
                            .property(
                                "content",
                                ObjectBuilder::new()
                                    .description(Some("The contents of the message"))
                                    .build(),
                            )
                            .build(),
                    )
                    .build(),
            )
            .property(
                "temperature",
                ObjectBuilder::new()
                    .description(Some("Sampling temperature between 0 and 2. Higher values make output more random"))
                    .build(),
            )
            .property(
                "max_tokens",
                ObjectBuilder::new()
                    .description(Some("Maximum number of tokens to generate"))
                    .build(),
            )
            .property(
                "stream",
                ObjectBuilder::new()
                    .description(Some("Whether to stream back partial progress"))
                    .build(),
            )
            .required("model")
            .required("messages")
            .build(),
    ))
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

/// Create schema for completion request
fn create_completion_schema() -> RefOr<utoipa::openapi::schema::Schema> {
    use utoipa::openapi::schema::ObjectBuilder;

    RefOr::T(utoipa::openapi::schema::Schema::Object(
        ObjectBuilder::new()
            .property(
                "model",
                ObjectBuilder::new()
                    .description(Some("ID of the model to use"))
                    .build(),
            )
            .property(
                "prompt",
                ObjectBuilder::new()
                    .description(Some("The prompt to generate completions for"))
                    .build(),
            )
            .property(
                "temperature",
                ObjectBuilder::new()
                    .description(Some("Sampling temperature between 0 and 2"))
                    .build(),
            )
            .property(
                "max_tokens",
                ObjectBuilder::new()
                    .description(Some("Maximum number of tokens to generate"))
                    .build(),
            )
            .property(
                "stream",
                ObjectBuilder::new()
                    .description(Some("Whether to stream back partial progress"))
                    .build(),
            )
            .required("model")
            .required("prompt")
            .build(),
    ))
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
    use utoipa::openapi::schema::ObjectBuilder;

    RefOr::T(utoipa::openapi::schema::Schema::Object(
        ObjectBuilder::new()
            .property(
                "model",
                ObjectBuilder::new()
                    .description(Some("ID of the model to use"))
                    .build(),
            )
            .property(
                "input",
                ObjectBuilder::new()
                    .description(Some(
                        "Input text to embed, encoded as a string or array of strings",
                    ))
                    .build(),
            )
            .required("model")
            .required("input")
            .build(),
    ))
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
    use utoipa::openapi::schema::ObjectBuilder;

    RefOr::T(utoipa::openapi::schema::Schema::Object(
        ObjectBuilder::new()
            .property(
                "model",
                ObjectBuilder::new()
                    .description(Some("ID of the model to use"))
                    .build(),
            )
            .property(
                "input",
                ObjectBuilder::new()
                    .description(Some("The input text"))
                    .build(),
            )
            .required("model")
            .required("input")
            .build(),
    ))
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

/// Create router for OpenAPI documentation endpoints
pub fn openapi_router(route_docs: Vec<RouteDoc>, _path: Option<String>) -> (Vec<RouteDoc>, Router) {
    use utoipa_swagger_ui::SwaggerUi;

    // Generate the OpenAPI spec from route docs
    let openapi_spec = generate_openapi_spec(&route_docs);

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
}
