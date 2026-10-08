// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use dynamo_decisions::protocols::{
    openai as decisions_oai, sglang as decisions_native, systemone as decisions_jev,
};
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
    ),
    components(
        schemas(
            crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest,
            crate::protocols::openai::completions::NvCreateCompletionRequest,
            crate::protocols::openai::embeddings::NvCreateEmbeddingRequest,
            crate::protocols::openai::responses::NvCreateResponse,
            decisions_jev::Request,
            decisions_jev::Response,
            decisions_oai::Request,
            decisions_oai::Response,
            decisions_native::Request,
            decisions_native::Response
        )
    )
)]
struct ApiDoc;

/// Generate OpenAPI specification from route documentation
///
/// This is the core helper used both by the embedded Swagger UI and by
/// external tools (for example CI) which need to materialize the
/// same frontend OpenAPI specification without running the HTTP service.
pub fn generate_openapi_spec(route_docs: &[RouteDoc]) -> utoipa::openapi::OpenApi {
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
        let operation = create_operation_for_route(method, route.documentation_path());

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

    if matches!(
        path,
        super::systemone::DEFAULT_PATH | super::systemone::DECISIONS_PATH
    ) {
        use utoipa::openapi::ContentBuilder;
        let schema = if path == super::systemone::DEFAULT_PATH {
            <decisions_jev::Response as utoipa::PartialSchema>::schema()
        } else {
            utoipa::openapi::Schema::OneOf(
                utoipa::openapi::schema::OneOfBuilder::new()
                    .item(<decisions_oai::Response as utoipa::PartialSchema>::schema())
                    .item(<decisions_native::Response as utoipa::PartialSchema>::schema())
                    .build(),
            )
            .into()
        };
        operation = operation.response(
            "200",
            ResponseBuilder::new()
                .description("Ordered typed answers with uncalibrated label probabilities")
                .content(
                    "application/json",
                    ContentBuilder::new().schema(Some(schema)).build(),
                )
                .build(),
        );
        for (status, description) in [
            ("413", "Request body exceeds 4 MiB"),
            ("429", "Decision scoring admission capacity is exhausted"),
            (
                "422",
                "Invalid question, unsupported control, or prompt/token budget",
            ),
            ("499", "Request cancelled"),
            ("500", "Malformed or incomplete native scoring response"),
            ("504", "Decision preflight or execution deadline exceeded"),
        ] {
            operation = operation.response(
                status,
                ResponseBuilder::new().description(description).build(),
            );
        }
    }

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

fn decisions_request_schema() -> RefOr<utoipa::openapi::schema::Schema> {
    use utoipa::openapi::schema::{
        AdditionalProperties, ObjectBuilder, OneOfBuilder, Schema, Type,
    };
    let variants = [
        (
            <decisions_oai::Request as utoipa::PartialSchema>::schema(),
            "oai",
            false,
        ),
        (
            <decisions_native::Request as utoipa::PartialSchema>::schema(),
            "sglang_native",
            true,
        ),
    ];
    let mut combined = OneOfBuilder::new();
    for (mut schema, format, required) in variants {
        let mut extension = ObjectBuilder::new()
            .additional_properties(Some(AdditionalProperties::FreeForm(false)))
            .property(
                "format",
                ObjectBuilder::new()
                    .schema_type(Type::String)
                    .enum_values(Some([format])),
            );
        if required {
            extension = extension.required("format");
        }
        if let RefOr::T(Schema::Object(object)) = &mut schema {
            object
                .properties
                .insert("nvext".into(), extension.build().into());
            if required {
                object.required.push("nvext".into());
            }
        }
        combined = combined.item(schema);
    }
    utoipa::openapi::Schema::OneOf(combined.build()).into()
}

/// Add request body schema for POST endpoints
fn add_request_body_for_path(
    operation: utoipa::openapi::path::OperationBuilder,
    path: &str,
) -> utoipa::openapi::path::OperationBuilder {
    use utoipa::openapi::ContentBuilder;
    use utoipa::openapi::request_body::RequestBodyBuilder;

    let (description, schema, example) = match path {
        "/v1/systemone" => (
            "Jev text evaluation; unsupported fields and controls are rejected",
            <decisions_jev::Request as utoipa::PartialSchema>::schema(),
            serde_json::json!({
                "model": "Qwen/Qwen3.8-27B", "state": "The payment failed.",
                "questions": {"route": {"type": "choice", "criteria": {
                    "billing": "Payment issues", "technical": "Integration failures"
                }}}
            }),
        ),
        "/v1/decisions" => (
            "OpenAI Decisions by default; nvext.format=sglang_native selects SGLang request and response schemas",
            decisions_request_schema(),
            serde_json::json!({
                "model":"Qwen/Qwen3.8-27B", "input":"The payment failed.",
                "questions":[{"type":"choice", "name":"route", "instructions":"Choose a team", "choices":[
                    {"value":"billing","description":"Payment issues"},
                    {"value":"technical","description":"Integration failures"}
                ]}]
            }),
        ),
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
    // Schema derived from actual NvCreateChatCompletionRequest type via ToSchema
    <crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest as utoipa::PartialSchema>::schema()
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
    <crate::protocols::openai::completions::NvCreateCompletionRequest as utoipa::PartialSchema>::schema()
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
        "/v1/systemone" => "Score System One questions".to_string(),
        "/v1/decisions" => "Evaluate bounded decision questions".to_string(),
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
        "/v1/systemone" => "Experimental aggregate SGLang API. Each question is scored in one prefill-only operation; answers contain probabilities normalized over the permitted labels, not calibrated probabilities of correctness.".to_string(),
        "/v1/decisions" => "Experimental text decisions over a qualified SGLang worker. OpenAI format is the default; nvext.format=sglang_native selects native validation and serialization. No streaming or image inputs are supported.".to_string(),
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
    fn systemone_openapi_describes_typed_requests_answers_and_errors() {
        let spec =
            generate_openapi_spec(&[RouteDoc::new(axum::http::Method::POST, "/v1/systemone")]);
        let value = serde_json::to_value(spec).unwrap();
        let operation = &value["paths"]["/v1/systemone"]["post"];
        assert_eq!(operation["summary"], "Score System One questions");
        assert!(operation["requestBody"]["content"]["application/json"]["schema"].is_object());
        assert!(operation["responses"]["200"]["content"]["application/json"]["schema"].is_object());
        for status in ["400", "404", "413", "422", "499", "500", "503", "529"] {
            assert!(
                operation["responses"].get(status).is_some(),
                "missing {status}"
            );
        }
        let schemas = &value["components"]["schemas"];
        assert!(schemas["DecisionSystemOneQuestion"]["oneOf"].is_array());
        assert!(schemas["DecisionSystemOneAnswer"]["oneOf"].is_array());
    }

    #[test]
    fn decisions_openapi_documents_selector_and_both_contracts() {
        let spec =
            generate_openapi_spec(&[RouteDoc::new(axum::http::Method::POST, "/v1/decisions")]);
        let value = serde_json::to_value(spec).unwrap();
        let operation = &value["paths"]["/v1/decisions"]["post"];
        let variants = operation["requestBody"]["content"]["application/json"]["schema"]["oneOf"]
            .as_array()
            .unwrap();
        assert_eq!(variants.len(), 2);
        assert_eq!(
            variants[0]["properties"]["nvext"]["properties"]["format"]["enum"],
            serde_json::json!(["oai"])
        );
        assert_eq!(
            variants[1]["properties"]["nvext"]["properties"]["format"]["enum"],
            serde_json::json!(["sglang_native"])
        );
        assert!(
            variants[1]["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("nvext"))
        );
        assert!(operation["responses"].get("429").is_some());
    }

    #[test]
    fn systemone_custom_path_preserves_its_schema() {
        let spec = generate_openapi_spec(&[RouteDoc::new(
            axum::http::Method::POST,
            "/experimental/score",
        )
        .with_documentation_path(super::super::systemone::DEFAULT_PATH)]);
        let value = serde_json::to_value(spec).unwrap();
        assert_eq!(
            value["paths"]["/experimental/score"]["post"]["summary"],
            "Score System One questions"
        );
        assert!(value["paths"].get("/v1/systemone").is_none());
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
