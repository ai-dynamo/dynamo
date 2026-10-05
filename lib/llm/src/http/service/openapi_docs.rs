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
use utoipa::OpenApi;
use utoipa::openapi::{PathItem, Paths, RefOr};

use crate::http::service::RouteDoc;
use crate::protocols::openai::compatibility::profile::Endpoint;

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
            crate::protocols::openai::compatibility::catalog::ModelCompatibilityCatalog,
            crate::http::service::openai::CompletionErrorResponse
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
        tracing::debug!("Adding route to OpenAPI spec: {}", route);
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
    openapi
}

/// Create an OpenAPI operation for a specific route
fn create_operation_for_route(route: &RouteDoc) -> utoipa::openapi::path::Operation {
    use utoipa::openapi::ResponseBuilder;
    use utoipa::openapi::path::OperationBuilder;

    let method = route.method().as_str();
    let path = route.path();
    let completion_endpoint = route
        .completion_endpoint
        .filter(|_| route.method() == axum::http::Method::POST);
    let description_path = completion_endpoint.map(Endpoint::as_str).unwrap_or(path);

    let operation_id = format!(
        "{}_{}",
        method.to_lowercase(),
        path.replace('/', "_").trim_matches('_')
    );
    let (summary, description) = if completion_endpoint.is_none()
        && matches!(path, "/v1/chat/completions" | "/v1/completions")
    {
        (
            format!("Endpoint: {path}"),
            format!("Endpoint for path: {path}"),
        )
    } else {
        (
            generate_summary_for_path(description_path),
            generate_description_for_path(description_path),
        )
    };

    let mut operation = OperationBuilder::new()
        .operation_id(Some(operation_id))
        .summary(Some(summary))
        .description(Some(description));

    if path.split('/').any(|segment| segment == "{model_id}") {
        operation = operation.parameter(
            utoipa::openapi::path::ParameterBuilder::new()
                .name("model_id")
                .parameter_in(utoipa::openapi::path::ParameterIn::Path)
                .required(utoipa::openapi::Required::True)
                .description(Some("Registered model ID, which may contain slashes. Exact registered names take precedence over diagnostic suffixes."))
                .schema(Some(utoipa::openapi::ObjectBuilder::new()
                    .schema_type(utoipa::openapi::schema::Type::String)))
                .build(),
        );
    }

    // Add request body for POST methods
    if method.to_uppercase() == "POST" {
        operation = add_request_body_for_path(operation, path, completion_endpoint);
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

    if method == "GET" && path.ends_with("/{model_id}/compatibility") {
        operation = operation
            .summary(Some("Inspect registered pipeline admission rules"))
            .description(Some("Dynamo-specific diagnostic snapshot, not routing eligibility or end-to-end conformance. Unlisted fields are not catalogued. Exact model names take precedence over this suffix."))
            .response("200", ResponseBuilder::new()
                .description("Partial admission catalog for a committed model, including unready pipelines")
                .content("application/json", utoipa::openapi::ContentBuilder::new()
                    .schema(Some(utoipa::openapi::Ref::from_schema_name("ModelCompatibilityCatalog")))
                    .build())
                .build());
    }

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

    if completion_endpoint.is_some() {
        for (code, description) in [
            ("400", "Invalid request"),
            ("404", "Model not found"),
            ("413", "Request body too large"),
            ("415", "Unsupported media type"),
            ("429", "Request deadline or rate limit"),
            ("499", "Request cancelled"),
            ("500", "Internal error"),
            ("501", "Unsupported capability"),
            ("503", "Service unavailable"),
            ("529", "Service overloaded"),
        ] {
            operation = operation.response(
                code,
                ResponseBuilder::new()
                    .description(description)
                    .content(
                        "application/json",
                        utoipa::openapi::ContentBuilder::new()
                            .schema(Some(utoipa::openapi::Ref::from_schema_name(
                                "CompletionErrorResponse",
                            )))
                            .build(),
                    )
                    .build(),
            );
        }
    }
    operation.build()
}

/// Add request body schema for POST endpoints
fn add_request_body_for_path(
    operation: utoipa::openapi::path::OperationBuilder,
    path: &str,
    completion_endpoint: Option<Endpoint>,
) -> utoipa::openapi::path::OperationBuilder {
    use utoipa::openapi::ContentBuilder;
    use utoipa::openapi::request_body::RequestBodyBuilder;

    let (description, schema, example) = match (completion_endpoint, path) {
        (Some(Endpoint::Chat), _) => (
            "Chat completion request with model, messages, and optional parameters",
            create_chat_completion_schema(),
            create_chat_completion_example(),
        ),
        (Some(Endpoint::Completion), _) => (
            "Text completion request with model, prompt, and optional parameters",
            create_completion_schema(),
            create_completion_example(),
        ),
        (None, "/v1/embeddings") => (
            "Embedding request with model and input text",
            create_embedding_schema(),
            create_embedding_example(),
        ),
        (None, "/v1/responses") => (
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
    fn completion_error_schema_is_nested_and_endpoint_scoped() {
        let routes = [
            RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions")
                .with_completion_endpoint(Endpoint::Chat),
            RouteDoc::new(axum::http::Method::POST, "/v1/completions")
                .with_completion_endpoint(Endpoint::Completion),
            RouteDoc::new(axum::http::Method::POST, "/v1/responses"),
        ];
        let document = serde_json::to_value(generate_openapi_spec(&routes)).unwrap();
        for path in ["/v1/chat/completions", "/v1/completions"] {
            assert_eq!(
                document["paths"][path]["post"]["responses"]["400"]["content"]["application/json"]
                    ["schema"]["$ref"],
                "#/components/schemas/CompletionErrorResponse"
            );
        }
        assert!(
            document["paths"]["/v1/responses"]["post"]["responses"]["400"]
                .get("content")
                .is_none()
        );
        let schemas = &document["components"]["schemas"];
        assert_eq!(
            schemas["CompletionErrorResponse"]["properties"]["error"]["$ref"],
            "#/components/schemas/CompletionErrorInfo"
        );
        for field in ["message", "type", "param", "code", "details"] {
            assert!(
                schemas["CompletionErrorInfo"]["properties"]
                    .get(field)
                    .is_some()
            );
        }
    }

    #[test]
    fn chat_only_controls_belong_to_the_chat_endpoint_openapi_contract() {
        use serde_json::Value;

        fn has_property(document: &Value, schema: &Value, field: &str) -> bool {
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
                let pointer = reference.strip_prefix('#').expect("local schema reference");
                return has_property(document, document.pointer(pointer).unwrap(), field);
            }
            schema
                .get("properties")
                .is_some_and(|properties| properties.get(field).is_some())
                || ["allOf", "anyOf", "oneOf"].iter().any(|composition| {
                    schema
                        .get(*composition)
                        .and_then(Value::as_array)
                        .is_some_and(|schemas| {
                            schemas
                                .iter()
                                .any(|item| has_property(document, item, field))
                        })
                })
        }

        let routes = [
            RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions")
                .with_completion_endpoint(Endpoint::Chat),
            RouteDoc::new(axum::http::Method::POST, "/v1/completions")
                .with_completion_endpoint(Endpoint::Completion),
        ];
        // This is the same generator used by /openapi.json, including path-to-schema references.
        let document = serde_json::to_value(generate_openapi_spec(&routes)).unwrap();
        for (path, expected) in [("/v1/chat/completions", true), ("/v1/completions", false)] {
            let schema = &document["paths"][path]["post"]["requestBody"]["content"]["application/json"]
                ["schema"];
            assert!(!schema.is_null(), "missing request schema for {path}");
            for field in ["add_generation_prompt", "continue_final_message"] {
                assert_eq!(
                    has_property(&document, schema, field),
                    expected,
                    "{path}: {field}"
                );
            }
        }
    }

    #[test]
    fn test_generate_openapi_spec() {
        let routes = vec![
            RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions")
                .with_completion_endpoint(Endpoint::Chat),
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

    #[test]
    fn completion_schemas_require_handler_identity_and_post_method() {
        for path in [
            "/v1/chat/completions",
            "/v1/completions",
            "/other/chat/completions",
        ] {
            let routes = [
                RouteDoc::new(axum::http::Method::POST, path),
                // Even marked metadata cannot give a GET the POST handler's contract.
                RouteDoc::new(axum::http::Method::GET, path)
                    .with_completion_endpoint(Endpoint::Chat),
            ];
            let document = serde_json::to_value(generate_openapi_spec(&routes)).unwrap();
            for method in ["get", "post"] {
                let operation = &document["paths"][path][method];
                assert!(
                    operation["requestBody"]["content"]["application/json"]["schema"].is_null()
                );
                assert!(operation["responses"]["400"].get("content").is_none());
            }
        }
    }

    #[tokio::test]
    async fn served_completion_schemas_follow_registered_handlers_at_custom_paths() {
        use crate::http::service::{openai, service_v2::HttpService};
        use serde_json::{Value, json};

        let service = HttpService::builder().build().unwrap();
        let state = service.state_clone();
        let mut app = Router::new();
        let mut docs = Vec::new();
        let cases = [
            (Endpoint::Chat, None),
            (Endpoint::Completion, None),
            (Endpoint::Chat, Some("/tenant/generate")),
            // Deliberately resembles chat: the text-completion handler wins.
            (Endpoint::Completion, Some("/tenant/chat/completions")),
        ];
        for (endpoint, path) in cases {
            let (route_docs, router) = match endpoint {
                Endpoint::Chat => {
                    openai::chat_completions_router(state.clone(), None, path.map(str::to_owned))
                }
                Endpoint::Completion => {
                    openai::completions_router(state.clone(), path.map(str::to_owned))
                }
            };
            docs.extend(route_docs);
            app = app.merge(router);
        }
        app = app.merge(openapi_router(docs, None).1);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task = tokio::spawn(async move { axum::serve(listener, app).await });
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();
        let document: Value = client
            .get(format!("http://{addr}/openapi.json"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let mut responses = Vec::new();
        for (endpoint, path) in cases {
            let path = path.unwrap_or(endpoint.as_str());
            let mut payload = json!({"model":"missing", "prompt_logprobs":-2});
            match endpoint {
                Endpoint::Chat => payload["messages"] = json!([{"role":"user","content":"hello"}]),
                Endpoint::Completion => payload["prompt"] = json!("hello"),
            };
            let response = client
                .post(format!("http://{addr}{path}"))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            responses.push((
                endpoint,
                path,
                status,
                response.json::<Value>().await.unwrap(),
            ));
        }
        task.abort();
        let _ = task.await;

        for (endpoint, path, status, response) in responses {
            assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
            assert_eq!(response["error"]["code"], 400);
            assert_eq!(response["error"]["param"], "prompt_logprobs");
            let operation = &document["paths"][path]["post"];
            let canonical = &document["paths"][endpoint.as_str()]["post"];
            for field in ["requestBody", "responses", "summary", "description"] {
                assert_eq!(operation[field], canonical[field], "{path}: {field}");
            }
            assert_eq!(
                operation["summary"],
                match endpoint {
                    Endpoint::Chat => "Create chat completion",
                    Endpoint::Completion => "Create text completion",
                }
            );
            assert!(operation["requestBody"]["content"]["application/json"]["schema"].is_object());
            assert_eq!(
                operation["responses"]["400"]["content"]["application/json"]["schema"]["$ref"],
                "#/components/schemas/CompletionErrorResponse"
            );
            assert!(operation.get("parameters").is_none());
        }
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
