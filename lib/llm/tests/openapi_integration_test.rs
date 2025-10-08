// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for OpenAPI documentation endpoints

#[cfg(test)]
mod tests {
    use axum::http::Method;
    use dynamo_llm::http::service::{RouteDoc, openapi_docs};

    #[test]
    fn test_openapi_router_creation() {
        // Create some sample route docs
        let route_docs = vec![
            RouteDoc::new(Method::POST, "/v1/chat/completions"),
            RouteDoc::new(Method::POST, "/v1/completions"),
            RouteDoc::new(Method::GET, "/v1/models"),
            RouteDoc::new(Method::GET, "/health"),
        ];

        // Create the OpenAPI router
        let (docs, _router) = openapi_docs::openapi_router(route_docs, None);

        // Verify that the OpenAPI docs include the expected routes
        assert_eq!(docs.len(), 2); // /openapi.json and /docs

        let paths: Vec<String> = docs.iter().map(|d| d.to_string()).collect();
        assert!(paths.iter().any(|p| p.contains("/openapi.json")));
        assert!(paths.iter().any(|p| p.contains("/docs")));
    }

    #[test]
    fn test_openapi_router_custom_path() {
        let route_docs = vec![RouteDoc::new(Method::POST, "/v1/chat/completions")];

        // Create the OpenAPI router with custom path
        let (docs, _router) =
            openapi_docs::openapi_router(route_docs, Some("/api/docs.json".to_string()));

        // Verify custom path is used
        let paths: Vec<String> = docs.iter().map(|d| d.to_string()).collect();
        assert!(paths.iter().any(|p| p.contains("/api/docs.json")));
    }
}
