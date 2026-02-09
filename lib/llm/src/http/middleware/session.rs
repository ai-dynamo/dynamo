// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request session extraction middleware
//!
//! Extracts tenant_id and session_id from trusted headers provided by upstream
//! authentication/gateway service. Dynamo runs in a trusted environment and
//! does not perform authentication itself.

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};

/// Request session extracted from trusted upstream headers
///
/// # Deployment Context
/// Dynamo runs behind a VPN/private network. An upstream service handles:
/// - Authentication and authorization
/// - Assignment of tenant_id (identifies the customer/organization)
/// - Assignment of session_id (represents a conversation context)
///
/// All responses within a session share the same conversation history.
#[derive(Debug, Clone)]
pub struct RequestSession {
    /// Tenant identifier (from x-tenant-id header)
    ///
    /// Used for tenant isolation - different tenants cannot access
    /// each other's data.
    pub tenant_id: String,

    /// Session identifier (from x-session-id header)
    ///
    /// Represents a conversation context. All responses within a session
    /// share the same conversation history and can reference each other
    /// via previous_response_id.
    pub session_id: String,
}

/// Maximum allowed length for header values (tenant_id, session_id)
const MAX_HEADER_LENGTH: usize = 256;

/// Validate a header value: must be non-empty, at most MAX_HEADER_LENGTH chars,
/// and contain only alphanumeric characters, hyphens, underscores, and dots.
fn validate_header_value(value: &str, header_name: &str) -> Result<(), (StatusCode, String)> {
    if value.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("{header_name} must not be empty"),
        ));
    }
    if value.len() > MAX_HEADER_LENGTH {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("{header_name} exceeds maximum length of {MAX_HEADER_LENGTH} characters"),
        ));
    }
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "{header_name} contains invalid characters; \
                 only alphanumeric, hyphens, underscores, and dots are allowed"
            ),
        ));
    }
    Ok(())
}

/// Middleware to extract request session from headers
///
/// # Headers
/// - `x-tenant-id` (required): Tenant identifier
/// - `x-session-id` (required): Session/conversation identifier
///
/// # Validation
/// Both headers must be non-empty, at most 256 characters, and contain only
/// alphanumeric characters, hyphens, underscores, and dots (`^[a-zA-Z0-9._-]+$`).
///
/// # Errors
/// Returns 400 Bad Request if required headers are missing or invalid.
pub async fn extract_session_middleware(
    mut request: Request,
    next: Next,
) -> Result<Response, Response> {
    let headers = request.headers();

    // Extract and validate tenant_id (required)
    let tenant_id = headers
        .get("x-tenant-id")
        .ok_or_else(|| bad_request("Missing required header: x-tenant-id"))?
        .to_str()
        .map_err(|_| bad_request("x-tenant-id contains invalid characters"))?
        .to_string();

    validate_header_value(&tenant_id, "x-tenant-id")
        .map_err(|(code, msg)| (code, msg).into_response())?;

    // Extract and validate session_id (required)
    let session_id = headers
        .get("x-session-id")
        .ok_or_else(|| bad_request("Missing required header: x-session-id"))?
        .to_str()
        .map_err(|_| bad_request("x-session-id contains invalid characters"))?
        .to_string();

    validate_header_value(&session_id, "x-session-id")
        .map_err(|(code, msg)| (code, msg).into_response())?;

    // Insert context into request extensions for downstream handlers
    request.extensions_mut().insert(RequestSession {
        tenant_id,
        session_id,
    });

    Ok(next.run(request).await)
}

/// Helper to create a 400 Bad Request response with a message body
fn bad_request(msg: &str) -> Response {
    (StatusCode::BAD_REQUEST, msg.to_string()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
        middleware,
        response::IntoResponse,
        routing::get,
    };
    use tower::ServiceExt;

    async fn test_handler(
        axum::Extension(ctx): axum::Extension<RequestSession>,
    ) -> impl IntoResponse {
        format!("tenant={}, session={}", ctx.tenant_id, ctx.session_id)
    }

    fn create_test_router() -> Router {
        Router::new()
            .route("/test", get(test_handler))
            .layer(middleware::from_fn(extract_session_middleware))
    }

    #[tokio::test]
    async fn test_valid_headers_extracted() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "tenant_123")
            .header("x-session-id", "session_456")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        assert!(body_str.contains("tenant=tenant_123"));
        assert!(body_str.contains("session=session_456"));
    }

    #[tokio::test]
    async fn test_missing_tenant_id() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-session-id", "session_456")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_missing_session_id() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "tenant_123")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_empty_tenant_id() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "")
            .header("x-session-id", "session_456")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_empty_session_id() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "tenant_123")
            .header("x-session-id", "")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_tenant_id_too_long() {
        let app = create_test_router();
        let long_tenant = "a".repeat(257);

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", &long_tenant)
            .header("x-session-id", "session_456")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_session_id_too_long() {
        let app = create_test_router();
        let long_session = "b".repeat(257);

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "tenant_123")
            .header("x-session-id", &long_session)
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_tenant_id_invalid_chars() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "tenant/123")
            .header("x-session-id", "session_456")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_session_id_invalid_chars() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "tenant_123")
            .header("x-session-id", "session 456")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_valid_header_with_dots_and_hyphens() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "org-123.prod")
            .header("x-session-id", "sess_2024-01.test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_max_length_header_accepted() {
        let app = create_test_router();
        let max_tenant = "a".repeat(256);

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", &max_tenant)
            .header("x-session-id", "session_456")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
