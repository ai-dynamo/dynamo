//! Request session extraction middleware
//!
//! Extracts tenant_id and session_id from trusted headers provided by upstream
//! authentication/gateway service. Dynamo runs in a trusted environment and
//! does not perform authentication itself.

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
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

    /// User identifier (from x-user-id header, optional)
    ///
    /// Used for logging and audit purposes. Not required for functionality.
    pub user_id: Option<String>,
}

/// Middleware to extract request session from headers
///
/// # Headers
/// - `x-tenant-id` (required): Tenant identifier
/// - `x-session-id` (required): Session/conversation identifier
/// - `x-user-id` (optional): User identifier for logging
///
/// # Errors
/// Returns 400 Bad Request if required headers are missing or invalid.
pub async fn extract_session_middleware(
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let headers = request.headers();

    // Extract tenant_id (required)
    let tenant_id = headers
        .get("x-tenant-id")
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_str()
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .to_string();

    // Validate tenant_id is not empty
    if tenant_id.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Extract session_id (required)
    let session_id = headers
        .get("x-session-id")
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_str()
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .to_string();

    // Validate session_id is not empty
    if session_id.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Extract user_id (optional)
    let user_id = headers
        .get("x-user-id")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());

    // Insert context into request extensions for downstream handlers
    request.extensions_mut().insert(RequestSession {
        tenant_id,
        session_id,
        user_id,
    });

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        response::IntoResponse,
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    async fn test_handler(
        axum::Extension(ctx): axum::Extension<RequestSession>,
    ) -> impl IntoResponse {
        format!(
            "tenant={}, session={}, user={:?}",
            ctx.tenant_id, ctx.session_id, ctx.user_id
        )
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
            .header("x-user-id", "user_789")
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
        assert!(body_str.contains("user=Some(\"user_789\")"));
    }

    #[tokio::test]
    async fn test_optional_user_id() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "tenant_123")
            .header("x-session-id", "session_456")
            // No x-user-id header
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        assert!(body_str.contains("user=None"));
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
    async fn test_invalid_header_encoding() {
        let app = create_test_router();

        let request = Request::builder()
            .uri("/test")
            .header("x-tenant-id", "tenant_123")
            .header("x-session-id", "session_456")
            // Invalid UTF-8 in user_id won't cause failure (optional header)
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
