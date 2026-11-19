// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::engine_auth::{EngineControlAuth, authorize};
use axum::{
    Json, Router,
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode},
    middleware,
    routing::{any, get},
};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tower::ServiceExt;

const TOKEN: &str = "example-control-token-with-at-least-32-bytes";

fn router(secret: Option<&str>, calls: Arc<AtomicUsize>) -> Router {
    Router::new()
        .route(
            "/engine/{*path}",
            any(move |Json(body): Json<Value>| async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Json(body)
            }),
        )
        .route_layer(middleware::from_fn_with_state(
            EngineControlAuth::from_secret(secret).unwrap(),
            authorize,
        ))
        .route("/health", get(|| async { StatusCode::OK }))
}

#[tokio::test]
async fn rejects_unauthorized_controls_before_dispatch_without_echoing_credentials() {
    for supplied in [None, Some("Bearer wrong"), Some(TOKEN)] {
        let calls = Arc::new(AtomicUsize::new(0));
        let app = router(Some(TOKEN), calls.clone());
        let mut request = Request::post("/engine/update/update_weights")
            .header("content-type", "application/json");
        if let Some(value) = supplied {
            request = request.header("authorization", value);
        }
        let response = app
            .oneshot(request.body(Body::from(r#"{"update_info":{}}"#)).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()["www-authenticate"], "Bearer");
        let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(!text.contains(TOKEN));
        assert!(!text.contains("wrong"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn authenticates_controls_and_preserves_the_body() {
    let calls = Arc::new(AtomicUsize::new(0));
    let body = json!({"session_id": "session-a", "seq": 1, "update_info": {"names": ["weight"]}});
    let response = router(Some(TOKEN), calls.clone())
        .oneshot(
            Request::post("/engine/update/update_weights")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {TOKEN}"))
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), body);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn rejects_duplicate_authorization_headers() {
    let calls = Arc::new(AtomicUsize::new(0));
    let response = router(Some(TOKEN), calls.clone())
        .oneshot(
            Request::post("/engine/start_weight_update")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {TOKEN}"))
                .header("authorization", format!("Bearer {TOKEN}"))
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn leaves_health_checks_public_and_preserves_unconfigured_controls() {
    let calls = Arc::new(AtomicUsize::new(0));
    let response = router(Some(TOKEN), calls.clone())
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let response = router(None, calls.clone())
        .oneshot(
            Request::post("/engine/get_weight_version")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn configured_control_secret_must_be_strong_and_header_safe() {
    for secret in [
        "",
        "short",
        "a secret with spaces that is long enough",
        "invalid\nheader-value-that-is-long-enough",
    ] {
        let error = EngineControlAuth::from_secret(Some(secret)).err().unwrap();
        assert!(!error.to_string().contains(secret) || secret.is_empty());
    }
    assert!(EngineControlAuth::from_secret(None).is_ok());
    assert!(EngineControlAuth::from_secret(Some(TOKEN)).is_ok());
}

#[tokio::test]
async fn actual_system_listener_authenticates_legacy_and_typed_routes() {
    use crate::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::actual_system_listener_authenticates_legacy_and_typed_routes"
        ),
        &[],
    ) {
        return;
    }
    for secret in [Some(TOKEN), None] {
        temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_HOST", Some("127.0.0.1")),
                ("DYN_SYSTEM_PORT", Some("0")),
                ("DYN_HEALTH_CHECK_ENABLED", Some("false")),
                ("DYN_SYSTEM_ENGINE_AUTH_TOKEN", secret),
                ("DYN_LORA_ENABLED", Some("true")),
            ],
            async {
                let runtime = Runtime::from_current().unwrap();
                let drt = DistributedRuntime::new_with_probe_policy(
                    runtime.clone(),
                    DistributedConfig::process_local(),
                    super::SystemProbePolicy::RuntimeOnly,
                )
                .await
                .unwrap();
                let info = drt.system_status_server_info().unwrap();
                let base = format!("http://{}", info.socket_addr);
                let client = reqwest::Client::builder()
                    .no_proxy()
                    .timeout(std::time::Duration::from_secs(3))
                    .build()
                    .unwrap();
                for (method, path) in [
                    (Method::POST, "/engine/update/update_weights"),
                    (Method::POST, "/engine/update_weights"),
                    (Method::POST, "/engine/collective_rpc"),
                    (Method::GET, "/v1/loras"),
                    (Method::POST, "/v1/loras"),
                    (Method::DELETE, "/v1/loras/adapter"),
                ] {
                    let response = client
                        .request(method.clone(), format!("{base}{path}"))
                        .body("not-json")
                        .send()
                        .await
                        .unwrap();
                    assert_eq!(
                        response.status().as_u16() == 401,
                        secret.is_some(),
                        "{method} {path}"
                    );
                    let response = client
                        .request(method, format!("{base}{path}"))
                        .bearer_auth(TOKEN)
                        .body("not-json")
                        .send()
                        .await
                        .unwrap();
                    assert_ne!(response.status().as_u16(), 401);
                }
                assert_eq!(
                    client
                        .get(format!("{base}/live"))
                        .send()
                        .await
                        .unwrap()
                        .status()
                        .as_u16(),
                    200
                );
                runtime.shutdown();
            },
        )
        .await;
    }
}
