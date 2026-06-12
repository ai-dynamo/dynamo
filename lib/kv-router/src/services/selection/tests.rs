// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use axum::response::Response;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

use super::*;

fn test_config() -> crate::config::KvRouterConfig {
    let mut config = crate::config::KvRouterConfig::default();
    config.use_kv_events = false;
    config.router_queue_threshold = None;
    config
}

fn app() -> Router {
    let core = Arc::new(SelectionCore::new(
        test_config(),
        1,
        CancellationToken::new(),
    ));
    create_router(Arc::new(AppState { core }))
}

async fn response_json(response: Response) -> serde_json::Value {
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response body");
    serde_json::from_slice(&body).expect("response JSON")
}

async fn post(app: Router, uri: &str, body: &str) -> Response {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn register_worker(app: Router, max_tokens: Option<u64>) -> Response {
    register_worker_id(app, 1, max_tokens).await
}

async fn register_worker_id(app: Router, worker_id: u64, max_tokens: Option<u64>) -> Response {
    let mut body = serde_json::json!({
        "worker_id": worker_id,
        "model_name": "model",
        "endpoint": format!("http://worker-{worker_id}:8000"),
        "block_size": 4
    });
    if let Some(max_tokens) = max_tokens {
        body["max_num_batched_tokens"] = serde_json::json!(max_tokens);
    }
    post(app, "/workers", &body.to_string()).await
}

#[tokio::test]
async fn ready_and_select_report_not_ready_without_schedulable_workers() {
    let app = app();
    let ready_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready_response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let select_response = post(
        app,
        "/select",
        r#"{"model_name":"model","token_ids":[1,2,3,4],"selection_id":"s1"}"#,
    )
    .await;
    assert_eq!(select_response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn incomplete_worker_is_accepted_but_not_schedulable() {
    let mut config = test_config();
    config.router_queue_threshold = Some(1.0);
    let core = Arc::new(SelectionCore::new(config, 1, CancellationToken::new()));
    let app = create_router(Arc::new(AppState { core }));

    let response = post(
        app.clone(),
        "/workers",
        r#"{"worker_id":1,"model_name":"model","endpoint":"http://worker-1:8000","block_size":4}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = response_json(response).await;
    assert_eq!(body["lifecycle"], "incomplete");
    assert!(
        body["not_schedulable_reasons"][0]
            .as_str()
            .unwrap()
            .contains("max_num_batched_tokens")
    );

    let select_response = post(
        app,
        "/select",
        r#"{"model_name":"model","token_ids":[1,2,3,4]}"#,
    )
    .await;
    assert_eq!(select_response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn select_echoes_selection_id_and_does_not_book_load() {
    let app = app();
    assert_eq!(
        register_worker(app.clone(), None).await.status(),
        StatusCode::CREATED
    );

    let response = post(
        app.clone(),
        "/select",
        r#"{"model_name":"model","token_ids":[1,2,3,4],"selection_id":"sel-a"}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["selection_id"], "sel-a");
    assert_eq!(body["worker_id"], 1);

    let loads_response = app
        .oneshot(
            Request::builder()
                .uri("/loads")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(loads_response.status(), StatusCode::OK);
    let loads = response_json(loads_response).await;
    assert_eq!(loads[0]["loads"][0]["active_requests"], 0);
}

#[tokio::test]
async fn select_and_reserve_books_and_duplicate_reservation_conflicts() {
    let app = app();
    assert_eq!(
        register_worker(app.clone(), None).await.status(),
        StatusCode::CREATED
    );

    let response = post(
        app.clone(),
        "/select_and_reserve",
        r#"{"model_name":"model","token_ids":[1,2,3,4],"reservation_id":"res-a"}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["reservation_id"], "res-a");

    let duplicate = post(
        app.clone(),
        "/select_and_reserve",
        r#"{"model_name":"model","token_ids":[1,2,3,4],"reservation_id":"res-a"}"#,
    )
    .await;
    assert_eq!(duplicate.status(), StatusCode::CONFLICT);

    let free = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/reservations/res-a")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(free.status(), StatusCode::OK);
}

#[tokio::test]
async fn explicit_reservation_books_after_select() {
    let app = app();
    assert_eq!(
        register_worker(app.clone(), None).await.status(),
        StatusCode::CREATED
    );

    let select_response = post(
        app.clone(),
        "/select",
        r#"{"model_name":"model","token_ids":[1,2,3,4]}"#,
    )
    .await;
    assert_eq!(select_response.status(), StatusCode::OK);
    let selected = response_json(select_response).await;

    let reservation = serde_json::json!({
        "model_name": "model",
        "reservation_id": "res-b",
        "worker_id": selected["worker_id"],
        "dp_rank": selected["dp_rank"],
        "sequence_hashes": [1],
        "isl_tokens": 4
    });
    let reserve_response = post(app.clone(), "/reservations", &reservation.to_string()).await;
    assert_eq!(reserve_response.status(), StatusCode::CREATED);

    let prefill_response = post(app, "/reservations/res-b/prefill_complete", "{}").await;
    assert_eq!(prefill_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn explicit_reservation_rejects_unschedulable_worker() {
    let app = app();
    let incomplete = post(
        app.clone(),
        "/workers",
        r#"{"worker_id":1,"model_name":"model","block_size":4}"#,
    )
    .await;
    assert_eq!(incomplete.status(), StatusCode::CREATED);
    assert_eq!(response_json(incomplete).await["lifecycle"], "incomplete");
    assert_eq!(
        register_worker_id(app.clone(), 2, None).await.status(),
        StatusCode::CREATED
    );

    let reservation = serde_json::json!({
        "model_name": "model",
        "reservation_id": "res-unschedulable",
        "worker_id": 1,
        "sequence_hashes": [1],
        "isl_tokens": 4
    });
    let reserve_response = post(app, "/reservations", &reservation.to_string()).await;
    assert_eq!(reserve_response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn reconcile_rolls_back_partial_listener_registration() {
    let mut config = test_config();
    config.use_kv_events = true;
    let core = Arc::new(SelectionCore::new(config, 1, CancellationToken::new()));
    let app = create_router(Arc::new(AppState { core }));

    let response = post(
        app.clone(),
        "/workers",
        r#"{
            "worker_id": 7,
            "model_name": "model",
            "endpoint": "http://worker-7:8000",
            "block_size": 4,
            "data_parallel_size": 2,
            "kv_events_endpoints": {
                "0": "tcp://127.0.0.1:5557",
                "1": "not-a-zmq-endpoint"
            }
        }"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = response_json(response).await;
    assert_eq!(body["lifecycle"], "incomplete");
    assert!(
        body["not_schedulable_reasons"][0]
            .as_str()
            .unwrap()
            .contains("reconciliation failed")
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/workers/7")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "kv_events_endpoints": {
                            "0": "tcp://127.0.0.1:5557",
                            "1": "tcp://127.0.0.1:5558"
                        }
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["lifecycle"], "schedulable");
}

#[tokio::test]
async fn hash_path_validation_returns_bad_request() {
    let app = app();
    assert_eq!(
        register_worker(app.clone(), None).await.status(),
        StatusCode::CREATED
    );

    let response = post(
        app,
        "/select",
        r#"{"model_name":"model","block_hashes":[1],"sequence_hashes":[1,2],"isl_tokens":4}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
