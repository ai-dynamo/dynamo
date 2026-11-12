// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_runtime::discovery::{MockDiscovery, SharedMockRegistry};
use std::collections::HashSet;

/// A worker's session API: sessions are created under the body's `id`, and
/// every answer names the worker.
async fn session_worker(name: &'static str) -> String {
    use axum::extract::Path;
    use axum::routing::{get, post};

    let sessions = Arc::new(parking_lot::Mutex::new(HashSet::<String>::new()));
    let held = move |sessions: &parking_lot::Mutex<HashSet<String>>, id: &str| {
        if sessions.lock().contains(id) {
            (
                StatusCode::OK,
                axum::Json(serde_json::json!({"id": id, "worker": name})),
            )
        } else {
            (
                StatusCode::NOT_FOUND,
                axum::Json(serde_json::json!({"worker": name})),
            )
        }
    };
    let app = axum::Router::new()
        .route(
            "/v1/sessions",
            post({
                let sessions = sessions.clone();
                move |body: Bytes| async move {
                    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                    let id = body["id"]
                        .as_str()
                        .expect("the frontend names every session");
                    if !sessions.lock().insert(id.to_string()) {
                        return (
                            StatusCode::CONFLICT,
                            axum::Json(serde_json::json!({"worker": name})),
                        );
                    }
                    (
                        StatusCode::OK,
                        axum::Json(serde_json::json!({"id": id, "worker": name})),
                    )
                }
            }),
        )
        .route(
            "/v1/sessions/{id}",
            get({
                let sessions = sessions.clone();
                move |Path(id): Path<String>| async move { held(&sessions, &id) }
            })
            .delete({
                let sessions = sessions.clone();
                move |Path(id): Path<String>| async move {
                    let answer = held(&sessions, &id);
                    sessions.lock().remove(&id);
                    answer
                }
            }),
        )
        .route(
            "/v1/sessions/{id}/warm",
            post({
                let sessions = sessions.clone();
                move |Path(id): Path<String>| async move { held(&sessions, &id) }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

/// Registers worker `id`'s model card in `namespace`, declaring `prefixes`
/// at `url`.
async fn register(
    registry: &SharedMockRegistry,
    id: u64,
    namespace: &str,
    url: &str,
    prefixes: &[&str],
) -> (MockDiscovery, DiscoveryInstance) {
    let discovery = MockDiscovery::new(Some(id), registry.clone());
    let card = serde_json::json!({
        "display_name": "model",
        "runtime_config": {"runtime_data": {HTTP_ROUTES_KEY: {"url": url, "prefixes": prefixes}}},
    });
    let spec = dynamo_runtime::discovery::DiscoverySpec::from_model(
        namespace.to_string(),
        "backend".to_string(),
        "generate".to_string(),
        &card,
    )
    .unwrap();
    let instance = discovery.register(spec).await.unwrap();
    (discovery, instance)
}

/// Registers a session worker named `name` as worker `id`.
async fn sessions(
    registry: &SharedMockRegistry,
    id: u64,
    namespace: &str,
    name: &'static str,
) -> (MockDiscovery, DiscoveryInstance) {
    let url = session_worker(name).await;
    register(registry, id, namespace, &url, &["/v1/sessions"]).await
}

/// A frontend's backend routes, once `workers` workers declare `/v1/sessions`.
async fn frontend(
    registry: &SharedMockRegistry,
    namespaces: NamespaceFilter,
    workers: usize,
) -> BackendRoutes {
    let routes = BackendRoutes::new(WorkerDiscovery {
        discovery: Arc::new(MockDiscovery::new(None, registry.clone())),
        namespaces,
        cancel: CancellationToken::new(),
    })
    .unwrap();
    wait_for_workers(&routes, workers).await;
    routes
}

async fn wait_for_workers(routes: &BackendRoutes, count: usize) {
    tokio::time::timeout(
        Duration::from_secs(5),
        routes.wait_for_workers("/v1/sessions", count),
    )
    .await
    .expect("the frontend sees the workers");
}

async fn call(
    routes: &BackendRoutes,
    method: &str,
    path: &str,
    body: &str,
) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .body(Body::from(body.to_string()))
        .unwrap();
    let target = routes
        .resolve(request.uri())
        .expect("a route covers the request");
    let response = routes
        .forward(request, target, CancellationToken::new(), ())
        .await;
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap_or_default())
}

#[tokio::test]
async fn requests_follow_the_session_to_its_worker() {
    let registry = SharedMockRegistry::new();
    sessions(&registry, 1, "ns", "a").await;
    sessions(&registry, 2, "ns", "b").await;
    let first = frontend(&registry, NamespaceFilter::Global, 2).await;
    let second = frontend(&registry, NamespaceFilter::Global, 2).await;

    // Created without an id: the frontend mints one, so it knows the worker.
    let (status, created) = call(&first, "POST", "/v1/sessions", "").await;
    assert_eq!(status, StatusCode::OK);
    let id = created["id"].as_str().unwrap().to_string();
    assert!(
        id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()),
        "{id}"
    );
    let owner = created["worker"].clone();
    for _ in 0..3 {
        let (status, warmed) = call(&first, "POST", &format!("/v1/sessions/{id}/warm"), "").await;
        assert_eq!((status, &warmed["worker"]), (StatusCode::OK, &owner));
    }
    // Another frontend asks the workers.
    let (status, warmed) = call(&second, "POST", &format!("/v1/sessions/{id}/warm"), "").await;
    assert_eq!((status, &warmed["worker"]), (StatusCode::OK, &owner));

    // A chosen id some worker holds goes to that worker, whichever frontend creates it.
    let (status, chosen) = call(&first, "POST", "/v1/sessions", r#"{"id": "agent-1"}"#).await;
    assert_eq!(status, StatusCode::OK);
    let (status, again) = call(&second, "POST", "/v1/sessions", r#"{"id": "agent-1"}"#).await;
    assert_eq!(
        (status, &again["worker"]),
        (StatusCode::CONFLICT, &chosen["worker"])
    );

    // Deleted through one frontend, gone through the other, which held its worker.
    let (status, _) = call(&second, "DELETE", &format!("/v1/sessions/{id}"), "").await;
    assert_eq!(status, StatusCode::OK);
    for _ in 0..2 {
        let (status, _) = call(&first, "GET", &format!("/v1/sessions/{id}"), "").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
    let (status, missing) = call(&second, "GET", "/v1/sessions/nope", "").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(missing["message"], "no worker holds this resource");

    let (status, _) = call(&first, "GET", "/v1/sessions", "").await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    for bad in [r#"{"id": 7}"#, r#"{"id": "a/b"}"#, "[]"] {
        let (status, _) = call(&first, "POST", "/v1/sessions", bad).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
    }
}

#[tokio::test]
async fn new_sessions_spread_over_the_workers() {
    let registry = SharedMockRegistry::new();
    sessions(&registry, 1, "ns", "a").await;
    sessions(&registry, 2, "ns", "b").await;
    let routes = frontend(&registry, NamespaceFilter::Global, 2).await;

    let mut workers = HashSet::new();
    for _ in 0..32 {
        let (status, created) = call(&routes, "POST", "/v1/sessions", "").await;
        assert_eq!(status, StatusCode::OK);
        let id = created["id"].as_str().unwrap();
        let (status, warmed) = call(&routes, "POST", &format!("/v1/sessions/{id}/warm"), "").await;
        assert_eq!(
            (status, &warmed["worker"]),
            (StatusCode::OK, &created["worker"])
        );
        workers.insert(created["worker"].to_string());
    }
    assert_eq!(workers.len(), 2, "{workers:?}");
}

#[tokio::test]
async fn a_session_whose_worker_left_is_not_found() {
    let registry = SharedMockRegistry::new();
    let a = sessions(&registry, 1, "ns", "a").await;
    let b = sessions(&registry, 2, "ns", "b").await;
    let routes = frontend(&registry, NamespaceFilter::Global, 2).await;

    let (status, created) = call(&routes, "POST", "/v1/sessions", "").await;
    assert_eq!(status, StatusCode::OK);
    let id = created["id"].as_str().unwrap().to_string();
    let (discovery, card) = if created["worker"] == "a" { a } else { b };
    discovery.unregister(card).await.unwrap();
    wait_for_workers(&routes, 1).await;

    let (status, _) = call(&routes, "POST", &format!("/v1/sessions/{id}/warm"), "").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_worker_that_cannot_answer_makes_a_lookup_unavailable() {
    let registry = SharedMockRegistry::new();
    sessions(&registry, 1, "ns", "a").await;
    let routes = frontend(&registry, NamespaceFilter::Global, 1).await;
    let (status, created) = call(&routes, "POST", "/v1/sessions", "").await;
    assert_eq!(status, StatusCode::OK);
    let id = created["id"].as_str().unwrap().to_string();

    register(&registry, 2, "ns", "http://127.0.0.1:1", &["/v1/sessions"]).await;
    let other = frontend(&registry, NamespaceFilter::Global, 2).await;
    // The owner answers, so the silent worker does not matter.
    let (status, warmed) = call(&other, "POST", &format!("/v1/sessions/{id}/warm"), "").await;
    assert_eq!(
        (status, &warmed["worker"]),
        (StatusCode::OK, &created["worker"])
    );
    // Without an owner, the silent worker might hold it: not a 404.
    let (status, _) = call(&other, "GET", "/v1/sessions/nope", "").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn only_workers_in_the_served_namespaces_take_sessions() {
    let registry = SharedMockRegistry::new();
    sessions(&registry, 1, "ns", "a").await;
    sessions(&registry, 2, "other", "b").await;
    let routes = frontend(&registry, NamespaceFilter::Exact("ns".to_string()), 1).await;
    for _ in 0..8 {
        let (status, created) = call(&routes, "POST", "/v1/sessions", "").await;
        assert_eq!(
            (status, &created["worker"]),
            (StatusCode::OK, &serde_json::json!("a"))
        );
    }
}

#[tokio::test]
async fn only_declared_prefixes_are_forwarded() {
    let registry = SharedMockRegistry::new();
    let none = frontend(&registry, NamespaceFilter::Global, 0).await;
    assert!(none.resolve(&Uri::from_static("/v1/sessions")).is_none());

    register(
        &registry,
        1,
        "ns",
        "http://worker:1",
        &[
            "/v1/sessions/",
            "/v1/sessions/admin",
            "/",
            "/v1/../x",
            "/v1/a b",
        ],
    )
    .await;
    let routes = frontend(&registry, NamespaceFilter::Global, 1).await;
    let prefix = |path: &'static str| routes.resolve(&Uri::from_static(path)).map(|t| t.0);
    assert_eq!(prefix("/v1/sessions").as_deref(), Some("/v1/sessions"));
    assert_eq!(
        prefix("/v1/sessions/s1/events").as_deref(),
        Some("/v1/sessions")
    );
    assert_eq!(
        prefix("/v1/sessions/admin/x").as_deref(),
        Some("/v1/sessions/admin")
    );
    for path in [
        "/v1/sessionsx",
        "/v1/models",
        "/x",
        "/v1/sessions/../metrics",
        "/v1/sessions/%2E%2e/x",
        "/v1/sessions/./x",
        "/v1/sessions/.%2e/x",
        "/v1/sessions/%2e./x",
        "/v1/sessions/%2e/x",
        "/v1/sessions/a\\b",
        "/v1/sessions/a%2f..%2fmetrics",
        "/v1/sessions/a%5C..%5Cmetrics",
        "/v1/sessions/a%2F%2e%2E%2Fmetrics",
    ] {
        assert_eq!(prefix(path), None, "{path} must not be forwarded");
    }
}
