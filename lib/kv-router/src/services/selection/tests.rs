// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use axum::response::Response;
use std::net::TcpListener as StdTcpListener;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use super::input::{MmRoutingInfoRequest, PromptRequest};
use super::server::create_router;
use super::*;
use crate::indexer::{LowerTierMatchDetails, MatchDetails, TieredMatchDetails};
use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, BlockMmObjectInfo, OverlapScores, StorageTier,
    WorkerWithDpRank, compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use crate::scheduling::config::RouterConfigOverride;
use crate::scheduling::overlap::build_overlap_scores_response;

fn test_config() -> crate::config::KvRouterConfig {
    crate::config::KvRouterConfig {
        use_kv_events: false,
        router_queue_threshold: None,
        ..Default::default()
    }
}

fn app() -> Router {
    let service = Arc::new(SelectionService::new_local_for_test(test_config(), 1));
    create_router(Arc::new(AppState { service }))
}

async fn response_json(response: Response) -> serde_json::Value {
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response body");
    serde_json::from_slice(&body).expect("response JSON")
}

#[tokio::test]
async fn replica_sync_routes_are_mounted() {
    let response = app()
        .oneshot(
            Request::builder()
                .uri("/replica_sync/peers")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

async fn post(app: Router, uri: &str, body: &str) -> Response {
    post_with_policy_class(app, uri, body, None).await
}

async fn post_with_policy_class(
    app: Router,
    uri: &str,
    body: &str,
    policy_class: Option<&str>,
) -> Response {
    let mut request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(policy_class) = policy_class {
        request = request.header("x-dynamo-meta-policy-class", policy_class);
    }
    app.oneshot(request.body(Body::from(body.to_string())).unwrap())
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

#[test]
fn prompt_normalization_uses_mm_routing_info_and_eagle_hashing() {
    let mm_infos = vec![Some(BlockExtraInfo {
        mm_objects: vec![BlockMmObjectInfo {
            mm_hash: 42,
            offsets: vec![(0, 2)],
        }],
    })];
    let request = PromptRequest {
        token_ids: Some(vec![1, 2, 3, 4]),
        mm_routing_info: Some(MmRoutingInfoRequest {
            routing_token_ids: vec![10, 11, 12, 13, 14, 15, 16, 17],
            block_mm_infos: mm_infos.clone(),
        }),
        block_mm_infos: None,
        block_hashes: None,
        sequence_hashes: None,
        isl_tokens: None,
        lora_name: Some("adapter".to_string()),
        cache_namespace: Some("tenant-a".to_string()),
        is_eagle: Some(true),
    };

    let normalized = request
        .normalize_for_selection(4, false)
        .expect("normalize prompt");
    let expected_block_hashes = compute_block_hash_for_seq(
        &[10, 11, 12, 13, 14, 15, 16, 17],
        4,
        BlockHashOptions {
            block_mm_infos: Some(&mm_infos),
            lora_name: Some("adapter"),
            cache_namespace: Some("tenant-a"),
            is_eagle: Some(true),
        },
    );
    assert_eq!(normalized.block_hashes, expected_block_hashes);
    assert_eq!(
        normalized.sequence_hashes,
        compute_seq_hash_for_block(&expected_block_hashes)
    );
    assert_eq!(normalized.isl_tokens, 8);
}

#[test]
fn prompt_request_cache_salt_changes_normalized_hashes() {
    let salted: PromptRequest = serde_json::from_value(serde_json::json!({
        "token_ids": [1, 2, 3, 4],
        "cache_salt": "tenant-a"
    }))
    .expect("deserialize cache_salt");
    let unsalted: PromptRequest = serde_json::from_value(serde_json::json!({
        "token_ids": [1, 2, 3, 4]
    }))
    .expect("deserialize unsalted prompt");

    let salted = salted
        .normalize_for_selection(4, false)
        .expect("normalize salted prompt");
    let unsalted = unsalted
        .normalize_for_selection(4, false)
        .expect("normalize unsalted prompt");

    assert_ne!(salted.block_hashes, unsalted.block_hashes);
    assert_ne!(salted.sequence_hashes, unsalted.sequence_hashes);
}

#[test]
fn overlap_scores_response_honors_override_and_includes_python_shape_fields() {
    let worker = WorkerWithDpRank::new(1, 0);
    let idle_worker = WorkerWithDpRank::new(2, 0);
    let mut device_scores = OverlapScores::new();
    device_scores.scores.insert(worker, 2);
    device_scores.frequencies = vec![1, 1];
    let mut host = LowerTierMatchDetails::default();
    host.hits.insert(worker, 1);
    let mut tiered = TieredMatchDetails {
        device: MatchDetails {
            overlap_scores: device_scores,
            last_matched_hashes: Default::default(),
            router_hint_root_candidates: None,
        },
        lower_tier: Default::default(),
    };
    tiered.lower_tier.insert(StorageTier::HostPinned, host);

    let mut config = test_config();
    config.host_cache_hit_weight = 0.75;
    let override_config = RouterConfigOverride {
        overlap_score_credit: Some(0.5),
        ..Default::default()
    };
    let response = build_overlap_scores_response(
        &config,
        Some(&override_config),
        &tiered,
        4,
        2,
        [worker, idle_worker],
        false,
        None,
        None,
    );

    assert_eq!(response.workers.len(), 2);
    let selected = response
        .workers
        .iter()
        .find(|row| row.worker_id == 1)
        .expect("worker row");
    assert_eq!(selected.device_blocks, 2);
    assert_eq!(selected.host_pinned_blocks, 3);
    assert_eq!(selected.disk_blocks, 3);
    assert_eq!(selected.host_pinned_extension_blocks, 1);
    assert_eq!(selected.shared_beyond_device_blocks, None);
    assert_eq!(selected.router_credit_blocks, 1.75);
    assert!(!response.shared_cache.enabled);
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
    let service = Arc::new(SelectionService::new_local_for_test(config, 1));
    let app = create_router(Arc::new(AppState { service }));

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
    assert_eq!(body["routing_group"], "default");
    assert!(body.get("tenant_id").is_none());
    assert_eq!(body["worker_id"], 1);
    assert_eq!(body["effective_prefill_tokens"], 4);
    assert_eq!(body["overlap"]["longest_matched"], 0);
    assert_eq!(body["overlap"]["dp"]["0"], 0);
    assert!(body.get("cached_tokens").is_none());
    assert!(body.get("effective_overlap_blocks").is_none());

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
async fn overlap_scores_returns_all_schedulable_worker_ranks() {
    let app = app();
    assert_eq!(
        register_worker(app.clone(), None).await.status(),
        StatusCode::CREATED
    );
    let response = post(
        app.clone(),
        "/workers",
        r#"{
            "worker_id": 2,
            "model_name": "model",
            "endpoint": "http://worker-2:8000",
            "block_size": 4,
            "data_parallel_start_rank": 2,
            "data_parallel_size": 2
        }"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::CREATED);

    let response = post(
        app,
        "/overlap_scores",
        r#"{"model_name":"model","token_ids":[1,2,3,4]}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["num_blocks"], 1);
    let workers = body["workers"].as_array().expect("workers array");
    let ids: Vec<_> = workers
        .iter()
        .map(|row| {
            (
                row["worker_id"].as_u64().unwrap(),
                row["dp_rank"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(ids, vec![(1, 0), (2, 2), (2, 3)]);
    assert_eq!(workers[0]["host_pinned_blocks"], 0);
    assert_eq!(workers[0]["disk_blocks"], 0);
    assert_eq!(body["shared_cache"]["enabled"], false);
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
    assert_eq!(body["effective_prefill_tokens"], 4);

    let loads_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/loads")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let loads = response_json(loads_response).await;
    assert_eq!(loads[0]["loads"][0]["potential_prefill_tokens"], 4);

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
async fn standalone_policy_classes_apply_header_thresholds_and_structured_rejection() {
    let policy_file = tempfile::NamedTempFile::new().expect("create policy file");
    std::fs::write(
        policy_file.path(),
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: latency
    policy_family: latency
    cache_bucket: all
    queue_policy: fcfs
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 0
  - name: batch
    policy_family: batch
    cache_bucket: all
    queue_policy: wspt
    quantum: 4
    prefill_busy_threshold: 1024
"#,
    )
    .expect("write policy file");

    let mut config = test_config();
    config.router_policy_config = Some(policy_file.path().to_string_lossy().into_owned());
    let service = Arc::new(SelectionService::new_local_for_test(config, 1));
    let app = create_router(Arc::new(AppState { service }));

    assert_eq!(
        register_worker(app.clone(), Some(4)).await.status(),
        StatusCode::CREATED
    );

    let reserved = post_with_policy_class(
        app.clone(),
        "/select_and_reserve",
        r#"{"model_name":"model","token_ids":[1,2,3,4],"reservation_id":"latency-active"}"#,
        Some("latency"),
    )
    .await;
    assert_eq!(reserved.status(), StatusCode::OK);

    let batch = post_with_policy_class(
        app.clone(),
        "/select",
        r#"{"model_name":"model","token_ids":[5,6,7,8]}"#,
        Some("batch"),
    )
    .await;
    assert_eq!(batch.status(), StatusCode::OK);

    let rejected = post_with_policy_class(
        app,
        "/select",
        r#"{"model_name":"model","token_ids":[9,10,11,12]}"#,
        Some("latency"),
    )
    .await;
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response_json(rejected).await;
    assert_eq!(body["details"]["policy_class"], "latency");
    assert_eq!(body["details"]["limit_kind"], "requests");
    assert_eq!(body["details"]["current"], 0);
    assert_eq!(body["details"]["limit"], 0);
}

#[tokio::test]
async fn output_block_endpoint_updates_reserved_load() {
    let app = app();
    assert_eq!(
        register_worker(app.clone(), None).await.status(),
        StatusCode::CREATED
    );

    let response = post(
        app.clone(),
        "/select_and_reserve",
        r#"{"model_name":"model","token_ids":[1,2,3,4],"reservation_id":"res-output"}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let loads_before = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/loads")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let before = response_json(loads_before).await;
    let before_blocks = before[0]["loads"][0]["potential_decode_blocks"]
        .as_u64()
        .unwrap();

    let response = post(
        app.clone(),
        "/reservations/res-output/output_block",
        r#"{}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let invalid = post(
        app.clone(),
        "/reservations/res-output/output_block",
        r#"{"decay_fraction":1.5}"#,
    )
    .await;
    assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);

    let loads_after = app
        .oneshot(
            Request::builder()
                .uri("/loads")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let after = response_json(loads_after).await;
    let after_blocks = after[0]["loads"][0]["potential_decode_blocks"]
        .as_u64()
        .unwrap();
    assert_eq!(after_blocks, before_blocks + 1);
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
        "isl_tokens": 4,
        "effective_prefill_tokens": selected["effective_prefill_tokens"]
    });
    let reserve_response = post(app.clone(), "/reservations", &reservation.to_string()).await;
    assert_eq!(reserve_response.status(), StatusCode::CREATED);

    let loads_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/loads")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let loads = response_json(loads_response).await;
    assert_eq!(loads[0]["loads"][0]["potential_prefill_tokens"], 4);

    let prefill_response = post(app, "/reservations/res-b/prefill_complete", "{}").await;
    assert_eq!(prefill_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn explicit_reservation_rejects_effective_prefill_above_isl() {
    let app = app();
    assert_eq!(
        register_worker(app.clone(), None).await.status(),
        StatusCode::CREATED
    );
    let reservation = serde_json::json!({
        "model_name": "model",
        "reservation_id": "res-too-large",
        "worker_id": 1,
        "sequence_hashes": [1],
        "isl_tokens": 4,
        "effective_prefill_tokens": 5
    });

    let response = post(app, "/reservations", &reservation.to_string()).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        response_json(response).await["error"]
            .as_str()
            .unwrap()
            .contains("must not exceed isl_tokens")
    );
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn selector_replica_sync_propagates_request_lifecycle() {
    let port_a = reserve_tcp_port();
    let port_b = reserve_tcp_port();
    let config_a = SelectionServiceConfig {
        port: 8092,
        threads: 1,
        indexer_peers: Vec::new(),
        replica_sync_port: Some(port_a),
        replica_sync_peers: Vec::new(),
        kv_router_config: test_config(),
    };
    let service_a = Arc::new(config_a.service_builder().build().await.unwrap());
    let service_b = Arc::new(
        SelectionServiceBuilder::new(test_config())
            .indexer_threads(1)
            .replica_sync(port_b, Vec::new())
            .build()
            .await
            .unwrap(),
    );
    service_b
        .register_replica_peer(format!("tcp://127.0.0.1:{port_a}"))
        .await
        .unwrap();

    let app_a = create_router(Arc::new(AppState {
        service: Arc::clone(&service_a),
    }));
    assert_eq!(
        register_worker(app_a.clone(), None).await.status(),
        StatusCode::CREATED
    );
    let worker: WorkerRequest = serde_json::from_value(serde_json::json!({
        "worker_id": 1,
        "model_name": "model",
        "endpoint": "http://worker-1:8000",
        "block_size": 4
    }))
    .unwrap();
    service_b.upsert_worker(worker).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let response = post(
        app_a.clone(),
        "/select_and_reserve",
        r#"{"model_name":"model","token_ids":[1,2,3,4],"reservation_id":"replicated"}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    wait_for_service_load(&service_b, 1, 1, 4).await;

    let response = post(
        app_a.clone(),
        "/reservations/replicated/prefill_complete",
        "{}",
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    wait_for_service_load(&service_b, 1, 1, 0).await;

    let response = app_a
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/reservations/replicated")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    wait_for_service_load(&service_b, 0, 0, 0).await;

    service_a.shutdown().await;
    service_b.shutdown().await;
}

async fn wait_for_service_load(
    service: &SelectionService,
    expected_requests: usize,
    expected_blocks: usize,
    expected_tokens: usize,
) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let loads = service.loads(Some("model"), Some("default"));
            if let Some(load) = loads.first().and_then(|model| model.loads.first())
                && load.active_requests == expected_requests
                && load.potential_decode_blocks == expected_blocks
                && load.potential_prefill_tokens == expected_tokens
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

fn reserve_tcp_port() -> u16 {
    StdTcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[tokio::test]
async fn dump_exposes_compatible_indexer_snapshot() {
    let app = app();
    assert_eq!(
        register_worker(app.clone(), None).await.status(),
        StatusCode::CREATED
    );

    let response = app
        .oneshot(Request::builder().uri("/dump").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["model:default"]["block_size"], 4);
    assert!(body["model:default"]["events"].is_array());
}

#[tokio::test]
async fn reconcile_rolls_back_partial_listener_registration() {
    let mut config = test_config();
    config.use_kv_events = true;
    let service = Arc::new(SelectionService::new_local_for_test(config, 1));
    let app = create_router(Arc::new(AppState { service }));

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
