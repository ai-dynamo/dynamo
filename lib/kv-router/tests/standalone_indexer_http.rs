// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP-level integration tests for the standalone KV indexer.
//!
//! Patterned after `lib/llm/tests/http-service.rs`: bind a random port, spawn
//! `axum::serve` in a background tokio task, poll `/health` until ready, then
//! drive `reqwest` against the live service. Tests bypass the ZMQ listener
//! path and pre-populate the `WorkerRegistry`'s `Indexer` directly via
//! `apply_event_routed`, so they exercise the HTTP layer end-to-end without
//! requiring a running ZMQ publisher.

#![cfg(feature = "standalone-indexer")]

use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier, compute_seq_hash_for_block,
};
use dynamo_kv_router::standalone_indexer::registry::{IndexerKey, WorkerRegistry};
use dynamo_kv_router::standalone_indexer::server::{AppState, create_router};
use dynamo_kv_router::zmq_wire::{BlockHashValue, RawKvEvent};
use serde_json::json;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// Construct a STORE [`RouterEvent`] for `local_hashes` chaining off
/// `prefix_hashes`. Mirrors the helper used by the in-crate `Indexer` tests so
/// we exercise identical event shapes from the integration-test surface.
fn store_event(
    worker_id: u64,
    dp_rank: u32,
    event_id: u64,
    prefix_hashes: &[u64],
    local_hashes: &[u64],
    storage_tier: StorageTier,
) -> RouterEvent {
    let prefix_block_hashes: Vec<LocalBlockHash> =
        prefix_hashes.iter().copied().map(LocalBlockHash).collect();
    let parent_hash = compute_seq_hash_for_block(&prefix_block_hashes)
        .last()
        .copied()
        .map(ExternalSequenceBlockHash);

    let full_hashes: Vec<LocalBlockHash> = prefix_hashes
        .iter()
        .chain(local_hashes.iter())
        .copied()
        .map(LocalBlockHash)
        .collect();
    let full_sequence_hashes = compute_seq_hash_for_block(&full_hashes);
    let new_sequence_hashes = &full_sequence_hashes[prefix_hashes.len()..];
    let blocks = local_hashes
        .iter()
        .zip(new_sequence_hashes.iter())
        .map(|(&local_hash, &sequence_hash)| KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(sequence_hash),
            tokens_hash: LocalBlockHash(local_hash),
            mm_extra_info: None,
        })
        .collect();

    RouterEvent::with_storage_tier(
        worker_id,
        KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash,
                start_position: None,
                blocks,
            }),
            dp_rank,
        },
        storage_tier,
    )
}

/// Bind to an OS-assigned port, return the listener and the resolved address.
async fn bind_localhost() -> (TcpListener, std::net::SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind localhost:0");
    let addr = listener.local_addr().expect("local_addr");
    (listener, addr)
}

/// Poll `/health` until the service responds 200 or the timeout elapses.
async fn wait_for_health(base_url: &str) {
    let client = reqwest::Client::new();
    let url = format!("{base_url}/health");
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(5);
    loop {
        if let Ok(resp) = client.get(&url).send().await
            && resp.status().is_success()
        {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("standalone indexer /health did not become ready");
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }
}

/// Build a registry seeded with one indexer for `(model, tenant)` and feed
/// `events` through `apply_event_routed`. This bypasses the ZMQ listener path
/// so we can drive HTTP queries against deterministic state.
async fn registry_with_events(
    model: &str,
    tenant: &str,
    block_size: u32,
    events: Vec<RouterEvent>,
) -> Arc<WorkerRegistry> {
    let registry = Arc::new(WorkerRegistry::new(1));
    registry.signal_ready();
    let indexer = registry.get_or_create_indexer(
        IndexerKey {
            model_name: model.to_string(),
            tenant_id: tenant.to_string(),
        },
        block_size,
    );
    for event in events {
        indexer.apply_event_routed(event).await;
    }
    // Force the in-flight events through KvIndexer's mpsc channel before the
    // first query lands; otherwise the test races the indexer worker.
    if let Some(entry) = registry.get_indexer(&IndexerKey {
        model_name: model.to_string(),
        tenant_id: tenant.to_string(),
    }) {
        let dump = entry.indexer.dump_events().await.expect("dump events");
        // dump_events provides the FIFO barrier we need; we don't care about
        // its contents here.
        drop(dump);
    }
    registry
}

/// Spawn the HTTP server with `state` on a random localhost port and return
/// the base URL plus the cancellation token used to shut it down.
async fn spawn_indexer_http(
    state: Arc<AppState>,
) -> (String, CancellationToken, tokio::task::JoinHandle<()>) {
    let (listener, addr) = bind_localhost().await;
    let app = create_router(state);
    let cancel = CancellationToken::new();
    let cancel_for_serve = cancel.clone();
    let task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move { cancel_for_serve.cancelled().await })
            .await
            .expect("axum::serve");
    });
    let base_url = format!("http://{addr}");
    wait_for_health(&base_url).await;
    (base_url, cancel, task)
}

#[cfg(feature = "metrics")]
fn make_app_state(registry: Arc<WorkerRegistry>) -> Arc<AppState> {
    Arc::new(AppState {
        registry,
        prom_registry: prometheus::Registry::new(),
    })
}

#[cfg(not(feature = "metrics"))]
fn make_app_state(registry: Arc<WorkerRegistry>) -> Arc<AppState> {
    Arc::new(AppState { registry })
}

/// `/query_by_hash` against a populated registry must surface both the legacy
/// flat shape (for backward-compat callers) and the Mooncake RFC #1403 shape
/// (per-instance `gpu`/`cpu`/`disk`/`longest_matched`).
#[tokio::test]
async fn query_by_hash_returns_per_instance_tier_breakdown() {
    const BLOCK_SIZE: u32 = 4;
    const MODEL: &str = "test-model";
    const TENANT: &str = "default";

    // Worker 7: 2 device blocks + 1 host-pinned extension.
    // Worker 8: 2 device blocks, no lower-tier.
    let events = vec![
        store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device),
        store_event(8, 0, 1, &[], &[11, 12], StorageTier::Device),
        store_event(7, 0, 2, &[11, 12], &[13], StorageTier::HostPinned),
    ];
    let registry = registry_with_events(MODEL, TENANT, BLOCK_SIZE, events).await;
    let state = make_app_state(registry);
    let (base_url, cancel, task) = spawn_indexer_http(state).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/query_by_hash"))
        .json(&json!({
            "block_hashes": [11_i64, 12, 13],
            "model_name": MODEL,
            "tenant_id": TENANT,
        }))
        .send()
        .await
        .expect("POST /query_by_hash");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let body: serde_json::Value = resp.json().await.expect("parse JSON body");

    // Legacy flat shape — device-tier overlap, scaled to tokens.
    assert_eq!(
        body["scores"]["7"]["0"],
        (2 * BLOCK_SIZE) as u64,
        "legacy scores should still carry worker 7's device match"
    );
    assert_eq!(
        body["scores"]["8"]["0"],
        (2 * BLOCK_SIZE) as u64,
        "legacy scores should still carry worker 8's device match"
    );

    // Mooncake-shape per-instance breakdown.
    let inst7 = &body["instances"]["7"];
    assert_eq!(inst7["gpu"], (2 * BLOCK_SIZE) as u64, "instance 7 gpu");
    assert_eq!(
        inst7["cpu"],
        (3 * BLOCK_SIZE) as u64,
        "instance 7 cpu cumulative through host-pinned"
    );
    assert_eq!(
        inst7["disk"],
        (3 * BLOCK_SIZE) as u64,
        "instance 7 disk falls back to cpu when no disk extension exists"
    );
    assert_eq!(
        inst7["dp"]["0"],
        (2 * BLOCK_SIZE) as u64,
        "instance 7 dp_rank=0 device count"
    );
    assert_eq!(
        inst7["longest_matched"],
        (3 * BLOCK_SIZE) as u64,
        "instance 7 longest_matched is the max across tiers"
    );

    let inst8 = &body["instances"]["8"];
    assert_eq!(inst8["gpu"], (2 * BLOCK_SIZE) as u64);
    assert_eq!(
        inst8["cpu"],
        (2 * BLOCK_SIZE) as u64,
        "instance 8 cpu falls back to device when no host extension exists"
    );
    assert_eq!(inst8["longest_matched"], (2 * BLOCK_SIZE) as u64);

    cancel.cancel();
    task.await.expect("server task join");
}

/// `/query` against an unknown `(model, tenant)` must return 404 with an
/// error body, not 500 or a panic.
#[tokio::test]
async fn query_returns_404_for_unknown_model() {
    let registry = Arc::new(WorkerRegistry::new(1));
    registry.signal_ready();
    let state = make_app_state(registry);
    let (base_url, cancel, task) = spawn_indexer_http(state).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/query"))
        .json(&json!({
            "token_ids": [1_u32, 2, 3],
            "model_name": "no-such-model",
        }))
        .send()
        .await
        .expect("POST /query");
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);

    let body: serde_json::Value = resp.json().await.expect("parse error body");
    assert!(
        body["error"]
            .as_str()
            .map(|s| s.contains("no indexer for model="))
            .unwrap_or(false),
        "unexpected 404 body: {body}"
    );

    cancel.cancel();
    task.await.expect("server task join");
}

/// Smoke test that the axum router is wired correctly end-to-end and
/// reports `/health` 200. Catches misuse of feature flags / route gates.
#[tokio::test]
async fn health_returns_ok() {
    let registry = Arc::new(WorkerRegistry::new(1));
    let state = make_app_state(registry);
    let (base_url, cancel, task) = spawn_indexer_http(state).await;

    let resp = reqwest::Client::new()
        .get(format!("{base_url}/health"))
        .send()
        .await
        .expect("GET /health");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    cancel.cancel();
    task.await.expect("server task join");
}

// =============================================================================
// ZMQ-publisher e2e — drive a real PUB socket through the listener path
// =============================================================================

/// Reserve an OS-assigned TCP port by binding+dropping a `std::net::TcpListener`
/// (matches the pattern used in `standalone_indexer/listener.rs` tests).
fn reserve_zmq_endpoint() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind probe listener");
    let port = listener
        .local_addr()
        .expect("local_addr on probe listener")
        .port();
    drop(listener);
    format!("tcp://127.0.0.1:{port}")
}

/// Build a `RawKvEvent::BlockStored` with the minimum fields the engine
/// emits over ZMQ. `medium` is the wire spelling of the storage tier:
/// `"GPU"` → Device, `"CPU_TIER1"` → HostPinned, `"CPU_TIER2"` → Disk.
fn raw_block_stored(
    block_hash: u64,
    parent_block_hash: Option<u64>,
    token_ids: Vec<u32>,
    block_size: usize,
    medium: &str,
) -> RawKvEvent {
    RawKvEvent::BlockStored {
        block_hashes: vec![BlockHashValue::Unsigned(block_hash)],
        parent_block_hash: parent_block_hash.map(BlockHashValue::Unsigned),
        token_ids,
        block_size,
        medium: Some(medium.to_string()),
        lora_name: None,
        block_mm_infos: None,
        is_eagle: None,
        group_idx: None,
        kv_cache_spec_kind: None,
        kv_cache_spec_sliding_window: None,
    }
}

/// Encode a one-event batch into the wire payload the listener expects:
/// msgpack-array of `(timestamp, [events], dp_rank)`.
fn encode_batch(event: RawKvEvent, dp_rank: i32) -> Vec<u8> {
    rmp_serde::to_vec_named(&(0.0_f64, vec![event], Some(dp_rank))).expect("serialize KvEventBatch")
}

/// Send one ZMQ multipart message in the listener's expected frame layout:
/// `[topic, seq_be_bytes, payload]`. Topic is unused (SUB filter is `b""`).
fn send_live_message(pub_socket: &zmq::Socket, seq: u64, payload: &[u8]) {
    pub_socket
        .send_multipart(
            [b"" as &[u8], &seq.to_be_bytes(), payload],
            0, // no flags
        )
        .expect("send_multipart");
}

/// Poll `/query` against `(model, tenant)` until `instances[instance_id].cpu`
/// reaches `expected_cpu_tokens`, or panic on timeout. The listener applies
/// events asynchronously after `/register`, so the first few queries may see
/// the indexer still cold; this loop is the e2e-test equivalent of awaiting
/// convergence.
async fn await_instance_cpu(
    base_url: &str,
    model: &str,
    tenant: &str,
    instance_id: u64,
    token_ids: Vec<u32>,
    expected_cpu_tokens: u64,
) -> serde_json::Value {
    let client = reqwest::Client::new();
    let url = format!("{base_url}/query");
    let body = json!({
        "token_ids": token_ids,
        "model_name": model,
        "tenant_id": tenant,
    });
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let key = instance_id.to_string();
    loop {
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .expect("POST /query");
        if resp.status() == reqwest::StatusCode::OK {
            let value: serde_json::Value = resp.json().await.expect("parse /query body");
            if value["instances"][&key]["cpu"]
                .as_u64()
                .map(|n| n >= expected_cpu_tokens)
                .unwrap_or(false)
            {
                return value;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            panic!(
                "timed out waiting for instance {instance_id} cpu>={expected_cpu_tokens}; \
                 last response did not converge"
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Real ZMQ PUB → listener → indexer → HTTP query round trip.
///
/// Bind a PUB socket on a random localhost port, register a worker that
/// subscribes to it, publish one Device-tier and one HostPinned-tier
/// `BlockStored` event chained together, and assert the HTTP `/query`
/// response surfaces the per-tier breakdown — which it can only do if the
/// listener correctly normalized `medium="CPU_TIER1"` into a HostPinned
/// `RouterEvent` and the standalone `Indexer::apply_event_routed` dispatched
/// it to the lower-tier slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zmq_published_tiered_events_appear_in_http_query() {
    const BLOCK_SIZE: u32 = 4;
    const MODEL: &str = "test-model";
    const TENANT: &str = "default";
    const INSTANCE_ID: u64 = 7;

    // Spin up the HTTP server with an empty registry.
    let registry = Arc::new(WorkerRegistry::new(1));
    registry.signal_ready();
    let state = make_app_state(registry);
    let (base_url, server_cancel, server_task) = spawn_indexer_http(state).await;
    let client = reqwest::Client::new();

    // Bind a PUB socket BEFORE /register so the listener finds someone to
    // subscribe to once it spawns. ZMQ's slow-joiner problem still means a
    // few early messages may be lost; we send a probe before the real
    // payloads to flush the connection.
    let zmq_endpoint = reserve_zmq_endpoint();
    let zmq_ctx = zmq::Context::new();
    let pub_socket = zmq_ctx.socket(zmq::PUB).expect("create PUB socket");
    pub_socket.set_linger(0).expect("set_linger");
    pub_socket.bind(&zmq_endpoint).expect("bind PUB socket");

    // Register the worker — this triggers the listener to connect a SUB.
    let resp = client
        .post(format!("{base_url}/register"))
        .json(&json!({
            "instance_id": INSTANCE_ID,
            "endpoint": zmq_endpoint,
            "model_name": MODEL,
            "tenant_id": TENANT,
            "block_size": BLOCK_SIZE,
            "dp_rank": 0,
        }))
        .send()
        .await
        .expect("POST /register");
    assert_eq!(resp.status(), reqwest::StatusCode::CREATED);

    // Give the SUB time to connect to our PUB. ZMQ buffers most things, but
    // empirically a short settle plus a probe message reduces flake.
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Token sequences for two adjacent blocks. block_size = 4.
    let device_tokens = vec![10_u32, 11, 12, 13];
    let host_tokens = vec![20_u32, 21, 22, 23];
    let device_block_hash: u64 = 0xD0_E0_AA_AA_BB_BB_CC_CC;
    let host_block_hash: u64 = 0x40_E0_AA_AA_BB_BB_CC_CC;

    // seq 0: Device-tier store (parent=None).
    let device_event = raw_block_stored(
        device_block_hash,
        None,
        device_tokens.clone(),
        BLOCK_SIZE as usize,
        "GPU",
    );
    send_live_message(&pub_socket, 0, &encode_batch(device_event, 0));

    // seq 1: HostPinned-tier store anchored on the device hash.
    let host_event = raw_block_stored(
        host_block_hash,
        Some(device_block_hash),
        host_tokens.clone(),
        BLOCK_SIZE as usize,
        "CPU_TIER1",
    );
    send_live_message(&pub_socket, 1, &encode_batch(host_event, 0));

    // Compose the full prefix for the query: device tokens followed by host
    // tokens. The server will hash this and match it against the radix tree
    // (device blocks) plus the lower-tier index (host extension).
    let mut full_tokens = device_tokens.clone();
    full_tokens.extend(host_tokens.clone());

    // Wait for the listener to apply both events; assert via the per-instance
    // tier breakdown.
    let body = await_instance_cpu(
        &base_url,
        MODEL,
        TENANT,
        INSTANCE_ID,
        full_tokens,
        // Cumulative through host = 2 blocks * BLOCK_SIZE tokens.
        (2 * BLOCK_SIZE) as u64,
    )
    .await;

    let inst = &body["instances"][INSTANCE_ID.to_string()];
    assert_eq!(
        inst["gpu"], BLOCK_SIZE as u64,
        "device-tier hits should equal one block of tokens"
    );
    assert_eq!(
        inst["cpu"],
        (2 * BLOCK_SIZE) as u64,
        "host-pinned cumulative should be device + host extension"
    );
    assert_eq!(
        inst["longest_matched"],
        (2 * BLOCK_SIZE) as u64,
        "longest_matched should reflect the full device + host reach"
    );

    server_cancel.cancel();
    server_task.await.expect("server task join");
    drop(pub_socket);
}
