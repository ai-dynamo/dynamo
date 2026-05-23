// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for the KV indexer feature: two mock `PUB` publishers push
//! `KvbmCacheEvents` over ZMQ to the hub's `SUB` ingest socket; the test then
//! asserts the index via the feature's own HTTP surface
//! (`/v1/features/indexer/...`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_tokens::TokenBlockSequence;
use futures::SinkExt;
use kvbm_hub::{HubServer, IndexerConfigResponse};
use kvbm_logical::events::{KvCacheEvents, KvbmCacheEvents};
use kvbm_logical::{KvbmSequenceHashProvider, SequenceHash};
use serde_json::Value;
use tmq::{Context, Multipart, publish::Publish, publish::publish};

const BLOCK_SIZE: u32 = 4;
const MAX_SEQ_LEN: usize = 64;

async fn start_hub() -> HubServer {
    let manager = kvbm_hub::IndexerManager::new(
        MAX_SEQ_LEN,
        BLOCK_SIZE as usize,
        Some("tcp://127.0.0.1:0".to_string()),
        Some("127.0.0.1".to_string()),
    )
    .expect("build indexer manager");

    kvbm_hub::create_server_builder()
        .bind_addr("127.0.0.1".parse().unwrap())
        .discovery_port(0)
        .control_port(0)
        .heartbeat_interval(Duration::from_secs(3600))
        .heartbeat_max_failures(u32::MAX)
        .registration_ttl(Duration::from_secs(3600))
        .add_feature_manager(Arc::new(manager) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start hub")
}

/// Builds `n` PLHs at positions 0..n by laying down `n * BLOCK_SIZE` tokens.
fn plhs(n: usize, salt: u64) -> Vec<SequenceHash> {
    let tokens: Vec<u32> = (0..(BLOCK_SIZE as usize * n) as u32).collect();
    let seq = TokenBlockSequence::from_slice(&tokens, BLOCK_SIZE, Some(salt));
    seq.blocks()
        .iter()
        .map(|b| b.kvbm_sequence_hash())
        .collect()
}

fn connect_pub(endpoint: &str) -> Publish {
    let ctx = Context::new();
    publish(&ctx)
        .set_linger(0)
        .connect(endpoint)
        .expect("connect PUB")
}

async fn send_batch(sock: &mut Publish, events: KvCacheEvents, instance_id: u128) {
    let batch = KvbmCacheEvents {
        events,
        instance_id,
    };
    let payload = rmp_serde::to_vec(&batch).expect("encode batch");
    let frames: Vec<Vec<u8>> = vec![b"kvbm.kv_index".to_vec(), payload];
    sock.send(Multipart::from(frames)).await.expect("PUB send");
}

async fn get_json(http: &reqwest::Client, base: &str, path: &str) -> Value {
    http.get(format!("{base}/v1/features/indexer{path}"))
        .send()
        .await
        .expect("GET")
        .json()
        .await
        .expect("json")
}

fn instances(entry: &Value) -> Vec<String> {
    let mut v: Vec<String> = entry["instances"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    v.sort();
    v
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_instances_publish_index_and_query() {
    let server = start_hub().await;
    let base = format!("http://{}", server.discovery_addr());
    let http = reqwest::Client::new();

    // GET /config: feature present, sizing reported, ZMQ endpoint advertised.
    let cfg: IndexerConfigResponse =
        serde_json::from_value(get_json(&http, &base, "/config").await).expect("config");
    assert_eq!(cfg.block_size, BLOCK_SIZE as usize);
    assert_eq!(cfg.max_seq_len, MAX_SEQ_LEN);
    assert_eq!(cfg.num_positions, MAX_SEQ_LEN / BLOCK_SIZE as usize);
    assert!(
        cfg.zmq_endpoint.starts_with("tcp://127.0.0.1:"),
        "endpoint: {}",
        cfg.zmq_endpoint
    );

    // Two workers holding the same 3-block prefix.
    let id_a: u128 = 0xA0;
    let id_b: u128 = 0xB0;
    let hashes = plhs(3, 1337);

    let mut pub_a = connect_pub(&cfg.zmq_endpoint);
    let mut pub_b = connect_pub(&cfg.zmq_endpoint);
    // Give the SUB time to register the PUB connections before the first send.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // by_position/0 lists both instances. Resend each poll to defeat ZMQ's
    // slow-joiner (a PUB drops messages sent before the SUB connection
    // completes). Create is idempotent in the index.
    let deadline = Instant::now() + Duration::from_secs(8);
    let body = loop {
        send_batch(&mut pub_a, KvCacheEvents::Create(hashes.clone()), id_a).await;
        send_batch(&mut pub_b, KvCacheEvents::Create(hashes.clone()), id_b).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let body = get_json(&http, &base, "/hashes/by_position/0").await;
        let ready = body["entries"]
            .as_array()
            .map(|e| !e.is_empty() && instances(&e[0]).len() == 2)
            .unwrap_or(false);
        if ready {
            break body;
        }
        assert!(
            Instant::now() < deadline,
            "timed out indexing creates: {body}"
        );
    };
    assert_eq!(
        instances(&body["entries"][0]),
        vec![id_a.to_string(), id_b.to_string()]
    );

    // POST /query with the full sequence → deepest match (position 2).
    let resp: Value = http
        .post(format!("{base}/v1/features/indexer/query"))
        .json(&serde_json::json!({ "hashes": hashes }))
        .send()
        .await
        .expect("POST query")
        .json()
        .await
        .expect("query json");
    let hit = &resp["hit"];
    assert_eq!(hit["position"].as_u64(), Some(2));
    assert_eq!(instances(hit), vec![id_a.to_string(), id_b.to_string()]);

    // Remove instance A's blocks → only B remains at position 0.
    let deadline = Instant::now() + Duration::from_secs(8);
    let body = loop {
        send_batch(&mut pub_a, KvCacheEvents::Remove(hashes.clone()), id_a).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let body = get_json(&http, &base, "/hashes/by_position/0").await;
        let ready = body["entries"]
            .as_array()
            .and_then(|e| e.first())
            .map(|e0| instances(e0) == vec![id_b.to_string()])
            .unwrap_or(false);
        if ready {
            break body;
        }
        assert!(
            Instant::now() < deadline,
            "timed out indexing remove: {body}"
        );
    };
    assert_eq!(instances(&body["entries"][0]), vec![id_b.to_string()]);

    server.shutdown().await.expect("shutdown");
}
