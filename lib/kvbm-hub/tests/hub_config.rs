// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub configuration-management contract (P1):
//! - `GET /v1/config` aggregate (primary + per-feature descriptors + dep graph,
//!   with KV-index sizing inherited from `primary`),
//! - register-time dependency-closure enforcement (generalised CD→P2P),
//! - register-time must-match validation against `primary`,
//! - KV-index instances register and have their index entries swept on
//!   unregister (closing the reclaim gap).

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_tokens::TokenBlockSequence;
use futures::SinkExt;
use kvbm_hub::{
    BlockLayoutMode, ConditionalDisaggConfig, ConditionalDisaggManager, ConditionalDisaggRole,
    Feature, FeatureManager, HubConfigResponse, HubServer, IndexerFeatureConfig, IndexerManager,
    P2pManager, PrimaryConfig, RuntimeConfigSummary,
};
use kvbm_logical::events::{KvCacheEvents, KvbmCacheEvents};
use kvbm_logical::{KvbmSequenceHashProvider, SequenceHash};
use serde_json::{Value, json};
use tmq::{Context, Multipart, publish::Publish, publish::publish};
use velo_ext::{InstanceId, PeerInfo, WorkerAddress};

const BLOCK_SIZE: usize = 16;
const MAX_SEQ_LEN: usize = 1024;

/// Hub with the full feature stack and an authoritative `primary` config.
async fn start_hub() -> HubServer {
    let kv = IndexerManager::new(
        MAX_SEQ_LEN,
        BLOCK_SIZE,
        Some("tcp://127.0.0.1:0".to_string()),
        Some("127.0.0.1".to_string()),
    )
    .expect("build indexer manager");

    let p2p = Arc::new(P2pManager::new());

    kvbm_hub::create_server_builder()
        .bind_addr("127.0.0.1".parse().unwrap())
        .discovery_port(0)
        .control_port(0)
        .heartbeat_interval(Duration::from_secs(3600))
        .heartbeat_max_failures(u32::MAX)
        .registration_ttl(Duration::from_secs(3600))
        .primary_config(PrimaryConfig {
            block_size: Some(BLOCK_SIZE),
            max_seq_len: Some(MAX_SEQ_LEN),
            block_layout: BlockLayoutMode::Operational,
            g2_memory_gib: Some(2.0),
            g2_blocks: None,
            advertise_host: Some("127.0.0.1".to_string()),
        })
        .add_feature_manager(p2p as Arc<dyn FeatureManager>)
        .add_feature_manager(Arc::new(ConditionalDisaggManager::new()) as Arc<dyn FeatureManager>)
        .add_feature_manager(Arc::new(kv) as Arc<dyn FeatureManager>)
        .serve()
        .await
        .expect("start hub")
}

fn peer() -> (InstanceId, PeerInfo) {
    let id = InstanceId::new_v4();
    (
        id,
        PeerInfo::new(id, WorkerAddress::from_encoded(b"t".to_vec())),
    )
}

async fn register(
    http: &reqwest::Client,
    control_base: &str,
    peer_info: PeerInfo,
    features: Vec<Feature>,
    runtime: Option<RuntimeConfigSummary>,
) -> reqwest::Response {
    http.post(format!("{control_base}/v1/instances"))
        .json(&json!({
            "peer_info": peer_info,
            "features": features,
            "runtime": runtime,
        }))
        .send()
        .await
        .expect("POST /v1/instances")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aggregate_config_reports_primary_and_features() {
    let server = start_hub().await;
    let base = format!("http://{}", server.discovery_addr());
    let http = reqwest::Client::new();

    let resp: HubConfigResponse = http
        .get(format!("{base}/v1/config"))
        .send()
        .await
        .expect("GET /v1/config")
        .json()
        .await
        .expect("decode HubConfigResponse");

    // Primary echoed back.
    assert_eq!(resp.primary.block_size, Some(BLOCK_SIZE));
    assert_eq!(resp.primary.max_seq_len, Some(MAX_SEQ_LEN));
    assert_eq!(resp.primary.block_layout, BlockLayoutMode::Operational);

    // Features present, deterministically ordered by key.
    let keys: Vec<String> = resp.features.iter().map(|f| f.key.to_string()).collect();
    assert_eq!(keys, vec!["disagg", "indexer", "p2p"], "got {keys:?}");

    // CD declares its P2P dependency in the aggregate.
    let cd = resp
        .features
        .iter()
        .find(|f| f.key.as_str() == "disagg")
        .unwrap();
    assert_eq!(
        cd.dependencies
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>(),
        vec!["p2p"]
    );

    // KV-index descriptor advertises sizing (inherited) + ZMQ endpoint so a
    // connector can wire its publisher straight from the aggregate.
    let kv = resp
        .features
        .iter()
        .find(|f| f.key.as_str() == "indexer")
        .unwrap();
    assert_eq!(kv.config["block_size"].as_u64(), Some(BLOCK_SIZE as u64));
    assert_eq!(kv.config["max_seq_len"].as_u64(), Some(MAX_SEQ_LEN as u64));
    assert!(
        kv.config["zmq_endpoint"]
            .as_str()
            .unwrap()
            .starts_with("tcp://127.0.0.1:"),
        "zmq_endpoint: {}",
        kv.config["zmq_endpoint"]
    );

    server.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cd_without_p2p_rejected_by_dependency_closure() {
    let server = start_hub().await;
    let base = format!("http://{}", server.control_addr());
    let http = reqwest::Client::new();

    let (_, p) = peer();
    let resp = register(
        &http,
        &base,
        p,
        vec![Feature::ConditionalDisagg(ConditionalDisaggConfig {
            role: ConditionalDisaggRole::Decode,
        })],
        None,
    )
    .await;
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("p2p"),
        "expected dep error mentioning p2p: {body}"
    );

    server.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn indexer_register_block_size_mismatch_rejected() {
    let server = start_hub().await;
    let base = format!("http://{}", server.control_addr());
    let http = reqwest::Client::new();

    let (_, p) = peer();
    let resp = register(
        &http,
        &base,
        p,
        vec![Feature::Indexer(IndexerFeatureConfig::default())],
        Some(RuntimeConfigSummary {
            block_size: Some(BLOCK_SIZE * 2), // wrong
            block_layout: None,
        }),
    )
    .await;
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("block_size"),
        "expected block_size mismatch error: {body}"
    );

    // A present-but-empty summary must not slip through: a required field left
    // `None` is rejected (this is the residual the reconciled `primary` closes
    // — validation has an authoritative value to compare against).
    let (_, p_empty) = peer();
    let empty = register(
        &http,
        &base,
        p_empty,
        vec![Feature::Indexer(IndexerFeatureConfig::default())],
        Some(RuntimeConfigSummary::default()),
    )
    .await;
    assert_eq!(empty.status(), reqwest::StatusCode::BAD_REQUEST);
    assert!(
        empty.text().await.unwrap().contains("block_size"),
        "empty summary should be rejected for a required field"
    );

    // A matching summary registers fine.
    let (_, p_ok) = peer();
    let ok = register(
        &http,
        &base,
        p_ok,
        vec![Feature::Indexer(IndexerFeatureConfig::default())],
        Some(RuntimeConfigSummary {
            block_size: Some(BLOCK_SIZE),
            block_layout: None,
        }),
    )
    .await;
    assert!(ok.status().is_success(), "status: {}", ok.status());

    server.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn indexer_register_without_runtime_summary_rejected() {
    // A KV-index registrant must declare its sizing — omitting the runtime
    // summary must NOT bypass must-match validation (the legacy escape hatch
    // is only for P2P / CD).
    let server = start_hub().await;
    let base = format!("http://{}", server.control_addr());
    let http = reqwest::Client::new();

    let (_, p) = peer();
    let resp = register(
        &http,
        &base,
        p,
        vec![Feature::Indexer(IndexerFeatureConfig::default())],
        None,
    )
    .await;
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("runtime config summary"),
        "expected mandatory-summary error: {body}"
    );

    server.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn indexer_register_grows_capacity() {
    // A registrant reporting a larger max_seq_len grows the index capacity
    // (never shrinks); a smaller one leaves it unchanged.
    let server = start_hub().await;
    let disc = format!("http://{}", server.discovery_addr());
    let ctrl = format!("http://{}", server.control_addr());
    let http = reqwest::Client::new();

    let kv_config = |http: reqwest::Client, disc: String| async move {
        http.get(format!("{disc}/v1/features/indexer/config"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["max_seq_len"]
            .as_u64()
            .unwrap() as usize
    };

    assert_eq!(kv_config(http.clone(), disc.clone()).await, MAX_SEQ_LEN);

    // Register with a larger max_seq_len → capacity grows.
    let (_, p) = peer();
    let resp = register(
        &http,
        &ctrl,
        p,
        vec![Feature::Indexer(IndexerFeatureConfig {
            max_seq_len: Some(MAX_SEQ_LEN * 2),
        })],
        Some(RuntimeConfigSummary {
            block_size: Some(BLOCK_SIZE),
            block_layout: None,
        }),
    )
    .await;
    assert!(resp.status().is_success(), "status {}", resp.status());
    assert_eq!(kv_config(http.clone(), disc.clone()).await, MAX_SEQ_LEN * 2);

    // A smaller report never shrinks it.
    let (_, p2) = peer();
    register(
        &http,
        &ctrl,
        p2,
        vec![Feature::Indexer(IndexerFeatureConfig {
            max_seq_len: Some(BLOCK_SIZE),
        })],
        Some(RuntimeConfigSummary {
            block_size: Some(BLOCK_SIZE),
            block_layout: None,
        }),
    )
    .await;
    assert_eq!(kv_config(http, disc).await, MAX_SEQ_LEN * 2);

    server.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn indexer_unregister_sweeps_index() {
    let server = start_hub().await;
    let disc = format!("http://{}", server.discovery_addr());
    let ctrl = format!("http://{}", server.control_addr());
    let http = reqwest::Client::new();

    // Discover the ZMQ ingest endpoint from the aggregate.
    let cfg: HubConfigResponse = http
        .get(format!("{disc}/v1/config"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let endpoint = cfg
        .features
        .iter()
        .find(|f| f.key.as_str() == "indexer")
        .unwrap()
        .config["zmq_endpoint"]
        .as_str()
        .unwrap()
        .to_string();

    // A worker with a known InstanceId publishes a 3-block prefix, then
    // registers under that same id (publisher stamps `instance_id.as_u128()`,
    // which is what `on_unregister` sweeps).
    let (id, p) = peer();
    let id_u128 = id.as_u128();
    let hashes = plhs(3, 1337);

    let ctx = Context::new();
    let mut sock: Publish = publish(&ctx).set_linger(0).connect(&endpoint).unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Resend to defeat ZMQ slow-joiner until the index shows our instance.
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        send_create(&mut sock, hashes.clone(), id_u128).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        if index_has_instance(&http, &disc, &id_u128.to_string()).await {
            break;
        }
        assert!(Instant::now() < deadline, "timed out indexing creates");
    }

    // Register the instance. KV-index mandates a runtime summary; supply a
    // matching one (the point of this test is the registry↔index reclaim
    // wiring, not the mismatch path).
    let resp = register(
        &http,
        &ctrl,
        p,
        vec![Feature::Indexer(IndexerFeatureConfig::default())],
        Some(RuntimeConfigSummary {
            block_size: Some(BLOCK_SIZE),
            block_layout: None,
        }),
    )
    .await;
    assert!(
        resp.status().is_success(),
        "register status {}",
        resp.status()
    );

    // The instance now appears in the registered-instances set — this is
    // driven by registration (declaring `Feature::Indexer`), distinct from the
    // index contents driven by emitted KV events.
    let registered: kvbm_hub::InstancesResponse = http
        .get(format!("{disc}/v1/features/indexer/instances"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        registered.instances.contains(&id_u128.to_string()),
        "registered instance set {:?} missing {id_u128}",
        registered.instances
    );

    // Unregister → on_unregister sweeps the index entries for this instance.
    let del = http
        .delete(format!("{ctrl}/v1/instances/{id}"))
        .send()
        .await
        .unwrap();
    assert!(del.status().is_success(), "delete status {}", del.status());

    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if !index_has_instance(&http, &disc, &id_u128.to_string()).await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            Instant::now() < deadline,
            "index entries not swept after unregister"
        );
    }

    // Unregister also drops the instance from the registered-instances set.
    let registered: kvbm_hub::InstancesResponse = http
        .get(format!("{disc}/v1/features/indexer/instances"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        !registered.instances.contains(&id_u128.to_string()),
        "registered instance set still has {id_u128} after unregister"
    );

    server.shutdown().await.expect("shutdown");
}

// ---- helpers shared by the sweep test ----

fn plhs(n: usize, salt: u64) -> Vec<SequenceHash> {
    let tokens: Vec<u32> = (0..(BLOCK_SIZE * n) as u32).collect();
    let seq = TokenBlockSequence::from_slice(&tokens, BLOCK_SIZE as u32, Some(salt));
    seq.blocks()
        .iter()
        .map(|b| b.kvbm_sequence_hash())
        .collect()
}

async fn send_create(sock: &mut Publish, hashes: Vec<SequenceHash>, instance_id: u128) {
    let batch = KvbmCacheEvents {
        events: KvCacheEvents::Create(hashes),
        instance_id,
    };
    let payload = rmp_serde::to_vec(&batch).expect("encode batch");
    let frames: Vec<Vec<u8>> = vec![b"kvbm.kv_index".to_vec(), payload];
    sock.send(Multipart::from(frames)).await.expect("PUB send");
}

async fn index_has_instance(http: &reqwest::Client, disc_base: &str, id: &str) -> bool {
    let body: Value = http
        .get(format!(
            "{disc_base}/v1/features/indexer/hashes/by_position/0"
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    body["entries"]
        .as_array()
        .map(|entries| {
            entries.iter().any(|e| {
                e["instances"]
                    .as_array()
                    .map(|is| is.iter().any(|v| v.as_str() == Some(id)))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}
