// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The render path is gated behind the `kvbmctl` feature (pulls CUDA via
// kvbm-config). Compile this suite to nothing unless that feature is on; run it
// with `cargo test -p kvbm-hub --features kvbmctl`.
#![cfg(feature = "kvbmctl")]

//! Integration test for the `kvbmctl` render path against a live hub.
//!
//! Stands up a real `HubServer`, fetches `GET /v1/config` through `HubClient`
//! (built from a URL the same way the binary does), and renders the vLLM CLI
//! fragment. The render's own validation round-trips the emitted
//! `kv_connector_extra_config` through the connector parser, so a green test
//! means the hub aggregate → CLI fragment → connector-config path is sound.

use std::sync::Arc;
use std::time::Duration;

use kvbm_hub::features::indexer::cli::IndexerCli;
use kvbm_hub::render::{VllmRenderOptions, render_vllm_cli};
use kvbm_hub::{
    BlockLayoutMode, ConditionalDisaggManager, FeatureCli, FeatureManager, HubClientBuilder,
    HubServer, IndexerManager, P2pManager, PrimaryConfig,
};

const BLOCK_SIZE: usize = 16;
const MAX_SEQ_LEN: usize = 8192;

async fn start_hub() -> HubServer {
    let kv = IndexerManager::new(
        MAX_SEQ_LEN,
        BLOCK_SIZE,
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
        .primary_config(PrimaryConfig {
            block_size: Some(BLOCK_SIZE),
            max_seq_len: Some(MAX_SEQ_LEN),
            block_layout: BlockLayoutMode::Operational,
            g2_memory_gib: Some(2.0),
            g2_blocks: None,
            advertise_host: Some("127.0.0.1".to_string()),
        })
        .add_feature_manager(Arc::new(P2pManager::new()) as Arc<dyn FeatureManager>)
        .add_feature_manager(Arc::new(ConditionalDisaggManager::new()) as Arc<dyn FeatureManager>)
        .add_feature_manager(Arc::new(kv) as Arc<dyn FeatureManager>)
        .serve()
        .await
        .expect("start hub")
}

fn base_opts() -> VllmRenderOptions {
    VllmRenderOptions {
        features: vec![],
        role: None,
        kvbm_overrides: vec![],
        kvbm_config: None,
        kv_connector: "DynamoConnector".to_string(),
        kv_role: "kv_both".to_string(),
        kv_load_failure_policy: "recompute".to_string(),
        kv_connector_module_path: "kvbm.v2.vllm.connector".to_string(),
    }
}

fn extract_extra_config(cli: &str) -> serde_json::Value {
    let marker = "--kv-transfer-config '";
    let start = cli.find(marker).expect("has kv-transfer-config") + marker.len();
    let end = cli.rfind('\'').expect("closing quote");
    let tc: serde_json::Value =
        serde_json::from_str(&cli[start..end]).expect("valid transfer json");
    tc["kv_connector_extra_config"].clone()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn render_against_live_hub_indexer_only() {
    let server = start_hub().await;
    let hub_url = format!("http://{}", server.discovery_addr());

    let client = HubClientBuilder::from_url(&hub_url)
        .expect("from_url")
        .build()
        .expect("build client");
    let aggregate = client.get_config().await.expect("get_config");

    let mut opts = base_opts();
    opts.features = vec!["indexer".to_string()];
    let cli = render_vllm_cli(&aggregate, &hub_url, &opts).expect("render");

    assert!(cli.contains(&format!("--block-size {BLOCK_SIZE}")));
    assert!(cli.contains(&format!("--max-model-len {MAX_SEQ_LEN}")));
    let extra = extract_extra_config(&cli);
    assert_eq!(extra["leader"]["hub"]["url"], hub_url);
    assert_eq!(
        extra["leader"]["hub"]["features"],
        serde_json::json!(["indexer"])
    );
    assert_eq!(extra["default"]["block_layout"], "operational");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn render_against_live_hub_cd_dep_closes_p2p() {
    let server = start_hub().await;
    let hub_url = format!("http://{}", server.discovery_addr());

    let client = HubClientBuilder::from_url(&hub_url)
        .expect("from_url")
        .build()
        .expect("build client");
    let aggregate = client.get_config().await.expect("get_config");

    let mut opts = base_opts();
    opts.features = vec!["conditional_disagg".to_string()];
    opts.role = Some("decode".to_string());
    let cli = render_vllm_cli(&aggregate, &hub_url, &opts).expect("render");

    let extra = extract_extra_config(&cli);
    let feats = extra["leader"]["hub"]["features"].as_array().unwrap();
    assert!(feats.contains(&serde_json::json!("p2p")));
    assert!(feats.contains(&serde_json::json!("conditional_disagg")));
    assert_eq!(extra["leader"]["disagg"]["role"], "decode");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn render_against_live_hub_omitted_features_uses_full_enabled_set() {
    let server = start_hub().await;
    let hub_url = format!("http://{}", server.discovery_addr());

    let client = HubClientBuilder::from_url(&hub_url)
        .expect("from_url")
        .build()
        .expect("build client");
    let aggregate = client.get_config().await.expect("get_config");

    // No --features and no --role: CD is enabled on the hub, so the full
    // enabled set includes conditional_disagg and the render must demand a role.
    let err = render_vllm_cli(&aggregate, &hub_url, &base_opts()).unwrap_err();
    assert!(err.to_string().contains("--role"), "got: {err}");
}

// --- FeatureCli: indexer subcommands against a live hub -----------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn indexer_cli_against_live_hub() {
    let server = start_hub().await;
    let hub_url = format!("http://{}", server.discovery_addr());
    let client = HubClientBuilder::from_url(&hub_url)
        .expect("from_url")
        .build()
        .expect("build client");
    let cli = IndexerCli;

    // `config` — primary sizing surfaces through the feature CLI.
    let m = cli
        .command()
        .try_get_matches_from(["indexer", "config"])
        .expect("parse config");
    let v = cli.run(&client, &m).await.expect("config run");
    assert_eq!(v["block_size"], BLOCK_SIZE);
    assert_eq!(v["max_seq_len"], MAX_SEQ_LEN);

    // `get-instances` — fresh hub, nothing registered to the indexer yet.
    let m = cli
        .command()
        .try_get_matches_from(["indexer", "get-instances"])
        .expect("parse get-instances");
    let v = cli.run(&client, &m).await.expect("get-instances run");
    assert_eq!(
        v["instances"].as_array().unwrap().len(),
        0,
        "expected empty registered set, got {v}"
    );

    // `by-pos 0` — fresh index, empty bucket.
    let m = cli
        .command()
        .try_get_matches_from(["indexer", "by-pos", "0"])
        .expect("parse by-pos");
    let v = cli.run(&client, &m).await.expect("by-pos run");
    assert_eq!(v["position"], 0);
    assert_eq!(v["entries"].as_array().unwrap().len(), 0);

    // `query <decimal-u128>` — the CLI packs it into the 16-byte wire shape;
    // a green run proves the hub's byte decoder accepts what we sent (no hit
    // on an empty index).
    let m = cli
        .command()
        .try_get_matches_from(["indexer", "query", "166542759488764189892533901512933376"])
        .expect("parse query");
    let v = cli.run(&client, &m).await.expect("query run");
    assert!(
        v["hit"].is_null(),
        "expected no hit on empty index, got {v}"
    );
}
