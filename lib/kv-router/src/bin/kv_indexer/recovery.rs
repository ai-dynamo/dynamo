// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Deserialize;

use dynamo_kv_router::protocols::RouterEvent;

use super::registry::{IndexerKey, WorkerRegistry};

#[derive(Deserialize)]
struct DumpEntry {
    block_size: u32,
    events: Vec<RouterEvent>,
}

pub async fn recover_from_peers(peers: &[String], registry: &WorkerRegistry) -> Result<bool> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("failed to build HTTP client")?;

    // Brief delay to let ZMQ subscription handshakes complete. Workers were
    // registered (ZMQ sockets connected) before this function is called, but
    // the PUB-side subscription processing is asynchronous. Without this
    // delay, the PUB may not yet know about our SUB sockets and would
    // silently drop events sent during the handshake window.
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    for peer_url in peers {
        match try_recover_from_peer(&client, peer_url, registry).await {
            Ok(()) => {
                tracing::info!(peer = %peer_url, "recovery from peer succeeded");
                return Ok(true);
            }
            Err(e) => {
                tracing::warn!(peer = %peer_url, error = %e, "recovery from peer failed, trying next");
            }
        }
    }

    Ok(false)
}

async fn try_recover_from_peer(
    client: &reqwest::Client,
    peer_url: &str,
    registry: &WorkerRegistry,
) -> Result<()> {
    let dump_url = format!("{peer_url}/dump");
    tracing::info!(url = %dump_url, "fetching dump from peer");

    let resp = client
        .get(&dump_url)
        .send()
        .await
        .context("HTTP request failed")?;

    if !resp.status().is_success() {
        anyhow::bail!("peer returned status {}", resp.status());
    }

    let dump: HashMap<String, DumpEntry> =
        resp.json().await.context("failed to parse dump response")?;

    let mut total_events = 0usize;
    for (map_key, entry) in dump {
        let (model_name, tenant_id) = map_key
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("invalid dump key format: {map_key}"))?;

        let key = IndexerKey {
            model_name: model_name.to_string(),
            tenant_id: tenant_id.to_string(),
        };

        let indexer = registry.get_or_create_indexer(key, entry.block_size);

        for event in entry.events {
            indexer.apply_event(event).await;
            total_events += 1;
        }
    }

    tracing::info!(total_events, "applied dump events from peer");
    Ok(())
}
