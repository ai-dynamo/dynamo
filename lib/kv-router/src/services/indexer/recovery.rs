// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::identity::RoutingPartitionId;
use crate::protocols::RouterEvent;

use super::registry::WorkerRegistry;

/// Timeout for one peer `/dump` fetch (HTTP request + body transfer). A full
/// KV-index snapshot can be tens of MB on a busy deployment; 10s only fits
/// small indexes once serialization and parsing are counted. Configurable via
/// `DYN_EPP_RECOVERY_HTTP_TIMEOUT_MS` for larger clusters.
const DEFAULT_RECOVERY_HTTP_TIMEOUT_MS: u64 = 30_000;
const RECOVERY_HTTP_TIMEOUT_ENV: &str = "DYN_EPP_RECOVERY_HTTP_TIMEOUT_MS";
/// Safety ceiling on the accepted `/dump` response body. The whole snapshot is
/// materialized in memory on both sides today (streaming is a follow-up), so
/// an unbounded body either OOMs or trips the timeout and then the retry loop.
/// Fail fast with a clear error instead. Configurable via
/// `DYN_EPP_RECOVERY_MAX_DUMP_BYTES`.
const DEFAULT_MAX_DUMP_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DUMP_BYTES_ENV: &str = "DYN_EPP_RECOVERY_MAX_DUMP_BYTES";

#[derive(Deserialize)]
struct DumpEntry {
    block_size: u32,
    events: Vec<RouterEvent>,
}

fn parse_u64(value: Option<String>, default: u64) -> u64 {
    value
        .as_deref()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    parse_u64(std::env::var(key).ok(), default)
}

pub async fn recover_from_peers(peers: &[String], registry: &WorkerRegistry) -> Result<bool> {
    let timeout = Duration::from_millis(env_u64(
        RECOVERY_HTTP_TIMEOUT_ENV,
        DEFAULT_RECOVERY_HTTP_TIMEOUT_MS,
    ));
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .context("failed to build HTTP client")?;

    tokio::time::sleep(Duration::from_secs(1)).await;

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
    // Pass the accepted budget to the peer so it can reject an over-budget
    // snapshot with 413 *before* serializing/transmitting it. `0` means no
    // budget (unbounded). The Content-Length check below remains as receiver
    // defense in depth.
    let max_dump_bytes = env_u64(MAX_DUMP_BYTES_ENV, DEFAULT_MAX_DUMP_BYTES);
    let dump_url = if max_dump_bytes > 0 {
        format!("{peer_url}/dump?max_bytes={max_dump_bytes}")
    } else {
        format!("{peer_url}/dump")
    };
    tracing::info!(url = %dump_url, "fetching dump from peer");

    let resp = client
        .get(&dump_url)
        .send()
        .await
        .context("HTTP request failed")?;

    if !resp.status().is_success() {
        anyhow::bail!("peer returned status {}", resp.status());
    }

    // Fail fast on an oversized snapshot before reading the body: the dump is
    // materialized fully in memory on both sides, so a large body either OOMs
    // or trips the request timeout and the retry loop. The peer already rejects
    // over-budget bodies with 413, so this only fires for a peer without the
    // budget-aware endpoint.
    if let Some(len) = resp.content_length()
        && max_dump_bytes > 0
        && len > max_dump_bytes
    {
        anyhow::bail!(
            "peer dump is too large: {len} bytes exceeds limit {max_dump_bytes} \
             (raise {MAX_DUMP_BYTES_ENV} to accept larger snapshots)"
        );
    }

    let dump: HashMap<String, DumpEntry> =
        resp.json().await.context("failed to parse dump response")?;
    let mut total_events = 0usize;
    for (map_key, entry) in dump {
        let (model_name, routing_group) = map_key
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("invalid dump key format: {map_key}"))?;

        let key = RoutingPartitionId::new(model_name, routing_group);

        let indexer = registry.get_or_create_indexer(key, entry.block_size);

        for event in entry.events {
            // Use the tier-aware dispatcher so HostPinned/Disk events from the
            // peer's dump land in the matching lower-tier slot rather than the
            // device primary. The peer side retags lower-tier events in
            // `Indexer::dump_events`, so the `storage_tier` here is correct.
            //
            // Re-application is idempotent by construction: the radix index
            // keys blocks by `tokens_hash`, so applying the same (or an
            // overlapping) peer dump again is a no-op for blocks already
            // present. A cancelled-then-retried attempt can therefore leave a
            // partially applied snapshot that the next attempt safely re-applies
            // on top of, without inflating or duplicating residency state.
            indexer
                .apply_event_routed(event)
                .await
                .context("peer recovery event was rejected by the local indexer")?;
            total_events += 1;
        }
    }

    // An empty dump is a valid recovery. Recovery candidates are restricted to
    // already-serving peers (see `recovery_peer_urls`), so a zero-event dump
    // means the serving peer genuinely holds no KV index yet (idle cluster),
    // not a transient race. Rejecting it would deadlock cold starts and idle
    // rollouts: the joining replica would wait forever for events that no
    // serving peer holds.
    tracing::info!(total_events, "applied dump events from peer");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_u64_falls_back_on_missing_or_invalid() {
        assert_eq!(parse_u64(None, 30_000), 30_000);
        assert_eq!(parse_u64(Some("not-a-number".to_string()), 30_000), 30_000);
        assert_eq!(parse_u64(Some(" 123 ".to_string()), 30_000), 123);
        assert_eq!(parse_u64(Some("0".to_string()), 30_000), 0);
    }

    #[test]
    fn parse_u64_accepts_zero() {
        // Zero is a valid explicit value (e.g. disabling a cap); it must not be
        // treated as a parse failure that falls back to the default.
        assert_eq!(parse_u64(Some("0".to_string()), 512), 0);
    }
}
