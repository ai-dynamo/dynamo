// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peer KV-index recovery over each peer's `/dump` endpoint.
//!
//! [`recover_from_peers`] consumes the single-JSON-object format served by the
//! standalone indexer / selection / python P2P recovery and by the EPP's peer
//! `/dump`. The EPP reuses this shared path rather than introducing a second
//! wire contract: the receiver buffers the body with a bounded read
//! ([`read_dump_body`]) and applies each event idempotently (the radix index
//! keys blocks by `tokens_hash`).

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
        // Freeze the exact connected listener attempts covered by this dump
        // request. Topology reconciliation may add or restart listeners while
        // the HTTP response is in flight; those attempts must retain normal
        // first-batch gap detection because the snapshot cannot prove that it
        // contains their history.
        let listener_snapshot = registry.snapshot_buffering_listeners();
        match try_recover_from_peer(&client, peer_url, registry).await {
            Ok(()) => {
                listener_snapshot.mark_snapshot_bootstrapped();
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

    let mut resp = client
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

    // Read the body as a bounded stream: a chunked response without
    // Content-Length must not buffer without bound (resp.json() would).
    // `0` disables the cap, consistent with the documented behavior.
    let body = read_dump_body(&mut resp, max_dump_bytes).await?;

    let dump: HashMap<String, DumpEntry> =
        serde_json::from_slice(&body).context("failed to parse dump response")?;
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

        indexer
            .flush_pending()
            .await
            .context("failed to flush peer recovery events")?;
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

/// Read the `/dump` response body as a bounded stream, failing as soon as the
/// configured cap is exceeded so a chunked response without `Content-Length`
/// cannot buffer without bound. `max_dump_bytes == 0` disables the cap,
/// consistent with the documented behavior.
async fn read_dump_body(resp: &mut reqwest::Response, max_dump_bytes: u64) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .context("failed to read dump response body")?
    {
        if max_dump_bytes > 0
            && (body.len() as u64).saturating_add(chunk.len() as u64) > max_dump_bytes
        {
            anyhow::bail!(
                "peer dump is too large: exceeds limit {max_dump_bytes} bytes \
                 (raise {MAX_DUMP_BYTES_ENV} to accept larger snapshots)"
            );
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
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

    /// Serve one `chunked` HTTP response with no `Content-Length` header, the
    /// shape `read_dump_body` must bound without relying on the header.
    async fn serve_chunked_without_content_length(body: &[u8]) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("read test addr");
        let body = body.to_vec();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept test client");
            // Drain the request line + headers.
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await.expect("read request");
            // Transfer-Encoding: chunked, deliberately no Content-Length.
            let mut resp = String::from(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n",
            );
            for chunk in body.chunks(7) {
                resp.push_str(&format!("{:x}\r\n", chunk.len()));
                resp.push_str(&String::from_utf8_lossy(chunk));
                resp.push_str("\r\n");
            }
            resp.push_str("0\r\n\r\n");
            socket
                .write_all(resp.as_bytes())
                .await
                .expect("write response");
        });
        let url = format!("http://{addr}/dump");
        // Detach: the server task finishes once the client has consumed the
        // body; the test runtime aborts it when the test ends.
        drop(server);
        url
    }

    #[tokio::test]
    async fn read_dump_body_bounds_chunked_response_without_content_length() {
        // Small body, cap not hit: bounded read succeeds and returns the body.
        let url =
            serve_chunked_without_content_length(br#"{"m:default":{"block_size":16,"events":[]}}"#)
                .await;
        let client = reqwest::Client::new();
        let mut resp = client.get(&url).send().await.expect("GET dump");
        assert!(
            resp.content_length().is_none(),
            "fixture must not declare Content-Length"
        );
        let body = read_dump_body(&mut resp, 1024)
            .await
            .expect("bounded read under cap");
        assert_eq!(
            body,
            br#"{"m:default":{"block_size":16,"events":[]}}"#.to_vec()
        );
    }

    #[tokio::test]
    async fn read_dump_body_rejects_oversized_chunked_response_without_content_length() {
        // Body larger than the cap, no Content-Length: the bounded reader must
        // fail instead of buffering without bound (resp.json() would).
        let url = serve_chunked_without_content_length(&[b'x'; 64]).await;
        let client = reqwest::Client::new();
        let mut resp = client.get(&url).send().await.expect("GET dump");
        assert!(resp.content_length().is_none());
        let err = read_dump_body(&mut resp, 32)
            .await
            .expect_err("cap must be enforced without Content-Length");
        assert!(
            err.to_string().contains("peer dump is too large"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn read_dump_body_zero_cap_is_unbounded() {
        // max_dump_bytes == 0 disables the cap, even for a body larger than
        // any plausible default and with no Content-Length header.
        let url = serve_chunked_without_content_length(&[b'x'; 512]).await;
        let client = reqwest::Client::new();
        let mut resp = client.get(&url).send().await.expect("GET dump");
        let body = read_dump_body(&mut resp, 0)
            .await
            .expect("zero disables the cap");
        assert_eq!(body.len(), 512);
    }
}
