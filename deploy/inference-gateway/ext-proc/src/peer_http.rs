// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup KV-index dump endpoint for sibling EPP replicas.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::services::selection::SelectionService;

#[derive(Clone)]
struct AppState {
    service: Arc<SelectionService>,
    recovered: Arc<AtomicBool>,
}

#[derive(serde::Deserialize)]
struct DumpQuery {
    /// Caller's accepted snapshot budget in bytes. The peer rejects over-budget
    /// snapshots with 413 *before* writing the body. Absent or `0` = unbounded.
    max_bytes: Option<u64>,
}

/// Bind the dump listener before returning so peer recovery cannot race server startup.
pub(crate) async fn spawn(
    service: Arc<SelectionService>,
    port: u16,
    pod_ip: IpAddr,
    cancel: CancellationToken,
    recovered: Arc<AtomicBool>,
) -> Result<()> {
    let address = listener_addr(pod_ip, port);
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("binding EPP peer HTTP server on {address}"))?;
    let state = AppState { service, recovered };
    let app = Router::new().route("/dump", get(dump)).with_state(state);

    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app)
            .with_graceful_shutdown(cancel.cancelled_owned())
            .await
        {
            tracing::error!(%error, port, "EPP peer HTTP server exited");
        }
    });
    Ok(())
}

fn listener_addr(pod_ip: IpAddr, port: u16) -> SocketAddr {
    match pod_ip {
        IpAddr::V4(_) => SocketAddr::from((Ipv4Addr::UNSPECIFIED, port)),
        IpAddr::V6(_) => SocketAddr::from((Ipv6Addr::UNSPECIFIED, port)),
    }
}

/// True when the indexer snapshot contains a per-model `{"error": ...}`
/// entry, which happens when one model's indexer failed to dump. The
/// recovery consumer deserializes each entry as `DumpEntry` (`block_size`
/// plus `events`), so an error entry would make the whole body unparseable;
/// the dump handler fails such snapshots with a non-success status.
fn snapshot_has_failed_dump(snapshot: &serde_json::Value) -> bool {
    snapshot
        .as_object()
        .is_some_and(|entries| entries.values().any(|entry| entry.get("error").is_some()))
}

async fn dump(State(state): State<AppState>, Query(query): Query<DumpQuery>) -> Response {
    // Do not serve a snapshot until local recovery/bootstrap has finished: the
    // index is empty during recovery, and an early /dump could let a sibling
    // latch onto an empty index while a warm one exists elsewhere.
    if !state.recovered.load(Ordering::Acquire) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "peer KV index not yet recovered",
        )
            .into_response();
    }
    let snapshot = state.service.indexer_snapshot().await;
    // A per-model indexer dump failure is surfaced as an `{"error": ...}`
    // entry inside the snapshot. The recovery consumer expects `DumpEntry`
    // values and cannot parse that shape, so fail the whole dump with a
    // non-success status instead of returning a 200 body the consumer would
    // reject (and then fall back to an empty index on).
    if snapshot_has_failed_dump(&snapshot) {
        tracing::warn!("Peer KV-index snapshot contains an indexer dump failure");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "peer KV-index snapshot generation failed",
        )
            .into_response();
    }
    let bytes = match serde_json::to_vec(&snapshot) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, "Failed to serialize peer KV-index snapshot");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "snapshot serialization failed",
            )
                .into_response();
        }
    };
    if let Some(max) = query.max_bytes
        && max > 0
        && bytes.len() as u64 > max
    {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            "peer KV index snapshot exceeds max_bytes",
        )
            .into_response();
    }
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        bytes,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::config::KvRouterConfig;
    use dynamo_kv_router::services::selection::SelectionServiceBuilder;

    async fn service() -> Arc<SelectionService> {
        Arc::new(
            SelectionServiceBuilder::new(KvRouterConfig::default())
                .indexer_threads(1)
                .build()
                .await
                .expect("build selection service"),
        )
    }

    fn free_tcp_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("reserve port")
            .local_addr()
            .expect("read local address")
            .port()
    }

    #[tokio::test]
    async fn dump_listener_is_bound_before_spawn_returns() {
        let service = service().await;
        let cancel = CancellationToken::new();
        let port = free_tcp_port();

        spawn(
            service.clone(),
            port,
            "127.0.0.1".parse().unwrap(),
            cancel.clone(),
            Arc::new(AtomicBool::new(true)),
        )
        .await
        .expect("spawn peer HTTP server");
        tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .expect("listener must already be bound");

        cancel.cancel();
        service.shutdown().await;
    }

    #[tokio::test]
    async fn dump_endpoint_matches_selection_service_snapshot() {
        let service = service().await;
        let cancel = CancellationToken::new();
        let port = free_tcp_port();
        spawn(
            service.clone(),
            port,
            "127.0.0.1".parse().unwrap(),
            cancel.clone(),
            Arc::new(AtomicBool::new(true)),
        )
        .await
        .expect("spawn peer HTTP server");

        let response: serde_json::Value = reqwest::get(format!("http://127.0.0.1:{port}/dump"))
            .await
            .expect("request dump")
            .json()
            .await
            .expect("decode dump");
        assert_eq!(response, service.indexer_snapshot().await);

        cancel.cancel();
        service.shutdown().await;
    }

    #[tokio::test]
    async fn dump_returns_503_until_recovered() {
        let service = service().await;
        let cancel = CancellationToken::new();
        let port = free_tcp_port();
        spawn(
            service.clone(),
            port,
            "127.0.0.1".parse().unwrap(),
            cancel.clone(),
            Arc::new(AtomicBool::new(false)),
        )
        .await
        .expect("spawn peer HTTP server");

        let resp = reqwest::get(format!("http://127.0.0.1:{port}/dump"))
            .await
            .expect("request dump");
        assert_eq!(resp.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);

        cancel.cancel();
        service.shutdown().await;
    }

    #[tokio::test]
    async fn dump_rejects_over_budget_snapshot() {
        let service = service().await;
        let cancel = CancellationToken::new();
        let port = free_tcp_port();
        spawn(
            service.clone(),
            port,
            "127.0.0.1".parse().unwrap(),
            cancel.clone(),
            Arc::new(AtomicBool::new(true)),
        )
        .await
        .expect("spawn peer HTTP server");

        // max_bytes=1 is smaller than any serialized snapshot ("{}" is 2 bytes).
        let resp = reqwest::get(format!("http://127.0.0.1:{port}/dump?max_bytes=1"))
            .await
            .expect("request dump");
        assert_eq!(resp.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);

        cancel.cancel();
        service.shutdown().await;
    }

    #[test]
    fn listener_addr_matches_pod_ip_family() {
        assert_eq!(
            listener_addr("192.0.2.1".parse().unwrap(), 9093),
            "0.0.0.0:9093".parse().unwrap()
        );
        assert_eq!(
            listener_addr("2001:db8::1".parse().unwrap(), 9093),
            "[::]:9093".parse().unwrap()
        );
    }

    #[test]
    fn snapshot_with_failed_indexer_dump_is_detected() {
        // A per-model indexer dump failure surfaces as `{"error": ...}`; the
        // recovery consumer expects `DumpEntry` values, so such snapshots must
        // be rejected as a whole.
        let failed = serde_json::json!({
            "model:default": {"error": "indexer dump failed"},
        });
        assert!(snapshot_has_failed_dump(&failed));

        let mixed = serde_json::json!({
            "model:a": {"block_size": 16, "events": []},
            "model:b": {"error": "boom"},
        });
        assert!(snapshot_has_failed_dump(&mixed));
    }

    #[test]
    fn snapshot_without_failed_indexer_dump_is_accepted() {
        let healthy = serde_json::json!({
            "model:default": {"block_size": 16, "events": []},
        });
        assert!(!snapshot_has_failed_dump(&healthy));

        let empty = serde_json::json!({});
        assert!(!snapshot_has_failed_dump(&empty));
    }

    #[tokio::test]
    async fn ipv6_pod_binds_an_ipv6_listener() {
        let listener = TcpListener::bind(listener_addr("::1".parse().unwrap(), 0))
            .await
            .expect("IPv6 loopback listener should bind");
        assert!(listener.local_addr().unwrap().is_ipv6());
    }
}
