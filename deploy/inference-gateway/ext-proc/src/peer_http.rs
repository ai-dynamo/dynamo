// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup KV-index dump endpoint for sibling EPP replicas.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
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

async fn dump(State(state): State<AppState>) -> Response {
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
    // `dump_registry` embeds per-model `{"error": ...}` entries when an indexer
    // dump fails; surface those as a whole-snapshot 500 so the recovery consumer
    // does not deserialize an incompatible entry (and fall back to empty state).
    let failed = snapshot
        .as_object()
        .into_iter()
        .flatten()
        .any(|(_key, entry)| entry.get("error").is_some());
    if failed {
        tracing::warn!("peer KV-index snapshot contains a failed indexer dump");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "peer KV-index snapshot generation failed",
        )
            .into_response();
    }

    (StatusCode::OK, Json(snapshot)).into_response()
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
    async fn dump_returns_json_snapshot() {
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

        let resp = reqwest::get(format!("http://127.0.0.1:{port}/dump"))
            .await
            .expect("request dump");
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        let body = resp.text().await.expect("read body");
        // An empty index yields an empty snapshot object; it must parse as the
        // `HashMap<String, DumpEntry>` shape `recover_from_peers` consumes.
        let snapshot: serde_json::Value =
            serde_json::from_str(&body).expect("dump body must be valid JSON");
        assert!(
            snapshot.is_object(),
            "snapshot must be a JSON object, got: {snapshot}"
        );

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

    #[tokio::test]
    async fn ipv6_pod_binds_an_ipv6_listener() {
        let listener = TcpListener::bind(listener_addr("::1".parse().unwrap(), 0))
            .await
            .expect("IPv6 loopback listener should bind");
        assert!(listener.local_addr().unwrap().is_ipv6());
    }
}
