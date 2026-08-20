// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup KV-index dump endpoint for sibling EPP replicas.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
    routing::get,
};
use futures::StreamExt;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::services::indexer::server::StreamDumpRecord;
use dynamo_kv_router::services::selection::SelectionService;

/// Media type of the streaming NDJSON dump (one [`StreamDumpRecord`] per line).
const STREAM_DUMP_MEDIA_TYPE: &str = "application/x-ndjson";

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

    let records = match state.service.indexer_stream_records().await {
        Ok(records) => records,
        Err(error) => {
            tracing::warn!(%error, "Failed to collect peer KV-index dump records");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "peer KV-index snapshot generation failed",
            )
                .into_response();
        }
    };

    // Stream NDJSON, one record per line. The receiver applies each event and
    // drops it, so the whole snapshot is never buffered on either side and no
    // max-bytes budget is needed.
    let stream = futures::stream::iter(records).map(|record: StreamDumpRecord| {
        match serde_json::to_vec(&record) {
            Ok(mut bytes) => {
                bytes.push(b'\n');
                Ok::<_, std::convert::Infallible>(bytes)
            }
            Err(error) => {
                tracing::warn!(%error, "Failed to serialize dump record");
                // Signal a mid-stream failure with an empty frame; the receiver
                // treats a truncated record as an error and retries.
                Ok::<_, std::convert::Infallible>(Vec::new())
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, STREAM_DUMP_MEDIA_TYPE)
        .body(Body::from_stream(stream))
        .expect("static response builder")
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
    async fn streaming_dump_emits_ndjson_records() {
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
            Some("application/x-ndjson")
        );
        let body = resp.text().await.expect("read body");
        // An empty index yields an empty stream; every non-empty line must be a
        // parseable StreamDumpRecord.
        for line in body.lines() {
            if line.trim().is_empty() {
                continue;
            }
            serde_json::from_str::<StreamDumpRecord>(line).expect("valid NDJSON record");
        }

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
