// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup KV-index dump endpoint for sibling EPP replicas.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{Json, Router, extract::State, routing::get};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::services::selection::SelectionService;

/// Bind the dump listener before returning so peer recovery cannot race server startup.
pub(crate) async fn spawn(
    service: Arc<SelectionService>,
    port: u16,
    pod_ip: IpAddr,
    cancel: CancellationToken,
) -> Result<()> {
    let address = listener_addr(pod_ip, port);
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("binding EPP peer HTTP server on {address}"))?;
    let app = Router::new().route("/dump", get(dump)).with_state(service);

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

async fn dump(State(service): State<Arc<SelectionService>>) -> Json<serde_json::Value> {
    Json(service.indexer_snapshot().await)
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
