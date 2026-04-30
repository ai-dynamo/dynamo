// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone Rust EPP binary.
//!
//! Replaces the Go EPP + CGO bridge with a single native Rust binary that
//! implements the Envoy ext_proc gRPC service and uses Dynamo's KV-aware
//! router for endpoint selection.
//!
//! The ext-proc port (9002) serves TLS (self-signed cert, matching the Go EPP).
//! The health port (9003) is plaintext (K8s probes don't need TLS).

use std::sync::Arc;

use anyhow::Result;
use dynamo_ext_proc::{ExtProcServer, Router};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

const GRPC_PORT: u16 = 9002;
const HEALTH_PORT: u16 = 9003;
const HEALTH_SERVICE_NAME: &str = "inference-extension";

struct Config {
    namespace: String,
    component: String,
    enforce_disagg: bool,
}

impl Config {
    fn from_env() -> Self {
        let namespace = env_or("DYN_NAMESPACE_PREFIX", "")
            .or_else(|| env_or("DYN_NAMESPACE", ""))
            .unwrap_or_else(|| "vllm-agg".to_string());

        Self {
            namespace,
            component: env_or("DYN_COMPONENT_NAME", "").unwrap_or_else(|| "backend".to_string()),
            enforce_disagg: parse_env("DYN_ENFORCE_DISAGG", false),
        }
    }
}

fn env_or(key: &str, empty_means_unset: &str) -> Option<String> {
    std::env::var(key).ok().and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() || trimmed == empty_means_unset {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn parse_env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Generate a self-signed TLS acceptor for the ext-proc gRPC server.
fn create_tls_acceptor() -> Result<TlsAcceptor> {
    use rcgen::{CertificateParams, KeyPair};
    use rustls::ServerConfig;
    use tokio_rustls::rustls;

    let key_pair = KeyPair::generate()?;
    let mut params = CertificateParams::new(vec!["localhost".to_string()])?;
    params
        .subject_alt_names
        .push(rcgen::SanType::IpAddress(std::net::IpAddr::V4(
            std::net::Ipv4Addr::UNSPECIFIED,
        )));
    let cert = params.self_signed(&key_pair)?;

    let cert_pem = cert.pem();
    let key_pem = key_pair.serialize_pem();

    let certs = rustls_pemfile::certs(&mut cert_pem.as_bytes())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let key = rustls_pemfile::private_key(&mut key_pem.as_bytes())?
        .ok_or_else(|| anyhow::anyhow!("No private key found in PEM"))?;

    let mut tls_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?;
    tls_config.alpn_protocols = vec![b"h2".to_vec()];

    tracing::info!("Generated self-signed TLS certificate for ext-proc server");
    Ok(TlsAcceptor::from(Arc::new(tls_config)))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env();

    tracing::info!(
        port = GRPC_PORT,
        health_port = HEALTH_PORT,
        namespace = %config.namespace,
        component = %config.component,
        enforce_disagg = config.enforce_disagg,
        "Starting Dynamo Rust EPP"
    );

    // Start plaintext gRPC health server immediately (NOT_SERVING until router ready).
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_service_status(HEALTH_SERVICE_NAME, tonic_health::ServingStatus::NotServing)
        .await;

    let health_addr = format!("0.0.0.0:{HEALTH_PORT}").parse()?;
    tracing::info!(%health_addr, "Starting gRPC health server (plaintext)");
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(health_service)
            .serve(health_addr),
    );

    tracing::info!("Initializing KV-aware router from discovery...");
    let router =
        Router::from_discovery(&config.namespace, &config.component, config.enforce_disagg).await?;

    health_reporter
        .set_service_status(HEALTH_SERVICE_NAME, tonic_health::ServingStatus::Serving)
        .await;
    tracing::info!("Router initialized, health status set to SERVING");

    // Serve ext-proc over TLS on port 9002.
    let tls_acceptor = create_tls_acceptor()?;
    let picker = Arc::new(router);
    let server = ExtProcServer::new(picker);
    let svc = server.into_service();

    let listener = TcpListener::bind(format!("0.0.0.0:{GRPC_PORT}")).await?;
    tracing::info!(port = GRPC_PORT, "Listening for ext_proc connections (TLS)");

    loop {
        let (tcp_stream, remote_addr) = listener.accept().await?;
        let tls_acceptor = tls_acceptor.clone();
        let svc = svc.clone();

        tokio::spawn(async move {
            let tls_stream = match tls_acceptor.accept(tcp_stream).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::debug!(%remote_addr, error = %e, "TLS handshake failed");
                    return;
                }
            };

            let io = hyper_util::rt::TokioIo::new(tls_stream);
            let hyper_svc = hyper_util::service::TowerToHyperService::new(svc);
            if let Err(e) =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .serve_connection(io, hyper_svc)
                    .await
            {
                tracing::debug!(%remote_addr, error = %e, "Connection ended");
            }
        });
    }
}
