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
use dynamo_ext_proc::{EppMode, EppStandaloneConfig, ExtProcServer, Router};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

const GRPC_PORT: u16 = 9002;
const HEALTH_PORT: u16 = 9003;
const HEALTH_SERVICE_NAME: &str = "inference-extension";
/// Cap concurrent in-flight TLS handshakes + active gRPC streams. Prevents a
/// connection flood from exhausting fds / memory. Tuned for an inference EPP
/// where a single Envoy upstream typically holds <100 concurrent streams.
const MAX_CONCURRENT_CONNECTIONS: usize = 1024;
/// Max time to wait for the TLS handshake to complete before dropping the
/// connection. Without this, a client that finishes the TCP connect but
/// stalls the TLS handshake holds a connection-limit permit indefinitely;
/// enough such stalls exhaust all permits and starve legitimate ext_proc
/// traffic (slowloris-style). Only the handshake is bounded — established
/// connections may stay open for the lifetime of their bidi stream.
const TLS_HANDSHAKE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

struct Config {
    namespace: String,
    component: String,
}

impl Config {
    fn from_env() -> Self {
        let namespace = env_or("DYN_NAMESPACE_PREFIX", "")
            .or_else(|| env_or("DYN_NAMESPACE", ""))
            .unwrap_or_else(|| "vllm-agg".to_string());

        if parse_env("DYN_ENFORCE_DISAGG", false) {
            tracing::warn!(
                "DYN_ENFORCE_DISAGG is deprecated and ignored; routing topology and readiness are determined from registered worker types"
            );
        }

        Self {
            namespace,
            component: env_or("DYN_COMPONENT_NAME", "").unwrap_or_else(|| "backend".to_string()),
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

    // Build with an explicit crypto provider. This crate compiles in BOTH
    // rustls providers via feature unification (our direct `ring` feature plus
    // `aws-lc-rs` pulled in transitively by `kube`), so the parameterless
    // `ServerConfig::builder()` cannot auto-select a process-default provider
    // and would panic. Pin to `ring`, matching the rustls feature we enable
    // for our own serving path.
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut tls_config = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
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

    let standalone = matches!(EppMode::from_env()?, EppMode::Standalone);

    let config = Config::from_env();

    tracing::info!(
        port = GRPC_PORT,
        health_port = HEALTH_PORT,
        namespace = %config.namespace,
        component = %config.component,
        standalone,
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

    if standalone {
        let selector_cfg = EppStandaloneConfig::from_env()?;
        tracing::info!(
            inference_pool_name = %selector_cfg.inference_pool_name,
            model_name = %selector_cfg.model_name,
            block_size = selector_cfg.block_size,
            "Initializing standalone selector mode (no Dynamo runtime)..."
        );
        let router = dynamo_ext_proc::EppRouter::from_selector(selector_cfg).await?;
        let ready = router.reflector_ready();
        serve(Arc::new(router), ready, health_reporter).await
    } else {
        tracing::info!("Initializing KV-aware router from discovery...");
        let router = Router::from_discovery(&config.namespace, &config.component).await?;
        let ready = router.pod_store_ready();
        serve(Arc::new(router), ready, health_reporter).await
    }
}

/// Gate gRPC health SERVING on `ready`, then serve the ext_proc endpoint with
/// the given picker. Shared by both Dynamo-discovery and standalone modes.
async fn serve<P: dynamo_ext_proc::EndpointPicker>(
    picker: Arc<P>,
    ready: Arc<std::sync::atomic::AtomicBool>,
    health_reporter: tonic_health::server::HealthReporter,
) -> Result<()> {
    // Gate SERVING on the picker's readiness flag. The router constructor may
    // return before the pod reflector's initial LIST lands; `pick()` returns a
    // routing error until then, so reporting SERVING unconditionally would
    // advertise a healthy pod that rejects every request. Flip to SERVING once
    // the reflector cache is usable; in the common path this is immediate.
    if ready.load(std::sync::atomic::Ordering::Acquire) {
        health_reporter
            .set_service_status(HEALTH_SERVICE_NAME, tonic_health::ServingStatus::Serving)
            .await;
        tracing::info!("Router initialized, health status set to SERVING");
    } else {
        tracing::warn!(
            "Router initialized but pod reflector cache not ready yet; \
             keeping health NOT_SERVING until the initial LIST completes"
        );
        let health_reporter = health_reporter.clone();
        let ready = ready.clone();
        tokio::spawn(async move {
            while !ready.load(std::sync::atomic::Ordering::Acquire) {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
            health_reporter
                .set_service_status(HEALTH_SERVICE_NAME, tonic_health::ServingStatus::Serving)
                .await;
            tracing::info!("Pod reflector now ready, health status set to SERVING");
        });
    }

    let server = ExtProcServer::new(picker);
    // Default to TLS to match the Go EPP behavior. Verified working with
    // kGateway (`appProtocol: http2` upstreams negotiate h2 over TLS via ALPN
    // when the cert is presented). Set DYN_SECURE_SERVING=false to fall back
    // to plaintext h2c, e.g. for local debugging or non-TLS gateways.
    let secure_serving = parse_env("DYN_SECURE_SERVING", true);
    let addr: std::net::SocketAddr = format!("0.0.0.0:{GRPC_PORT}").parse()?;

    if secure_serving {
        let tls_acceptor = create_tls_acceptor()?;
        let svc = server.into_service();
        let listener = TcpListener::bind(addr).await?;
        let conn_semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_CONNECTIONS));
        tracing::info!(
            %addr,
            max_connections = MAX_CONCURRENT_CONNECTIONS,
            "Listening for ext_proc connections (TLS)"
        );

        loop {
            // Acquire permit before accept() so we backpressure the listener
            // instead of accepting and immediately dropping connections.
            let permit = conn_semaphore.clone().acquire_owned().await?;
            let (tcp_stream, remote_addr) = listener.accept().await?;
            let tls_acceptor = tls_acceptor.clone();
            let svc = svc.clone();

            tokio::spawn(async move {
                let _permit = permit; // released when this task exits (incl. handshake timeout)
                let tls_stream = match tokio::time::timeout(
                    TLS_HANDSHAKE_TIMEOUT,
                    tls_acceptor.accept(tcp_stream),
                )
                .await
                {
                    Ok(Ok(s)) => s,
                    Ok(Err(e)) => {
                        tracing::debug!(%remote_addr, error = %e, "TLS handshake failed");
                        return;
                    }
                    Err(_) => {
                        tracing::debug!(
                            %remote_addr,
                            timeout_secs = TLS_HANDSHAKE_TIMEOUT.as_secs(),
                            "TLS handshake timed out; dropping connection"
                        );
                        return;
                    }
                };

                let io = hyper_util::rt::TokioIo::new(tls_stream);
                let hyper_svc = hyper_util::service::TowerToHyperService::new(svc);
                if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                    hyper_util::rt::TokioExecutor::new(),
                )
                .serve_connection(io, hyper_svc)
                .await
                {
                    tracing::debug!(%remote_addr, error = %e, "Connection ended");
                }
            });
        }
    } else {
        tracing::info!(%addr, "Listening for ext_proc connections (plaintext h2)");
        tonic::transport::Server::builder()
            .add_service(server.into_service())
            .serve(addr)
            .await?;
        Ok(())
    }
}
