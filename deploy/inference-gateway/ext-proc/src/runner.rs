// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standard Rust EPP process bootstrap.
//!
//! Replaces the Go EPP + CGO bridge with a single native Rust binary that
//! implements the Envoy ext_proc gRPC service and uses Dynamo's KV-aware
//! router for endpoint selection.
//!
//! The ext-proc port (9002) serves TLS (self-signed cert, matching the Go EPP).
//! The health port (9003) is plaintext (K8s probes don't need TLS).

use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;
use tokio_util::sync::CancellationToken;

use crate::{EppMode, EppStandaloneConfig, ExtProcServer, Router, metrics};

const GRPC_PORT: u16 = 9002;
const HEALTH_PORT: u16 = 9003;
const HEALTH_SERVICE_NAME: &str = "inference-extension";
/// Cap concurrent in-flight TLS handshakes + active gRPC streams. Prevents a
/// connection flood from exhausting fds / memory. Tuned for an inference EPP
/// where a single Envoy upstream typically holds <100 concurrent streams.
const MAX_CONCURRENT_CONNECTIONS: usize = 1024;
/// Propagation window after a shutdown signal before the server stops
/// accepting new connections. The gateway stops routing to this EPP as soon
/// as health flips to NOT_SERVING. Configurable via
/// `DYN_EPP_GRACEFUL_SHUTDOWN_PROPAGATION_SECS`.
const DEFAULT_GRACEFUL_SHUTDOWN_PROPAGATION_SECS: u64 = 5;
const GRACEFUL_SHUTDOWN_PROPAGATION_ENV: &str = "DYN_EPP_GRACEFUL_SHUTDOWN_PROPAGATION_SECS";
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

/// Run EPP until it exits.
pub async fn run(policy_registry: Option<WorkerSelectionPolicyRegistry>) -> Result<()> {
    init_tracing();
    let mode = EppMode::from_env()?;
    if !matches!(mode, EppMode::Standalone)
        && policy_registry
            .as_ref()
            .is_some_and(|registry| !registry.is_empty())
    {
        anyhow::bail!("linked worker-selection policies require DYN_EPP_MODE=standalone")
    }
    run_inner(mode, policy_registry).await
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

/// Wait for SIGTERM (kubelet pod termination) or SIGINT (Ctrl-C).
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

async fn run_inner(
    mode: EppMode,
    policy_registry: Option<WorkerSelectionPolicyRegistry>,
) -> Result<()> {
    let standalone = matches!(mode, EppMode::Standalone);

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

    // Start before router init so scrapes during a slow discovery bootstrap
    // return an empty exposition rather than a connection refusal.
    let metrics_port = parse_env(metrics::METRICS_PORT_ENV, metrics::DEFAULT_METRICS_PORT);
    if metrics_port == 0 {
        tracing::info!("Metrics server disabled");
    } else {
        tokio::spawn(async move {
            if let Err(e) = metrics::serve(metrics_port).await {
                tracing::error!(error = %e, port = metrics_port, "Metrics server exited");
            }
        });
    }

    // Shutdown coordination: on SIGTERM/SIGINT, flip health to NOT_SERVING
    // (the gateway stops routing new requests to this EPP), allow the endpoint
    // propagation window to elapse, then stop accepting connections. The
    // protocol-correct drain deadline and forced close of long-lived HTTP/2
    // connections are handled by the follow-up connection-lifecycle work.
    let draining = CancellationToken::new();
    let shutdown = CancellationToken::new();
    {
        let draining = draining.clone();
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            tracing::info!("Shutdown signal received; starting endpoint withdrawal");
            // The readiness mirror owns the health transition so it cannot
            // race with a final SERVING update. If initialization has not
            // reached `serve` yet, health is already NOT_SERVING and the
            // draining cancellation below makes initialization return.
            draining.cancel();
            let propagation_secs = parse_env(
                GRACEFUL_SHUTDOWN_PROPAGATION_ENV,
                DEFAULT_GRACEFUL_SHUTDOWN_PROPAGATION_SECS,
            );
            tracing::info!(
                propagation_secs,
                "EPP health set to NOT_SERVING; allowing endpoint propagation"
            );
            tokio::time::sleep(std::time::Duration::from_secs(propagation_secs)).await;
            shutdown.cancel();
            tracing::info!("EPP endpoint propagation complete; stopping accepts");
        });
    }

    if standalone {
        let selector_cfg = EppStandaloneConfig::from_env()?;
        tracing::info!(
            inference_pool_name = %selector_cfg.inference_pool_name,
            model_name = %selector_cfg.model_name,
            block_size = selector_cfg.block_size,
            "Initializing standalone selector mode (no Dynamo runtime)..."
        );
        metrics::set_served_model(&selector_cfg.model_name);
        let router = tokio::select! {
            _ = draining.cancelled() => {
                tracing::info!("Shutdown received during standalone EPP initialization");
                return Ok(());
            }
            router = crate::EppRouter::from_selector(selector_cfg, policy_registry) => Arc::new(router?),
        };
        if draining.is_cancelled() {
            tracing::info!("Shutdown received before standalone EPP serving started");
            return Ok(());
        }
        let ready_router = router.clone();
        serve(
            router,
            move || ready_router.is_ready(),
            health_reporter,
            draining,
            shutdown,
        )
        .await
    } else {
        tracing::info!("Initializing KV-aware router from discovery...");
        let router = tokio::select! {
            _ = draining.cancelled() => {
                tracing::info!("Shutdown received during Dynamo discovery initialization");
                return Ok(());
            }
            router = Router::from_discovery(&config.namespace, &config.component) => router?,
        };
        if draining.is_cancelled() {
            tracing::info!("Shutdown received before Dynamo discovery serving started");
            return Ok(());
        }
        metrics::set_served_model(router.served_model());
        let ready = router.pod_store_ready();
        serve(
            Arc::new(router),
            move || ready.load(std::sync::atomic::Ordering::Acquire),
            health_reporter,
            draining,
            shutdown,
        )
        .await
    }
}

/// Mirror the picker's live readiness onto the gRPC health status, then serve
/// the ext_proc endpoint. Shared by both Dynamo-discovery and standalone modes.
async fn serve<P: crate::EndpointPicker>(
    picker: Arc<P>,
    is_ready: impl Fn() -> bool + Send + 'static,
    health_reporter: tonic_health::server::HealthReporter,
    draining: CancellationToken,
    shutdown: CancellationToken,
) -> Result<()> {
    // Continuously mirror readiness onto the health status. `is_ready()` is a
    // *live* signal that can flip both ways — standalone discovery clears it when
    // the InferencePool is deleted/invalid (nothing routable), and in replicated
    // mode it stays false until the peer set finishes its initial sync. A latch
    // (set SERVING once) would strand those states, so a background task tracks
    // transitions and moves the health status in lock-step, dropping out of
    // SERVING when readiness drops and recovering when it returns. Health starts
    // NOT_SERVING (set during startup); the mirror polls a cheap closure (atomic loads).
    let readiness_task = {
        let health_reporter = health_reporter.clone();
        let draining = draining.clone();
        tokio::spawn(async move {
            let mut last: Option<bool> = None;
            loop {
                if draining.is_cancelled() {
                    health_reporter
                        .set_service_status(
                            HEALTH_SERVICE_NAME,
                            tonic_health::ServingStatus::NotServing,
                        )
                        .await;
                    tracing::info!("EPP readiness mirror stopped; health set to NOT_SERVING");
                    break;
                }

                let now = is_ready();
                if !draining.is_cancelled() && last != Some(now) {
                    let status = if now {
                        tonic_health::ServingStatus::Serving
                    } else {
                        tonic_health::ServingStatus::NotServing
                    };
                    health_reporter
                        .set_service_status(HEALTH_SERVICE_NAME, status)
                        .await;
                    if draining.is_cancelled() {
                        health_reporter
                            .set_service_status(
                                HEALTH_SERVICE_NAME,
                                tonic_health::ServingStatus::NotServing,
                            )
                            .await;
                        tracing::info!("EPP readiness mirror stopped; health set to NOT_SERVING");
                        break;
                    }
                    tracing::info!(ready = now, "EPP readiness changed; health status updated");
                    last = Some(now);
                }

                tokio::select! {
                    _ = draining.cancelled() => {}
                    _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {}
                }
            }
        })
    };

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

        let result: Result<()> = loop {
            // Acquire permit before accept() so we backpressure the listener
            // instead of accepting and immediately dropping connections. Stop
            // accepting once the endpoint propagation window has elapsed.
            let permit = tokio::select! {
                _ = shutdown.cancelled() => break Ok(()),
                permit = conn_semaphore.clone().acquire_owned() => match permit {
                    Ok(permit) => permit,
                    Err(error) => break Err(error.into()),
                },
            };
            let (tcp_stream, remote_addr) = tokio::select! {
                _ = shutdown.cancelled() => break Ok(()),
                accepted = listener.accept() => match accepted {
                    Ok(accepted) => accepted,
                    Err(error) => break Err(error.into()),
                },
            };
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
        };
        readiness_task.abort();
        let _ = readiness_task.await;
        result
    } else {
        tracing::info!(%addr, "Listening for ext_proc connections (plaintext h2)");
        tonic::transport::Server::builder()
            .add_service(server.into_service())
            .serve_with_shutdown(addr, shutdown.cancelled_owned())
            .await?;
        readiness_task.abort();
        let _ = readiness_task.await;
        Ok(())
    }
}
