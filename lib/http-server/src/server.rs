// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::Router;
use axum_server::tls_rustls::RustlsConfig;
use dynamo_runtime::config::environment_names::llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS;
use tokio_util::sync::CancellationToken;

use super::lifecycle::ServiceObserver;

pub struct HttpServerConfig {
    pub host: String,
    pub port: u16,
    pub enable_tls: bool,
    pub tls_cert_path: Option<PathBuf>,
    pub tls_key_path: Option<PathBuf>,
}

enum Transport {
    Http(tokio::net::TcpListener),
    Https {
        address: SocketAddr,
        config: RustlsConfig,
    },
}

pub struct HttpServer {
    router: Router,
    transport: Transport,
}

fn get_graceful_shutdown_timeout() -> usize {
    std::env::var(DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5)
}

impl HttpServer {
    /// Perform fallible server setup before runtime-owned background tasks start.
    pub async fn prepare(
        router: Router,
        config: HttpServerConfig,
        listener: Option<tokio::net::TcpListener>,
    ) -> Result<Self> {
        let address = format!("{}:{}", config.host, config.port);
        let protocol = if config.enable_tls { "HTTPS" } else { "HTTP" };
        tracing::info!(protocol, address, "Starting HTTP(S) service");

        let transport = if config.enable_tls {
            if listener.is_some() {
                return Err(anyhow::anyhow!(
                    "Pre-bound listener is not supported in TLS mode; \
                     axum_server::bind_rustls owns its own bind. \
                     Use run()/spawn() (which bind internally) when enable_tls is set."
                ));
            }
            let address: SocketAddr = address
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid address '{}': {}", address, e))?;
            let cert_path = config
                .tls_cert_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("TLS certificate path not provided"))?;
            let key_path = config
                .tls_key_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("TLS private key path not provided"))?;

            // aws_lc_rs is the default but other crates pull in `ring` also,
            // so rustls doesn't know which one to use. Tell it.
            if let Err(e) = rustls::crypto::aws_lc_rs::default_provider().install_default() {
                tracing::debug!("TLS crypto provider already installed: {e:?}");
            }

            let config = RustlsConfig::from_pem_file(cert_path, key_path)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create TLS config: {}", e))?;
            Transport::Https { address, config }
        } else {
            let listener = match listener {
                Some(listener) => listener,
                None => {
                    let addr: SocketAddr = address
                        .parse()
                        .map_err(|e| anyhow::anyhow!("Invalid address '{}': {}", address, e))?;
                    tokio::net::TcpListener::bind(addr).await.map_err(|e| {
                        tracing::error!(
                            protocol = %protocol,
                            address = %address,
                            error = %e,
                            "Failed to bind server to address"
                        );
                        match e.kind() {
                            std::io::ErrorKind::AddrInUse => anyhow::anyhow!(
                                "Failed to start {} server: port {} already in use. Use --http-port to specify a different port.",
                                protocol,
                                config.port
                            ),
                            _ => anyhow::anyhow!(
                                "Failed to start {} server on {}: {}",
                                protocol,
                                address,
                                e
                            ),
                        }
                    })?
                }
            };
            Transport::Http(listener)
        };

        Ok(Self { router, transport })
    }

    pub async fn serve(
        self,
        state: Arc<ServiceObserver>,
        state_cancel: CancellationToken,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let observer = cancel_token.child_token();
        match self.transport {
            Transport::Https { address, config } => {
                let handle = axum_server::Handle::new();
                let server = axum_server::bind_rustls(address, config)
                    .handle(handle.clone())
                    .serve(self.router.into_make_service());

                tokio::select! {
                    result = server => {
                        let result = result.map_err(|e| anyhow::anyhow!("HTTPS server error: {}", e));
                        state.start_stopping();
                        cancel_token.cancel();
                        result?;
                    }
                    _ = observer.cancelled() => {
                        state.start_draining();
                        tracing::info!("HTTPS server shutdown requested");
                        let shutdown_timeout =
                            Duration::from_secs(get_graceful_shutdown_timeout() as u64);
                        handle.graceful_shutdown(Some(shutdown_timeout));
                        if !state.wait_inflight_zero_or_timeout(shutdown_timeout).await {
                            tracing::warn!(
                                inflight_requests = state.inflight_count(),
                                "Timed out waiting for inflight inference requests to drain"
                            );
                        }
                        state.start_stopping();
                        state_cancel.cancel();
                    }
                }
            }
            Transport::Http(listener) => {
                let shutdown_state = state.clone();
                axum::serve(listener, self.router)
                    .with_graceful_shutdown(async move {
                        observer.cancelled_owned().await;
                        shutdown_state.start_draining();
                        tracing::info!("HTTP server shutdown requested");
                        let shutdown_timeout =
                            Duration::from_secs(get_graceful_shutdown_timeout() as u64);
                        if !shutdown_state
                            .wait_inflight_zero_or_timeout(shutdown_timeout)
                            .await
                        {
                            tracing::warn!(
                                inflight_requests = shutdown_state.inflight_count(),
                                "Timed out waiting for inflight inference requests to drain"
                            );
                        }
                        shutdown_state.start_stopping();
                        state_cancel.cancel();
                    })
                    .await
                    .inspect_err(|_| {
                        state.start_stopping();
                        cancel_token.cancel()
                    })?;
                state.start_stopping();
                cancel_token.cancel();
            }
        }

        Ok(())
    }
}
