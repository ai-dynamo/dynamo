// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sidecar probes must answer before either the engine or runtime is available.
//! One listener serves these probes throughout startup and shutdown; the usual
//! metrics, metadata and engine routes become available when the runtime connects.

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use axum::{Router, extract::Request, http::StatusCode, response::IntoResponse, routing::get};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

use super::{SystemStatusServerInfo, serve_system_status, system_status_router};
use crate::{DistributedRuntime, config::RuntimeConfig, discovery::DiscoveryMetadata};

struct ConnectedRuntime {
    drt: DistributedRuntime,
    routes: Router,
}

/// HTTP listener owned by the sidecar process, not by engine initialization.
pub struct SidecarStatusServer {
    connected: Arc<OnceLock<ConnectedRuntime>>,
    info: Arc<SystemStatusServerInfo>,
    stop: CancellationToken,
}

impl SidecarStatusServer {
    /// Bind before any remote connection attempt. Disabled system HTTP stays disabled.
    pub async fn start(
        config: &RuntimeConfig,
        shutdown: CancellationToken,
    ) -> anyhow::Result<Option<Self>> {
        if !config.system_server_enabled() {
            return Ok(None);
        }
        let connected = Arc::new(OnceLock::<ConnectedRuntime>::new());
        let readiness_state = connected.clone();
        let readiness_shutdown = shutdown.clone();
        let runtime_routes = connected.clone();
        let last_ready = Arc::new(AtomicBool::new(false));
        let app = Router::new()
            .route(&config.system_live_path, get(|| async { StatusCode::OK }))
            .route(
                &config.system_health_path,
                get(move || {
                    let connected = readiness_state.clone();
                    let shutdown = readiness_shutdown.clone();
                    let last_ready = last_ready.clone();
                    async move {
                        let result = async {
                            anyhow::ensure!(!shutdown.is_cancelled(), "sidecar is shutting down");
                            let state = connected
                                .get()
                                .ok_or_else(|| anyhow::anyhow!("runtime initializing"))?;
                            tokio::time::timeout(
                                Duration::from_secs(1),
                                state.drt.check_dependencies(),
                            )
                            .await
                            .map_err(|_| anyhow::anyhow!("runtime dependency check timed out"))??;
                            anyhow::ensure!(!shutdown.is_cancelled(), "sidecar is shutting down");
                            Ok::<_, anyhow::Error>(())
                        }
                        .await;
                        let ready = result.is_ok();
                        // Log transitions, not every kubelet probe.
                        if last_ready.swap(ready, Ordering::AcqRel) != ready {
                            match result {
                                Ok(()) => {
                                    tracing::info!("Sidecar ready; runtime dependencies available")
                                }
                                Err(error) => tracing::warn!(%error, "Sidecar not ready"),
                            }
                        }
                        let (code, status) = if ready {
                            (StatusCode::OK, "ready")
                        } else {
                            (StatusCode::SERVICE_UNAVAILABLE, "notready")
                        };
                        (code, axum::Json(serde_json::json!({ "status": status })))
                    }
                }),
            )
            .fallback(move |request: Request| {
                let connected = runtime_routes.clone();
                async move {
                    match connected.get() {
                        Some(state) => match state.routes.clone().oneshot(request).await {
                            Ok(response) => response,
                            Err(infallible) => match infallible {},
                        },
                        None => StatusCode::SERVICE_UNAVAILABLE.into_response(),
                    }
                }
            });
        let stop = CancellationToken::new();
        let (address, handle) = serve_system_status(
            &config.system_host,
            config.system_port as u16,
            stop.clone(),
            app,
        )
        .await?;
        tracing::info!(%address, "Sidecar probes started; runtime initializing");
        Ok(Some(Self {
            connected,
            info: Arc::new(SystemStatusServerInfo::new(address, Some(handle))),
            stop,
        }))
    }

    pub(crate) fn attach(
        &self,
        drt: Arc<DistributedRuntime>,
        metadata: Option<Arc<tokio::sync::RwLock<DiscoveryMetadata>>>,
    ) -> anyhow::Result<()> {
        let routes = system_status_router(drt.clone(), metadata)?;
        self.connected
            .set(ConnectedRuntime {
                drt: (*drt).clone(),
                routes,
            })
            .map_err(|_| anyhow::anyhow!("sidecar runtime already attached"))
    }

    pub(crate) fn info(&self) -> Arc<SystemStatusServerInfo> {
        self.info.clone()
    }
}

impl Drop for SidecarStatusServer {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Runtime, config::HealthStatus, distributed::DistributedConfig};

    async fn status(client: &reqwest::Client, base: &str, path: &str) -> u16 {
        client
            .get(format!("{base}{path}"))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16()
    }

    #[tokio::test]
    async fn probes_follow_runtime_and_shutdown_without_engine_registration() {
        let shutdown = CancellationToken::new();
        let config = RuntimeConfig {
            system_host: "127.0.0.1".into(),
            system_port: 0,
            system_live_path: "/custom-live".into(),
            system_health_path: "/custom-health".into(),
            ..Default::default()
        };
        let server = SidecarStatusServer::start(&config, shutdown.clone())
            .await
            .unwrap()
            .unwrap();
        let base = format!("http://{}", server.info.socket_addr);
        let client = reqwest::Client::new();
        assert_eq!(status(&client, &base, "/custom-live").await, 200);
        assert_eq!(status(&client, &base, "/custom-health").await, 503);
        assert_eq!(status(&client, &base, "/metrics").await, 503);

        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new_with_sidecar_status(
            runtime.clone(),
            DistributedConfig::process_local(),
            Some(&server),
        )
        .await
        .unwrap();
        // Existing worker health/registration state must not control sidecar probes.
        drt.system_health()
            .lock()
            .set_health_status(HealthStatus::NotReady);
        assert_eq!(status(&client, &base, "/custom-health").await, 200);
        assert_eq!(status(&client, &base, "/custom-live").await, 200);
        assert_eq!(status(&client, &base, "/metrics").await, 200);
        assert_eq!(status(&client, &base, "/missing").await, 404);

        shutdown.cancel();
        assert_eq!(status(&client, &base, "/custom-health").await, 503);
        assert_eq!(status(&client, &base, "/custom-live").await, 200);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn runtime_shutdown_withdraws_readiness_without_stopping_liveness() {
        let config = RuntimeConfig {
            system_host: "127.0.0.1".into(),
            system_port: 0,
            ..Default::default()
        };
        let server = SidecarStatusServer::start(&config, CancellationToken::new())
            .await
            .unwrap()
            .unwrap();
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new_with_sidecar_status(
            runtime.clone(),
            DistributedConfig::process_local(),
            Some(&server),
        )
        .await
        .unwrap();
        let base = format!("http://{}", server.info.socket_addr);
        let client = reqwest::Client::new();
        assert_eq!(status(&client, &base, "/health").await, 200);
        // Hold Phase 2 open: readiness must fail before transport teardown.
        let guard = runtime.graceful_shutdown_tracker().register_task();
        runtime.shutdown();
        assert_eq!(status(&client, &base, "/health").await, 503);
        assert_eq!(status(&client, &base, "/live").await, 200);
        assert!(!drt.runtime().primary_token().is_cancelled());
        drop(guard);
    }

    #[tokio::test]
    async fn dependency_outage_only_fails_readiness_and_can_recover() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::sync::Notify;

        // A minimal NATS peer lets us close and restore the real client connection
        // without an external daemon or inference requests.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let disconnect = Arc::new(Notify::new());
        let reconnect = Arc::new(Notify::new());
        let peer_disconnect = disconnect.clone();
        let peer_reconnect = reconnect.clone();
        let peer = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(async move {
            for attempt in 0..2 {
                if attempt > 0 {
                    peer_reconnect.notified().await;
                }
                let (socket, _) = listener.accept().await.unwrap();
                let (reader, mut writer) = socket.into_split();
                writer.write_all(b"INFO {\"server_id\":\"probe-test\",\"version\":\"2.10.0\",\"proto\":1,\"max_payload\":1048576}\r\n").await.unwrap();
                let connection = async {
                    let mut lines = BufReader::new(reader).lines();
                    while let Some(line) = lines.next_line().await.unwrap() {
                        if line == "PING" {
                            writer.write_all(b"PONG\r\n").await.unwrap();
                        }
                    }
                };
                tokio::select! {
                    _ = peer_disconnect.notified() => {},
                    _ = connection => {},
                }
            }
        }));
        let config = RuntimeConfig {
            system_host: "127.0.0.1".into(),
            system_port: 0,
            ..Default::default()
        };
        let server = SidecarStatusServer::start(&config, CancellationToken::new())
            .await
            .unwrap()
            .unwrap();
        let runtime = Runtime::from_current().unwrap();
        let distributed = DistributedConfig {
            nats_config: Some(
                crate::transports::nats::ClientOptions::builder()
                    .server(format!("nats://{address}"))
                    .build()
                    .unwrap(),
            ),
            ..DistributedConfig::process_local()
        };
        let _drt = DistributedRuntime::new_with_sidecar_status(
            runtime.clone(),
            distributed,
            Some(&server),
        )
        .await
        .unwrap();
        let base = format!("http://{}", server.info.socket_addr);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap();
        assert_eq!(status(&client, &base, "/health").await, 200);
        disconnect.notify_one();
        tokio::time::timeout(Duration::from_secs(5), async {
            while status(&client, &base, "/health").await != 503 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(status(&client, &base, "/live").await, 200);
        reconnect.notify_one();
        tokio::time::timeout(Duration::from_secs(10), async {
            while status(&client, &base, "/health").await != 200 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        runtime.shutdown();
        drop(peer);
    }
}
