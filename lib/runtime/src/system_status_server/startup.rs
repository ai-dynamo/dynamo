// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One runtime-owned listener, available before remote dependencies connect.

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use axum::{Router, extract::Request, http::StatusCode, response::IntoResponse, routing::get};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;
use tower_http::trace::TraceLayer;

use super::{SystemStatusServerInfo, health_handler, serve_system_status, system_status_router};
use crate::{
    DistributedRuntime, Runtime, SystemHealth, config::RuntimeConfig, discovery::DiscoveryMetadata,
};

/// Selects probe semantics without changing ownership of the runtime HTTP server.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SystemProbePolicy {
    /// Preserve the configured worker health handler for both probes.
    #[default]
    Worker,
    /// Static liveness and readiness based on runtime dependencies, not the engine.
    RuntimeOnly,
}

struct ConnectedState {
    drt: DistributedRuntime,
    routes: Router,
}

/// Owns the listener while DRT construction can still fail or be cancelled.
pub(crate) struct PendingSystemStatusServer {
    connected: Arc<OnceLock<ConnectedState>>,
    info: Arc<SystemStatusServerInfo>,
    stop: CancellationToken,
    attached: bool,
}

impl PendingSystemStatusServer {
    pub(crate) async fn start(
        config: &RuntimeConfig,
        runtime: &Runtime,
        health: Arc<parking_lot::Mutex<SystemHealth>>,
        policy: SystemProbePolicy,
    ) -> anyhow::Result<Self> {
        let connected = Arc::new(OnceLock::<ConnectedState>::new());
        let runtime_routes = connected.clone();
        let mut app = Router::new();
        match policy {
            SystemProbePolicy::Worker => {
                let live_health = health.clone();
                app = app
                    .route(
                        &config.system_live_path,
                        get(move || health_handler(live_health.clone())),
                    )
                    .route(
                        &config.system_health_path,
                        get(move || health_handler(health.clone())),
                    );
            }
            SystemProbePolicy::RuntimeOnly => {
                let readiness_state = connected.clone();
                let shutdown = runtime.shutdown_started_token();
                let last_ready = Arc::new(AtomicBool::new(false));
                app = app
                    .route(&config.system_live_path, get(|| async { StatusCode::OK }))
                    .route(
                        &config.system_health_path,
                        get(move || {
                            let connected = readiness_state.clone();
                            let shutdown = shutdown.clone();
                            let last_ready = last_ready.clone();
                            async move {
                                let result = async {
                                    anyhow::ensure!(
                                        !shutdown.is_cancelled(),
                                        "runtime is shutting down"
                                    );
                                    let state = connected
                                        .get()
                                        .ok_or_else(|| anyhow::anyhow!("runtime initializing"))?;
                                    tokio::time::timeout(
                                        Duration::from_secs(1),
                                        state.drt.check_dependencies(),
                                    )
                                    .await
                                    .map_err(|_| {
                                        anyhow::anyhow!("runtime dependency check timed out")
                                    })??;
                                    anyhow::ensure!(
                                        !shutdown.is_cancelled(),
                                        "runtime is shutting down"
                                    );
                                    Ok::<_, anyhow::Error>(())
                                }
                                .await;
                                let ready = result.is_ok();
                                if last_ready.swap(ready, Ordering::AcqRel) != ready {
                                    match result {
                                        Ok(()) => {
                                            tracing::info!("Runtime ready; dependencies available")
                                        }
                                        Err(error) => tracing::warn!(%error, "Runtime not ready"),
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
                    );
            }
        }
        let app = app
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
            })
            .layer(
                TraceLayer::new_for_http().make_span_with(crate::logging::make_system_request_span),
            );
        // Sidecars keep HTTP through unregister/drain (Phase 2). Ordinary workers
        // retain their existing endpoint-shutdown lifetime (Phase 1).
        let stop = match policy {
            SystemProbePolicy::Worker => runtime.child_token(),
            SystemProbePolicy::RuntimeOnly => runtime.primary_token().child_token(),
        };
        let (address, handle) = serve_system_status(
            &config.system_host,
            config.system_port as u16,
            stop.clone(),
            app,
        )
        .await?;
        tracing::info!(%address, ?policy, "System HTTP listener started; runtime initializing");
        Ok(Self {
            connected,
            info: Arc::new(SystemStatusServerInfo::new(address, Some(handle))),
            stop,
            attached: false,
        })
    }

    pub(crate) fn info(&self) -> Arc<SystemStatusServerInfo> {
        self.info.clone()
    }

    /// Commit only after every fallible initialization step has succeeded.
    pub(crate) fn attach(
        mut self,
        drt: DistributedRuntime,
        metadata: Option<Arc<tokio::sync::RwLock<DiscoveryMetadata>>>,
    ) -> anyhow::Result<()> {
        let routes = system_status_router(Arc::new(drt.clone()), metadata)?;
        self.connected
            .set(ConnectedState { drt, routes })
            .map_err(|_| anyhow::anyhow!("runtime HTTP routes already attached"))?;
        self.attached = true;
        Ok(())
    }
}

impl Drop for PendingSystemStatusServer {
    fn drop(&mut self) {
        if !self.attached {
            self.stop.cancel();
            // Cancellation alone could wait for an unfinished HTTP request.
            // Failed construction must release the listener and all connections.
            if let Some(handle) = &self.info.handle {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests;
