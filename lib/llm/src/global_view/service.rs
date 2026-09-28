// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One Global Router replica: Relay-fed Global View and the serving listener.

use std::future::IntoFuture;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use dynamo_kv_router::global_view::eligibility::has_ready_aggregated_pool;
use dynamo_kv_router::global_view::state::{FreshnessPolicy, PoolStateRepository};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use super::http_forward::GlobalRouterHttp;
use crate::kv_dc_relay::global_view_consumer::GlobalViewRuntime;

struct ReadinessState {
    repository: Arc<dyn PoolStateRepository>,
    freshness: FreshnessPolicy,
}

async fn readyz(State(state): State<Arc<ReadinessState>>) -> StatusCode {
    let now_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|elapsed| u64::try_from(elapsed.as_millis()).ok());
    if now_unix_ms.is_some_and(|now| {
        has_ready_aggregated_pool(state.repository.as_ref(), now, &state.freshness)
    }) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

pub struct GlobalRouterService {
    view: Arc<GlobalViewRuntime>,
    forwarder: Arc<GlobalRouterHttp>,
    freshness: FreshnessPolicy,
}

impl GlobalRouterService {
    pub fn new(view: Arc<GlobalViewRuntime>, freshness: FreshnessPolicy) -> Result<Self> {
        let forwarder = Arc::new(
            GlobalRouterHttp::new(view.repository(), freshness)
                .context("create private Frontend HTTP client")?,
        );
        Ok(Self {
            view,
            forwarder,
            freshness,
        })
    }

    /// The same listener serves routed inference and, when enabled, read-only
    /// Global View inspection. Routing reads the repository directly.
    pub fn router(&self) -> Router {
        let readiness = Arc::new(ReadinessState {
            repository: self.view.repository(),
            freshness: self.freshness,
        });
        let routes = self.forwarder.clone().router().merge(
            Router::new()
                .route("/readyz", get(readyz))
                .with_state(readiness),
        );
        #[cfg(feature = "global-view-diagnostics")]
        let routes = routes.merge(self.view.diagnostics_router(self.freshness));
        routes
    }

    /// Stop both serving and relay subscriptions when either side exits.
    /// The caller provides the already-bound listener and shutdown token.
    pub async fn run(&self, listener: TcpListener, cancel: CancellationToken) -> Result<()> {
        let view_cancel = cancel.child_token();
        let server_cancel = cancel.child_token();
        let view_run = self.view.run(view_cancel.clone());
        let server_run = axum::serve(listener, self.router())
            .with_graceful_shutdown(server_cancel.clone().cancelled_owned())
            .into_future();
        tokio::pin!(view_run);
        tokio::pin!(server_run);

        tokio::select! {
            result = &mut view_run => {
                server_cancel.cancel();
                server_run.await.context("stop Global Router HTTP listener")?;
                result.context("Global View subscription stopped")
            }
            result = &mut server_run => {
                view_cancel.cancel();
                view_run.await.context("stop Global View subscriptions")?;
                result.context("Global Router HTTP listener failed")?;
                if cancel.is_cancelled() {
                    Ok(())
                } else {
                    Err(anyhow!("Global Router HTTP listener stopped before shutdown"))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::global_view::RelayPoolScope;
    use crate::kv_dc_relay::global_view_consumer::RelayDgdSource;
    use dynamo_kv_router::global_view::PoolKey;
    use dynamo_kv_router::global_view::state::PoolLocation;

    fn service() -> GlobalRouterService {
        let channel = tonic::transport::Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let view = Arc::new(
            GlobalViewRuntime::new(
                vec![RelayDgdSource {
                    key: PoolKey::new("ohio", "dynamo", "mocker").unwrap(),
                    location: PoolLocation {
                        region: "us-east-2".into(),
                        availability_zone: None,
                        cluster: None,
                        datacenter: None,
                    },
                    scope: RelayPoolScope {
                        runtime_namespace: "mocker".into(),
                        frontend_endpoint: "http://127.0.0.1:1".into(),
                    },
                    model: "model".into(),
                    subscriber_id: "global-router-ohio".into(),
                    relay_channel: channel.clone(),
                    stats_channel: channel,
                }],
                Duration::from_secs(10),
            )
            .unwrap(),
        );
        GlobalRouterService::new(
            view,
            FreshnessPolicy {
                catalog_max_age_ms: 60_000,
                readiness_max_age_ms: 60_000,
                capacity_max_age_ms: 60_000,
                load_max_age_ms: 60_000,
                kv_usage_max_age_ms: 60_000,
                kv_overlap_max_age_ms: 60_000,
            },
        )
        .unwrap()
    }

    #[tokio::test]
    async fn shutdown_stops_http_and_relay_tasks() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), service().run(listener, cancel))
            .await
            .unwrap()
            .unwrap();
    }

    #[cfg(feature = "global-view-diagnostics")]
    #[tokio::test]
    async fn one_listener_mounts_diagnostics_and_serving_routes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let service = service();
        let task = tokio::spawn(async move {
            axum::serve(listener, service.router()).await.unwrap();
        });
        let client = reqwest::Client::new();
        let pools = client
            .get(format!("http://{address}/pools"))
            .send()
            .await
            .unwrap();
        assert_eq!(pools.status(), reqwest::StatusCode::OK);
        let readiness = client
            .get(format!("http://{address}/readyz"))
            .send()
            .await
            .unwrap();
        assert_eq!(readiness.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let inference = client
            .post(format!("http://{address}/v1/completions"))
            .body(r#"{"model":"model","prompt":"hi"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(inference.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        task.abort();
    }
}
