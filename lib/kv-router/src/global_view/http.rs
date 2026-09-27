// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional read-only diagnostics for the in-process Global View.
//! Serving-path routing uses PoolStateRepository directly.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;

use super::state::{FreshnessPolicy, PoolState, PoolStateRepository};

#[derive(Clone)]
struct DiagnosticsState {
    repository: Arc<dyn PoolStateRepository>,
    freshness: FreshnessPolicy,
}

#[derive(Deserialize)]
struct PoolQuery {
    model: Option<String>,
}

/// Mount under a diagnostic-only listener. This does not expose CKF contents.
pub fn pool_diagnostics_router(
    repository: Arc<dyn PoolStateRepository>,
    freshness: FreshnessPolicy,
) -> Router {
    Router::new()
        .route("/pools", get(list_pools))
        .with_state(DiagnosticsState {
            repository,
            freshness,
        })
}

async fn list_pools(
    State(state): State<DiagnosticsState>,
    Query(query): Query<PoolQuery>,
) -> Json<Vec<PoolState>> {
    let now_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    let mut pools = state.repository.list(now_unix_ms, &state.freshness);
    if let Some(model) = query.model {
        pools.retain(|pool| pool.descriptors.models.iter().any(|name| name == &model));
    }
    Json(pools)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_view::source::{PoolObservation, PoolObservationAssembler, SourcePlane};
    use crate::global_view::state::{
        InMemoryPoolStateRepository, PoolLocation, PoolRole, SignalState, SignalStatus,
    };
    use crate::global_view::{PoolKey, V1PoolIdDeriver};
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn freshness() -> FreshnessPolicy {
        FreshnessPolicy {
            catalog_max_age_ms: 60_000,
            readiness_max_age_ms: 60_000,
            capacity_max_age_ms: 60_000,
            load_max_age_ms: 60_000,
            kv_usage_max_age_ms: 60_000,
            kv_overlap_max_age_ms: 60_000,
        }
    }

    fn add_pool(
        repo: Arc<InMemoryPoolStateRepository>,
        site: &str,
        model: &str,
    ) -> PoolObservationAssembler {
        let key = PoolKey::new(site, "dynamo", "mocker").unwrap();
        let assembler = PoolObservationAssembler::new(
            &key,
            PoolLocation {
                region: site.into(),
                availability_zone: None,
                cluster: None,
                datacenter: None,
            },
            &V1PoolIdDeriver,
            repo,
        );
        let lease = assembler.open(SourcePlane::Catalog).unwrap();
        let received_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        assembler
            .apply(
                lease,
                PoolObservation::Catalog {
                    models: vec![model.into()],
                    roles: vec![PoolRole::Aggregated],
                    frontend_endpoint: Some("frontend.generate".into()),
                    hardware: Vec::new(),
                    status: SignalStatus {
                        state: SignalState::Complete,
                        received_at_unix_ms: Some(received_at_unix_ms),
                        ..Default::default()
                    },
                },
            )
            .unwrap();
        assembler
    }

    #[tokio::test]
    async fn lists_model_pools_with_nested_state_without_ckf_payload() {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let ohio = add_pool(repo.clone(), "ohio", "model-a");
        let _west = add_pool(repo.clone(), "west", "model-b");
        let app = pool_diagnostics_router(repo, freshness());

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/pools?model=model-a")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let pools: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(pools.as_array().unwrap().len(), 1);
        assert_eq!(pools[0]["pool_id"], ohio.pool_id().as_str());
        assert_eq!(pools[0]["descriptors"]["site_id"], "ohio");
        assert_eq!(pools[0]["signal_status"]["catalog"]["state"], "complete");
        assert_eq!(
            pools[0]["signal_status"]["kv_overlap"]["state"],
            "unavailable"
        );
        assert!(!String::from_utf8(bytes.to_vec()).unwrap().contains("ckf"));
    }
}
