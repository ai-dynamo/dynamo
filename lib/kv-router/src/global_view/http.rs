// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional read-only diagnostics for the in-process Global View.
//! Serving-path routing uses PoolStateRepository directly.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use super::PoolId;
use super::eligibility::eligible_pools;
use super::overlap::KvOverlapScorer;
use super::state::{FreshnessPolicy, PoolState, PoolStateRepository, SignalState, SignalStatus};

#[derive(Clone)]
struct DiagnosticsState {
    repository: Arc<dyn PoolStateRepository>,
    freshness: FreshnessPolicy,
}

#[derive(Deserialize)]
struct PoolQuery {
    model: Option<String>,
}
#[derive(Clone)]
struct OverlapDiagnosticsState {
    repository: Arc<dyn PoolStateRepository>,
    scorer: Arc<dyn KvOverlapScorer>,
    freshness: FreshnessPolicy,
}

#[derive(Deserialize)]
struct OverlapRequest {
    model: String,
    token_ids: Vec<u32>,
    pool_ids: Option<Vec<PoolId>>,
    #[serde(flatten)]
    unsupported_fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct OverlapScore {
    pool_id: PoolId,
    matched_prefix_tokens: Option<u64>,
    signal_status: SignalStatus,
}

#[derive(Serialize)]
struct OverlapResponse {
    scores: Vec<OverlapScore>,
}

#[derive(Serialize)]
struct DiagnosticError {
    error: &'static str,
    fields: Vec<String>,
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
    let mut pools = state.repository.list(now_unix_ms(), &state.freshness);
    if let Some(model) = query.model {
        pools.retain(|pool| pool.descriptors.models.iter().any(|name| name == &model));
    }
    Json(pools)
}

/// Mount request-specific overlap diagnostics next to pool diagnostics.
pub fn overlap_diagnostics_router(
    repository: Arc<dyn PoolStateRepository>,
    freshness: FreshnessPolicy,
    scorer: Arc<dyn KvOverlapScorer>,
) -> Router {
    Router::new()
        .route("/overlap_scores", post(overlap_scores))
        .with_state(OverlapDiagnosticsState {
            repository,
            scorer,
            freshness,
        })
}

pub fn global_view_diagnostics_router(
    repository: Arc<dyn PoolStateRepository>,
    freshness: FreshnessPolicy,
    scorer: Arc<dyn KvOverlapScorer>,
) -> Router {
    pool_diagnostics_router(repository.clone(), freshness)
        .merge(overlap_diagnostics_router(repository, freshness, scorer))
}

async fn overlap_scores(
    State(state): State<OverlapDiagnosticsState>,
    Json(request): Json<OverlapRequest>,
) -> Result<Json<OverlapResponse>, (StatusCode, Json<DiagnosticError>)> {
    if !request.unsupported_fields.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(DiagnosticError {
                error: "unsupported_request_variant",
                fields: request.unsupported_fields.into_keys().collect(),
            }),
        ));
    }
    if request.model.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(DiagnosticError {
                error: "invalid_model",
                fields: Vec::new(),
            }),
        ));
    }
    let candidate_ids = request
        .pool_ids
        .map(|ids| ids.into_iter().collect::<HashSet<_>>());
    let pools = eligible_pools(
        state.repository.as_ref(),
        &request.model,
        now_unix_ms(),
        &state.freshness,
    );
    let scores = pools
        .into_iter()
        .filter(|pool| {
            candidate_ids
                .as_ref()
                .is_none_or(|ids| ids.contains(&pool.pool_id))
        })
        .map(|pool| {
            let mut signal_status = pool.signal_status.kv_overlap;
            let matched_prefix_tokens = if matches!(
                signal_status.state,
                SignalState::Complete | SignalState::Degraded
            ) {
                state.scorer.estimate_matched_prefix_tokens(
                    &pool.pool_id,
                    &request.model,
                    &request.token_ids,
                )
            } else {
                None
            };
            if matched_prefix_tokens.is_none()
                && matches!(
                    signal_status.state,
                    SignalState::Complete | SignalState::Degraded
                )
            {
                signal_status.state = SignalState::Unavailable;
            }
            OverlapScore {
                pool_id: pool.pool_id,
                matched_prefix_tokens,
                signal_status,
            }
        })
        .collect();
    Ok(Json(OverlapResponse { scores }))
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
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

    struct FixedScorer {
        pool_id: PoolId,
    }

    impl KvOverlapScorer for FixedScorer {
        fn estimate_matched_prefix_tokens(
            &self,
            pool_id: &PoolId,
            _model: &str,
            _token_ids: &[u32],
        ) -> Option<u64> {
            (pool_id == &self.pool_id).then_some(2)
        }
    }

    fn mark_ready_for_overlap(assembler: &PoolObservationAssembler, model: &str) {
        let received_at_unix_ms = now_unix_ms();
        let status = SignalStatus {
            state: SignalState::Complete,
            received_at_unix_ms: Some(received_at_unix_ms),
            ..Default::default()
        };
        let readiness = assembler.open(SourcePlane::Readiness).unwrap();
        assembler
            .apply(
                readiness,
                PoolObservation::Readiness {
                    models: BTreeMap::from([(
                        model.into(),
                        crate::global_view::state::ServingReadiness::Ready,
                    )]),
                    status: status.clone(),
                },
            )
            .unwrap();
        let overlap = assembler.open(SourcePlane::KvOverlap).unwrap();
        assembler
            .apply(overlap, PoolObservation::KvOverlap { status })
            .unwrap();
    }

    #[tokio::test]
    async fn overlap_scores_only_eligible_candidates_and_rejects_variants() {
        let repo = Arc::new(InMemoryPoolStateRepository::default());
        let ohio = add_pool(repo.clone(), "ohio", "model-a");
        mark_ready_for_overlap(&ohio, "model-a");
        let west = add_pool(repo.clone(), "west", "model-a");
        mark_ready_for_overlap(&west, "model-a");
        let app = global_view_diagnostics_router(
            repo,
            freshness(),
            Arc::new(FixedScorer {
                pool_id: ohio.pool_id(),
            }),
        );
        let request_body = serde_json::json!({
            "model": "model-a",
            "token_ids": [1, 2, 3, 4],
            "pool_ids": [ohio.pool_id()]
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/overlap_scores")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let result: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(result["scores"].as_array().unwrap().len(), 1);
        assert_eq!(result["scores"][0]["pool_id"], ohio.pool_id().as_str());
        assert_eq!(result["scores"][0]["matched_prefix_tokens"], 2);
        assert_eq!(result["scores"][0]["signal_status"]["state"], "complete");

        let unsupported = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/overlap_scores")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "model-a",
                            "token_ids": [1, 2],
                            "lora_adapter": "adapter"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unsupported.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(unsupported.into_body(), usize::MAX).await.unwrap();
        let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(error["error"], "unsupported_request_variant");
        assert_eq!(error["fields"], serde_json::json!(["lora_adapter"]));
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
