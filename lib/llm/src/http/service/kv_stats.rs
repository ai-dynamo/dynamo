// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical frontend projection of worker-owned KV capacity and occupancy.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderValue, Method, header};
use axum::response::Response;
use axum::routing::get;
use serde::Serialize;

use crate::discovery::ModelManager;
use crate::worker_type::WorkerType;

use super::{RouteDoc, service_v2};

const KV_STATS_PATH: &str = "/v1/kv-cache/stats/stream";
const SNAPSHOT_INTERVAL: Duration = Duration::from_secs(1);
static NEXT_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Serialize)]
struct KvStatsSnapshot {
    v: u8,
    #[serde(rename = "type")]
    event_type: &'static str,
    snapshot_id: u64,
    observed_at_unix_ms: u64,
    models: Vec<ModelKvStats>,
}

#[derive(Debug, Serialize)]
struct ModelKvStats {
    model: String,
    aliases: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    routing_cache: Option<RoutingCacheStats>,
    pools: Vec<KvPoolStats>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct RoutingCacheStats {
    role: WorkerType,
    capacity_tokens: u64,
    used_tokens: u64,
    free_tokens: u64,
}

#[derive(Debug, Serialize)]
struct KvPoolStats {
    namespace: String,
    component: String,
    endpoint: String,
    role: WorkerType,
    storage_tier: &'static str,
    block_size_tokens: u32,
    expected_ranks: u64,
    observed_ranks: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    capacity_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    used_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    free_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_decode_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_prefill_tokens: Option<u64>,
    complete: bool,
}

pub(super) fn router(state: Arc<service_v2::State>) -> (Vec<RouteDoc>, Router) {
    let docs = vec![RouteDoc::new(Method::GET, KV_STATS_PATH)];
    let router = Router::new()
        .route(KV_STATS_PATH, get(kv_stats_stream_handler))
        .with_state(state);
    (docs, router)
}

async fn kv_stats_stream_handler(State(state): State<Arc<service_v2::State>>) -> Response {
    kv_stats_stream_response(state.manager_clone(), state.cancel_token().clone())
}

fn kv_stats_stream_response(
    manager: Arc<ModelManager>,
    shutdown: tokio_util::sync::CancellationToken,
) -> Response {
    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(SNAPSHOT_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = interval.tick() => {
                    match serde_json::to_vec(&build_snapshot(&manager)) {
                        Ok(mut line) => {
                            line.push(b'\n');
                            yield Ok::<Bytes, Infallible>(Bytes::from(line));
                        }
                        Err(error) => tracing::warn!(%error, "failed to serialize KV stats snapshot"),
                    }
                }
            }
        }
    };
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

fn build_snapshot(manager: &ModelManager) -> KvStatsSnapshot {
    let models = manager
        .committed_model_views()
        .into_iter()
        .filter_map(|view| {
            let mut pools = Vec::new();
            for worker_set in view
                .worker_sets
                .into_iter()
                .filter(|worker_set| worker_set.card().lora.is_none())
            {
                let Some(monitor) = worker_set.worker_monitor() else {
                    continue;
                };
                let role = worker_set
                    .card()
                    .worker_type
                    .unwrap_or(WorkerType::Aggregated);
                for pool in monitor.pool_snapshots(role) {
                    let endpoint = pool.endpoint;
                    pools.push(KvPoolStats {
                        namespace: endpoint.namespace,
                        component: endpoint.component,
                        endpoint: endpoint.name,
                        role: pool.role,
                        storage_tier: "device",
                        block_size_tokens: worker_set.card().kv_cache_block_size,
                        expected_ranks: pool.expected_ranks,
                        observed_ranks: pool.observed_ranks,
                        capacity_blocks: pool.capacity_blocks,
                        used_blocks: pool.used_blocks,
                        free_blocks: pool.free_blocks,
                        active_decode_blocks: pool.active_decode_blocks,
                        active_prefill_tokens: pool.active_prefill_tokens,
                        complete: pool.complete,
                    });
                }
            }
            pools.sort_unstable_by(|left, right| {
                (
                    left.namespace.as_str(),
                    left.component.as_str(),
                    left.endpoint.as_str(),
                    left.role.as_str(),
                )
                    .cmp(&(
                        right.namespace.as_str(),
                        right.component.as_str(),
                        right.endpoint.as_str(),
                        right.role.as_str(),
                    ))
            });
            (!pools.is_empty()).then(|| ModelKvStats {
                model: view.name,
                aliases: view.aliases,
                routing_cache: routing_cache(&pools),
                pools,
            })
        })
        .collect();
    KvStatsSnapshot {
        v: 1,
        event_type: "kv_stats_snapshot",
        snapshot_id: NEXT_SNAPSHOT_ID.fetch_add(1, Ordering::Relaxed),
        observed_at_unix_ms: current_unix_millis(),
        models,
    }
}

fn routing_cache(pools: &[KvPoolStats]) -> Option<RoutingCacheStats> {
    let role = if pools.iter().any(|pool| pool.role == WorkerType::Aggregated) {
        WorkerType::Aggregated
    } else if pools.iter().any(|pool| pool.role == WorkerType::Decode) {
        WorkerType::Decode
    } else {
        return None;
    };
    let selected = pools
        .iter()
        .filter(|pool| pool.role == role)
        .collect::<Vec<_>>();
    if !selected.iter().all(|pool| pool.complete) {
        return None;
    }
    let (capacity, used, free) =
        selected
            .iter()
            .try_fold((0_u64, 0_u64, 0_u64), |totals, pool| {
                let block_size = u64::from(pool.block_size_tokens);
                Some((
                    totals
                        .0
                        .checked_add(pool.capacity_blocks?.checked_mul(block_size)?)?,
                    totals
                        .1
                        .checked_add(pool.used_blocks?.checked_mul(block_size)?)?,
                    totals
                        .2
                        .checked_add(pool.free_blocks?.checked_mul(block_size)?)?,
                ))
            })?;
    if capacity == 0 || used.checked_add(free) != Some(capacity) {
        return None;
    }
    Some(RoutingCacheStats {
        role,
        capacity_tokens: capacity,
        used_tokens: used,
        free_tokens: free,
    })
}

fn current_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn pool(role: WorkerType, capacity: u64, used: u64, complete: bool) -> KvPoolStats {
        KvPoolStats {
            namespace: "ns".to_string(),
            component: role.as_str().to_string(),
            endpoint: "generate".to_string(),
            role,
            storage_tier: "device",
            block_size_tokens: 16,
            expected_ranks: 1,
            observed_ranks: u64::from(complete),
            capacity_blocks: Some(capacity),
            used_blocks: Some(used),
            free_blocks: capacity.checked_sub(used),
            active_decode_blocks: None,
            active_prefill_tokens: None,
            complete,
        }
    }

    #[test]
    fn routing_cache_uses_decode_instead_of_prefill() {
        let stats = routing_cache(&[
            pool(WorkerType::Prefill, 10, 9, true),
            pool(WorkerType::Decode, 100, 25, true),
        ])
        .unwrap();
        assert_eq!(stats.role, WorkerType::Decode);
        assert_eq!(stats.capacity_tokens, 1_600);
        assert_eq!(stats.used_tokens, 400);
        assert_eq!(stats.free_tokens, 1_200);
        assert!(
            serde_json::to_value(&stats)
                .unwrap()
                .get("complete")
                .is_none()
        );
    }

    #[test]
    fn routing_cache_uses_aggregated_instead_of_role_specific_pools() {
        let stats = routing_cache(&[
            pool(WorkerType::Aggregated, 200, 50, true),
            pool(WorkerType::Decode, 100, 25, true),
        ])
        .unwrap();
        assert_eq!(stats.role, WorkerType::Aggregated);
        assert_eq!(stats.capacity_tokens, 3_200);
        assert_eq!(stats.used_tokens, 800);
        assert_eq!(stats.free_tokens, 2_400);
    }

    #[test]
    fn incomplete_pool_does_not_export_partial_totals() {
        assert!(routing_cache(&[pool(WorkerType::Aggregated, 100, 25, false)]).is_none());
    }

    #[test]
    fn zero_capacity_pool_is_not_routing_cache_state() {
        assert!(routing_cache(&[pool(WorkerType::Aggregated, 0, 0, true)]).is_none());
    }

    #[tokio::test]
    async fn stream_emits_an_immediate_empty_snapshot() {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let response = kv_stats_stream_response(Arc::new(ModelManager::new()), shutdown.clone());
        let mut body = response.into_body().into_data_stream();
        let line = tokio::time::timeout(Duration::from_millis(100), body.next())
            .await
            .expect("snapshot should be immediate")
            .expect("stream should remain open")
            .expect("snapshot body should be readable");
        let value: serde_json::Value = serde_json::from_slice(&line).unwrap();
        assert_eq!(value["type"], "kv_stats_snapshot");
        assert_eq!(value["models"], serde_json::json!([]));
        shutdown.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn stream_repeats_unchanged_snapshots_and_closes_on_shutdown() {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let response = kv_stats_stream_response(Arc::new(ModelManager::new()), shutdown.clone());
        let mut body = response.into_body().into_data_stream();

        let first = body.next().await.unwrap().unwrap();
        tokio::time::advance(SNAPSHOT_INTERVAL).await;
        let second = body.next().await.unwrap().unwrap();
        let first: serde_json::Value = serde_json::from_slice(&first).unwrap();
        let second: serde_json::Value = serde_json::from_slice(&second).unwrap();
        assert_eq!(first["models"], second["models"]);
        assert_ne!(first["snapshot_id"], second["snapshot_id"]);

        shutdown.cancel();
        assert!(body.next().await.is_none());
    }
}
