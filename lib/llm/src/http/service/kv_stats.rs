// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical frontend projection of worker-owned KV capacity and occupancy.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::discovery::ModelManager;
use crate::worker_type::WorkerType;

static NEXT_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Serialize)]
pub(crate) struct KvStatsSnapshot {
    pub(crate) v: u8,
    #[serde(rename = "type")]
    pub(crate) event_type: &'static str,
    pub(crate) snapshot_id: u64,
    pub(crate) observed_at_unix_ms: u64,
    pub(crate) models: Vec<ModelKvStats>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelKvStats {
    pub(crate) model: String,
    pub(crate) aliases: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) routing_cache: Option<RoutingCacheStats>,
    pub(crate) pools: Vec<KvPoolStats>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct RoutingCacheStats {
    pub(crate) role: WorkerType,
    pub(crate) capacity_tokens: u64,
    pub(crate) used_tokens: u64,
    pub(crate) free_tokens: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct KvPoolStats {
    pub(crate) namespace: String,
    pub(crate) component: String,
    pub(crate) endpoint: String,
    pub(crate) role: WorkerType,
    pub(crate) storage_tier: &'static str,
    pub(crate) block_size_tokens: u32,
    pub(crate) expected_ranks: u64,
    pub(crate) observed_ranks: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) capacity_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) used_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) free_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) active_decode_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) active_prefill_tokens: Option<u64>,
    pub(crate) complete: bool,
}

pub(crate) fn build_snapshot(manager: &ModelManager) -> KvStatsSnapshot {
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
}
