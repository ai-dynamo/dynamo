// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's KV index: the shared `dynamo_kv_router` [`Indexer`] built
//! from router configuration with runtime-backed metrics and, when configured,
//! a request-plane remote primary.

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed,
    approx::PruneConfig,
    config::{ApproximateCachePolicyKind, KvRouterConfig},
    indexer::{
        ApproximateRetentionConfig, KvIndexer, KvIndexerMetrics, LowerTierIndexers,
        ThreadPoolIndexer,
    },
};

// Re-export tiered-match types so internal callers (`indexer::TieredMatchDetails`)
// keep working after these types moved to `dynamo-kv-router`.
pub(crate) use dynamo_kv_router::indexer::TieredMatchDetails;
#[allow(unused_imports)]
pub(crate) use dynamo_kv_router::indexer::WireTieredMatchDetails;
pub use dynamo_kv_router::services::indexer::backend::{Indexer, RemotePrimary, SideIndexer};
pub(crate) use dynamo_kv_router::services::indexer::recording::ApproximateRequestLease;
use dynamo_runtime::component::Component;
pub(crate) use ingress::{RuntimeIngress, RuntimeIngressArgs};
use tokio_util::sync::CancellationToken;

mod embedding_cache;
mod ingress;
mod recovery;
pub mod remote;

pub use self::embedding_cache::{
    EmbeddingCacheIndexer, preprocessed_multimodal_cache_keys, try_build_cache_indexer,
};
use self::remote::RemoteIndexer;
pub use self::remote::{ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
#[cfg(feature = "ckf-diagnostics")]
pub(crate) use recovery::WorkerQueryHealthSnapshot;
pub(crate) use recovery::{
    DEFAULT_RECOVERY_ATTEMPT_TIMEOUT, KvEventSubscriptionHandle, RecoveryResetReason,
    RecoverySupervisor, RecoveryTarget, TargetFaultDisposition, start_target_subscriber,
};
#[cfg(test)]
pub(crate) use recovery::{WorkerQueryClient, WorkerQueryTransport};
pub(crate) use recovery::{
    start_subscriber, start_worker_kv_query_endpoint, start_worker_kv_query_endpoint_with_status,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedApproximatePrimaryPolicy {
    Disabled,
    Ttl,
    Lru,
    TtlRemoteFallback,
}

fn resolve_approximate_primary_policy(
    config: &KvRouterConfig,
) -> Result<ResolvedApproximatePrimaryPolicy> {
    if config.use_kv_events
        && config.router_approximate_cache_policy == ApproximateCachePolicyKind::Lru
    {
        anyhow::bail!(
            "router_approximate_cache_policy=lru requires use_kv_events=false; the local side indexer is TTL-only"
        );
    }
    if config.overlap_score_credit <= 0.0 {
        return Ok(ResolvedApproximatePrimaryPolicy::Disabled);
    }
    if config.use_kv_events
        || config.router_approximate_cache_policy == ApproximateCachePolicyKind::Ttl
    {
        return Ok(ResolvedApproximatePrimaryPolicy::Ttl);
    }
    if config.use_remote_indexer || config.serve_indexer {
        return Ok(ResolvedApproximatePrimaryPolicy::TtlRemoteFallback);
    }
    Ok(ResolvedApproximatePrimaryPolicy::Lru)
}

/// Build the frontend router's index for `kv_router_config`.
pub(crate) async fn build(
    component: &Component,
    kv_router_config: &KvRouterConfig,
    block_size: u32,
    model_name: Option<&str>,
    cancellation_token: CancellationToken,
) -> Result<Indexer> {
    let approximate_policy = resolve_approximate_primary_policy(kv_router_config)?;
    if approximate_policy == ResolvedApproximatePrimaryPolicy::Disabled {
        return Ok(Indexer::None);
    }

    if approximate_policy == ResolvedApproximatePrimaryPolicy::TtlRemoteFallback {
        tracing::warn!(
            use_remote_indexer = kv_router_config.use_remote_indexer,
            serve_indexer = kv_router_config.serve_indexer,
            "Approximate LRU requires a router-local primary indexer; falling back to TTL"
        );
    }

    if kv_router_config.router_predicted_ttl_secs.is_some() && !kv_router_config.use_kv_events {
        anyhow::bail!(
            "router_predicted_ttl_secs requires use_kv_events=true; \
             do not combine a primary approximate indexer with a side approximate indexer"
        );
    }
    if kv_router_config.use_remote_indexer {
        let model_name = model_name
            .ok_or_else(|| {
                anyhow::anyhow!("model_name is required when use_remote_indexer is configured")
            })?
            .to_string();
        let indexer_component_name = component.name();
        tracing::info!(
            indexer_component = %indexer_component_name,
            model_name,
            "Using remote KV indexer"
        );
        let remote =
            RemoteIndexer::new(component, model_name, kv_router_config.use_kv_events).await?;
        let approx = predict_on_route_side_indexer(
            component,
            kv_router_config,
            block_size,
            cancellation_token.child_token(),
        );
        return Ok(Indexer::Remote {
            primary: Arc::new(remote) as Arc<dyn RemotePrimary>,
            approx,
            primary_records_routing_decisions: !kv_router_config.use_kv_events,
        });
    }

    if !kv_router_config.use_kv_events {
        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        let prune_config = PruneConfig {
            ttl: Duration::from_secs_f64(kv_router_config.router_ttl_secs),
        };
        let retention = if approximate_policy == ResolvedApproximatePrimaryPolicy::Lru {
            tracing::info!(
                "Starting local primary approximate indexer with capacity-bounded LRU retention"
            );
            ApproximateRetentionConfig::Lru {
                fallback_ttl: prune_config,
            }
        } else {
            ApproximateRetentionConfig::Ttl(prune_config)
        };
        if kv_router_config.router_event_threads > 1 {
            return Ok(Indexer::Concurrent {
                primary: Arc::new(
                    ThreadPoolIndexer::new_with_metrics_and_approximate_retention(
                        ConcurrentRadixTreeCompressed::new(),
                        kv_router_config.router_event_threads as usize,
                        block_size,
                        Some(kv_indexer_metrics.clone()),
                        Some(retention),
                    ),
                ),
                lower_tier: LowerTierIndexers::new_with_metrics(
                    kv_router_config.router_event_threads as usize,
                    block_size,
                    Some(kv_indexer_metrics),
                ),
                approx: None,
                primary_records_routing_decisions: true,
            });
        }

        return Ok(Indexer::Single {
            primary: KvIndexer::new_with_approximate_retention(
                cancellation_token.child_token(),
                block_size,
                kv_indexer_metrics.clone(),
                Some(retention),
            ),
            lower_tier: LowerTierIndexers::new_with_metrics(
                1,
                block_size,
                Some(kv_indexer_metrics),
            ),
            approx: None,
            primary_records_routing_decisions: true,
        });
    }

    let approx = predict_on_route_side_indexer(
        component,
        kv_router_config,
        block_size,
        cancellation_token.child_token(),
    );

    if kv_router_config.router_event_threads > 1 {
        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        return Ok(Indexer::Concurrent {
            primary: Arc::new(ThreadPoolIndexer::new_with_metrics(
                ConcurrentRadixTreeCompressed::new(),
                kv_router_config.router_event_threads as usize,
                block_size,
                Some(kv_indexer_metrics.clone()),
            )),
            lower_tier: LowerTierIndexers::new_with_metrics(
                kv_router_config.router_event_threads as usize,
                block_size,
                Some(kv_indexer_metrics),
            ),
            approx,
            primary_records_routing_decisions: false,
        });
    }

    let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
    Ok(Indexer::Single {
        primary: KvIndexer::new_with_pruning(
            cancellation_token.child_token(),
            block_size,
            kv_indexer_metrics.clone(),
            None,
        ),
        lower_tier: LowerTierIndexers::new_with_metrics(1, block_size, Some(kv_indexer_metrics)),
        approx,
        primary_records_routing_decisions: false,
    })
}

/// Predict-on-route side indexer for `router_predicted_ttl_secs`, with the
/// component's indexer metrics.
fn predict_on_route_side_indexer(
    component: &Component,
    kv_router_config: &KvRouterConfig,
    block_size: u32,
    cancellation_token: CancellationToken,
) -> Option<SideIndexer> {
    let ttl_secs = kv_router_config.router_predicted_ttl_secs?;
    tracing::info!(
        ttl_secs,
        "Starting predict-on-route side indexer (short-TTL approximate)"
    );
    Some(SideIndexer::new(
        Duration::from_secs_f64(ttl_secs),
        block_size,
        kv_router_config.router_event_threads as usize,
        KvIndexerMetrics::from_component(component),
        cancellation_token,
    ))
}

#[cfg(test)]
pub(super) mod test_util {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
        compute_seq_hash_for_block,
    };

    pub(crate) fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        prefix_hashes: &[u64],
        local_hashes: &[u64],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        let prefix_block_hashes: Vec<LocalBlockHash> =
            prefix_hashes.iter().copied().map(LocalBlockHash).collect();
        let parent_hash = compute_seq_hash_for_block(&prefix_block_hashes)
            .last()
            .copied()
            .map(ExternalSequenceBlockHash);

        let full_hashes: Vec<LocalBlockHash> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .map(LocalBlockHash)
            .collect();
        let full_sequence_hashes = compute_seq_hash_for_block(&full_hashes);
        let new_sequence_hashes = &full_sequence_hashes[prefix_hashes.len()..];
        let blocks = local_hashes
            .iter()
            .zip(new_sequence_hashes.iter())
            .map(|(&local_hash, &sequence_hash)| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(sequence_hash),
                tokens_hash: LocalBlockHash(local_hash),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            },
            storage_tier,
        )
    }
}
