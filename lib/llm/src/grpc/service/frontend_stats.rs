// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use crate::http::service::engine_stats::{self, StatsUpdate};
use crate::http::service::kv_placements::{self, PlacementEventBatch, PlacementStreamEvent};
use crate::http::service::kv_stats;
use crate::http::service::service_v2::State;
use crate::worker_type::WorkerType;
use dynamo_kv_router::protocols::{
    BlockExtraInfo, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
    RouterEvent, StorageTier, WireResidencyDomain,
};

pub(crate) mod proto {
    tonic::include_proto!("dynamo.frontend.stats.v1");
}

use proto::frontend_stats_server::{FrontendStats, FrontendStatsServer};

#[derive(Clone)]
struct FrontendStatsService {
    state: Arc<State>,
}

pub(crate) fn router(state: Arc<State>) -> axum::Router {
    tonic::service::Routes::new(FrontendStatsServer::new(FrontendStatsService { state }))
        .into_axum_router()
}

#[tonic::async_trait]
impl FrontendStats for FrontendStatsService {
    type WatchStatsStream =
        Pin<Box<dyn Stream<Item = Result<proto::StatsUpdate, Status>> + Send + 'static>>;
    type WatchKvPlacementsStream =
        Pin<Box<dyn Stream<Item = Result<proto::KvPlacementUpdate, Status>> + Send + 'static>>;

    #[allow(clippy::result_large_err)]
    async fn watch_stats(
        &self,
        _request: Request<proto::WatchStatsRequest>,
    ) -> Result<Response<Self::WatchStatsStream>, Status> {
        let stream = engine_stats::stats_stream(
            self.state.engine_stats().clone(),
            self.state.manager_clone(),
            self.state.cancel_token().clone(),
        )
        .map(|update| Ok(update.into()));
        Ok(Response::new(Box::pin(stream)))
    }

    #[allow(clippy::result_large_err)]
    async fn watch_kv_placements(
        &self,
        _request: Request<proto::WatchKvPlacementsRequest>,
    ) -> Result<Response<Self::WatchKvPlacementsStream>, Status> {
        let stream = kv_placements::placement_stream(
            self.state.manager_clone(),
            self.state.cancel_token().clone(),
        )
        .map(|update| Ok(update.into()));
        Ok(Response::new(Box::pin(stream)))
    }
}

impl From<StatsUpdate> for proto::StatsUpdate {
    fn from(update: StatsUpdate) -> Self {
        let update = match update {
            StatsUpdate::Request(request) => {
                proto::stats_update::Update::RequestStats(proto::RequestStats {
                    request_id: request.request_id,
                    model: request.model,
                    tokens_processed: request.tokens_processed,
                    tokens_generated: request.tokens_generated,
                    finished: request.finished,
                })
            }
            StatsUpdate::Kv(snapshot) => proto::stats_update::Update::KvStats(snapshot.into()),
        };
        Self {
            update: Some(update),
        }
    }
}

impl From<kv_stats::KvStatsSnapshot> for proto::KvStatsSnapshot {
    fn from(snapshot: kv_stats::KvStatsSnapshot) -> Self {
        Self {
            snapshot_id: snapshot.snapshot_id,
            observed_at_unix_ms: snapshot.observed_at_unix_ms,
            models: snapshot.models.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<kv_stats::ModelKvStats> for proto::ModelKvStats {
    fn from(model: kv_stats::ModelKvStats) -> Self {
        Self {
            model: model.model,
            aliases: model.aliases,
            routing_cache: model.routing_cache.map(Into::into),
            pools: model.pools.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<kv_stats::RoutingCacheStats> for proto::RoutingCacheStats {
    fn from(stats: kv_stats::RoutingCacheStats) -> Self {
        Self {
            role: worker_role(stats.role),
            capacity_tokens: stats.capacity_tokens,
            used_tokens: stats.used_tokens,
            free_tokens: stats.free_tokens,
        }
    }
}

impl From<kv_stats::KvPoolStats> for proto::KvPoolStats {
    fn from(stats: kv_stats::KvPoolStats) -> Self {
        Self {
            namespace: stats.namespace,
            component: stats.component,
            endpoint: stats.endpoint,
            role: worker_role(stats.role),
            storage_tier: proto::StorageTier::Device as i32,
            block_size_tokens: stats.block_size_tokens,
            expected_ranks: stats.expected_ranks,
            observed_ranks: stats.observed_ranks,
            capacity_blocks: stats.capacity_blocks,
            used_blocks: stats.used_blocks,
            free_blocks: stats.free_blocks,
            active_decode_blocks: stats.active_decode_blocks,
            active_prefill_tokens: stats.active_prefill_tokens,
            complete: stats.complete,
        }
    }
}

fn worker_role(role: WorkerType) -> i32 {
    (match role {
        WorkerType::Aggregated => proto::WorkerRole::Aggregated,
        WorkerType::Prefill => proto::WorkerRole::Prefill,
        WorkerType::Decode => proto::WorkerRole::Decode,
        WorkerType::Encode => proto::WorkerRole::Encode,
    }) as i32
}

impl From<PlacementStreamEvent> for proto::KvPlacementUpdate {
    fn from(event: PlacementStreamEvent) -> Self {
        let update = match event {
            PlacementStreamEvent::SnapshotBegin { snapshot_id } => {
                proto::kv_placement_update::Update::SnapshotBegin(
                    proto::KvPlacementSnapshotBoundary {
                        snapshot_id,
                        complete: false,
                        cursors: Vec::new(),
                    },
                )
            }
            PlacementStreamEvent::SnapshotEvents(batch) => {
                proto::kv_placement_update::Update::SnapshotEvents(batch.into())
            }
            PlacementStreamEvent::SnapshotEnd {
                snapshot_id,
                complete,
                cursors,
            } => proto::kv_placement_update::Update::SnapshotEnd(
                proto::KvPlacementSnapshotBoundary {
                    snapshot_id,
                    complete,
                    cursors: cursors
                        .into_iter()
                        .map(|cursor| proto::KvPlacementCursor {
                            model: cursor.model,
                            namespace: cursor.namespace,
                            component: cursor.component,
                            endpoint: cursor.endpoint,
                            cursor: cursor.cursor,
                        })
                        .collect(),
                },
            ),
            PlacementStreamEvent::Events(batch) => {
                proto::kv_placement_update::Update::Events(batch.into())
            }
            PlacementStreamEvent::SourceError { source, reason } => {
                proto::kv_placement_update::Update::SourceError(proto::KvPlacementSourceError {
                    source: Some(source.into()),
                    reason,
                })
            }
        };
        Self {
            update: Some(update),
        }
    }
}

impl From<PlacementEventBatch> for proto::KvPlacementEvents {
    fn from(batch: PlacementEventBatch) -> Self {
        Self {
            snapshot_id: batch.snapshot_id,
            source: Some(batch.source.into()),
            cursor: batch.cursor,
            batch_index: u32::try_from(batch.batch_index)
                .expect("placement batch index must fit in u32"),
            batch_count: u32::try_from(batch.batch_count)
                .expect("placement batch count must fit in u32"),
            events: batch.events.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<kv_placements::PlacementSourceInfo> for proto::KvPlacementSource {
    fn from(source: kv_placements::PlacementSourceInfo) -> Self {
        Self {
            model: source.model,
            namespace: source.endpoint.namespace,
            component: source.endpoint.component,
            endpoint: source.endpoint.name,
            block_size_tokens: source.block_size_tokens,
        }
    }
}

impl From<RouterEvent> for proto::RouterEvent {
    fn from(event: RouterEvent) -> Self {
        Self {
            worker_id: event.worker_id,
            storage_tier: storage_tier(event.storage_tier),
            residency_domain: Some(residency_domain(event.residency_domain)),
            state_source: event.state_source.map(|source| source.to_string()),
            event_id: event.event.event_id,
            dp_rank: event.event.dp_rank,
            data: Some(match event.event.data {
                KvCacheEventData::Stored(stored) => {
                    proto::router_event::Data::Stored(stored.into())
                }
                KvCacheEventData::Removed(removed) => {
                    proto::router_event::Data::Removed(removed.into())
                }
                KvCacheEventData::Cleared => {
                    proto::router_event::Data::Cleared(proto::KvCacheClear {})
                }
            }),
        }
    }
}

impl From<KvCacheStoreData> for proto::KvCacheStore {
    fn from(stored: KvCacheStoreData) -> Self {
        Self {
            parent_hash: stored.parent_hash.map(|hash| hash.0),
            start_position: stored.start_position,
            blocks: stored.blocks.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<KvCacheStoredBlockData> for proto::KvCacheBlock {
    fn from(block: KvCacheStoredBlockData) -> Self {
        Self {
            block_hash: block.block_hash.0,
            tokens_hash: block.tokens_hash.0,
            multimodal_objects: block
                .mm_extra_info
                .map(multimodal_objects)
                .unwrap_or_default(),
        }
    }
}

fn multimodal_objects(extra: BlockExtraInfo) -> Vec<proto::MultimodalObject> {
    extra
        .mm_objects
        .into_iter()
        .map(|object| proto::MultimodalObject {
            hash: object.mm_hash,
            offsets: object
                .offsets
                .into_iter()
                .map(|(start, end)| proto::TokenRange {
                    start: u64::try_from(start).expect("token offset must fit in u64"),
                    end: u64::try_from(end).expect("token offset must fit in u64"),
                })
                .collect(),
        })
        .collect()
}

impl From<KvCacheRemoveData> for proto::KvCacheRemove {
    fn from(removed: KvCacheRemoveData) -> Self {
        Self {
            block_hashes: removed
                .block_hashes
                .into_iter()
                .map(|hash| hash.0)
                .collect(),
        }
    }
}

fn storage_tier(tier: StorageTier) -> i32 {
    (match tier {
        StorageTier::Device => proto::StorageTier::Device,
        StorageTier::HostPinned => proto::StorageTier::HostPinned,
        StorageTier::Disk => proto::StorageTier::Disk,
        StorageTier::External => proto::StorageTier::External,
    }) as i32
}

fn residency_domain(domain: WireResidencyDomain) -> proto::ResidencyDomain {
    use proto::residency_domain::Kind;

    let (kind, unknown_value) = match domain {
        WireResidencyDomain::Missing => (Kind::Missing, String::new()),
        WireResidencyDomain::Known(dynamo_kv_router::protocols::ResidencyDomain::Worker) => {
            (Kind::Worker, String::new())
        }
        WireResidencyDomain::Known(dynamo_kv_router::protocols::ResidencyDomain::CacheOwner) => {
            (Kind::CacheOwner, String::new())
        }
        WireResidencyDomain::Unknown(value) => (Kind::Unknown, value.into()),
        WireResidencyDomain::Invalid => (Kind::Invalid, String::new()),
    };
    proto::ResidencyDomain {
        kind: kind as i32,
        unknown_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::service::service_v2;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn grpc_and_http_debug_stats_share_the_frontend_listener() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let service = service_v2::HttpService::builder()
            .port(port)
            .build()
            .unwrap();
        let shutdown = CancellationToken::new();
        let handle = service
            .spawn_with_listener(shutdown.clone(), listener)
            .await;

        let endpoint = format!("http://127.0.0.1:{port}");
        let mut client =
            proto::frontend_stats_client::FrontendStatsClient::connect(endpoint.clone())
                .await
                .unwrap();
        let mut stats = client
            .watch_stats(proto::WatchStatsRequest {})
            .await
            .unwrap()
            .into_inner();
        let first = stats.message().await.unwrap().unwrap();
        assert!(matches!(
            first.update,
            Some(proto::stats_update::Update::KvStats(_))
        ));

        let mut placements = client
            .watch_kv_placements(proto::WatchKvPlacementsRequest {})
            .await
            .unwrap()
            .into_inner();
        let begin = placements.message().await.unwrap().unwrap();
        let end = placements.message().await.unwrap().unwrap();
        assert!(matches!(
            begin.update,
            Some(proto::kv_placement_update::Update::SnapshotBegin(_))
        ));
        assert!(matches!(
            end.update,
            Some(proto::kv_placement_update::Update::SnapshotEnd(boundary)) if boundary.complete
        ));

        let response = reqwest::get(format!("{endpoint}/v1/stats/stream"))
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let removed = reqwest::get(format!("{endpoint}/v1/kv-cache/stats/stream"))
            .await
            .unwrap();
        assert_eq!(removed.status(), reqwest::StatusCode::NOT_FOUND);

        drop(stats);
        drop(placements);
        shutdown.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), handle)
            .await
            .expect("frontend should stop after cancellation")
            .unwrap()
            .unwrap();
    }
}
