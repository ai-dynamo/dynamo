// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_kv_router::{
    SharedCacheQuery, SharedKvCache,
    protocols::{KvCacheEventData, KvCacheStoreData, LocalBlockHash},
    zmq_wire::{RawKvEvent, ZmqEventNormalizer},
};
use dynamo_runtime::component::{Instance, TransportType};
use serde_json::json;
use tokio::sync::watch;

use super::*;
use crate::kv_router::shared_cache::vllm_mooncake_store::test_runtime_config;

const BLOCK_SIZE: u32 = 16;

fn instance(worker_id: WorkerId) -> Instance {
    Instance {
        namespace: "test".to_string(),
        component: "worker".to_string(),
        endpoint: "generate".to_string(),
        instance_id: worker_id,
        transport: TransportType::Nats(format!("test-worker-{worker_id}")),
        device_type: None,
        request_plane_codec: None,
    }
}

fn changed_contract() -> ModelRuntimeConfig {
    let mut runtime = test_runtime_config(BLOCK_SIZE);
    runtime.runtime_data.get_mut("vllm_mooncake_store").unwrap()["coordinator"]["partial_hash_hits"] =
        json!(false);
    runtime
}

struct WatchFixture {
    admitted: watch::Sender<Vec<Instance>>,
    configs: watch::Sender<HashMap<WorkerId, ModelRuntimeConfig>>,
    watch: Arc<parking_lot::Mutex<MooncakeContractWatch>>,
    cache: Arc<MooncakeStoreCache>,
}

impl WatchFixture {
    fn new(admitted: &[WorkerId], configs: HashMap<WorkerId, ModelRuntimeConfig>) -> Self {
        let (admitted, admitted_rx) =
            watch::channel(admitted.iter().copied().map(instance).collect());
        let (configs, configs_rx) = watch::channel(configs);
        let (watch, cache) = MooncakeContractWatch {
            admitted: admitted_rx,
            configs: configs_rx,
            block_size: BLOCK_SIZE,
        }
        .passive_cache();
        Self {
            admitted,
            configs,
            watch,
            cache,
        }
    }

    fn single_worker() -> Self {
        Self::new(&[7], HashMap::from([(7, test_runtime_config(BLOCK_SIZE))]))
    }

    fn revalidate(&self) {
        self.watch.lock().revalidate(&self.cache);
    }
}

struct CacheProbe {
    data: KvCacheStoreData,
    tokens: Vec<u32>,
    hashes: Vec<LocalBlockHash>,
    digest: String,
}

impl CacheProbe {
    fn new(seed: u32) -> Self {
        let mut tokens: Vec<_> = (seed..seed + BLOCK_SIZE).collect();
        let external = 0x1000_u64 + u64::from(seed);
        let payload = rmp_serde::to_vec_named(&json!({
            "type": "BlockStored",
            "block_hashes": [external],
            "parent_block_hash": null,
            "token_ids": tokens,
            "block_size": BLOCK_SIZE,
            "medium": "GPU",
        }))
        .unwrap();
        let raw: RawKvEvent = rmp_serde::from_slice(&payload).unwrap();
        let event = ZmqEventNormalizer::new(BLOCK_SIZE)
            .normalize(raw, 1, WorkerWithDpRank::new(7, 0))
            .unwrap()
            .into_router_event()
            .unwrap();
        let KvCacheEventData::Stored(data) = event.event.data else {
            panic!("expected normalized store");
        };
        assert!(data.shared_cache_eligible);
        let hashes = data.blocks.iter().map(|block| block.tokens_hash).collect();
        tokens.push(seed + BLOCK_SIZE);
        Self {
            data,
            tokens,
            hashes,
            digest: format!("{external:064x}"),
        }
    }

    fn observe(&self, cache: &MooncakeStoreCache, source: WorkerWithDpRank) {
        cache.observe_stored(source, &self.data);
    }

    fn store(&self, cache: &MooncakeStoreCache, sequence: u64) {
        let events: Vec<_> = (0..2)
            .map(|rank| {
                json!({
                    "event_type": "stored",
                    "tenant_id": "default",
                    "object_key": format!("deployment@g0@r{rank}@{}", self.digest),
                    "medium": "cpu",
                })
            })
            .collect();
        cache.apply_test_frames(&[
            Vec::new(),
            sequence.to_be_bytes().to_vec(),
            rmp_serde::to_vec_named(&(0_i64, events, 0_u32)).unwrap(),
        ]);
    }

    async fn hits(&self, cache: &MooncakeStoreCache) -> u32 {
        cache
            .check_blocks(SharedCacheQuery {
                block_hashes: &self.hashes,
                tokens: &self.tokens,
                block_size: BLOCK_SIZE,
                cache_namespace: None,
                shared_cache_eligible: true,
            })
            .await
            .unwrap()
            .total_hits
    }
}

#[test]
fn mooncake_members_use_only_admitted_workers_and_exact_dp_ranks() {
    let mut runtime = test_runtime_config(BLOCK_SIZE);
    runtime.data_parallel_start_rank = 2;
    runtime.data_parallel_size = 2;
    let configs = HashMap::from([
        (7, runtime.clone()),
        (8, test_runtime_config(BLOCK_SIZE)),
        (99, ModelRuntimeConfig::default()),
    ]);
    let (contract, sources) = validate_mooncake_members([7, 8], &configs, BLOCK_SIZE).unwrap();
    assert_eq!(contract, ValidatedContract::from_runtime(&runtime).unwrap());
    assert_eq!(
        sources,
        HashSet::from([
            WorkerWithDpRank::new(7, 2),
            WorkerWithDpRank::new(7, 3),
            WorkerWithDpRank::new(8, 0),
        ])
    );
    assert!(validate_mooncake_members([7, 8, 99], &configs, BLOCK_SIZE).is_err());
    let (contract, sources) = validate_mooncake_members([], &configs, BLOCK_SIZE).unwrap();
    assert!(contract.is_none());
    assert!(sources.is_empty());
}

#[test]
fn mooncake_members_require_compatible_metadata_for_every_admission() {
    let valid = test_runtime_config(BLOCK_SIZE);
    assert!(validate_mooncake_members([7], &HashMap::new(), BLOCK_SIZE).is_err());
    assert!(
        validate_mooncake_members(
            [7],
            &HashMap::from([(7, ModelRuntimeConfig::default())]),
            BLOCK_SIZE,
        )
        .is_err()
    );
    assert!(
        validate_mooncake_members([7], &HashMap::from([(7, valid.clone())]), BLOCK_SIZE * 2)
            .is_err()
    );
    assert!(
        validate_mooncake_members(
            [7, 8],
            &HashMap::from([(7, valid.clone()), (8, changed_contract())]),
            BLOCK_SIZE,
        )
        .is_err()
    );
    for mutate in [
        |runtime: &mut ModelRuntimeConfig| runtime.data_parallel_size = 0,
        |runtime: &mut ModelRuntimeConfig| runtime.data_parallel_start_rank = u32::MAX,
        |runtime: &mut ModelRuntimeConfig| {
            runtime.runtime_data.get_mut("vllm_mooncake_store").unwrap()["schema_version"] =
                json!(999);
        },
    ] {
        let mut invalid = valid.clone();
        mutate(&mut invalid);
        assert!(
            validate_mooncake_members(
                [7, 8],
                &HashMap::from([(7, valid.clone()), (8, invalid)]),
                BLOCK_SIZE,
            )
            .is_err()
        );
    }
}

#[test]
fn mooncake_passive_construction_needs_no_runtime_and_retains_no_tasks() {
    let fixture = WatchFixture::single_worker();
    let cache_weak = Arc::downgrade(&fixture.cache);
    let watch_weak = Arc::downgrade(&fixture.watch);
    assert_eq!(fixture.admitted.receiver_count(), 1);
    assert_eq!(fixture.configs.receiver_count(), 1);
    let WatchFixture {
        admitted,
        configs,
        watch,
        cache,
    } = fixture;
    drop(watch);
    drop(cache);
    assert!(cache_weak.upgrade().is_none());
    assert!(watch_weak.upgrade().is_none());
    assert_eq!(admitted.receiver_count(), 0);
    assert_eq!(configs.receiver_count(), 0);
}

#[tokio::test]
async fn mooncake_pending_watch_changes_withhold_queries_and_new_learning() {
    for change_configs in [false, true] {
        let fixture = WatchFixture::single_worker();
        fixture.revalidate();
        let old = CacheProbe::new(1);
        old.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
        old.store(&fixture.cache, 1);
        assert_eq!(old.hits(&fixture.cache).await, 1);

        if change_configs {
            fixture.configs.send_modify(|configs| {
                configs.get_mut(&7).unwrap().context_length = Some(4096);
            });
        } else {
            fixture.admitted.send_replace(vec![instance(7)]);
        }
        assert_eq!(old.hits(&fixture.cache).await, 0);
        let pending = CacheProbe::new(100);
        pending.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
        fixture.revalidate();
        assert_eq!(old.hits(&fixture.cache).await, 1);
        pending.store(&fixture.cache, 2);
        assert_eq!(pending.hits(&fixture.cache).await, 0);
        pending.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
        assert_eq!(pending.hits(&fixture.cache).await, 1);
    }
}

#[tokio::test]
async fn mooncake_pending_store_batches_cannot_populate_residency() {
    let fixture = WatchFixture::single_worker();
    fixture.revalidate();
    let probe = CacheProbe::new(1);
    probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    fixture.admitted.send_replace(vec![instance(7)]);
    probe.store(&fixture.cache, 1);
    fixture.revalidate();
    assert_eq!(probe.hits(&fixture.cache).await, 0);
    probe.store(&fixture.cache, 2);
    assert_eq!(probe.hits(&fixture.cache).await, 1);
}

#[tokio::test]
async fn mooncake_compatible_membership_preserves_state_but_fences_departed_sources() {
    let fixture = WatchFixture::new(
        &[7],
        HashMap::from([
            (7, test_runtime_config(BLOCK_SIZE)),
            (8, test_runtime_config(BLOCK_SIZE)),
        ]),
    );
    fixture.revalidate();
    let old = CacheProbe::new(1);
    old.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    old.store(&fixture.cache, 1);
    assert_eq!(old.hits(&fixture.cache).await, 1);

    fixture.admitted.send_replace(vec![instance(8)]);
    fixture.revalidate();
    assert_eq!(old.hits(&fixture.cache).await, 1);
    let new = CacheProbe::new(100);
    new.store(&fixture.cache, 2);
    for source in [WorkerWithDpRank::new(7, 0), WorkerWithDpRank::new(8, 1)] {
        new.observe(&fixture.cache, source);
        assert_eq!(new.hits(&fixture.cache).await, 0);
    }
    new.observe(&fixture.cache, WorkerWithDpRank::new(8, 0));
    assert_eq!(new.hits(&fixture.cache).await, 1);
}

#[tokio::test]
async fn mooncake_rejected_endpoint_members_neither_disable_nor_teach_cache() {
    let fixture = WatchFixture::single_worker();
    fixture.revalidate();
    let old = CacheProbe::new(1);
    old.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    old.store(&fixture.cache, 1);
    fixture.configs.send_modify(|configs| {
        configs.insert(99, ModelRuntimeConfig::default());
    });
    fixture.revalidate();
    assert_eq!(old.hits(&fixture.cache).await, 1);

    let rejected = CacheProbe::new(100);
    rejected.observe(&fixture.cache, WorkerWithDpRank::new(99, 0));
    rejected.store(&fixture.cache, 2);
    assert_eq!(rejected.hits(&fixture.cache).await, 0);
    rejected.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    assert_eq!(rejected.hits(&fixture.cache).await, 1);
}

#[tokio::test]
async fn mooncake_contract_replacement_clears_both_identity_and_residency() {
    let fixture = WatchFixture::single_worker();
    fixture.revalidate();
    let probe = CacheProbe::new(1);
    probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    probe.store(&fixture.cache, 1);
    assert_eq!(probe.hits(&fixture.cache).await, 1);

    fixture
        .configs
        .send_replace(HashMap::from([(7, changed_contract())]));
    fixture.revalidate();
    assert_eq!(probe.hits(&fixture.cache).await, 0);
    probe.store(&fixture.cache, 1);
    assert_eq!(probe.hits(&fixture.cache).await, 0);
    probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    assert_eq!(probe.hits(&fixture.cache).await, 1);

    fixture
        .configs
        .send_replace(HashMap::from([(7, test_runtime_config(BLOCK_SIZE))]));
    fixture.revalidate();
    probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    assert_eq!(probe.hits(&fixture.cache).await, 0);
    probe.store(&fixture.cache, 1);
    assert_eq!(probe.hits(&fixture.cache).await, 1);
}

#[tokio::test]
async fn mooncake_missing_admitted_metadata_disables_and_discards_old_state() {
    for missing_config in [false, true] {
        let fixture = WatchFixture::single_worker();
        fixture.revalidate();
        let probe = CacheProbe::new(1);
        probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
        probe.store(&fixture.cache, 1);
        assert_eq!(probe.hits(&fixture.cache).await, 1);
        fixture.configs.send_modify(|configs| {
            if missing_config {
                configs.remove(&7);
            } else {
                configs.insert(7, ModelRuntimeConfig::default());
            }
        });
        fixture.revalidate();
        assert_eq!(probe.hits(&fixture.cache).await, 0);
        fixture
            .configs
            .send_replace(HashMap::from([(7, test_runtime_config(BLOCK_SIZE))]));
        fixture.revalidate();
        assert_eq!(probe.hits(&fixture.cache).await, 0);
    }
}

#[tokio::test]
async fn mooncake_closed_watch_channels_fail_closed_without_revalidation() {
    for close_configs in [false, true] {
        let fixture = WatchFixture::single_worker();
        fixture.revalidate();
        let probe = CacheProbe::new(1);
        probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
        probe.store(&fixture.cache, 1);
        assert_eq!(probe.hits(&fixture.cache).await, 1);
        let WatchFixture {
            admitted,
            configs,
            watch,
            cache,
        } = fixture;
        let mut admitted = Some(admitted);
        let mut configs = Some(configs);
        if close_configs {
            drop(configs.take());
        } else {
            drop(admitted.take());
        }
        assert_eq!(probe.hits(&cache).await, 0);
        watch.lock().revalidate(&cache);
        assert_eq!(probe.hits(&cache).await, 0);
    }
}

#[tokio::test]
async fn mooncake_same_endpoint_successor_has_fresh_cache_and_admission_channel() {
    let old = WatchFixture::single_worker();
    old.revalidate();
    let successor = WatchFixture::new(&[8], HashMap::from([(8, test_runtime_config(BLOCK_SIZE))]));
    successor.revalidate();
    assert_eq!(instance(7).endpoint_id(), instance(8).endpoint_id());
    let probe = CacheProbe::new(1);
    probe.observe(&old.cache, WorkerWithDpRank::new(7, 0));
    probe.store(&old.cache, 1);
    assert_eq!(probe.hits(&old.cache).await, 1);
    assert_eq!(probe.hits(&successor.cache).await, 0);
    probe.store(&successor.cache, 1);
    probe.observe(&successor.cache, WorkerWithDpRank::new(7, 0));
    assert_eq!(probe.hits(&successor.cache).await, 0);
    probe.observe(&successor.cache, WorkerWithDpRank::new(8, 0));
    assert_eq!(probe.hits(&successor.cache).await, 1);
    drop(old.admitted);
    assert_eq!(probe.hits(&old.cache).await, 0);
    assert_eq!(probe.hits(&successor.cache).await, 1);
}

#[tokio::test]
async fn mooncake_spawned_watch_revalidates_changed_contract_snapshots() {
    let fixture = WatchFixture::single_worker();
    let cancellation = CancellationToken::new();
    let task = MooncakeContractWatch::spawn(
        Arc::clone(&fixture.watch),
        Arc::clone(&fixture.cache),
        cancellation.clone(),
    );
    let probe = CacheProbe::new(1);
    probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    probe.store(&fixture.cache, 1);
    assert_eq!(probe.hits(&fixture.cache).await, 1);
    fixture
        .configs
        .send_replace(HashMap::from([(7, changed_contract())]));
    assert_eq!(probe.hits(&fixture.cache).await, 0);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if matches!(fixture.watch.lock().configs.has_changed(), Ok(false)) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    assert_eq!(probe.hits(&fixture.cache).await, 0);
    probe.store(&fixture.cache, 1);
    assert_eq!(probe.hits(&fixture.cache).await, 1);
    cancellation.cancel();
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn mooncake_spawned_watch_stops_when_either_channel_closes() {
    for close_configs in [false, true] {
        let fixture = WatchFixture::single_worker();
        let task = MooncakeContractWatch::spawn(
            Arc::clone(&fixture.watch),
            Arc::clone(&fixture.cache),
            CancellationToken::new(),
        );
        let probe = CacheProbe::new(1);
        probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
        probe.store(&fixture.cache, 1);
        assert_eq!(probe.hits(&fixture.cache).await, 1);
        let WatchFixture {
            admitted,
            configs,
            watch,
            cache,
        } = fixture;
        let mut admitted = Some(admitted);
        let mut configs = Some(configs);
        if close_configs {
            drop(configs.take());
        } else {
            drop(admitted.take());
        }
        tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(probe.hits(&cache).await, 0);
        if let Some(admitted) = admitted {
            assert_eq!(admitted.receiver_count(), 1);
        }
        if let Some(configs) = configs {
            assert_eq!(configs.receiver_count(), 1);
        }
        drop(watch);
    }
}

#[tokio::test]
async fn mooncake_spawned_watch_stops_on_cancellation_and_releases_receivers() {
    let fixture = WatchFixture::single_worker();
    let cancellation = CancellationToken::new();
    let task = MooncakeContractWatch::spawn(
        Arc::clone(&fixture.watch),
        Arc::clone(&fixture.cache),
        cancellation.clone(),
    );
    let probe = CacheProbe::new(1);
    probe.observe(&fixture.cache, WorkerWithDpRank::new(7, 0));
    probe.store(&fixture.cache, 1);
    assert_eq!(probe.hits(&fixture.cache).await, 1);
    cancellation.cancel();
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(probe.hits(&fixture.cache).await, 0);
    assert_eq!(fixture.admitted.receiver_count(), 1);
    assert_eq!(fixture.configs.receiver_count(), 1);
}
