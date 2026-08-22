// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use tokio::sync::{Mutex, watch};
use tokio_util::sync::CancellationToken;

use super::mooncake_ha::MooncakeLeaderResolver;
use super::*;

struct SequenceLeaderResolver {
    leaders: Mutex<VecDeque<String>>,
    calls: AtomicUsize,
}

#[async_trait]
impl MooncakeLeaderResolver for SequenceLeaderResolver {
    async fn current_leader(&self) -> anyhow::Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.leaders
            .lock()
            .await
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("no leader left in test sequence"))
    }
}

fn mooncake_config() -> SglangHicacheMooncakeConfig {
    SglangHicacheMooncakeConfig {
        backend: "mooncake".to_string(),
        page_size: 4,
        tp_size: 1,
        pp_size: 1,
        is_mla_model: false,
        is_eagle: false,
        tp_lcm_size: None,
        should_split_heads: false,
        extra_backend_tag: None,
        kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
        ha: MooncakeHaConfig::default(),
    }
}

fn ha_config(master_server_address: &str, cluster_id: &str) -> MooncakeHaConfig {
    MooncakeHaConfig {
        master_server_address: Some(master_server_address.to_string()),
        cluster_id: Some(cluster_id.to_string()),
        redis_db_index: None,
    }
}

fn runtime_watch_with_config(config: SglangHicacheMooncakeConfig) -> RuntimeConfigWatch {
    let mut runtime_config = ModelRuntimeConfig::new();
    runtime_config
        .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, config)
        .unwrap();
    let (_tx, rx) = watch::channel(HashMap::from([(1, runtime_config)]));
    rx
}

#[tokio::test]
async fn test_resolved_kv_events_endpoint_follows_ha_leader_change() {
    let config = SglangHicacheMooncakeConfig {
        ha: ha_config("etcd://unused:2379", "tenant-a"),
        ..mooncake_config()
    };
    let resolver = Arc::new(SequenceLeaderResolver {
        leaders: Mutex::new(VecDeque::from([
            "10.0.0.1:50051".to_string(),
            "10.0.0.2:50051".to_string(),
        ])),
        calls: AtomicUsize::new(0),
    });
    let cache = HicacheSharedKvCache::new(runtime_watch_with_config(config));
    let ha = cache.resolve_mooncake_config_and_endpoint().unwrap().0.ha;
    cache
        .ha_resolver
        .replace_resolver_for_test(&ha, resolver.clone())
        .await
        .unwrap();

    let first = cache.resolved_kv_events_endpoint(false).await.unwrap();
    let refreshed = cache.resolved_kv_events_endpoint(true).await.unwrap();

    assert_eq!(first.as_deref(), Some("tcp://10.0.0.1:5557"));
    assert_eq!(refreshed.as_deref(), Some("tcp://10.0.0.2:5557"));
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_frontend_endpoint_override_is_not_rewritten_for_ha() {
    let config = SglangHicacheMooncakeConfig {
        ha: ha_config("etcd://unused:2379", "tenant-a"),
        ..mooncake_config()
    };
    let cache = HicacheSharedKvCache::new_with_cancellation_and_endpoint(
        runtime_watch_with_config(config),
        CancellationToken::new(),
        Some("tcp://mooncake-events.dynamo.svc:5557".to_string()),
    );

    assert_eq!(
        cache
            .resolved_kv_events_endpoint(true)
            .await
            .unwrap()
            .as_deref(),
        Some("tcp://mooncake-events.dynamo.svc:5557")
    );
}

#[tokio::test]
async fn test_unsupported_ha_backend_keeps_configured_event_endpoint() {
    let config = SglangHicacheMooncakeConfig {
        ha: ha_config("consul://consul:8500", "tenant-a"),
        ..mooncake_config()
    };
    let cache = HicacheSharedKvCache::new(runtime_watch_with_config(config));

    assert_eq!(
        cache
            .resolved_kv_events_endpoint(true)
            .await
            .unwrap()
            .as_deref(),
        Some("tcp://127.0.0.1:5557")
    );
}

#[tokio::test]
async fn test_unparseable_ha_locator_keeps_configured_event_endpoint() {
    let config = SglangHicacheMooncakeConfig {
        ha: ha_config("redis://user:password@redis:6379", "tenant-a"),
        ..mooncake_config()
    };
    let cache = HicacheSharedKvCache::new(runtime_watch_with_config(config));

    assert_eq!(
        cache
            .resolved_kv_events_endpoint(true)
            .await
            .unwrap()
            .as_deref(),
        Some("tcp://127.0.0.1:5557")
    );
}

#[tokio::test]
async fn test_unrewritable_event_endpoint_keeps_configured_endpoint() {
    let config = SglangHicacheMooncakeConfig {
        kv_events_endpoint: Some("http://127.0.0.1:5557".to_string()),
        ha: ha_config("etcd://unused:2379", "tenant-a"),
        ..mooncake_config()
    };
    let resolver = Arc::new(SequenceLeaderResolver {
        leaders: Mutex::new(VecDeque::from(["10.0.0.1:50051".to_string()])),
        calls: AtomicUsize::new(0),
    });
    let cache = HicacheSharedKvCache::new(runtime_watch_with_config(config));
    let ha = cache.resolve_mooncake_config_and_endpoint().unwrap().0.ha;
    cache
        .ha_resolver
        .replace_resolver_for_test(&ha, resolver)
        .await
        .unwrap();

    assert_eq!(
        cache
            .resolved_kv_events_endpoint(true)
            .await
            .unwrap()
            .as_deref(),
        Some("http://127.0.0.1:5557")
    );
}

#[test]
fn test_leader_unavailable_refresh_clears_state_before_reconnect() {
    let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
    cache.apply_batch(
        1,
        vec![MooncakeObjectEvent {
            event_type: "stored".to_string(),
            object_key: Some("stale-key".to_string()),
            tenant_id: "default".to_string(),
            group_id: Some("stale-group".to_string()),
        }],
    );

    let should_reconnect = cache.should_reconnect_after_endpoint_resolution(
        "tcp://10.0.0.1:5557",
        Err(mooncake_ha::leader_unavailable_for_test()),
    );

    assert!(should_reconnect);
    assert!(cache.present_keys.is_empty());
    assert!(cache.group_states.is_empty());
    assert!(!cache.has_sequence.load(Ordering::Acquire));
}

#[test]
fn test_mooncake_runtime_config_without_ha_fields_is_backward_compatible() {
    let mut value = serde_json::to_value(mooncake_config()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("master_server_address");
    value.as_object_mut().unwrap().remove("cluster_id");

    let config: SglangHicacheMooncakeConfig = serde_json::from_value(value).unwrap();

    assert_eq!(config.ha.master_server_address, None);
    assert_eq!(config.ha.cluster_id, None);
}

#[test]
fn test_mooncake_ha_metadata_wire_format_remains_flat() {
    let mut config = mooncake_config();
    config.ha = MooncakeHaConfig {
        master_server_address: Some("redis://redis:6379".to_string()),
        cluster_id: Some("tenant-a".to_string()),
        redis_db_index: Some(7),
    };

    let value = serde_json::to_value(&config).unwrap();

    assert_eq!(value["master_server_address"], "redis://redis:6379");
    assert_eq!(value["cluster_id"], "tenant-a");
    assert_eq!(value["redis_db_index"], 7);
    assert!(value.get("ha").is_none());
    let roundtrip: SglangHicacheMooncakeConfig = serde_json::from_value(value).unwrap();
    assert_eq!(roundtrip.ha, config.ha);
}

#[test]
fn test_ha_metadata_tolerates_old_worker_omission() {
    let mut advertised = mooncake_config();
    advertised.ha = ha_config("etcd://etcd:2379", "tenant-a");
    let mut old_worker = advertised.clone();
    old_worker.ha = MooncakeHaConfig::default();
    let mut advertised_runtime = ModelRuntimeConfig::new();
    advertised_runtime
        .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, advertised)
        .unwrap();
    let mut old_runtime = ModelRuntimeConfig::new();
    old_runtime
        .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, old_worker)
        .unwrap();
    let (_tx, runtime_configs) =
        watch::channel(HashMap::from([(1, old_runtime), (2, advertised_runtime)]));
    let cache = HicacheSharedKvCache::new(runtime_configs);

    let (resolved, _) = cache.resolve_mooncake_config_and_endpoint().unwrap();

    assert_eq!(
        resolved.ha.master_server_address.as_deref(),
        Some("etcd://etcd:2379")
    );
    assert_eq!(resolved.ha.cluster_id.as_deref(), Some("tenant-a"));
}

#[test]
fn test_conflicting_ha_metadata_disables_shared_cache() {
    let mut first = mooncake_config();
    first.ha = ha_config("etcd://etcd-a:2379", "tenant-a");
    let mut second = mooncake_config();
    second.ha = ha_config("etcd://etcd-b:2379", "tenant-a");
    let mut first_runtime = ModelRuntimeConfig::new();
    first_runtime
        .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, first)
        .unwrap();
    let mut second_runtime = ModelRuntimeConfig::new();
    second_runtime
        .set_engine_specific(SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY, second)
        .unwrap();
    let (_tx, runtime_configs) =
        watch::channel(HashMap::from([(1, first_runtime), (2, second_runtime)]));
    let cache = HicacheSharedKvCache::new(runtime_configs);

    assert!(cache.resolve_mooncake_config_and_endpoint().is_none());
}

#[test]
fn test_layout_snapshot_is_enriched_after_mixed_version_upgrade() {
    let cache = HicacheSharedKvCache::new(runtime_watch_with_config(mooncake_config()));
    let old_layout = mooncake_config();
    cache.clear_on_layout_change(&old_layout);
    cache.present_keys.insert("survives-enrichment".to_string());

    let mut enriched = old_layout.clone();
    enriched.ha = ha_config("etcd://etcd:2379", "cluster-a");
    cache.clear_on_layout_change(&enriched);

    let saved = cache.last_layout.load_full().unwrap();
    assert_eq!(
        saved.ha.master_server_address.as_deref(),
        Some("etcd://etcd:2379")
    );
    assert_eq!(saved.ha.cluster_id.as_deref(), Some("cluster-a"));
    assert!(cache.present_keys.contains("survives-enrichment"));

    let mut changed = enriched;
    changed.ha.cluster_id = Some("cluster-b".to_string());
    cache.clear_on_layout_change(&changed);
    assert!(cache.present_keys.is_empty());
}
