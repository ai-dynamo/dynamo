// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-JSON and legacy-environment worker configuration boundaries.

use super::*;
use crate::router_valkey_config::RouterValkeyConfig;

const ROUTER_VALKEY_CONFIG_ENV: &str = "DYN_ROUTER_VALKEY_CONFIG";
const VALKEY_WORKER_EVENTS_ENV: &str = "DYN_ROUTER_VALKEY_WORKER_EVENTS";
const VALKEY_REQUIRED_REPLICA_ACKS_ENV: &str = "DYN_ROUTER_VALKEY_REQUIRED_REPLICA_ACKS";
pub(crate) const VALKEY_WORKER_UNREGISTER_TIMEOUT: Duration = Duration::from_secs(2);

fn router_valkey_json_from_env() -> Result<Option<RouterValkeyConfig>> {
    let Some(raw) = std::env::var(ROUTER_VALKEY_CONFIG_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(None);
    };
    RouterValkeyConfig::parse(&raw)
        .with_context(|| format!("invalid {ROUTER_VALKEY_CONFIG_ENV}"))
        .map(Some)
}

fn worker_config_from_json(
    component: &Component,
    config: RouterValkeyConfig,
) -> Result<Option<ValkeyWorkerConfig>> {
    if !config.worker_events {
        return Ok(None);
    }
    let index_scope = config
        .index_scope
        .expect("canonical config requires index_scope for worker events");
    let gc_interval_ms = if config.gc_interval_ms == 0 {
        None
    } else {
        Some(config.gc_interval_ms)
    };
    let (sentinel_urls, sentinel_master_name, sentinel_quorum) = match config.sentinel {
        Some(sentinel) => {
            let urls = sentinel.urls.join(",");
            (Some(urls), Some(sentinel.master_name), sentinel.quorum)
        }
        None => (None, None, None),
    };

    let component_namespace = component.namespace().name();
    Ok(Some(ValkeyWorkerConfig {
        urls: config.urls.join(","),
        index_scope,
        index_namespace: valkey_index_namespace(
            &component_namespace,
            std::env::var("DYN_NAMESPACE").ok().as_deref(),
            std::env::var("DYN_NAMESPACE_WORKER_SUFFIX").ok().as_deref(),
        ),
        // The shared JSON pool controls frontend MATCH concurrency. Workers
        // use a separate, bounded direct-event budget so frontend scale
        // tuning cannot multiply sockets across the worker fleet.
        direct_event_pool_size: crate::kv_router::indexer::valkey::DEFAULT_WORKER_DIRECT_EVENT_LANES
            as u32,
        required_replica_acks: config.required_replica_acks,
        sentinel_urls,
        sentinel_master_name,
        sentinel_quorum,
        allow_degraded_writes: config.allow_degraded_writes,
        worker_lease_ms: config.worker_lease_ms,
        gc_interval_ms,
        gc_inspection_budget: config.gc_inspection_budget,
    }))
}

pub(crate) fn parse_valkey_gc_interval_ms(value: &str) -> Result<Option<u64>> {
    let interval_ms = value.trim().parse::<u64>().map_err(|error| {
        anyhow::anyhow!(
            "{VALKEY_GC_INTERVAL_MS_ENV} must be an integer number of milliseconds: {error}"
        )
    })?;
    if interval_ms == 0 {
        return Ok(None);
    }
    if !(MIN_VALKEY_GC_INTERVAL_MS..=MAX_VALKEY_GC_INTERVAL_MS).contains(&interval_ms) {
        anyhow::bail!(
            "{VALKEY_GC_INTERVAL_MS_ENV} must be 0 (disabled) or in {MIN_VALKEY_GC_INTERVAL_MS}..={MAX_VALKEY_GC_INTERVAL_MS} milliseconds; got {interval_ms}"
        );
    }
    Ok(Some(interval_ms))
}

pub(crate) fn parse_valkey_gc_inspection_budget(value: &str) -> Result<u32> {
    let inspection_budget = value.trim().parse::<u32>().map_err(|error| {
        anyhow::anyhow!("{VALKEY_GC_INSPECTION_BUDGET_ENV} must be a positive integer: {error}")
    })?;
    if inspection_budget == 0 || inspection_budget > MAX_VALKEY_GC_INSPECTION_BUDGET {
        anyhow::bail!(
            "{VALKEY_GC_INSPECTION_BUDGET_ENV} must be in 1..={MAX_VALKEY_GC_INSPECTION_BUDGET}; got {inspection_budget}"
        );
    }
    Ok(inspection_budget)
}

fn mix_owner_nonce(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn valkey_gc_initial_delay_ms(interval_ms: u64, owner_nonce: u64) -> u64 {
    debug_assert!(interval_ms > 0);
    interval_ms + mix_owner_nonce(owner_nonce) % interval_ms
}

/// Build the worker-side direct Valkey writer from the deployment-wide router
/// environment without allowing a worker suffix to change the shared index.
pub(crate) fn valkey_index_namespace(
    component_namespace: &str,
    raw_namespace: Option<&str>,
    worker_suffix: Option<&str>,
) -> String {
    let raw_namespace = raw_namespace
        .map(str::trim)
        .filter(|namespace| !namespace.is_empty());
    let worker_suffix = worker_suffix
        .map(str::trim)
        .filter(|suffix| !suffix.is_empty());

    match (raw_namespace, worker_suffix) {
        (Some(namespace), Some(suffix))
            if component_namespace == format!("{namespace}-{suffix}") =>
        {
            namespace.to_string()
        }
        _ => component_namespace.to_string(),
    }
}

/// Whether workers are configured to write KV metadata directly to Valkey.
///
/// Worker startup uses the same predicate as publisher construction so it can
/// fail closed before advertising a model whose ranks were not registered.
pub fn valkey_worker_events_enabled() -> Result<bool> {
    if let Some(config) = router_valkey_json_from_env()? {
        return Ok(config.worker_events);
    }
    std::env::var(VALKEY_WORKER_EVENTS_ENV)
        .ok()
        .as_deref()
        .map(parse_valkey_worker_events_enabled)
        .transpose()
        .map(|enabled| enabled.unwrap_or(false))
}

pub(crate) fn parse_valkey_worker_events_enabled(value: &str) -> Result<bool> {
    parse_boolean_env(VALKEY_WORKER_EVENTS_ENV, value)
}

fn parse_boolean_env(name: &str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(anyhow::anyhow!("{name} must be a boolean; got {value:?}")),
    }
}

pub(crate) fn parse_valkey_required_replica_acks(value: &str) -> Result<u32> {
    let parsed = value.parse::<u32>().map_err(|error| {
        anyhow::anyhow!(
            "{VALKEY_REQUIRED_REPLICA_ACKS_ENV} must be a non-negative integer: {error}"
        )
    })?;
    if parsed > i32::MAX as u32 {
        anyhow::bail!("{VALKEY_REQUIRED_REPLICA_ACKS_ENV} exceeds the Valkey WAIT command limit");
    }
    Ok(parsed)
}

fn parse_valkey_connection_pool_size(value: &str) -> Result<u32> {
    let parsed = value.parse::<u32>().map_err(|error| {
        anyhow::anyhow!("{VALKEY_CONNECTION_POOL_SIZE_ENV} must be a positive integer: {error}")
    })?;
    if !(1..=crate::kv_router::indexer::valkey::MAX_WORKER_DIRECT_EVENT_LANES as u32)
        .contains(&parsed)
    {
        anyhow::bail!(
            "legacy worker {VALKEY_CONNECTION_POOL_SIZE_ENV} must be in 1..={}",
            crate::kv_router::indexer::valkey::MAX_WORKER_DIRECT_EVENT_LANES
        );
    }
    Ok(parsed)
}

pub fn valkey_worker_config_from_env(component: &Component) -> Result<Option<ValkeyWorkerConfig>> {
    if let Some(config) = router_valkey_json_from_env()? {
        return worker_config_from_json(component, config);
    }
    if !valkey_worker_events_enabled()? {
        return Ok(None);
    }
    let allow_insecure_plaintext = std::env::var(VALKEY_ALLOW_INSECURE_PLAINTEXT_ENV)
        .ok()
        .map(|value| parse_boolean_env(VALKEY_ALLOW_INSECURE_PLAINTEXT_ENV, &value))
        .transpose()?
        .unwrap_or(false);
    if !allow_insecure_plaintext {
        anyhow::bail!(
            "legacy plaintext Valkey worker events require {VALKEY_ALLOW_INSECURE_PLAINTEXT_ENV}=true and a separate tenant-isolated trusted network"
        );
    }
    let urls = std::env::var(VALKEY_URLS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{VALKEY_URLS_ENV} is required when {VALKEY_WORKER_EVENTS_ENV} is enabled"
            )
        })?;
    let index_scope = std::env::var(VALKEY_INDEX_SCOPE_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{VALKEY_INDEX_SCOPE_ENV} is required when {VALKEY_WORKER_EVENTS_ENV} is enabled"
            )
        })?;
    let direct_event_pool_size = std::env::var(VALKEY_CONNECTION_POOL_SIZE_ENV)
        .ok()
        .map(|value| parse_valkey_connection_pool_size(&value))
        .transpose()?
        .unwrap_or(4);
    let required_replica_acks = std::env::var(VALKEY_REQUIRED_REPLICA_ACKS_ENV)
        .ok()
        .map(|value| parse_valkey_required_replica_acks(&value))
        .transpose()?;
    let sentinel_urls = std::env::var(VALKEY_SENTINEL_URLS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty());
    let sentinel_master_name = std::env::var(VALKEY_SENTINEL_MASTER_NAME_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty());
    if sentinel_urls.is_some() != sentinel_master_name.is_some() {
        anyhow::bail!(
            "{VALKEY_SENTINEL_URLS_ENV} and {VALKEY_SENTINEL_MASTER_NAME_ENV} must be configured together"
        );
    }
    let sentinel_quorum = std::env::var(VALKEY_SENTINEL_QUORUM_ENV)
        .ok()
        .map(|value| {
            value.parse::<usize>().map_err(|error| {
                anyhow::anyhow!("{VALKEY_SENTINEL_QUORUM_ENV} must be a positive integer: {error}")
            })
        })
        .transpose()?;
    if sentinel_quorum.is_some() && sentinel_urls.is_none() {
        anyhow::bail!("{VALKEY_SENTINEL_QUORUM_ENV} requires {VALKEY_SENTINEL_URLS_ENV}");
    }
    let allow_degraded_writes = std::env::var(VALKEY_ALLOW_DEGRADED_WRITES_ENV)
        .ok()
        .map(|value| parse_boolean_env(VALKEY_ALLOW_DEGRADED_WRITES_ENV, &value))
        .transpose()?
        .unwrap_or(false);
    if allow_degraded_writes && sentinel_urls.is_none() {
        anyhow::bail!(
            "{VALKEY_ALLOW_DEGRADED_WRITES_ENV}=true requires {VALKEY_SENTINEL_URLS_ENV}"
        );
    }
    if allow_degraded_writes && required_replica_acks.is_none_or(|acks| acks == 0) {
        anyhow::bail!(
            "{VALKEY_ALLOW_DEGRADED_WRITES_ENV}=true requires a positive {VALKEY_REQUIRED_REPLICA_ACKS_ENV}"
        );
    }
    let worker_lease_ms = std::env::var(VALKEY_WORKER_LEASE_MS_ENV)
        .ok()
        .map(|value| {
            value.parse::<u64>().map_err(|error| {
                anyhow::anyhow!("{VALKEY_WORKER_LEASE_MS_ENV} must be an integer: {error}")
            })
        })
        .transpose()?
        .unwrap_or(DEFAULT_VALKEY_WORKER_LEASE_MS);
    if !(MIN_VALKEY_WORKER_LEASE_MS..=MAX_VALKEY_WORKER_LEASE_MS).contains(&worker_lease_ms) {
        anyhow::bail!(
            "{VALKEY_WORKER_LEASE_MS_ENV} must be in {MIN_VALKEY_WORKER_LEASE_MS}..={MAX_VALKEY_WORKER_LEASE_MS} milliseconds; got {worker_lease_ms}"
        );
    }
    let gc_interval_ms = std::env::var(VALKEY_GC_INTERVAL_MS_ENV)
        .ok()
        .map(|value| parse_valkey_gc_interval_ms(&value))
        .transpose()?
        .unwrap_or(Some(DEFAULT_VALKEY_GC_INTERVAL_MS));
    let gc_inspection_budget = std::env::var(VALKEY_GC_INSPECTION_BUDGET_ENV)
        .ok()
        .map(|value| parse_valkey_gc_inspection_budget(&value))
        .transpose()?
        .unwrap_or(DEFAULT_VALKEY_GC_INSPECTION_BUDGET);
    let component_namespace = component.namespace().name();
    let index_namespace = valkey_index_namespace(
        &component_namespace,
        std::env::var("DYN_NAMESPACE").ok().as_deref(),
        std::env::var("DYN_NAMESPACE_WORKER_SUFFIX").ok().as_deref(),
    );
    Ok(Some(ValkeyWorkerConfig {
        urls,
        index_scope,
        index_namespace,
        direct_event_pool_size,
        required_replica_acks,
        sentinel_urls,
        sentinel_master_name,
        sentinel_quorum,
        allow_degraded_writes,
        worker_lease_ms,
        gc_interval_ms,
        gc_inspection_budget,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_json_worker_boundary_accepts_shared_router_and_tokenizer_config() {
        let config = RouterValkeyConfig::parse(
            r#"{
                "allow_insecure_plaintext":true,
                "urls":["valkey://router:6379"],
                "index_scope":"scope-a",
                "worker_events":true,
                "worker_lease_ms":45000,
                "gc_interval_ms":0,
                "gc_inspection_budget":512,
                "sentinel":{"urls":["s0:26379","s1:26379","s2:26379"],"master_name":"router","quorum":2},
                "tokenizer_cache":{"enabled":true,"sentinel_master_name":"tokenizer"}
            }"#,
        )
        .unwrap();

        assert!(config.worker_events);
        assert_eq!(config.worker_lease_ms, 45_000);
        assert_eq!(config.gc_interval_ms, 0);
        assert_eq!(config.gc_inspection_budget, 512);
        assert_eq!(config.sentinel.unwrap().quorum, Some(2));
    }

    #[test]
    fn single_json_worker_boundary_rejects_unknown_duplicate_and_oversized_input() {
        assert!(RouterValkeyConfig::parse(r#"{"worker_events":true,"unknown":1}"#).is_err());
        assert!(RouterValkeyConfig::parse(r#"{"tokenizer_cache":{"unknown":1}}"#).is_err());
        assert!(
            RouterValkeyConfig::parse(r#"{"worker_events":true,"worker_events":false}"#).is_err()
        );
        assert!(RouterValkeyConfig::parse(&" ".repeat(64 * 1024 + 1)).is_err());
    }

    #[test]
    fn legacy_worker_boundary_bounds_connection_pool_allocation() {
        assert_eq!(parse_valkey_connection_pool_size("16").unwrap(), 16);
        for value in ["0", "17", "64", "4294967295"] {
            assert!(parse_valkey_connection_pool_size(value).is_err());
        }
    }
}
