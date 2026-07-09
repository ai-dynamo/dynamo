// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical JSON contract shared by frontend and worker Valkey consumers.

use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};
use dynamo_kv_router::config::{KvRouterConfig, ValkeyRouterConfig};
use serde::{Deserialize, Serialize};

use crate::tokenizer_cache::{TokenizerCacheConfig, TokenizerCacheL2Config};
use crate::valkey_transport::{ValkeySentinelConfig, parse_endpoint};

pub const MAX_ROUTER_VALKEY_CONFIG_BYTES: usize = 64 * 1024;
pub const MAX_ROUTER_VALKEY_ENDPOINTS: usize = 64;
pub const MIN_WORKER_LEASE_MS: u64 = 10_000;
pub const MAX_WORKER_LEASE_MS: u64 = 600_000;
pub const MIN_GC_INTERVAL_MS: u64 = 1_000;
pub const MAX_GC_INTERVAL_MS: u64 = 86_400_000;
pub const MAX_GC_INSPECTION_BUDGET: u32 = 1_048_576;

const fn default_router_pool_size() -> u32 {
    4
}

const fn default_admission_lease_ms() -> u64 {
    30_000
}

const fn default_worker_lease_ms() -> u64 {
    30_000
}

const fn default_gc_interval_ms() -> u64 {
    60_000
}

const fn default_gc_inspection_budget() -> u32 {
    256
}

const fn default_tokenizer_l1_bytes() -> u64 {
    64 * 1024 * 1024
}

const fn default_tokenizer_ttl_seconds() -> u64 {
    3_600
}

const fn default_tokenizer_timeout_ms() -> u64 {
    20
}

const fn default_tokenizer_pool_size() -> usize {
    8
}

const fn default_tokenizer_pending_writes() -> usize {
    128
}

fn default_tokenizer_scope() -> String {
    "default".to_string()
}

fn default_tokenizer_key_prefix() -> String {
    "dynamo:tokenizer:v1".to_string()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterValkeySentinelConfig {
    pub urls: Vec<String>,
    pub master_name: String,
    pub quorum: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouterTokenizerCacheConfig {
    pub enabled: bool,
    pub url: Option<String>,
    pub sentinel_master_name: Option<String>,
    pub scope: String,
    pub key_prefix: String,
    pub ttl_seconds: u64,
    pub timeout_ms: u64,
    pub connection_pool_size: usize,
    pub max_pending_writes: usize,
    pub l1_bytes: u64,
    pub extend: bool,
}

impl Default for RouterTokenizerCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            url: None,
            sentinel_master_name: None,
            scope: default_tokenizer_scope(),
            key_prefix: default_tokenizer_key_prefix(),
            ttl_seconds: default_tokenizer_ttl_seconds(),
            timeout_ms: default_tokenizer_timeout_ms(),
            connection_pool_size: default_tokenizer_pool_size(),
            max_pending_writes: default_tokenizer_pending_writes(),
            l1_bytes: default_tokenizer_l1_bytes(),
            extend: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouterValkeyConfig {
    pub urls: Vec<String>,
    pub index_scope: Option<String>,
    pub connection_pool_size: u32,
    pub required_replica_acks: Option<u32>,
    pub sentinel: Option<RouterValkeySentinelConfig>,
    pub allow_insecure_plaintext: bool,
    pub allow_degraded_writes: bool,
    pub worker_events: bool,
    pub authoritative_admission: bool,
    pub admission_lease_ms: u64,
    pub worker_lease_ms: u64,
    pub gc_interval_ms: u64,
    pub gc_inspection_budget: u32,
    pub tokenizer_cache: RouterTokenizerCacheConfig,
}

impl Default for RouterValkeyConfig {
    fn default() -> Self {
        Self {
            urls: Vec::new(),
            index_scope: None,
            connection_pool_size: default_router_pool_size(),
            required_replica_acks: None,
            sentinel: None,
            allow_insecure_plaintext: false,
            allow_degraded_writes: false,
            worker_events: false,
            authoritative_admission: false,
            admission_lease_ms: default_admission_lease_ms(),
            worker_lease_ms: default_worker_lease_ms(),
            gc_interval_ms: default_gc_interval_ms(),
            gc_inspection_budget: default_gc_inspection_budget(),
            tokenizer_cache: RouterTokenizerCacheConfig::default(),
        }
    }
}

impl RouterValkeyConfig {
    pub fn parse(raw: &str) -> Result<Self> {
        if raw.len() > MAX_ROUTER_VALKEY_CONFIG_BYTES {
            bail!("router Valkey config must be at most {MAX_ROUTER_VALKEY_CONFIG_BYTES} bytes");
        }
        if !raw.trim_start().starts_with('{') {
            bail!("router Valkey config must be a JSON object");
        }
        let mut config: Self = serde_json::from_str(raw)
            .map_err(|error| anyhow::anyhow!("router Valkey config is invalid: {error}"))?;
        config.validate_and_normalize()?;
        Ok(config)
    }

    pub fn normalized_json(raw: &str) -> Result<String> {
        serde_json::to_string(&Self::parse(raw)?).context("serialize normalized Valkey config")
    }

    /// Apply the canonical JSON routing section to the runtime router config.
    /// Tokenizer-only fields remain owned by [`Self::tokenizer_cache_config`].
    pub fn apply_to_kv_router(&self, router: &mut KvRouterConfig) {
        let sentinel = self.sentinel.as_ref().filter(|_| !self.urls.is_empty());
        router.valkey = ValkeyRouterConfig {
            urls: (!self.urls.is_empty()).then(|| self.urls.join(",")),
            index_scope: self.index_scope.clone(),
            connection_pool_size: self.connection_pool_size,
            required_replica_acks: self.required_replica_acks,
            sentinel_urls: sentinel.map(|config| config.urls.join(",")),
            sentinel_master_name: sentinel.map(|config| config.master_name.clone()),
            sentinel_quorum: sentinel
                .and_then(|config| config.quorum)
                .map(|quorum| quorum as u32),
            allow_insecure_plaintext: self.allow_insecure_plaintext,
            allow_degraded_writes: self.allow_degraded_writes,
            worker_events: self.worker_events,
            authoritative_admission: self.authoritative_admission,
            admission_lease_ms: self.admission_lease_ms,
        };
        if self.authoritative_admission {
            router.apply_valkey_authoritative_admission_preset();
        }
    }

    fn validate_and_normalize(&mut self) -> Result<()> {
        validate_endpoints(&self.urls, "urls")?;
        if let Some(scope) = self.index_scope.as_deref() {
            validate_key_segment(scope, "index_scope")?;
        }
        if !(1..=64).contains(&self.connection_pool_size) {
            bail!("connection_pool_size must be in 1..=64");
        }
        if self
            .required_replica_acks
            .is_some_and(|acks| acks > i32::MAX as u32)
        {
            bail!("required_replica_acks exceeds the Valkey WAIT command limit");
        }
        if !(10_000..=600_000).contains(&self.admission_lease_ms) {
            bail!("admission_lease_ms must be in 10000..=600000");
        }
        if !(MIN_WORKER_LEASE_MS..=MAX_WORKER_LEASE_MS).contains(&self.worker_lease_ms) {
            bail!("worker_lease_ms must be in {MIN_WORKER_LEASE_MS}..={MAX_WORKER_LEASE_MS}");
        }
        if self.gc_interval_ms != 0
            && !(MIN_GC_INTERVAL_MS..=MAX_GC_INTERVAL_MS).contains(&self.gc_interval_ms)
        {
            bail!("gc_interval_ms must be 0 or in {MIN_GC_INTERVAL_MS}..={MAX_GC_INTERVAL_MS}");
        }
        if !(1..=MAX_GC_INSPECTION_BUDGET).contains(&self.gc_inspection_budget) {
            bail!("gc_inspection_budget must be in 1..={MAX_GC_INSPECTION_BUDGET}");
        }
        if self.worker_events && self.urls.is_empty() {
            bail!("worker_events requires urls");
        }
        if self.worker_events && self.index_scope.is_none() {
            bail!("worker_events requires index_scope");
        }

        if let Some(sentinel) = &mut self.sentinel {
            validate_master_name(&sentinel.master_name, "sentinel.master_name")?;
            let urls = sentinel.urls.join(",");
            let validated =
                ValkeySentinelConfig::new(&urls, &sentinel.master_name, sentinel.quorum)?;
            sentinel.quorum = Some(validated.quorum);
            if self.allow_degraded_writes {
                validated.validate_degraded_writes()?;
            }
        } else if self.allow_degraded_writes {
            bail!("allow_degraded_writes requires sentinel");
        }
        if self.allow_degraded_writes && self.required_replica_acks.is_none_or(|acks| acks == 0) {
            bail!("allow_degraded_writes requires positive required_replica_acks");
        }
        self.tokenizer_cache_config()?;
        let uses_network_valkey = !self.urls.is_empty()
            || self.sentinel.is_some()
            || self.tokenizer_cache.url.is_some()
            || self.tokenizer_cache.sentinel_master_name.is_some();
        if uses_network_valkey && !self.allow_insecure_plaintext {
            bail!(
                "plaintext Valkey transport requires allow_insecure_plaintext=true and a separate tenant-isolated trusted network; TLS and ACL credentials are not yet supported"
            );
        }
        Ok(())
    }

    pub fn tokenizer_cache_config(&self) -> Result<TokenizerCacheConfig> {
        let tokenizer = &self.tokenizer_cache;
        validate_key_segment(&tokenizer.scope, "tokenizer_cache.scope")?;
        validate_key_segment(&tokenizer.key_prefix, "tokenizer_cache.key_prefix")?;
        if !(1..=604_800).contains(&tokenizer.ttl_seconds) {
            bail!("tokenizer_cache.ttl_seconds must be in 1..=604800");
        }
        if !(1..=10_000).contains(&tokenizer.timeout_ms) {
            bail!("tokenizer_cache.timeout_ms must be in 1..=10000");
        }
        if !(1..=64).contains(&tokenizer.connection_pool_size) {
            bail!("tokenizer_cache.connection_pool_size must be in 1..=64");
        }
        if !(1..=4096).contains(&tokenizer.max_pending_writes) {
            bail!("tokenizer_cache.max_pending_writes must be in 1..=4096");
        }
        if let Some(url) = tokenizer.url.as_deref() {
            parse_endpoint(url).context("invalid tokenizer_cache.url")?;
        }
        if tokenizer.url.is_some() && tokenizer.sentinel_master_name.is_some() {
            bail!("tokenizer_cache.url and sentinel_master_name are mutually exclusive");
        }
        if tokenizer.sentinel_master_name.is_some() && self.sentinel.is_none() {
            bail!("tokenizer_cache.sentinel_master_name requires top-level sentinel");
        }
        if let Some(master_name) = tokenizer.sentinel_master_name.as_deref() {
            validate_master_name(master_name, "tokenizer_cache.sentinel_master_name")?;
        }

        let l2 = if !tokenizer.enabled {
            None
        } else {
            match (
                tokenizer.url.as_deref(),
                tokenizer.sentinel_master_name.as_deref(),
            ) {
                (Some(_), Some(_)) => unreachable!("validated mutually exclusive backends"),
                (None, Some(_)) if self.sentinel.is_none() => {
                    unreachable!("validated Sentinel dependency")
                }
                (None, None) => None,
                (url, sentinel_master_name) => Some(TokenizerCacheL2Config {
                    allow_insecure_plaintext: self.allow_insecure_plaintext,
                    url: url.map(str::to_string),
                    sentinel_urls: self
                        .sentinel
                        .as_ref()
                        .map(|sentinel| sentinel.urls.join(",")),
                    sentinel_master_name: sentinel_master_name.map(str::to_string),
                    sentinel_quorum: self.sentinel.as_ref().and_then(|sentinel| sentinel.quorum),
                    scope: tokenizer.scope.clone(),
                    key_prefix: tokenizer.key_prefix.clone(),
                    ttl_seconds: tokenizer.ttl_seconds,
                    timeout_ms: tokenizer.timeout_ms,
                    connection_pool_size: tokenizer.connection_pool_size,
                    max_pending_writes: tokenizer.max_pending_writes,
                }),
            }
        };
        let config = TokenizerCacheConfig {
            enabled: tokenizer.enabled,
            l1_bytes: tokenizer.l1_bytes,
            extend: tokenizer.extend,
            l2,
        };
        config.validate()?;
        Ok(config)
    }
}

fn validate_endpoints(urls: &[String], field: &str) -> Result<()> {
    if urls.len() > MAX_ROUTER_VALKEY_ENDPOINTS {
        bail!("{field} must contain at most {MAX_ROUTER_VALKEY_ENDPOINTS} endpoints");
    }
    let mut unique = BTreeSet::new();
    for url in urls {
        let endpoint = parse_endpoint(url)?;
        if !unique.insert(endpoint.clone()) {
            bail!("{field} endpoints must be distinct; duplicate {endpoint:?}");
        }
    }
    Ok(())
}

fn validate_key_segment(value: &str, field: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b':' | b'-' | b'_' | b'.'))
    {
        bail!("{field} must contain 1..=128 ASCII letters, digits, ':', '-', '_' or '.'");
    }
    Ok(())
}

fn validate_master_name(value: &str, field: &str) -> Result<()> {
    if value.is_empty() || value.len() > 128 || value.bytes().any(|byte| byte.is_ascii_whitespace())
    {
        bail!("{field} must be one non-empty token of at most 128 bytes");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_contract_rejects_unknown_duplicate_and_oversized_input() {
        assert!(RouterValkeyConfig::parse("[]").is_err());
        assert!(RouterValkeyConfig::parse(r#"{"unknown":1}"#).is_err());
        assert!(
            RouterValkeyConfig::parse(r#"{"worker_events":true,"worker_events":false}"#).is_err()
        );
        assert!(RouterValkeyConfig::parse(&" ".repeat(64 * 1024 + 1)).is_err());
    }

    #[test]
    fn plaintext_valkey_requires_explicit_network_isolation_opt_in() {
        let error = RouterValkeyConfig::parse(r#"{"urls":["valkey://router:6379"]}"#).unwrap_err();
        assert!(error.to_string().contains("allow_insecure_plaintext"));

        RouterValkeyConfig::parse(
            r#"{"urls":["valkey://router:6379"],"allow_insecure_plaintext":true}"#,
        )
        .unwrap();

        let tokenizer_error =
            RouterValkeyConfig::parse(r#"{"tokenizer_cache":{"url":"valkey://tokenizer:6379"}}"#)
                .unwrap_err();
        assert!(
            tokenizer_error
                .to_string()
                .contains("allow_insecure_plaintext")
        );
    }

    #[test]
    fn one_contract_builds_tokenizer_sentinel_config() {
        let config = RouterValkeyConfig::parse(
            r#"{
                "allow_insecure_plaintext":true,
                "sentinel":{"urls":["s0:26379","s1:26379","s2:26379"],"master_name":"router","quorum":2},
                "tokenizer_cache":{"sentinel_master_name":"tokenizer"}
            }"#,
        )
        .unwrap();
        let tokenizer = config.tokenizer_cache_config().unwrap();
        let l2 = tokenizer.l2.unwrap();
        assert_eq!(l2.sentinel_master_name.as_deref(), Some("tokenizer"));
        assert_eq!(l2.sentinel_quorum, Some(2));
    }

    #[test]
    fn one_contract_bounds_sentinel_endpoint_count() {
        let urls = (0..17)
            .map(|index| format!("127.0.0.1:{}", 20_000 + index))
            .collect::<Vec<_>>();
        let raw = serde_json::json!({
            "allow_insecure_plaintext": true,
            "sentinel": {
                "urls": urls,
                "master_name": "router"
            }
        })
        .to_string();

        let error = RouterValkeyConfig::parse(&raw).unwrap_err();

        assert!(error.to_string().contains("at most 16"));
    }

    #[test]
    fn one_contract_applies_routing_and_authoritative_preset() {
        let contract = RouterValkeyConfig::parse(
            r#"{
                "allow_insecure_plaintext":true,
                "urls":["router-primary:6379","router-replica:6379"],
                "index_scope":"shared-router",
                "connection_pool_size":12,
                "required_replica_acks":1,
                "sentinel":{"urls":["s0:26379","s1:26379","s2:26379"],"master_name":"router","quorum":2},
                "worker_events":true,
                "authoritative_admission":true,
                "admission_lease_ms":45000
            }"#,
        )
        .unwrap();
        let mut router = dynamo_kv_router::config::KvRouterConfig::default();

        contract.apply_to_kv_router(&mut router);

        assert_eq!(
            router.valkey.urls.as_deref(),
            Some("router-primary:6379,router-replica:6379")
        );
        assert_eq!(router.valkey.index_scope.as_deref(), Some("shared-router"));
        assert_eq!(router.valkey.connection_pool_size, 12);
        assert_eq!(router.valkey.required_replica_acks, Some(1));
        assert_eq!(
            router.valkey.sentinel_urls.as_deref(),
            Some("s0:26379,s1:26379,s2:26379")
        );
        assert_eq!(
            router.valkey.sentinel_master_name.as_deref(),
            Some("router")
        );
        assert_eq!(router.valkey.sentinel_quorum, Some(2));
        assert!(router.valkey.worker_events);
        assert!(router.valkey.authoritative_admission);
        assert_eq!(router.valkey.admission_lease_ms, 45_000);
        assert_eq!(router.router_queue_threshold, None);
        assert!(!router.router_track_prefill_tokens);
        assert_eq!(router.host_cache_hit_weight, 0.0);
        assert_eq!(router.disk_cache_hit_weight, 0.0);
    }
}
