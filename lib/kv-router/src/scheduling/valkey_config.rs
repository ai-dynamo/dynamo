// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Focused configuration and invariants for the Valkey-backed router.

use serde::{Deserialize, Serialize};
use validator::ValidationError;

const fn default_connection_pool_size() -> u32 {
    4
}

const fn default_admission_lease_ms() -> u64 {
    30_000
}

const MAX_CONNECTION_POOL_SIZE: u32 = 64;
const MIN_ADMISSION_LEASE_MS: u64 = 10_000;
const MAX_ADMISSION_LEASE_MS: u64 = 600_000;

/// Valkey's WAIT command accepts a non-negative signed `int` replica count.
pub const MAX_REQUIRED_REPLICA_ACKS: u32 = i32::MAX as u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValkeyRouterConfig {
    #[serde(default, rename = "valkey_urls")]
    pub urls: Option<String>,
    #[serde(default, rename = "valkey_index_scope")]
    pub index_scope: Option<String>,
    #[serde(
        default = "default_connection_pool_size",
        rename = "valkey_connection_pool_size"
    )]
    pub connection_pool_size: u32,
    #[serde(default, rename = "valkey_required_replica_acks")]
    pub required_replica_acks: Option<u32>,
    #[serde(default, rename = "valkey_sentinel_urls")]
    pub sentinel_urls: Option<String>,
    #[serde(default, rename = "valkey_sentinel_master_name")]
    pub sentinel_master_name: Option<String>,
    #[serde(default, rename = "valkey_sentinel_quorum")]
    pub sentinel_quorum: Option<u32>,
    #[serde(default, rename = "valkey_allow_insecure_plaintext")]
    pub allow_insecure_plaintext: bool,
    #[serde(default, rename = "valkey_allow_degraded_writes")]
    pub allow_degraded_writes: bool,
    #[serde(default, rename = "valkey_worker_events")]
    pub worker_events: bool,
    #[serde(default, rename = "valkey_authoritative_admission")]
    pub authoritative_admission: bool,
    #[serde(
        default = "default_admission_lease_ms",
        rename = "valkey_admission_lease_ms"
    )]
    pub admission_lease_ms: u64,
}

impl Default for ValkeyRouterConfig {
    fn default() -> Self {
        Self {
            urls: None,
            index_scope: None,
            connection_pool_size: default_connection_pool_size(),
            required_replica_acks: None,
            sentinel_urls: None,
            sentinel_master_name: None,
            sentinel_quorum: None,
            allow_insecure_plaintext: false,
            allow_degraded_writes: false,
            worker_events: false,
            authoritative_admission: false,
            admission_lease_ms: default_admission_lease_ms(),
        }
    }
}

pub(crate) struct ValkeyValidationContext {
    pub use_remote_indexer: bool,
    pub serve_indexer: bool,
    pub use_kv_events: bool,
    pub durable_kv_events: bool,
    pub overlap_enabled: bool,
    pub router_replica_sync: bool,
    pub immediate_admission: bool,
    pub frontend_load_tracking: bool,
    pub frontend_scoring: bool,
}

impl ValkeyRouterConfig {
    pub(crate) fn validate(&self, context: ValkeyValidationContext) -> Result<(), ValidationError> {
        self.validate_topology(&context)?;
        self.validate_sentinel()?;
        self.validate_admission(&context)
    }

    fn validate_topology(&self, context: &ValkeyValidationContext) -> Result<(), ValidationError> {
        if !(1..=MAX_CONNECTION_POOL_SIZE).contains(&self.connection_pool_size) {
            return Err(ValidationError::new(
                "valkey_connection_pool_size must be in 1..=64",
            ));
        }
        if !(MIN_ADMISSION_LEASE_MS..=MAX_ADMISSION_LEASE_MS).contains(&self.admission_lease_ms) {
            return Err(ValidationError::new(
                "valkey_admission_lease_ms must be in 10000..=600000",
            ));
        }
        if let Some(urls) = self.urls.as_deref() {
            if urls.trim().is_empty() {
                return Err(ValidationError::new(
                    "valkey_urls must contain at least one endpoint",
                ));
            }
            if context.use_remote_indexer || context.serve_indexer {
                return Err(ValidationError::new(
                    "valkey_urls is mutually exclusive with use_remote_indexer and serve_indexer",
                ));
            }
            if !context.use_kv_events {
                return Err(ValidationError::new(
                    "valkey_urls requires use_kv_events=true",
                ));
            }
            if context.durable_kv_events {
                return Err(ValidationError::new(
                    "valkey_urls does not support durable_kv_events; Valkey provides persistent router metadata",
                ));
            }
            if !self.allow_insecure_plaintext {
                return Err(ValidationError::new(
                    "valkey_urls requires valkey_allow_insecure_plaintext=true and a separate tenant-isolated trusted network",
                ));
            }
        }
        if self
            .index_scope
            .as_deref()
            .is_some_and(|scope| scope.trim().is_empty())
        {
            return Err(ValidationError::new(
                "valkey_index_scope must not be empty when configured",
            ));
        }
        if self.required_replica_acks.is_some() && self.urls.is_none() {
            return Err(ValidationError::new(
                "valkey_required_replica_acks requires valkey_urls",
            ));
        }
        if self
            .required_replica_acks
            .is_some_and(|acks| acks > MAX_REQUIRED_REPLICA_ACKS)
        {
            return Err(ValidationError::new(
                "valkey_required_replica_acks exceeds the Valkey WAIT command limit",
            ));
        }
        if self.worker_events && self.urls.is_none() {
            return Err(ValidationError::new(
                "valkey_worker_events requires valkey_urls",
            ));
        }
        if self.worker_events && self.index_scope.is_none() {
            return Err(ValidationError::new(
                "valkey_worker_events requires valkey_index_scope so workers and frontends use one index key",
            ));
        }
        Ok(())
    }

    fn validate_sentinel(&self) -> Result<(), ValidationError> {
        match (
            self.sentinel_urls.as_deref(),
            self.sentinel_master_name.as_deref(),
        ) {
            (None, None) => {}
            (Some(urls), Some(master_name)) => {
                if self.urls.is_none() {
                    return Err(ValidationError::new(
                        "valkey_sentinel_urls requires valkey_urls",
                    ));
                }
                let witness_count = witness_count(urls);
                if witness_count == 0 {
                    return Err(ValidationError::new(
                        "valkey_sentinel_urls must contain at least one endpoint",
                    ));
                }
                if master_name.trim().is_empty() {
                    return Err(ValidationError::new(
                        "valkey_sentinel_master_name must not be empty",
                    ));
                }
                if let Some(quorum) = self.sentinel_quorum {
                    let quorum = quorum as usize;
                    if quorum <= witness_count / 2 || quorum > witness_count {
                        return Err(ValidationError::new(
                            "valkey_sentinel_quorum must be a strict majority of configured Sentinel endpoints",
                        ));
                    }
                }
            }
            _ => {
                return Err(ValidationError::new(
                    "valkey_sentinel_urls and valkey_sentinel_master_name must be configured together",
                ));
            }
        }
        if self.sentinel_quorum.is_some() && self.sentinel_urls.is_none() {
            return Err(ValidationError::new(
                "valkey_sentinel_quorum requires valkey_sentinel_urls",
            ));
        }
        if self.allow_degraded_writes {
            let Some(sentinel_urls) = self.sentinel_urls.as_deref() else {
                return Err(ValidationError::new(
                    "valkey_allow_degraded_writes requires Sentinel primary discovery",
                ));
            };
            if witness_count(sentinel_urls) < 3 {
                return Err(ValidationError::new(
                    "valkey_allow_degraded_writes requires at least three distinct Sentinel witnesses",
                ));
            }
            if self.required_replica_acks.is_none_or(|acks| acks == 0) {
                return Err(ValidationError::new(
                    "valkey_allow_degraded_writes requires an explicit positive valkey_required_replica_acks",
                ));
            }
        }
        Ok(())
    }

    fn validate_admission(&self, context: &ValkeyValidationContext) -> Result<(), ValidationError> {
        if !self.authoritative_admission {
            return Ok(());
        }
        if !self.worker_events {
            return Err(ValidationError::new(
                "valkey_authoritative_admission requires valkey_worker_events so the module has fleet-wide KV ownership",
            ));
        }
        if !context.overlap_enabled {
            return Err(ValidationError::new(
                "valkey_authoritative_admission requires overlap_score_credit > 0 so the Valkey indexer is enabled",
            ));
        }
        if context.router_replica_sync {
            return Err(ValidationError::new(
                "valkey_authoritative_admission is mutually exclusive with router_replica_sync; the module owns active admission",
            ));
        }
        if !context.immediate_admission {
            return Err(ValidationError::new(
                "valkey_authoritative_admission currently supports immediate admission only; disable router queues and policy config",
            ));
        }
        if context.frontend_load_tracking {
            return Err(ValidationError::new(
                "valkey_authoritative_admission does not support frontend-local output, prefill, or predicted-load accounting",
            ));
        }
        if context.frontend_scoring {
            return Err(ValidationError::new(
                "valkey_authoritative_admission does not support frontend-local cache credits, overlap decay, or temperature",
            ));
        }
        Ok(())
    }
}

fn witness_count(urls: &str) -> usize {
    urls.split(',')
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .count()
}
