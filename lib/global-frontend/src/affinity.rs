// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Construction and lifecycle of the configured session-affinity backend.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use aws_config::meta::region::RegionProviderChain;
use aws_sdk_dynamodb::config::{Region, retry::RetryConfig, timeout::TimeoutConfig};
use clap::{Args, ValueEnum};
use dynamo_global_router::affinity::{
    AffinityConfig, AffinityStats, AffinityStore, BreakerConfig, CacheStats, CachedAffinityStore,
    InMemoryAffinityStore, SessionAffinity, StoreBackedSessionAffinity,
};
use dynamo_global_router_store_dynamodb::DynamoDbAffinityStore;
use tokio_util::sync::CancellationToken;
use url::Url;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum Store {
    #[default]
    Memory,
    Dynamodb,
}

#[derive(Args)]
pub(crate) struct AffinityArgs {
    /// Persistence backend for --router-policy affinity.
    #[arg(
        long = "affinity-store",
        env = "DYN_GLOBAL_AFFINITY_STORE",
        value_enum,
        default_value = "memory"
    )]
    store: Store,
    /// Shared deployment/tenant scope; required for DynamoDB. Never derived from client headers.
    #[arg(
        long = "affinity-scope",
        env = "DYN_GLOBAL_AFFINITY_SCOPE",
        required_if_eq("store", "dynamodb")
    )]
    scope: Option<String>,
    /// Stable namespace for compatible affinity policies, shared across router replicas.
    #[arg(
        long = "affinity-policy",
        env = "DYN_GLOBAL_AFFINITY_POLICY",
        default_value = "headroom-v1"
    )]
    policy: String,
    /// Total affinity lookup/write/retry budget before routing without persistence.
    #[arg(long = "affinity-timeout-ms", env = "DYN_GLOBAL_AFFINITY_TIMEOUT_MS", default_value_t = 20, value_parser = clap::value_parser!(u64).range(1..))]
    timeout_ms: u64,
    /// How long a looked-up binding is reused from memory before the store is
    /// read again. 0 reads the store on every request.
    #[arg(
        long = "affinity-cache-ttl-ms",
        env = "DYN_GLOBAL_AFFINITY_CACHE_TTL_MS",
        default_value_t = 1000
    )]
    cache_ttl_ms: u64,
    /// Sessions held in the read-through cache. Further sessions read the store.
    #[arg(long = "affinity-cache-entries", env = "DYN_GLOBAL_AFFINITY_CACHE_ENTRIES", default_value_t = 100_000, value_parser = clap::value_parser!(usize))]
    cache_entries: usize,
    /// Consecutive storage failures before affinity is taken out of the request path.
    #[arg(long = "affinity-breaker-failures", env = "DYN_GLOBAL_AFFINITY_BREAKER_FAILURES", default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..))]
    breaker_failures: u32,
    /// How long to route without affinity before one request retests the store.
    #[arg(long = "affinity-breaker-cooldown-ms", env = "DYN_GLOBAL_AFFINITY_BREAKER_COOLDOWN_MS", default_value_t = 5_000, value_parser = clap::value_parser!(u64).range(1..))]
    breaker_cooldown_ms: u64,
    /// Existing table with string itemHashKey and sortKey keys; TTL attribute expiresAt.
    #[arg(
        long = "affinity-ddb-table",
        env = "DYN_GLOBAL_AFFINITY_DDB_TABLE",
        required_if_eq("store", "dynamodb")
    )]
    table: Option<String>,
    /// AWS region override. Otherwise use the standard AWS region provider chain.
    #[arg(long = "affinity-ddb-region", env = "DYN_GLOBAL_AFFINITY_DDB_REGION")]
    ddb_region: Option<String>,
    /// Optional endpoint override, for example http://localhost:8000 for DynamoDB Local.
    #[arg(
        long = "affinity-ddb-endpoint",
        env = "DYN_GLOBAL_AFFINITY_DDB_ENDPOINT"
    )]
    endpoint: Option<Url>,
    /// Strong reads within the contacted region; does not provide global consistency on MREC.
    #[arg(long = "affinity-consistent-reads", env = "DYN_GLOBAL_AFFINITY_CONSISTENT_READS", default_value_t = true, action = clap::ArgAction::Set)]
    consistent_reads: bool,
}

impl AffinityArgs {
    pub(crate) fn uses_dynamodb(&self) -> bool {
        self.store == Store::Dynamodb
    }

    /// The configured affinity mechanism, its selector configuration, and the
    /// counters the metrics endpoint exports.
    pub(crate) async fn build(
        &self,
        idle: Duration,
        shutdown: CancellationToken,
    ) -> Result<(Arc<dyn SessionAffinity>, AffinityConfig, AffinityMetrics)> {
        let scope = self.scope.as_deref().unwrap_or("default");
        if scope.trim().is_empty() || self.policy.trim().is_empty() {
            bail!("--affinity-scope and --affinity-policy must be nonempty");
        }
        let config = AffinityConfig {
            scope: scope.into(),
            policy: self.policy.clone(),
            timeout: Duration::from_millis(self.timeout_ms),
            breaker: BreakerConfig {
                failures: self.breaker_failures,
                cooldown: Duration::from_millis(self.breaker_cooldown_ms),
            },
        };
        let metrics = AffinityMetrics::default();
        let store: Arc<dyn AffinityStore> = match self.store {
            Store::Memory => {
                let store = Arc::new(InMemoryAffinityStore::default());
                tokio::spawn(sweep_pins(store.clone(), shutdown));
                store
            }
            Store::Dynamodb => {
                // Also enforce this here for construction outside the CLI parser.
                self.scope
                    .as_ref()
                    .context("--affinity-scope is required for DynamoDB")?;
                let table = self
                    .table
                    .clone()
                    .context("--affinity-ddb-table is required")?;
                let region =
                    RegionProviderChain::first_try(self.ddb_region.clone().map(Region::new))
                        .or_default_provider();
                let sdk = aws_config::defaults(aws_config::BehaviorVersion::latest())
                    .region(region)
                    .load()
                    .await;
                if sdk.region().is_none() {
                    bail!("configure an AWS region or pass --affinity-ddb-region");
                }
                let mut client = aws_sdk_dynamodb::config::Builder::from(&sdk)
                    .retry_config(RetryConfig::standard().with_max_attempts(2))
                    .timeout_config(
                        TimeoutConfig::builder()
                            .operation_timeout(config.timeout)
                            .operation_attempt_timeout(config.timeout)
                            .build(),
                    );
                if let Some(endpoint) = &self.endpoint {
                    if !matches!(endpoint.scheme(), "http" | "https")
                        || endpoint.host_str().is_none()
                    {
                        bail!("--affinity-ddb-endpoint must be an HTTP(S) URL");
                    }
                    client = client.endpoint_url(endpoint.as_str());
                }
                let store: Arc<dyn AffinityStore> = Arc::new(DynamoDbAffinityStore::new(
                    aws_sdk_dynamodb::Client::from_conf(client.build()),
                    table,
                    self.consistent_reads,
                )?);
                // The memory store is already memory, so only a remote store
                // is worth a cache. A hit skips the network round trip
                // entirely; a stale entry costs one conditional-write
                // conflict, which the selector's retry reads through.
                match self.cache_ttl_ms {
                    0 => store,
                    ttl_ms => {
                        let cache = Arc::new(CachedAffinityStore::with_stats(
                            store,
                            Duration::from_millis(ttl_ms),
                            self.cache_entries,
                            metrics.cache.clone(),
                        )?);
                        tokio::spawn(sweep_cache(cache.clone(), shutdown));
                        cache
                    }
                }
            }
        };
        Ok((
            Arc::new(StoreBackedSessionAffinity::with_stats(
                store,
                idle,
                metrics.affinity.clone(),
            )?),
            config,
            metrics,
        ))
    }
}

/// Counters the affinity path fills and the metrics endpoint exports.
#[derive(Clone, Debug, Default)]
pub(crate) struct AffinityMetrics {
    pub(crate) affinity: Arc<AffinityStats>,
    pub(crate) cache: Arc<CacheStats>,
}

async fn sweep_cache(cache: Arc<CachedAffinityStore>, cancel: CancellationToken) {
    let mut ticks = tokio::time::interval(Duration::from_secs(60));
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = ticks.tick() => cache.sweep(),
        }
    }
}

async fn sweep_pins(store: Arc<InMemoryAffinityStore>, cancel: CancellationToken) {
    let mut ticks = tokio::time::interval(Duration::from_secs(60));
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = ticks.tick() => {
                if let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) {
                    store.sweep(now.as_secs());
                }
            }
        }
    }
}
