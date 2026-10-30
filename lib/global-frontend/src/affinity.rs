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
    AffinityConfig, AffinityStore, InMemoryAffinityStore, SessionAffinity,
    StoreBackedSessionAffinity,
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
    #[arg(long = "affinity-timeout-ms", env = "DYN_GLOBAL_AFFINITY_TIMEOUT_MS", default_value_t = 100, value_parser = clap::value_parser!(u64).range(1..))]
    timeout_ms: u64,
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

    pub(crate) async fn build(
        &self,
        idle: Duration,
        shutdown: CancellationToken,
    ) -> Result<(Arc<dyn SessionAffinity>, AffinityConfig)> {
        let scope = self.scope.as_deref().unwrap_or("default");
        if scope.trim().is_empty() || self.policy.trim().is_empty() {
            bail!("--affinity-scope and --affinity-policy must be nonempty");
        }
        let config = AffinityConfig {
            scope: scope.into(),
            policy: self.policy.clone(),
            timeout: Duration::from_millis(self.timeout_ms),
        };
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
                Arc::new(DynamoDbAffinityStore::new(
                    aws_sdk_dynamodb::Client::from_conf(client.build()),
                    table,
                    self.consistent_reads,
                )?)
            }
        };
        Ok((
            Arc::new(StoreBackedSessionAffinity::new(store, idle)?),
            config,
        ))
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
