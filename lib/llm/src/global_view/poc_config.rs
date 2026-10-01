// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! File-backed configuration for the first regional Global Router POC.
//! Pool IDs are derived by Global View from configured site, namespace, and DGD.

use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use dynamo_kv_router::global_view::state::{FreshnessPolicy, PoolLocation};
use dynamo_kv_router::global_view::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};
use serde::Deserialize;
use tonic::transport::{Channel, Endpoint};

use super::RelayPoolScope;
use super::http_forward::parse_private_frontend_base;
use super::scheduler_metrics::{MetricsConfig, MetricsSource, SchedulerMetrics};
use super::service::GlobalRouterService;
use crate::kv_dc_relay::global_view_consumer::{GlobalViewRuntime, RelayDgdSource};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PocRouterConfig {
    listen: SocketAddr,
    freshness: FreshnessConfig,
    overlap_max_age_ms: u64,
    #[serde(default)]
    kv_aware_token_id_completions: bool,
    #[serde(default)]
    experimental_credit_policy: Option<CreditPolicyConfig>,
    pools: Vec<PoolConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreditPolicyConfig {
    overlap_credit: f64,
    scrape_interval_ms: u64,
    max_sample_age_ms: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FreshnessConfig {
    catalog_max_age_ms: u64,
    readiness_max_age_ms: u64,
    capacity_max_age_ms: u64,
    load_max_age_ms: u64,
    kv_usage_max_age_ms: u64,
    kv_overlap_max_age_ms: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PoolConfig {
    site_id: String,
    namespace: String,
    dgd_name: String,
    region: String,
    #[serde(default)]
    availability_zone: Option<String>,
    #[serde(default)]
    cluster: Option<String>,
    #[serde(default)]
    datacenter: Option<String>,
    runtime_namespace: String,
    model: String,
    #[serde(default)]
    stats_only_aggregated: bool,
    #[serde(default)]
    experimental_scheduler_metrics: Option<MetricsConfig>,
    private_frontend_base_url: String,
    relay_grpc_url: String,
    stats_grpc_url: String,
    subscriber_id: String,
}

impl PocRouterConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read(path)
            .with_context(|| format!("read Global Router config {}", path.display()))?;
        serde_json::from_slice(&contents)
            .with_context(|| format!("parse Global Router config {}", path.display()))
    }

    pub fn build(self) -> Result<(SocketAddr, GlobalRouterService)> {
        let freshness = self.freshness.validate()?;
        if self.overlap_max_age_ms == 0 || self.pools.is_empty() {
            bail!("Global Router needs a positive overlap age and at least one pool");
        }
        if self
            .experimental_credit_policy
            .as_ref()
            .is_some_and(|p| !p.overlap_credit.is_finite() || p.overlap_credit < 0.0)
        {
            bail!("overlap_credit must be finite and nonnegative");
        }
        let mut metrics_sources = Vec::new();
        let mut sources = Vec::with_capacity(self.pools.len());
        for pool in self.pools {
            let key = PoolKey::new(&pool.site_id, &pool.namespace, &pool.dgd_name)
                .with_context(|| format!("invalid pool identity at site {}", pool.site_id))?;
            if pool.region.trim().is_empty() {
                bail!("pool {} needs a region", pool.site_id);
            }
            if parse_private_frontend_base(&pool.private_frontend_base_url).is_none() {
                bail!(
                    "pool {} has an invalid private Frontend base URL",
                    pool.site_id
                );
            }
            match (
                self.experimental_credit_policy.is_some(),
                pool.experimental_scheduler_metrics,
            ) {
                (true, Some(config)) => {
                    config.validate()?;
                    metrics_sources.push(MetricsSource {
                        pool_id: V1PoolIdDeriver.derive(&key),
                        model: pool.model.clone(),
                        config,
                    });
                }
                (true, None) => {
                    bail!("credit experiment requires scheduler metrics for every pool")
                }
                (false, Some(_)) => bail!("scheduler metrics require experimental_credit_policy"),
                (false, None) => {}
            }
            sources.push(RelayDgdSource {
                key,
                location: PoolLocation {
                    region: pool.region,
                    availability_zone: pool.availability_zone,
                    cluster: pool.cluster,
                    datacenter: pool.datacenter,
                },
                scope: RelayPoolScope {
                    runtime_namespace: pool.runtime_namespace,
                    frontend_endpoint: pool.private_frontend_base_url,
                },
                model: pool.model,
                stats_only_aggregated: pool.stats_only_aggregated,
                subscriber_id: pool.subscriber_id,
                relay_channel: channel(&pool.relay_grpc_url).context("configure Relay channel")?,
                stats_channel: channel(&pool.stats_grpc_url)
                    .context("configure Relay stats channel")?,
            });
        }
        let view = Arc::new(GlobalViewRuntime::new(
            sources,
            Duration::from_millis(self.overlap_max_age_ms),
        )?);
        let mut service = GlobalRouterService::new_with_kv_token_id_completions(
            view,
            freshness,
            self.kv_aware_token_id_completions,
        )?;
        if let Some(policy) = self.experimental_credit_policy {
            let metrics = SchedulerMetrics::new(
                metrics_sources,
                policy.scrape_interval_ms,
                policy.max_sample_age_ms,
            )?;
            service = service.with_experimental_credit(metrics, policy.overlap_credit)?;
        }
        Ok((self.listen, service))
    }
}

impl FreshnessConfig {
    fn validate(self) -> Result<FreshnessPolicy> {
        let ages = [
            self.catalog_max_age_ms,
            self.readiness_max_age_ms,
            self.capacity_max_age_ms,
            self.load_max_age_ms,
            self.kv_usage_max_age_ms,
            self.kv_overlap_max_age_ms,
        ];
        if ages.contains(&0) {
            bail!("all Global View freshness age limits must be positive");
        }
        Ok(FreshnessPolicy {
            catalog_max_age_ms: self.catalog_max_age_ms,
            readiness_max_age_ms: self.readiness_max_age_ms,
            capacity_max_age_ms: self.capacity_max_age_ms,
            load_max_age_ms: self.load_max_age_ms,
            kv_usage_max_age_ms: self.kv_usage_max_age_ms,
            kv_overlap_max_age_ms: self.kv_overlap_max_age_ms,
        })
    }
}

fn channel(raw: &str) -> Result<Channel> {
    let endpoint = Endpoint::from_shared(raw.to_owned())
        .with_context(|| format!("invalid gRPC endpoint {raw}"))?
        .connect_timeout(Duration::from_secs(5));
    Ok(endpoint.connect_lazy())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> String {
        serde_json::json!({
            "listen": "127.0.0.1:8090",
            "freshness": {
                "catalog_max_age_ms": 45000,
                "readiness_max_age_ms": 45000,
                "capacity_max_age_ms": 30000,
                "load_max_age_ms": 30000,
                "kv_usage_max_age_ms": 30000,
                "kv_overlap_max_age_ms": 30000
            },
            "overlap_max_age_ms": 30000,
            "pools": [
                {
                    "site_id": "ohio",
                    "namespace": "mocker",
                    "dgd_name": "mocker-a",
                    "region": "us-east-2",
                    "runtime_namespace": "mocker-a",
                    "model": "model",
                    "private_frontend_base_url": "http://127.0.0.1:8000",
                    "relay_grpc_url": "http://127.0.0.1:9000",
                    "stats_grpc_url": "http://127.0.0.1:9001",
                    "subscriber_id": "router-ohio-a"
                },
                {
                    "site_id": "west",
                    "namespace": "mocker",
                    "dgd_name": "mocker-b",
                    "region": "us-west-2",
                    "runtime_namespace": "mocker-b",
                    "model": "model",
                    "private_frontend_base_url": "http://127.0.0.1:8001",
                    "relay_grpc_url": "http://127.0.0.1:9010",
                    "stats_grpc_url": "http://127.0.0.1:9011",
                    "subscriber_id": "router-ohio-b"
                }
            ]
        })
        .to_string()
    }

    #[tokio::test]
    async fn builds_two_region_router_without_opening_connections() {
        let config: PocRouterConfig = serde_json::from_str(&config()).unwrap();
        let (listen, _service) = config.build().unwrap();
        assert_eq!(listen.to_string(), "127.0.0.1:8090");
    }

    #[tokio::test]
    async fn rejects_non_base_frontend_url() {
        let mut config: serde_json::Value = serde_json::from_str(&config()).unwrap();
        config["pools"][0]["private_frontend_base_url"] =
            "https://us-east-2.api.dynamo.com/v1/chat/completions".into();
        let config: PocRouterConfig = serde_json::from_value(config).unwrap();
        assert!(config.build().is_err());
    }
}
