// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run one DC Relay with both Global View gRPC contracts from a JSON POC config.

use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::Path;

use anyhow::{Context, Result, bail};
use dynamo_llm::kv_dc_relay::{
    KvDcRelay, KvDcRelayConfig, KvDcRelayDiscoveryConfig, KvDcRelaySources,
    wan::grpc::KvDcRelayGrpcConfig,
};
use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelayPocConfig {
    dc_id: String,
    runtime_namespace: String,
    #[serde(default)]
    watch_namespaces: Vec<String>,
    wan_listen: SocketAddr,
    stats_listen: SocketAddr,
    #[serde(default)]
    stats_allow_non_loopback: bool,
}

impl RelayPocConfig {
    fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents =
            fs::read(path).with_context(|| format!("read Relay config {}", path.display()))?;
        let config: Self = serde_json::from_slice(&contents)
            .with_context(|| format!("parse Relay config {}", path.display()))?;
        if config.dc_id.trim().is_empty() || config.runtime_namespace.trim().is_empty() {
            bail!("Relay dc_id and runtime_namespace must be nonempty");
        }
        if config.wan_listen == config.stats_listen {
            bail!("Relay WAN and stats listeners must use different addresses");
        }
        Ok(config)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let mut args = env::args_os().skip(1);
    let Some(config_path) = args.next() else {
        bail!("usage: kv-dc-relay-poc CONFIG.json");
    };
    if args.next().is_some() {
        bail!("usage: kv-dc-relay-poc CONFIG.json");
    }
    let poc = RelayPocConfig::load(config_path)?;
    let runtime = Runtime::from_current()?;
    let distributed = DistributedRuntime::new(runtime, DistributedConfig::try_from_settings()?)
        .await
        .context("connect Relay to Dynamo discovery")?;
    let component = distributed
        .namespace(&poc.runtime_namespace)?
        .component("kv_dc_relay")?;
    let config = KvDcRelayConfig {
        sources: KvDcRelaySources::Discovery(KvDcRelayDiscoveryConfig {
            watch_all: poc.watch_namespaces.is_empty(),
            namespaces: poc.watch_namespaces,
            endpoint_prefixes: Vec::new(),
        }),
        producer: dynamo_llm::kv_dc_relay::KvDcRelayProducerConfig {
            grpc_listen_address: Some(poc.stats_listen),
            stats_allow_non_loopback: poc.stats_allow_non_loopback,
            ..Default::default()
        },
        transport: Some(KvDcRelayGrpcConfig::new(poc.wan_listen)),
    };
    let relay = KvDcRelay::start(component, poc.dc_id, config).await?;
    tracing::info!(wan = %poc.wan_listen, stats = %poc.stats_listen, "KV DC Relay listening");
    let failed = tokio::select! {
        signal = shutdown_signal() => {
            signal?;
            false
        }
        _ = relay.wait_for_shutdown() => true,
    };
    let health = relay.health().await;
    relay.shutdown().await?;
    if failed {
        bail!(
            "KV DC Relay stopped: {}",
            health
                .host_last_error
                .unwrap_or_else(|| "supervisor stopped".to_owned())
        );
    }
    Ok(())
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).context("register SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.context("wait for Ctrl-C")?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.context("wait for Ctrl-C")?;
    Ok(())
}
