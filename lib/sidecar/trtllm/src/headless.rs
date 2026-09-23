// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Node-local telemetry without model registration or inference ingress.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use dynamo_backend_common::{DisaggregationMode, KvEventSource};
use dynamo_llm::discovery::{RuntimeConfigWatch, runtime_config_watch};
use dynamo_llm::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::config::HealthStatus;
use dynamo_runtime::distributed::{DistributedConfig, DistributedRuntime};
use dynamo_runtime::{Runtime, logging};
use futures::stream::{FuturesUnordered, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::TrtllmClient;
use crate::node::{NODE_METADATA_KEY, NodeMetadata};
use crate::proto as pb;

const POLL_INTERVAL: Duration = Duration::from_secs(5);

pub(crate) fn run(args: Args, node: NodeMetadata) -> Result<()> {
    ensure!(
        !args.sidecar.common.route_to_encoder && !args.sidecar.common.enable_rl,
        "Telemetry-only sidecars cannot register encoder or RL routes"
    );
    logging::init();
    let runtime = Runtime::from_settings()?;
    let result = runtime.secondary().block_on(async {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        let mut interrupt =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
        let cancel = runtime.primary_token();
        tokio::select! {
            result = relay(runtime.clone(), args, node, cancel.clone()) => result,
            _ = terminate.recv() => Ok(()),
            _ = interrupt.recv() => Ok(()),
            _ = cancel.cancelled() => Ok(()),
        }
    });
    runtime.shutdown();
    result
}

async fn relay(
    runtime: Runtime,
    args: Args,
    node: NodeMetadata,
    cancel: CancellationToken,
) -> Result<()> {
    let transport = args.sidecar.grpc.config();
    let client = TrtllmClient::connect(&args.sidecar.grpc_endpoint, transport).await?;
    let info = client
        .server_info()
        .await?
        .context("Follower metadata disappeared")?;
    ensure!(
        NodeMetadata::from_info(Some(&info))?.as_ref() == Some(&node),
        "Engine incarnation changed during startup"
    );
    let mut sources = client.kv_event_sources().await?;
    node.select_local_sources(&mut sources)?;
    let block_size = info
        .capacity
        .as_ref()
        .and_then(|c| c.kv_block_size)
        .unwrap_or(0);
    let dp_size = info
        .parallelism
        .as_ref()
        .and_then(|p| p.data_parallel_size)
        .unwrap_or(1);
    validate_local_sources(&sources, dp_size)?;
    ensure!(
        sources.is_empty() || block_size > 0,
        "Local publisher has no KV block size"
    );
    let drt = DistributedRuntime::new(runtime, DistributedConfig::from_settings()).await?;
    let common = &args.sidecar.common;
    let component = if common.disaggregation_mode == DisaggregationMode::Aggregated {
        common.component.as_str()
    } else {
        common.disaggregation_mode.discovery_component()
    };
    let endpoint = drt
        .namespace(&common.namespace)?
        .component(component)?
        .endpoint(&common.endpoint);
    let mut configs = runtime_config_watch(&endpoint, cancel.clone()).await?;
    let wait = async {
        loop {
            if let Some(leader) = matching_leader(&configs.borrow(), &node)? {
                return Ok::<_, anyhow::Error>(leader);
            }
            configs.changed().await.context("Leader discovery closed")?;
        }
    };
    let leader = tokio::select! {
        result = tokio::time::timeout(transport.startup_deadline, wait) => result.context("Timed out waiting for engine leader")??,
        result = monitor_engine(&client, &node, &sources) => { result?; unreachable!() },
        _ = cancel.cancelled() => return Ok(()),
    };
    let (worker_id, leader_config) = leader;
    ensure!(
        leader_config.data_parallel_size == dp_size && leader_config.data_parallel_start_rank == 0,
        "Leader registration does not cover the engine's DP ranks"
    );
    ensure!(
        sources.is_empty() || node.kv_block_size == block_size,
        "Leader and follower KV block sizes disagree"
    );
    let heartbeat = crate::engine::kv_heartbeat_timeout(
        info.extra
            .as_ref()
            .and_then(|extra| extra.fields.get("kv_event_heartbeat_interval_ms")),
    );
    let mut publishers: Vec<_> = sources
        .iter()
        .map(|source| {
            let KvEventSource::Zmq {
                endpoint: address,
                topic,
                dp_rank: rank,
                heartbeat_timeout,
            } = crate::engine::to_kv_event_source(source, heartbeat)?
            else {
                bail!("Local KV event source must use ZMQ")
            };
            KvEventPublisher::new_with_local_indexer_and_worker_id_at(
                endpoint.clone(),
                leader_config.effective_kv_state_endpoint(&endpoint.id()),
                Some(worker_id),
                block_size,
                Some(KvEventSourceConfig::Zmq {
                    endpoint: address,
                    topic,
                    image_token_id: None,
                    video_token_id: None,
                    liveness: heartbeat_timeout.map(|timeout| (rank, timeout)),
                }),
                leader_config.enable_local_indexer,
                rank,
                None,
            )
        })
        .collect::<Result<_>>()?;
    let initialize = async {
        futures::future::try_join_all(publishers.iter_mut().map(KvEventPublisher::ready)).await
    };
    tokio::select! {
        result = tokio::time::timeout(transport.startup_deadline, initialize) => {
            result.context("Timed out initializing local KV publishers")??;
        }
        _ = cancel.cancelled() => return Ok(()),
    }
    let mut terminations: FuturesUnordered<_> = publishers
        .iter_mut()
        .map(KvEventPublisher::terminated)
        .collect();
    drt.system_health()
        .lock()
        .set_health_status(HealthStatus::Ready);
    tracing::info!(worker_id, node = %node.node_id, sources = sources.len(), "TRTLLM node-local telemetry ready; no inference registration");
    tokio::select! {
        _ = terminations.next(), if !terminations.is_empty() => {
            bail!("Local KV event publisher stopped; restarting the telemetry sidecar is required")
        }
        result = monitor_engine(&client, &node, &sources) => result,
        result = monitor_leader(&mut configs, &node, worker_id, &leader_config) => result,
        _ = cancel.cancelled() => Ok(()),
    }
}

fn matching_leader(
    configs: &HashMap<u64, ModelRuntimeConfig>,
    node: &NodeMetadata,
) -> Result<Option<(u64, ModelRuntimeConfig)>> {
    let mut matching = Vec::new();
    for (&id, config) in configs {
        let Some(value) = config.runtime_data.get(NODE_METADATA_KEY) else {
            continue;
        };
        // Unrelated engines may advertise a different metadata version.
        if value.get("engine_id").and_then(serde_json::Value::as_str)
            != Some(node.engine_id.as_str())
        {
            continue;
        }
        let leader: NodeMetadata =
            serde_json::from_value(value.clone()).context("Invalid registered node metadata")?;
        if node.matches_leader(&leader) {
            matching.push((id, config.clone()));
        }
    }
    ensure!(
        matching.len() <= 1,
        "Multiple serving leaders for this engine incarnation"
    );
    Ok(matching.pop())
}

async fn monitor_leader(
    configs: &mut RuntimeConfigWatch,
    node: &NodeMetadata,
    worker_id: u64,
    original: &ModelRuntimeConfig,
) -> Result<()> {
    loop {
        // Check before waiting: discovery may have changed while publishers started.
        let (id, current) = matching_leader(&configs.borrow_and_update(), node)?
            .context("Engine leader disappeared; restart engine and sidecars together")?;
        ensure!(
            id == worker_id
                && current.runtime_data == original.runtime_data
                && current.enable_local_indexer == original.enable_local_indexer
                && current.kv_state_endpoint == original.kv_state_endpoint
                && current.data_parallel_size == original.data_parallel_size,
            "Engine leader changed; restart engine and sidecars together"
        );
        configs.changed().await.context("Leader discovery closed")?;
    }
}

async fn monitor_engine(
    client: &TrtllmClient,
    node: &NodeMetadata,
    sources: &[pb::KvEventSource],
) -> Result<()> {
    let mut ticks = tokio::time::interval(POLL_INTERVAL);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticks.tick().await;
        let info = client
            .server_info()
            .await?
            .context("Local metadata endpoint disappeared")?;
        ensure!(
            NodeMetadata::from_info(Some(&info))?.as_ref() == Some(node),
            "Local engine changed; restart engine and sidecars together"
        );
        if client.kv_event_sources().await? != sources {
            bail!("Local event sources changed; restart engine and sidecars together");
        }
    }
}

fn validate_local_sources(sources: &[pb::KvEventSource], dp_size: u32) -> Result<()> {
    let mut ranks = HashSet::new();
    for source in sources {
        let rank = crate::engine::validate_kv_source(source, dp_size)?;
        ensure!(ranks.insert(rank), "Duplicate local DP rank {rank}");
    }
    Ok(())
}
