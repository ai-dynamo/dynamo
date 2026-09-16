// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Node-local KV relaying. No engine, model registration, or request ingress.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use dynamo_backend_common::{CommonArgs, DynamoError};
use dynamo_llm::discovery::{RuntimeConfigWatch, runtime_config_watch};
use dynamo_llm::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::distributed::{DistributedConfig, DistributedRuntime};
use dynamo_runtime::{Runtime, logging};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client;
use crate::context::{KV_CONFIG_KEY, SidecarContext, WORKER_GROUP_KEY};

pub(crate) struct HeadlessSidecar {
    context: SidecarContext,
    group_id: String,
    common: CommonArgs,
    discovery_timeout: Duration,
}

#[derive(Clone, Debug, Deserialize)]
struct LeaderKvConfig {
    block_size: u32,
    local_dp_ranks: Vec<u32>,
}

impl HeadlessSidecar {
    pub(crate) fn from_args(args: Args) -> Result<Self, DynamoError> {
        let context = args
            .sidecar_context
            .expect("dispatcher checked telemetry context");
        let group_id = context
            .worker_group_id()
            .map_err(|error| client::invalid_arg(error.to_string()))?
            .ok_or_else(|| client::invalid_arg("telemetry mode requires a multinode group"))?;
        if args.sidecar.common.route_to_encoder || args.sidecar.common.enable_rl {
            return Err(client::invalid_arg(
                "telemetry mode cannot register encoder or RL request routes",
            ));
        }
        Ok(Self {
            context,
            group_id,
            common: args.sidecar.common,
            discovery_timeout: Duration::from_secs(args.leader_discovery_timeout_secs),
        })
    }

    pub(crate) fn run(self) -> Result<()> {
        logging::init();
        let runtime = Runtime::from_settings()?;
        runtime.secondary().block_on(async {
            let shutdown = CancellationToken::new();
            let result = async {
                // Install signal listeners before discovery so a follower waiting
                // for its leader can still shut down immediately.
                let mut terminate =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
                let mut interrupt =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
                tokio::select! {
                    result = self.run_inner(runtime.clone(), shutdown.clone()) => result,
                    _ = terminate.recv() => Ok(()),
                    _ = interrupt.recv() => Ok(()),
                    _ = runtime.primary_token().cancelled_owned() => Ok(()),
                }
            }
            .await;
            shutdown.cancel();
            runtime.shutdown();
            result
        })
    }

    async fn run_inner(&self, runtime: Runtime, shutdown: CancellationToken) -> Result<()> {
        let drt = DistributedRuntime::new(runtime, DistributedConfig::from_settings()).await?;
        let endpoint = drt
            .namespace(&self.common.namespace)?
            .component(&self.common.component)?
            .endpoint(&self.common.endpoint);
        tracing::info!(node_rank = self.context.node_rank, group = %self.group_id,
            endpoint = %endpoint.id(), "Waiting for SGLang leader for local KV publishing");
        let mut configs = runtime_config_watch(&endpoint, shutdown.clone()).await?;
        let (worker_id, config) = tokio::time::timeout(self.discovery_timeout,
            wait_for_leader(&mut configs, &self.group_id, &shutdown)).await
            .context("timed out waiting for SGLang leader; check namespace, component, endpoint, and dist_init_addr")??;
        let leader = validate_leader(&self.context, &config)?;
        let _publishers = start_publishers(&endpoint, &self.context, worker_id, &config, &leader)?;
        tracing::info!(node_rank = self.context.node_rank, worker_id, group = %self.group_id,
            local_ranks = ?self.context.kv_event_sources.iter().map(|source| source.dp_rank).collect::<Vec<_>>(),
            "SGLang headless sidecar publishing local KV events");

        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                result = configs.changed() => {
                    result.context("leader discovery watch closed")?;
                    check_leader_unchanged(&configs.borrow_and_update(), &self.group_id, worker_id, &config)?;
                }
            }
        }
        // Publisher Drop cancels local subscriptions; the follower never owns
        // the leader's serving registration. Independent sidecar restart is not
        // supported by SGLang's managed lifecycle.
    }
}

fn check_leader_unchanged(
    configs: &HashMap<u64, ModelRuntimeConfig>,
    group_id: &str,
    worker_id: u64,
    original: &ModelRuntimeConfig,
) -> Result<()> {
    let (current_id, current) = matching_leader(configs, group_id)?
        .context("SGLang leader disappeared; restart the distributed engine instance")?;
    ensure!(
        current_id == worker_id
            && current.data_parallel_start_rank == original.data_parallel_start_rank
            && current.data_parallel_size == original.data_parallel_size
            && current.runtime_data.get(KV_CONFIG_KEY) == original.runtime_data.get(KV_CONFIG_KEY)
            && current.enable_local_indexer == original.enable_local_indexer
            && current.kv_state_endpoint == original.kv_state_endpoint,
        "SGLang leader identity or KV publishing configuration changed; restart the distributed engine instance"
    );
    Ok(())
}

fn matching_leader(
    configs: &HashMap<u64, ModelRuntimeConfig>,
    group_id: &str,
) -> Result<Option<(u64, ModelRuntimeConfig)>> {
    let mut matches = configs.iter().filter(|(_, config)| {
        config
            .runtime_data
            .get(WORKER_GROUP_KEY)
            .and_then(|value| value.as_str())
            == Some(group_id)
    });
    let first = matches.next();
    ensure!(
        matches.next().is_none(),
        "multiple SGLang leaders registered for group {group_id}"
    );
    Ok(first.map(|(id, config)| (*id, config.clone())))
}

async fn wait_for_leader(
    configs: &mut RuntimeConfigWatch,
    group_id: &str,
    shutdown: &CancellationToken,
) -> Result<(u64, ModelRuntimeConfig)> {
    loop {
        if let Some(leader) = matching_leader(&configs.borrow_and_update(), group_id)? {
            return Ok(leader);
        }
        tokio::select! {
            _ = shutdown.cancelled() => bail!("leader lookup cancelled"),
            result = configs.changed() => result.context("leader discovery watch closed")?,
        }
    }
}

fn validate_leader(
    context: &SidecarContext,
    config: &ModelRuntimeConfig,
) -> Result<LeaderKvConfig> {
    ensure!(
        config.data_parallel_start_rank == 0,
        "SGLang leader must serve the complete global DP range"
    );
    let metadata = config
        .runtime_data
        .get(KV_CONFIG_KEY)
        .context("leader does not advertise node-local sidecar KV metadata")?;
    let leader: LeaderKvConfig =
        serde_json::from_value(metadata.clone()).context("invalid leader KV metadata")?;
    context.validate_registration(config.data_parallel_size, Some(leader.block_size))?;
    for source in &context.kv_event_sources {
        ensure!(
            !leader.local_dp_ranks.contains(&source.dp_rank),
            "local KV rank {} is already published by the leader",
            source.dp_rank
        );
    }
    Ok(leader)
}

fn start_publishers(
    endpoint: &Endpoint,
    context: &SidecarContext,
    worker_id: u64,
    config: &ModelRuntimeConfig,
    leader: &LeaderKvConfig,
) -> Result<Vec<KvEventPublisher>> {
    context
        .kv_event_sources
        .iter()
        .map(|source| {
            KvEventPublisher::new_with_local_indexer_and_worker_id_at(
                endpoint.clone(),
                config.effective_kv_state_endpoint(&endpoint.id()),
                Some(worker_id),
                leader.block_size,
                Some(KvEventSourceConfig::Zmq {
                    endpoint: source.endpoint.clone(),
                    topic: source.topic.clone(),
                    // Same SGLang normalization contract as the full worker relay.
                    image_token_id: None,
                    video_token_id: None,
                }),
                config.enable_local_indexer,
                source.dp_rank,
                None,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::tests::context_json;
    use serde_json::json;
    use tokio::sync::watch;

    fn leader(group: &str) -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            data_parallel_start_rank: 0,
            data_parallel_size: 8,
            runtime_data: HashMap::from([
                (WORKER_GROUP_KEY.into(), json!(group)),
                (
                    KV_CONFIG_KEY.into(),
                    json!({"block_size":64,"local_dp_ranks":[0,1,2,3]}),
                ),
            ]),
            ..Default::default()
        }
    }

    #[test]
    fn matches_exact_group_not_arbitrary_worker() {
        let mut configs = HashMap::from([(42, leader("A")), (99, leader("B"))]);
        assert_eq!(matching_leader(&configs, "A").unwrap().unwrap().0, 42);
        assert!(matching_leader(&configs, "C").unwrap().is_none());
        configs.insert(100, leader("A"));
        assert!(matching_leader(&configs, "A").is_err());
    }

    #[test]
    fn leader_restart_or_topology_change_requires_group_restart() {
        let original = leader("A");
        let mut configs = HashMap::from([(42, original.clone())]);
        check_leader_unchanged(&configs, "A", 42, &original).unwrap();
        // Changes to unrelated workers must not interrupt this group's relay.
        configs.insert(99, leader("B"));
        check_leader_unchanged(&configs, "A", 42, &original).unwrap();
        configs.remove(&42);
        assert!(check_leader_unchanged(&configs, "A", 42, &original).is_err());
        configs.insert(43, original.clone());
        assert!(check_leader_unchanged(&configs, "A", 42, &original).is_err());
        configs.remove(&43);
        configs.insert(42, original.clone());
        configs.get_mut(&42).unwrap().enable_local_indexer = !original.enable_local_indexer;
        assert!(check_leader_unchanged(&configs, "A", 42, &original).is_err());
    }

    #[test]
    fn rejects_incompatible_or_duplicate_rank_ownership() {
        let context: SidecarContext = context_json("telemetry").to_string().parse().unwrap();
        let mut config = leader("A");
        validate_leader(&context, &config).unwrap();
        config.data_parallel_size = 4;
        assert!(validate_leader(&context, &config).is_err());
        config.data_parallel_size = 8;
        config.runtime_data.get_mut(KV_CONFIG_KEY).unwrap()["local_dp_ranks"] = json!([4]);
        assert!(validate_leader(&context, &config).is_err());
    }

    #[tokio::test]
    async fn follower_can_start_before_leader() {
        let (tx, mut rx) = watch::channel(HashMap::new());
        let cancel = CancellationToken::new();
        let lookup = wait_for_leader(&mut rx, "A", &cancel);
        let publish = async {
            tokio::task::yield_now().await;
            tx.send(HashMap::from([(42, leader("A"))])).unwrap();
        };
        let (found, ()) = tokio::join!(lookup, publish);
        assert_eq!(found.unwrap().0, 42);
    }

    #[tokio::test]
    async fn shutdown_interrupts_a_pending_lookup() {
        let (_tx, mut rx) = watch::channel(HashMap::new());
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(wait_for_leader(&mut rx, "A", &cancel).await.is_err());
    }

    #[tokio::test]
    async fn local_zmq_events_keep_global_rank_and_leader_identity_without_serving() {
        use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, RouterEvent};
        use dynamo_runtime::discovery::{DiscoveryQuery, EventSourceQuery};
        use dynamo_runtime::transports::event_plane::EventSubscriber;
        use futures::SinkExt;

        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("headless-kv-test")
            .unwrap()
            .component("backend")
            .unwrap()
            .endpoint("generate");
        let socket_dir = tempfile::tempdir().unwrap();
        let source_address = format!("ipc://{}/kv.sock", socket_dir.path().display());
        let zmq_context = tmq::Context::new();
        let mut source = tmq::publish::publish(&zmq_context)
            .set_linger(0)
            .bind(&source_address)
            .unwrap();
        let mut raw = context_json("telemetry");
        raw["kv_event_sources"][0]["endpoint"] = json!(source_address);
        let context: SidecarContext = raw.to_string().parse().unwrap();
        let config = leader("A");
        let metadata = validate_leader(&context, &config).unwrap();
        let mut subscriber = EventSubscriber::for_endpoint(&endpoint, KV_EVENT_SUBJECT)
            .await
            .unwrap()
            .typed::<Vec<RouterEvent>>();
        let publishers = start_publishers(&endpoint, &context, 42, &config, &metadata).unwrap();

        // Repeat until the real ZMQ subscriptions are connected; no fixed sleep.
        let received = tokio::time::timeout(Duration::from_secs(5), async {
            let mut ticks = tokio::time::interval(Duration::from_millis(50));
            let mut seq = 0_u64;
            loop {
                tokio::select! {
                    batch = subscriber.next() => {
                        let (envelope, events) = batch.unwrap().unwrap();
                        if let Some(event) = events.first() {
                            break (envelope, event.clone());
                        }
                    }
                    _ = ticks.tick() => {
                        seq += 1;
                        let payload = rmp_serde::to_vec_named(&json!([
                            0.0, [{"type":"BlockRemoved", "block_hashes":[42]}], 4
                        ])).unwrap();
                        source.send(vec![Vec::new(), seq.to_be_bytes().to_vec(), payload]).await.unwrap();
                    }
                }
            }
        }).await.expect("headless relay should forward local events");
        assert_eq!(received.1.worker_id, 42);
        assert_eq!(received.1.event.dp_rank, 4);
        assert_ne!(received.0.publisher_id, 42);
        assert!(
            drt.discovery()
                .list(DiscoveryQuery::Endpoint {
                    namespace: "headless-kv-test".into(),
                    component: "backend".into(),
                    endpoint: "generate".into(),
                })
                .await
                .unwrap()
                .is_empty(),
            "telemetry must not register an inference worker"
        );
        let source_query = DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
            endpoint.id(),
            KV_EVENT_SUBJECT,
        ));
        assert_eq!(
            drt.discovery()
                .list(source_query.clone())
                .await
                .unwrap()
                .len(),
            1
        );
        drop(publishers);
        tokio::time::timeout(Duration::from_secs(5), async {
            while !drt
                .discovery()
                .list(source_query.clone())
                .await
                .unwrap()
                .is_empty()
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("dropping a relay must unregister only its local source");
        drt.shutdown();
    }
}
