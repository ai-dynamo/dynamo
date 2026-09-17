// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Node-local KV relaying using the follower engine's metadata-only gRPC server.
//! No model registration or request ingress.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use dynamo_backend_common::{CommonArgs, DisaggregationMode, DynamoError};
use dynamo_llm::discovery::{RuntimeConfigWatch, runtime_config_watch};
use dynamo_llm::kv_router::publisher::{KvEventPublisher, KvEventSourceConfig};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::config::HealthStatus;
use dynamo_runtime::distributed::{DistributedConfig, DistributedRuntime};
use dynamo_runtime::prelude::DistributedRuntimeProvider;
use dynamo_runtime::{Runtime, logging};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use serde::Deserialize;
use serde_json::Value;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::Args;
use crate::client::{self, KV_CONFIG_KEY, NodeMetadata, WORKER_GROUP_KEY};

const METADATA_POLL_INTERVAL: Duration = Duration::from_secs(5);

pub(crate) struct HeadlessSidecar {
    grpc_endpoint: GrpcEndpoint,
    transport: GrpcTransportConfig,
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
        if args.sidecar.common.route_to_encoder || args.sidecar.common.enable_rl {
            return Err(client::invalid_arg(
                "telemetry mode cannot register encoder or RL request routes",
            ));
        }
        Ok(Self {
            grpc_endpoint: args.sidecar.grpc_endpoint,
            transport: args.sidecar.grpc.config(),
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
        // Followers only implement GetServerInfo. Do not use the full engine's
        // model discovery, HealthCheck, connection pool, or native HTTP client.
        let (mut client, metadata, mode) = self.connect_local_engine().await?;
        let group_id = metadata
            .worker_group_id()?
            .context("telemetry-only mode requires a multinode group")?;

        // Supervise the local engine while waiting for the leader too. Engines
        // and sidecars are managed externally and require a coordinated restart
        // after a failure; the live KV stream has no replay/recovery here.
        tokio::select! {
            _ = shutdown.cancelled() => Ok(()),
            result = monitor_local_engine(&mut client, &metadata, mode, self.transport.connect_attempt_timeout) => result,
            result = self.relay(runtime, shutdown.clone(), &metadata, mode, &group_id) => result,
        }
    }

    async fn connect_local_engine(
        &self,
    ) -> Result<(client::Client, NodeMetadata, DisaggregationMode)> {
        let deadline = Instant::now() + self.transport.startup_deadline;
        let mut client = client::connect(&self.grpc_endpoint, &self.transport, deadline).await?;
        let info = client::get_server_info(&mut client, deadline).await?;
        let (metadata, mode) = follower_metadata(&info)?;
        Ok((client, metadata, mode))
    }

    async fn relay(
        &self,
        runtime: Runtime,
        shutdown: CancellationToken,
        metadata: &NodeMetadata,
        mode: DisaggregationMode,
        group_id: &str,
    ) -> Result<()> {
        let drt = DistributedRuntime::new(runtime, DistributedConfig::from_settings()).await?;
        let component = if mode == DisaggregationMode::Aggregated {
            &self.common.component
        } else {
            mode.discovery_component()
        };
        let endpoint = drt
            .namespace(&self.common.namespace)?
            .component(component)?
            .endpoint(&self.common.endpoint);
        tracing::info!(node_rank = metadata.node_rank, group = group_id,
            endpoint = %endpoint.id(), "Waiting for SGLang leader for local KV publishing");
        let mut configs = runtime_config_watch(&endpoint, shutdown.clone()).await?;
        let (worker_id, config) = tokio::time::timeout(self.discovery_timeout,
            wait_for_leader(&mut configs, group_id, &shutdown)).await
            .context("timed out waiting for SGLang leader; check namespace, component, endpoint, and dist_init_addr")??;
        let leader = validate_leader(metadata, &config)?;
        let _publishers = start_publishers(&endpoint, metadata, worker_id, &config, &leader)?;
        tracing::info!(node_rank = metadata.node_rank, worker_id, group = group_id,
            local_ranks = ?metadata.kv_event_sources.iter().map(|source| source.dp_rank).collect::<Vec<_>>(),
            "SGLang headless sidecar publishing local KV events");

        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                result = configs.changed() => {
                    result.context("leader discovery watch closed")?;
                    check_leader_unchanged(&configs.borrow_and_update(), group_id, worker_id, &config)?;
                }
            }
        }
    }
}

fn follower_metadata(info: &Value) -> Result<(NodeMetadata, DisaggregationMode)> {
    let metadata = NodeMetadata::from_server_info(info)?
        .context("telemetry-only mode requires GetServerInfo.kv_event_sources; upgrade SGLang")?;
    ensure!(
        metadata.node_rank > 0 && metadata.nnodes > 1,
        "--telemetry-only requires a follower node; run the leader sidecar without this flag"
    );
    ensure!(
        !metadata.kv_event_sources.is_empty(),
        "telemetry-only node has no local KV sources; do not launch a sidecar on a TP-only follower without a publisher"
    );
    Ok((metadata, client::discovery_mode(info)?))
}

async fn monitor_local_engine(
    client: &mut client::Client,
    original: &NodeMetadata,
    mode: DisaggregationMode,
    timeout: Duration,
) -> Result<()> {
    let mut ticks = tokio::time::interval_at(
        Instant::now() + METADATA_POLL_INTERVAL,
        METADATA_POLL_INTERVAL,
    );
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        ticks.tick().await;
        check_local_engine(client, original, mode, timeout).await?;
    }
}

async fn check_local_engine(
    client: &mut client::Client,
    original: &NodeMetadata,
    mode: DisaggregationMode,
    timeout: Duration,
) -> Result<()> {
    let info = client::get_server_info(client, Instant::now() + timeout)
        .await
        .context("local SGLang metadata server unavailable; restart the distributed engine instance and its sidecars")?;
    let (current, current_mode) = follower_metadata(&info)
        .context("local SGLang metadata became invalid; restart the distributed engine instance and its sidecars")?;
    ensure!(
        &current == original && current_mode == mode,
        "local SGLang topology or KV source configuration changed; restart the distributed engine instance and its sidecars"
    );
    Ok(())
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

fn validate_leader(context: &NodeMetadata, config: &ModelRuntimeConfig) -> Result<LeaderKvConfig> {
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
    context: &NodeMetadata,
    worker_id: u64,
    config: &ModelRuntimeConfig,
    leader: &LeaderKvConfig,
) -> Result<Vec<KvEventPublisher>> {
    let publishers = context
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
        .collect::<Result<Vec<_>>>()?;
    // There is no serving worker to mark this process ready. Publish readiness
    // only after all local publishers have been created.
    endpoint
        .drt()
        .system_health()
        .lock()
        .set_health_status(HealthStatus::Ready);
    Ok(publishers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context as TaskContext, Poll};

    use clap::Parser;
    use serde_json::json;
    use tokio::sync::watch;
    use tonic::body::Body;
    use tonic::codegen::{BoxFuture, Service, http};

    use crate::proto as pb;

    // One real gRPC method, matching the upstream follower server. Recording
    // every URI makes accidental full-engine discovery/health RPCs visible.
    #[derive(Clone)]
    struct MetadataService {
        info: Arc<Mutex<Value>>,
        calls: Arc<Mutex<Vec<String>>>,
        unavailable: Arc<AtomicBool>,
    }

    impl tonic::server::NamedService for MetadataService {
        const NAME: &'static str = "sglang.runtime.v1.SglangService";
    }

    impl tonic::server::UnaryService<pb::GetServerInfoRequest> for MetadataService {
        type Response = pb::GetServerInfoResponse;
        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;

        fn call(&mut self, _: tonic::Request<pb::GetServerInfoRequest>) -> Self::Future {
            let json_info = self.info.lock().unwrap().to_string();
            let unavailable = self.unavailable.load(Ordering::Relaxed);
            Box::pin(async move {
                if unavailable {
                    return Err(tonic::Status::unavailable("local engine stopped"));
                }
                Ok(tonic::Response::new(pb::GetServerInfoResponse {
                    json_info,
                }))
            })
        }
    }

    impl Service<http::Request<Body>> for MetadataService {
        type Response = http::Response<Body>;
        type Error = Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;

        fn poll_ready(&mut self, _: &mut TaskContext<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, request: http::Request<Body>) -> Self::Future {
            let path = request.uri().path().to_owned();
            self.calls.lock().unwrap().push(path.clone());
            let service = self.clone();
            Box::pin(async move {
                if path != "/sglang.runtime.v1.SglangService/GetServerInfo" {
                    return Ok(tonic::Status::unimplemented("metadata-only follower").into_http());
                }
                let mut grpc = tonic::server::Grpc::new(tonic::codec::ProstCodec::default());
                Ok(grpc.unary(service, request).await)
            })
        }
    }

    struct MetadataServer {
        endpoint: String,
        service: MetadataService,
        task: tokio::task::JoinHandle<()>,
    }

    impl MetadataServer {
        async fn start() -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let service = MetadataService {
                info: Arc::new(Mutex::new(follower_info())),
                calls: Arc::default(),
                unavailable: Arc::default(),
            };
            let incoming = futures::stream::unfold(listener, |listener| async move {
                let connection = listener.accept().await.map(|(stream, _)| stream);
                Some((connection, listener))
            });
            let server = tonic::transport::Server::builder().add_service(service.clone());
            let task =
                tokio::spawn(async move { server.serve_with_incoming(incoming).await.unwrap() });
            Self {
                endpoint,
                service,
                task,
            }
        }
    }

    impl Drop for MetadataServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    fn follower_info() -> Value {
        json!({"node_rank":1,"nnodes":2,"dp_size":8,
            "dist_init_addr":"127.0.0.1:2345", "disaggregation_mode":"prefill",
            "kv_event_sources":[{"dp_rank":4,"endpoint":"tcp://127.0.0.1:5561",
                "topic":"","block_size":64}]})
    }

    #[test]
    fn telemetry_requires_follower_metadata_and_local_sources() {
        let mut info = follower_info();
        let (metadata, mode) = follower_metadata(&info).unwrap();
        assert_eq!(metadata.node_rank, 1);
        assert_eq!(mode, DisaggregationMode::Prefill);
        info["node_rank"] = json!(0);
        assert!(follower_metadata(&info).is_err());
        info["node_rank"] = json!(1);
        info["kv_event_sources"] = json!([]);
        assert!(follower_metadata(&info).is_err());
        info.as_object_mut().unwrap().remove("kv_event_sources");
        assert!(follower_metadata(&info).is_err());
    }

    #[tokio::test]
    async fn follower_discovers_and_monitors_using_only_server_info() {
        let server = MetadataServer::start().await;
        let args = Args::try_parse_from([
            "sidecar",
            "--telemetry-only",
            "--grpc-endpoint",
            &server.endpoint,
            "--grpc-startup-deadline-secs",
            "5",
        ])
        .unwrap();
        let sidecar = HeadlessSidecar::from_args(args).unwrap();
        let (mut client, original, mode) = sidecar.connect_local_engine().await.unwrap();
        assert_eq!(mode, DisaggregationMode::Prefill);
        let timeout = Duration::from_secs(1);
        // A misplaced full sidecar fails with an actionable mode error, without
        // asking the follower for any of the unsupported inference RPCs.
        let error = client::discover(&mut client, Instant::now() + timeout)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("use --telemetry-only"));
        check_local_engine(&mut client, &original, mode, timeout)
            .await
            .unwrap();
        // Unrelated live server fields do not invalidate the KV contract.
        server.service.info.lock().unwrap()["uptime"] = json!(42);
        check_local_engine(&mut client, &original, mode, timeout)
            .await
            .unwrap();
        server.service.info.lock().unwrap()["kv_event_sources"][0]["endpoint"] =
            json!("tcp://127.0.0.1:5562");
        assert!(
            check_local_engine(&mut client, &original, mode, timeout)
                .await
                .is_err()
        );
        *server.service.info.lock().unwrap() = follower_info();
        server.service.info.lock().unwrap()["disaggregation_mode"] = json!("decode");
        assert!(
            check_local_engine(&mut client, &original, mode, timeout)
                .await
                .is_err()
        );
        assert_eq!(
            server.service.calls.lock().unwrap().as_slice(),
            vec!["/sglang.runtime.v1.SglangService/GetServerInfo"; 6]
        );
        server.service.unavailable.store(true, Ordering::Relaxed);
        let error = check_local_engine(&mut client, &original, mode, timeout)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("local SGLang metadata server unavailable")
        );
    }

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
        let (context, _) = follower_metadata(&follower_info()).unwrap();
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
        let mut raw = follower_info();
        raw["kv_event_sources"][0]["endpoint"] = json!(source_address);
        let (context, _) = follower_metadata(&raw).unwrap();
        let config = leader("A");
        let metadata = validate_leader(&context, &config).unwrap();
        let mut subscriber = EventSubscriber::for_endpoint(&endpoint, KV_EVENT_SUBJECT)
            .await
            .unwrap()
            .typed::<Vec<RouterEvent>>();
        assert!(!drt.system_health().lock().get_health_status().0);
        let publishers = start_publishers(&endpoint, &context, 42, &config, &metadata).unwrap();
        assert!(drt.system_health().lock().get_health_status().0);

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
