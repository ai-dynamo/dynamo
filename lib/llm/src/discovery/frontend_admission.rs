// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publishes which workers this frontend routes requests to, so a rolling
//! update can wait for every frontend to serve a replacement worker generation
//! before it retires the previous one.
//!
//! The frontend keeps one discovery event source on [`FRONTEND_ADMISSION_TOPIC`]
//! whose metadata holds the record protocol, whether the frontend's view is
//! synced with discovery, and the workers it routes to: for each endpoint
//! path, the hex IDs of its routable instances, which together form their base
//! model-card keys (see [`FrontendAdmissions`]). The frontend publishes it under
//! Kubernetes discovery, where the record lives in the frontend pod's
//! `DynamoWorkerMetadata` and the Dynamo operator reads it. The operator
//! matches the topic and record fields defined here.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::discovery::{
    Discovery, DiscoveryInstance, DiscoverySpec, EventScope, discovery_safe_publisher_id,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::ModelManager;
use super::model_manager::RoutableMembers;

pub(crate) const FRONTEND_ADMISSION_TOPIC: &str = "frontend-model-admission";
/// Record format version. The operator ignores versions it does not know.
const FRONTEND_ADMISSION_PROTOCOL: &str = "v1";
const RETRY_INTERVAL: Duration = Duration::from_secs(1);
const MIN_PUBLISH_INTERVAL: Duration = Duration::from_millis(200);
/// Record bytes one frontend writes per second at most, so the churn of a large
/// deployment does not turn into heavy API-server traffic. Each write carries
/// the whole record.
const MAX_PUBLISH_BYTES_PER_SECOND: f64 = (1 << 20) as f64;
/// How often to look again while a routing hop activates its target, which
/// completes in the background without a change notification.
const ACTIVATION_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// What one frontend publishes about the workers it routes requests to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrontendAdmissions {
    /// Whether `members` reflect live discovery: false until the frontend
    /// applies its first full discovery snapshot, and again after its
    /// discovery stream ends.
    pub(crate) synced: bool,
    /// Instances that request routing can reach, by endpoint path.
    pub(crate) members: RoutableMembers,
}

impl FrontendAdmissions {
    fn record(&self) -> serde_json::Value {
        let members: BTreeMap<&str, Vec<String>> = self
            .members
            .iter()
            .map(|(endpoint, ids)| {
                let ids = ids.iter().map(|id| format!("{id:x}")).collect();
                (endpoint.as_str(), ids)
            })
            .collect();
        serde_json::json!({
            "protocol": FRONTEND_ADMISSION_PROTOCOL,
            "synced": self.synced,
            "members": members,
        })
    }
}

/// The least time from the start of a pass that wrote `written` record bytes
/// to the start of the next pass.
fn publish_interval(written: usize) -> Duration {
    MIN_PUBLISH_INTERVAL.max(Duration::from_secs_f64(
        written as f64 / MAX_PUBLISH_BYTES_PER_SECOND,
    ))
}

/// The discovery event source that carries `admissions`.
fn admission_source(
    frontend_namespace: &str,
    admissions: &FrontendAdmissions,
    publisher_id: u64,
) -> DiscoverySpec {
    DiscoverySpec::EventSource {
        scope: EventScope::Namespace {
            name: frontend_namespace.to_string(),
        },
        topic: FRONTEND_ADMISSION_TOPIC.to_string(),
        publisher_id,
        metadata: admissions.record(),
    }
}

struct AdmissionPublisher {
    discovery: Arc<dyn Discovery>,
    frontend_namespace: String,
    published: Option<(FrontendAdmissions, DiscoveryInstance)>,
}

impl AdmissionPublisher {
    fn new(discovery: Arc<dyn Discovery>, frontend_namespace: String) -> Self {
        Self {
            discovery,
            frontend_namespace,
            published: None,
        }
    }

    /// Converges the published record on `admissions`. Returns the bytes of the
    /// record written, zero when it already matched, or `None` when the write
    /// failed and must be retried.
    ///
    /// A changed record is replaced, never withdrawn first: a reader must not
    /// see this frontend without a record while it still serves workers.
    async fn reconcile(&mut self, admissions: &FrontendAdmissions) -> Option<usize> {
        if self
            .published
            .as_ref()
            .is_some_and(|(published, _)| published == admissions)
        {
            return Some(0);
        }
        let spec = admission_source(
            &self.frontend_namespace,
            admissions,
            discovery_safe_publisher_id(rand::random()),
        );
        let DiscoverySpec::EventSource { metadata, .. } = &spec else {
            unreachable!("admission records are event sources");
        };
        let written = serde_json::to_vec(metadata).map_or(0, |record| record.len());
        let result = match self.published.take() {
            None => self
                .discovery
                .register(spec)
                .await
                .map_err(|error| (error, None)),
            Some((published, instance)) => self
                .discovery
                .replace_event_source(instance.clone(), spec)
                .await
                .map_err(|error| (error, Some((published, instance)))),
        };
        match result {
            Ok(instance) => {
                tracing::debug!(
                    synced = admissions.synced,
                    members = admissions
                        .members
                        .values()
                        .map(|ids| ids.len())
                        .sum::<usize>(),
                    bytes = written,
                    "Published frontend admissions"
                );
                self.published = Some((admissions.clone(), instance));
                Some(written)
            }
            Err((error, previous)) => {
                tracing::warn!(
                    error = format!("{error:#}"),
                    "Failed to publish frontend admissions; retrying"
                );
                self.published = previous;
                None
            }
        }
    }

    async fn withdraw(&mut self) {
        let Some((_, instance)) = self.published.take() else {
            return;
        };
        if let Err(error) = self.discovery.unregister(instance).await {
            tracing::debug!(
                error = format!("{error:#}"),
                "Failed to withdraw frontend admissions during shutdown"
            );
        }
    }
}

/// Publishes the workers `manager` routes requests to until `cancellation`
/// fires, then withdraws the record.
///
/// `synced` reports whether this frontend's discovery view is live. A pass runs
/// whenever the manager's routing state or `synced` changes, at most every
/// [`MIN_PUBLISH_INTERVAL`] and no more often than keeps the written bytes under
/// [`MAX_PUBLISH_BYTES_PER_SECOND`], so worker churn costs a bounded number of
/// discovery writes. Each pass runs to completion so the publisher always knows
/// which record it registered; cancellation is observed between passes.
pub(crate) async fn run_admission_publisher(
    discovery: Arc<dyn Discovery>,
    frontend_namespace: String,
    manager: Arc<ModelManager>,
    mut synced: watch::Receiver<bool>,
    cancellation: CancellationToken,
) {
    let mut publisher = AdmissionPublisher::new(discovery, frontend_namespace);
    let mut routing_state = manager.subscribe_serving_state();
    let mut synced_open = true;
    loop {
        let pass_started = tokio::time::Instant::now();
        routing_state.borrow_and_update();
        let is_synced = *synced.borrow_and_update();
        let serving = manager.serving_members();
        let admissions = FrontendAdmissions {
            synced: is_synced,
            members: serving.members,
        };
        let written = publisher.reconcile(&admissions).await;
        tokio::select! {
            _ = cancellation.cancelled() => break,
            changed = routing_state.changed() => {
                if changed.is_err() {
                    break;
                }
            }
            changed = synced.changed(), if synced_open => {
                // The owner is gone; keep its last value.
                synced_open = changed.is_ok();
            }
            _ = tokio::time::sleep(RETRY_INTERVAL), if written.is_none() => {}
            _ = tokio::time::sleep(ACTIVATION_POLL_INTERVAL), if serving.awaiting_activation => {}
        }
        tokio::select! {
            _ = cancellation.cancelled() => break,
            _ = tokio::time::sleep_until(pass_started + publish_interval(written.unwrap_or(0))) => {}
        }
    }
    publisher.withdraw().await;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use dynamo_runtime::discovery::{DiscoveryQuery, EventSourceQuery, KVStoreDiscovery};
    use dynamo_runtime::storage::kv;

    use super::*;

    async fn published_records(discovery: &dyn Discovery) -> Vec<serde_json::Value> {
        discovery
            .list(DiscoveryQuery::EventSources(EventSourceQuery::all()))
            .await
            .unwrap()
            .into_iter()
            .filter_map(|instance| match instance {
                DiscoveryInstance::EventSource {
                    topic, metadata, ..
                } if topic == FRONTEND_ADMISSION_TOPIC => Some(metadata),
                _ => None,
            })
            .collect()
    }

    async fn wait_for_records(discovery: &dyn Discovery, expected: serde_json::Value) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while published_records(discovery).await != vec![expected.clone()] {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("expected admission record {expected}"));
    }

    #[tokio::test]
    async fn publisher_follows_routing_state_and_sync() {
        let discovery: Arc<dyn Discovery> = Arc::new(KVStoreDiscovery::new(
            kv::Manager::memory(),
            CancellationToken::new(),
        ));
        let manager = Arc::new(ModelManager::new());
        let (synced_tx, synced_rx) = watch::channel(false);
        let cancellation = CancellationToken::new();
        let publisher = tokio::spawn(run_admission_publisher(
            Arc::clone(&discovery),
            "frontend".to_string(),
            manager.clone(),
            synced_rx,
            cancellation.clone(),
        ));
        wait_for_records(
            discovery.as_ref(),
            serde_json::json!({"protocol": "v1", "synced": false, "members": {}}),
        )
        .await;

        // Discovery syncs, and a committed aggregated worker becomes routable.
        synced_tx.send_replace(true);
        let mut card = crate::model_card::ModelDeploymentCard::default();
        card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
        let (routable_tx, routable_rx) = watch::channel(vec![0x2a_u64]);
        let mut worker_set = super::super::WorkerSet::new("graph".into(), "mdc".into(), card);
        worker_set.set_instance_watcher(routable_rx);
        worker_set.set_endpoint_id(dynamo_runtime::protocols::EndpointId {
            namespace: "graph".to_string(),
            component: "backend".to_string(),
            name: "generate".to_string(),
        });
        assert!(manager.add_worker_set("model", "graph", worker_set));
        wait_for_records(
            discovery.as_ref(),
            serde_json::json!({
                "protocol": "v1",
                "synced": true,
                "members": {"graph/backend/generate": ["2a"]},
            }),
        )
        .await;

        // A worker the client can no longer route to leaves the record.
        routable_tx.send_replace(Vec::new());
        manager.notify_serving_state_changed();
        wait_for_records(
            discovery.as_ref(),
            serde_json::json!({"protocol": "v1", "synced": true, "members": {}}),
        )
        .await;

        cancellation.cancel();
        publisher.await.unwrap();
        assert!(published_records(discovery.as_ref()).await.is_empty());
    }

    /// A prefill hop whose activation the test controls.
    struct ActivatingHop(std::sync::atomic::AtomicBool);

    impl crate::kv_router::prefill_router::PrefillRouterLifecycle for ActivatingHop {
        fn set_target(&self, _: Option<crate::discovery::WorkerSetTarget>) {}

        fn awaiting_activation(&self) -> bool {
            self.0.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[tokio::test]
    async fn publisher_lists_a_disaggregated_generation_once_its_prefill_hop_activates() {
        use crate::worker_type::WorkerType;

        let discovery: Arc<dyn Discovery> = Arc::new(KVStoreDiscovery::new(
            kv::Manager::memory(),
            CancellationToken::new(),
        ));
        let manager = Arc::new(ModelManager::new());
        let (_synced_tx, synced_rx) = watch::channel(true);
        let cancellation = CancellationToken::new();
        let publisher = tokio::spawn(run_admission_publisher(
            Arc::clone(&discovery),
            "frontend".to_string(),
            manager.clone(),
            synced_rx,
            cancellation.clone(),
        ));

        // Commit a prefill and a decode WorkerSet whose prefill hop is still activating.
        let hop = Arc::new(ActivatingHop(std::sync::atomic::AtomicBool::new(true)));
        let mut routable = Vec::new();
        for (component, worker_type, needs, instance_id) in [
            ("prefill", WorkerType::Prefill, WorkerType::Decode, 1_u64),
            ("decode", WorkerType::Decode, WorkerType::Prefill, 2_u64),
        ] {
            let mut card = crate::model_card::ModelDeploymentCard::default();
            card.worker_type = Some(worker_type);
            card.needs = vec![vec![needs]];
            let (routable_tx, routable_rx) = watch::channel(vec![instance_id]);
            routable.push(routable_tx);
            let mut worker_set =
                super::super::WorkerSet::new("graph".into(), format!("{component}-mdc"), card);
            worker_set.set_instance_watcher(routable_rx);
            worker_set.set_endpoint_id(dynamo_runtime::protocols::EndpointId {
                namespace: "graph".to_string(),
                component: component.to_string(),
                name: "generate".to_string(),
            });
            if worker_type == WorkerType::Decode {
                worker_set.prefill_router = Some(hop.clone());
            }
            assert!(manager.add_worker_set("model", component, worker_set));
        }

        // Requests would skip remote prefill, so the generation is not listed.
        tokio::time::sleep(5 * ACTIVATION_POLL_INTERVAL).await;
        assert_eq!(
            published_records(discovery.as_ref()).await,
            vec![serde_json::json!({"protocol": "v1", "synced": true, "members": {}})]
        );

        // Activation completes in the background, without a change notification.
        hop.0.store(false, std::sync::atomic::Ordering::SeqCst);
        wait_for_records(
            discovery.as_ref(),
            serde_json::json!({
                "protocol": "v1",
                "synced": true,
                "members": {
                    "graph/decode/generate": ["2"],
                    "graph/prefill/generate": ["1"],
                },
            }),
        )
        .await;

        cancellation.cancel();
        publisher.await.unwrap();
    }

    /// The Dynamo operator reads these records from DynamoWorkerMetadata, whose
    /// `spec.data` is the serialized discovery metadata of one process. A golden
    /// file that the operator's tests also read pins the format on both sides;
    /// regenerate it with `UPDATE_GOLDEN=1`. Worker model cards have been
    /// serialized this way since v1.4.0.
    #[test]
    fn kubernetes_records_match_the_operator_contract() {
        use dynamo_runtime::discovery::{DiscoveryMetadata, ModelCardInstanceId};

        let card = |model_suffix: Option<&str>| ModelCardInstanceId {
            namespace: "graph-1".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 0x2a,
            model_suffix: model_suffix.map(str::to_string),
        };
        let mut worker = DiscoveryMetadata::new();
        for mcid in [card(None), card(Some("adapter"))] {
            worker
                .register_model_card(DiscoveryInstance::Model {
                    namespace: mcid.namespace,
                    component: mcid.component,
                    endpoint: mcid.endpoint,
                    instance_id: mcid.instance_id,
                    card_json: serde_json::json!({"display_name": "Qwen/Qwen3-0.6B"}),
                    model_suffix: mcid.model_suffix,
                })
                .unwrap();
        }
        let mut frontend = DiscoveryMetadata::new();
        let admissions = FrontendAdmissions {
            synced: true,
            members: RoutableMembers::from([(
                "graph-1/backend/generate".to_string(),
                [0x2a].into(),
            )]),
        };
        frontend
            .register_event_source(admission_source("graph", &admissions, 7).into_instance(1))
            .unwrap();
        let produced = serde_json::json!({"frontend": frontend, "worker": worker});

        let golden = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../deploy/operator/internal/workermetadata/testdata/discovery_records.json");
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            // Sort keys so regenerating the file does not reorder hash maps.
            fn sorted(value: serde_json::Value) -> serde_json::Value {
                match value {
                    serde_json::Value::Object(map) => map
                        .into_iter()
                        .map(|(key, value)| (key, sorted(value)))
                        .collect::<BTreeMap<_, _>>()
                        .into_iter()
                        .collect(),
                    other => other,
                }
            }
            let pretty = serde_json::to_string_pretty(&sorted(produced.clone())).unwrap();
            std::fs::write(&golden, pretty + "\n").unwrap();
        }
        let expected: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&golden).unwrap()).unwrap();
        assert_eq!(
            produced,
            expected,
            "the records changed; update the operator's reader and regenerate {} with UPDATE_GOLDEN=1",
            golden.display()
        );
    }

    /// A committed WorkerSet of `worker_type` at `{namespace}/{component}/generate`
    /// whose routable instances follow the returned sender.
    fn routable_worker_set(
        namespace: &str,
        component: &str,
        worker_type: crate::worker_type::WorkerType,
        needs: Vec<Vec<crate::worker_type::WorkerType>>,
        instance_ids: Vec<u64>,
    ) -> (super::super::WorkerSet, watch::Sender<Vec<u64>>) {
        let mut card = crate::model_card::ModelDeploymentCard::default();
        card.worker_type = Some(worker_type);
        card.needs = needs;
        let (routable_tx, routable_rx) = watch::channel(instance_ids);
        let mut worker_set =
            super::super::WorkerSet::new(namespace.into(), format!("{component}-mdc"), card);
        worker_set.set_instance_watcher(routable_rx);
        worker_set.set_endpoint_id(dynamo_runtime::protocols::EndpointId {
            namespace: namespace.to_string(),
            component: component.to_string(),
            name: "generate".to_string(),
        });
        (worker_set, routable_tx)
    }

    #[tokio::test(start_paused = true)]
    async fn publisher_writes_at_most_once_per_interval_under_churn() {
        use futures::{FutureExt, StreamExt};

        let discovery: Arc<dyn Discovery> = Arc::new(KVStoreDiscovery::new(
            kv::Manager::memory(),
            CancellationToken::new(),
        ));
        let mut records = discovery
            .list_and_watch(DiscoveryQuery::EventSources(EventSourceQuery::all()), None)
            .await
            .unwrap();
        let manager = Arc::new(ModelManager::new());
        let (worker_set, routable_tx) = routable_worker_set(
            "graph",
            "backend",
            crate::worker_type::WorkerType::Aggregated,
            Vec::new(),
            vec![1],
        );
        assert!(manager.add_worker_set("model", "backend", worker_set));
        let (_synced_tx, synced_rx) = watch::channel(true);
        let cancellation = CancellationToken::new();
        let publisher = tokio::spawn(run_admission_publisher(
            Arc::clone(&discovery),
            "frontend".to_string(),
            manager.clone(),
            synced_rx,
            cancellation.clone(),
        ));

        // Two seconds of churn: the routable instances change every 10 ms.
        let changes = 200;
        for change in 0..changes {
            routable_tx.send_replace(vec![1, 2 + change % 2]);
            manager.notify_serving_state_changed();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
        cancellation.cancel();
        publisher.await.unwrap();

        // Each write registers one record; passes start at most every interval.
        let mut writes = 0;
        while let Some(Some(event)) = records.next().now_or_never() {
            if matches!(
                event,
                Ok(dynamo_runtime::discovery::DiscoveryEvent::Added(_))
            ) {
                writes += 1;
            }
        }
        let bound = (Duration::from_secs(3).as_millis() / MIN_PUBLISH_INTERVAL.as_millis()) + 1;
        assert!(
            (2..=bound as usize).contains(&writes),
            "{writes} record writes for {changes} routing changes; expected at most {bound}"
        );
    }

    /// Measures what a publisher pass computes and writes, by cluster size. Run
    /// with `cargo test --release -p dynamo-llm --lib -- --ignored --nocapture
    /// admission_cost_by_cluster_size`.
    #[test]
    #[ignore = "measurement that prints a table"]
    fn admission_cost_by_cluster_size() {
        use crate::worker_type::WorkerType;

        println!("generations workers/role members  recompute/pass  record bytes");
        for (generations, workers_per_role) in [
            (2_u64, 50_u64),
            (2, 500),
            (2, 2_500),
            (2, 10_000),
            (500, 10),
        ] {
            let manager = ModelManager::new();
            let mut routable = Vec::new();
            for generation in 0..generations {
                let namespace = format!("dynamo-graph-{generation:08x}");
                for (component, worker_type, needs) in [
                    ("prefill", WorkerType::Prefill, WorkerType::Decode),
                    ("backend", WorkerType::Decode, WorkerType::Prefill),
                ] {
                    let instance_ids = (0..workers_per_role)
                        .map(|worker| 0x694d_a126_c857_0000 + (generation << 20) + (worker << 1))
                        .collect();
                    let (worker_set, routable_tx) = routable_worker_set(
                        &namespace,
                        component,
                        worker_type,
                        vec![vec![needs]],
                        instance_ids,
                    );
                    routable.push(routable_tx);
                    assert!(manager.add_worker_set(
                        "model",
                        &format!("{namespace}/{component}"),
                        worker_set
                    ));
                }
            }

            let passes = 20;
            let started = std::time::Instant::now();
            let mut serving = manager.serving_members();
            for _ in 1..passes {
                serving = manager.serving_members();
            }
            let per_pass = started.elapsed() / passes;
            let members: usize = serving.members.values().map(|ids| ids.len()).sum();
            let record = FrontendAdmissions {
                synced: true,
                members: serving.members,
            }
            .record();
            let bytes = serde_json::to_vec(&record).unwrap().len();
            println!(
                "{generations:>11} {workers_per_role:>12} {members:>7} {per_pass:>15.2?} {bytes:>13}"
            );
        }
    }

    #[test]
    fn large_records_are_published_less_often() {
        assert_eq!(publish_interval(0), MIN_PUBLISH_INTERVAL);
        assert_eq!(publish_interval(100 << 10), MIN_PUBLISH_INTERVAL);
        assert_eq!(publish_interval(2 << 20), Duration::from_secs(2));
    }

    #[tokio::test]
    async fn publisher_replaces_its_record_and_withdraws_it_on_shutdown() {
        let discovery: Arc<dyn Discovery> = Arc::new(KVStoreDiscovery::new(
            kv::Manager::memory(),
            CancellationToken::new(),
        ));
        let mut publisher = AdmissionPublisher::new(Arc::clone(&discovery), "frontend".into());

        // A frontend that has not synced yet still announces the protocol.
        let unsynced_bytes = r#"{"members":{},"protocol":"v1","synced":false}"#.len();
        assert_eq!(
            publisher.reconcile(&FrontendAdmissions::default()).await,
            Some(unsynced_bytes)
        );
        assert_eq!(
            published_records(discovery.as_ref()).await,
            vec![serde_json::json!({"protocol": "v1", "synced": false, "members": {}})]
        );

        // Each change replaces the one record.
        let synced = FrontendAdmissions {
            synced: true,
            members: RoutableMembers::from([("graph/backend/generate".to_string(), [1].into())]),
        };
        assert!(
            publisher
                .reconcile(&synced)
                .await
                .is_some_and(|bytes| bytes > 0)
        );
        assert_eq!(
            publisher.reconcile(&synced).await,
            Some(0),
            "unchanged admissions need no write"
        );
        assert_eq!(
            published_records(discovery.as_ref()).await,
            vec![serde_json::json!({
                "protocol": "v1",
                "synced": true,
                "members": {"graph/backend/generate": ["1"]},
            })]
        );

        publisher.withdraw().await;
        assert!(published_records(discovery.as_ref()).await.is_empty());
    }
}
