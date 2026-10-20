// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-observed membership. Engines publish facts; only the worker writes discovery.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::component::{Endpoint, Instance};
use dynamo_runtime::config::HealthStatus;
use dynamo_runtime::discovery::{Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec};
use dynamo_runtime::system_health::ReadinessHold;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use tokio::sync::{Mutex, watch};
use tokio_util::sync::CancellationToken;

/// A full observation of one engine incarnation, not a controller enable command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineServingState {
    pub instance_id: u64,
    pub revision: u64,
    pub ready: bool,
    /// Changes on reconnect even if engine incarnation/revision are unchanged.
    pub observation_epoch: u64,
    pub weight_version: Option<String>,
    /// Complete engine-owned taint set, projected before registering membership.
    pub taints: HashSet<String>,
}

/// `None` means unknown/disconnected and withdraws membership. Closing the sender
/// also withdraws membership. Engines must never mutate discovery themselves.
pub type EngineServingStates = watch::Receiver<Option<EngineServingState>>;

/// Discovery operations for the exact request-plane instance started by the worker.
pub(crate) struct EndpointMembership {
    endpoint: Endpoint,
    discovery: Arc<dyn Discovery>,
    instance: Instance,
    timeout: Duration,
}

impl EndpointMembership {
    pub(crate) fn new(endpoint: Endpoint, instance: &Instance, timeout: Duration) -> Self {
        Self {
            discovery: endpoint.drt().discovery(),
            endpoint,
            instance: instance.clone(),
            timeout,
        }
    }

    async fn register(&self) -> anyhow::Result<()> {
        let instance = &self.instance;
        self.discovery
            .register(DiscoverySpec::Endpoint {
                namespace: instance.namespace.clone(),
                component: instance.component.clone(),
                endpoint: instance.endpoint.clone(),
                transport: instance.transport.clone(),
                device_type: instance.device_type.clone(),
                request_plane_codec: instance.request_plane_codec,
            })
            .await?;
        Ok(())
    }

    async fn unregister(&self) -> anyhow::Result<()> {
        self.discovery
            .unregister(DiscoveryInstance::Endpoint(self.instance.clone()))
            .await
    }
}

pub(crate) async fn bounded_discovery<T>(
    timeout: Duration,
    operation: impl std::future::Future<Output = anyhow::Result<T>>,
) -> anyhow::Result<T> {
    tokio::time::timeout(timeout, operation)
        .await
        .map_err(|_| anyhow::anyhow!("discovery operation exceeded {timeout:?}; outcome unknown"))?
}

/// Runs only after the request handler and model card exist. A deferred endpoint
/// remains absent until the first eligible observation. The mutation lock is also
/// used by shutdown, so no local registration can start after final withdrawal.
pub(crate) async fn follow_engine_state(
    membership: EndpointMembership,
    mut states: EngineServingStates,
    mutation: Arc<Mutex<()>>,
    shutdown: CancellationToken,
    readiness_hold: ReadinessHold,
    admission: Option<Arc<crate::admission::ControllerAdmission>>,
    managed_prefix: Option<&'static str>,
) {
    let endpoint = &membership.endpoint;
    let mut registered_instance = None;
    let mut readiness_hold = Some(readiness_hold);
    let mut closed = false;
    let mut commands = admission.as_ref().map(|gate| gate.changes());
    loop {
        let result = {
            let _guard = tokio::select! {
                biased;
                _ = shutdown.cancelled() => return,
                guard = mutation.lock() => guard,
            };
            if shutdown.is_cancelled() {
                return;
            }
            let mut observation = if closed {
                None
            } else {
                states.borrow_and_update().clone()
            };
            if let Some(gate) = &admission
                && !gate.permits(observation.as_ref())
            {
                observation = None;
            }
            let result = tokio::select! {
                biased;
                _ = shutdown.cancelled() => Err(anyhow::anyhow!(
                    "discovery reconciliation cancelled; outcome unknown"
                )),
                result = bounded_discovery(membership.timeout, reconcile(
                    &membership,
                    &mut registered_instance,
                    &mut readiness_hold,
                    observation.as_ref(),
                    managed_prefix,
                )) => result,
            };
            if result.is_err() {
                registered_instance = None;
                readiness_hold.get_or_insert_with(|| {
                    ReadinessHold::take(endpoint.drt().system_health(), endpoint.name())
                });
                super::worker::set_worker_health(endpoint, HealthStatus::NotReady);
            }
            if let Some(gate) = &admission {
                gate.applied(
                    result.as_ref().ok().map(|_| registered_instance.is_some()),
                    result.as_ref().err().map(ToString::to_string),
                );
            }
            result
        };
        let retry = match result {
            Ok(()) => false,
            Err(error) => {
                // A failed RPC may have taken effect remotely. Forget the local
                // cache and force an unregister before attempting registration again.
                tracing::warn!(%error, "engine membership reconciliation failed; retrying");
                true
            }
        };
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => return,
            changed = states.changed(), if !closed => {
                closed = changed.is_err();
            }
            _ = async { match &mut commands {
                Some(commands) => { let _ = commands.changed().await; },
                None => std::future::pending().await,
            }} => {}
            _ = tokio::time::sleep(Duration::from_millis(250)), if retry => {}
        }
    }
}

async fn reconcile(
    membership: &EndpointMembership,
    registered_instance: &mut Option<(u64, HashSet<String>)>,
    readiness_hold: &mut Option<ReadinessHold>,
    observation: Option<&EngineServingState>,
    managed_prefix: Option<&str>,
) -> anyhow::Result<()> {
    let endpoint = &membership.endpoint;
    let eligible = observation.filter(|state| state.ready);
    if eligible.is_some_and(|state| {
        registered_instance
            .as_ref()
            .is_some_and(|(instance, taints)| {
                *instance == state.instance_id && taints == &state.taints
            })
    }) {
        return Ok(());
    }

    // Also withdraw on initial/unknown state and after an ambiguous register
    // failure. Idempotence is provided by the deferred-endpoint runtime primitive.
    readiness_hold.get_or_insert_with(|| {
        ReadinessHold::take(endpoint.drt().system_health(), endpoint.name())
    });
    super::worker::set_worker_health(endpoint, HealthStatus::NotReady);
    membership.unregister().await?;
    *registered_instance = None;
    if let Some(prefix) = managed_prefix {
        let empty = HashSet::new();
        project_taints(
            endpoint,
            prefix,
            eligible.map_or(&empty, |state| &state.taints),
        )
        .await?;
    }
    if let Some(state) = eligible {
        membership.register().await?;
        *registered_instance = Some((state.instance_id, state.taints.clone()));
        drop(readiness_hold.take());
        super::worker::set_worker_health(endpoint, HealthStatus::Ready);
    }
    Ok(())
}

/// Preserve unrelated labels and let the existing taint API regenerate topology.
/// The worker rejects competing full-set updates while this writer is active.
async fn project_taints(
    endpoint: &Endpoint,
    prefix: &str,
    owned: &HashSet<String>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        owned.iter().all(|taint| taint.starts_with(prefix)),
        "engine taint outside its owned prefix"
    );
    let id = endpoint.id();
    let models = endpoint
        .drt()
        .discovery()
        .list(DiscoveryQuery::EndpointModels {
            namespace: id.namespace,
            component: id.component,
            endpoint: id.name,
        })
        .await?;
    let card = models
        .iter()
        .find_map(|model| match model {
            DiscoveryInstance::Model {
                instance_id,
                model_suffix: None,
                card_json,
                ..
            } if *instance_id == endpoint.drt().connection_id() => Some(card_json),
            _ => None,
        })
        .ok_or_else(|| anyhow::anyhow!("engine-owned taints require the local base model card"))?;
    let mut taints: HashSet<String> = card["runtime_config"]["taints"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .filter(|taint| !taint.starts_with(prefix) && !taint.starts_with("dynamo.topology/"))
        .map(str::to_owned)
        .collect();
    taints.extend(owned.iter().cloned());
    dynamo_llm::local_model::update_model_taints(endpoint, taints).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::DiscoveryQuery,
        distributed::DistributedConfig,
        pipeline::{ManyOut, SingleIn, network::Ingress},
        protocols::annotated::Annotated,
        traits::DistributedRuntimeProvider,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    struct StalledDiscovery {
        inner: Arc<dyn Discovery>,
        publish_before_stall: bool,
        entered: Notify,
        release: Notify,
        dropped: Notify,
        completed: AtomicUsize,
    }

    struct NotifyOnDrop<'a>(&'a Notify);

    impl Drop for NotifyOnDrop<'_> {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }

    #[async_trait::async_trait]
    impl Discovery for StalledDiscovery {
        fn instance_id(&self) -> u64 {
            self.inner.instance_id()
        }

        async fn register_internal(
            &self,
            spec: DiscoverySpec,
        ) -> anyhow::Result<DiscoveryInstance> {
            let _dropped = NotifyOnDrop(&self.dropped);
            if self.publish_before_stall {
                self.inner.register(spec.clone()).await?;
            }
            self.entered.notify_one();
            self.release.notified().await;
            let instance = self.inner.register(spec).await?;
            self.completed.fetch_add(1, Ordering::SeqCst);
            Ok(instance)
        }

        async fn unregister(&self, instance: DiscoveryInstance) -> anyhow::Result<()> {
            self.inner.unregister(instance).await
        }

        async fn list(&self, query: DiscoveryQuery) -> anyhow::Result<Vec<DiscoveryInstance>> {
            self.inner.list(query).await
        }

        async fn list_and_watch(
            &self,
            query: DiscoveryQuery,
            cancel: Option<CancellationToken>,
        ) -> anyhow::Result<dynamo_runtime::discovery::DiscoveryStream> {
            self.inner.list_and_watch(query, cancel).await
        }
    }

    #[tokio::test]
    async fn stalled_registration_releases_shutdown_lock_and_withdraws_ambiguous_publication() {
        tokio::time::timeout(Duration::from_secs(5), async {
            for (cancel, publish_before_stall) in [(true, false), (true, true), (false, true)] {
                let runtime = Runtime::from_current().unwrap();
                let drt =
                    DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                        .await
                        .unwrap();
                let endpoint = drt
                    .namespace("stalled_registration")
                    .unwrap()
                    .component("backend")
                    .unwrap()
                    .endpoint("generate");
                let started = endpoint
                    .endpoint_builder()
                    .handler(Ingress::<SingleIn<String>, ManyOut<Annotated<String>>>::new())
                    .start_without_registration()
                    .await
                    .unwrap();
                let discovery = Arc::new(StalledDiscovery {
                    inner: drt.discovery(),
                    publish_before_stall,
                    entered: Notify::new(),
                    release: Notify::new(),
                    dropped: Notify::new(),
                    completed: AtomicUsize::new(0),
                });
                let mut membership = EndpointMembership::new(
                    endpoint.clone(),
                    started.instance(),
                    if cancel {
                        Duration::from_secs(30)
                    } else {
                        Duration::from_millis(20)
                    },
                );
                membership.discovery = discovery.clone();
                let (states, receiver) = watch::channel(Some(EngineServingState {
                    instance_id: 1,
                    revision: 1,
                    ready: true,
                    observation_epoch: 1,
                    weight_version: Some("1".into()),
                    taints: HashSet::new(),
                }));
                let mutation = Arc::new(Mutex::new(()));
                let shutdown = CancellationToken::new();
                let task = tokio::spawn(follow_engine_state(
                    membership,
                    receiver,
                    mutation.clone(),
                    shutdown.clone(),
                    ReadinessHold::take(drt.system_health(), endpoint.name()),
                    None,
                    None,
                ));
                discovery.entered.notified().await;
                assert!(mutation.try_lock().is_err());
                assert_eq!(count(&endpoint).await, usize::from(publish_before_stall));
                if cancel {
                    shutdown.cancel();
                }
                discovery.dropped.notified().await;
                if !cancel {
                    assert!(!drt.system_health().lock().get_health_status().0);
                    shutdown.cancel();
                }
                let guard = mutation.lock().await;
                endpoint.unregister_endpoint_instance().await.unwrap();
                drop(guard);
                task.await.unwrap();
                // Releasing the peer and sending a fresh observation must not
                // resume a detached local registration after final withdrawal.
                discovery.release.notify_one();
                states.send_modify(|state| state.as_mut().unwrap().revision += 1);
                tokio::task::yield_now().await;
                assert_eq!(discovery.completed.load(Ordering::SeqCst), 0);
                assert_eq!(count(&endpoint).await, 0);
                started.shutdown().await.unwrap();
                runtime.shutdown();
            }
        })
        .await
        .expect("stalled discovery must not block shutdown");
    }

    async fn count(endpoint: &Endpoint) -> usize {
        let id = endpoint.id();
        endpoint
            .drt()
            .discovery()
            .list(DiscoveryQuery::Endpoint {
                namespace: id.namespace,
                component: id.component,
                endpoint: id.name,
            })
            .await
            .unwrap()
            .len()
    }

    async fn wait_count(endpoint: &Endpoint, expected: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while count(endpoint).await != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn version_projection_preserves_other_labels_and_fails_closed() {
        use dynamo_runtime::discovery::DiscoverySpec;
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("observed_taints")
            .unwrap()
            .component("backend")
            .unwrap()
            .endpoint("generate");
        let started = endpoint
            .endpoint_builder()
            .handler(Ingress::<SingleIn<String>, ManyOut<Annotated<String>>>::new())
            .start_without_registration()
            .await
            .unwrap();
        let membership =
            EndpointMembership::new(endpoint.clone(), started.instance(), Duration::from_secs(1));
        let mut registered = None;
        let mut readiness = None;
        let mut state = EngineServingState {
            instance_id: 7,
            revision: 1,
            ready: true,
            observation_epoch: 1,
            weight_version: Some("1".into()),
            taints: HashSet::from(["dynamo.policy/version=run:1".into()]),
        };
        // Missing model metadata must never publish membership.
        assert!(
            reconcile(
                &membership,
                &mut registered,
                &mut readiness,
                Some(&state),
                Some("dynamo.policy/")
            )
            .await
            .is_err()
        );
        assert_eq!(count(&endpoint).await, 0);
        let id = endpoint.id();
        drt.discovery().register(DiscoverySpec::Model {
            namespace: id.namespace, component: id.component, endpoint: id.name,
            card_json: serde_json::json!({"display_name": "mock", "runtime_config": {
                "taints": ["capacity/fast", "dynamo.policy/version=stale", "dynamo.topology/zone=west"],
                "topology_domains": {"zone": "west"}
            }}), model_suffix: None,
        }).await.unwrap();
        for version in ["1", "2"] {
            state.revision += 1;
            let expected = format!("dynamo.policy/version=run:{version}");
            state.taints = HashSet::from([expected.clone()]);
            reconcile(
                &membership,
                &mut registered,
                &mut readiness,
                Some(&state),
                Some("dynamo.policy/"),
            )
            .await
            .unwrap();
            assert_eq!(count(&endpoint).await, 1);
            let labels = current_taints(&endpoint).await;
            assert_eq!(
                labels,
                HashSet::from([
                    expected,
                    "capacity/fast".into(),
                    "dynamo.topology/zone=west".into()
                ])
            );
        }
        // An unrelated revision does not churn discovery.
        state.revision += 1;
        reconcile(
            &membership,
            &mut registered,
            &mut readiness,
            Some(&state),
            Some("dynamo.policy/"),
        )
        .await
        .unwrap();
        assert_eq!(count(&endpoint).await, 1);
        state.ready = false;
        reconcile(
            &membership,
            &mut registered,
            &mut readiness,
            Some(&state),
            Some("dynamo.policy/"),
        )
        .await
        .unwrap();
        assert_eq!(count(&endpoint).await, 0);
        assert!(
            !current_taints(&endpoint)
                .await
                .iter()
                .any(|taint| taint.starts_with("dynamo.policy/"))
        );
        state.ready = true;
        reconcile(
            &membership,
            &mut registered,
            &mut readiness,
            Some(&state),
            Some("dynamo.policy/"),
        )
        .await
        .unwrap();
        reconcile(
            &membership,
            &mut registered,
            &mut readiness,
            None,
            Some("dynamo.policy/"),
        )
        .await
        .unwrap();
        assert_eq!(count(&endpoint).await, 0);
        assert!(
            !current_taints(&endpoint)
                .await
                .iter()
                .any(|taint| taint.starts_with("dynamo.policy/"))
        );
        state.taints = HashSet::from(["unowned/value".into()]);
        assert!(
            reconcile(
                &membership,
                &mut registered,
                &mut readiness,
                Some(&state),
                Some("dynamo.policy/")
            )
            .await
            .is_err()
        );
        assert_eq!(count(&endpoint).await, 0);
        started.shutdown().await.unwrap();
        runtime.shutdown();
    }

    async fn current_taints(endpoint: &Endpoint) -> HashSet<String> {
        let id = endpoint.id();
        let models = endpoint
            .drt()
            .discovery()
            .list(DiscoveryQuery::EndpointModels {
                namespace: id.namespace,
                component: id.component,
                endpoint: id.name,
            })
            .await
            .unwrap();
        let [DiscoveryInstance::Model { card_json, .. }] = models.as_slice() else {
            panic!("one model required");
        };
        card_json["runtime_config"]["taints"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect()
    }

    #[tokio::test]
    async fn observed_lifecycle_defers_withdraws_and_stops_before_shutdown() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("observed_lifecycle_test")
            .unwrap()
            .component("backend")
            .unwrap()
            .endpoint("generate");
        let started = endpoint
            .endpoint_builder()
            .handler(Ingress::<SingleIn<String>, ManyOut<Annotated<String>>>::new())
            .start_without_registration()
            .await
            .unwrap();
        let (tx, rx) = watch::channel(None);
        let shutdown = CancellationToken::new();
        let mutation = Arc::new(Mutex::new(()));
        let task = tokio::spawn(follow_engine_state(
            EndpointMembership::new(endpoint.clone(), started.instance(), Duration::from_secs(1)),
            rx,
            mutation.clone(),
            shutdown.clone(),
            ReadinessHold::take(endpoint.drt().system_health(), endpoint.name()),
            None,
            None,
        ));
        assert_eq!(count(&endpoint).await, 0);

        tx.send_replace(Some(EngineServingState {
            instance_id: 1,
            revision: 1,
            ready: false,
            observation_epoch: 1,
            weight_version: Some("1".into()),
            taints: HashSet::new(),
        }));
        assert_eq!(count(&endpoint).await, 0);
        tx.send_replace(Some(EngineServingState {
            instance_id: 1,
            revision: 2,
            ready: true,
            observation_epoch: 1,
            weight_version: Some("1".into()),
            taints: HashSet::new(),
        }));
        wait_count(&endpoint, 1).await;
        tx.send_replace(None);
        wait_count(&endpoint, 0).await;
        // A late successful transport/canary signal cannot make a withdrawn
        // observed worker ready while the membership owner retains its hold.
        {
            let health = endpoint.drt().system_health();
            let mut health = health.lock();
            health.set_endpoint_health_status(endpoint.name(), HealthStatus::Ready);
            health.set_health_status(HealthStatus::Ready);
            assert!(!health.get_health_status().0);
        }
        tx.send_replace(Some(EngineServingState {
            instance_id: 2,
            revision: 1,
            ready: true,
            observation_epoch: 1,
            weight_version: Some("1".into()),
            taints: HashSet::new(),
        }));
        wait_count(&endpoint, 1).await;
        tx.send_replace(Some(EngineServingState {
            instance_id: 2,
            revision: 2,
            ready: false,
            observation_epoch: 1,
            weight_version: Some("1".into()),
            taints: HashSet::new(),
        }));
        wait_count(&endpoint, 0).await;
        tx.send_replace(Some(EngineServingState {
            instance_id: 2,
            revision: 3,
            ready: true,
            observation_epoch: 1,
            weight_version: Some("1".into()),
            taints: HashSet::new(),
        }));
        wait_count(&endpoint, 1).await;

        shutdown.cancel();
        let guard = mutation.lock().await;
        endpoint.unregister_endpoint_instance().await.unwrap();
        drop(guard);
        tx.send_replace(Some(EngineServingState {
            instance_id: 3,
            revision: 1,
            ready: true,
            observation_epoch: 1,
            weight_version: Some("1".into()),
            taints: HashSet::new(),
        }));
        task.await.unwrap();
        assert_eq!(count(&endpoint).await, 0);
        started.shutdown().await.unwrap();
        runtime.shutdown();
    }

    #[tokio::test]
    async fn sender_exit_withdraws_a_registered_worker() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("observed_sender_exit")
            .unwrap()
            .component("backend")
            .unwrap()
            .endpoint("generate");
        let started = endpoint
            .endpoint_builder()
            .handler(Ingress::<SingleIn<String>, ManyOut<Annotated<String>>>::new())
            .start_without_registration()
            .await
            .unwrap();
        let (tx, rx) = watch::channel(Some(EngineServingState {
            instance_id: 1,
            revision: 1,
            ready: true,
            observation_epoch: 1,
            weight_version: Some("1".into()),
            taints: HashSet::new(),
        }));
        let shutdown = CancellationToken::new();
        let task = tokio::spawn(follow_engine_state(
            EndpointMembership::new(endpoint.clone(), started.instance(), Duration::from_secs(1)),
            rx,
            Arc::new(Mutex::new(())),
            shutdown.clone(),
            ReadinessHold::take(endpoint.drt().system_health(), endpoint.name()),
            None,
            None,
        ));
        wait_count(&endpoint, 1).await;
        drop(tx);
        wait_count(&endpoint, 0).await;
        shutdown.cancel();
        task.await.unwrap();
        started.shutdown().await.unwrap();
        runtime.shutdown();
    }
}
