// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};

use anyhow::Result;
use dashmap::DashMap;
use dynamo_kv_router::indexer::{
    IndexerQueryRequest, IndexerQueryResponse, IndexerRecordRoutingDecisionRequest,
    IndexerRecordRoutingDecisionResponse, KV_INDEXER_QUERY_ENDPOINT,
    KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT,
};
use dynamo_kv_router::protocols::{LocalBlockHash, WorkerWithDpRank};
use dynamo_runtime::component::{Client, Component, StartedEndpoint};
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, RouterMode, SingleIn,
    async_trait, network::Ingress, network::egress::push_router::PushRouter,
};
use dynamo_runtime::stream;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_tokens::SequenceHash;
use futures::StreamExt;
use parking_lot::RwLock;
use tokio::sync::Mutex;

use crate::kv_router::metrics::RemoteIndexerMetrics;

use super::{Indexer, TieredMatchDetails};

pub struct RemoteIndexer {
    query_router: PushRouter<IndexerQueryRequest, IndexerQueryResponse>,
    query_client: Client,
    record_router: Option<
        PushRouter<IndexerRecordRoutingDecisionRequest, IndexerRecordRoutingDecisionResponse>,
    >,
    record_client: Client,
    component: Component,
    model_name: String,
    metrics: Arc<RemoteIndexerMetrics>,
    use_kv_events: bool,
}

impl RemoteIndexer {
    pub(super) async fn new(
        component: &Component,
        model_name: String,
        use_kv_events: bool,
    ) -> Result<Self> {
        let query_client = component
            .endpoint(KV_INDEXER_QUERY_ENDPOINT)
            .client()
            .await?;
        let query_router = PushRouter::from_client_no_fault_detection(
            query_client.clone(),
            RouterMode::RoundRobin,
        )
        .await?;
        let record_client = component
            .endpoint(KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT)
            .client()
            .await?;
        let record_router = if use_kv_events {
            None
        } else {
            Some(
                PushRouter::from_client_no_fault_detection(
                    record_client.clone(),
                    RouterMode::RoundRobin,
                )
                .await?,
            )
        };
        let metrics = RemoteIndexerMetrics::from_component(component);
        Ok(Self {
            query_router,
            query_client,
            record_router,
            record_client,
            component: component.clone(),
            model_name,
            metrics,
            use_kv_events,
        })
    }

    pub(super) async fn find_matches_by_tier(
        &self,
        block_hashes: Vec<LocalBlockHash>,
        device_only: bool,
    ) -> Result<TieredMatchDetails> {
        self.validate_topology_if_ready().await.inspect_err(|_| {
            self.metrics.increment_query_failures();
        })?;

        let request = IndexerQueryRequest {
            model_name: self.model_name.clone(),
            block_hashes,
            device_only,
        };
        let mut stream: ManyOut<IndexerQueryResponse> = self
            .query_router
            .round_robin(SingleIn::new(request))
            .await
            .inspect_err(|_| {
                self.metrics.increment_query_failures();
            })?;

        match stream.next().await {
            Some(IndexerQueryResponse::TieredScores(wire)) => Ok(wire.into()),
            Some(IndexerQueryResponse::Error(msg)) => {
                self.metrics.increment_query_failures();
                Err(anyhow::anyhow!("Remote indexer error: {}", msg))
            }
            None => {
                self.metrics.increment_query_failures();
                Err(anyhow::anyhow!("Remote indexer returned empty response"))
            }
        }
    }

    pub(super) async fn record_hashed_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<()> {
        self.validate_topology_if_ready().await.inspect_err(|_| {
            self.metrics.increment_write_failures();
        })?;

        let record_router = self.record_router.as_ref().ok_or_else(|| {
            self.metrics.increment_write_failures();
            anyhow::anyhow!("remote approximate indexer is not configured for writes")
        })?;
        let request = IndexerRecordRoutingDecisionRequest {
            model_name: self.model_name.clone(),
            worker,
            local_hashes,
            sequence_hashes,
        };
        let mut stream: ManyOut<IndexerRecordRoutingDecisionResponse> = record_router
            .round_robin(SingleIn::new(request))
            .await
            .inspect_err(|_| {
                self.metrics.increment_write_failures();
            })?;

        match stream.next().await {
            Some(IndexerRecordRoutingDecisionResponse::Recorded) => Ok(()),
            Some(IndexerRecordRoutingDecisionResponse::Error(msg)) => {
                self.metrics.increment_write_failures();
                Err(anyhow::anyhow!("Remote indexer write error: {}", msg))
            }
            None => {
                self.metrics.increment_write_failures();
                Err(anyhow::anyhow!(
                    "Remote indexer returned empty write response"
                ))
            }
        }
    }

    async fn validate_topology_if_ready(&self) -> Result<()> {
        let query_instances = cached_instance_ids(&self.query_client);
        let record_instances = cached_instance_ids(&self.record_client);

        if query_instances.is_empty() && record_instances.is_empty() {
            return Ok(());
        }

        if self.use_kv_events {
            if !record_instances.is_empty() {
                anyhow::bail!(
                    "remote indexer component {}.{} mixes event-driven and approximate endpoints",
                    self.component.namespace().name(),
                    self.component.name()
                );
            }
            return Ok(());
        }

        if query_instances.len() != 1 || record_instances.len() != 1 {
            anyhow::bail!(
                "approximate remote indexer component {}.{} must expose exactly one query endpoint and one record endpoint",
                self.component.namespace().name(),
                self.component.name()
            );
        }
        if query_instances != record_instances {
            anyhow::bail!(
                "approximate remote indexer component {}.{} must expose query and record endpoints from the same singleton instance",
                self.component.namespace().name(),
                self.component.name()
            );
        }

        Ok(())
    }

    pub(super) fn use_kv_events(&self) -> bool {
        self.use_kv_events
    }
}

fn cached_instance_ids(client: &Client) -> HashSet<u64> {
    client.instance_ids_avail().iter().copied().collect()
}

type ServiceKey = (u64, String, String);

static SERVED_INDEXER_SERVICES: LazyLock<DashMap<ServiceKey, Arc<ServedIndexerService>>> =
    LazyLock::new(DashMap::new);
static SERVICE_CREATION_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServedIndexerMode {
    EventDriven,
    Approximate,
}

impl ServedIndexerMode {
    pub fn from_use_kv_events(use_kv_events: bool) -> Self {
        if use_kv_events {
            Self::EventDriven
        } else {
            Self::Approximate
        }
    }

    fn topology_label(self) -> &'static str {
        match self {
            Self::EventDriven => "event-driven",
            Self::Approximate => "approximate",
        }
    }
}

struct ServedIndexerService {
    mode: ServedIndexerMode,
    bindings: Arc<RwLock<HashMap<String, Indexer>>>,
    // Set under the bindings lock and never cleared.
    retired: AtomicBool,
    endpoints: parking_lot::Mutex<Vec<StartedEndpoint>>,
}

impl ServedIndexerService {
    async fn start(
        component: Component,
        mode: ServedIndexerMode,
        ignored_instance_ids: &HashSet<u64>,
    ) -> Result<Arc<Self>> {
        verify_service_topology(&component, mode, ignored_instance_ids).await?;

        let bindings = Arc::new(RwLock::new(HashMap::new()));
        let mut endpoints = vec![start_query_endpoint(component.clone(), bindings.clone()).await?];
        if mode == ServedIndexerMode::Approximate {
            match start_record_endpoint(component.clone(), bindings.clone()).await {
                Ok(endpoint) => endpoints.push(endpoint),
                Err(error) => {
                    // A partial start must not leave the query endpoint registered.
                    shutdown_endpoints(endpoints).await;
                    return Err(error);
                }
            }
        }

        Ok(Arc::new(Self {
            mode,
            bindings,
            retired: AtomicBool::new(false),
            endpoints: parking_lot::Mutex::new(endpoints),
        }))
    }

    fn retire_if_unused(&self) -> bool {
        let bindings = self.bindings.write();
        if !bindings.is_empty() {
            return false;
        }
        self.retired.store(true, Ordering::SeqCst);
        true
    }

    fn mark_retired(&self) {
        let _bindings = self.bindings.write();
        self.retired.store(true, Ordering::SeqCst);
    }

    fn is_retired(&self) -> bool {
        self.retired.load(Ordering::SeqCst)
    }

    // Pop one at a time so cancellation leaves unreached endpoints owned by the service.
    async fn stop_endpoints(&self) -> HashSet<u64> {
        let mut instance_ids = HashSet::new();
        while let Some(endpoint) = self.take_endpoint() {
            instance_ids.insert(endpoint.instance().instance_id);
            shutdown_endpoint(endpoint).await;
        }
        instance_ids
    }

    fn take_endpoint(&self) -> Option<StartedEndpoint> {
        self.endpoints.lock().pop()
    }
}

async fn shutdown_endpoints(endpoints: Vec<StartedEndpoint>) {
    for endpoint in endpoints {
        shutdown_endpoint(endpoint).await;
    }
}

async fn shutdown_endpoint(endpoint: StartedEndpoint) {
    if let Err(error) = endpoint.shutdown().await {
        tracing::warn!(error = %error, "served indexer endpoint shutdown failed");
    }
}

pub struct ServedIndexerHandle {
    service: Arc<ServedIndexerService>,
    model_name: String,
}

impl Drop for ServedIndexerHandle {
    fn drop(&mut self) {
        // Endpoint retirement is deferred to the next async registry access.
        self.service.bindings.write().remove(&self.model_name);
    }
}

pub async fn ensure_served_indexer_service(
    component: Component,
    mode: ServedIndexerMode,
    model_name: String,
    indexer: Indexer,
) -> Result<ServedIndexerHandle> {
    enum BindOutcome {
        Bound,
        AlreadyRegistered,
        Retired,
    }

    // Retry once if a concurrent mode switch retires the selected service.
    for _ in 0..2 {
        let service = get_or_start_service(component.clone(), mode).await?;

        if service.mode != mode {
            anyhow::bail!(
                "cannot mix {} and {} served indexers under {}.{}",
                service.mode.topology_label(),
                mode.topology_label(),
                component.namespace().name(),
                component.name()
            );
        }

        let outcome = {
            let mut bindings = service.bindings.write();
            if service.is_retired() {
                BindOutcome::Retired
            } else if bindings.contains_key(&model_name) {
                BindOutcome::AlreadyRegistered
            } else {
                bindings.insert(model_name.clone(), indexer.clone());
                BindOutcome::Bound
            }
        };

        match outcome {
            BindOutcome::Bound => {
                return Ok(ServedIndexerHandle {
                    service,
                    model_name,
                });
            }
            BindOutcome::AlreadyRegistered => anyhow::bail!(
                "served indexer for model {} is already registered under {}.{}",
                model_name,
                component.namespace().name(),
                component.name(),
            ),
            BindOutcome::Retired => continue,
        }
    }

    anyhow::bail!(
        "served indexer service under {}.{} kept being retired while registering model {}",
        component.namespace().name(),
        component.name(),
        model_name
    )
}

async fn get_or_start_service(
    component: Component,
    mode: ServedIndexerMode,
) -> Result<Arc<ServedIndexerService>> {
    let key = service_key(&component);
    if let Some(existing) = cached_service(&key)
        && existing.mode == mode
        && !existing.is_retired()
    {
        return Ok(existing);
    }

    let _guard = SERVICE_CREATION_LOCK.lock().await;
    let mut ignored_instance_ids = HashSet::new();
    if let Some(existing) = cached_service(&key) {
        if existing.mode == mode && !existing.is_retired() {
            return Ok(existing);
        }

        // Replace only entries without live bindings.
        if !existing.retire_if_unused() {
            return Ok(existing);
        }

        ignored_instance_ids = existing.stop_endpoints().await;
        // Drop the removed entry after the DashMap shard guard is released.
        let removed =
            SERVED_INDEXER_SERVICES.remove_if(&key, |_, entry| Arc::ptr_eq(entry, &existing));
        drop(removed);
    }

    let service =
        ServedIndexerService::start(component.clone(), mode, &ignored_instance_ids).await?;
    SERVED_INDEXER_SERVICES.insert(key.clone(), service.clone());
    spawn_teardown_eviction(&component, key, &service);
    Ok(service)
}

fn cached_service(key: &ServiceKey) -> Option<Arc<ServedIndexerService>> {
    SERVED_INDEXER_SERVICES
        .get(key)
        .map(|entry| Arc::clone(entry.value()))
}

fn spawn_teardown_eviction(
    component: &Component,
    key: ServiceKey,
    service: &Arc<ServedIndexerService>,
) {
    let token = component.drt().child_token();
    let service = Arc::downgrade(service);
    tokio::spawn(async move {
        token.cancelled().await;
        let Some(service) = service.upgrade() else {
            return;
        };
        // Retire before removal so a cloned service cannot accept a new binding.
        service.mark_retired();
        let removed =
            SERVED_INDEXER_SERVICES.remove_if(&key, |_, entry| Arc::ptr_eq(entry, &service));
        drop(removed);
        service.stop_endpoints().await;
    });
}

async fn verify_service_topology(
    component: &Component,
    mode: ServedIndexerMode,
    ignored_instance_ids: &HashSet<u64>,
) -> Result<()> {
    let discovery = component.drt().discovery();
    let endpoints = discovery
        .list(DiscoveryQuery::ComponentEndpoints {
            namespace: component.namespace().name(),
            component: component.name().to_string(),
        })
        .await?;

    let namespace = component.namespace().name();
    validate_service_topology(
        &namespace,
        component.name(),
        mode,
        endpoints,
        ignored_instance_ids,
    )
}

fn validate_service_topology(
    namespace: &str,
    component: &str,
    mode: ServedIndexerMode,
    endpoints: impl IntoIterator<Item = DiscoveryInstance>,
    ignored_instance_ids: &HashSet<u64>,
) -> Result<()> {
    let mut query_instances = HashSet::new();
    let mut record_instances = HashSet::new();

    for endpoint in endpoints {
        let DiscoveryInstance::Endpoint(instance) = endpoint else {
            continue;
        };
        if ignored_instance_ids.contains(&instance.instance_id) {
            continue;
        }
        match instance.endpoint.as_str() {
            KV_INDEXER_QUERY_ENDPOINT => {
                query_instances.insert(instance.instance_id);
            }
            KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT => {
                record_instances.insert(instance.instance_id);
            }
            _ => {}
        }
    }

    match mode {
        ServedIndexerMode::EventDriven => {
            if !record_instances.is_empty() {
                anyhow::bail!(
                    "cannot start event-driven served indexer on {}.{}: approximate endpoint already exists",
                    namespace,
                    component
                );
            }
        }
        ServedIndexerMode::Approximate => {
            if !query_instances.is_empty() || !record_instances.is_empty() {
                anyhow::bail!(
                    "cannot start approximate served indexer on {}.{}: indexer endpoint already exists",
                    namespace,
                    component
                );
            }
        }
    }

    Ok(())
}

async fn start_query_endpoint(
    component: Component,
    bindings: Arc<RwLock<HashMap<String, Indexer>>>,
) -> Result<StartedEndpoint> {
    let engine = Arc::new(ServedIndexerQueryEngine { bindings });
    let ingress =
        Ingress::<SingleIn<IndexerQueryRequest>, ManyOut<IndexerQueryResponse>>::for_engine(
            engine,
        )?;
    component
        .endpoint(KV_INDEXER_QUERY_ENDPOINT)
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start_with_registration()
        .await
}

async fn start_record_endpoint(
    component: Component,
    bindings: Arc<RwLock<HashMap<String, Indexer>>>,
) -> Result<StartedEndpoint> {
    let engine = Arc::new(ServedIndexerRecordEngine { bindings });
    let ingress = Ingress::<
        SingleIn<IndexerRecordRoutingDecisionRequest>,
        ManyOut<IndexerRecordRoutingDecisionResponse>,
    >::for_engine(engine)?;
    component
        .endpoint(KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT)
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start_with_registration()
        .await
}

struct ServedIndexerQueryEngine {
    bindings: Arc<RwLock<HashMap<String, Indexer>>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<IndexerQueryRequest>, ManyOut<IndexerQueryResponse>, anyhow::Error>
    for ServedIndexerQueryEngine
{
    async fn generate(
        &self,
        request: SingleIn<IndexerQueryRequest>,
    ) -> Result<ManyOut<IndexerQueryResponse>> {
        let (request, ctx) = request.into_parts();
        let indexer = self.bindings.read().get(&request.model_name).cloned();

        let response = match indexer {
            Some(indexer) => {
                // Skip the per-tier walk when the caller only needs the device
                // overlap; saves server-side CPU and wire bytes.
                let result: Result<TieredMatchDetails, _> = if request.device_only {
                    indexer
                        .find_primary_match_details(request.block_hashes)
                        .await
                        .map(|device| TieredMatchDetails {
                            device,
                            lower_tier: HashMap::new(),
                        })
                } else {
                    indexer
                        .find_primary_matches_by_tier(request.block_hashes)
                        .await
                };
                match result {
                    Ok(tiered) => IndexerQueryResponse::TieredScores((&tiered).into()),
                    Err(error) => IndexerQueryResponse::Error(error.to_string()),
                }
            }
            None => IndexerQueryResponse::Error(format!(
                "served indexer model {} is not registered",
                request.model_name
            )),
        };

        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![response])),
            ctx.context(),
        ))
    }
}

struct ServedIndexerRecordEngine {
    bindings: Arc<RwLock<HashMap<String, Indexer>>>,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<IndexerRecordRoutingDecisionRequest>,
        ManyOut<IndexerRecordRoutingDecisionResponse>,
        anyhow::Error,
    > for ServedIndexerRecordEngine
{
    async fn generate(
        &self,
        request: SingleIn<IndexerRecordRoutingDecisionRequest>,
    ) -> Result<ManyOut<IndexerRecordRoutingDecisionResponse>> {
        let (request, ctx) = request.into_parts();
        let indexer = self.bindings.read().get(&request.model_name).cloned();

        let response = match indexer {
            Some(indexer) => match indexer
                .record_hashed_routing_decision(
                    request.worker,
                    request.local_hashes,
                    request.sequence_hashes,
                )
                .await
            {
                Ok(()) => IndexerRecordRoutingDecisionResponse::Recorded,
                Err(error) => IndexerRecordRoutingDecisionResponse::Error(error.to_string()),
            },
            None => IndexerRecordRoutingDecisionResponse::Error(format!(
                "served indexer model {} is not registered",
                request.model_name
            )),
        };

        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![response])),
            ctx.context(),
        ))
    }
}

fn service_key(component: &Component) -> ServiceKey {
    (
        component.drt().connection_id(),
        component.namespace().name(),
        component.name().to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::indexer::test_util::store_event;
    use crate::kv_router::indexer::{LowerTierIndexers, SideIndexer};
    use std::time::Duration;

    use dynamo_kv_router::indexer::{
        KvIndexer, KvIndexerInterface, KvIndexerMetrics, pruning::PruneConfig,
    };
    use dynamo_kv_router::protocols::{StorageTier, WorkerWithDpRank, compute_seq_hash_for_block};
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use tokio_util::sync::CancellationToken;

    async fn registry_test_component(name: &str) -> (DistributedRuntime, Component) {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let namespace = drt.namespace(format!("served-indexer-ns-{name}")).unwrap();
        let component = namespace
            .component(format!("served-indexer-component-{name}"))
            .unwrap();
        (drt, component)
    }

    async fn shutdown_and_settle(drt: DistributedRuntime) {
        let token = drt.primary_token();
        drt.shutdown();
        if tokio::time::timeout(Duration::from_secs(10), token.cancelled())
            .await
            .is_err()
        {
            panic!("runtime did not finish shutting down");
        }
        tokio::task::yield_now().await;
    }

    fn binding_count(key: &ServiceKey) -> usize {
        SERVED_INDEXER_SERVICES
            .get(key)
            .expect("service should still be registered while its runtime is alive")
            .bindings
            .read()
            .len()
    }

    fn discovered_indexer_endpoint(endpoint: &str, instance_id: u64) -> DiscoveryInstance {
        DiscoveryInstance::Endpoint(dynamo_runtime::component::Instance {
            namespace: "test".to_string(),
            component: "router".to_string(),
            endpoint: endpoint.to_string(),
            instance_id,
            transport: dynamo_runtime::component::TransportType::Nats(String::new()),
            device_type: None,
            request_plane_codec: None,
        })
    }

    #[test]
    fn topology_ignores_endpoints_from_the_retired_instance() {
        let stale_instance_id = 7;
        let endpoints = vec![
            discovered_indexer_endpoint(KV_INDEXER_QUERY_ENDPOINT, stale_instance_id),
            discovered_indexer_endpoint(
                KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT,
                stale_instance_id,
            ),
        ];
        let ignored = HashSet::from([stale_instance_id]);

        assert!(
            validate_service_topology(
                "test",
                "router",
                ServedIndexerMode::Approximate,
                endpoints.clone(),
                &ignored,
            )
            .is_ok()
        );

        let mut endpoints_with_other_instance = endpoints.clone();
        endpoints_with_other_instance.push(discovered_indexer_endpoint(
            KV_INDEXER_QUERY_ENDPOINT,
            stale_instance_id + 1,
        ));
        assert!(
            validate_service_topology(
                "test",
                "router",
                ServedIndexerMode::Approximate,
                endpoints_with_other_instance,
                &ignored,
            )
            .is_err()
        );
        assert!(
            validate_service_topology(
                "test",
                "router",
                ServedIndexerMode::Approximate,
                endpoints,
                &HashSet::new(),
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn served_indexer_retires_and_allows_mode_switch() {
        let _zmq_gate = crate::kv_router::indexer::ZMQ_TEST_ISOLATION.lock().await;
        let (drt, component) = registry_test_component("mode-switch").await;
        let key = service_key(&component);

        let handle = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::EventDriven,
            "model-a".to_string(),
            Indexer::None,
        )
        .await
        .expect("event-driven served indexer should start");

        assert_eq!(binding_count(&key), 1);
        drop(handle);
        assert_eq!(
            binding_count(&key),
            0,
            "dropping the final handle must remove its model binding"
        );

        let handle = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::Approximate,
            "model-a".to_string(),
            Indexer::None,
        )
        .await
        .expect("approximate served indexer should start once the event-driven one is retired");
        drop(handle);

        let handle = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::EventDriven,
            "model-a".to_string(),
            Indexer::None,
        )
        .await
        .expect("event-driven served indexer should start once the approximate one is retired");
        drop(handle);

        shutdown_and_settle(drt).await;
    }

    #[tokio::test]
    async fn served_indexer_mid_retirement_entry_is_not_handed_out() {
        let _zmq_gate = crate::kv_router::indexer::ZMQ_TEST_ISOLATION.lock().await;
        let (drt, component) = registry_test_component("mid-retirement").await;
        let key = service_key(&component);

        let handle = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::EventDriven,
            "model-a".to_string(),
            Indexer::None,
        )
        .await
        .expect("event-driven served indexer should start");
        drop(handle);

        let retiring = cached_service(&key).expect("service should still be registered");
        assert!(
            retiring.retire_if_unused(),
            "an unused service should retire"
        );

        let handle = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::EventDriven,
            "model-a".to_string(),
            Indexer::None,
        )
        .await
        .expect("a same-mode caller must be served by a replacement, not by the retired service");

        let replacement = cached_service(&key).expect("a replacement should be registered");
        assert!(
            !Arc::ptr_eq(&replacement, &retiring),
            "the retired service must have been replaced, not reused"
        );
        assert_eq!(binding_count(&key), 1);

        drop(handle);
        drop(retiring);
        shutdown_and_settle(drt).await;
    }

    #[tokio::test]
    async fn served_indexer_live_binding_rejects_conflicting_mode() {
        let _zmq_gate = crate::kv_router::indexer::ZMQ_TEST_ISOLATION.lock().await;
        let (drt, component) = registry_test_component("live-binding").await;

        let handle = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::EventDriven,
            "model-a".to_string(),
            Indexer::None,
        )
        .await
        .expect("event-driven served indexer should start");

        assert!(
            ensure_served_indexer_service(
                component.clone(),
                ServedIndexerMode::Approximate,
                "model-b".to_string(),
                Indexer::None,
            )
            .await
            .is_err(),
            "a live event-driven binding must reject an approximate indexer"
        );

        drop(handle);
        shutdown_and_settle(drt).await;
    }

    #[tokio::test]
    async fn served_indexer_survives_partial_handle_drop() {
        let _zmq_gate = crate::kv_router::indexer::ZMQ_TEST_ISOLATION.lock().await;
        let (drt, component) = registry_test_component("shared-service").await;
        let key = service_key(&component);

        let handle_a = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::EventDriven,
            "model-a".to_string(),
            Indexer::None,
        )
        .await
        .expect("first event-driven served indexer should start");
        let handle_b = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::EventDriven,
            "model-b".to_string(),
            Indexer::None,
        )
        .await
        .expect("second model should share the event-driven service");

        assert_eq!(binding_count(&key), 2);
        drop(handle_a);
        assert_eq!(
            binding_count(&key),
            1,
            "dropping one user must leave the other binding intact"
        );

        assert!(
            ensure_served_indexer_service(
                component.clone(),
                ServedIndexerMode::Approximate,
                "model-c".to_string(),
                Indexer::None,
            )
            .await
            .is_err(),
            "the surviving binding must keep the shared service in event-driven mode"
        );

        drop(handle_b);
        shutdown_and_settle(drt).await;
    }

    #[tokio::test]
    async fn served_indexer_registry_evicted_on_runtime_teardown() {
        let _zmq_gate = crate::kv_router::indexer::ZMQ_TEST_ISOLATION.lock().await;
        let (drt, component) = registry_test_component("teardown").await;
        let key = service_key(&component);

        let handle = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::EventDriven,
            "model-a".to_string(),
            Indexer::None,
        )
        .await
        .expect("event-driven served indexer should start");
        assert!(SERVED_INDEXER_SERVICES.contains_key(&key));

        let shutdown_complete = drt.primary_token();
        drt.shutdown();

        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while SERVED_INDEXER_SERVICES.contains_key(&key) {
            assert!(
                tokio::time::Instant::now() < deadline,
                "registry entry outlived its runtime"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        drop(handle);

        // Wait for phase-three socket cleanup before the next ZeroMQ test.
        if tokio::time::timeout(Duration::from_secs(10), shutdown_complete.cancelled())
            .await
            .is_err()
        {
            panic!("runtime did not finish shutting down");
        }
        tokio::task::yield_now().await;
    }

    #[tokio::test]
    async fn served_indexer_interrupted_teardown_retains_unreached_endpoints() {
        let _zmq_gate = crate::kv_router::indexer::ZMQ_TEST_ISOLATION.lock().await;
        let (drt, component) = registry_test_component("interrupted-teardown").await;
        let key = service_key(&component);

        let handle = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::Approximate,
            "model-a".to_string(),
            Indexer::None,
        )
        .await
        .expect("approximate served indexer should start");
        let service = cached_service(&key).expect("service should be registered");
        assert_eq!(service.endpoints.lock().len(), 2);

        // Poll once without yielding; a zero-duration timeout would let teardown finish.
        let first_poll = {
            let mut teardown = std::pin::pin!(service.stop_endpoints());
            let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
            std::future::Future::poll(teardown.as_mut(), &mut cx)
        };
        assert!(
            first_poll.is_pending(),
            "teardown of a live endpoint should not complete in a single poll"
        );
        assert!(
            !service.endpoints.lock().is_empty(),
            "an interrupted teardown must leave the unreached handles with the service"
        );

        service.stop_endpoints().await;
        assert!(
            service.endpoints.lock().is_empty(),
            "a completed teardown must own no endpoints"
        );

        drop(handle);
        shutdown_and_settle(drt).await;
    }

    #[tokio::test]
    async fn query_engine_supports_multiple_model_bindings() {
        let bindings = Arc::new(RwLock::new(HashMap::from([
            ("model-a".to_string(), Indexer::None),
            ("model-b".to_string(), Indexer::None),
        ])));
        let engine = ServedIndexerQueryEngine { bindings };
        let request = SingleIn::new(IndexerQueryRequest {
            model_name: "model-b".to_string(),
            block_hashes: vec![LocalBlockHash(1)],
            device_only: false,
        });

        let mut stream = engine.generate(request).await.unwrap();

        assert!(matches!(
            stream.next().await,
            Some(IndexerQueryResponse::TieredScores(_))
        ));
    }

    #[tokio::test]
    async fn query_engine_does_not_serve_side_indexer_matches() {
        let worker = WorkerWithDpRank::new(7, 0);
        let block_hashes = vec![LocalBlockHash(11), LocalBlockHash(12)];
        let sequence_hashes = compute_seq_hash_for_block(&block_hashes);
        let side = KvIndexer::new_with_pruning(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            Some(PruneConfig {
                ttl: Duration::from_secs(60),
            }),
        );
        side.process_routing_decision_with_hashes(worker, block_hashes.clone(), sequence_hashes)
            .await
            .unwrap();
        let _ = side.flush().await;

        let indexer = Indexer::KvIndexer {
            primary: KvIndexer::new(
                CancellationToken::new(),
                4,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            ),
            lower_tier: LowerTierIndexers::new(1, 4),
            approx: Some(SideIndexer::KvIndexer(side)),
            primary_records_routing_decisions: false,
        };

        assert_eq!(
            indexer
                .find_match_details(block_hashes.clone())
                .await
                .unwrap()
                .overlap_scores
                .scores
                .get(&worker)
                .copied(),
            Some(2),
            "local router queries should still use the side indexer"
        );

        let bindings = Arc::new(RwLock::new(HashMap::from([("m".to_string(), indexer)])));
        let engine = ServedIndexerQueryEngine { bindings };
        let mut stream = engine
            .generate(SingleIn::new(IndexerQueryRequest {
                model_name: "m".to_string(),
                block_hashes,
                device_only: false,
            }))
            .await
            .unwrap();

        let Some(IndexerQueryResponse::TieredScores(wire)) = stream.next().await else {
            panic!("expected TieredScores response");
        };

        assert!(
            wire.device.scores.iter().all(|(w, _)| *w != worker),
            "served indexer queries must expose only the primary indexer, got {:?}",
            wire.device.scores
        );
    }

    /// Verifies the served query engine returns a tiered payload with populated
    /// lower-tier hits, and that the wire value round-trips through the client
    /// conversion back to native `TieredMatchDetails`.
    #[tokio::test]
    async fn query_engine_returns_tiered_scores_with_lower_tier() {
        let worker = WorkerWithDpRank::new(7, 0);
        let indexer = Indexer::KvIndexer {
            primary: KvIndexer::new(
                CancellationToken::new(),
                4,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            ),
            lower_tier: LowerTierIndexers::new(1, 4),
            approx: None,
            primary_records_routing_decisions: false,
        };

        // Worker owns [11, 12] on device and [11, 12, 13] on host-pinned.
        indexer
            .apply_event(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;

        let Indexer::KvIndexer {
            primary,
            lower_tier,
            ..
        } = &indexer
        else {
            unreachable!()
        };
        let _ = primary.flush().await;
        for lt in lower_tier.all() {
            let _ = lt.dump_events().await.unwrap();
        }

        let bindings = Arc::new(RwLock::new(HashMap::from([("m".to_string(), indexer)])));
        let engine = ServedIndexerQueryEngine { bindings };
        let mut stream = engine
            .generate(SingleIn::new(IndexerQueryRequest {
                model_name: "m".to_string(),
                block_hashes: vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)],
                device_only: false,
            }))
            .await
            .unwrap();

        let Some(IndexerQueryResponse::TieredScores(wire)) = stream.next().await else {
            panic!("expected TieredScores response");
        };

        assert_eq!(
            wire.device
                .scores
                .iter()
                .find(|(w, _)| *w == worker)
                .map(|(_, s)| *s),
            Some(2),
            "device should report 2 overlap blocks"
        );
        let (_, host_hits) = wire
            .lower_tier
            .iter()
            .find(|(tier, _)| *tier == StorageTier::HostPinned)
            .expect("host-pinned tier should be present");
        assert_eq!(
            host_hits
                .hits
                .iter()
                .find(|(w, _)| *w == worker)
                .map(|(_, h)| *h),
            Some(1),
            "host-pinned should report 1 hit beyond device"
        );

        // Round-trip through the client conversion mirrors what RemoteIndexer does.
        let native: TieredMatchDetails = wire.into();
        assert_eq!(
            native.device.overlap_scores.scores.get(&worker).copied(),
            Some(2)
        );
        assert_eq!(
            native
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|d| d.hits.get(&worker).copied()),
            Some(1)
        );
    }

    /// `device_only=true` must skip the lower-tier walk: the response carries
    /// the device overlap but no per-tier entries, even when the underlying
    /// indexer holds host-pinned data.
    #[tokio::test]
    async fn query_engine_device_only_skips_lower_tiers() {
        let worker = WorkerWithDpRank::new(7, 0);
        let indexer = Indexer::KvIndexer {
            primary: KvIndexer::new(
                CancellationToken::new(),
                4,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            ),
            lower_tier: LowerTierIndexers::new(1, 4),
            approx: None,
            primary_records_routing_decisions: false,
        };

        indexer
            .apply_event(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;

        let Indexer::KvIndexer {
            primary,
            lower_tier,
            ..
        } = &indexer
        else {
            unreachable!()
        };
        let _ = primary.flush().await;
        for lt in lower_tier.all() {
            let _ = lt.dump_events().await.unwrap();
        }

        let bindings = Arc::new(RwLock::new(HashMap::from([("m".to_string(), indexer)])));
        let engine = ServedIndexerQueryEngine { bindings };
        let mut stream = engine
            .generate(SingleIn::new(IndexerQueryRequest {
                model_name: "m".to_string(),
                block_hashes: vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)],
                device_only: true,
            }))
            .await
            .unwrap();

        let Some(IndexerQueryResponse::TieredScores(wire)) = stream.next().await else {
            panic!("expected TieredScores response");
        };

        assert_eq!(
            wire.device
                .scores
                .iter()
                .find(|(w, _)| *w == worker)
                .map(|(_, s)| *s),
            Some(2),
            "device should still report 2 overlap blocks"
        );
        assert!(
            wire.lower_tier.is_empty(),
            "device_only=true must omit lower-tier entries, got {:?}",
            wire.lower_tier
        );
    }
}
