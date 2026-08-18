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
    /// Set once, under the `bindings` write lock, when the service is retired.
    /// Terminal: a retired service is never reused, only replaced.
    retired: AtomicBool,
    /// The endpoints this service registered in discovery. They must be stopped
    /// when the service retires, because `verify_service_topology` rejects a new
    /// mode for as long as the old mode's endpoints are still advertised.
    endpoints: parking_lot::Mutex<Vec<StartedEndpoint>>,
}

impl ServedIndexerService {
    async fn start(component: Component, mode: ServedIndexerMode) -> Result<Arc<Self>> {
        verify_service_topology(&component, mode).await?;

        let bindings = Arc::new(RwLock::new(HashMap::new()));
        let mut endpoints = vec![start_query_endpoint(component.clone(), bindings.clone()).await?];
        if mode == ServedIndexerMode::Approximate {
            match start_record_endpoint(component.clone(), bindings.clone()).await {
                Ok(endpoint) => endpoints.push(endpoint),
                Err(error) => {
                    // Leaving the query endpoint registered would make every later
                    // attempt on this component fail topology verification.
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

    /// Marks the service retired if it holds no model bindings, reporting whether
    /// it did.
    ///
    /// The flag is set under the same write lock that guards binding insertion, so
    /// a concurrent binder either observes the flag and retries against the
    /// replacement service, or wins the race and keeps this service alive.
    fn retire_if_unused(&self) -> bool {
        let bindings = self.bindings.write();
        if !bindings.is_empty() {
            return false;
        }
        self.retired.store(true, Ordering::SeqCst);
        true
    }

    /// Marks the service retired whether or not it still holds bindings.
    ///
    /// Only runtime teardown uses this. The endpoints are going away regardless of
    /// who is still bound, so a binder that cloned this `Arc` before cancellation
    /// fired must not be allowed to attach to it afterwards.
    fn mark_retired(&self) {
        let _bindings = self.bindings.write();
        self.retired.store(true, Ordering::SeqCst);
    }

    fn is_retired(&self) -> bool {
        self.retired.load(Ordering::SeqCst)
    }

    /// Stops every endpoint this service registered.
    ///
    /// Handles are taken one at a time rather than drained up front so that
    /// cancelling this future cannot strand the endpoints it has not reached yet:
    /// those stay owned by the service and a later call finishes them. The handle
    /// in flight when a cancellation lands is safe either way, because
    /// `StartedEndpoint::shutdown` cancels the endpoint's token before it awaits
    /// the endpoint task, so its cleanup runs whether or not anyone awaits it.
    async fn stop_endpoints(&self) {
        while let Some(endpoint) = self.take_endpoint() {
            shutdown_endpoint(endpoint).await;
        }
    }

    /// Separate from `stop_endpoints` so the lock guard is released by the time the
    /// caller awaits.
    fn take_endpoint(&self) -> Option<StartedEndpoint> {
        self.endpoints.lock().pop()
    }
}

/// Stops endpoints and waits for their discovery registrations to be withdrawn.
///
/// Awaiting matters: a spawned shutdown would leave a window in which discovery
/// still advertises the retired endpoints and a replacement service in the other
/// mode would be rejected.
///
/// How completely awaiting closes that window is backend-specific, and this is a
/// weaker guarantee than it looks. The KV-store backend answers
/// `DiscoveryClient::list` with a direct read, so an awaited unregistration is
/// visible to the next `verify_service_topology`. The Kubernetes backend answers
/// from an asynchronously refreshed watch snapshot, so it can still report the
/// retired endpoints for a short time afterwards and reject the incoming mode
/// once. That failure is transient rather than the wedge this change fixes: the
/// registry entry is already gone by then, so a retry succeeds instead of hitting
/// the same rejection forever.
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
        // Removing the binding is all that happens here. Retiring the service needs
        // to await endpoint shutdown, which a synchronous drop cannot do, and
        // spawning would panic when the last handle is dropped during runtime
        // teardown. Retirement is therefore deferred to `get_or_start_service`,
        // which already runs on an async path serialized by the creation lock.
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

    // A concurrent mode switch can retire the service between the lookup and the
    // binding insert. The retired flag is terminal for a given service and
    // `get_or_start_service` never hands back a service already carrying it, so the
    // retry lands on a live replacement; only a second mode switch inside the same
    // narrow window could retire that one too, and that is reported rather than
    // looped on.
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
    // The lock-free path is only valid when the cached mode is the one being asked
    // for *and* the entry is still live. A mismatch has to reach the slow path so it
    // can be retired, and so does a retired entry: retirement sets the flag before it
    // awaits endpoint shutdown, so the map advertises a retired service for the whole
    // of that window. Handing one back here would let a same-mode caller spin its
    // bounded retry against a service that can never accept a binding, and would wedge
    // the key permanently if the retiring task were cancelled before it removed it.
    if let Some(existing) = cached_service(&key)
        && existing.mode == mode
        && !existing.is_retired()
    {
        return Ok(existing);
    }

    let _guard = SERVICE_CREATION_LOCK.lock().await;
    if let Some(existing) = cached_service(&key) {
        if existing.mode == mode && !existing.is_retired() {
            return Ok(existing);
        }

        // A retired entry observed under the lock is a leftover: mode-switch retirement
        // runs while holding this lock, so no retirement of that kind is in flight. On
        // that path its bindings are empty by construction (`retire_if_unused` only sets
        // the flag over an empty map, and insertion refuses once it is set), so the call
        // below reports true and the entry is replaced rather than returned.
        //
        // Runtime teardown is the exception, and the reason this is not phrased as an
        // invariant: `mark_retired` sets the flag whether or not bindings remain. Such
        // an entry is removed from the registry immediately afterwards, so it is
        // visible here only for the width of that window, and if it does still hold
        // bindings the call below reports false and the caller is told about the
        // conflict rather than having a live service replaced underneath it.
        if !existing.retire_if_unused() {
            // Still in use by another router; the caller reports the conflict.
            return Ok(existing);
        }

        existing.stop_endpoints().await;
        // Bind the removed entry so the shard guard is released before the `Arc` is
        // dropped, and check identity so a service started by someone else under
        // this key is left alone.
        let removed =
            SERVED_INDEXER_SERVICES.remove_if(&key, |_, entry| Arc::ptr_eq(entry, &existing));
        drop(removed);
    }

    let service = ServedIndexerService::start(component.clone(), mode).await?;
    SERVED_INDEXER_SERVICES.insert(key.clone(), service.clone());
    spawn_teardown_eviction(&component, key, &service);
    Ok(service)
}

fn cached_service(key: &ServiceKey) -> Option<Arc<ServedIndexerService>> {
    SERVED_INDEXER_SERVICES
        .get(key)
        .map(|entry| Arc::clone(entry.value()))
}

/// Drops the registry entry when the owning runtime starts shutting down.
///
/// `child_token()` descends from the runtime's phase-1 endpoint shutdown token, so
/// this fires at the start of `Runtime::shutdown`. `primary_token()` would only
/// fire in phase 3, behind the graceful drain and its timeout.
///
/// The task holds a `Weak`, so a service retired earlier by a mode switch is not
/// kept alive by it, and the identity check stops it from evicting whatever
/// replaced that service under the same key.
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
        // Before the entry leaves the map, so the resurrection window the mode-switch
        // path closes is closed here too: a caller holding an `Arc` cloned just before
        // cancellation would otherwise bind to a service that is no longer registered
        // and whose endpoints are being torn down.
        service.mark_retired();
        let removed =
            SERVED_INDEXER_SERVICES.remove_if(&key, |_, entry| Arc::ptr_eq(entry, &service));
        drop(removed);
        service.stop_endpoints().await;
    });
}

async fn verify_service_topology(component: &Component, mode: ServedIndexerMode) -> Result<()> {
    let discovery = component.drt().discovery();
    let endpoints = discovery
        .list(DiscoveryQuery::ComponentEndpoints {
            namespace: component.namespace().name(),
            component: component.name().to_string(),
        })
        .await?;

    let mut query_instances = HashSet::new();
    let mut record_instances = HashSet::new();

    for endpoint in endpoints {
        let DiscoveryInstance::Endpoint(instance) = endpoint else {
            continue;
        };
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
                    component.namespace().name(),
                    component.name()
                );
            }
        }
        ServedIndexerMode::Approximate => {
            if !query_instances.is_empty() || !record_instances.is_empty() {
                anyhow::bail!(
                    "cannot start approximate served indexer on {}.{}: indexer endpoint already exists",
                    component.namespace().name(),
                    component.name()
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

    /// Builds an isolated in-process runtime plus a component whose namespace is
    /// unique to the caller, so registry and discovery state cannot leak between
    /// tests sharing this process.
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

    /// Shuts the runtime down and waits for teardown to finish.
    ///
    /// `DistributedRuntime::shutdown` only spawns the three-phase coordination task
    /// onto the runtime's own handle. Inside `#[tokio::test]` that handle is dropped
    /// the moment the test returns, so a bare `shutdown()` lets the test race its own
    /// cleanup: endpoint tasks never unregister and their sockets are left open in the
    /// process-wide ZMQ context for the rest of the test binary, which slows every
    /// later event-plane test in the same process. Awaiting the primary token means
    /// phase 3 has run before the test returns.
    async fn shutdown_and_settle(drt: DistributedRuntime) {
        let token = drt.primary_token();
        drt.shutdown();
        if tokio::time::timeout(Duration::from_secs(10), token.cancelled())
            .await
            .is_err()
        {
            panic!("runtime did not finish shutting down");
        }
        // Let the endpoint cleanup tasks woken by phase 3 close their sockets.
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

    /// The reported defect: once the final handle is dropped the service has no
    /// bindings, so a router rebuilt with the opposite `use_kv_events` setting must
    /// be able to claim the same component.
    ///
    /// Both transitions are exercised because they fail for different reasons.
    /// Evicting the registry entry alone fixes neither: `verify_service_topology`
    /// rejects `Approximate` while the retired query endpoint is still registered in
    /// discovery, and rejects `EventDriven` while the retired record endpoint is.
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

    /// A same-mode caller arriving mid-retirement must not be handed the retiring
    /// service.
    ///
    /// Retirement sets the retired flag and only then awaits endpoint shutdown, so
    /// there is a window — as long as two discovery round trips — in which the map
    /// still advertises a retired service under its original mode. A caller asking
    /// for that mode has to wait for the replacement instead of binding to the
    /// corpse. The same state outlives the window entirely if the retiring task is
    /// cancelled before it removes the key, and then nothing but an opposite-mode
    /// caller would ever clear it.
    ///
    /// The window is reproduced by leaving the registry in exactly the state the
    /// retiring task leaves it in while it is parked — flag set, key still present —
    /// rather than by racing a real mode switch, because a real interleaving cannot
    /// be pinned to that window without a production test hook.
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

    /// The guard must not degenerate into "always evict": while a model binding is
    /// live, a conflicting mode is still a genuine conflict.
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

        let error = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::Approximate,
            "model-b".to_string(),
            Indexer::None,
        )
        .await
        .err()
        .expect("a live event-driven binding must reject an approximate indexer");
        assert!(
            error.to_string().contains("cannot mix"),
            "unexpected error: {error}"
        );

        drop(handle);
        shutdown_and_settle(drt).await;
    }

    /// Dropping one of two concurrent users must not retire the shared service.
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

        let error = ensure_served_indexer_service(
            component.clone(),
            ServedIndexerMode::Approximate,
            "model-c".to_string(),
            Indexer::None,
        )
        .await
        .err()
        .expect("the surviving binding must keep the shared service in event-driven mode");
        assert!(
            error.to_string().contains("cannot mix"),
            "unexpected error: {error}"
        );

        drop(handle_b);
        shutdown_and_settle(drt).await;
    }

    /// The registry is process-global but keyed on the runtime's connection id, so
    /// nothing else ever reclaims an entry belonging to a stopped runtime.
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

        // See `shutdown_and_settle`: leaving teardown unfinished leaks sockets into
        // the process-wide ZMQ context, so an incomplete teardown fails this test
        // rather than silently slowing every later event-plane test in the binary.
        if tokio::time::timeout(Duration::from_secs(10), shutdown_complete.cancelled())
            .await
            .is_err()
        {
            panic!("runtime did not finish shutting down");
        }
        tokio::task::yield_now().await;
    }

    /// Interrupting endpoint teardown must not strand the endpoints it never reached.
    ///
    /// Draining every handle out of the service before the first await would leave a
    /// dropped teardown future's unreached endpoints cancelled by nobody, still
    /// advertised in discovery, and owned by nothing that could stop them later.
    #[tokio::test]
    async fn served_indexer_interrupted_teardown_retains_unreached_endpoints() {
        let _zmq_gate = crate::kv_router::indexer::ZMQ_TEST_ISOLATION.lock().await;
        let (drt, component) = registry_test_component("interrupted-teardown").await;
        let key = service_key(&component);

        // Approximate mode registers the record endpoint as well as the query one, so
        // there is a second handle for an interrupted run to leave behind.
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

        // Polls teardown exactly once, then drops it. One poll gets as far as awaiting
        // the first endpoint's task, which cannot have finished yet: this is a
        // current-thread runtime and the task has had no chance to run since its token
        // was cancelled. `tokio::time::timeout` is deliberately not used here — with a
        // zero duration it still yields to the runtime, which lets teardown run to
        // completion and makes the assertion below vacuous.
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

        // The interrupted attempt cost nothing: a second one still drains the rest.
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
