// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use anyhow::Result;
use dashmap::DashMap;
use dynamo_kv_router::indexer::{
    IndexerQueryRequest, IndexerQueryResponse, IndexerRecordRoutingDecisionRequest,
    IndexerRecordRoutingDecisionResponse, KV_INDEXER_QUERY_ENDPOINT,
    KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT,
};
use dynamo_kv_router::protocols::{LocalBlockHash, OverlapScores, WorkerWithDpRank};
use dynamo_runtime::component::Component;
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

use super::Indexer;

pub struct RemoteIndexer {
    query_router: PushRouter<IndexerQueryRequest, IndexerQueryResponse>,
    record_router: Option<
        PushRouter<IndexerRecordRoutingDecisionRequest, IndexerRecordRoutingDecisionResponse>,
    >,
    component: Component,
    model_name: String,
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
        let query_router =
            PushRouter::from_client_no_fault_detection(query_client, RouterMode::RoundRobin)
                .await?;
        let record_router = if use_kv_events {
            None
        } else {
            let record_client = component
                .endpoint(KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT)
                .client()
                .await?;
            Some(
                PushRouter::from_client_no_fault_detection(record_client, RouterMode::RoundRobin)
                    .await?,
            )
        };
        Ok(Self {
            query_router,
            record_router,
            component: component.clone(),
            model_name,
            use_kv_events,
        })
    }

    pub(super) async fn find_matches(
        &self,
        block_hashes: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores> {
        self.validate_topology_if_ready().await?;

        let request = IndexerQueryRequest {
            model_name: self.model_name.clone(),
            block_hashes,
        };
        let mut stream: ManyOut<IndexerQueryResponse> = self
            .query_router
            .round_robin(SingleIn::new(request))
            .await?;

        match stream.next().await {
            Some(IndexerQueryResponse::Scores(scores)) => Ok(scores.into()),
            Some(IndexerQueryResponse::Error(msg)) => {
                Err(anyhow::anyhow!("Remote indexer error: {}", msg))
            }
            None => Err(anyhow::anyhow!("Remote indexer returned empty response")),
        }
    }

    pub(super) async fn record_hashed_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<()> {
        self.validate_topology_if_ready().await?;

        let record_router = self.record_router.as_ref().ok_or_else(|| {
            anyhow::anyhow!("remote approximate indexer is not configured for writes")
        })?;
        let request = IndexerRecordRoutingDecisionRequest {
            model_name: self.model_name.clone(),
            worker,
            local_hashes,
            sequence_hashes,
        };
        let mut stream: ManyOut<IndexerRecordRoutingDecisionResponse> =
            record_router.round_robin(SingleIn::new(request)).await?;

        match stream.next().await {
            Some(IndexerRecordRoutingDecisionResponse::Recorded) => Ok(()),
            Some(IndexerRecordRoutingDecisionResponse::Error(msg)) => {
                Err(anyhow::anyhow!("Remote indexer write error: {}", msg))
            }
            None => Err(anyhow::anyhow!(
                "Remote indexer returned empty write response"
            )),
        }
    }

    async fn validate_topology_if_ready(&self) -> Result<()> {
        let endpoints = self
            .component
            .drt()
            .discovery()
            .list(DiscoveryQuery::ComponentEndpoints {
                namespace: self.component.namespace().name(),
                component: self.component.name().to_string(),
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

#[derive(Clone)]
struct ServedIndexerBinding {
    model_name: String,
    indexer: Indexer,
}

struct ServedIndexerService {
    mode: ServedIndexerMode,
    binding: Arc<RwLock<Option<ServedIndexerBinding>>>,
}

impl ServedIndexerService {
    async fn start(component: Component, mode: ServedIndexerMode) -> Result<Arc<Self>> {
        verify_service_topology(&component, mode).await?;

        let binding = Arc::new(RwLock::new(None));
        start_query_endpoint(component.clone(), binding.clone())?;
        if mode == ServedIndexerMode::Approximate {
            start_record_endpoint(component.clone(), binding.clone())?;
        }

        Ok(Arc::new(Self { mode, binding }))
    }
}

pub struct ServedIndexerHandle {
    service_key: ServiceKey,
    service: Arc<ServedIndexerService>,
}

impl Drop for ServedIndexerHandle {
    fn drop(&mut self) {
        {
            let mut binding = self.service.binding.write();
            *binding = None;
        }

        if Arc::strong_count(&self.service) != 2 {
            return;
        }

        let should_remove = SERVED_INDEXER_SERVICES
            .get(&self.service_key)
            .is_some_and(|service| Arc::ptr_eq(service.value(), &self.service));
        if should_remove {
            SERVED_INDEXER_SERVICES.remove(&self.service_key);
        }
    }
}

pub async fn ensure_served_indexer_service(
    component: Component,
    mode: ServedIndexerMode,
    model_name: String,
    indexer: Indexer,
) -> Result<ServedIndexerHandle> {
    let service_key = service_key(&component);
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

    {
        let mut binding = service.binding.write();
        if binding.is_some() {
            anyhow::bail!(
                "served indexer is already registered under {}.{}",
                component.namespace().name(),
                component.name(),
            );
        }

        *binding = Some(ServedIndexerBinding {
            model_name: model_name.clone(),
            indexer,
        });
    }

    Ok(ServedIndexerHandle {
        service_key,
        service,
    })
}

async fn get_or_start_service(
    component: Component,
    mode: ServedIndexerMode,
) -> Result<Arc<ServedIndexerService>> {
    let key = service_key(&component);
    if let Some(existing) = SERVED_INDEXER_SERVICES.get(&key) {
        return Ok(existing.clone());
    }

    let _guard = SERVICE_CREATION_LOCK.lock().await;
    if let Some(existing) = SERVED_INDEXER_SERVICES.get(&key) {
        return Ok(existing.clone());
    }

    let service = ServedIndexerService::start(component, mode).await?;
    SERVED_INDEXER_SERVICES.insert(key, service.clone());
    Ok(service)
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

fn start_query_endpoint(
    component: Component,
    binding: Arc<RwLock<Option<ServedIndexerBinding>>>,
) -> Result<()> {
    let engine = Arc::new(ServedIndexerQueryEngine { binding });
    let ingress =
        Ingress::<SingleIn<IndexerQueryRequest>, ManyOut<IndexerQueryResponse>>::for_engine(
            engine,
        )?;
    tokio::spawn(async move {
        if let Err(error) = component
            .endpoint(KV_INDEXER_QUERY_ENDPOINT)
            .endpoint_builder()
            .handler(ingress)
            .graceful_shutdown(true)
            .start()
            .await
        {
            tracing::error!(error = %error, "served indexer query endpoint failed");
        }
    });
    Ok(())
}

fn start_record_endpoint(
    component: Component,
    binding: Arc<RwLock<Option<ServedIndexerBinding>>>,
) -> Result<()> {
    let engine = Arc::new(ServedIndexerRecordEngine { binding });
    let ingress = Ingress::<
        SingleIn<IndexerRecordRoutingDecisionRequest>,
        ManyOut<IndexerRecordRoutingDecisionResponse>,
    >::for_engine(engine)?;
    tokio::spawn(async move {
        if let Err(error) = component
            .endpoint(KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT)
            .endpoint_builder()
            .handler(ingress)
            .graceful_shutdown(true)
            .start()
            .await
        {
            tracing::error!(error = %error, "served indexer record endpoint failed");
        }
    });
    Ok(())
}

struct ServedIndexerQueryEngine {
    binding: Arc<RwLock<Option<ServedIndexerBinding>>>,
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
        let binding = self.binding.read().clone();

        let response = match binding {
            Some(binding) if binding.model_name == request.model_name => {
                match binding.indexer.find_matches(request.block_hashes).await {
                    Ok(scores) => IndexerQueryResponse::Scores(scores.into()),
                    Err(error) => IndexerQueryResponse::Error(error.to_string()),
                }
            }
            Some(binding) => IndexerQueryResponse::Error(format!(
                "served indexer model mismatch: requested={}, served={}",
                request.model_name, binding.model_name
            )),
            None => IndexerQueryResponse::Error("served indexer is not registered".to_string()),
        };

        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![response])),
            ctx.context(),
        ))
    }
}

struct ServedIndexerRecordEngine {
    binding: Arc<RwLock<Option<ServedIndexerBinding>>>,
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
        let binding = self.binding.read().clone();

        let response = match binding {
            Some(binding) if binding.model_name == request.model_name => match binding
                .indexer
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
            Some(binding) => IndexerRecordRoutingDecisionResponse::Error(format!(
                "served indexer model mismatch: requested={}, served={}",
                request.model_name, binding.model_name
            )),
            None => IndexerRecordRoutingDecisionResponse::Error(
                "served indexer is not registered".to_string(),
            ),
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
