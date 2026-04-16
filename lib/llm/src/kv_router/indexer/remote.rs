// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::Result;
use dashmap::DashMap;
use dynamo_kv_router::indexer::{
    IndexerQueryRequest, IndexerQueryResponse, IndexerRecordRoutingDecisionRequest,
    IndexerRecordRoutingDecisionResponse, KV_INDEXER_QUERY_ENDPOINT,
    KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT,
};
use dynamo_kv_router::protocols::{LocalBlockHash, OverlapScores, WorkerWithDpRank};
use dynamo_runtime::component::{Client, Component};
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

use super::Indexer;

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

    pub(super) async fn find_matches(
        &self,
        block_hashes: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores> {
        self.validate_topology_if_ready().await.inspect_err(|_| {
            self.metrics.increment_query_failures();
        })?;

        let request = IndexerQueryRequest {
            model_name: self.model_name.clone(),
            block_hashes,
        };
        let mut stream: ManyOut<IndexerQueryResponse> = self
            .query_router
            .round_robin(SingleIn::new(request))
            .await
            .inspect_err(|_| {
                self.metrics.increment_query_failures();
            })?;

        match stream.next().await {
            Some(IndexerQueryResponse::Scores(scores)) => Ok(scores.into()),
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
        ttl_override: Option<Duration>,
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
            ttl_ms: ttl_override.map(|d| d.as_millis() as u64),
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
}

impl ServedIndexerService {
    async fn start(component: Component, mode: ServedIndexerMode) -> Result<Arc<Self>> {
        verify_service_topology(&component, mode).await?;

        let bindings = Arc::new(RwLock::new(HashMap::new()));
        start_query_endpoint(component.clone(), bindings.clone())?;
        if mode == ServedIndexerMode::Approximate {
            start_record_endpoint(component.clone(), bindings.clone())?;
        }

        Ok(Arc::new(Self { mode, bindings }))
    }
}

pub struct ServedIndexerHandle {
    service: Arc<ServedIndexerService>,
    model_name: String,
}

impl Drop for ServedIndexerHandle {
    fn drop(&mut self) {
        self.service.bindings.write().remove(&self.model_name);
    }
}

pub async fn ensure_served_indexer_service(
    component: Component,
    mode: ServedIndexerMode,
    model_name: String,
    indexer: Indexer,
) -> Result<ServedIndexerHandle> {
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
        let mut bindings = service.bindings.write();
        if bindings.contains_key(&model_name) {
            anyhow::bail!(
                "served indexer for model {} is already registered under {}.{}",
                model_name,
                component.namespace().name(),
                component.name(),
            );
        }

        bindings.insert(model_name.clone(), indexer);
    }

    Ok(ServedIndexerHandle {
        service,
        model_name,
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
    bindings: Arc<RwLock<HashMap<String, Indexer>>>,
) -> Result<()> {
    let engine = Arc::new(ServedIndexerQueryEngine { bindings });
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
    bindings: Arc<RwLock<HashMap<String, Indexer>>>,
) -> Result<()> {
    let engine = Arc::new(ServedIndexerRecordEngine { bindings });
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
            Some(indexer) => match indexer.find_matches(request.block_hashes).await {
                Ok(scores) => IndexerQueryResponse::Scores(scores.into()),
                Err(error) => IndexerQueryResponse::Error(error.to_string()),
            },
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
                    request.ttl_ms.map(Duration::from_millis),
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
        });

        let mut stream = engine.generate(request).await.unwrap();

        assert!(matches!(
            stream.next().await,
            Some(IndexerQueryResponse::Scores(_))
        ));
    }
}
