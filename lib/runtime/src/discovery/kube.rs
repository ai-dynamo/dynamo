// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{CancellationToken, Result};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid;

use super::{Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, DiscoveryStream};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{
    Api, Client as KubeClient,
    runtime::{watcher, watcher::Config, reflector, WatchStreamExt},
};
use tokio::task::JoinHandle;

/// Snapshot of all discovered instances and their metadata
#[derive(Clone, Debug)]
struct MetadataSnapshot {
    /// Map of instance_id -> metadata
    instances: HashMap<u64, Arc<DiscoveryMetadata>>,
    /// Sequence number for debugging
    sequence: u64,
    /// Timestamp for observability
    timestamp: std::time::Instant,
}

impl MetadataSnapshot {
    fn empty() -> Self {
        Self {
            instances: HashMap::new(),
            sequence: 0,
            timestamp: std::time::Instant::now(),
        }
    }
}

/// Hash a pod name to get a consistent instance ID
pub fn hash_pod_name(pod_name: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    pod_name.hash(&mut hasher);
    hasher.finish()
}

/// Key for organizing metadata internally
/// Format: "namespace/component/endpoint"
fn make_endpoint_key(namespace: &str, component: &str, endpoint: &str) -> String {
    format!("{}/{}/{}", namespace, component, endpoint)
}

/// Metadata stored on each pod and exposed via HTTP endpoint
/// This struct holds all discovery registrations for this pod instance
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiscoveryMetadata {
    /// Registered endpoint instances (key: "namespace/component/endpoint")
    endpoints: HashMap<String, DiscoveryInstance>,
    /// Registered model card instances (key: "namespace/component/endpoint")
    model_cards: HashMap<String, DiscoveryInstance>,
}

impl DiscoveryMetadata {
    /// Create a new empty metadata store
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            model_cards: HashMap::new(),
        }
    }

    /// Register an endpoint instance
    pub fn register_endpoint(&mut self, instance: DiscoveryInstance) -> Result<()> {
        if let DiscoveryInstance::Endpoint(ref inst) = instance {
            let key = make_endpoint_key(&inst.namespace, &inst.component, &inst.endpoint);
            self.endpoints.insert(key, instance);
            Ok(())
        } else {
            crate::raise!("Cannot register non-endpoint instance as endpoint")
        }
    }

    /// Register a model card instance
    pub fn register_model_card(&mut self, instance: DiscoveryInstance) -> Result<()> {
        if let DiscoveryInstance::Model {
            ref namespace,
            ref component,
            ref endpoint,
            ..
        } = instance
        {
            let key = make_endpoint_key(namespace, component, endpoint);
            self.model_cards.insert(key, instance);
            Ok(())
        } else {
            crate::raise!("Cannot register non-model-card instance as model card")
        }
    }

    /// Get all registered endpoints
    pub fn get_all_endpoints(&self) -> Vec<DiscoveryInstance> {
        self.endpoints.values().cloned().collect()
    }

    /// Get all registered model cards
    pub fn get_all_model_cards(&self) -> Vec<DiscoveryInstance> {
        self.model_cards.values().cloned().collect()
    }

    /// Get all registered instances (endpoints and model cards)
    pub fn get_all(&self) -> Vec<DiscoveryInstance> {
        self.endpoints
            .values()
            .chain(self.model_cards.values())
            .cloned()
            .collect()
    }
}

impl Default for DiscoveryMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Cached metadata from a remote pod
struct CachedMetadata {
    metadata: Arc<DiscoveryMetadata>,
    pod_ip: String,
    fetched_at: std::time::Instant,
}

/// Pod information extracted from environment
#[derive(Debug, Clone)]
struct PodInfo {
    pod_name: String,
    pod_namespace: String,
    system_port: u16,
}

impl PodInfo {
    /// Discover pod information from environment variables
    fn from_env() -> Result<Self> {
        let pod_name = std::env::var("POD_NAME")
            .map_err(|_| crate::error!("POD_NAME environment variable not set"))?;
        
        let pod_namespace = std::env::var("POD_NAMESPACE")
            .unwrap_or_else(|_| {
                tracing::warn!("POD_NAMESPACE not set, defaulting to 'default'");
                "default".to_string()
            });
        
        // Read system server port from config
        let config = crate::config::RuntimeConfig::from_settings().unwrap_or_default();
        let system_port = config.system_port as u16;
        
        Ok(Self {
            pod_name,
            pod_namespace,
            system_port,
        })
    }
}

/// Discovery client implementation backed by Kubernetes EndpointSlices
#[derive(Clone)]
pub struct KubeDiscoveryClient {
    /// Instance ID derived from pod name hash
    instance_id: u64,
    /// Local pod's metadata (shared with system server)
    metadata: Arc<RwLock<DiscoveryMetadata>>,
    /// HTTP client for fetching remote metadata
    http_client: reqwest::Client,
    /// Cache of remote pod metadata (instance_id -> metadata)
    metadata_cache: Arc<RwLock<HashMap<u64, CachedMetadata>>>,
    /// Pod information
    pod_info: PodInfo,
    /// Cancellation token
    cancel_token: CancellationToken,
    /// Kubernetes client
    kube_client: KubeClient,
    /// Mock mode for testing (skips HTTP calls and returns mock metadata)
    mock_metadata: bool,
    /// Watch channel receiver for daemon broadcasts
    metadata_watch: tokio::sync::watch::Receiver<Arc<MetadataSnapshot>>,
    /// Daemon task handle for graceful shutdown
    daemon_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
}

impl KubeDiscoveryClient {
    /// Create a new Kubernetes discovery client
    /// 
    /// # Arguments
    /// * `metadata` - Shared metadata store (also used by system server)
    /// * `cancel_token` - Cancellation token for shutdown
    pub async fn new(
        metadata: Arc<RwLock<DiscoveryMetadata>>,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        let pod_info = PodInfo::from_env()?;
        let instance_id = hash_pod_name(&pod_info.pod_name);
        
        tracing::info!(
            "Initializing KubeDiscoveryClient: pod_name={}, instance_id={:x}, namespace={}",
            pod_info.pod_name,
            instance_id,
            pod_info.pod_namespace
        );
        
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| crate::error!("Failed to create HTTP client: {}", e))?;
        
        let kube_client = KubeClient::try_default()
            .await
            .map_err(|e| crate::error!("Failed to create Kubernetes client: {}", e))?;
        
        // Create watch channel with initial empty snapshot
        let (watch_tx, watch_rx) = tokio::sync::watch::channel(Arc::new(MetadataSnapshot::empty()));
        
        let client = Self {
            instance_id,
            metadata,
            http_client,
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            pod_info,
            cancel_token: cancel_token.clone(),
            kube_client,
            mock_metadata: false,
            metadata_watch: watch_rx,
            // why can't this simply be an option
            daemon_handle: Arc::new(RwLock::new(None)),
        };
        
        // Spawn daemon task
        let daemon_handle = tokio::spawn({
            let client = client.clone();
            async move {
                if let Err(e) = client.run_metadata_daemon(watch_tx).await {
                    tracing::error!("Metadata daemon failed: {}", e);
                }
            }
        });
        
        *client.daemon_handle.write().await = Some(daemon_handle);
        
        tracing::info!("Metadata daemon started");
        
        Ok(client)
    }

    /// Create a new client for testing (doesn't require environment variables)
    /// 
    /// This method is intended for testing only and bypasses the normal
    /// environment variable requirements. When `mock_metadata` is true,
    /// HTTP calls are skipped and mock metadata is returned.
    #[doc(hidden)]
    pub async fn new_for_testing(
        kube_client: KubeClient,
        pod_name: String,
        pod_namespace: String,
        mock_metadata: bool,
    ) -> Result<Self> {
        let instance_id = hash_pod_name(&pod_name);
        let metadata = Arc::new(RwLock::new(DiscoveryMetadata::new()));
        let cancel_token = CancellationToken::new();
        
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| crate::error!("Failed to create HTTP client: {}", e))?;
        
        let pod_info = PodInfo {
            pod_name,
            pod_namespace,
            system_port: 8080,
        };
        
        // Create watch channel with initial empty snapshot
        let (watch_tx, watch_rx) = tokio::sync::watch::channel(Arc::new(MetadataSnapshot::empty()));
        
        let client = Self {
            instance_id,
            metadata,
            http_client,
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            pod_info,
            cancel_token: cancel_token.clone(),
            kube_client,
            mock_metadata,
            metadata_watch: watch_rx,
            daemon_handle: Arc::new(RwLock::new(None)),
        };
        
        // Spawn daemon task
        let daemon_handle = tokio::spawn({
            let client = client.clone();
            async move {
                if let Err(e) = client.run_metadata_daemon(watch_tx).await {
                    tracing::error!("Metadata daemon failed: {}", e);
                }
            }
        });

        // what is this for?
        *client.daemon_handle.write().await = Some(daemon_handle);
        
        Ok(client)
    }

    /// Generate mock metadata for testing
    /// Returns a DiscoveryMetadata with one endpoint instance
    fn create_mock_metadata(pod_name: &str) -> DiscoveryMetadata {
        use crate::component::{Instance, TransportType};
        
        let mut metadata = DiscoveryMetadata::new();
        let instance_id = hash_pod_name(pod_name);
        
        // Create a mock endpoint instance
        let endpoint = DiscoveryInstance::Endpoint(Instance {
            namespace: "test-namespace".to_string(),
            component: "test-component".to_string(),
            endpoint: "test-endpoint".to_string(),
            instance_id,
            transport: TransportType::NatsTcp("nats://test:4222".to_string()),
        });
        
        // Ignore errors in mock data creation
        let _ = metadata.register_endpoint(endpoint);
        
        metadata
    }

    /// Get metadata for a remote pod, using cache if available
    async fn get_metadata(&self, pod_name: &str, pod_ip: &str) -> Result<Arc<DiscoveryMetadata>> {
        let instance_id = hash_pod_name(pod_name);
        
        // Mock mode: return mock metadata without HTTP calls
        if self.mock_metadata {
            tracing::debug!(
                "Mock mode: returning mock metadata for pod_name={}, instance_id={:x}",
                pod_name,
                instance_id
            );
            let metadata = Self::create_mock_metadata(pod_name);
            return Ok(Arc::new(metadata));
        }
        
        // Local test mode: parse port from pod name and use localhost
        let target_host = if std::env::var("DYN_LOCAL_KUBE_TEST").is_ok() {
            if let Some(port) = Self::parse_port_from_pod_name(pod_name) {
                tracing::trace!(
                    "Local test mode: using localhost:{} for pod {}",
                    port,
                    pod_name
                );
                format!("localhost:{}", port)
            } else {
                tracing::warn!(
                    "Local test mode enabled but couldn't parse port from pod name: {}",
                    pod_name
                );
                format!("{}:{}", pod_ip, self.pod_info.system_port)
            }
        } else {
            format!("{}:{}", pod_ip, self.pod_info.system_port)
        };
        
        // Fast path: check cache
        {
            let cache = self.metadata_cache.read().await;
            if let Some(cached) = cache.get(&instance_id) {
                tracing::trace!(
                    "Cache hit for pod_name={}, instance_id={:x}",
                    pod_name,
                    instance_id
                );
                return Ok(cached.metadata.clone());
            }
        }
        
        // Cache miss: fetch from remote pod
        tracing::info!(
            "Cache miss for pod_name={}, instance_id={:x}, fetching from {}",
            pod_name,
            instance_id,
            target_host
        );
        self.fetch_and_cache_from_host(instance_id, pod_name, &target_host).await
    }
    
    /// Parse port number from pod name (format: pod-name-<port>)
    /// Returns Some(port) if successfully parsed, None otherwise
    fn parse_port_from_pod_name(pod_name: &str) -> Option<u16> {
        // Split by '-' and try to parse the last segment as a port number
        pod_name.rsplit('-')
            .next()
            .and_then(|last| last.parse::<u16>().ok())
    }

    /// Fetch metadata from a remote pod and cache it
    async fn fetch_and_cache_from_host(
        &self,
        instance_id: u64,
        pod_name: &str,
        target_host: &str,
    ) -> Result<Arc<DiscoveryMetadata>> {
        let url = format!("http://{}/metadata", target_host);
        
        tracing::debug!("Fetching metadata from {}", url);
        
        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error!("Failed to fetch metadata from {}: {}", url, e))?;
        
        let metadata: DiscoveryMetadata = response
            .json()
            .await
            .map_err(|e| crate::error!("Failed to parse metadata from {}: {}", url, e))?;
        
        let metadata = Arc::new(metadata);
        
        // Store in cache
        {
            let mut cache = self.metadata_cache.write().await;
            // Check again in case another task inserted while we were fetching
            if let Some(existing) = cache.get(&instance_id) {
                tracing::debug!(
                    "Another task cached metadata for instance_id={:x} while we were fetching",
                    instance_id
                );
                return Ok(existing.metadata.clone());
            }
            
            cache.insert(
                instance_id,
                CachedMetadata {
                    metadata: metadata.clone(),
                    pod_ip: target_host.to_string(),
                    fetched_at: std::time::Instant::now(),
                },
            );
            
            tracing::debug!(
                "Cached metadata for pod_name={}, instance_id={:x}",
                pod_name,
                instance_id
            );
        }
        
        Ok(metadata)
    }

    /// Run the metadata daemon that maintains global discovery state
    async fn run_metadata_daemon(
        &self,
        watch_tx: tokio::sync::watch::Sender<Arc<MetadataSnapshot>>,
    ) -> Result<()> {
        use futures::StreamExt;
        
        tracing::info!("Metadata daemon starting");
        
        // Create reflector for ALL EndpointSlices in our namespace
        let endpoint_slices: Api<EndpointSlice> = Api::namespaced(
            self.kube_client.clone(),
            &self.pod_info.pod_namespace,
        );
        
        let (reader, writer) = reflector::store();
        
        // Apply label selector to only watch discovery-enabled EndpointSlices
        let watch_config = Config::default()
            .labels("dynamo.nvidia.com/discovery=enabled");
        
        tracing::info!("Daemon watching EndpointSlices with label: dynamo.nvidia.com/discovery=enabled");
        
        // Spawn reflector task (runs independently)
        let reflector_stream = reflector(writer, watcher(endpoint_slices, watch_config))
            .default_backoff()
            .touched_objects()
            .for_each(|res| {
                futures::future::ready(match res {
                    Ok(obj) => {
                        tracing::debug!(
                            slice_name = obj.metadata.name.as_deref().unwrap_or("unknown"),
                            "Daemon reflector updated EndpointSlice"
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Daemon reflector error: {}", e);
                    }
                })
            });
        
        tokio::spawn(reflector_stream);
        
        // Polling loop
        let mut sequence = 0u64;
        let mut prev_instance_ids: HashSet<u64> = HashSet::new();
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    match self.fetch_snapshot(&reader, sequence).await {
                        Ok(snapshot) => {
                            // Compare instance IDs to detect changes
                            // Assumption: metadata is immutable once set, so we only need to compare keys
                            let current_instance_ids: HashSet<u64> = 
                                snapshot.instances.keys().copied().collect();
                            
                            let instances_changed = current_instance_ids != prev_instance_ids;
                            
                            if instances_changed {
                                // Compute what was added and removed
                                let added: Vec<u64> = current_instance_ids
                                    .difference(&prev_instance_ids)
                                    .copied()
                                    .collect();
                                
                                let removed: Vec<u64> = prev_instance_ids
                                    .difference(&current_instance_ids)
                                    .copied()
                                    .collect();
                                
                                tracing::info!(
                                    "Daemon snapshot (seq={}): instances changed, total={}, added=[{}], removed=[{}]",
                                    sequence,
                                    current_instance_ids.len(),
                                    added.iter().map(|id| format!("{:x}", id)).collect::<Vec<_>>().join(", "),
                                    removed.iter().map(|id| format!("{:x}", id)).collect::<Vec<_>>().join(", ")
                                );
                                
                                // Prune cache for removed instances
                                if !removed.is_empty() {
                                    self.prune_cache_entries(&removed).await;
                                }
                                
                                // Broadcast the snapshot (only when changed)
                                if watch_tx.send(Arc::new(snapshot)).is_err() {
                                    tracing::debug!("No watch subscribers, daemon stopping");
                                    break;
                                }
                                
                                prev_instance_ids = current_instance_ids;
                            } else {
                                tracing::trace!(
                                    "Daemon snapshot (seq={}): no changes, {} instances",
                                    sequence,
                                    current_instance_ids.len()
                                );
                            }
                            
                            sequence += 1;
                        }
                        Err(e) => {
                            tracing::error!("Failed to fetch snapshot: {}", e);
                            // Continue on errors - don't crash daemon
                        }
                    }
                }
                _ = self.cancel_token.cancelled() => {
                    tracing::info!("Metadata daemon received cancellation");
                    break;
                }
            }
        }
        
        tracing::info!("Metadata daemon stopped");
        Ok(())
    }

    /// Fetch a complete snapshot of all instances and their metadata
    async fn fetch_snapshot(
        &self,
        reader: &reflector::Store<EndpointSlice>,
        sequence: u64,
    ) -> Result<MetadataSnapshot> {
        use futures::StreamExt;
        
        let start = std::time::Instant::now();
        
        // Extract ALL ready endpoints (instance_id, pod_name, pod_ip) directly from reflector
        // No need to clone EndpointSlices - just iterate and extract
        let all_endpoints: Vec<(u64, String, String)> = reader
            .state()
            .iter()
            .flat_map(|arc_slice| Self::extract_endpoint_info(arc_slice.as_ref()))
            .collect();
        
        tracing::trace!(
            "Daemon found {} ready endpoints to fetch",
            all_endpoints.len()
        );
        
        // Concurrent fetch: Fetch metadata for all endpoints in parallel
        let fetch_futures = all_endpoints.into_iter().map(|(instance_id, pod_name, pod_ip)| {
            let client = self.clone();
            async move {
                match client.get_metadata(&pod_name, &pod_ip).await {
                    Ok(metadata) => Some((instance_id, metadata)),
                    Err(e) => {
                        tracing::warn!(
                            "Failed to fetch metadata for pod {} (instance_id={:x}): {}",
                            pod_name,
                            instance_id,
                            e
                        );
                        None
                    }
                }
            }
        });
        
        // Execute fetches concurrently with bounded parallelism
        let results: Vec<_> = futures::stream::iter(fetch_futures)
            .buffer_unordered(20)
            .collect()
            .await;
        
        // Build the snapshot
        let mut instances = HashMap::new();
        for result in results {
            if let Some((instance_id, metadata)) = result {
                instances.insert(instance_id, metadata);
            }
        }
        
        let elapsed = start.elapsed();
        
        tracing::trace!(
            "Daemon snapshot complete (seq={}): {} instances in {:?}",
            sequence,
            instances.len(),
            elapsed
        );
        
        Ok(MetadataSnapshot {
            instances,
            sequence,
            timestamp: std::time::Instant::now(),
        })
    }

    /// Prune cache entries for removed instances
    async fn prune_cache_entries(&self, removed: &[u64]) {
        let mut cache = self.metadata_cache.write().await;
        for instance_id in removed {
            if cache.remove(instance_id).is_some() {
                tracing::debug!("Pruned cache for removed instance_id={:x}", instance_id);
            }
        }
    }

    /// Invalidate cache entry for a given instance
    async fn invalidate_cache(&self, instance_id: u64) {
        let mut cache = self.metadata_cache.write().await;
        if cache.remove(&instance_id).is_some() {
            tracing::debug!("Invalidated cache for instance_id={:x}", instance_id);
        }
    }

    /// Extract endpoint information from an EndpointSlice
    /// Returns (instance_id, pod_name, pod_ip) tuples for ready endpoints
    fn extract_endpoint_info(slice: &EndpointSlice) -> Vec<(u64, String, String)> {
        let mut result = Vec::new();
        
        let endpoints = &slice.endpoints;
        
        for endpoint in endpoints {
            // Check if endpoint is ready
            let is_ready = endpoint.conditions.as_ref()
                .and_then(|c| c.ready)
                .unwrap_or(false);
            
            if !is_ready {
                continue;
            }
            
            // Get pod name from targetRef
            let pod_name = match endpoint.target_ref.as_ref() {
                Some(target_ref) => target_ref.name.as_deref().unwrap_or(""),
                None => continue,
            };
            
            if pod_name.is_empty() {
                continue;
            }
            
            let instance_id = hash_pod_name(pod_name);
            
            // Get IP addresses
            for ip in &endpoint.addresses {
                result.push((instance_id, pod_name.to_string(), ip.clone()));
            }
        }
        
        result
    }

    /// Filter metadata instances by DiscoveryQuery
    fn filter_metadata(
        metadata: &DiscoveryMetadata,
        key: &DiscoveryQuery,
        _instance_id: u64,
    ) -> Vec<DiscoveryInstance> {
        let mut result = Vec::new();
        
        match key {
            DiscoveryQuery::AllEndpoints => {
                result.extend(metadata.get_all_endpoints());
            }
            DiscoveryQuery::NamespacedEndpoints { namespace } => {
                for instance in metadata.get_all_endpoints() {
                    if let DiscoveryInstance::Endpoint(ref inst) = instance {
                        if &inst.namespace == namespace {
                            result.push(instance);
                        }
                    }
                }
            }
            DiscoveryQuery::ComponentEndpoints { namespace, component } => {
                for instance in metadata.get_all_endpoints() {
                    if let DiscoveryInstance::Endpoint(ref inst) = instance {
                        if &inst.namespace == namespace && &inst.component == component {
                            result.push(instance);
                        }
                    }
                }
            }
            DiscoveryQuery::Endpoint { namespace, component, endpoint } => {
                for instance in metadata.get_all_endpoints() {
                    if let DiscoveryInstance::Endpoint(ref inst) = instance {
                        if &inst.namespace == namespace 
                            && &inst.component == component 
                            && &inst.endpoint == endpoint {
                            result.push(instance);
                        }
                    }
                }
            }
            DiscoveryQuery::AllModels => {
                result.extend(metadata.get_all_model_cards());
            }
            DiscoveryQuery::NamespacedModels { namespace } => {
                for instance in metadata.get_all_model_cards() {
                    if let DiscoveryInstance::Model { namespace: ns, .. } = &instance {
                        if ns == namespace {
                            result.push(instance);
                        }
                    }
                }
            }
            DiscoveryQuery::ComponentModels { namespace, component } => {
                for instance in metadata.get_all_model_cards() {
                    if let DiscoveryInstance::Model { 
                        namespace: ns, 
                        component: comp, 
                        .. 
                    } = &instance {
                        if ns == namespace && comp == component {
                            result.push(instance);
                        }
                    }
                }
            }
            DiscoveryQuery::EndpointModels { namespace, component, endpoint } => {
                for instance in metadata.get_all_model_cards() {
                    if let DiscoveryInstance::Model { 
                        namespace: ns, 
                        component: comp, 
                        endpoint: ep,
                        .. 
                    } = &instance {
                        if ns == namespace && comp == component && ep == endpoint {
                            result.push(instance);
                        }
                    }
                }
            }
        }
        
        result
    }
}

#[async_trait]
impl Discovery for KubeDiscoveryClient {
    fn instance_id(&self) -> u64 {
        self.instance_id
    }

    async fn register(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        let instance_id = self.instance_id();
        let instance = spec.with_instance_id(instance_id);
        
        tracing::debug!(
            "Registering instance: {:?} with instance_id={:x}",
            instance,
            instance_id
        );
        
        // Write to local metadata
        let mut metadata = self.metadata.write().await;
        match &instance {
            DiscoveryInstance::Endpoint(inst) => {
                tracing::info!(
                    "Registered endpoint: namespace={}, component={}, endpoint={}, instance_id={:x}",
                    inst.namespace,
                    inst.component,
                    inst.endpoint,
                    instance_id
                );
                metadata.register_endpoint(instance.clone())?;
            }
            DiscoveryInstance::Model {
                namespace,
                component,
                endpoint,
                ..
            } => {
                tracing::info!(
                    "Registered model card: namespace={}, component={}, endpoint={}, instance_id={:x}",
                    namespace,
                    component,
                    endpoint,
                    instance_id
                );
                metadata.register_model_card(instance.clone())?;
            }
        }
        
        Ok(instance)
    }

    async fn list(&self, key: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
        tracing::debug!("KubeDiscoveryClient::list called with key={:?}", key);
        
        // Wait for daemon to fetch at least once (skip initial empty snapshot)
        let snapshot = {
            let mut rx = self.metadata_watch.clone();
            let snapshot = rx.borrow_and_update().clone();
            
            // If we got the initial empty snapshot, wait for first real update
            if snapshot.sequence == 0 {
                tracing::debug!("Waiting for daemon to fetch first snapshot...");
                rx.changed().await
                    .map_err(|_| crate::error!("Daemon channel closed before first snapshot"))?;
                rx.borrow_and_update().clone()
            } else {
                snapshot
            }
        };
        
        tracing::debug!(
            "List using snapshot seq={} with {} instances",
            snapshot.sequence,
            snapshot.instances.len()
        );
        
        // Filter snapshot by query
        let mut instances = Vec::new();
        for (&instance_id, metadata) in &snapshot.instances {
            let filtered = Self::filter_metadata(metadata, &key, instance_id);
            instances.extend(filtered);
        }
        
        tracing::info!(
            "KubeDiscoveryClient::list returning {} instances for key={:?}",
            instances.len(),
            key
        );
        
        Ok(instances)
    }

    async fn list_and_watch(&self, key: DiscoveryQuery, _cancel_token: Option<CancellationToken>) -> Result<DiscoveryStream> {
        use tokio::sync::mpsc;

        tracing::info!(
            "KubeDiscoveryClient::list_and_watch started for key={:?}",
            key
        );
        
        // Clone the watch receiver
        let mut watch_rx = self.metadata_watch.clone();
        
        // Create output stream
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        
        // Generate unique stream identifier for tracing
        let stream_id = uuid::Uuid::new_v4();
        
        // Spawn task to process snapshots
        tokio::spawn(async move {
            let mut known_instances = HashSet::<u64>::new();
            
            tracing::debug!(
                stream_id = %stream_id,
                "Watch started for key={:?}",
                key
            );
            
            loop {
                // Wait for next snapshot
                match watch_rx.changed().await {
                    Ok(()) => {
                        // Get latest snapshot
                        let snapshot = watch_rx.borrow_and_update().clone();
                        
                        // Filter snapshot by query
                        let current_instances: HashSet<u64> = snapshot
                            .instances
                            .iter()
                            .filter_map(|(&instance_id, metadata)| {
                                let filtered = Self::filter_metadata(metadata, &key, instance_id);
                                if !filtered.is_empty() {
                                    Some(instance_id)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        
                        // Compute diff
                        let added: Vec<u64> = current_instances
                            .difference(&known_instances)
                            .copied()
                            .collect();
                        
                        let removed: Vec<u64> = known_instances
                            .difference(&current_instances)
                            .copied()
                            .collect();
                        
                        // Only log if there are changes
                        if !added.is_empty() || !removed.is_empty() {
                            tracing::debug!(
                                stream_id = %stream_id,
                                seq = snapshot.sequence,
                                added = added.len(),
                                removed = removed.len(),
                                total = current_instances.len(),
                                "Watch detected changes"
                            );
                        }
                        
                        // Emit Added events
                        for instance_id in added {
                            if let Some(metadata) = snapshot.instances.get(&instance_id) {
                                let instances = Self::filter_metadata(metadata, &key, instance_id);
                                for instance in instances {
                                    tracing::info!(
                                        stream_id = %stream_id,
                                        instance_id = format!("{:x}", instance.instance_id()),
                                        "Emitting Added event"
                                    );
                                    if event_tx.send(Ok(DiscoveryEvent::Added(instance))).is_err() {
                                        tracing::debug!(stream_id = %stream_id, "Watch receiver dropped");
                                        return;
                                    }
                                }
                            }
                        }
                        
                        // Emit Removed events
                        for instance_id in removed {
                            tracing::info!(
                                stream_id = %stream_id,
                                instance_id = format!("{:x}", instance_id),
                                "Emitting Removed event"
                            );
                            if event_tx.send(Ok(DiscoveryEvent::Removed(instance_id))).is_err() {
                                tracing::debug!(stream_id = %stream_id, "Watch receiver dropped");
                                return;
                            }
                        }
                        
                        // Update known set
                        known_instances = current_instances;
                    }
                    Err(_) => {
                        tracing::info!(
                            stream_id = %stream_id,
                            "Watch channel closed (daemon stopped)"
                        );
                        break;
                    }
                }
            }
        });
        
        // Convert receiver to stream
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(event_rx);
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::TransportType;

    #[test]
    fn test_hash_consistency() {
        let pod_name = "test-pod-123";
        let hash1 = hash_pod_name(pod_name);
        let hash2 = hash_pod_name(pod_name);
        assert_eq!(hash1, hash2, "Hash should be consistent");
    }

    #[test]
    fn test_hash_uniqueness() {
        let hash1 = hash_pod_name("pod-1");
        let hash2 = hash_pod_name("pod-2");
        assert_ne!(hash1, hash2, "Different pods should have different hashes");
    }

    #[test]
    fn test_metadata_serde() {
        let mut metadata = DiscoveryMetadata::new();
        
        // Add an endpoint
        let instance = DiscoveryInstance::Endpoint(crate::component::Instance {
            namespace: "test".to_string(),
            component: "comp1".to_string(),
            endpoint: "ep1".to_string(),
            instance_id: 123,
            transport: TransportType::NatsTcp("nats://localhost:4222".to_string()),
        });
        
        metadata.register_endpoint(instance).unwrap();
        
        // Serialize
        let json = serde_json::to_string(&metadata).unwrap();
        
        // Deserialize
        let deserialized: DiscoveryMetadata = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.endpoints.len(), 1);
        assert_eq!(deserialized.model_cards.len(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_registration() {
        let metadata = Arc::new(RwLock::new(DiscoveryMetadata::new()));
        
        // Spawn multiple tasks registering concurrently
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let metadata = metadata.clone();
                tokio::spawn(async move {
                    let mut meta = metadata.write().await;
                    let instance = DiscoveryInstance::Endpoint(crate::component::Instance {
                        namespace: "test".to_string(),
                        component: "comp1".to_string(),
                        endpoint: format!("ep{}", i),
                        instance_id: i,
                        transport: TransportType::NatsTcp("nats://localhost:4222".to_string()),
                    });
                    meta.register_endpoint(instance).unwrap();
                })
            })
            .collect();
        
        // Wait for all to complete
        for handle in handles {
            handle.await.unwrap();
        }
        
        // Verify all registrations succeeded
        let meta = metadata.read().await;
        assert_eq!(meta.endpoints.len(), 10);
    }

    #[test]
    fn test_endpoint_key() {
        let key1 = make_endpoint_key("ns1", "comp1", "ep1");
        let key2 = make_endpoint_key("ns1", "comp1", "ep1");
        let key3 = make_endpoint_key("ns1", "comp1", "ep2");
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert_eq!(key1, "ns1/comp1/ep1");
    }

    #[test]
    fn test_parse_port_from_pod_name() {
        // Valid port numbers
        assert_eq!(
            KubeDiscoveryClient::parse_port_from_pod_name("dynamo-test-worker-8080"),
            Some(8080)
        );
        assert_eq!(
            KubeDiscoveryClient::parse_port_from_pod_name("my-service-9000"),
            Some(9000)
        );
        assert_eq!(
            KubeDiscoveryClient::parse_port_from_pod_name("test-3000"),
            Some(3000)
        );
        assert_eq!(
            KubeDiscoveryClient::parse_port_from_pod_name("a-b-c-80"),
            Some(80)
        );
        
        // Invalid - no port number at end
        assert_eq!(
            KubeDiscoveryClient::parse_port_from_pod_name("dynamo-test-worker"),
            None
        );
        assert_eq!(
            KubeDiscoveryClient::parse_port_from_pod_name("8080-worker"),
            None  // Port at beginning, not end
        );
        assert_eq!(
            KubeDiscoveryClient::parse_port_from_pod_name("worker-abc"),
            None  // Not a number
        );
        assert_eq!(
            KubeDiscoveryClient::parse_port_from_pod_name(""),
            None  // Empty string
        );
    }

    #[tokio::test]
    async fn test_metadata_accessors() {
        let mut metadata = DiscoveryMetadata::new();
        
        // Register endpoints
        for i in 0..3 {
            let instance = DiscoveryInstance::Endpoint(crate::component::Instance {
                namespace: "test".to_string(),
                component: "comp1".to_string(),
                endpoint: format!("ep{}", i),
                instance_id: i,
                transport: TransportType::NatsTcp("nats://localhost:4222".to_string()),
            });
            metadata.register_endpoint(instance).unwrap();
        }
        
        // Register model cards
        for i in 0..2 {
            let instance = DiscoveryInstance::Model {
                namespace: "test".to_string(),
                component: "comp1".to_string(),
                endpoint: format!("ep{}", i),
                instance_id: i,
                card_json: serde_json::json!({"model": "test"}),
            };
            metadata.register_model_card(instance).unwrap();
        }
        
        assert_eq!(metadata.get_all_endpoints().len(), 3);
        assert_eq!(metadata.get_all_model_cards().len(), 2);
        assert_eq!(metadata.get_all().len(), 5);
    }
}

