// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic HTTP endpoint watcher for native HTTP paths.
//!
//! This watcher maintains a small, in-memory mapping from HTTP path -> set of
//! `EndpointId` and a cache of `EndpointId` -> `Client` (one per endpoint).
//! It consumes etcd watch events for instance records and updates the mapping
//! on PUT/DELETE. The HTTP hot path performs a read-only lookup to get Clients
//! and does not touch etcd.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::{RwLock, mpsc::Receiver};

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::Client;
use dynamo_runtime::component::Instance;
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::transports::etcd::WatchEvent;

fn normalize_path(path: &str) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{}", path)
    }
}

#[derive(Default)]
struct RegistryInner {
    // Only 1 entry per EndpointId
    paths: HashMap<String, HashSet<EndpointId>>,
    endpoint_clients: HashMap<EndpointId, Client>,
    // Maps etcd key to its (path, endpoint) for easier deletes
    instance_index: HashMap<String, (String, EndpointId)>,
}

#[derive(Clone)]
pub struct DynamicEndpointWatcher {
    drt: Option<DistributedRuntime>,
    inner: Arc<RwLock<RegistryInner>>,
}

impl DynamicEndpointWatcher {
    pub fn new(drt: Option<DistributedRuntime>) -> Self {
        Self {
            drt,
            inner: Arc::new(RwLock::new(RegistryInner::default())),
        }
    }

    pub async fn watch(&self, mut rx: Receiver<WatchEvent>) {
        while let Some(evt) = rx.recv().await {
            match evt {
                WatchEvent::Put(kv) => {
                    let key = match kv.key_str() {
                        Ok(k) => k.to_string(),
                        Err(e) => {
                            tracing::warn!("Invalid UTF-8 in instance key: {e:?}");
                            continue;
                        }
                    };
                    match serde_json::from_slice::<Instance>(kv.value()) {
                        Ok(instance) => {
                            if let Err(e) = self.add_instance(&key, instance).await {
                                tracing::warn!("Failed to process instance PUT: {e:?}");
                            }
                        }
                        Err(err) => {
                            tracing::warn!("Failed to parse instance on PUT: {err:?}");
                        }
                    }
                }
                WatchEvent::Delete(kv) => {
                    let key = match kv.key_str() {
                        Ok(k) => k.to_string(),
                        Err(e) => {
                            tracing::warn!("Invalid UTF-8 in instance key on DELETE: {e:?}");
                            continue;
                        }
                    };
                    self.remove_instance(&key).await;
                }
            }
        }
    }

    async fn ensure_client(&self, eid: &EndpointId) -> anyhow::Result<Client> {
        if let Some(c) = self.inner.read().await.endpoint_clients.get(eid) {
            return Ok(c.clone());
        }
        let drt = self
            .drt
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No DistributedRuntime available"))?;
        let ns = drt
            .namespace(eid.namespace.clone())
            .map_err(|e| anyhow::anyhow!("namespace(): {e}"))?;
        let comp = ns
            .component(eid.component.clone())
            .map_err(|e| anyhow::anyhow!("component(): {e}"))?;
        let ep = comp.endpoint(eid.name.clone());
        let client = ep.client().await?;
        // Ensure at least one instance is observed before publishing the client
        let _ = client.wait_for_instances().await?;
        self.inner
            .write()
            .await
            .endpoint_clients
            .insert(eid.clone(), client.clone());
        tracing::info!(
            path = %eid.as_url(),
            namespace = %eid.namespace,
            component = %eid.component,
            endpoint = %eid.name,
            "Dynamic HTTP endpoint client ready"
        );
        Ok(client)
    }

    async fn add_instance(&self, key: &str, instance: Instance) -> anyhow::Result<()> {
        let Some(path) = instance.http_endpoint_path.as_ref() else {
            // not a dynamic HTTP endpoint; ignore
            return Ok(());
        };
        let path = normalize_path(path);

        let endpoint_id = EndpointId {
            namespace: instance.namespace,
            component: instance.component,
            name: instance.endpoint,
        };

        let mut guard = self.inner.write().await;

        guard
            .instance_index
            .insert(key.to_string(), (path.clone(), endpoint_id.clone()));

        let set = guard.paths.entry(path.clone()).or_insert_with(HashSet::new);
        let inserted_new = set.insert(endpoint_id.clone());
        let need_client = inserted_new && !guard.endpoint_clients.contains_key(&endpoint_id);
        drop(guard);

        if need_client {
            if let Err(e) = self.ensure_client(&endpoint_id).await {
                tracing::warn!("Failed to create client for dynamic endpoint triple: {e:?}");
            }
            tracing::info!(
                http_path = %path,
                namespace = %endpoint_id.namespace,
                component = %endpoint_id.component,
                endpoint = %endpoint_id.name,
                "Registered dynamic HTTP endpoint path"
            );
        }

        Ok(())
    }

    async fn remove_instance(&self, key: &str) {
        let (_path, endpoint_id) = {
            let mut guard = self.inner.write().await;
            match guard.instance_index.remove(key) {
                Some(v) => {
                    if let Some(set) = guard.paths.get_mut(&v.0) {
                        set.remove(&v.1);
                        if set.is_empty() {
                            guard.paths.remove(&v.0);
                        }
                    }
                    v
                }
                None => return,
            }
        };

        let still_used = {
            let guard = self.inner.read().await;
            guard.paths.values().any(|set| set.contains(&endpoint_id))
        };
        if !still_used {
            let mut guard = self.inner.write().await;
            if guard.endpoint_clients.remove(&endpoint_id).is_some() {
                tracing::info!(
                    namespace = %endpoint_id.namespace,
                    component = %endpoint_id.component,
                    endpoint = %endpoint_id.name,
                    "Removed dynamic HTTP endpoint client"
                );
            }
        }
    }

    /// Get a cloned list of clients for a path. Returns None if the path is unknown.
    pub async fn get_clients(&self, path: &str) -> Option<Vec<Client>> {
        let path = normalize_path(path);
        let guard = self.inner.read().await;
        let triples: Vec<EndpointId> = guard
            .paths
            .get(&path)
            .map(|set| set.iter().cloned().collect())?;
        let clients = triples
            .into_iter()
            .filter_map(|t| guard.endpoint_clients.get(&t).cloned())
            .collect::<Vec<_>>();
        Some(clients)
    }
}
