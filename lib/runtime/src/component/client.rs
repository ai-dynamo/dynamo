// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::{collections::HashMap, time::Duration};

use anyhow::Result;
use arc_swap::ArcSwap;
use futures::StreamExt;
use tokio::net::unix::pipe::Receiver;

use crate::discovery::{DiscoveryEvent, DiscoveryInstance};
use crate::{
    component::{Endpoint, Instance},
    pipeline::async_trait,
    pipeline::{
        AddressedPushRouter, AddressedRequest, AsyncEngine, Data, ManyOut, PushRouter, RouterMode,
        SingleIn,
    },
    storage::key_value_store::{KeyValueStoreManager, WatchEvent},
    traits::DistributedRuntimeProvider,
    transports::etcd::Client as EtcdClient,
};

#[derive(Clone, Debug)]
pub struct Client {
    // This is me
    pub endpoint: Endpoint,
    // These are the remotes I know about from watching key-value store
    pub instance_source: Arc<tokio::sync::watch::Receiver<Vec<Instance>>>,
    // These are the instance source ids less those reported as down from sending rpc
    instance_avail: Arc<ArcSwap<Vec<u64>>>,
    // These are the instance source ids less those reported as busy (above threshold)
    instance_free: Arc<ArcSwap<Vec<u64>>>,
    // Watch sender for available instance IDs (for sending updates)
    instance_avail_tx: Arc<tokio::sync::watch::Sender<Vec<u64>>>,
    // Watch receiver for available instance IDs (for cloning to external subscribers)
    instance_avail_rx: tokio::sync::watch::Receiver<Vec<u64>>,
}

impl Client {
    // Client with auto-discover instances using key-value store
    pub(crate) async fn new(endpoint: Endpoint) -> Result<Self> {
        tracing::trace!(
            "Client::new_dynamic: Creating dynamic client for endpoint: {}",
            endpoint.path()
        );
        let instance_source = Self::get_or_create_dynamic_instance_source(&endpoint).await?;

        let (avail_tx, avail_rx) = tokio::sync::watch::channel(vec![]);
        let client = Client {
            endpoint: endpoint.clone(),
            instance_source: instance_source.clone(),
            instance_avail: Arc::new(ArcSwap::from(Arc::new(vec![]))),
            instance_free: Arc::new(ArcSwap::from(Arc::new(vec![]))),
            instance_avail_tx: Arc::new(avail_tx),
            instance_avail_rx: avail_rx,
        };
        client.monitor_instance_source();
        Ok(client)
    }

    pub fn path(&self) -> String {
        self.endpoint.path()
    }

    /// The root etcd path we watch in etcd to discover new instances to route to.
    pub fn etcd_root(&self) -> String {
        self.endpoint.etcd_root()
    }

    /// Instances available from watching key-value store
    pub fn instances(&self) -> Vec<Instance> {
        self.instance_source.borrow().clone()
    }

    pub fn instance_ids(&self) -> Vec<u64> {
        self.instances().into_iter().map(|ep| ep.id()).collect()
    }

    pub fn instance_ids_avail(&self) -> arc_swap::Guard<Arc<Vec<u64>>> {
        self.instance_avail.load()
    }

    pub fn instance_ids_free(&self) -> arc_swap::Guard<Arc<Vec<u64>>> {
        self.instance_free.load()
    }

    /// Get a watcher for available instance IDs
    pub fn instance_avail_watcher(&self) -> tokio::sync::watch::Receiver<Vec<u64>> {
        self.instance_avail_rx.clone()
    }

    /// Wait for at least one Instance to be available for this Endpoint
    pub async fn wait_for_instances(&self) -> Result<Vec<Instance>> {
        tracing::trace!(
            "wait_for_instances: Starting wait for endpoint: {}",
            self.endpoint.path()
        );
        let mut rx = self.instance_source.as_ref().clone();
        // wait for there to be 1 or more endpoints
        let mut instances: Vec<Instance>;
        loop {
            instances = rx.borrow_and_update().to_vec();
            if instances.is_empty() {
                rx.changed().await?;
            } else {
                tracing::info!(
                    "wait_for_instances: Found {} instance(s) for endpoint: {}",
                    instances.len(),
                    self.endpoint.path()
                );
                break;
            }
        }
        Ok(instances)
    }

    /// Mark an instance as down/unavailable
    pub fn report_instance_down(&self, instance_id: u64) {
        let filtered = self
            .instance_ids_avail()
            .iter()
            .filter_map(|&id| if id == instance_id { None } else { Some(id) })
            .collect::<Vec<_>>();
        self.instance_avail.store(Arc::new(filtered.clone()));

        // Notify watch channel subscribers about the change
        let _ = self.instance_avail_tx.send(filtered);

        tracing::debug!("inhibiting instance {instance_id}");
    }

    /// Update the set of free instances based on busy instance IDs
    pub fn update_free_instances(&self, busy_instance_ids: &[u64]) {
        let all_instance_ids = self.instance_ids();
        let free_ids: Vec<u64> = all_instance_ids
            .into_iter()
            .filter(|id| !busy_instance_ids.contains(id))
            .collect();
        self.instance_free.store(Arc::new(free_ids));
    }

    /// Monitor the key-value instance source and update instance_avail.
    fn monitor_instance_source(&self) {
        let cancel_token = self.endpoint.drt().primary_token();
        let client = self.clone();
        let endpoint_path = self.endpoint.path();
        tokio::task::spawn(async move {
            let mut rx = client.instance_source.as_ref().clone();
            while !cancel_token.is_cancelled() {
                let instance_ids: Vec<u64> = rx
                    .borrow_and_update()
                    .iter()
                    .map(|instance| instance.id())
                    .collect();

                // TODO: this resets both tracked available and free instances
                client.instance_avail.store(Arc::new(instance_ids.clone()));
                client.instance_free.store(Arc::new(instance_ids.clone()));

                // Send update to watch channel subscribers
                let _ = client.instance_avail_tx.send(instance_ids);

                if let Err(err) = rx.changed().await {
                    tracing::error!(
                        "monitor_instance_source: The Sender is dropped: {}, endpoint={}",
                        err,
                        endpoint_path
                    );
                    cancel_token.cancel();
                }
            }
        });
    }

    async fn get_or_create_dynamic_instance_source(
        endpoint: &Endpoint,
    ) -> Result<Arc<tokio::sync::watch::Receiver<Vec<Instance>>>> {
        let drt = endpoint.drt();
        let instance_sources = drt.instance_sources();
        let mut instance_sources = instance_sources.lock().await;

        if let Some(instance_source) = instance_sources.get(endpoint) {
            if let Some(instance_source) = instance_source.upgrade() {
                return Ok(instance_source);
            } else {
                instance_sources.remove(endpoint);
            }
        }

        let discovery = drt.discovery();
        let discovery_query = crate::discovery::DiscoveryQuery::Endpoint {
            namespace: endpoint.component.namespace.name.clone(),
            component: endpoint.component.name.clone(),
            endpoint: endpoint.name.clone(),
        };

        let mut discovery_stream = discovery
            .list_and_watch(discovery_query.clone(), None)
            .await?;
        let (watch_tx, watch_rx) = tokio::sync::watch::channel(vec![]);

        let secondary = endpoint.component.drt.runtime().secondary().clone();

        secondary.spawn(async move {
            tracing::trace!("endpoint_watcher: Starting for discovery query: {:?}", discovery_query);
            let mut map: HashMap<u64, Instance> = HashMap::new();

            loop {
                let discovery_event = tokio::select! {
                    _ = watch_tx.closed() => {
                        break;
                    }
                    discovery_event = discovery_stream.next() => {
                        match discovery_event {
                            Some(Ok(event)) => {
                                event
                            },
                            Some(Err(e)) => {
                                tracing::error!("endpoint_watcher: discovery stream error: {}; shutting down for discovery query: {:?}", e, discovery_query);
                                break;
                            }
                            None => {
                                break;
                            }
                        }
                    }
                };

                match discovery_event {
                    DiscoveryEvent::Added(discovery_instance) => {
                        if let DiscoveryInstance::Endpoint(instance) = discovery_instance {

                                map.insert(instance.instance_id, instance);
                        }
                    }
                    DiscoveryEvent::Removed(instance_id) => {
                        map.remove(&instance_id);
                    }
                }

                let instances: Vec<Instance> = map.values().cloned().collect();
                if watch_tx.send(instances).is_err() {
                    break;
                }
            }
            let _ = watch_tx.send(vec![]);
        });

        let instance_source = Arc::new(watch_rx);
        instance_sources.insert(endpoint.clone(), Arc::downgrade(&instance_source));
        Ok(instance_source)
    }
}
