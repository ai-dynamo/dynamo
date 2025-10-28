// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::pipeline::{
    AddressedPushRouter, AddressedRequest, AsyncEngine, Data, ManyOut, PushRouter, RouterMode,
    SingleIn,
};
use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::unix::pipe::Receiver;

use crate::{
    discovery::{DiscoveryClient, DiscoveryEvent, DiscoveryKey},
    pipeline::async_trait,
    transports::etcd::{Client as EtcdClient, WatchEvent},
};

use super::*;

/// Each state will be have a nonce associated with it
/// The state will be emitted in a watch channel, so we can observe the
/// critical state transitions.
enum MapState {
    /// The map is empty; value = nonce
    Empty(u64),

    /// The map is not-empty; values are (nonce, count)
    NonEmpty(u64, u64),

    /// The watcher has finished, no more events will be emitted
    Finished,
}

enum EndpointEvent {
    Put(String, u64),
    Delete(String),
}

#[derive(Clone, Debug)]
pub struct Client {
    // This is me
    pub endpoint: Endpoint,
    // These are the remotes I know about from watching etcd
    pub instance_source: Arc<InstanceSource>,
    // These are the instance source ids less those reported as down from sending rpc
    instance_avail: Arc<ArcSwap<Vec<u64>>>,
    // These are the instance source ids less those reported as busy (above threshold)
    instance_free: Arc<ArcSwap<Vec<u64>>>,
}

#[derive(Clone, Debug)]
pub enum InstanceSource {
    Static,
    Dynamic(tokio::sync::watch::Receiver<Vec<Instance>>),
}

impl Client {
    // Client will only talk to a single static endpoint
    pub(crate) async fn new_static(endpoint: Endpoint) -> Result<Self> {
        Ok(Client {
            endpoint,
            instance_source: Arc::new(InstanceSource::Static),
            instance_avail: Arc::new(ArcSwap::from(Arc::new(vec![]))),
            instance_free: Arc::new(ArcSwap::from(Arc::new(vec![]))),
        })
    }

    // Client with auto-discover instances using discovery client
    pub(crate) async fn new_dynamic(endpoint: Endpoint) -> Result<Self> {
        const INSTANCE_REFRESH_PERIOD: Duration = Duration::from_secs(1);

        // Get the discovery client from DRT
        let discovery_client = endpoint.component.drt.discovery_client().await?;

        let instance_source =
            Self::get_or_create_dynamic_instance_source(discovery_client, &endpoint).await?;

        let client = Client {
            endpoint,
            instance_source: instance_source.clone(),
            instance_avail: Arc::new(ArcSwap::from(Arc::new(vec![]))),
            instance_free: Arc::new(ArcSwap::from(Arc::new(vec![]))),
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

    /// Instances available from watching etcd
    pub fn instances(&self) -> Vec<Instance> {
        match self.instance_source.as_ref() {
            InstanceSource::Static => vec![],
            InstanceSource::Dynamic(watch_rx) => watch_rx.borrow().clone(),
        }
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

    /// Wait for at least one Instance to be available for this Endpoint
    pub async fn wait_for_instances(&self) -> Result<Vec<Instance>> {
        let mut instances: Vec<Instance> = vec![];
        if let InstanceSource::Dynamic(mut rx) = self.instance_source.as_ref().clone() {
            // wait for there to be 1 or more endpoints
            loop {
                instances = rx.borrow_and_update().to_vec();
                if instances.is_empty() {
                    rx.changed().await?;
                } else {
                    break;
                }
            }
        }
        Ok(instances)
    }

    /// Is this component know at startup and not discovered via etcd?
    pub fn is_static(&self) -> bool {
        matches!(self.instance_source.as_ref(), InstanceSource::Static)
    }

    /// Mark an instance as down/unavailable
    pub fn report_instance_down(&self, instance_id: u64) {
        let filtered = self
            .instance_ids_avail()
            .iter()
            .filter_map(|&id| if id == instance_id { None } else { Some(id) })
            .collect::<Vec<_>>();
        self.instance_avail.store(Arc::new(filtered));

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

    /// Monitor the ETCD instance source and update instance_avail.
    fn monitor_instance_source(&self) {
        let cancel_token = self.endpoint.drt().primary_token();
        let client = self.clone();
        tokio::task::spawn(async move {
            let mut rx = match client.instance_source.as_ref() {
                InstanceSource::Static => {
                    tracing::error!("Static instance source is not watchable");
                    return;
                }
                InstanceSource::Dynamic(rx) => rx.clone(),
            };
            while !cancel_token.is_cancelled() {
                let instance_ids: Vec<u64> = rx
                    .borrow_and_update()
                    .iter()
                    .map(|instance| instance.id())
                    .collect();

                // TODO: this resets both tracked available and free instances
                client.instance_avail.store(Arc::new(instance_ids.clone()));
                client.instance_free.store(Arc::new(instance_ids));

                tracing::debug!("instance source updated");

                if let Err(err) = rx.changed().await {
                    tracing::error!("The Sender is dropped: {}", err);
                    cancel_token.cancel();
                }
            }
        });
    }

    async fn get_or_create_dynamic_instance_source(
        discovery_client: Arc<dyn DiscoveryClient>,
        endpoint: &Endpoint,
    ) -> Result<Arc<InstanceSource>> {
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

        // Create discovery key from endpoint info
        let discovery_key = DiscoveryKey::Endpoint {
            namespace: endpoint.component.namespace.name.clone(),
            component: endpoint.component.name.clone(),
            endpoint: endpoint.name.clone(),
        };

        // Start list_and_watch on the discovery client
        let mut discovery_stream = discovery_client
            .list_and_watch(discovery_key.clone())
            .await?;

        let (watch_tx, watch_rx) = tokio::sync::watch::channel(vec![]);

        let secondary = endpoint.component.drt.runtime.secondary().clone();
        let endpoint_path = endpoint.path();

        // Spawn task to handle discovery events
        secondary.spawn(async move {
            tracing::debug!("Starting discovery watcher for endpoint: {}", endpoint_path);
            let mut map = HashMap::new();

            // Use StreamExt to iterate over the discovery stream
            use futures::StreamExt;

            loop {
                let discovery_event = tokio::select! {
                    _ = watch_tx.closed() => {
                        tracing::debug!("all watchers have closed; shutting down discovery watcher for endpoint: {endpoint_path}");
                        break;
                    }
                    event = discovery_stream.next() => {
                        match event {
                            Some(Ok(event)) => event,
                            Some(Err(e)) => {
                                tracing::error!("discovery stream error: {}; shutting down discovery watcher for endpoint: {endpoint_path}", e);
                                break;
                            }
                            None => {
                                tracing::debug!("discovery stream closed; shutting down discovery watcher for endpoint: {endpoint_path}");
                                break;
                            }
                        }
                    }
                };

                match discovery_event {
                    DiscoveryEvent::Added(discovery_instance) => {
                        // Extract fields from discovery instance
                        let (namespace, component, endpoint_name, instance_id) = match &discovery_instance {
                            crate::discovery::DiscoveryInstance::Endpoint {
                                namespace,
                                component,
                                endpoint,
                                instance_id
                            } => (
                                namespace.clone(),
                                component.clone(),
                                endpoint.clone(),
                                *instance_id
                            ),
                        };

                        // For now, create a test Instance with NatsTcp transport
                        // TODO: Discovery should store full Instance details, not just metadata
                        let instance = Instance {
                            component,
                            endpoint: endpoint_name,
                            namespace,
                            instance_id,
                            transport: TransportType::NatsTcp("".to_string()),
                        };

                        tracing::debug!(
                            instance_id = %instance.id(),
                            endpoint = %endpoint_path,
                            "Discovery: Added instance"
                        );

                        // Use instance_id as the key in the HashMap
                        map.insert(instance_id.to_string(), instance);
                    }
                    DiscoveryEvent::Removed(instance_id) => {
                        tracing::debug!(
                            instance_id = %instance_id,
                            endpoint = %endpoint_path,
                            "Discovery: Removed instance"
                        );
                        map.remove(&instance_id.to_string());
                    }
                }

                // Send complete snapshot of current instances
                let instances: Vec<Instance> = map.values().cloned().collect();

                if watch_tx.send(instances).is_err() {
                    tracing::debug!("Unable to send watch updates; shutting down discovery watcher for endpoint: {}", endpoint_path);
                    break;
                }
            }

            tracing::debug!("Completed discovery watcher for endpoint: {endpoint_path}");
            let _ = watch_tx.send(vec![]);
        });

        let instance_source = Arc::new(InstanceSource::Dynamic(watch_rx));
        instance_sources.insert(endpoint.clone(), Arc::downgrade(&instance_source));
        Ok(instance_source)
    }
}
