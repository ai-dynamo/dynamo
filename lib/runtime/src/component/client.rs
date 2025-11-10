// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::{collections::HashMap, time::Duration};

use anyhow::Result;
use arc_swap::ArcSwap;
use futures::StreamExt;
use tokio::net::unix::pipe::Receiver;

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

    // Client with auto-discover instances using etcd
    pub(crate) async fn new_dynamic(endpoint: Endpoint) -> Result<Self> {
        tracing::debug!(
            "Client::new_dynamic: Creating dynamic client for endpoint: {}",
            endpoint.path()
        );
        const INSTANCE_REFRESH_PERIOD: Duration = Duration::from_secs(1);

        let instance_source = Self::get_or_create_dynamic_instance_source(&endpoint).await?;
        tracing::debug!(
            "Client::new_dynamic: Got instance source for endpoint: {}",
            endpoint.path()
        );

        let client = Client {
            endpoint: endpoint.clone(),
            instance_source: instance_source.clone(),
            instance_avail: Arc::new(ArcSwap::from(Arc::new(vec![]))),
            instance_free: Arc::new(ArcSwap::from(Arc::new(vec![]))),
        };
        tracing::debug!(
            "Client::new_dynamic: Starting instance source monitor for endpoint: {}",
            endpoint.path()
        );
        client.monitor_instance_source();
        tracing::debug!(
            "Client::new_dynamic: Successfully created dynamic client for endpoint: {}",
            endpoint.path()
        );
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
        tracing::debug!(
            "wait_for_instances: Starting wait for endpoint: {}",
            self.endpoint.path()
        );
        let mut instances: Vec<Instance> = vec![];
        if let InstanceSource::Dynamic(mut rx) = self.instance_source.as_ref().clone() {
            // wait for there to be 1 or more endpoints
            let mut iteration = 0;
            loop {
                instances = rx.borrow_and_update().to_vec();
                tracing::debug!(
                    "wait_for_instances: iteration={}, current_instance_count={}, endpoint={}",
                    iteration,
                    instances.len(),
                    self.endpoint.path()
                );
                if instances.is_empty() {
                    tracing::debug!(
                        "wait_for_instances: No instances yet, waiting for change notification for endpoint: {}",
                        self.endpoint.path()
                    );
                    rx.changed().await?;
                    tracing::debug!(
                        "wait_for_instances: Change notification received for endpoint: {}",
                        self.endpoint.path()
                    );
                } else {
                    tracing::info!(
                        "wait_for_instances: Found {} instance(s) for endpoint: {}",
                        instances.len(),
                        self.endpoint.path()
                    );
                    break;
                }
                iteration += 1;
            }
        } else {
            tracing::debug!(
                "wait_for_instances: Static instance source, no dynamic discovery for endpoint: {}",
                self.endpoint.path()
            );
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
        let endpoint_path = self.endpoint.path();
        tracing::debug!(
            "monitor_instance_source: Starting monitor for endpoint: {}",
            endpoint_path
        );
        tokio::task::spawn(async move {
            let mut rx = match client.instance_source.as_ref() {
                InstanceSource::Static => {
                    tracing::error!(
                        "monitor_instance_source: Static instance source is not watchable"
                    );
                    return;
                }
                InstanceSource::Dynamic(rx) => rx.clone(),
            };
            let mut iteration = 0;
            while !cancel_token.is_cancelled() {
                let instance_ids: Vec<u64> = rx
                    .borrow_and_update()
                    .iter()
                    .map(|instance| instance.id())
                    .collect();

                tracing::debug!(
                    "monitor_instance_source: iteration={}, instance_count={}, instance_ids={:?}, endpoint={}",
                    iteration,
                    instance_ids.len(),
                    instance_ids,
                    endpoint_path
                );

                // TODO: this resets both tracked available and free instances
                client.instance_avail.store(Arc::new(instance_ids.clone()));
                client.instance_free.store(Arc::new(instance_ids.clone()));

                tracing::debug!(
                    "monitor_instance_source: instance source updated, endpoint={}",
                    endpoint_path
                );

                if let Err(err) = rx.changed().await {
                    tracing::error!(
                        "monitor_instance_source: The Sender is dropped: {}, endpoint={}",
                        err,
                        endpoint_path
                    );
                    cancel_token.cancel();
                }
                iteration += 1;
            }
            tracing::debug!(
                "monitor_instance_source: Monitor loop exiting for endpoint: {}",
                endpoint_path
            );
        });
    }

    async fn get_or_create_dynamic_instance_source(
        endpoint: &Endpoint,
    ) -> Result<Arc<InstanceSource>> {
        let drt = endpoint.drt();
        let instance_sources = drt.instance_sources();
        let mut instance_sources = instance_sources.lock().await;

        tracing::debug!(
            "get_or_create_dynamic_instance_source: Checking cache for endpoint: {}",
            endpoint.path()
        );

        if let Some(instance_source) = instance_sources.get(endpoint) {
            if let Some(instance_source) = instance_source.upgrade() {
                tracing::debug!(
                    "get_or_create_dynamic_instance_source: Found cached instance source for endpoint: {}",
                    endpoint.path()
                );
                return Ok(instance_source);
            } else {
                tracing::debug!(
                    "get_or_create_dynamic_instance_source: Cached instance source was dropped, removing for endpoint: {}",
                    endpoint.path()
                );
                instance_sources.remove(endpoint);
            }
        }

        tracing::debug!(
            "get_or_create_dynamic_instance_source: Creating new instance source for endpoint: {}",
            endpoint.path()
        );

        let discovery = drt.discovery();
        let discovery_query = crate::discovery::DiscoveryQuery::Endpoint {
            namespace: endpoint.component.namespace.name.clone(),
            component: endpoint.component.name.clone(),
            endpoint: endpoint.name.clone(),
        };

        tracing::debug!(
            "get_or_create_dynamic_instance_source: Calling discovery.list_and_watch for query: {:?}",
            discovery_query
        );

        let mut discovery_stream = discovery
            .list_and_watch(discovery_query.clone(), None)
            .await?;

        tracing::debug!(
            "get_or_create_dynamic_instance_source: Got discovery stream for query: {:?}",
            discovery_query
        );

        let (watch_tx, watch_rx) = tokio::sync::watch::channel(vec![]);

        let secondary = endpoint.component.drt.runtime().secondary().clone();

        secondary.spawn(async move {
            tracing::debug!("endpoint_watcher: Starting for discovery query: {:?}", discovery_query);
            let mut map: HashMap<u64, Instance> = HashMap::new();
            let mut event_count = 0;

            loop {
                let discovery_event = tokio::select! {
                    _ = watch_tx.closed() => {
                        tracing::debug!("endpoint_watcher: all watchers have closed; shutting down for discovery query: {:?}", discovery_query);
                        break;
                    }
                    discovery_event = discovery_stream.next() => {
                        tracing::debug!("endpoint_watcher: Received stream event for discovery query: {:?}", discovery_query);
                        match discovery_event {
                            Some(Ok(event)) => {
                                tracing::debug!("endpoint_watcher: Got Ok event: {:?}", event);
                                event
                            },
                            Some(Err(e)) => {
                                tracing::error!("endpoint_watcher: discovery stream error: {}; shutting down for discovery query: {:?}", e, discovery_query);
                                break;
                            }
                            None => {
                                tracing::debug!("endpoint_watcher: watch stream has closed; shutting down for discovery query: {:?}", discovery_query);
                                break;
                            }
                        }
                    }
                };

                event_count += 1;
                tracing::debug!("endpoint_watcher: Processing event #{} for discovery query: {:?}", event_count, discovery_query);

                match discovery_event {
                    crate::discovery::DiscoveryEvent::Added(discovery_instance) => {
                        match discovery_instance {
                            crate::discovery::DiscoveryInstance::Endpoint(instance) => {
                                tracing::debug!(
                                    "endpoint_watcher: Added endpoint instance_id={}, namespace={}, component={}, endpoint={}",
                                    instance.instance_id,
                                    instance.namespace,
                                    instance.component,
                                    instance.endpoint
                                );
                                map.insert(instance.instance_id, instance);
                            }
                            _ => {
                                tracing::debug!("endpoint_watcher: Ignoring non-endpoint instance (Model, etc.) for discovery query: {:?}", discovery_query);
                            }
                        }
                    }
                    crate::discovery::DiscoveryEvent::Removed(instance_id) => {
                        tracing::debug!(
                            "endpoint_watcher: Removed instance_id={} for discovery query: {:?}",
                            instance_id,
                            discovery_query
                        );
                        map.remove(&instance_id);
                    }
                }

                let instances: Vec<Instance> = map.values().cloned().collect();
                tracing::debug!(
                    "endpoint_watcher: Current map size={}, sending update for discovery query: {:?}",
                    instances.len(),
                    discovery_query
                );

                if watch_tx.send(instances).is_err() {
                    tracing::debug!("endpoint_watcher: Unable to send watch updates; shutting down for discovery query: {:?}", discovery_query);
                    break;
                }
            }

            tracing::debug!("endpoint_watcher: Completed for discovery query: {:?}, total events processed: {}", discovery_query, event_count);
            let _ = watch_tx.send(vec![]);
        });

        let instance_source = Arc::new(InstanceSource::Dynamic(watch_rx));
        instance_sources.insert(endpoint.clone(), Arc::downgrade(&instance_source));
        tracing::debug!(
            "get_or_create_dynamic_instance_source: Successfully created and cached instance source for endpoint: {}",
            endpoint.path()
        );
        Ok(instance_source)
    }
}
