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

    // Client with auto-discover instances using etcd
    pub(crate) async fn new_dynamic(endpoint: Endpoint) -> Result<Self> {
        const INSTANCE_REFRESH_PERIOD: Duration = Duration::from_secs(1);

        // create live endpoint watcher
        let Some(etcd_client) = &endpoint.component.drt.etcd_client else {
            anyhow::bail!("Attempt to create a dynamic client on a static endpoint");
        };

        let instance_source =
            Self::get_or_create_dynamic_instance_source(etcd_client, &endpoint).await?;

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
        etcd_client: &EtcdClient,
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

        let prefix_watcher = etcd_client
            .kv_get_and_watch_prefix(endpoint.etcd_root())
            .await?;

        let (prefix, mut kv_event_rx) = prefix_watcher.dissolve();

        let (watch_tx, watch_rx) = tokio::sync::watch::channel(vec![]);

        let secondary = endpoint.component.drt.runtime.secondary().clone();

        // this task should be included in the registry
        // currently this is created once per client, but this object/task should only be instantiated
        // once per worker/instance
        secondary.spawn(async move {
            tracing::debug!("Starting endpoint watcher for prefix: {}", prefix);
            let mut map = HashMap::new();

            loop {
                let kv_event = tokio::select! {
                    _ = watch_tx.closed() => {
                        tracing::debug!("all watchers have closed; shutting down endpoint watcher for prefix: {prefix}");
                        break;
                    }
                    kv_event = kv_event_rx.recv() => {
                        match kv_event {
                            Some(kv_event) => kv_event,
                            None => {
                                tracing::debug!("watch stream has closed; shutting down endpoint watcher for prefix: {prefix}");
                                break;
                            }
                        }
                    }
                };

                match kv_event {
                    WatchEvent::Put(kv) => {
                        let key = String::from_utf8(kv.key().to_vec());
                        let val = serde_json::from_slice::<Instance>(kv.value());
                        if let (Ok(key), Ok(val)) = (key, val) {
                            map.insert(key.clone(), val);
                        } else {
                            tracing::error!("Unable to parse put endpoint event; shutting down endpoint watcher for prefix: {prefix}");
                            break;
                        }
                    }
                    WatchEvent::Delete(kv) => {
                        match String::from_utf8(kv.key().to_vec()) {
                            Ok(key) => { map.remove(&key); }
                            Err(_) => {
                                tracing::error!("Unable to parse delete endpoint event; shutting down endpoint watcher for prefix: {}", prefix);
                                break;
                            }
                        }
                    }
                }

                let instances: Vec<Instance> = map.values().cloned().collect();

                if watch_tx.send(instances).is_err() {
                    tracing::debug!("Unable to send watch updates; shutting down endpoint watcher for prefix: {}", prefix);
                    break;
                }

            }

            tracing::debug!("Completed endpoint watcher for prefix: {prefix}");
            let _ = watch_tx.send(vec![]);
        });

        let instance_source = Arc::new(InstanceSource::Dynamic(watch_rx));
        instance_sources.insert(endpoint.clone(), Arc::downgrade(&instance_source));
        Ok(instance_source)
    }
}
