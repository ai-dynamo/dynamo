// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::pipeline::{
    AddressedPushRouter, AddressedRequest, AsyncEngine, Data, ManyOut, PushRouter, RouterMode,
    SingleIn,
};
use arc_swap::ArcSwap;
use rand::Rng;
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Instant;
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
    Put(String, i64),
    Delete(String),
}

#[derive(Clone, Debug)]
pub struct Client {
    // This is me
    pub endpoint: Endpoint,
    // These are the remotes I know about from watching etcd
    pub instance_source: Arc<InstanceSource>,
    // These are the instances that are reported as down from sending rpc
    instance_inhibited: Arc<Mutex<HashMap<i64, Instant>>>,
    // The current active IDs
    instance_cache: Arc<ArcSwap<Vec<i64>>>,
}

#[derive(Clone, Debug)]
pub enum InstanceSource {
    Static,
    Dynamic(tokio::sync::watch::Receiver<Vec<Instance>>),
}

// TODO: Avoid returning a full clone of `Vec<Instance>` everytime from Client
//       See instances() and instances_avail() methods
impl Client {
    // Client will only talk to a single static endpoint
    pub(crate) async fn new_static(endpoint: Endpoint) -> Result<Self> {
        Ok(Client {
            endpoint,
            instance_source: Arc::new(InstanceSource::Static),
            instance_inhibited: Arc::new(Mutex::new(HashMap::new())),
            instance_cache: Arc::new(ArcSwap::from(Arc::new(vec![]))),
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

        let cancel_token = endpoint.drt().primary_token();
        let client = Client {
            endpoint,
            instance_source,
            instance_inhibited: Arc::new(Mutex::new(HashMap::new())),
            instance_cache: Arc::new(ArcSwap::from(Arc::new(vec![]))),
        };

        let instance_source_c = client.instance_source.clone();
        let instance_inhibited_c = Arc::clone(&client.instance_inhibited);
        let instance_cache_c = Arc::clone(&client.instance_cache);
        tokio::task::spawn(async move {
            while !cancel_token.is_cancelled() {
                refresh_instances(&instance_source_c, &instance_inhibited_c, &instance_cache_c);
                tokio::select! {
                    _ = cancel_token.cancelled() => {}
                    _ = tokio::time::sleep(INSTANCE_REFRESH_PERIOD) => {}
                }
            }
        });
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
        instances_inner(self.instance_source.as_ref())
    }

    pub fn instance_ids(&self) -> Vec<i64> {
        self.instances().into_iter().map(|ep| ep.id()).collect()
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

    /// Instances available from watching etcd minus those reported as down
    pub fn instance_ids_avail(&self) -> arc_swap::Guard<Arc<Vec<i64>>> {
        self.instance_cache.load()
    }

    /// Mark an instance as down/unavailable
    pub fn report_instance_down(&self, instance_id: i64) {
        self.instance_inhibited
            .lock()
            .unwrap()
            .insert(instance_id, Instant::now());

        tracing::debug!("inhibiting instance {instance_id}");
    }

    /// Is this component know at startup and not discovered via etcd?
    pub fn is_static(&self) -> bool {
        matches!(self.instance_source.as_ref(), InstanceSource::Static)
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

        let (prefix, _watcher, mut kv_event_rx) = prefix_watcher.dissolve();

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

/// Update the instance id cache
fn refresh_instances(
    instance_source: &InstanceSource,
    instance_inhibited: &Arc<Mutex<HashMap<i64, Instant>>>,
    instance_cache: &Arc<ArcSwap<Vec<i64>>>,
) {
    const ETCD_LEASE_TTL: u64 = 10; // seconds

    // TODO: Can we get the remaining TTL from the lease for the instance?
    let now = Instant::now();

    let instances = instances_inner(instance_source);
    let mut inhibited = instance_inhibited.lock().unwrap();

    // 1. Remove inhibited instances that are no longer in `self.instances()`
    // 2. Remove inhibited instances that have expired
    // 3. Only return instances that are not inhibited after removals
    let mut new_inhibited = HashMap::<i64, Instant>::new();
    let filtered: Vec<i64> = instances
        .into_iter()
        .filter_map(|instance| {
            let id = instance.id();
            if let Some(&timestamp) = inhibited.get(&id) {
                if now.duration_since(timestamp).as_secs() > ETCD_LEASE_TTL {
                    Some(id)
                } else {
                    new_inhibited.insert(id, timestamp);
                    None
                }
            } else {
                Some(id)
            }
        })
        .collect();

    *inhibited = new_inhibited;
    instance_cache.store(Arc::new(filtered));
}

fn instances_inner(instance_source: &InstanceSource) -> Vec<Instance> {
    match instance_source {
        InstanceSource::Static => vec![],
        InstanceSource::Dynamic(watch_rx) => watch_rx.borrow().clone(),
    }
}
