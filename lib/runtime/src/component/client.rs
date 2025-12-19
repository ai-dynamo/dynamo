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
    traits::DistributedRuntimeProvider,
    transports::etcd::Client as EtcdClient,
};

/// Default interval for periodic reconciliation of instance_avail with instance_source
const DEFAULT_RECONCILE_INTERVAL: Duration = Duration::from_secs(5);

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
    /// Interval for periodic reconciliation of instance_avail with instance_source.
    /// This ensures instances removed via `report_instance_down` are eventually restored.
    reconcile_interval: Duration,
}

impl Client {
    // Client with auto-discover instances using key-value store
    pub(crate) async fn new(endpoint: Endpoint) -> Result<Self> {
        Self::with_reconcile_interval(endpoint, DEFAULT_RECONCILE_INTERVAL).await
    }

    /// Create a client with a custom reconcile interval.
    /// The reconcile interval controls how often `instance_avail` is reset to match
    /// `instance_source`, restoring any instances removed via `report_instance_down`.
    pub(crate) async fn with_reconcile_interval(
        endpoint: Endpoint,
        reconcile_interval: Duration,
    ) -> Result<Self> {
        tracing::trace!(
            "Client::new_dynamic: Creating dynamic client for endpoint: {}",
            endpoint.id()
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
            reconcile_interval,
        };
        client.monitor_instance_source();
        Ok(client)
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
            self.endpoint.id()
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
                    self.endpoint.id()
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
    ///
    /// This function also performs periodic reconciliation: if `instance_source` hasn't
    /// changed for `reconcile_interval`, we reset `instance_avail` to match
    /// `instance_source`. This ensures instances removed via `report_instance_down`
    /// are eventually restored even if the discovery source doesn't emit updates.
    fn monitor_instance_source(&self) {
        let reconcile_interval = self.reconcile_interval;
        let cancel_token = self.endpoint.drt().primary_token();
        let client = self.clone();
        let endpoint_id = self.endpoint.id();
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

                tokio::select! {
                    result = rx.changed() => {
                        if let Err(err) = result {
                            tracing::error!(
                                "monitor_instance_source: The Sender is dropped: {err}, endpoint={endpoint_id}",
                            );
                            cancel_token.cancel();
                        }
                    }
                    _ = tokio::time::sleep(reconcile_interval) => {
                        tracing::trace!(
                            "monitor_instance_source: periodic reconciliation for endpoint={endpoint_id}",
                        );
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    /// Test that instances removed via report_instance_down are restored after
    /// the reconciliation interval elapses.
    #[tokio::test]
    async fn test_instance_reconciliation() {
        const TEST_RECONCILE_INTERVAL: Duration = Duration::from_millis(100);

        let rt = Runtime::from_current().unwrap();
        // Use process_local config to avoid needing etcd/nats
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_reconciliation".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        // Use a short reconcile interval for faster tests
        let client = Client::with_reconcile_interval(endpoint, TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();

        // Initially, instance_avail should be empty (no registered instances)
        assert!(client.instance_ids_avail().is_empty());

        // For this test, we'll directly manipulate instance_avail and verify reconciliation
        // Store some test IDs
        client.instance_avail.store(Arc::new(vec![1, 2, 3]));

        assert_eq!(**client.instance_ids_avail(), vec![1u64, 2, 3]);

        // Simulate report_instance_down removing instance 2
        client.report_instance_down(2);
        assert_eq!(**client.instance_ids_avail(), vec![1u64, 3]);

        // Wait for reconciliation interval + buffer
        // The monitor_instance_source will reset instance_avail to match instance_source
        // Since instance_source is empty, after reconciliation instance_avail should be empty
        tokio::time::sleep(TEST_RECONCILE_INTERVAL + Duration::from_millis(50)).await;

        // After reconciliation, instance_avail should match instance_source (which is empty)
        assert!(
            client.instance_ids_avail().is_empty(),
            "After reconciliation, instance_avail should match instance_source"
        );

        rt.shutdown();
    }

    /// Test that report_instance_down correctly removes an instance from instance_avail.
    #[tokio::test]
    async fn test_report_instance_down() {
        let rt = Runtime::from_current().unwrap();
        // Use process_local config to avoid needing etcd/nats
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_report_down".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = endpoint.client().await.unwrap();

        // Manually set up instance_avail with test instances
        client.instance_avail.store(Arc::new(vec![1, 2, 3]));
        assert_eq!(**client.instance_ids_avail(), vec![1u64, 2, 3]);

        // Report instance 2 as down
        client.report_instance_down(2);

        // Verify instance 2 is removed
        let avail = client.instance_ids_avail();
        assert!(avail.contains(&1), "Instance 1 should still be available");
        assert!(
            !avail.contains(&2),
            "Instance 2 should be removed after report_instance_down"
        );
        assert!(avail.contains(&3), "Instance 3 should still be available");

        rt.shutdown();
    }

    /// Test that instance_avail_watcher receives updates when instances change.
    #[tokio::test]
    async fn test_instance_avail_watcher() {
        let rt = Runtime::from_current().unwrap();
        // Use process_local config to avoid needing etcd/nats
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_watcher".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = endpoint.client().await.unwrap();
        let watcher = client.instance_avail_watcher();

        // Set initial instances
        client.instance_avail.store(Arc::new(vec![1, 2, 3]));

        // Report instance down - this should notify the watcher
        client.report_instance_down(2);

        // The watcher should receive the update
        // Note: We need to check if changed() was signaled
        let current = watcher.borrow().clone();
        assert_eq!(current, vec![1, 3]);

        rt.shutdown();
    }
}
