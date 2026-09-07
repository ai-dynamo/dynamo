// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result};
use dashmap::{DashMap, mapref::entry::Entry as DashEntry};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::{
    Discovery, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, DiscoverySpec,
    EndpointInstanceId,
};

struct RegistrationEntry {
    instance: DiscoveryInstance,
    leases: usize,
    owned: bool,
}

type RegistrationSlot = Arc<Mutex<Option<RegistrationEntry>>>;

/// Coordinates endpoint registration ownership across all users of one distributed runtime.
pub(crate) struct EndpointRegistrationManager {
    discovery: Arc<dyn Discovery>,
    slots: DashMap<EndpointInstanceId, RegistrationSlot>,
    runtime: tokio::runtime::Handle,
    cancellation: CancellationToken,
}

impl EndpointRegistrationManager {
    pub(crate) fn new(
        discovery: Arc<dyn Discovery>,
        runtime: tokio::runtime::Handle,
        cancellation: CancellationToken,
    ) -> Arc<Self> {
        Arc::new(Self {
            discovery,
            slots: DashMap::new(),
            runtime,
            cancellation,
        })
    }

    pub(crate) async fn register(
        self: &Arc<Self>,
        spec: DiscoverySpec,
    ) -> Result<EndpointRegistrationLease> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let manager = self.clone();
        self.runtime.spawn(async move {
            let result = manager.acquire(spec).await;
            let _ = result_tx.send(result);
        });
        result_rx
            .await
            .context("endpoint discovery registration task ended without a result")?
    }

    async fn acquire(self: &Arc<Self>, spec: DiscoverySpec) -> Result<EndpointRegistrationLease> {
        let candidate = spec.clone().into_instance(self.discovery.instance_id());
        let DiscoveryInstanceId::Endpoint(id) = candidate.id() else {
            anyhow::bail!("endpoint registration leases require an endpoint specification");
        };
        let slot = self
            .slots
            .entry(id.clone())
            .or_insert_with(|| Arc::new(Mutex::new(None)))
            .clone();
        let mut entry = slot.lock().await;
        if let Some(entry) = entry.as_mut() {
            anyhow::ensure!(
                entry.instance == candidate,
                "endpoint registration lease conflicts with the existing specification"
            );
            entry.leases += 1;
            return Ok(EndpointRegistrationLease::new(self.clone(), id));
        }

        let registration = async {
            let existing = self
                .discovery
                .list(DiscoveryQuery::Endpoint {
                    namespace: id.namespace.clone(),
                    component: id.component.clone(),
                    endpoint: id.endpoint.clone(),
                })
                .await?
                .into_iter()
                .find(|instance| instance.id() == DiscoveryInstanceId::Endpoint(id.clone()));
            match existing {
                Some(existing) => {
                    anyhow::ensure!(
                        existing == candidate,
                        "endpoint registration lease conflicts with a pre-existing specification"
                    );
                    Ok((existing, false))
                }
                None => Ok((self.discovery.register(spec).await?, true)),
            }
        }
        .await;
        let (instance, owned) = match registration {
            Ok(registration) => registration,
            Err(error) => {
                drop(entry);
                self.remove_unused_slot(&id, &slot);
                return Err(error);
            }
        };
        *entry = Some(RegistrationEntry {
            instance,
            leases: 1,
            owned,
        });
        Ok(EndpointRegistrationLease::new(self.clone(), id))
    }

    async fn release(self: Arc<Self>, id: EndpointInstanceId) {
        let Some(slot) = self.slots.get(&id).map(|slot| slot.clone()) else {
            return;
        };
        let mut entry = slot.lock().await;
        let Some(registration) = entry.as_mut() else {
            return;
        };
        registration.leases = registration.leases.saturating_sub(1);
        if registration.leases != 0 {
            return;
        }
        if registration.owned {
            let mut retry_delay = Duration::from_millis(50);
            loop {
                match self
                    .discovery
                    .unregister(registration.instance.clone())
                    .await
                {
                    Ok(()) => break,
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            instance = ?registration.instance.id(),
                            "Failed to release endpoint discovery registration; retrying"
                        );
                    }
                }
                tokio::select! {
                    _ = self.cancellation.cancelled() => return,
                    _ = tokio::time::sleep(retry_delay) => {}
                }
                retry_delay = (retry_delay * 2).min(Duration::from_secs(5));
            }
        }
        *entry = None;
        drop(entry);
        self.remove_unused_slot(&id, &slot);
    }

    fn remove_unused_slot(&self, id: &EndpointInstanceId, slot: &RegistrationSlot) {
        if let DashEntry::Occupied(entry) = self.slots.entry(id.clone())
            && Arc::ptr_eq(entry.get(), slot)
            && Arc::strong_count(slot) == 2
        {
            entry.remove();
        }
    }
}

/// Keeps a discovery endpoint registered until the last lease is dropped.
pub struct EndpointRegistrationLease {
    manager: Arc<EndpointRegistrationManager>,
    id: Option<EndpointInstanceId>,
}

impl EndpointRegistrationLease {
    fn new(manager: Arc<EndpointRegistrationManager>, id: EndpointInstanceId) -> Self {
        Self {
            manager,
            id: Some(id),
        }
    }
}

impl std::fmt::Debug for EndpointRegistrationLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EndpointRegistrationLease")
            .field("id", &self.id)
            .finish()
    }
}

impl Drop for EndpointRegistrationLease {
    fn drop(&mut self) {
        let Some(id) = self.id.take() else {
            return;
        };
        let manager = self.manager.clone();
        self.manager.runtime.spawn(async move {
            manager.release(id).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        component::TransportType,
        discovery::{DiscoveryStream, MockDiscovery, SharedMockRegistry},
    };
    use async_trait::async_trait;

    struct BlockingRegistrationDiscovery {
        inner: MockDiscovery,
        registered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl Discovery for BlockingRegistrationDiscovery {
        fn instance_id(&self) -> u64 {
            self.inner.instance_id()
        }

        async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
            let instance = self.inner.register_internal(spec).await?;
            self.registered.notify_one();
            self.release.notified().await;
            Ok(instance)
        }

        async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
            self.inner.unregister(instance).await
        }

        async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
            self.inner.list(query).await
        }

        async fn list_and_watch(
            &self,
            query: DiscoveryQuery,
            cancel_token: Option<CancellationToken>,
        ) -> Result<DiscoveryStream> {
            self.inner.list_and_watch(query, cancel_token).await
        }
    }

    fn endpoint_spec() -> DiscoverySpec {
        DiscoverySpec::Endpoint {
            namespace: "ns".to_string(),
            component: "frontend".to_string(),
            endpoint: "router".to_string(),
            transport: TransportType::Tcp("127.0.0.1:1/7/router".to_string()),
            device_type: None,
            request_plane_codec: None,
        }
    }

    async fn wait_for_endpoint_count(discovery: &dyn Discovery, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let instances = discovery.list(DiscoveryQuery::AllEndpoints).await.unwrap();
                if instances.len() == expected {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("endpoint registration count did not converge");
    }

    #[tokio::test]
    async fn cancelled_registration_releases_endpoint_after_register_completes() {
        let registered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let discovery: Arc<dyn Discovery> = Arc::new(BlockingRegistrationDiscovery {
            inner: MockDiscovery::new(Some(7), SharedMockRegistry::new()),
            registered: registered.clone(),
            release: release.clone(),
        });
        let manager = EndpointRegistrationManager::new(
            discovery.clone(),
            tokio::runtime::Handle::current(),
            CancellationToken::new(),
        );
        let caller = tokio::spawn({
            let manager = manager.clone();
            async move { manager.register(endpoint_spec()).await }
        });

        registered.notified().await;
        caller.abort();
        let _ = caller.await;
        wait_for_endpoint_count(discovery.as_ref(), 1).await;
        release.notify_one();
        wait_for_endpoint_count(discovery.as_ref(), 0).await;
    }

    #[tokio::test]
    async fn endpoint_remains_registered_until_last_lease_drops() {
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(7), SharedMockRegistry::new()));
        let manager = EndpointRegistrationManager::new(
            discovery.clone(),
            tokio::runtime::Handle::current(),
            CancellationToken::new(),
        );
        let first = manager.register(endpoint_spec()).await.unwrap();
        let second = manager.register(endpoint_spec()).await.unwrap();
        wait_for_endpoint_count(discovery.as_ref(), 1).await;

        drop(first);
        tokio::task::yield_now().await;
        wait_for_endpoint_count(discovery.as_ref(), 1).await;
        drop(second);
        wait_for_endpoint_count(discovery.as_ref(), 0).await;
    }
}
