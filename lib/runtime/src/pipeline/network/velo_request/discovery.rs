// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, Weak},
    time::Duration,
};

use anyhow::Result;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use velo::InstanceId;

use super::decode_address;
use crate::{
    component::TransportType,
    discovery::{
        Discovery, DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery,
        EndpointInstanceId,
    },
    pipeline::network::velo_response::VeloResponseService,
};

#[derive(Default)]
pub(in crate::pipeline::network) struct DiscoveryPeers {
    refs: HashMap<InstanceId, usize>,
    departed: HashMap<InstanceId, u64>,
}

impl DiscoveryPeers {
    pub(in crate::pipeline::network) fn recently_departed(&self, id: InstanceId, now: u64) -> bool {
        self.departed.get(&id).is_some_and(|&expires| expires > now)
    }

    pub(in crate::pipeline::network) fn prune(&mut self, now: u64) {
        self.departed.retain(|_, expires| *expires > now);
    }

    fn add(&mut self, service: &VeloResponseService, peer: InstanceId) {
        self.departed.remove(&peer);
        let count = self.refs.entry(peer).or_default();
        if *count == 0 {
            service.rediscover_request_peer(peer);
        }
        *count += 1;
    }

    fn remove(&mut self, service: &VeloResponseService, peer: InstanceId) {
        let Entry::Occupied(mut entry) = self.refs.entry(peer) else {
            return;
        };
        *entry.get_mut() -= 1;
        if *entry.get() == 0 {
            entry.remove();
            let now = service.peer_clock();
            self.prune(now);
            self.departed.insert(
                peer,
                now + crate::pipeline::network::tombstones::TOMBSTONE_TTL.as_secs(),
            );
            service.unregister_request_peer(peer);
        }
    }

    #[cfg(test)]
    pub(in crate::pipeline::network) fn count(&self, peer: InstanceId) -> usize {
        self.refs.get(&peer).copied().unwrap_or_default()
    }
}

pub(in crate::pipeline::network) async fn start(
    service: &Arc<VeloResponseService>,
    discovery: Arc<dyn Discovery>,
    cancel: CancellationToken,
) -> Result<()> {
    let mut stream = discovery
        .list_and_watch(DiscoveryQuery::AllEndpoints, Some(cancel.clone()))
        .await?;
    let mut watch = PeerWatch {
        service: Arc::downgrade(service),
        endpoints: HashMap::new(),
    };
    service.velo.runtime().spawn(async move {
        loop {
            let event = tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                event = stream.next() => event,
            };
            match event {
                Some(Ok(event)) => {
                    let Some(service) = watch.service.upgrade() else {
                        return;
                    };
                    watch.apply(&service, event);
                }
                Some(Err(error)) => {
                    tracing::warn!(%error, "Velo request peer discovery event failed")
                }
                None => loop {
                    let next = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return,
                        next = async {
                            tokio::time::sleep(Duration::from_secs(1)).await;
                            discovery.list_and_watch(DiscoveryQuery::AllEndpoints, Some(cancel.clone())).await
                        } => next,
                    };
                    match next {
                        Ok(next) => {
                            stream = next;
                            break;
                        }
                        Err(error) => {
                            tracing::warn!(%error, "Velo request peer discovery reconnect failed")
                        }
                    }
                },
            }
        }
    });
    Ok(())
}

struct PeerWatch {
    service: Weak<VeloResponseService>,
    endpoints: HashMap<EndpointInstanceId, InstanceId>,
}

impl PeerWatch {
    fn apply(&mut self, service: &VeloResponseService, event: DiscoveryEvent) {
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint(instance)) => {
                let id = instance.endpoint_instance_id();
                self.update(service, id, peer_id(&instance.transport));
            }
            DiscoveryEvent::Removed(DiscoveryInstanceId::Endpoint(id)) => {
                self.update(service, id, None);
            }
            DiscoveryEvent::Resync(instances) => {
                let next = instances
                    .into_iter()
                    .filter_map(|instance| match instance {
                        DiscoveryInstance::Endpoint(instance) => Some((
                            instance.endpoint_instance_id(),
                            peer_id(&instance.transport)?,
                        )),
                        _ => None,
                    })
                    .collect();
                self.replace(service, next);
            }
            _ => {}
        }
    }

    fn update(
        &mut self,
        service: &VeloResponseService,
        endpoint: EndpointInstanceId,
        next: Option<InstanceId>,
    ) {
        let previous = match next {
            Some(peer) => self.endpoints.insert(endpoint, peer),
            None => self.endpoints.remove(&endpoint),
        };
        if previous == next {
            return;
        }
        let mut refs = service.discovered_request_peers.lock();
        if let Some(peer) = next {
            refs.add(service, peer);
        }
        if let Some(peer) = previous {
            refs.remove(service, peer);
        }
    }

    fn replace(
        &mut self,
        service: &VeloResponseService,
        next: HashMap<EndpointInstanceId, InstanceId>,
    ) {
        let mut refs = service.discovered_request_peers.lock();
        // Apply additions first so a changed endpoint key can keep the same peer.
        for (endpoint, &peer) in &next {
            if self.endpoints.get(endpoint) != Some(&peer) {
                refs.add(service, peer);
            }
        }
        for (endpoint, &peer) in &self.endpoints {
            if next.get(endpoint) != Some(&peer) {
                refs.remove(service, peer);
            }
        }
        self.endpoints = next;
    }
}

impl Drop for PeerWatch {
    fn drop(&mut self) {
        if let Some(service) = self.service.upgrade() {
            self.replace(&service, HashMap::new());
        }
    }
}

fn peer_id(transport: &TransportType) -> Option<InstanceId> {
    match transport {
        TransportType::Velo(address) => Some(decode_address(address).ok()?.instance_id),
        _ => None,
    }
}
