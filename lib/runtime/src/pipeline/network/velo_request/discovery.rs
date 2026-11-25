// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use futures::StreamExt;
use tokio::sync::Notify;
use tokio_util::sync::{CancellationToken, DropGuard};
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
        if self.release(peer) {
            let now = service.peer_clock();
            self.prune(now);
            self.departed.insert(
                peer,
                now + crate::pipeline::network::tombstones::TOMBSTONE_TTL.as_secs(),
            );
            service.unregister_request_peer(peer);
        }
    }

    fn release(&mut self, peer: InstanceId) -> bool {
        let Entry::Occupied(mut entry) = self.refs.entry(peer) else {
            return false;
        };
        *entry.get_mut() -= 1;
        if *entry.get() == 0 {
            entry.remove();
            return true;
        }
        false
    }

    #[cfg(test)]
    pub(in crate::pipeline::network) fn count(&self, peer: InstanceId) -> usize {
        self.refs.get(&peer).copied().unwrap_or_default()
    }
}

pub(in crate::pipeline::network) struct RequestPeerWatch {
    pub(in crate::pipeline::network) ready: Arc<DiscoveryReady>,
    _cancel: DropGuard,
}

#[derive(Default)]
pub(in crate::pipeline::network) struct DiscoveryReady {
    established: AtomicBool,
    notify: Notify,
}

impl DiscoveryReady {
    pub(super) fn is_established(&self) -> bool {
        self.established.load(Ordering::Acquire)
    }

    pub(super) async fn wait(&self) {
        // Register before checking so the initial snapshot cannot race the wait.
        let ready = self.notify.notified();
        if !self.is_established() {
            ready.await;
        }
    }

    fn mark_established(&self) {
        if !self.established.swap(true, Ordering::Release) {
            self.notify.notify_waiters();
        }
    }
}

pub(in crate::pipeline::network) fn start(
    service: &Arc<VeloResponseService>,
    discovery: Arc<dyn Discovery>,
    cancel: CancellationToken,
) -> RequestPeerWatch {
    let guard = cancel.clone().drop_guard();
    let ready = Arc::new(DiscoveryReady::default());
    let signal = ready.clone();
    let mut watch = PeerWatch {
        service: Arc::downgrade(service),
        endpoints: HashMap::new(),
    };
    service.velo.runtime().spawn(async move {
        loop {
            match tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                stream = discovery.list_and_watch(DiscoveryQuery::AllEndpoints, Some(cancel.clone())) => stream,
            } {
                Ok(mut stream) => loop {
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
                            let is_snapshot = matches!(event, DiscoveryEvent::Resync(_));
                            watch.apply(&service, event);
                            if is_snapshot {
                                signal.mark_established();
                            }
                        }
                        Some(Err(error)) => tracing::warn!(%error, "Velo request peer discovery event failed"),
                        None => break,
                    }
                },
                Err(error) => tracing::warn!(%error, "Velo request peer discovery watch failed"),
            }
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(Duration::from_secs(1)) => {}
            }
        }
    });
    RequestPeerWatch {
        ready,
        _cancel: guard,
    }
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
            // Ending a watch releases its references without recording a discovery delete.
            let mut refs = service.discovered_request_peers.lock();
            for (_, peer) in self.endpoints.drain() {
                refs.release(peer);
            }
        }
    }
}

fn peer_id(transport: &TransportType) -> Option<InstanceId> {
    match transport {
        TransportType::Velo(address) => Some(decode_address(address).ok()?.instance_id),
        _ => None,
    }
}
