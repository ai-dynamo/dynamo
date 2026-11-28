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
    refs: HashMap<InstanceId, PeerRefs>,
    departed: HashMap<InstanceId, u64>,
}

#[derive(Default)]
struct PeerRefs {
    count: usize,
    /// A watch saw the peer leave discovery while another watch still held it.
    left_discovery: bool,
}

impl DiscoveryPeers {
    pub(in crate::pipeline::network) fn recently_departed(&self, id: InstanceId, now: u64) -> bool {
        self.departed.get(&id).is_some_and(|&expires| expires > now)
    }

    /// Drops expired tombstones. The peer sweep calls this, so a burst of
    /// removals does not rescan the map once per peer.
    pub(in crate::pipeline::network) fn prune(&mut self, now: u64) {
        self.departed.retain(|_, expires| *expires > now);
    }

    fn add(&mut self, service: &VeloResponseService, peer: InstanceId) {
        self.departed.remove(&peer);
        let refs = self.refs.entry(peer).or_default();
        if refs.count == 0 {
            service.rediscover_request_peer(peer);
        }
        refs.count += 1;
        refs.left_discovery = false;
    }

    /// Drops one watch's reference. `left_discovery` says that watch saw the
    /// peer's last endpoint deleted, as opposed to one of several endpoints
    /// going away or the watch itself ending. The peer is retired with the
    /// last reference if any watch saw it leave.
    fn release(&mut self, service: &VeloResponseService, peer: InstanceId, left_discovery: bool) {
        let Entry::Occupied(mut entry) = self.refs.entry(peer) else {
            return;
        };
        let refs = entry.get_mut();
        refs.left_discovery |= left_discovery;
        refs.count -= 1;
        if refs.count > 0 || !entry.remove().left_discovery {
            return;
        }
        self.departed.insert(
            peer,
            service.peer_clock() + crate::pipeline::network::tombstones::TOMBSTONE_TTL.as_secs(),
        );
        service.unregister_request_peer(peer);
    }

    #[cfg(test)]
    pub(in crate::pipeline::network) fn count(&self, peer: InstanceId) -> usize {
        self.refs.get(&peer).map_or(0, |refs| refs.count)
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
    let mut watch = PeerWatch::new(service);
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
    /// How many of this watch's endpoints each peer serves. A peer has left
    /// discovery, as this watch sees it, when its last endpoint is deleted.
    peers: HashMap<InstanceId, usize>,
}

impl PeerWatch {
    fn new(service: &Arc<VeloResponseService>) -> Self {
        Self {
            service: Arc::downgrade(service),
            endpoints: HashMap::new(),
            peers: HashMap::new(),
        }
    }

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
            self.add(&mut refs, service, peer);
        }
        if let Some(peer) = previous {
            self.delete(&mut refs, service, peer);
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
                self.add(&mut refs, service, peer);
            }
        }
        let previous = std::mem::replace(&mut self.endpoints, next);
        for (endpoint, peer) in previous {
            if self.endpoints.get(&endpoint) != Some(&peer) {
                self.delete(&mut refs, service, peer);
            }
        }
    }

    fn add(&mut self, refs: &mut DiscoveryPeers, service: &VeloResponseService, peer: InstanceId) {
        *self.peers.entry(peer).or_default() += 1;
        refs.add(service, peer);
    }

    /// One of this watch's endpoints left discovery.
    fn delete(
        &mut self,
        refs: &mut DiscoveryPeers,
        service: &VeloResponseService,
        peer: InstanceId,
    ) {
        let Entry::Occupied(mut listed) = self.peers.entry(peer) else {
            return;
        };
        *listed.get_mut() -= 1;
        let left_discovery = *listed.get() == 0;
        if left_discovery {
            listed.remove();
        }
        refs.release(service, peer, left_discovery);
    }
}

impl Drop for PeerWatch {
    fn drop(&mut self) {
        if let Some(service) = self.service.upgrade() {
            // Ending a watch is not a discovery delete. A peer is retired here
            // only if another watch already saw it deleted.
            let mut refs = service.discovered_request_peers.lock();
            for (_, peer) in self.endpoints.drain() {
                refs.release(&service, peer, false);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        component::Instance,
        error::ErrorType,
        pipeline::network::velo_response::{ResponseTransport, VeloServiceOptions},
    };

    fn endpoint(transport: TransportType) -> DiscoveryEvent {
        DiscoveryEvent::Added(DiscoveryInstance::Endpoint(Instance {
            namespace: "ns".into(),
            component: "worker".into(),
            endpoint: "generate".into(),
            instance_id: 1,
            transport,
            device_type: None,
            request_plane_codec: None,
        }))
    }
    #[tokio::test]
    async fn a_closing_watch_preserves_another_owners_discovery_delete() {
        let service = VeloResponseService::new_with_options(
            ResponseTransport::Tcp,
            "127.0.0.1:0".parse().unwrap(),
            VeloServiceOptions {
                request_plane_enabled: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let watch = || PeerWatch::new(&service);
        let mut live = watch();
        let mut closing = watch();
        let peer = InstanceId::new_v4();
        let added = || {
            endpoint(TransportType::Velo(format!(
                "velo://{peer}/AA/1/ns/worker/generate"
            )))
        };
        live.apply(&service, added());
        closing.apply(&service, added());
        live.apply(&service, DiscoveryEvent::Resync(vec![]));
        assert_eq!(service.discovered_request_peers.lock().count(peer), 1);

        // Cancellation wins the second watch's select before it reads the same removal.
        drop(closing);
        assert_eq!(service.discovered_request_peers.lock().count(peer), 0);
        assert_eq!(
            service
                .reserve_request_peer(peer)
                .err()
                .map(|e| e.error_type()),
            Some(ErrorType::WorkerUnavailable),
            "the remaining owner can still contact a peer it saw leave"
        );
    }

    #[tokio::test]
    async fn in_place_plane_change_retires_and_restores_the_request_peer() {
        let service = VeloResponseService::new_with_options(
            ResponseTransport::Tcp,
            "127.0.0.1:0".parse().unwrap(),
            VeloServiceOptions {
                request_plane_enabled: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let peer = InstanceId::new_v4();
        let velo = || {
            endpoint(TransportType::Velo(format!(
                "velo://{peer}/AA/1/ns/worker/generate"
            )))
        };
        let mut watch = PeerWatch::new(&service);
        let count = || service.discovered_request_peers.lock().count(peer);

        watch.apply(&service, velo());
        assert_eq!(count(), 1);
        assert!(service.reserve_request_peer(peer).is_ok());

        // The same endpoint instance comes back on the tcp plane.
        watch.apply(
            &service,
            endpoint(TransportType::Tcp(
                "127.0.0.1:9/AA/1/ns/worker/generate".into(),
            )),
        );
        assert_eq!(count(), 0);
        assert_eq!(
            service
                .reserve_request_peer(peer)
                .err()
                .unwrap()
                .error_type(),
            ErrorType::WorkerUnavailable
        );

        watch.apply(&service, velo());
        assert_eq!(count(), 1);
        assert!(service.reserve_request_peer(peer).is_ok());
    }

    #[tokio::test]
    async fn closing_a_watch_after_one_sibling_leaves_preserves_the_serving_endpoint() {
        use super::super::{REQUEST_HANDLER, client::VeloRequestClient, encode_address};
        use crate::{
            discovery::{DiscoverySpec, MockDiscovery, SharedMockRegistry, startup_contract},
            pipeline::network::egress::unified_client::RequestPlaneClient,
            protocols::EndpointId,
        };
        use bytes::Bytes;

        let service = || {
            VeloResponseService::new_with_options(
                ResponseTransport::Tcp,
                "127.0.0.1:0".parse().unwrap(),
                VeloServiceOptions {
                    request_plane_enabled: true,
                    ..Default::default()
                },
            )
        };
        let frontend = service().await.unwrap();
        let worker = service().await.unwrap();
        worker
            .velo
            .register_handler(
                velo::Handler::unary_handler(REQUEST_HANDLER, |_| {
                    Ok(Some(Bytes::from_static(b"ack")))
                })
                .build(),
            )
            .unwrap();
        let peer = worker.velo.peer_info();
        let registry = MockDiscovery::new(Some(1), SharedMockRegistry::new());
        let mut closing = PeerWatch::new(&frontend);
        let mut stream = registry
            .list_and_watch(DiscoveryQuery::AllEndpoints, None)
            .await
            .unwrap();
        startup_contract::expect_empty_snapshot(&mut stream).await;
        // Another owner's all-endpoints watch is established but has not seen
        // the additions yet. Its per-endpoint lookup below already has generate.
        let mut other_stream = registry
            .list_and_watch(DiscoveryQuery::AllEndpoints, None)
            .await
            .unwrap();
        startup_contract::expect_empty_snapshot(&mut other_stream).await;
        let ready = Arc::new(DiscoveryReady::default());
        ready.mark_established();
        let client =
            VeloRequestClient::new(frontend.clone(), Duration::from_secs(2)).with_discovery(ready);
        let spec = |name: &str| {
            let endpoint = EndpointId {
                namespace: "ns".into(),
                component: "worker".into(),
                name: name.into(),
            };
            let address = encode_address(&peer, &endpoint, registry.instance_id()).unwrap();
            DiscoverySpec::Endpoint {
                namespace: endpoint.namespace,
                component: endpoint.component,
                endpoint: endpoint.name,
                transport: TransportType::Velo(address),
                device_type: None,
                request_plane_codec: None,
            }
        };
        let generate = registry.register(spec("generate")).await.unwrap();
        let tokenize = registry.register(spec("tokenize")).await.unwrap();
        closing.apply(&frontend, startup_contract::next(&mut stream).await);
        closing.apply(&frontend, startup_contract::next(&mut stream).await);
        registry.unregister(tokenize).await.unwrap();
        closing.apply(&frontend, startup_contract::next(&mut stream).await);
        assert_eq!(
            frontend
                .discovered_request_peers
                .lock()
                .count(peer.instance_id()),
            1
        );
        let serving = registry
            .list(DiscoveryQuery::Endpoint {
                namespace: "ns".into(),
                component: "worker".into(),
                endpoint: "generate".into(),
            })
            .await
            .unwrap();
        assert_eq!(serving, vec![generate]);
        let DiscoveryInstance::Endpoint(Instance {
            transport: TransportType::Velo(address),
            ..
        }) = &serving[0]
        else {
            panic!("generate has no Velo address");
        };
        let send = || client.send_request(address.clone(), Bytes::new(), Default::default());
        assert_eq!(send().await.unwrap(), "ack");
        drop(stream);
        drop(closing);
        assert_eq!(
            send()
                .await
                .expect("closing the watch retired the still-discovered generate endpoint"),
            "ack"
        );
    }
}
