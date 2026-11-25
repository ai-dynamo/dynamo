// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::Result;
use bytes::Bytes;
use tokio::{
    sync::{Mutex, OnceCell, Semaphore},
    time::Instant,
};
use velo::{Handler, InstanceId, PeerInfo, Velo, WorkerAddress, WorkerId};

use super::{PEER_HANDLER, REQUEST_HANDLER, STAGED_WINDOW, decode_identity, decode_worker_address};
use crate::{
    error::{DynamoError, ErrorType},
    pipeline::network::{
        ACK_UNAVAILABLE_PREFIX,
        velo_response::{PEER_IDLE_TTL, VeloResponseService},
    },
};

// Keep this below Velo's 30-second hello timeout.
pub(in crate::pipeline::network) const HELLO_CAP: Duration = Duration::from_secs(25);
const CONFIRMED: u64 = 1;
const FAILURE_TTL: Duration = Duration::from_secs(1);

pub(in crate::pipeline::network) struct PeerState {
    pub(in crate::pipeline::network) ready: OnceCell<()>,
    pub(in crate::pipeline::network) active: AtomicUsize,
    pub(in crate::pipeline::network) last_used: AtomicU64,
    pub(in crate::pipeline::network) pinned: AtomicBool,
    pub(in crate::pipeline::network) reverse_pin_until: AtomicU64,
    confirmed_until: AtomicU64,
    pub(in crate::pipeline::network) route: RouteState,
    pub(in crate::pipeline::network) retired: AtomicBool,
    pub(in crate::pipeline::network) registered_address: parking_lot::Mutex<Option<WorkerAddress>>,
    /// Limits staged sends in flight at the caller. A send owns its permit,
    /// so the permit outlives a caller that stops waiting for the ACK.
    pub(in crate::pipeline::network) staged: Arc<Semaphore>,
}

impl Default for PeerState {
    fn default() -> Self {
        Self {
            ready: OnceCell::new(),
            active: AtomicUsize::new(0),
            last_used: AtomicU64::new(0),
            pinned: AtomicBool::new(false),
            reverse_pin_until: AtomicU64::new(0),
            confirmed_until: AtomicU64::new(0),
            route: RouteState::default(),
            retired: AtomicBool::new(false),
            registered_address: parking_lot::Mutex::new(None),
            staged: Arc::new(Semaphore::new(STAGED_WINDOW)),
        }
    }
}

impl PeerState {
    pub(in crate::pipeline::network) fn confirmed_route(&self, now: u64) -> Option<u64> {
        self.route
            .confirmed_epoch()
            .filter(|_| self.confirmed_until.load(Ordering::Acquire) > now)
    }

    pub(in crate::pipeline::network) fn register(
        &self,
        velo: &Velo,
        target: PeerInfo,
        request: bool,
    ) -> Result<()> {
        let mut recorded = self.registered_address.lock();
        anyhow::ensure!(
            !request || !self.retired.load(Ordering::Acquire),
            "Velo request peer has departed"
        );
        let address = target.worker_address().clone();
        velo.register_peer(target)?;
        *recorded = Some(address);
        Ok(())
    }
}

#[derive(Default)]
pub(in crate::pipeline::network) struct RouteState {
    epoch: AtomicU64,
    first_contact: Mutex<Option<(Instant, DynamoError)>>,
}

impl RouteState {
    pub(in crate::pipeline::network) fn confirmed_epoch(&self) -> Option<u64> {
        let epoch = self.epoch.load(Ordering::Acquire);
        (epoch & CONFIRMED != 0).then_some(epoch)
    }

    fn confirm(&self) -> u64 {
        self.epoch.fetch_or(CONFIRMED, Ordering::AcqRel) | CONFIRMED
    }

    pub(in crate::pipeline::network) fn invalidate(&self, epoch: u64) {
        // A delayed failure must leave any newer contact intact.
        let _ = self.epoch.compare_exchange(
            epoch,
            epoch.wrapping_add(1),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

pub(in crate::pipeline::network) fn register_peer_handler(
    service: &Arc<VeloResponseService>,
) -> Result<()> {
    let weak = Arc::downgrade(service);
    let handler = Handler::unary_handler(PEER_HANDLER, move |ctx| {
        let result = (|| {
            let identity = decode_identity(&ctx.payload)?;
            let id = identity.instance_id;
            check_identity(ctx.sender_instance_id(), ctx.sender_worker_id(), id)?;
            let address = decode_worker_address(identity.worker_address)?;
            let service = weak
                .upgrade()
                .ok_or_else(|| anyhow::anyhow!("worker is shutting down"))?;
            #[cfg(test)]
            service.peer_pin_calls.fetch_add(1, Ordering::Relaxed);
            loop {
                let peer = service.peer_state(id);
                let mut recorded = peer.registered_address.lock();
                let known = ctx.sender_instance_id();
                check_identity(known, ctx.sender_worker_id(), id)?;
                anyhow::ensure!(
                    known.is_none()
                        || recorded
                            .as_ref()
                            .is_none_or(|existing| existing == &address),
                    "identity already registered"
                );

                // The shard also orders this pin against the sweep's unregister.
                let Some(previous_pin) = service.pin_reverse_current(id, &peer) else {
                    continue;
                };
                if known.is_none() {
                    #[cfg(test)]
                    if let Some(hook) = service.before_request_peer_register.lock().clone() {
                        hook(id);
                    }
                    if let Err(error) = service
                        .velo
                        .register_peer(PeerInfo::new(id, address.clone()))
                    {
                        service.restore_reverse_pin(id, &peer, previous_pin);
                        return Err(error);
                    }
                    *recorded = Some(address);
                }
                return Ok(());
            }
        })();
        let ack = match result {
            Ok(()) => Bytes::new(),
            Err(error) => Bytes::from(format!("{ACK_UNAVAILABLE_PREFIX} {error}")),
        };
        Ok(Some(ack))
    })
    .build();
    service.velo.messenger().register_streaming_handler(handler)
}

fn check_identity(known: Option<InstanceId>, sender: WorkerId, claimed: InstanceId) -> Result<()> {
    // The frame identity is a consistency check. It comes from the sender.
    let matches = known.map_or_else(|| claimed.worker_id() == sender, |id| id == claimed);
    anyhow::ensure!(matches, "sender identity mismatch");
    Ok(())
}

pub(in crate::pipeline::network) async fn establish_route(
    service: &VeloResponseService,
    id: InstanceId,
    address: &str,
    deadline: Instant,
) -> Result<(Arc<PeerState>, u64), DynamoError> {
    establish(service, id, address, deadline, true).await
}

/// Looks the route up again for a caller that already spent part of its
/// deadline waiting locally. If its contact times out, that says nothing about
/// the worker, so the timeout is not cached for other callers.
pub(in crate::pipeline::network) async fn renew_route(
    service: &VeloResponseService,
    id: InstanceId,
    address: &str,
    deadline: Instant,
) -> Result<(Arc<PeerState>, u64), DynamoError> {
    establish(service, id, address, deadline, false).await
}

async fn establish(
    service: &VeloResponseService,
    id: InstanceId,
    address: &str,
    deadline: Instant,
    full_budget: bool,
) -> Result<(Arc<PeerState>, u64), DynamoError> {
    if let Some(route) = service.confirmed_peer(id) {
        return Ok(route);
    }

    let lease = service.reserve_request_peer(id)?;
    let peer = &lease.peer;
    let epoch = {
        let mut recent = by_deadline(deadline, peer.route.first_contact.lock()).await?;
        let now = service.peer_clock();
        if let Some(epoch) = peer.confirmed_route(now) {
            epoch
        } else {
            if let Some(epoch) = peer.route.confirmed_epoch() {
                // An expired route must renew its reverse pin. Cache connection
                // failures and peer refusals even when the previous route worked.
                peer.route.invalidate(epoch);
            }
            if let Some((failed_at, cause)) = recent.as_ref()
                && failed_at.elapsed() < FAILURE_TTL
            {
                return Err(cause.clone());
            }
            if let Err(cause) = contact(service, peer, id, address, deadline, HELLO_CAP).await {
                let cached = match cause.error_type() {
                    // A timeout before the caller's own deadline came from the
                    // contact cap, so it is the worker's whatever the budget.
                    ErrorType::ConnectionTimeout => full_budget || Instant::now() < deadline,
                    ErrorType::CannotConnect | ErrorType::WorkerUnavailable => true,
                    _ => false,
                };
                if cached {
                    *recent = Some((Instant::now(), cause.clone()));
                }
                return Err(cause);
            }
            *recent = None;
            let _address = peer.registered_address.lock();
            service
                .pin_current(id, peer)
                .map_err(|err| error(ErrorType::WorkerUnavailable, err.to_string()))?
                .ok_or_else(|| {
                    error(
                        ErrorType::WorkerUnavailable,
                        "Velo request peer has departed",
                    )
                })?;
            // Renew well before the worker's reverse pin can expire.
            peer.confirmed_until
                .store(now + PEER_IDLE_TTL.as_secs() / 2, Ordering::Release);
            peer.route.confirm()
        }
    };
    Ok((Arc::clone(peer), epoch))
}

async fn contact(
    service: &VeloResponseService,
    peer: &PeerState,
    id: InstanceId,
    address: &str,
    deadline: Instant,
    hello_cap: Duration,
) -> Result<(), DynamoError> {
    let _slot = service.ack_slots.try_acquire().map_err(|_| {
        error(
            ErrorType::ResourceExhausted,
            "Velo request ACK slots are full",
        )
    })?;
    let identity = service
        .request_identity
        .clone()
        .ok_or_else(|| error(ErrorType::InvalidArgument, "Velo request plane is disabled"))?;
    let address = decode_worker_address(address).map_err(|err| {
        error(
            ErrorType::InvalidArgument,
            format!("invalid Velo worker address: {err}"),
        )
    })?;
    peer.register(&service.velo, PeerInfo::new(id, address), true)
        .map_err(|source| {
            if peer.retired.load(Ordering::Acquire) {
                error(
                    ErrorType::WorkerUnavailable,
                    "Velo request peer has departed",
                )
            } else {
                transport_error(source)
            }
        })?;
    let ack = by_deadline(deadline.min(Instant::now() + hello_cap), async {
        service
            .velo
            .messenger()
            .unary_streaming(PEER_HANDLER)
            .raw_payload(identity)
            .send_to(id)
            .await
    })
    .await?
    .map_err(transport_error)?;
    if !ack.is_empty() {
        return Err(error(
            ErrorType::WorkerUnavailable,
            String::from_utf8_lossy(&ack).into_owned(),
        ));
    }

    // Fetch the normal handler list only once the worker has pinned our return route.
    let hello_deadline = deadline.min(Instant::now() + hello_cap);
    let handlers = by_deadline(hello_deadline, async {
        service.velo.refresh_handlers(id).await?;
        service.velo.available_handlers(id).await
    })
    .await?
    .map_err(transport_error)?;
    if !handlers.iter().any(|name| name == REQUEST_HANDLER) {
        return Err(error(
            ErrorType::WorkerUnavailable,
            "worker has no Velo request handler",
        ));
    }
    Ok(())
}

pub(in crate::pipeline::network) async fn by_deadline<T>(
    deadline: Instant,
    future: impl Future<Output = T>,
) -> Result<T, DynamoError> {
    tokio::select! {
        biased;
        _ = tokio::time::sleep_until(deadline) => Err(error(ErrorType::ConnectionTimeout, "Velo request timed out")),
        result = future => Ok(result),
    }
}

pub(super) fn transport_error(source: anyhow::Error) -> DynamoError {
    let kind = if source.is::<velo::RequestTimeout>() {
        ErrorType::ConnectionTimeout
    } else {
        ErrorType::CannotConnect
    };
    error(kind, source.to_string())
}

pub(super) fn error(kind: ErrorType, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(kind)
        .message(message)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::network::velo_request::encode_worker_address;
    use crate::pipeline::network::velo_response::{ResponseTransport, VeloServiceOptions};
    use tokio::sync::Notify;

    async fn request_peer() -> Arc<VeloResponseService> {
        VeloResponseService::new_with_options(
            ResponseTransport::Tcp,
            "127.0.0.1:0".parse().unwrap(),
            VeloServiceOptions {
                request_plane_enabled: true,
                ..Default::default()
            },
        )
        .await
        .unwrap()
    }

    fn stall_handler(
        service: &VeloResponseService,
        name: &'static str,
    ) -> (Arc<AtomicUsize>, Arc<Notify>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(Notify::new());
        let count = calls.clone();
        let barrier = release.clone();
        service
            .velo
            .messenger()
            .register_streaming_handler(
                Handler::unary_handler_async(name, move |_| {
                    count.fetch_add(1, Ordering::Relaxed);
                    let barrier = barrier.clone();
                    async move {
                        barrier.notified().await;
                        Ok(None)
                    }
                })
                .build(),
            )
            .unwrap();
        (calls, release)
    }

    #[test]
    fn exact_sender_id_takes_precedence_over_the_frame_hash() {
        let claimed = InstanceId::new_v4();
        let known = InstanceId::new_v4();
        assert!(check_identity(Some(known), claimed.worker_id(), claimed).is_err());
        assert!(check_identity(Some(claimed), known.worker_id(), claimed).is_ok());
        assert!(check_identity(None, claimed.worker_id(), claimed).is_ok());
        assert!(check_identity(None, known.worker_id(), claimed).is_err());
    }

    async fn assert_contact_timeout(handler: &'static str, deadline: Duration, cap: Duration) {
        let frontend = request_peer().await;
        let worker = request_peer().await;
        if handler == PEER_HANDLER {
            worker
                .velo
                .register_peer(frontend.velo.peer_info())
                .unwrap();
        }
        let (seen, release) = stall_handler(&worker, handler);
        let target = worker.velo.peer_info();
        let address = encode_worker_address(target.worker_address()).unwrap();
        let peer = frontend.reserve_peer(target.instance_id());
        let start = Instant::now();
        let result = tokio::time::timeout(
            Duration::from_millis(850),
            contact(
                &frontend,
                &peer.peer,
                target.instance_id(),
                &address,
                start + deadline,
                cap,
            ),
        )
        .await;
        release.notify_waiters();
        assert_eq!(
            result
                .expect("contact exceeded its shorter timer")
                .unwrap_err()
                .error_type(),
            ErrorType::ConnectionTimeout
        );
        assert!(start.elapsed() >= deadline.min(cap));
        assert_eq!(seen.load(Ordering::Relaxed), 1);
        assert_eq!(frontend.velo.response_slots_in_use(), 0);
    }

    #[tokio::test]
    async fn silent_hello_is_bounded_below_the_request_deadline() {
        assert_contact_timeout("_hello", Duration::from_secs(2), Duration::from_millis(200)).await;
    }

    #[tokio::test]
    async fn staged_queue_exhaustion_during_route_renewal_is_local() {
        use crate::{
            pipeline::network::{
                egress::unified_client::RequestPlaneClient,
                velo_request::{client::VeloRequestClient, encode_address},
            },
            protocols::EndpointId,
        };

        let frontend = request_peer().await;
        let worker = request_peer().await;
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, |_| Ok(Some(Bytes::new()))).build(),
            )
            .unwrap();
        let info = worker.velo.peer_info();
        let id = info.instance_id();
        let address = encode_worker_address(info.worker_address()).unwrap();
        let target = encode_address(
            &info,
            &EndpointId {
                namespace: "ns".into(),
                component: "worker".into(),
                name: "generate".into(),
            },
            1,
        )
        .unwrap();
        let timeout = Duration::from_secs(2);
        let client = VeloRequestClient::new(frontend.clone(), timeout);
        client
            .send_request(target.clone(), Bytes::new(), Default::default())
            .await
            .unwrap();
        let (peer, _) = frontend.confirmed_peer(id).unwrap();
        let reserved = peer.staged.try_acquire_many(STAGED_WINDOW as u32).unwrap();
        tokio::time::pause();
        let until_refresh = peer.confirmed_until.load(Ordering::Acquire) - frontend.peer_clock();
        tokio::time::advance(Duration::from_secs(until_refresh - 1)).await;
        let mut staged = client.send_request(
            target.clone(),
            Bytes::from(vec![0; 300 * 1024]),
            Default::default(),
        );
        assert!(futures::poll!(&mut staged).is_pending());
        tokio::time::advance(timeout - Duration::from_millis(5)).await;

        // Another caller starts the periodic handshake while this request is still queued.
        let mut renewal = std::pin::pin!(establish_route(
            &frontend,
            id,
            &address,
            Instant::now() + timeout
        ));
        assert!(futures::poll!(&mut renewal).is_pending());
        assert!(peer.route.confirmed_epoch().is_none());
        drop(reserved);
        let result = match futures::poll!(&mut staged) {
            std::task::Poll::Ready(result) => result,
            std::task::Poll::Pending => {
                tokio::time::advance(Duration::from_millis(10)).await;
                staged.await
            }
        };
        tokio::time::resume();
        renewal.await.unwrap();
        client
            .send_request(target, Bytes::new(), Default::default())
            .await
            .unwrap();
        assert_eq!(
            result
                .unwrap_err()
                .downcast::<DynamoError>()
                .unwrap()
                .error_type(),
            ErrorType::ResourceExhausted,
            "the staged window consumed the send budget while the healthy route was renewing"
        );
    }

    #[tokio::test]
    async fn a_short_queued_renewal_does_not_inhibit_a_worker_for_a_fresh_request() {
        use crate::{
            DistributedRuntime, Runtime,
            component::TransportType,
            discovery::DiscoverySpec,
            distributed::{DistributedConfig, RequestPlaneMode},
            pipeline::{
                SingleIn,
                network::{
                    ResponsePlaneMode,
                    egress::{
                        push_router::{PushRouter, RouterMode},
                        unified_client::RequestPlaneClient,
                    },
                    velo_request::{client::VeloRequestClient, encode_address},
                },
            },
            protocols::annotated::Annotated,
        };
        let owner = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(
            owner.clone(),
            DistributedConfig {
                request_plane: RequestPlaneMode::Velo,
                response_plane: Some(ResponsePlaneMode::Velo),
                ..DistributedConfig::process_local()
            },
        )
        .await
        .unwrap();
        let frontend = drt.velo_response_service().await.unwrap();
        let worker = request_peer().await;
        let calls = Arc::new(AtomicUsize::new(0));
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, {
                    let calls = calls.clone();
                    move |ctx| {
                        let ack = if ctx.payload.is_empty() {
                            Bytes::new()
                        } else {
                            calls.fetch_add(1, Ordering::Relaxed);
                            Bytes::from_static(b"Server overloaded: test worker")
                        };
                        Ok(Some(ack))
                    }
                })
                .build(),
            )
            .unwrap();
        let endpoint = drt
            .namespace("queued_renewal")
            .unwrap()
            .component("worker")
            .unwrap()
            .endpoint("generate");
        let endpoint_id = endpoint.id();
        let discovery_id = drt.discovery().instance_id();
        let info = worker.velo.peer_info();
        let target = encode_address(&info, &endpoint_id, discovery_id).unwrap();
        drt.discovery()
            .register(DiscoverySpec::Endpoint {
                namespace: endpoint_id.namespace,
                component: endpoint_id.component,
                endpoint: endpoint_id.name,
                transport: TransportType::Velo(target.clone()),
                device_type: None,
                request_plane_codec: None,
            })
            .await
            .unwrap();
        let client = endpoint
            .client_with_inhibited_duration(Duration::from_secs(10))
            .await
            .unwrap();
        client.wait_for_instances().await.unwrap();
        let router = PushRouter::<String, Annotated<String>>::from_client(
            client.clone(),
            RouterMode::Direct,
        )
        .await
        .unwrap();
        let budget = Duration::from_secs(2);
        let transport = VeloRequestClient::new(frontend.clone(), budget);
        let eager = || transport.send_request(target.clone(), Bytes::new(), Default::default());
        eager().await.unwrap();
        let (peer, _) = frontend.confirmed_peer(info.instance_id()).unwrap();
        let reserved = peer.staged.try_acquire_many(STAGED_WINDOW as u32).unwrap();
        let (handshakes, gate) = stall_handler(&worker, PEER_HANDLER);

        tokio::time::pause();
        let until_refresh = peer.confirmed_until.load(Ordering::Acquire) - frontend.peer_clock();
        tokio::time::advance(Duration::from_secs(until_refresh - 1)).await;
        let mut queued = transport.send_request(
            target.clone(),
            Bytes::from(vec![1; 300 * 1024]),
            Default::default(),
        );
        assert!(futures::poll!(&mut queued).is_pending());
        tokio::time::advance(budget - Duration::from_millis(100)).await;
        drop(reserved);
        assert!(futures::poll!(&mut queued).is_pending());
        tokio::time::resume();
        let local_error = queued.await.unwrap_err().downcast::<DynamoError>().unwrap();
        assert_eq!(local_error.error_type(), ErrorType::ResourceExhausted);
        assert_eq!(handshakes.load(Ordering::Relaxed), 1);
        gate.notify_waiters();
        register_peer_handler(&worker).unwrap();

        let error = router
            .direct(SingleIn::new("fresh".to_string()), discovery_id)
            .await
            .expect_err("the worker returns its overload ACK for nonempty payloads");
        let eligible = client.instance_ids_avail().contains(&discovery_id);
        let received = calls.load(Ordering::Relaxed);
        tokio::time::pause();
        tokio::time::advance(FAILURE_TTL + Duration::from_millis(1)).await;
        tokio::time::resume();
        eager().await.unwrap();
        owner.shutdown();
        assert!(
            eligible && received == 1,
            "fresh request after a queued renewal timed out: worker eligible {eligible}, requests received {received}, error {:?}",
            error.downcast_ref::<DynamoError>().unwrap().error_type()
        );
    }

    #[tokio::test]
    async fn silent_hello_obeys_a_request_deadline_shorter_than_the_cap() {
        assert_contact_timeout("_hello", Duration::from_millis(200), Duration::from_secs(2)).await;
    }

    #[tokio::test]
    async fn silent_pin_ack_has_its_own_cap() {
        assert_contact_timeout(
            PEER_HANDLER,
            Duration::from_secs(2),
            Duration::from_millis(200),
        )
        .await;
    }

    #[tokio::test]
    async fn silent_pin_ack_obeys_a_request_deadline_shorter_than_the_cap() {
        assert_contact_timeout(
            PEER_HANDLER,
            Duration::from_millis(200),
            Duration::from_secs(2),
        )
        .await;
    }

    #[tokio::test]
    async fn a_queued_renewal_at_the_handshake_cap_shares_the_worker_timeout() {
        use crate::{
            pipeline::network::{
                egress::unified_client::RequestPlaneClient,
                velo_request::{client::VeloRequestClient, encode_address},
            },
            protocols::EndpointId,
        };

        let frontend = request_peer().await;
        let worker = request_peer().await;
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, |_| Ok(Some(Bytes::new()))).build(),
            )
            .unwrap();
        let info = worker.velo.peer_info();
        let target = encode_address(
            &info,
            &EndpointId {
                namespace: "ns".into(),
                component: "worker".into(),
                name: "generate".into(),
            },
            1,
        )
        .unwrap();
        let budget = Duration::from_secs(60);
        let client = VeloRequestClient::new(frontend.clone(), budget);
        let eager = || client.send_request(target.clone(), Bytes::new(), Default::default());
        eager().await.unwrap();
        let (peer, _) = frontend.confirmed_peer(info.instance_id()).unwrap();
        let full = peer.staged.try_acquire_many(STAGED_WINDOW as u32).unwrap();
        let (contacts, release) = stall_handler(&worker, PEER_HANDLER);
        let wait_for_contact = |n| {
            let contacts = contacts.clone();
            async move {
                tokio::time::timeout(Duration::from_secs(2), async {
                    while contacts.load(Ordering::Relaxed) < n {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                })
                .await
                .expect("worker did not receive the pin request");
            }
        };
        let advance_past_cap = || tokio::time::advance(HELLO_CAP + Duration::from_millis(1));

        // The confirmed route comes due while a large request waits in the full window.
        tokio::time::pause();
        let until_refresh = peer.confirmed_until.load(Ordering::Acquire) - frontend.peer_clock();
        tokio::time::advance(Duration::from_secs(until_refresh - 1)).await;
        let queued_deadline = Instant::now() + budget;
        let mut queued = client.send_request(
            target.clone(),
            Bytes::from(vec![1; 300 * 1024]),
            Default::default(),
        );
        assert!(futures::poll!(&mut queued).is_pending());
        tokio::time::advance(Duration::from_secs(2)).await;
        drop(full);
        assert!(futures::poll!(&mut queued).is_pending());
        tokio::time::resume();
        wait_for_contact(1).await;
        tokio::time::pause();
        advance_past_cap().await;
        let error = queued.await.unwrap_err().downcast::<DynamoError>().unwrap();
        assert_eq!(error.error_type(), ErrorType::ResourceExhausted);
        let left = queued_deadline.saturating_duration_since(Instant::now());
        assert!(left > HELLO_CAP, "the caller deadline ended the handshake");

        let start = Instant::now();
        let mut fresh = eager();
        let result = match futures::poll!(&mut fresh) {
            std::task::Poll::Ready(result) => result,
            std::task::Poll::Pending => {
                tokio::time::resume();
                wait_for_contact(2).await;
                tokio::time::pause();
                advance_past_cap().await;
                fresh.await
            }
        };
        assert_eq!(
            result
                .unwrap_err()
                .downcast::<DynamoError>()
                .unwrap()
                .error_type(),
            ErrorType::ConnectionTimeout
        );
        let extra_wait = start.elapsed();
        let mut next = eager();
        assert!(
            matches!(futures::poll!(&mut next), std::task::Poll::Ready(Err(_))),
            "ordinary first contact must cache its genuine timeout"
        );
        let count = contacts.load(Ordering::Relaxed);
        release.notify_waiters();
        tokio::time::resume();
        assert!(
            count == 1 && extra_wait < FAILURE_TTL,
            "worker pin reached its 25 s cap with {left:?} of the queued caller's budget left; the fresh caller waited {extra_wait:?}, worker saw {count} pin requests"
        );
    }

    #[tokio::test]
    async fn waiters_share_a_recent_first_contact_failure() {
        let frontend = request_peer().await;
        let worker = request_peer().await;
        let (seen, release) = stall_handler(&worker, PEER_HANDLER);
        let target = worker.velo.peer_info();
        let id = target.instance_id();
        let address = encode_worker_address(target.worker_address()).unwrap();
        let start = Instant::now();
        let (leader, waiters) = tokio::join!(
            establish_route(&frontend, id, &address, start + Duration::from_millis(250)),
            futures::future::join_all((0..12).map(|_| establish_route(
                &frontend,
                id,
                &address,
                start + Duration::from_millis(750)
            ))),
        );
        release.notify_waiters();
        let error = leader.err().unwrap();
        assert_eq!(error.error_type(), ErrorType::ConnectionTimeout);
        assert!(
            waiters
                .into_iter()
                .all(|result| result.err().is_some_and(|cause| cause == error))
        );
        assert_eq!(seen.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn a_staged_waiter_renews_a_reverse_pin_that_expired_in_the_queue() {
        use crate::{
            engine::AsyncEngineContextProvider,
            pipeline::{
                Context,
                network::{
                    egress::unified_client::RequestPlaneClient,
                    velo_request::{client::VeloRequestClient, encode_address},
                },
            },
            protocols::EndpointId,
        };

        let frontend = request_peer().await;
        let worker = request_peer().await;
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, |_| Ok(Some(Bytes::new()))).build(),
            )
            .unwrap();
        let info = worker.velo.peer_info();
        let worker_id = info.instance_id();
        let frontend_id = frontend.velo.instance_id();
        let target = encode_address(
            &info,
            &EndpointId {
                namespace: "ns".into(),
                component: "worker".into(),
                name: "generate".into(),
            },
            1,
        )
        .unwrap();
        let registration = frontend
            .register_response(Context::new(()).context(), false)
            .unwrap();
        frontend.hold_request_peer(registration.registration_id().unwrap(), &target);
        let client =
            VeloRequestClient::new(frontend.clone(), PEER_IDLE_TTL + Duration::from_secs(5));
        client
            .send_request(target.clone(), Bytes::new(), Default::default())
            .await
            .unwrap();
        let (peer, _) = frontend.confirmed_peer(worker_id).unwrap();
        let reserved = peer.staged.try_acquire_many(STAGED_WINDOW as u32).unwrap();
        tokio::time::pause();
        let mut staged = client.send_request(
            target.clone(),
            Bytes::from(vec![0; 300 * 1024]),
            Default::default(),
        );
        assert!(futures::poll!(&mut staged).is_pending());
        assert!(worker.velo.primary_transport_key(frontend_id).is_some());
        tokio::time::advance(PEER_IDLE_TTL + Duration::from_secs(2)).await;
        for _ in 0..32 {
            if worker.velo.primary_transport_key(frontend_id).is_none() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(worker.velo.primary_transport_key(frontend_id).is_none());
        assert!(frontend.confirmed_peer(worker_id).is_none());
        assert!(frontend.velo.primary_transport_key(worker_id).is_some());

        tokio::time::resume();
        drop(reserved);
        let result = staged.await;
        let pin_calls_at_ack = worker.peer_pin_calls.load(Ordering::Relaxed);
        tokio::time::timeout(
            Duration::from_secs(3),
            client.send_request(target, Bytes::new(), Default::default()),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(
            result.is_ok() && pin_calls_at_ack == 2,
            "the worker's reverse pin expired during the queue: staged result {result:?}, pin calls {pin_calls_at_ack}"
        );
    }
}
