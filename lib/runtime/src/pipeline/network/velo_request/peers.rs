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
    pipeline::network::{ACK_UNAVAILABLE_PREFIX, velo_response::VeloResponseService},
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
    pub(in crate::pipeline::network) route: RouteState,
    pub(in crate::pipeline::network) retired: AtomicBool,
    pub(in crate::pipeline::network) registered_address: parking_lot::Mutex<Option<WorkerAddress>>,
    /// Limits staged sends in flight at the caller.
    pub(in crate::pipeline::network) staged: Semaphore,
}

impl Default for PeerState {
    fn default() -> Self {
        Self {
            ready: OnceCell::new(),
            active: AtomicUsize::new(0),
            last_used: AtomicU64::new(0),
            pinned: AtomicBool::new(false),
            route: RouteState::default(),
            retired: AtomicBool::new(false),
            registered_address: parking_lot::Mutex::new(None),
            staged: Semaphore::new(STAGED_WINDOW),
        }
    }
}

impl PeerState {
    pub(in crate::pipeline::network) fn register(
        &self,
        velo: &Velo,
        target: PeerInfo,
    ) -> Result<()> {
        let mut recorded = self.registered_address.lock();
        anyhow::ensure!(
            !self.retired.load(Ordering::Acquire),
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
                let Some(was_pinned) = service.pin_current(id, &peer)? else {
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
                        if !was_pinned {
                            service.rollback_pin(id, &peer);
                        }
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
    if let Some(route) = service.confirmed_peer(id) {
        return Ok(route);
    }

    let lease = service.reserve_peer(id);
    let peer = &lease.peer;
    let epoch = {
        let mut recent = by_deadline(deadline, peer.route.first_contact.lock()).await?;
        if let Some(epoch) = peer.route.confirmed_epoch() {
            epoch
        } else {
            if let Some((failed_at, cause)) = recent.as_ref()
                && failed_at.elapsed() < FAILURE_TTL
            {
                return Err(cause.clone());
            }
            if let Err(cause) = contact(service, peer, id, address, deadline, HELLO_CAP).await {
                if matches!(
                    cause.error_type(),
                    ErrorType::ConnectionTimeout
                        | ErrorType::CannotConnect
                        | ErrorType::WorkerUnavailable
                ) {
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
    peer.register(&service.velo, PeerInfo::new(id, address))
        .map_err(transport_error)?;
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
        _ = tokio::time::sleep_until(deadline) => Err(error(ErrorType::ConnectionTimeout, "Velo first contact timed out")),
        result = future => Ok(result),
    }
}

fn transport_error(source: anyhow::Error) -> DynamoError {
    let kind = if source.is::<velo::RequestTimeout>() {
        ErrorType::ConnectionTimeout
    } else {
        ErrorType::CannotConnect
    };
    error(kind, source.to_string())
}

fn error(kind: ErrorType, message: impl Into<String>) -> DynamoError {
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
}
