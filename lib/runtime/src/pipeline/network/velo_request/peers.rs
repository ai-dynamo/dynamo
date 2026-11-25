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
use velo::{Handler, PeerInfo};

use super::{
    PEER_HANDLER, REQUEST_HANDLER, STAGED_WINDOW, decode_identity, decode_worker_address,
    encode_identity,
};
use crate::{
    error::{DynamoError, ErrorType},
    pipeline::network::{ACK_UNAVAILABLE_PREFIX, velo_response::VeloResponseService},
};

// Keep this below Velo's 30-second hello timeout.
pub(in crate::pipeline::network) const HELLO_CAP: Duration = Duration::from_secs(25);
const CONFIRMED: u64 = 1;

pub(in crate::pipeline::network) struct PeerState {
    pub(in crate::pipeline::network) ready: OnceCell<()>,
    pub(in crate::pipeline::network) active: AtomicUsize,
    pub(in crate::pipeline::network) last_used: AtomicU64,
    pub(in crate::pipeline::network) pinned: AtomicBool,
    pub(in crate::pipeline::network) route: RouteState,
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
            staged: Semaphore::new(STAGED_WINDOW),
        }
    }
}

#[derive(Default)]
pub(in crate::pipeline::network) struct RouteState {
    epoch: AtomicU64,
    first_contact: Mutex<()>,
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
            anyhow::ensure!(
                identity.instance_id.worker_id() == ctx.sender_worker_id(),
                "sender identity mismatch"
            );
            let address = decode_worker_address(identity.worker_address)?;
            let service = weak
                .upgrade()
                .ok_or_else(|| anyhow::anyhow!("worker is shutting down"))?;
            #[cfg(test)]
            service.peer_pin_calls.fetch_add(1, Ordering::Relaxed);

            // A sweep may have just removed the address. Pin before restoring it.
            service.pin(identity.instance_id);
            service
                .velo
                .register_peer(PeerInfo::new(identity.instance_id, address))
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

pub(in crate::pipeline::network) async fn establish_route(
    service: &VeloResponseService,
    target: &PeerInfo,
    deadline: Instant,
) -> Result<(Arc<PeerState>, u64), DynamoError> {
    let id = target.instance_id();
    if let Some(route) = service.confirmed_peer(id) {
        return Ok(route);
    }

    let peer = service.pin(id);
    let epoch = {
        let _contact = by_deadline(deadline, peer.route.first_contact.lock()).await?;
        if let Some(epoch) = peer.route.confirmed_epoch() {
            epoch
        } else {
            contact(service, target, deadline, HELLO_CAP).await?;
            peer.route.confirm()
        }
    };
    Ok((peer, epoch))
}

async fn contact(
    service: &VeloResponseService,
    target: &PeerInfo,
    deadline: Instant,
    hello_cap: Duration,
) -> Result<(), DynamoError> {
    let _slot = service.ack_slots.try_acquire().map_err(|_| {
        error(
            ErrorType::ResourceExhausted,
            "Velo request ACK slots are full",
        )
    })?;
    let identity = encode_identity(&service.velo.peer_info()).map_err(|err| {
        error(
            ErrorType::InvalidArgument,
            format!("cannot encode Velo peer identity: {err}"),
        )
    })?;
    let id = target.instance_id();
    service
        .velo
        .register_peer(target.clone())
        .map_err(transport_error)?;
    let ack = by_deadline(deadline, async {
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
    use crate::pipeline::network::velo_response::ResponseTransport;
    use tokio::sync::Notify;

    #[tokio::test]
    async fn silent_hello_is_bounded_below_the_request_deadline() {
        let address = "127.0.0.1:0".parse().unwrap();
        let frontend = VeloResponseService::new(ResponseTransport::Tcp, address)
            .await
            .unwrap();
        let worker = VeloResponseService::new(ResponseTransport::Tcp, address)
            .await
            .unwrap();
        let release = Arc::new(Notify::new());
        let seen = Arc::new(AtomicUsize::new(0));
        {
            let release = release.clone();
            let seen = seen.clone();
            worker
                .velo
                .messenger()
                .register_streaming_handler(
                    Handler::unary_handler_async("_hello", move |_| {
                        let release = release.clone();
                        async move {
                            release.notified().await;
                            Ok(None)
                        }
                    })
                    .build(),
                )
                .unwrap();
            worker
                .velo
                .register_handler(
                    Handler::unary_handler(REQUEST_HANDLER, move |_| {
                        seen.fetch_add(1, Ordering::Relaxed);
                        Ok(None)
                    })
                    .build(),
                )
                .unwrap();
        }
        let start = Instant::now();
        let error = contact(
            &frontend,
            &worker.velo.peer_info(),
            start + Duration::from_secs(3),
            Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        let elapsed = start.elapsed();
        release.notify_one();
        assert_eq!(error.error_type(), ErrorType::ConnectionTimeout);
        assert!(
            elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(3),
            "hello elapsed {elapsed:?}"
        );
        assert_eq!(worker.peer_pin_calls.load(Ordering::Relaxed), 1);
        assert_eq!(seen.load(Ordering::Relaxed), 0);
        assert_eq!(frontend.velo.response_slots_in_use(), 0);
    }
}
