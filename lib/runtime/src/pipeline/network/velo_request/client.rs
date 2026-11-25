// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use tokio::{
    sync::{OwnedSemaphorePermit, TryAcquireError},
    time::Instant,
};
use tracing::Instrument;
use velo::InstanceId;

use crate::{
    error::{DynamoError, ErrorType},
    pipeline::network::{
        VELO_SCHEME,
        egress::unified_client::{Headers, RequestPlaneClient},
        get_tcp_max_message_size,
        velo_response::{RuntimeService, VeloResponseService},
    },
};

use super::{
    REQUEST_HANDLER, decode_address,
    discovery::DiscoveryReady,
    peers::{PeerState, by_deadline, error, establish_route, transport_error},
    server::ENDPOINT_PATH_HEADER,
};

const MAX_HEADER_VALUE: usize = 1024;
// Velo reserves part of its 16 KiB header map for rendezvous metadata.
const HEADER_BUDGET: usize = 16 * 1024 - 512;
const MIN_STAGED_SEND_BUDGET: Duration = Duration::from_millis(10);

pub(in crate::pipeline::network) struct VeloRequestClient {
    service: Arc<VeloResponseService>,
    ack_timeout: Duration,
    discovery_ready: Option<Arc<DiscoveryReady>>,
    owner: Option<Arc<RuntimeService>>,
}

impl VeloRequestClient {
    pub(in crate::pipeline::network) fn new(
        service: Arc<VeloResponseService>,
        ack_timeout: Duration,
    ) -> Self {
        Self {
            service,
            ack_timeout,
            discovery_ready: None,
            owner: None,
        }
    }

    /// Ties the client to the runtime that created it. Another runtime in the
    /// process can keep the shared service alive after this one shuts down.
    pub(in crate::pipeline::network) fn with_owner(mut self, owner: Arc<RuntimeService>) -> Self {
        self.owner = Some(owner);
        self
    }

    fn owner_released(&self) -> bool {
        self.owner.as_ref().is_some_and(|owner| owner.is_released())
    }

    pub(in crate::pipeline::network) fn with_discovery(
        mut self,
        ready: Arc<DiscoveryReady>,
    ) -> Self {
        self.discovery_ready = Some(ready);
        self
    }
}

#[async_trait]
impl RequestPlaneClient for VeloRequestClient {
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        mut headers: Headers,
    ) -> Result<Bytes> {
        let deadline = Instant::now()
            .checked_add(self.ack_timeout)
            .ok_or_else(|| {
                error(
                    ErrorType::Internal,
                    "DYN_TCP_REQUEST_TIMEOUT exceeds the timer range",
                )
            })?;
        if self.owner_released() {
            return Err(error(ErrorType::Unavailable, "Velo request runtime is shut down").into());
        }
        if !address.starts_with(VELO_SCHEME) {
            return Err(error(
                ErrorType::Internal,
                "Velo request client requires a velo:// address. Check the local request plane configuration",
            ).into());
        }
        let target = decode_address(&address).map_err(|source| {
            error(
                ErrorType::InvalidArgument,
                format!("invalid Velo request address: {source:#}"),
            )
        })?;
        let limit = get_tcp_max_message_size();
        if payload.len() > limit {
            return Err(error(
                ErrorType::InvalidArgument,
                format!("request payload exceeds DYN_TCP_MAX_MESSAGE_SIZE ({limit} bytes)"),
            )
            .into());
        }
        headers.insert(ENDPOINT_PATH_HEADER.into(), target.path.into());
        check_headers(&headers)?;
        if target.instance_id == self.service.velo.instance_id() {
            return self.service.request_slot.handle_local(payload, headers);
        }

        if let Some(ready) = self
            .discovery_ready
            .as_ref()
            .filter(|ready| !ready.is_established())
        {
            by_deadline(deadline, ready.wait()).await.map_err(|_| {
                error(
                    ErrorType::Unavailable,
                    "Velo request discovery did not initialize before the request deadline",
                )
            })?;
        }
        let (mut peer, mut epoch) = establish_route(
            &self.service,
            target.instance_id,
            target.worker_address,
            deadline,
        )
        .await?;
        if payload.len() <= self.service.velo.staging_threshold() {
            let _slot = self
                .service
                .ack_slots
                .try_acquire()
                .map_err(|_| ack_slots_full())?;
            let ack = send(
                &self.service,
                target.instance_id,
                payload,
                headers,
                deadline,
            )
            .await;
            if ack.is_err() {
                peer.route.invalidate(epoch);
            }
            return Ok(ack?);
        }

        // The staged window bounds local capacity at each peer. Only requests
        // that wait for a permit spend part of their ACK budget on this queue.
        let mut waited = false;
        let permit = loop {
            let (permit, queued) = staged_permit(&peer, deadline).await?;
            waited |= queued;
            // The route can change during the wait: the worker left discovery,
            // or another request found the route broken or renewed it.
            if !queued || peer.route.confirmed_epoch() == Some(epoch) {
                break permit;
            }
            drop(permit);
            (peer, epoch) = establish_route(
                &self.service,
                target.instance_id,
                target.worker_address,
                deadline,
            )
            .await?;
        };
        if waited && deadline.saturating_duration_since(Instant::now()) < MIN_STAGED_SEND_BUDGET {
            return Err(error(
                ErrorType::ResourceExhausted,
                "staged window wait left too little time to send the Velo request",
            )
            .into());
        }
        let slot = Arc::clone(&self.service.ack_slots)
            .try_acquire_owned()
            .map_err(|_| ack_slots_full())?;
        // Velo keeps a staged payload until the worker fetches it or the send
        // times out, whatever the caller does. The send therefore runs in its
        // own task and keeps its place in the window if the caller goes away.
        let service = self.service.clone();
        let id = target.instance_id;
        let ack = tokio::spawn(
            async move {
                let _held = (permit, slot);
                send(&service, id, payload, headers, deadline).await
            }
            .in_current_span(),
        )
        .await
        .map_err(|cause| {
            error(
                ErrorType::Internal,
                format!("staged Velo send failed: {cause}"),
            )
        })?;
        match ack {
            Ok(ack) => Ok(ack),
            // The deadline includes time spent waiting for local capacity.
            // A timeout with that reduced budget leaves the peer route valid.
            Err(cause) if waited && cause.error_type() == ErrorType::ConnectionTimeout => {
                Err(error(
                    ErrorType::ResourceExhausted,
                    "Velo request timed out after waiting for the staged window",
                )
                .into())
            }
            Err(cause) => {
                peer.route.invalidate(epoch);
                Err(cause.into())
            }
        }
    }

    fn transport_name(&self) -> &'static str {
        "velo"
    }

    fn is_healthy(&self) -> bool {
        !self.service.is_closing() && !self.owner_released()
    }
}

/// Takes a place in the peer's staged window, waiting until the deadline for
/// one. The flag says whether the caller had to wait.
async fn staged_permit(
    peer: &PeerState,
    deadline: Instant,
) -> Result<(OwnedSemaphorePermit, bool), DynamoError> {
    let closed = || error(ErrorType::Internal, "staged window closed");
    match Arc::clone(&peer.staged).try_acquire_owned() {
        Ok(permit) => Ok((permit, false)),
        Err(TryAcquireError::Closed) => Err(closed()),
        Err(TryAcquireError::NoPermits) => {
            let permit = by_deadline(deadline, Arc::clone(&peer.staged).acquire_owned())
                .await
                .map_err(|_| {
                    error(
                        ErrorType::ResourceExhausted,
                        "staged window to worker stayed full until the request deadline",
                    )
                })?
                .map_err(|_| closed())?;
            Ok((permit, true))
        }
    }
}

async fn send(
    service: &VeloResponseService,
    id: InstanceId,
    payload: Bytes,
    headers: Headers,
    deadline: Instant,
) -> Result<Bytes, DynamoError> {
    by_deadline(deadline, async {
        service
            .velo
            .unary(REQUEST_HANDLER)?
            .raw_payload(payload)
            .headers(headers)
            .timeout(deadline.saturating_duration_since(Instant::now()))
            .send_to(id)
            .await
    })
    .await
    .and_then(|ack| ack.map_err(transport_error))
}

fn ack_slots_full() -> DynamoError {
    error(
        ErrorType::ResourceExhausted,
        "Velo request ACK slots are full",
    )
}

fn check_headers(headers: &Headers) -> Result<()> {
    // Msgpack uses at most five bytes for each string or map prefix.
    let mut upper_bound = 5;
    for (key, value) in headers {
        if value.len() > MAX_HEADER_VALUE {
            return Err(error(
                ErrorType::InvalidArgument,
                format!("Velo request header {key:.80} exceeds {MAX_HEADER_VALUE} bytes"),
            )
            .into());
        }
        if matches!(key.as_str(), "_rv" | "_rv_budget_ms") {
            return Err(error(
                ErrorType::InvalidArgument,
                "Velo request sets a reserved rendezvous header",
            )
            .into());
        }
        upper_bound += key.len() + value.len() + 10;
    }
    if upper_bound > HEADER_BUDGET && rmp_serde::to_vec(headers)?.len() > HEADER_BUDGET {
        return Err(error(
            ErrorType::InvalidArgument,
            format!("Velo request headers exceed {HEADER_BUDGET} bytes"),
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::VeloRequestClient;
    use crate::{
        DistributedRuntime, Runtime,
        component::TransportType,
        discovery::DiscoverySpec,
        distributed::{DistributedConfig, RequestPlaneMode},
        error::{DynamoError, ErrorType},
        pipeline::{
            SingleIn,
            network::{
                ResponsePlaneMode,
                egress::{
                    push_router::{PushRouter, RouterMode},
                    tcp_client::TcpRequestConfig,
                    unified_client::{Headers, RequestPlaneClient},
                },
                get_tcp_max_message_size,
                velo_request::{
                    REQUEST_HANDLER, STAGED_WINDOW, decode_address, peers::establish_route,
                    server::ENDPOINT_PATH_HEADER,
                },
                velo_response::VeloResponseService,
            },
        },
        protocols::annotated::Annotated,
    };
    use crate::{
        pipeline::network::velo_request::{ACK_SLOTS, encode_address, encode_worker_address},
        pipeline::network::velo_response::{ResponseTransport, VeloServiceOptions},
        protocols::EndpointId,
    };
    use anyhow::Result;
    use bytes::Bytes;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::{future::Future, sync::Arc, task::Poll, time::Duration};
    use tokio::sync::Semaphore;
    use tokio::time::Instant;
    use velo::{Handler, PeerInfo};

    async fn bounded<T>(future: impl Future<Output = T>) -> T {
        tokio::time::timeout(Duration::from_secs(5), future)
            .await
            .expect("operation stalled")
    }

    async fn service() -> Arc<VeloResponseService> {
        VeloResponseService::new_with_options(
            ResponseTransport::Tcp,
            "127.0.0.1:0".parse().unwrap(),
            VeloServiceOptions {
                request_plane_enabled: true,
                staged_payload_max: Some(get_tcp_max_message_size()),
            },
        )
        .await
        .unwrap()
    }

    fn address(peer: &PeerInfo) -> String {
        encode_address(
            peer,
            &EndpointId {
                namespace: "ns".into(),
                component: "worker".into(),
                name: "generate".into(),
            },
            1,
        )
        .unwrap()
    }

    fn error_type(result: Result<Bytes>) -> ErrorType {
        let error = result.unwrap_err();
        error
            .downcast_ref::<DynamoError>()
            .unwrap_or_else(|| panic!("{error:#}"))
            .error_type()
    }

    #[tokio::test]
    async fn live_requests_renew_reverse_pin_after_refresh_interval() {
        let frontend = service().await;
        let worker = service().await;
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, |_| Ok(Some(Bytes::from_static(b"ack"))))
                    .build(),
            )
            .unwrap();
        let client = VeloRequestClient::new(frontend.clone(), Duration::from_secs(2));
        let target = address(&worker.velo.peer_info());
        let send = || client.send_request(target.clone(), Bytes::new(), Headers::new());
        assert_eq!(send().await.unwrap(), "ack");
        let id = frontend.velo.instance_id();
        let pin = || {
            worker
                .peer_state(id)
                .reverse_pin_until
                .load(Ordering::Acquire)
        };
        let first_expiry = pin();
        assert_eq!(worker.peer_pin_calls.load(Ordering::Relaxed), 1);

        tokio::time::pause();
        tokio::time::advance(Duration::from_secs(301)).await;
        tokio::time::resume();

        assert_eq!(send().await.unwrap(), "ack");
        assert_eq!(worker.peer_pin_calls.load(Ordering::Relaxed), 2);
        assert!(pin() >= first_expiry + 301);
        assert_eq!(send().await.unwrap(), "ack");
        assert_eq!(worker.peer_pin_calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn non_velo_address_is_internal() {
        let frontend = service().await;
        let client = VeloRequestClient::new(frontend.clone(), Duration::from_secs(2));
        for input in ["127.0.0.1:9/ns/worker/generate", "worker.generate"] {
            assert_eq!(
                error_type(
                    client
                        .send_request(input.into(), Bytes::new(), Headers::new())
                        .await
                ),
                ErrorType::Internal
            );
        }
        assert_eq!(frontend.velo.response_slots_in_use(), 0);
    }

    #[tokio::test]
    async fn local_checks() {
        let frontend = service().await;
        let worker = service().await;
        let seen = Arc::new(AtomicUsize::new(0));
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, {
                    let seen = seen.clone();
                    move |_| {
                        seen.fetch_add(1, Ordering::Relaxed);
                        Ok(Some(Bytes::new()))
                    }
                })
                .build(),
            )
            .unwrap();
        let client = VeloRequestClient::new(frontend.clone(), Duration::from_secs(2));
        let local = address(&frontend.velo.peer_info());
        let remote = address(&worker.velo.peer_info());
        for target in [&local, &remote] {
            assert_eq!(
                error_type(
                    client
                        .send_request(
                            target.clone(),
                            Bytes::from(vec![0; get_tcp_max_message_size() + 1]),
                            Headers::new()
                        )
                        .await
                ),
                ErrorType::InvalidArgument
            );
            let bad = Headers::from([("x-request-id".into(), "x".repeat(1025))]);
            assert_eq!(
                error_type(client.send_request(target.clone(), Bytes::new(), bad).await),
                ErrorType::InvalidArgument
            );
        }
        for key in ["_rv", "_rv_budget_ms"] {
            let reserved = Headers::from([(key.into(), "10".into())]);
            assert_eq!(
                error_type(
                    client
                        .send_request(remote.clone(), Bytes::new(), reserved)
                        .await
                ),
                ErrorType::InvalidArgument
            );
        }
        assert_eq!(seen.load(Ordering::Relaxed), 0);
        assert_eq!(worker.peer_pin_calls.load(Ordering::Relaxed), 0);

        let good = Headers::from([("x-request-id".into(), "x".repeat(1024))]);
        assert!(
            client
                .send_request(local.clone(), Bytes::new(), good.clone())
                .await
                .unwrap()
                .starts_with(b"Server unavailable:")
        );
        assert_eq!(frontend.velo.response_slots_in_use(), 0);
        assert!(
            client
                .send_request(remote.clone(), Bytes::new(), good)
                .await
                .unwrap()
                .is_empty()
        );
        let parsed = decode_address(&remote).unwrap();
        let mut headers =
            Headers::from_iter((0..15).map(|i| (format!("header-{i}"), "x".repeat(1000))));
        headers.insert(ENDPOINT_PATH_HEADER.into(), parsed.path.into());
        let padding = 15_872 - rmp_serde::to_vec(&headers).unwrap().len();
        headers
            .get_mut("header-0")
            .unwrap()
            .push_str(&"x".repeat(24));
        headers.insert("padding".into(), "x".repeat(padding - 24 - 11));
        assert_eq!(rmp_serde::to_vec(&headers).unwrap().len(), 15_872);
        let payload = Bytes::from(vec![0; 300 * 1024]);
        assert!(
            client
                .send_request(remote.clone(), payload.clone(), headers.clone())
                .await
                .unwrap()
                .is_empty()
        );
        headers.get_mut("padding").unwrap().push('x');
        for target in [&local, &remote] {
            assert_eq!(
                error_type(
                    client
                        .send_request(target.clone(), payload.clone(), headers.clone())
                        .await
                ),
                ErrorType::InvalidArgument
            );
        }
        assert_eq!(seen.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn a_request_at_the_size_limit_reaches_the_worker() {
        let frontend = service().await;
        let worker = service().await;
        let seen = Arc::new(AtomicUsize::new(0));
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, {
                    let seen = seen.clone();
                    move |ctx| {
                        seen.store(ctx.payload.len(), Ordering::Relaxed);
                        Ok(Some(Bytes::new()))
                    }
                })
                .build(),
            )
            .unwrap();
        let client = VeloRequestClient::new(frontend.clone(), Duration::from_secs(10));
        let limit = get_tcp_max_message_size();
        let result = client
            .send_request(
                address(&worker.velo.peer_info()),
                Bytes::from(vec![0; limit]),
                Headers::new(),
            )
            .await;
        assert!(result.unwrap().is_empty());
        assert_eq!(seen.load(Ordering::Relaxed), limit);
    }

    #[tokio::test]
    async fn ack_slot_budget() {
        let frontend = service().await;
        let worker = service().await;
        let entered = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));
        worker
            .velo
            .register_handler(
                Handler::unary_handler_async(REQUEST_HANDLER, {
                    let entered = entered.clone();
                    let release = release.clone();
                    move |_| {
                        let entered = entered.clone();
                        let release = release.clone();
                        async move {
                            entered.add_permits(1);
                            release.acquire().await.unwrap().forget();
                            Ok(Some(Bytes::new()))
                        }
                    }
                })
                .build(),
            )
            .unwrap();

        let worker_info = worker.velo.peer_info();
        let client = Arc::new(VeloRequestClient::new(
            frontend.clone(),
            Duration::from_secs(4),
        ));
        let target = address(&worker_info);
        let worker_address = encode_worker_address(worker_info.worker_address()).unwrap();
        establish_route(
            &frontend,
            worker_info.instance_id(),
            &worker_address,
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap();

        let reserved = frontend
            .ack_slots
            .try_acquire_many((ACK_SLOTS - 4) as u32)
            .unwrap();
        let baseline = frontend.velo.response_slots_in_use();
        let mut calls = tokio::task::JoinSet::new();
        for _ in 0..4 {
            let (client, target) = (client.clone(), target.clone());
            calls.spawn(async move {
                client
                    .send_request(target, Bytes::new(), Headers::new())
                    .await
            });
        }
        tokio::time::timeout(Duration::from_secs(2), entered.acquire_many(4))
            .await
            .unwrap()
            .unwrap()
            .forget();
        assert_eq!(frontend.velo.response_slots_in_use(), baseline + 4);
        let fifth = tokio::time::timeout(
            Duration::from_millis(300),
            client.send_request(target, Bytes::new(), Headers::new()),
        )
        .await
        .unwrap();
        assert_eq!(error_type(fifth), ErrorType::ResourceExhausted);
        assert_eq!(frontend.velo.response_slots_in_use(), baseline + 4);
        assert_eq!(entered.available_permits(), 0);
        drop(reserved);
        release.add_permits(4);
        for result in calls.join_all().await {
            assert!(result.unwrap().is_empty());
        }
        assert_eq!(frontend.velo.response_slots_in_use(), baseline);
    }

    #[tokio::test]
    async fn cancelling_staged_sends_preserves_the_per_worker_buffer_limit() {
        let frontend = service().await;
        let worker = service().await;
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, |_| Ok(Some(Bytes::new()))).build(),
            )
            .unwrap();
        let info = worker.velo.peer_info();
        let target = address(&info);
        let timeout = Duration::from_secs(3);
        establish_route(
            &frontend,
            info.instance_id(),
            &encode_worker_address(info.worker_address()).unwrap(),
            Instant::now() + timeout,
        )
        .await
        .unwrap();
        let client = VeloRequestClient::new(frontend, timeout);
        let mut payloads = Vec::new();
        // Keep the current-thread executor in this poll until all callers have cancelled.
        for _ in 0..5 {
            let bytes: Arc<[u8]> = vec![0; 300 * 1024].into();
            payloads.push(Arc::downgrade(&bytes));
            let mut send =
                client.send_request(target.clone(), Bytes::from_owner(bytes), Headers::new());
            assert!(futures::poll!(&mut send).is_pending());
            drop(send);
        }
        let live = payloads
            .iter()
            .filter(|bytes| bytes.strong_count() > 0)
            .count();
        assert!(
            live <= super::super::STAGED_WINDOW,
            "{live} distinct staged payloads still live after cancellation"
        );
    }

    #[tokio::test]
    async fn staged_window_queues_requests_beyond_the_window() {
        let frontend = service().await;
        let worker = service().await;
        let entered = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));
        worker
            .velo
            .register_handler(
                Handler::unary_handler_async(REQUEST_HANDLER, {
                    let entered = entered.clone();
                    let release = release.clone();
                    move |_| {
                        let entered = entered.clone();
                        let release = release.clone();
                        async move {
                            entered.add_permits(1);
                            release.acquire().await.unwrap().forget();
                            Ok(Some(Bytes::from_static(b"ack")))
                        }
                    }
                })
                .build(),
            )
            .unwrap();

        let worker_info = worker.velo.peer_info();
        let client = Arc::new(VeloRequestClient::new(
            frontend.clone(),
            Duration::from_secs(4),
        ));
        let target = address(&worker_info);
        let worker_address = encode_worker_address(worker_info.worker_address()).unwrap();
        let (peer, _) = establish_route(
            &frontend,
            worker_info.instance_id(),
            &worker_address,
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap();
        // Leave a window of two.
        let _reserved = peer.staged.try_acquire_many(2).unwrap();
        let payload = Bytes::from(vec![0; 1024 * 1024]);
        let mut calls = tokio::task::JoinSet::new();
        for _ in 0..4 {
            let (client, target, payload) = (client.clone(), target.clone(), payload.clone());
            calls.spawn(async move { client.send_request(target, payload, Headers::new()).await });
        }
        let wait_for_two = || async {
            tokio::time::timeout(Duration::from_secs(2), entered.acquire_many(2))
                .await
                .unwrap()
                .unwrap()
                .forget();
        };
        wait_for_two().await;
        // The other two wait at the caller: none fails and none reaches the worker.
        assert!(
            tokio::time::timeout(Duration::from_millis(300), calls.join_next())
                .await
                .is_err()
        );
        assert_eq!(entered.available_permits(), 0);
        // A request below the staging threshold does not use the window.
        let small = tokio::spawn({
            let (client, target) = (client.clone(), target.clone());
            async move {
                client
                    .send_request(target, Bytes::new(), Headers::new())
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(2), entered.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();

        release.add_permits(3);
        assert_eq!(small.await.unwrap().unwrap(), "ack");
        wait_for_two().await;
        release.add_permits(2);
        for result in calls.join_all().await {
            assert_eq!(result.unwrap(), "ack");
        }
    }

    #[tokio::test]
    async fn staged_waiter_refuses_a_retired_request_route() {
        use crate::{engine::AsyncEngineContextProvider, pipeline::Context};

        let frontend = service().await;
        let worker = service().await;
        let received = Arc::new(AtomicUsize::new(0));
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, {
                    let received = received.clone();
                    move |_| {
                        received.fetch_add(1, Ordering::Relaxed);
                        Ok(Some(Bytes::new()))
                    }
                })
                .build(),
            )
            .unwrap();

        let info = worker.velo.peer_info();
        let id = info.instance_id();
        let target = address(&info);
        let timeout = Duration::from_secs(3);
        let (peer, _) = establish_route(
            &frontend,
            id,
            &encode_worker_address(info.worker_address()).unwrap(),
            Instant::now() + timeout,
        )
        .await
        .unwrap();
        let response = frontend
            .register_response(Context::new(()).context(), false)
            .unwrap();
        frontend.hold_request_peer(response.registration_id().unwrap(), &target);
        let reserved = peer.staged.try_acquire_many(4).unwrap();
        let client = VeloRequestClient::new(frontend.clone(), timeout);
        let mut send =
            client.send_request(target, Bytes::from(vec![0; 300 * 1024]), Headers::new());
        assert!(futures::poll!(&mut send).is_pending());

        frontend.unregister_request_peer(id);
        assert!(frontend.confirmed_peer(id).is_none());
        drop(reserved);
        let result = send.await;
        assert_eq!(
            received.load(Ordering::Relaxed),
            0,
            "queued request reached the retired worker, caller received {result:?}"
        );
        assert_eq!(error_type(result), ErrorType::WorkerUnavailable);
    }

    #[tokio::test]
    async fn staged_window_wait_ends_at_the_request_deadline() {
        let frontend = service().await;
        let worker = service().await;
        let seen = Arc::new(AtomicUsize::new(0));
        worker
            .velo
            .register_handler(
                Handler::unary_handler(REQUEST_HANDLER, {
                    let seen = seen.clone();
                    move |_| {
                        seen.fetch_add(1, Ordering::Relaxed);
                        Ok(Some(Bytes::new()))
                    }
                })
                .build(),
            )
            .unwrap();
        let worker_info = worker.velo.peer_info();
        let timeout = Duration::from_millis(500);
        let client = VeloRequestClient::new(frontend.clone(), timeout);
        let target = address(&worker_info);
        let worker_address = encode_worker_address(worker_info.worker_address()).unwrap();
        let (peer, epoch) = establish_route(
            &frontend,
            worker_info.instance_id(),
            &worker_address,
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap();
        let reserved = peer.staged.try_acquire_many(4).unwrap();

        let start = Instant::now();
        let result = bounded(client.send_request(
            target.clone(),
            Bytes::from(vec![0; 300 * 1024]),
            Headers::new(),
        ))
        .await;
        assert!(start.elapsed() >= timeout);
        // The caller ran out of time. The worker saw nothing and keeps its route.
        assert_eq!(error_type(result), ErrorType::ResourceExhausted);
        assert_eq!(seen.load(Ordering::Relaxed), 0);
        assert_eq!(
            frontend
                .confirmed_peer(worker_info.instance_id())
                .unwrap()
                .1,
            epoch
        );
        assert_eq!(frontend.velo.response_slots_in_use(), 0);

        drop(reserved);
        assert!(
            bounded(client.send_request(target, Bytes::from(vec![0; 300 * 1024]), Headers::new()))
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(seen.load(Ordering::Relaxed), 1);
    }

    #[rstest::rstest]
    #[case::too_late_to_send(Duration::from_millis(1), false)]
    #[case::ack_times_out(Duration::from_millis(100), true)]
    #[tokio::test]
    async fn staged_window_delay_leaves_the_worker_eligible(
        #[case] remaining: Duration,
        #[case] expect_send: bool,
    ) {
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
        let worker = service().await;
        worker
            .velo
            .register_handler(
                Handler::unary_handler_async(REQUEST_HANDLER, |ctx| async move {
                    if ctx.payload.is_empty() {
                        Ok(Some(Bytes::new()))
                    } else {
                        std::future::pending().await
                    }
                })
                .build(),
            )
            .unwrap();
        let endpoint = drt
            .namespace(format!("staged_window_{}", remaining.as_millis()))
            .unwrap()
            .component("worker")
            .unwrap()
            .endpoint("generate");
        let id = endpoint.id();
        let discovery_id = drt.discovery().instance_id();
        let worker_info = worker.velo.peer_info();
        let target = encode_address(&worker_info, &id, discovery_id).unwrap();
        drt.discovery()
            .register(DiscoverySpec::Endpoint {
                namespace: id.namespace,
                component: id.component,
                endpoint: id.name,
                transport: TransportType::Velo(target.clone()),
                device_type: None,
                request_plane_codec: None,
            })
            .await
            .unwrap();
        let client = endpoint
            .client_with_inhibited_duration(Duration::from_secs(2))
            .await
            .unwrap();
        bounded(client.wait_for_instances()).await.unwrap();
        let router = PushRouter::<String, Annotated<String>>::from_client(
            client.clone(),
            RouterMode::Direct,
        )
        .await
        .unwrap();
        let probe = drt.network_manager().create_client().unwrap();
        let send_probe = || probe.send_request(target.clone(), Bytes::new(), Headers::new());
        assert!(bounded(send_probe()).await.unwrap().is_empty());
        let peer_id = worker_info.instance_id();
        let (peer, epoch) = frontend.confirmed_peer(peer_id).unwrap();
        let reserved = peer.staged.try_acquire_many(STAGED_WINDOW as u32).unwrap();

        tokio::time::pause();
        let mut request =
            std::pin::pin!(router.direct(SingleIn::new("x".repeat(300 * 1024)), discovery_id,));
        assert!(futures::poll!(&mut request).is_pending());
        assert_eq!(frontend.ack_slots.available_permits(), ACK_SLOTS);
        let timeout = TcpRequestConfig::from_env().request_timeout;
        tokio::time::advance(timeout - remaining).await;
        drop(reserved);

        let (sent, result) = match futures::poll!(&mut request) {
            Poll::Ready(result) => (false, result),
            Poll::Pending => {
                assert_eq!(frontend.ack_slots.available_permits(), ACK_SLOTS - 1);
                (true, bounded(request).await)
            }
        };
        let err = result.expect_err("request must fail at the caller");
        assert_eq!(
            err.downcast_ref::<DynamoError>().unwrap().error_type(),
            ErrorType::ResourceExhausted,
            "{err:#}"
        );
        assert_eq!(sent, expect_send);
        assert!(client.instance_ids_avail().contains(&discovery_id));
        assert_eq!(frontend.confirmed_peer(peer_id).unwrap().1, epoch);
        assert_eq!(frontend.ack_slots.available_permits(), ACK_SLOTS);

        tokio::time::resume();
        assert!(bounded(send_probe()).await.unwrap().is_empty());
        owner.shutdown();
    }

    #[tokio::test]
    async fn timeout_after_route_resets_epoch() {
        let frontend = service().await;
        let worker = service().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(Semaphore::new(0));
        worker
            .velo
            .register_handler(
                Handler::unary_handler_async(REQUEST_HANDLER, {
                    let call_count = call_count.clone();
                    let release = release.clone();
                    move |_| {
                        let call = call_count.fetch_add(1, Ordering::Relaxed);
                        let release = release.clone();
                        async move {
                            if call == 1 {
                                release.acquire().await.unwrap().forget();
                            }
                            Ok(Some(Bytes::new()))
                        }
                    }
                })
                .build(),
            )
            .unwrap();
        let client = VeloRequestClient::new(frontend.clone(), Duration::from_millis(500));
        let target = address(&worker.velo.peer_info());
        let send = || client.send_request(target.clone(), Bytes::new(), Headers::new());
        assert!(send().await.unwrap().is_empty());
        let id = worker.velo.instance_id();
        let before = frontend.confirmed_peer(id).unwrap().1;
        let staged = client.send_request(
            target.clone(),
            Bytes::from(vec![0; 300 * 1024]),
            Headers::new(),
        );
        assert_eq!(
            error_type(bounded(staged).await),
            ErrorType::ConnectionTimeout
        );
        assert_eq!(call_count.load(Ordering::Relaxed), 2);
        release.add_permits(1);
        assert!(frontend.confirmed_peer(id).is_none());
        assert!(send().await.unwrap().is_empty());
        assert_ne!(frontend.confirmed_peer(id).unwrap().1, before);
        assert_eq!(worker.peer_pin_calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn closed_port_is_cannot_connect() {
        let velo = service().await;
        let peer = velo.velo.peer_info();
        let entry = peer.worker_address().get_entry("tcp").unwrap().unwrap();
        let endpoints = velo::transports::utils::interfaces::parse_endpoints(&entry).unwrap();
        let target = address(&peer);
        velo.velo
            .shutdown(velo::ShutdownPolicy::Timeout(Duration::from_secs(2)))
            .await;
        let socket = tokio::net::TcpSocket::new_v4().unwrap();
        socket.bind(endpoints[0].socket_addr().unwrap()).unwrap();
        let frontend = service().await;
        let client = VeloRequestClient::new(frontend, Duration::from_secs(2));
        assert_eq!(
            error_type(
                client
                    .send_request(target, Bytes::new(), Headers::new())
                    .await
            ),
            ErrorType::CannotConnect
        );
    }
}
