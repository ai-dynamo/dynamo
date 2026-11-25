// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use tokio::time::Instant;

use crate::{
    error::ErrorType,
    pipeline::network::{
        VELO_SCHEME,
        egress::unified_client::{Headers, RequestPlaneClient},
        get_tcp_max_message_size,
        velo_response::VeloResponseService,
    },
};

use super::{
    REQUEST_HANDLER, decode_address,
    discovery::DiscoveryReady,
    peers::{by_deadline, error, establish_route, transport_error},
    server::ENDPOINT_PATH_HEADER,
};

const MAX_HEADER_VALUE: usize = 1024;
// Velo reserves part of its 16 KiB header map for rendezvous metadata.
const HEADER_BUDGET: usize = 16 * 1024 - 512;

pub(in crate::pipeline::network) struct VeloRequestClient {
    service: Arc<VeloResponseService>,
    ack_timeout: Duration,
    discovery_ready: Option<Arc<DiscoveryReady>>,
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
        }
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
        let (peer, epoch) = establish_route(
            &self.service,
            target.instance_id,
            target.worker_address,
            deadline,
        )
        .await?;
        let _staged = (payload.len() > self.service.velo.staging_threshold())
            .then(|| {
                peer.staged.try_acquire().map_err(|_| {
                    error(ErrorType::ResourceExhausted, "staged window to worker full")
                })
            })
            .transpose()?;
        let _slot = self.service.ack_slots.try_acquire().map_err(|_| {
            error(
                ErrorType::ResourceExhausted,
                "Velo request ACK slots are full",
            )
        })?;
        let ack = by_deadline(deadline, async {
            self.service
                .velo
                .unary(REQUEST_HANDLER)?
                .raw_payload(payload)
                .headers(headers)
                .timeout(deadline.saturating_duration_since(Instant::now()))
                .send_to(target.instance_id)
                .await
        })
        .await
        .and_then(|ack| ack.map_err(transport_error));
        if ack.is_err() {
            peer.route.invalidate(epoch);
        }
        Ok(ack?)
    }

    fn transport_name(&self) -> &'static str {
        "velo"
    }

    fn is_healthy(&self) -> bool {
        !self.service.is_closing()
    }
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
        error::{DynamoError, ErrorType},
        pipeline::network::{
            egress::unified_client::{Headers, RequestPlaneClient},
            get_tcp_max_message_size,
            velo_request::{
                REQUEST_HANDLER, decode_address, peers::establish_route,
                server::ENDPOINT_PATH_HEADER,
            },
            velo_response::VeloResponseService,
        },
    };
    use crate::{
        pipeline::network::velo_request::{ACK_SLOTS, encode_address, encode_worker_address},
        pipeline::network::velo_response::{ResponseTransport, VeloServiceOptions},
        protocols::EndpointId,
    };
    use anyhow::Result;
    use bytes::Bytes;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::{sync::Arc, time::Duration};
    use tokio::sync::Semaphore;
    use tokio::time::Instant;
    use velo::{Handler, PeerInfo};

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
    async fn staged_window_refuses_before_sending() {
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
                            Ok(Some(Bytes::from_static(
                                b"Server overloaded: worker at capacity",
                            )))
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
        let _reserved = peer.staged.try_acquire_many(2).unwrap();
        let payload = Bytes::from(vec![0; 1024 * 1024]);
        let mut calls = tokio::task::JoinSet::new();
        for _ in 0..10 {
            let (client, target, payload) = (client.clone(), target.clone(), payload.clone());
            calls.spawn(async move { client.send_request(target, payload, Headers::new()).await });
        }
        tokio::time::timeout(Duration::from_secs(2), entered.acquire_many(2))
            .await
            .unwrap()
            .unwrap()
            .forget();
        for _ in 0..8 {
            let result = tokio::time::timeout(Duration::from_millis(300), calls.join_next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(error_type(result), ErrorType::ResourceExhausted);
        }
        assert_eq!(entered.available_permits(), 0);
        release.add_permits(2);
        for result in calls.join_all().await {
            assert!(result.unwrap().starts_with(b"Server overloaded:"));
        }
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
        assert_eq!(error_type(send().await), ErrorType::ConnectionTimeout);
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
