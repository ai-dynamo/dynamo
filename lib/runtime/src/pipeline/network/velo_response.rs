// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo owns response framing, batching, flow control, and stream lifecycle.

use anyhow::{Result, bail, ensure};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    pin::Pin,
    sync::{Arc, Weak},
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use velo::{
    PeerInfo, Velo,
    streaming::{
        MuxConfig, StreamAnchor, StreamAnchorHandle, StreamController, StreamFrame, StreamSender,
        control::StreamOpenTicket,
    },
};

use super::{
    ConnectionInfo, RegisteredStream, ResponseStreamPrologue, StreamPrologueError, StreamReceiver,
};
use crate::{
    discovery::EndpointInstanceId,
    engine::AsyncEngineContext,
    error::{DynamoError, ErrorType},
};

pub const TRANSPORT_NAME: &str = "velo_response";
const VERSION: u32 = 2;
const TOMBSTONE_TTL: Duration = Duration::from_secs(5);
static PROCESS_SERVICE: tokio::sync::Mutex<Weak<VeloResponseService>> =
    tokio::sync::Mutex::const_new(Weak::new());

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResponseTransport {
    Tcp,
    Ucx,
}

impl ResponseTransport {
    fn configured() -> Result<Self> {
        match std::env::var(
            crate::config::environment_names::response_plane::DYN_VELO_RESPONSE_TRANSPORT,
        )
        .as_deref()
        {
            Err(std::env::VarError::NotPresent) | Ok("tcp") => Ok(Self::Tcp),
            Ok("ucx") => Ok(Self::Ucx),
            _ => bail!("DYN_VELO_RESPONSE_TRANSPORT must be tcp or ucx"),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ResponseAddress {
    version: u32,
    transport: ResponseTransport,
    peer: PeerInfo,
    anchor: StreamAnchorHandle,
    ticket: StreamOpenTicket,
}

#[derive(Serialize, Deserialize)]
enum ResponseFrame {
    Prologue(ResponseStreamPrologue),
    Data(Bytes),
}

struct Registration {
    controller: StreamController,
    done: CancellationToken,
    instance: Option<EndpointInstanceId>,
}

#[derive(Default)]
struct Registrations {
    requests: HashMap<Uuid, Registration>,
    tombstones: HashMap<EndpointInstanceId, Instant>,
}

pub struct VeloResponseService {
    velo: Arc<Velo>,
    transport: ResponseTransport,
    registrations: Mutex<Registrations>,
}

impl VeloResponseService {
    /// A weak process cache lets the final runtime owner release the service.
    pub async fn shared() -> Result<Arc<Self>> {
        let mut shared = PROCESS_SERVICE.lock().await;
        if let Some(service) = shared.upgrade() {
            return Ok(service);
        }
        let transport = ResponseTransport::configured()?;
        let host = crate::utils::ip_resolver::host_override_from_env(
            crate::config::environment_names::tcp_response_stream::DYN_TCP_RESPONSE_STREAM_HOST,
        )?;
        let resolver = crate::utils::ip_resolver::DefaultIpResolver;
        let host = match host {
            Some(host) => crate::utils::ip_resolver::resolve_host_or_interface(&host, &resolver)?,
            None => crate::utils::ip_resolver::resolve_local_host(&resolver)?,
        };
        let service = Self::new(transport, SocketAddr::new(host.advertise_ip(), 0)).await?;
        *shared = Arc::downgrade(&service);
        Ok(service)
    }

    async fn new(transport: ResponseTransport, address: SocketAddr) -> Result<Arc<Self>> {
        let mut builder = Velo::builder()
            .stream_bind_addr(address.ip())
            .messenger_mux(MuxConfig {
                enabled: true,
                ..Default::default()
            })?;
        match transport {
            ResponseTransport::Tcp => {
                builder = builder.add_transport(Arc::new(
                    velo::transports::tcp::TcpTransportBuilder::new()
                        .bind_addr(address)
                        .build()?,
                ));
            }
            ResponseTransport::Ucx => {
                #[cfg(all(target_os = "linux", feature = "velo-ucx"))]
                {
                    builder = builder.add_transport(Arc::new(
                        velo::transports::ucx::UcxTransportBuilder::new().build()?,
                    ));
                }
                #[cfg(not(all(target_os = "linux", feature = "velo-ucx")))]
                bail!("Velo UCX responses require a Linux build with the velo-ucx feature");
            }
        }
        Ok(Arc::new(Self {
            velo: builder.build().await?,
            transport,
            registrations: Mutex::new(Registrations::default()),
        }))
    }

    pub fn register_response(
        self: &Arc<Self>,
        context: Arc<dyn AsyncEngineContext>,
    ) -> Result<RegisteredStream<StreamReceiver>> {
        let anchor = self.velo.create_anchor::<ResponseFrame>();
        let ticket = self
            .velo
            .prebind_anchor(anchor.handle())
            .ok_or_else(|| anyhow::anyhow!("Velo mux prebind unavailable"))?;
        let controller = anchor.controller();
        let info = ConnectionInfo {
            transport: TRANSPORT_NAME.into(),
            info: serde_json::to_string(&ResponseAddress {
                version: VERSION,
                transport: self.transport,
                peer: self.velo.peer_info(),
                anchor: anchor.handle(),
                ticket,
            })?,
        };
        let id = Uuid::new_v4();
        let done = CancellationToken::new();
        self.registrations.lock().requests.insert(
            id,
            Registration {
                controller: controller.clone(),
                done: done.clone(),
                instance: None,
            },
        );
        let lease = ResponseLease {
            service: self.clone(),
            id,
        };
        let (mut tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let mut anchor = anchor;
            let mut stopped = false;
            // This task handles setup and lifecycle only. The caller polls all
            // response data directly from the anchor after the prologue.
            let first = loop {
                tokio::select! {
                    biased;
                    _ = done.cancelled() => return,
                    _ = context.killed() => { controller.cancel(); return; }
                    _ = tx.closed() => return,
                    _ = context.stopped(), if !stopped => { controller.request_stop(); stopped = true; }
                    first = anchor.next() => break first,
                }
            };
            match first {
                Some(Ok(StreamFrame::Item(ResponseFrame::Prologue(prologue)))) => {
                    if let Some(error) = prologue.error {
                        let _ = tx.send(Err(StreamPrologueError {
                            message: error,
                            typed_error: prologue.typed_error,
                        }));
                        return;
                    }
                }
                other => {
                    let _ = tx.send(Err(StreamPrologueError::from_message(format!(
                        "Velo response ended before a valid prologue: {}",
                        first_kind(&other)
                    ))));
                    return;
                }
            }
            let receiver = VeloResponseReceiver {
                anchor,
                _lease: lease,
                ended: false,
            };
            if tx
                .send(Ok(StreamReceiver {
                    rx: super::ByteReceiver::Velo(Box::new(receiver)),
                }))
                .is_err()
            {
                return;
            }
            loop {
                tokio::select! {
                    biased;
                    _ = done.cancelled() => return,
                    _ = context.killed() => { controller.cancel(); return; }
                    _ = context.stopped(), if !stopped => { controller.request_stop(); stopped = true; }
                }
            }
        });
        let service = self.clone();
        Ok(RegisteredStream::new(info, rx)
            .with_registration_id(id)
            .with_cleanup(move || service.remove(id)))
    }

    fn remove(&self, id: Uuid) {
        let registration = self.registrations.lock().requests.remove(&id);
        if let Some(registration) = registration {
            registration.done.cancel();
            registration.controller.cancel();
        }
    }

    pub async fn associate_instance(&self, id: Uuid, instance: &EndpointInstanceId) -> bool {
        let mut state = self.registrations.lock();
        state
            .tombstones
            .retain(|_, when| when.elapsed() < TOMBSTONE_TTL);
        if state.tombstones.contains_key(instance) {
            return false;
        }
        if let Some(request) = state.requests.get_mut(&id) {
            request.instance = Some(instance.clone());
            true
        } else {
            false
        }
    }

    pub async fn cancel_response(&self, id: Uuid) {
        self.remove(id);
    }

    pub async fn cancel_instance_streams(&self, instance: &EndpointInstanceId) -> usize {
        let ids = {
            let mut state = self.registrations.lock();
            state
                .tombstones
                .retain(|_, when| when.elapsed() < TOMBSTONE_TTL);
            state.tombstones.insert(instance.clone(), Instant::now());
            state
                .requests
                .iter()
                .filter_map(|(id, r)| (r.instance.as_ref() == Some(instance)).then_some(*id))
                .collect::<Vec<_>>()
        };
        for id in &ids {
            self.remove(*id);
        }
        ids.len()
    }

    pub async fn clear_instance_tombstone(&self, instance: &EndpointInstanceId) {
        self.registrations.lock().tombstones.remove(instance);
    }

    pub async fn sender(
        self: &Arc<Self>,
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
        cancellation: Option<prometheus::IntCounter>,
    ) -> Result<VeloResponseSender> {
        ensure!(
            info.transport == TRANSPORT_NAME,
            "invalid Velo response transport"
        );
        let address: ResponseAddress = serde_json::from_str(&info.info)?;
        ensure!(
            address.version == VERSION
                && address.ticket.streaming_transport_key.as_str()
                    == velo::streaming::MESSENGER_MUX_KEY,
            "unsupported Velo response lifecycle version"
        );
        ensure!(
            address.transport == self.transport,
            "Velo response transport mismatch"
        );
        ensure!(
            address.anchor.unpack().0 == address.peer.worker_id(),
            "Velo anchor and peer identity differ"
        );
        let peer_id = address.peer.instance_id();
        self.velo.register_peer(address.peer)?;
        if peer_id != self.velo.instance_id() {
            // The cached Velo hello exchange also installs the reverse UCX
            // address. Stream slot opens do not perform that peer handshake.
            tokio::time::timeout(
                Duration::from_secs(10),
                self.velo.wait_for_handler(peer_id, "_stream_stop"),
            )
            .await??;
        }
        let sender = if address.anchor.unpack().0 == self.velo.instance_id().worker_id() {
            self.velo
                .attach_anchor::<ResponseFrame>(address.anchor)
                .await?
        } else {
            self.velo
                .open_anchor_stream::<ResponseFrame>(address.anchor, address.ticket)
                .await?
        };
        let stop = sender.stop_token();
        let cancel = sender.cancellation_token();
        let finished = CancellationToken::new();
        let monitor_done = finished.clone();
        tokio::spawn(async move {
            let mut stopped = false;
            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        if let Some(counter) = &cancellation { counter.inc(); }
                        context.kill(); return;
                    }
                    _ = monitor_done.cancelled() => return,
                    _ = stop.cancelled(), if !stopped => { context.stop_generating(); stopped = true; }
                }
            }
        });
        Ok(VeloResponseSender {
            sender: Some(sender),
            _service: self.clone(),
            finished,
            prologue_sent: false,
        })
    }
}

impl Drop for VeloResponseService {
    fn drop(&mut self) {
        let velo = self.velo.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                velo.graceful_shutdown(velo::ShutdownPolicy::Timeout(Duration::from_secs(5)))
                    .await;
            });
        }
    }
}

struct ResponseLease {
    service: Arc<VeloResponseService>,
    id: Uuid,
}
impl Drop for ResponseLease {
    fn drop(&mut self) {
        self.service.remove(self.id);
    }
}

fn first_kind(
    frame: &Option<Result<StreamFrame<ResponseFrame>, velo::streaming::StreamError>>,
) -> String {
    match frame {
        None => "closed".into(),
        Some(Err(error)) => error.to_string(),
        Some(Ok(_)) => "unexpected frame".into(),
    }
}

pub(super) struct VeloResponseReceiver {
    anchor: StreamAnchor<ResponseFrame>,
    _lease: ResponseLease,
    ended: bool,
}
impl Stream for VeloResponseReceiver {
    type Item = Result<Bytes, DynamoError>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        match std::task::ready!(Pin::new(&mut this.anchor).poll_next(cx)) {
            Some(Ok(StreamFrame::Item(ResponseFrame::Data(bytes)))) => Poll::Ready(Some(Ok(bytes))),
            Some(Ok(StreamFrame::Finalized)) => {
                this.ended = true;
                Poll::Ready(None)
            }
            terminal => {
                this.ended = true;
                Poll::Ready(Some(Err(DynamoError::builder()
                    .error_type(ErrorType::Disconnected)
                    .message(format!(
                        "Velo response ended without finalization: {}",
                        first_kind(&terminal)
                    ))
                    .build())))
            }
        }
    }
}

pub struct VeloResponseSender {
    sender: Option<StreamSender<ResponseFrame>>,
    _service: Arc<VeloResponseService>,
    finished: CancellationToken,
    prologue_sent: bool,
}

impl VeloResponseSender {
    pub async fn send(&self, bytes: Bytes) -> Result<()> {
        ensure!(self.prologue_sent, "Velo response prologue missing");
        self.sender
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Velo response closed"))?
            .send(ResponseFrame::Data(bytes))
            .await?;
        Ok(())
    }
    pub async fn send_prologue(&mut self, error: Option<StreamPrologueError>) -> Result<()> {
        ensure!(!self.prologue_sent, "Velo response prologue already sent");
        let (error, typed_error) = error.map_or((None, None), |e| (Some(e.message), e.typed_error));
        self.sender
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Velo response closed"))?
            .send(ResponseFrame::Prologue(ResponseStreamPrologue {
                error,
                typed_error,
            }))
            .await?;
        self.prologue_sent = true;
        Ok(())
    }
    pub async fn finish(&mut self) -> Result<()> {
        self.finished.cancel();
        if let Some(sender) = self.sender.take() {
            sender.finalize()?;
        }
        Ok(())
    }
    pub async fn abort(&mut self) -> Result<()> {
        self.finished.cancel();
        self.sender.take();
        Ok(())
    }
}

impl Drop for VeloResponseSender {
    fn drop(&mut self) {
        self.finished.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::AsyncEngineContextProvider;
    use crate::pipeline::Context as EngineContext;

    async fn pair(
        transport: ResponseTransport,
    ) -> (Arc<VeloResponseService>, Arc<VeloResponseService>) {
        let address = "127.0.0.1:0".parse().unwrap();
        (
            VeloResponseService::new(transport, address).await.unwrap(),
            VeloResponseService::new(transport, address).await.unwrap(),
        )
    }

    async fn lifecycle_suite(transport: ResponseTransport) {
        stop_drains_output_and_cancel_reaches_an_idle_engine(transport).await;
        completion_and_failure_keep_distinct_results(transport).await;
        prologue_preserves_typed_errors_and_rejects_version_mismatch(transport).await;
        blocked_stream_isolation_and_peer_failure(transport).await;
    }

    #[tokio::test]
    async fn tcp_lifecycle() {
        lifecycle_suite(ResponseTransport::Tcp).await;
    }

    #[cfg(feature = "velo-ucx")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ucx_lifecycle() {
        lifecycle_suite(ResponseTransport::Ucx).await;
    }

    async fn stop_drains_output_and_cancel_reaches_an_idle_engine(transport: ResponseTransport) {
        let (consumer, producer) = pair(transport).await;
        let client = EngineContext::new(()).context();
        let engine = EngineContext::new(()).context();
        let registered = consumer.register_response(client.clone()).unwrap();
        client.stop_generating();
        let mut sender = producer
            .sender(engine.clone(), registered.connection_info.clone(), None)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), engine.stopped())
            .await
            .unwrap();
        assert!(!engine.is_killed());
        sender.send_prologue(None).await.unwrap();
        let (_, provider) = registered.into_parts();
        let mut receiver = provider.await.unwrap().unwrap();
        let payload = Bytes::from(vec![7; 128 * 1024]);
        sender.send(payload.clone()).await.unwrap();
        assert_eq!(receiver.rx.next().await.unwrap().unwrap(), payload);
        drop(receiver);
        tokio::time::timeout(Duration::from_secs(5), engine.killed())
            .await
            .unwrap();
    }

    async fn completion_and_failure_keep_distinct_results(transport: ResponseTransport) {
        let (consumer, producer) = pair(transport).await;
        for complete in [true, false] {
            let registered = consumer
                .register_response(EngineContext::new(()).context())
                .unwrap();
            let mut sender = producer
                .sender(
                    EngineContext::new(()).context(),
                    registered.connection_info.clone(),
                    None,
                )
                .await
                .unwrap();
            sender.send_prologue(None).await.unwrap();
            let (_, provider) = registered.into_parts();
            let mut receiver = provider.await.unwrap().unwrap();
            sender.send(Bytes::from_static(b"last")).await.unwrap();
            if complete {
                sender.finish().await.unwrap();
            } else {
                sender.abort().await.unwrap();
            }
            assert_eq!(
                receiver.rx.next().await.unwrap().unwrap(),
                Bytes::from_static(b"last")
            );
            let terminal = receiver.rx.next().await;
            if complete {
                assert!(terminal.is_none());
            } else {
                assert!(terminal.unwrap().is_err());
            }
            assert!(receiver.rx.next().await.is_none());
        }
    }

    async fn blocked_stream_isolation_and_peer_failure(transport: ResponseTransport) {
        let (consumer, producer) = pair(transport).await;
        let mut streams = Vec::new();
        for _ in 0..2 {
            let registered = consumer
                .register_response(EngineContext::new(()).context())
                .unwrap();
            let engine = EngineContext::new(()).context();
            let mut sender = producer
                .sender(engine.clone(), registered.connection_info.clone(), None)
                .await
                .unwrap();
            sender.send_prologue(None).await.unwrap();
            let (_, provider) = registered.into_parts();
            streams.push((sender, provider.await.unwrap().unwrap(), engine));
        }
        let (blocked_sender, blocked_receiver, blocked_engine) = streams.remove(0);
        let (mut sender, mut receiver, engine) = streams.remove(0);
        let flood = tokio::spawn(async move {
            loop {
                blocked_sender.send(Bytes::from(vec![1; 4096])).await?;
            }
            #[allow(unreachable_code)]
            Ok::<(), anyhow::Error>(())
        });
        sender
            .send(Bytes::from_static(b"independent"))
            .await
            .unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), receiver.rx.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
            Bytes::from_static(b"independent")
        );
        drop(blocked_receiver);
        tokio::time::timeout(Duration::from_secs(5), blocked_engine.killed())
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(5), flood)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
        // Releasing one worker's service owner cannot shut down another stream.
        drop(producer);
        sender
            .send(Bytes::from_static(b"still alive"))
            .await
            .unwrap();
        assert!(receiver.rx.next().await.unwrap().is_ok());
        consumer
            .velo
            .graceful_shutdown(velo::ShutdownPolicy::Timeout(Duration::from_secs(1)))
            .await;
        tokio::time::timeout(Duration::from_secs(30), engine.killed())
            .await
            .unwrap();
        assert!(sender.send(Bytes::new()).await.is_err());
        sender.abort().await.unwrap();
    }

    async fn prologue_preserves_typed_errors_and_rejects_version_mismatch(
        transport: ResponseTransport,
    ) {
        let (consumer, producer) = pair(transport).await;
        let registered = consumer
            .register_response(EngineContext::new(()).context())
            .unwrap();
        let error = DynamoError::builder()
            .error_type(ErrorType::Disconnected)
            .message("backend failure")
            .build();
        let mut sender = producer
            .sender(
                EngineContext::new(()).context(),
                registered.connection_info.clone(),
                None,
            )
            .await
            .unwrap();
        sender
            .send_prologue(Some(StreamPrologueError::new(
                "backend failure",
                error.clone(),
            )))
            .await
            .unwrap();
        let (_, provider) = registered.into_parts();
        match provider.await.unwrap() {
            Err(actual) => {
                let actual = actual.typed_error.expect("typed prologue error");
                assert_eq!(actual.class(), error.class());
                assert_eq!(actual.reason(), error.reason());
                assert_eq!(actual.to_string(), error.to_string());
            }
            Ok(_) => panic!("error prologue succeeded"),
        }
        let registered = consumer
            .register_response(EngineContext::new(()).context())
            .unwrap();
        let mut address: ResponseAddress =
            serde_json::from_str(&registered.connection_info.info).unwrap();
        address.version -= 1;
        let info = ConnectionInfo {
            transport: TRANSPORT_NAME.into(),
            info: serde_json::to_string(&address).unwrap(),
        };
        assert!(
            producer
                .sender(EngineContext::new(()).context(), info, None)
                .await
                .is_err()
        );
    }
}
