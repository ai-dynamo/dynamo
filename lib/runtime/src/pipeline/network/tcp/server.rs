// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow};
use derive_builder::Builder;
use futures::{SinkExt, StreamExt};
use local_ip_address::{Error as LocalIpError, list_afinet_netifas, local_ip, local_ipv6};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::{
    io::AsyncWriteExt,
    sync::{mpsc, oneshot},
    time::Instant,
};
use tokio_util::codec::{FramedRead, FramedWrite};

use super::{CallHomeHandshake, TcpStreamConnectionInfo};
use crate::{
    discovery::EndpointInstanceId,
    engine::AsyncEngineContext,
    pipeline::{
        PipelineError,
        network::{
            ControlMessage, RegisteredStream, StreamSender, StreamType, TwoPartCodec,
            codec::TwoPartMessage,
        },
    },
};

const TOMBSTONE_TTL: Duration = Duration::from_secs(5);

pub trait IpResolver {
    fn local_ip(&self) -> Result<IpAddr, LocalIpError>;
    fn local_ipv6(&self) -> Result<IpAddr, LocalIpError>;
}

pub struct DefaultIpResolver;

impl IpResolver for DefaultIpResolver {
    fn local_ip(&self) -> Result<IpAddr, LocalIpError> {
        local_ip()
    }

    fn local_ipv6(&self) -> Result<IpAddr, LocalIpError> {
        local_ipv6()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Builder, Default)]
pub struct ServerOptions {
    #[builder(default = "0")]
    pub port: u16,
    #[builder(default)]
    pub interface: Option<String>,
}

impl ServerOptions {
    pub fn builder() -> ServerOptionsBuilder {
        ServerOptionsBuilder::default()
    }
}

pub struct TcpStreamServer {
    local_address: SocketAddr,
    state: Arc<Mutex<State>>,
}

struct RequestedSendConnection {
    context: Arc<dyn AsyncEngineContext>,
    connection: oneshot::Sender<Result<StreamSender, String>>,
    send_buffer_count: usize,
}

#[derive(Default)]
struct State {
    pending: HashMap<String, RequestedSendConnection>,
    subject_instance: HashMap<String, EndpointInstanceId>,
    instance_subjects: HashMap<EndpointInstanceId, HashSet<String>>,
    removed_instances: HashMap<EndpointInstanceId, Instant>,
    listener_task: Option<tokio::task::JoinHandle<Result<()>>>,
}

fn prune_tombstones(state: &mut State, now: Instant) {
    state
        .removed_instances
        .retain(|_, inserted| now.saturating_duration_since(*inserted) < TOMBSTONE_TTL);
}

impl TcpStreamServer {
    pub fn options_builder() -> ServerOptionsBuilder {
        ServerOptionsBuilder::default()
    }

    pub async fn new(options: ServerOptions) -> Result<Arc<Self>, PipelineError> {
        Self::new_with_resolver(options, DefaultIpResolver).await
    }

    pub async fn new_with_resolver<R: IpResolver>(
        options: ServerOptions,
        resolver: R,
    ) -> Result<Arc<Self>, PipelineError> {
        let ip = match options.interface {
            Some(interface) => {
                let interfaces: HashMap<String, IpAddr> =
                    list_afinet_netifas()?.into_iter().collect();
                *interfaces.get(&interface).ok_or_else(|| {
                    PipelineError::Generic(format!("Interface not found: {interface}"))
                })?
            }
            None => match resolver.local_ip().or_else(|error| match error {
                LocalIpError::LocalIpAddressNotFound => resolver.local_ipv6(),
                _ => Err(error),
            }) {
                Ok(ip) => ip,
                Err(LocalIpError::LocalIpAddressNotFound) => {
                    tracing::warn!("No routable local IP found; using 127.0.0.1");
                    IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
                }
                Err(error) => {
                    return Err(PipelineError::Generic(format!(
                        "Failed to resolve local IP address: {error}"
                    )));
                }
            },
        };

        let state = Arc::new(Mutex::new(State::default()));
        let local_address = start_listener(SocketAddr::new(ip, options.port), state.clone())
            .await
            .map_err(|error| {
                PipelineError::Generic(format!("Failed to start TCP callback listener: {error}"))
            })?;
        Ok(Arc::new(Self {
            local_address,
            state,
        }))
    }

    pub fn local_address(&self) -> Result<SocketAddr> {
        Ok(self.local_address)
    }

    pub fn register_request(
        &self,
        context: Arc<dyn AsyncEngineContext>,
        send_buffer_count: usize,
    ) -> RegisteredStream<StreamSender> {
        let subject = uuid::Uuid::new_v4().to_string();
        let (pending_tx, pending_rx) = oneshot::channel();
        self.state.lock().pending.insert(
            subject.clone(),
            RequestedSendConnection {
                context: context.clone(),
                connection: pending_tx,
                send_buffer_count,
            },
        );
        let connection_info = TcpStreamConnectionInfo {
            address: self.local_address.to_string(),
            subject: subject.clone(),
            context: context.id().to_string(),
            stream_type: StreamType::Request,
        }
        .into();
        let state = self.state.clone();
        RegisteredStream::new(connection_info, pending_rx).with_cleanup(move || {
            remove_subject(&state, &subject);
        })
    }

    pub async fn associate_request_instance(&self, subject: &str, id: &EndpointInstanceId) -> bool {
        let mut state = self.state.lock();
        let now = Instant::now();
        prune_tombstones(&mut state, now);
        if state.removed_instances.contains_key(id) {
            state.pending.remove(subject);
            return false;
        }
        state
            .subject_instance
            .insert(subject.to_string(), id.clone());
        state
            .instance_subjects
            .entry(id.clone())
            .or_default()
            .insert(subject.to_string());
        true
    }

    pub async fn cancel_send_stream(&self, subject: &str) {
        remove_subject(&self.state, subject);
    }

    pub async fn cancel_instance_streams(&self, id: &EndpointInstanceId) -> usize {
        let subjects = {
            let mut state = self.state.lock();
            let now = Instant::now();
            prune_tombstones(&mut state, now);
            state.removed_instances.insert(id.clone(), now);
            state.instance_subjects.remove(id).unwrap_or_default()
        };
        let count = subjects.len();
        for subject in subjects {
            remove_subject(&self.state, &subject);
        }
        count
    }

    pub async fn clear_instance_tombstone(&self, id: &EndpointInstanceId) {
        self.state.lock().removed_instances.remove(id);
    }
}

fn remove_subject(state: &Mutex<State>, subject: &str) {
    let mut state = state.lock();
    state.pending.remove(subject);
    if let Some(instance) = state.subject_instance.remove(subject)
        && let Some(subjects) = state.instance_subjects.get_mut(&instance)
    {
        subjects.remove(subject);
        if subjects.is_empty() {
            state.instance_subjects.remove(&instance);
        }
    }
}

fn take_request(state: &Mutex<State>, subject: &str) -> Option<RequestedSendConnection> {
    let mut state = state.lock();
    let pending = state.pending.remove(subject);
    if let Some(instance) = state.subject_instance.remove(subject)
        && let Some(subjects) = state.instance_subjects.get_mut(&instance)
    {
        subjects.remove(subject);
        if subjects.is_empty() {
            state.instance_subjects.remove(&instance);
        }
    }
    pending
}

async fn start_listener(address: SocketAddr, state: Arc<Mutex<State>>) -> Result<SocketAddr> {
    let (ready_tx, ready_rx) = oneshot::channel();
    let listener_state = state.clone();
    let task = tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(address).await {
            Ok(listener) => listener,
            Err(error) => {
                let _ = ready_tx.send(Err(anyhow!(error)));
                return Err(anyhow!("failed binding TCP callback listener on {address}"));
            }
        };
        let local_address = listener.local_addr()?;
        let _ = ready_tx.send(Ok(local_address));
        loop {
            let (stream, peer) = listener.accept().await?;
            stream.set_nodelay(true)?;
            let state = listener_state.clone();
            tokio::spawn(async move {
                if let Err(error) = process_stream(stream, state).await {
                    tracing::warn!(%peer, %error, "TCP request callback failed");
                }
            });
        }
    });
    state.lock().listener_task = Some(task);
    ready_rx
        .await
        .context("TCP callback listener exited before ready")?
}

async fn process_stream(stream: tokio::net::TcpStream, state: Arc<Mutex<State>>) -> Result<()> {
    let (read_half, write_half) = tokio::io::split(stream);
    let mut reader = FramedRead::new(read_half, TwoPartCodec::default());
    let writer = FramedWrite::new(write_half, TwoPartCodec::default());
    let first = reader
        .next()
        .await
        .ok_or_else(|| anyhow!("TCP callback closed before handshake"))??;
    let header = first
        .header()
        .ok_or_else(|| anyhow!("TCP callback handshake must be header-only"))?;
    let handshake: CallHomeHandshake = serde_json::from_slice(header)?;
    if handshake.stream_type != StreamType::Request {
        return Err(anyhow!(
            "TCP response callbacks are unsupported; coordinated QUIC upgrade required"
        ));
    }
    drop(reader);

    let requested = take_request(&state, &handshake.subject)
        .ok_or_else(|| anyhow!("unknown TCP request callback subject {}", handshake.subject))?;
    let (request_tx, request_rx) = mpsc::channel(requested.send_buffer_count.max(1));
    requested
        .connection
        .send(Ok(StreamSender { tx: request_tx }))
        .map_err(|_| anyhow!("TCP request callback registrant was dropped"))?;
    send_requests(writer, request_rx, requested.context).await;
    Ok(())
}

async fn send_requests(
    mut writer: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
    mut request_rx: mpsc::Receiver<TwoPartMessage>,
    context: Arc<dyn AsyncEngineContext>,
) {
    let killed = context.killed();
    let stopped = context.stopped();
    tokio::pin!(killed, stopped);
    let closing = loop {
        tokio::select! {
            biased;
            _ = &mut killed => break Some(ControlMessage::Kill),
            _ = &mut stopped => break Some(ControlMessage::Stop),
            frame = request_rx.recv() => match frame {
                Some(frame) => {
                    if writer.send(frame).await.is_err() { break None; }
                }
                None => break Some(ControlMessage::Sentinel),
            }
        }
    };
    if let Some(control) = closing
        && let Ok(header) = serde_json::to_vec(&control)
    {
        let _ = writer
            .send(TwoPartMessage::from_header(header.into()))
            .await;
    }
    let mut stream = writer.into_inner();
    let _ = stream.flush().await;
    let _ = stream.shutdown().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::AsyncEngineContextProvider, pipeline::Context};

    struct FailingResolver;

    impl IpResolver for FailingResolver {
        fn local_ip(&self) -> Result<IpAddr, LocalIpError> {
            Err(LocalIpError::LocalIpAddressNotFound)
        }

        fn local_ipv6(&self) -> Result<IpAddr, LocalIpError> {
            Err(LocalIpError::LocalIpAddressNotFound)
        }
    }

    #[tokio::test]
    async fn falls_back_to_loopback_when_no_routable_ip_exists() {
        let server = TcpStreamServer::new_with_resolver(ServerOptions::default(), FailingResolver)
            .await
            .unwrap();
        assert_eq!(
            server.local_address().unwrap().ip(),
            IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
        );
    }

    #[tokio::test]
    async fn tombstone_rejects_late_request_registration_association() {
        let server = TcpStreamServer::new_with_resolver(ServerOptions::default(), FailingResolver)
            .await
            .unwrap();
        let id = EndpointInstanceId {
            namespace: "n".to_string(),
            component: "c".to_string(),
            endpoint: "e".to_string(),
            instance_id: 1,
        };
        server.cancel_instance_streams(&id).await;
        let context = Context::new(());
        let registered = server.register_request(context.context(), 64);
        let info = TcpStreamConnectionInfo::try_from(registered.connection_info.clone()).unwrap();
        assert!(!server.associate_request_instance(&info.subject, &id).await);
        assert!(registered.stream_provider.await.is_err());
    }
}
