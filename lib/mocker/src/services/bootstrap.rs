// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Framed transport for mocker prefill/decode handoff sessions.
//!
//! Session and request ownership live in `dynamo-llm`. This module only
//! validates framed connections and hands them to that owner.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;
use futures::{FutureExt, SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Semaphore, mpsc, watch};
use tokio_util::codec::{Framed, LengthDelimitedCodec};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use uuid::Uuid;

use crate::common::handoff::{
    HandoffActionId, HandoffActionOutcome, HandoffFact, HandoffId, HandoffOrder,
    IssuedHandoffAction,
};
use crate::common::protocols::EngineType;

pub const BOOTSTRAP_PROTOCOL_VERSION: u16 = 1;
pub const MAX_BOOTSTRAP_FRAME_BYTES: usize = 64 * 1024;
const MAGIC: [u8; 4] = *b"DMHF";
const HEADER_BYTES: usize = 8;
const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug)]
pub struct BootstrapServerConfig {
    pub max_pending_connections: usize,
    pub registration_timeout: Duration,
}

impl Default for BootstrapServerConfig {
    fn default() -> Self {
        Self {
            max_pending_connections: 256,
            registration_timeout: RENDEZVOUS_TIMEOUT,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BootstrapParticipantRole {
    Destination,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BootstrapIdentity {
    pub handoff_id: HandoffId,
    pub bootstrap_room: u64,
    pub request_id: Uuid,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParticipantRegistration {
    pub role: BootstrapParticipantRole,
    pub dp_rank: u32,
    pub order: HandoffOrder,
    pub engine_type: EngineType,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BootstrapMessage {
    Register(ParticipantRegistration),
    Registered,
    Action(IssuedHandoffAction),
    ActionAck {
        action_id: HandoffActionId,
        outcome: HandoffActionOutcome,
    },
    Fact(HandoffFact),
    Complete,
    Abort {
        message: String,
    },
    Overloaded,
    ProtocolError {
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct BootstrapWireFrame {
    identity: BootstrapIdentity,
    message: BootstrapMessage,
}

pub struct IncomingBootstrapConnection {
    pub identity: BootstrapIdentity,
    pub registration: ParticipantRegistration,
    pub connection: BootstrapConnection,
}

pub struct BootstrapConnection {
    identity: BootstrapIdentity,
    framed: Framed<TcpStream, LengthDelimitedCodec>,
}

impl BootstrapConnection {
    pub fn identity(&self) -> &BootstrapIdentity {
        &self.identity
    }

    pub async fn send(&mut self, message: BootstrapMessage) -> Result<()> {
        let payload = serde_json::to_vec(&BootstrapWireFrame {
            identity: self.identity.clone(),
            message,
        })?;
        if payload.is_empty() || payload.len() > MAX_BOOTSTRAP_FRAME_BYTES {
            bail!(
                "bootstrap frame length {} is outside 1..={MAX_BOOTSTRAP_FRAME_BYTES}",
                payload.len()
            );
        }
        tokio::time::timeout(RENDEZVOUS_TIMEOUT, self.framed.send(Bytes::from(payload)))
            .await
            .map_err(|_| anyhow!("bootstrap send timed out"))??;
        Ok(())
    }

    pub async fn recv(&mut self) -> Result<Option<BootstrapMessage>> {
        let Some(frame) = self.framed.next().await.transpose()? else {
            return Ok(None);
        };
        if frame.is_empty() {
            bail!("bootstrap received an empty frame");
        }
        let frame: BootstrapWireFrame =
            serde_json::from_slice(&frame).context("bootstrap frame contains malformed JSON")?;
        if frame.identity != self.identity {
            bail!("bootstrap frame changed session identity");
        }
        Ok(Some(frame.message))
    }

    pub fn peer_closed_now(&self) -> Result<bool> {
        let mut byte = [0u8; 1];
        match self.framed.get_ref().peek(&mut byte).now_or_never() {
            Some(Ok(0)) => Ok(true),
            Some(Ok(_)) | None => Ok(false),
            Some(Err(error)) => Err(error.into()),
        }
    }
}

pub struct BootstrapServer {
    port: u16,
    incoming_rx: Mutex<Option<mpsc::Receiver<IncomingBootstrapConnection>>>,
    closed_rx: watch::Receiver<bool>,
    #[cfg(test)]
    accepted_with_slot_rx: watch::Receiver<u64>,
}

impl BootstrapServer {
    pub async fn start(
        port: u16,
        cancel: CancellationToken,
        config: BootstrapServerConfig,
    ) -> Result<Arc<Self>> {
        if config.max_pending_connections == 0 {
            bail!("bootstrap max_pending_connections must be at least one");
        }
        let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
        let actual_port = listener.local_addr()?.port();
        let (incoming_tx, incoming_rx) = mpsc::channel(config.max_pending_connections);
        let permits = Arc::new(Semaphore::new(config.max_pending_connections));
        let overload_permits = Arc::new(Semaphore::new(1));
        let (closed_tx, closed_rx) = watch::channel(false);
        #[cfg(test)]
        let (accepted_with_slot_tx, accepted_with_slot_rx) = watch::channel(0_u64);
        let server = Arc::new(Self {
            port: actual_port,
            incoming_rx: Mutex::new(Some(incoming_rx)),
            closed_rx,
            #[cfg(test)]
            accepted_with_slot_rx,
        });

        tokio::spawn(async move {
            let connections = TaskTracker::new();
            #[cfg(test)]
            let mut accepted_with_slot = 0_u64;
            loop {
                let accepted = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    accepted = listener.accept() => accepted,
                };
                let Ok((stream, _)) = accepted else {
                    tokio::select! {
                        _ = cancel.cancelled() => break,
                        _ = tokio::time::sleep(Duration::from_millis(10)) => {}
                    }
                    continue;
                };
                let Ok(permit) = permits.clone().try_acquire_owned() else {
                    let Ok(overload_permit) = overload_permits.clone().try_acquire_owned() else {
                        drop(stream);
                        continue;
                    };
                    let registration_timeout = config.registration_timeout;
                    let connection_cancel = cancel.clone();
                    connections.spawn(async move {
                        let _permit = overload_permit;
                        let incoming = tokio::select! {
                            biased;
                            _ = connection_cancel.cancelled() => return,
                            incoming = tokio::time::timeout(
                                registration_timeout,
                                accept_connection(stream),
                            ) => incoming,
                        };
                        let Ok(Ok(mut incoming)) = incoming else {
                            return;
                        };
                        let _ = incoming.connection.send(BootstrapMessage::Overloaded).await;
                    });
                    continue;
                };
                #[cfg(test)]
                {
                    accepted_with_slot = accepted_with_slot
                        .checked_add(1)
                        .expect("bootstrap accepted-connection test counter overflow");
                    let _ = accepted_with_slot_tx.send(accepted_with_slot);
                }
                let incoming_tx = incoming_tx.clone();
                let registration_timeout = config.registration_timeout;
                let connection_cancel = cancel.clone();
                connections.spawn(async move {
                    let _permit = permit;
                    let incoming = tokio::select! {
                        biased;
                        _ = connection_cancel.cancelled() => return,
                        incoming = tokio::time::timeout(
                            registration_timeout,
                            accept_connection(stream),
                        ) => incoming,
                    };
                    let Ok(incoming) = incoming else {
                        return;
                    };
                    let Ok(mut incoming) = incoming else {
                        return;
                    };
                    if let Err(error) = incoming_tx.try_send(incoming) {
                        incoming = error.into_inner();
                        let _ = incoming.connection.send(BootstrapMessage::Overloaded).await;
                    }
                });
            }
            connections.close();
            connections.wait().await;
            let _ = closed_tx.send(true);
        });

        Ok(server)
    }

    pub fn take_incoming_receiver(&self) -> Option<mpsc::Receiver<IncomingBootstrapConnection>> {
        self.incoming_rx
            .lock()
            .expect("bootstrap incoming receiver mutex poisoned")
            .take()
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub async fn wait_closed(&self) {
        let mut closed_rx = self.closed_rx.clone();
        if *closed_rx.borrow() {
            return;
        }
        let _ = closed_rx.wait_for(|closed| *closed).await;
    }

    #[cfg(test)]
    async fn wait_for_accepted_with_slot(&self, expected: u64) {
        let mut accepted = self.accepted_with_slot_rx.clone();
        let _ = accepted.wait_for(|count| *count >= expected).await;
    }
}

pub async fn connect_to_prefill(
    host: &str,
    port: u16,
    identity: BootstrapIdentity,
    registration: ParticipantRegistration,
) -> Result<BootstrapConnection> {
    let addr = bootstrap_addr(host, port);
    let mut stream = tokio::time::timeout(RENDEZVOUS_TIMEOUT, TcpStream::connect(&addr))
        .await
        .map_err(|_| anyhow!("bootstrap connect timeout to {addr}"))??;
    tokio::time::timeout(RENDEZVOUS_TIMEOUT, write_header(&mut stream))
        .await
        .map_err(|_| anyhow!("bootstrap header send timed out"))??;
    let mut connection = BootstrapConnection {
        identity,
        framed: framed(stream),
    };
    connection
        .send(BootstrapMessage::Register(registration))
        .await?;
    Ok(connection)
}

fn bootstrap_addr(host: &str, port: u16) -> String {
    let host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

async fn accept_connection(mut stream: TcpStream) -> Result<IncomingBootstrapConnection> {
    read_header(&mut stream).await?;
    let mut connection = BootstrapConnection {
        identity: BootstrapIdentity {
            handoff_id: HandoffId::default(),
            bootstrap_room: 0,
            request_id: Uuid::nil(),
        },
        framed: framed(stream),
    };
    let Some(frame) = connection.framed.next().await.transpose()? else {
        bail!("bootstrap connection closed before registration");
    };
    let frame: BootstrapWireFrame =
        serde_json::from_slice(&frame).context("bootstrap registration contains malformed JSON")?;
    let BootstrapMessage::Register(registration) = frame.message else {
        bail!("bootstrap first frame must register a participant");
    };
    if registration.role != BootstrapParticipantRole::Destination {
        bail!("bootstrap server accepts only destination participants");
    }
    connection.identity = frame.identity.clone();
    Ok(IncomingBootstrapConnection {
        identity: frame.identity,
        registration,
        connection,
    })
}

fn framed(stream: TcpStream) -> Framed<TcpStream, LengthDelimitedCodec> {
    LengthDelimitedCodec::builder()
        .little_endian()
        .length_field_type::<u32>()
        .max_frame_length(MAX_BOOTSTRAP_FRAME_BYTES)
        .new_framed(stream)
}

async fn write_header(stream: &mut TcpStream) -> Result<()> {
    let mut header = [0u8; HEADER_BYTES];
    header[..4].copy_from_slice(&MAGIC);
    header[4..6].copy_from_slice(&BOOTSTRAP_PROTOCOL_VERSION.to_le_bytes());
    stream.write_all(&header).await?;
    Ok(())
}

async fn read_header(stream: &mut TcpStream) -> Result<()> {
    let mut header = [0u8; HEADER_BYTES];
    stream.read_exact(&mut header).await?;
    if header[..4] != MAGIC {
        bail!("bootstrap protocol magic mismatch");
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != BOOTSTRAP_PROTOCOL_VERSION {
        bail!("unsupported bootstrap protocol version {version}");
    }
    let flags = u16::from_le_bytes([header[6], header[7]]);
    if flags != 0 {
        bail!("bootstrap protocol flags must be zero");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(value: u128) -> BootstrapIdentity {
        BootstrapIdentity {
            handoff_id: HandoffId::from(Uuid::from_u128(value)),
            bootstrap_room: value as u64,
            request_id: Uuid::from_u128(value + 100),
        }
    }

    fn registration() -> ParticipantRegistration {
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
        }
    }

    #[test]
    fn bootstrap_address_preserves_ipv6_literals() {
        assert_eq!(bootstrap_addr("[::1]", 1234), "[::1]:1234");
        assert_eq!(bootstrap_addr("::1", 1234), "[::1]:1234");
        assert_eq!(bootstrap_addr("127.0.0.1", 1234), "127.0.0.1:1234");
    }

    #[tokio::test]
    async fn send_rejects_oversized_frame_before_codec_allocation() {
        let cancel = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel.clone(), BootstrapServerConfig::default())
            .await
            .unwrap();
        let mut client =
            connect_to_prefill("127.0.0.1", server.port(), identity(2), registration())
                .await
                .unwrap();
        let error = client
            .send(BootstrapMessage::ProtocolError {
                message: "x".repeat(MAX_BOOTSTRAP_FRAME_BYTES),
            })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("outside"));
        cancel.cancel();
    }

    #[tokio::test]
    async fn bad_magic_never_enters_the_incoming_queue() {
        let cancel = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel.clone(), BootstrapServerConfig::default())
            .await
            .unwrap();
        let mut incoming_rx = server.take_incoming_receiver().unwrap();
        let mut stream = TcpStream::connect(("127.0.0.1", server.port()))
            .await
            .unwrap();
        stream.write_all(b"NOPE\x01\x00\x00\x00").await.unwrap();
        let mut byte = [0_u8; 1];
        let _ = stream.read(&mut byte).await;
        assert!(matches!(
            incoming_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        cancel.cancel();
    }

    #[tokio::test]
    async fn invalid_headers_and_frames_never_enter_session_ownership() {
        let cancel = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel.clone(), BootstrapServerConfig::default())
            .await
            .unwrap();
        let mut incoming_rx = server.take_incoming_receiver().unwrap();

        for header in [
            [MAGIC[0], MAGIC[1], MAGIC[2], MAGIC[3], 2, 0, 0, 0],
            [MAGIC[0], MAGIC[1], MAGIC[2], MAGIC[3], 1, 0, 1, 0],
        ] {
            let mut stream = TcpStream::connect(("127.0.0.1", server.port()))
                .await
                .unwrap();
            stream.write_all(&header).await.unwrap();
            let mut byte = [0_u8; 1];
            let _ = stream.read(&mut byte).await;
        }

        let mut malformed = TcpStream::connect(("127.0.0.1", server.port()))
            .await
            .unwrap();
        write_header(&mut malformed).await.unwrap();
        malformed.write_all(&1u32.to_le_bytes()).await.unwrap();
        malformed.write_all(b"{").await.unwrap();
        let mut byte = [0_u8; 1];
        let _ = malformed.read(&mut byte).await;

        let mut oversized = TcpStream::connect(("127.0.0.1", server.port()))
            .await
            .unwrap();
        write_header(&mut oversized).await.unwrap();
        oversized
            .write_all(&((MAX_BOOTSTRAP_FRAME_BYTES + 1) as u32).to_le_bytes())
            .await
            .unwrap();
        let _ = oversized.read(&mut byte).await;

        assert!(matches!(
            incoming_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        cancel.cancel();
        server.wait_closed().await;
    }

    #[tokio::test]
    async fn changed_identity_is_rejected_after_registration() {
        let cancel = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel.clone(), BootstrapServerConfig::default())
            .await
            .unwrap();
        let mut incoming_rx = server.take_incoming_receiver().unwrap();
        let mut client =
            connect_to_prefill("127.0.0.1", server.port(), identity(3), registration())
                .await
                .unwrap();
        let mut incoming = incoming_rx.recv().await.unwrap();

        client.identity = identity(4);
        client.send(BootstrapMessage::Complete).await.unwrap();
        let error = incoming.connection.recv().await.unwrap_err();
        assert!(error.to_string().contains("changed session identity"));
        cancel.cancel();
    }

    #[tokio::test]
    async fn first_frame_must_be_registration() {
        let cancel = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel.clone(), BootstrapServerConfig::default())
            .await
            .unwrap();
        let mut incoming_rx = server.take_incoming_receiver().unwrap();
        let mut stream = TcpStream::connect(("127.0.0.1", server.port()))
            .await
            .unwrap();
        write_header(&mut stream).await.unwrap();
        let frame = serde_json::to_vec(&BootstrapWireFrame {
            identity: identity(5),
            message: BootstrapMessage::Registered,
        })
        .unwrap();
        let mut framed_stream = framed(stream);
        framed_stream.send(Bytes::from(frame)).await.unwrap();
        let _ = framed_stream.next().await;

        assert!(matches!(
            incoming_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        cancel.cancel();
    }

    #[tokio::test]
    async fn full_incoming_queue_returns_overloaded() {
        let cancel = CancellationToken::new();
        let server = BootstrapServer::start(
            0,
            cancel.clone(),
            BootstrapServerConfig {
                max_pending_connections: 1,
                ..BootstrapServerConfig::default()
            },
        )
        .await
        .unwrap();
        let _incoming_rx = server.take_incoming_receiver().unwrap();
        let _first = connect_to_prefill("127.0.0.1", server.port(), identity(6), registration())
            .await
            .unwrap();
        server.wait_for_accepted_with_slot(1).await;
        let mut second =
            connect_to_prefill("127.0.0.1", server.port(), identity(7), registration())
                .await
                .unwrap();

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), second.recv())
                .await
                .unwrap()
                .unwrap(),
            Some(BootstrapMessage::Overloaded)
        );
        cancel.cancel();
    }

    #[tokio::test]
    async fn half_open_registration_saturation_returns_overloaded() {
        let cancel = CancellationToken::new();
        let server = BootstrapServer::start(
            0,
            cancel.clone(),
            BootstrapServerConfig {
                max_pending_connections: 1,
                ..BootstrapServerConfig::default()
            },
        )
        .await
        .unwrap();
        let _incoming_rx = server.take_incoming_receiver().unwrap();
        let _half_open = TcpStream::connect(("127.0.0.1", server.port()))
            .await
            .unwrap();
        server.wait_for_accepted_with_slot(1).await;

        let mut rejected =
            connect_to_prefill("127.0.0.1", server.port(), identity(8), registration())
                .await
                .unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), rejected.recv())
                .await
                .unwrap()
                .unwrap(),
            Some(BootstrapMessage::Overloaded)
        );
        cancel.cancel();
        server.wait_closed().await;
    }

    #[tokio::test]
    async fn shutdown_closes_half_open_registration() {
        let cancel = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel.clone(), BootstrapServerConfig::default())
            .await
            .unwrap();
        let mut stream = TcpStream::connect(("127.0.0.1", server.port()))
            .await
            .unwrap();
        server.wait_for_accepted_with_slot(1).await;
        cancel.cancel();

        let mut byte = [0u8; 1];
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), stream.read(&mut byte))
                .await
                .unwrap()
                .unwrap(),
            0
        );
        server.wait_closed().await;
    }
}
