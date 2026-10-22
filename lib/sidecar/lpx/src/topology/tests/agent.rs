// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The publisher's client of the Agent's `TopologyService`.

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tonic::Code;

use crate::topology::agent::AgentClient;
use crate::topology::proto::GetSnapshotRequest;

/// HTTP/2 frame types and error codes (RFC 9113 sections 6 and 7).
const HEADERS: u8 = 0x1;
const RST_STREAM: u8 = 0x3;
const SETTINGS: u8 = 0x4;
const GOAWAY: u8 = 0x7;
const SETTINGS_ACK: u8 = 0x1;
const NO_ERROR: u32 = 0x0;
const CANCEL: u32 = 0x8;
/// The client connection preface (RFC 9113 section 3.4).
const PREFACE: usize = 24;

fn frame(kind: u8, flags: u8, stream: u32, payload: &[u8]) -> Vec<u8> {
    let length = u32::try_from(payload.len()).unwrap().to_be_bytes();
    [&length[1..], &[kind, flags], &stream.to_be_bytes(), payload].concat()
}

/// Closes the connection without processing the call on `_stream`.
fn go_away(_stream: u32) -> Vec<u8> {
    let last_processed = 0u32;
    let payload = [last_processed.to_be_bytes(), NO_ERROR.to_be_bytes()].concat();
    frame(GOAWAY, 0, 0, &payload)
}

/// Cancels the call on `stream`.
fn reset(stream: u32) -> Vec<u8> {
    frame(RST_STREAM, 0, stream, &CANCEL.to_be_bytes())
}

/// The kind and stream of the next frame, or `None` once the client is gone.
async fn next_frame(connection: &mut UnixStream) -> Option<(u8, u8, u32)> {
    let mut header = [0u8; 9];
    connection.read_exact(&mut header).await.ok()?;
    let length = u32::from_be_bytes([0, header[0], header[1], header[2]]);
    let stream = u32::from_be_bytes([header[5], header[6], header[7], header[8]]) & 0x7fff_ffff;
    let mut payload = vec![0u8; usize::try_from(length).unwrap()];
    connection.read_exact(&mut payload).await.ok()?;
    Some((header[3], header[4], stream))
}

/// An HTTP/2 peer on the Agent socket that ends each call with `ending`
/// instead of a gRPC answer, as an Agent shutting down or restarting may.
async fn end_calls(listener: UnixListener, ending: fn(u32) -> Vec<u8>) {
    loop {
        let (mut connection, _) = listener.accept().await.unwrap();
        tokio::spawn(async move {
            let mut preface = [0u8; PREFACE];
            connection.read_exact(&mut preface).await.unwrap();
            connection
                .write_all(&frame(SETTINGS, 0, 0, &[]))
                .await
                .unwrap();
            while let Some((kind, flags, stream)) = next_frame(&mut connection).await {
                let reply = match kind {
                    SETTINGS if flags & SETTINGS_ACK == 0 => frame(SETTINGS, SETTINGS_ACK, 0, &[]),
                    HEADERS => ending(stream),
                    _ => continue,
                };
                if connection.write_all(&reply).await.is_err() {
                    return;
                }
            }
        });
    }
}

/// A call the Agent's connection ends without an answer did not reach the
/// Agent, so it is UNAVAILABLE, which the publisher retries, whatever code
/// the transport gives it.
#[tokio::test]
async fn a_call_ended_by_the_agent_connection_is_unavailable() {
    for (what, ending) in [
        ("the connection closes", go_away as fn(u32) -> Vec<u8>),
        ("the call is reset", reset),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("agent.sock");
        let peer = tokio::spawn(end_calls(UnixListener::bind(&socket).unwrap(), ending));
        let agent = AgentClient::new(&socket).unwrap();

        let status = agent
            .get_snapshot(GetSnapshotRequest::default(), Duration::from_secs(5))
            .await
            .expect_err(what);

        assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
        let unreachable = format!(
            "the Agent's topology socket {} is unreachable: ",
            socket.display()
        );
        assert!(
            status.message().starts_with(&unreachable),
            "{what}: {status:?}"
        );
        peer.abort();
    }
}

/// An Agent socket that does not exist yet or that nothing serves is
/// UNAVAILABLE. The conductor that finally sees the message runs in another
/// pod, so it names the Agent's socket and the cause.
#[tokio::test]
async fn an_unreachable_agent_socket_is_named_with_its_cause() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.sock");
    let refused = dir.path().join("refused.sock");
    // The socket file stays behind when its listener closes.
    drop(std::os::unix::net::UnixListener::bind(&refused).unwrap());
    for (socket, cause) in [
        (&missing, "No such file or directory"),
        (&refused, "Connection refused"),
    ] {
        let agent = AgentClient::new(socket).unwrap();

        let status = agent
            .get_snapshot(GetSnapshotRequest::default(), Duration::from_secs(5))
            .await
            .expect_err(cause);

        assert_eq!(status.code(), Code::Unavailable, "{cause}: {status:?}");
        let unreachable = format!(
            "the Agent's topology socket {} is unreachable: ",
            socket.display()
        );
        assert!(
            status.message().starts_with(&unreachable) && status.message().contains(cause),
            "{cause}: {status:?}"
        );
    }
}
