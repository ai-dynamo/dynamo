// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A stand-in frontend for driving `Ingress` through `handle_payload`.

#![allow(dead_code)]

use bytes::Bytes;
use futures::StreamExt;
use serde::Serialize;
use tokio_util::codec::FramedRead;

use dynamo_runtime::pipeline::network::{
    ConnectionInfo, ControlMessage, StreamType,
    codec::{TwoPartCodec, TwoPartMessage},
    tcp::TcpStreamConnectionInfo,
};

/// A frontend's response endpoint: accepts each worker connection, reads its
/// frames and closes it after the closing sentinel, as the frontend does. It
/// runs on its own thread, so a test that counts allocations on its own thread
/// does not count the peer's.
pub fn start_response_peer() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    listener.set_nonblocking(true).unwrap();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let sentinel = serde_json::to_vec(&ControlMessage::Sentinel).unwrap();
                    let mut frames = FramedRead::new(socket, TwoPartCodec::default());
                    while let Some(Ok(frame)) = frames.next().await {
                        if let (Some(header), None) = frame.optional_parts()
                            && header.as_ref() == sentinel.as_slice()
                        {
                            break;
                        }
                    }
                });
            }
        });
    });
    address
}

/// One single-in, many-out request-plane frame, as a frontend sends it. Its
/// response stream connects to `address`.
pub fn request_payload(address: &str, id: &str, body: &impl Serialize) -> Bytes {
    let connection: ConnectionInfo = TcpStreamConnectionInfo {
        address: address.to_string(),
        subject: id.to_string(),
        context: id.to_string(),
        stream_type: StreamType::Response,
    }
    .into();
    let header = serde_json::to_vec(&serde_json::json!({
        "id": id,
        "request_type": "single_in",
        "response_type": "many_out",
        "connection_info": connection,
    }))
    .unwrap();
    let body = serde_json::to_vec(body).unwrap();
    TwoPartCodec::default()
        .encode_message(TwoPartMessage::from_parts(header.into(), body.into()))
        .unwrap()
}
