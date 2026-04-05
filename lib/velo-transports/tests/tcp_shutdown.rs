// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for TCP graceful shutdown

#[macro_use]
mod common;

use common::{TcpShutdownClient, shutdown_scenarios};

transport_shutdown_tests!(tcp, TcpShutdownClient);

// --- Codec-level test (not transport-specific, but lives here for TCP coverage) ---
#[test]
fn test_shutting_down_frame_roundtrip() {
    use bytes::BytesMut;
    use tokio_util::codec::Decoder;
    use velo_transports::MessageType;
    use velo_transports::tcp::TcpFrameCodec;

    let header = b"correlation-header";
    let payload = b"";

    let mut buf = Vec::new();
    TcpFrameCodec::encode_frame_sync(&mut buf, MessageType::ShuttingDown, header, payload).unwrap();

    let mut codec = TcpFrameCodec::new();
    let mut bytes = BytesMut::from(&buf[..]);
    let (msg_type, decoded_header, decoded_payload) = codec.decode(&mut bytes).unwrap().unwrap();

    assert_eq!(msg_type, MessageType::ShuttingDown);
    assert_eq!(&decoded_header[..], header);
    assert_eq!(decoded_payload.len(), 0);
}
