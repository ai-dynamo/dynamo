// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "standalone-indexer")]

use std::time::Duration;

use bytes::Bytes;
use dynamo_kv_router::zmq_wire::{BlockHashValue, RawKvEvent};
use zeromq::{PubSocket, Socket, SocketSend, ZmqMessage};

fn usage() -> anyhow::Error {
    anyhow::anyhow!("usage: standalone_indexer_blackhole_publisher <bind-endpoint> [interval-ms]")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(endpoint) = args.next() else {
        return Err(usage());
    };
    let interval_ms = args
        .next()
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(100);

    let mut socket = PubSocket::new();
    socket.bind(&endpoint).await?;

    let mut seq = 0u64;
    loop {
        let event = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(seq + 1)],
            parent_block_hash: None,
            token_ids: vec![seq as u32 + 1],
            block_size: 1,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
            is_eagle: None,
        };
        let payload = rmp_serde::to_vec(&(0.0_f64, vec![event], Some(0_i32)))?;
        let frames = vec![
            Bytes::new(),
            Bytes::from(seq.to_be_bytes().to_vec()),
            Bytes::from(payload),
        ];
        let message = ZmqMessage::try_from(frames)
            .map_err(|error| anyhow::anyhow!("failed to build ZMQ message: {error}"))?;
        socket.send(message).await?;

        seq += 1;
        tokio::time::sleep(Duration::from_millis(interval_ms)).await;
    }
}
