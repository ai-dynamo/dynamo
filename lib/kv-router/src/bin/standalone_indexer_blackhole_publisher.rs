// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "standalone-indexer")]

use std::thread;
use std::time::Duration;

use anyhow::Result;
use dynamo_kv_router::zmq_wire::{BlockHashValue, RawKvEvent};

const ZMQ_SNDTIMEOUT_MS: i32 = 0;
const ZMQ_RECONNECT_IVL_MS: i32 = 100;
const ZMQ_RECONNECT_IVL_MAX_MS: i32 = 5000;
const ZMQ_TCP_KEEPALIVE: i32 = 1;
const ZMQ_HEARTBEAT_IVL_MS: i32 = 5000;
const ZMQ_HEARTBEAT_TIMEOUT_MS: i32 = 15000;
const ZMQ_HEARTBEAT_TTL_MS: i32 = 15000;
const ZMQ_LINGER_MS: i32 = 0;

fn usage() -> anyhow::Error {
    anyhow::anyhow!("usage: standalone_indexer_blackhole_publisher <bind-endpoint> [interval-ms]")
}

fn configure_pub_socket(socket: &zmq::Socket) -> Result<()> {
    socket.set_linger(ZMQ_LINGER_MS)?;
    socket.set_reconnect_ivl(ZMQ_RECONNECT_IVL_MS)?;
    socket.set_reconnect_ivl_max(ZMQ_RECONNECT_IVL_MAX_MS)?;
    socket.set_tcp_keepalive(ZMQ_TCP_KEEPALIVE)?;
    socket.set_heartbeat_ivl(ZMQ_HEARTBEAT_IVL_MS)?;
    socket.set_heartbeat_timeout(ZMQ_HEARTBEAT_TIMEOUT_MS)?;
    socket.set_heartbeat_ttl(ZMQ_HEARTBEAT_TTL_MS)?;
    socket.set_sndtimeo(ZMQ_SNDTIMEOUT_MS)?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(endpoint) = args.next() else {
        return Err(usage());
    };
    let interval_ms = args
        .next()
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(100);

    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::PUB)?;
    configure_pub_socket(&socket)?;
    socket.bind(&endpoint)?;

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
        socket.send_multipart(vec![Vec::new(), seq.to_be_bytes().to_vec(), payload], 0)?;

        seq += 1;
        thread::sleep(Duration::from_millis(interval_ms));
    }
}
