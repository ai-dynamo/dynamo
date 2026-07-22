// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ KV event publishing in the vLLM native wire format.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use bytes::Bytes;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, StorageTier};
use futures::{Sink, SinkExt, StreamExt};
use serde::Serialize;
use tmq::{
    Context, Multipart, SocketBuilder,
    publish::{Publish, publish},
    router::{Router, router},
};
use tokio::sync::{Mutex, mpsc};

use crate::common::protocols::{RawKvEvent, RawKvEventSink};

type MultipartMessage = Vec<Vec<u8>>;
type SharedPubSocket = Arc<Mutex<Publish>>;

const ZMQ_SNDTIMEOUT_MS: i32 = 0;
const ZMQ_RCVTIMEOUT_MS: i32 = 100;
const ZMQ_RECONNECT_IVL_MS: i32 = 100;
const ZMQ_RECONNECT_IVL_MAX_MS: i32 = 5000;
const ZMQ_TCP_KEEPALIVE: i32 = 1;
const ZMQ_LINGER_MS: i32 = 0;

/// Maximum number of entries in the replay ring buffer.
const REPLAY_BUFFER_CAPACITY: usize = 10_000;

#[derive(Serialize)]
#[serde(tag = "type")]
enum ZmqRawKvEvent {
    BlockStored {
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<u32>,
        block_size: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<&'static str>,
        group_idx: u32,
    },
    BlockRemoved {
        block_hashes: Vec<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        medium: Option<&'static str>,
        group_idx: u32,
    },
}

pub struct ZmqKvEventSink {
    tx: mpsc::UnboundedSender<RawKvEvent>,
}

impl ZmqKvEventSink {
    pub async fn new(
        port: u16,
        replay_port: Option<u16>,
        dp_rank: u32,
        block_size: u32,
    ) -> Result<Self> {
        let (tx, mut rx) = mpsc::unbounded_channel::<RawKvEvent>();

        let endpoint = format!("tcp://0.0.0.0:{port}");
        let pub_socket = bind_pub_socket(&endpoint)
            .await
            .map_err(|e| anyhow::anyhow!("ZMQ PUB bind to {endpoint} failed: {e}"))?;
        tracing::info!("ZmqKvEventSink bound to {endpoint} for dp_rank {dp_rank}");

        let mut router_socket = if let Some(rp) = replay_port {
            let replay_ep = format!("tcp://0.0.0.0:{rp}");
            let sock = bind_router_socket(&replay_ep)
                .await
                .map_err(|e| anyhow::anyhow!("ZMQ ROUTER bind to {replay_ep} failed: {e}"))?;
            tracing::info!(
                "ZmqKvEventSink replay ROUTER bound to {replay_ep} for dp_rank {dp_rank}"
            );
            Some(sock)
        } else {
            None
        };

        tokio::spawn(async move {
            let mut seq_num: u64 = 0;
            let mut ring_buffer: VecDeque<(u64, Bytes)> = VecDeque::new();

            loop {
                tokio::select! {
                    biased;

                    replay_result = async {
                        match router_socket.as_mut() {
                            Some(socket) => socket.next().await,
                            None => std::future::pending().await,
                        }
                    } => {
                        let req_msg = match replay_result {
                            Some(Ok(req_msg)) => multipart_message(req_msg),
                            Some(Err(error)) => {
                                tracing::warn!("Replay ROUTER recv error: {error}");
                                router_socket = None;
                                continue;
                            }
                            None => {
                                tracing::warn!("Replay ROUTER stream ended");
                                router_socket = None;
                                continue;
                            }
                        };
                        if req_msg.len() < 3 {
                            tracing::warn!("Unexpected replay request frame count: {}", req_msg.len());
                            continue;
                        }

                        let identity = Bytes::copy_from_slice(req_msg.first().unwrap());
                        let start_seq_bytes = req_msg.get(2).unwrap();
                        if start_seq_bytes.len() != 8 {
                            tracing::warn!("Invalid replay start_seq length: {}", start_seq_bytes.len());
                            continue;
                        }
                        let start_seq = u64::from_be_bytes(start_seq_bytes[..8].try_into().unwrap());

                        tracing::debug!(dp_rank, start_seq, buffer_len = ring_buffer.len(), "Replay request received");

                        let buffer_first = ring_buffer.front().map(|(seq, _)| *seq);
                        let buffer_last = ring_buffer.back().map(|(seq, _)| *seq);
                        let outside_buffer = match (buffer_first, buffer_last) {
                            (Some(first), Some(last)) => start_seq < first || start_seq > last,
                            _ => true,
                        };
                        if outside_buffer {
                            let buffer_first_display = buffer_first
                                .map(|seq| seq.to_string())
                                .unwrap_or_else(|| "empty".to_string());
                            let buffer_last_display = buffer_last
                                .map(|seq| seq.to_string())
                                .unwrap_or_else(|| "empty".to_string());
                            tracing::warn!(
                                dp_rank,
                                start_seq,
                                buffer_first = ?buffer_first,
                                buffer_last = ?buffer_last,
                                buffer_len = ring_buffer.len(),
                                "Replay request outside buffer: start_seq={start_seq}, buffer=[{},{}]",
                                buffer_first_display,
                                buffer_last_display,
                            );
                        }

                        let start_idx = ring_buffer.front()
                            .map(|(first_seq, _)| start_seq.saturating_sub(*first_seq) as usize)
                            .unwrap_or(0)
                            .min(ring_buffer.len());

                        let sock = router_socket.as_mut().unwrap();
                        for (seq, payload) in ring_buffer.iter().skip(start_idx) {
                            let frames = vec![
                                identity.clone().to_vec(),
                                Vec::new(),
                                seq.to_be_bytes().to_vec(),
                                payload.to_vec(),
                            ];
                            if let Err(e) = send_multipart_direct(sock, frames).await {
                                tracing::warn!("Replay send error: {e}");
                                break;
                            }
                        }

                        let sentinel_frames = vec![
                            identity.to_vec(),
                            Vec::new(),
                            (-1i64).to_be_bytes().to_vec(),
                            Vec::new(),
                        ];
                        let _ = send_multipart_direct(sock, sentinel_frames).await;
                    }

                    msg_opt = rx.recv() => {
                        let Some(msg) = msg_opt else { break };

                        let events = convert_to_zmq_events(
                            &msg.event,
                            msg.block_token_ids.as_deref(),
                            block_size,
                            msg.storage_tier,
                        );
                        if events.is_empty() {
                            continue;
                        }

                        let timestamp = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64();

                        let batch: (f64, Vec<ZmqRawKvEvent>, Option<i32>) =
                            (timestamp, events, Some(dp_rank as i32));
                        let payload: Bytes = match rmp_serde::to_vec(&batch) {
                            Ok(p) => p.into(),
                            Err(e) => {
                                tracing::warn!("Failed to serialize ZMQ KV event: {e}");
                                continue;
                            }
                        };

                        if router_socket.is_some() {
                            if ring_buffer.len() >= REPLAY_BUFFER_CAPACITY {
                                ring_buffer.pop_front();
                            }
                            ring_buffer.push_back((seq_num, payload.clone()));
                        }

                        let frames = vec![
                            Vec::new(),
                            seq_num.to_be_bytes().to_vec(),
                            payload.to_vec(),
                        ];
                        if let Err(e) = send_multipart(&pub_socket, frames).await {
                            tracing::warn!("Failed to send ZMQ KV event: {e}");
                        }

                        seq_num += 1;
                    }
                }
            }
        });

        Ok(Self { tx })
    }
}

impl RawKvEventSink for ZmqKvEventSink {
    fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
        self.tx
            .send(event)
            .map_err(|_| anyhow::anyhow!("ZMQ event sink channel closed"))
    }
}

fn convert_to_zmq_events(
    event: &KvCacheEvent,
    block_token_ids: Option<&[Vec<u32>]>,
    block_size: u32,
    storage_tier: StorageTier,
) -> Vec<ZmqRawKvEvent> {
    let medium = storage_tier.to_kv_medium();
    match &event.data {
        KvCacheEventData::Stored(store_data) => {
            let block_hashes: Vec<u64> = store_data.blocks.iter().map(|b| b.block_hash.0).collect();
            let parent_block_hash = store_data.parent_hash.map(|h| h.0);

            let token_ids: Vec<u32> = block_token_ids
                .map(|tids| tids.iter().flatten().copied().collect())
                .unwrap_or_default();

            assert_eq!(
                token_ids.len(),
                block_hashes.len() * block_size as usize,
                "token_ids length ({}) must equal block_hashes.len() ({}) * block_size ({block_size})",
                token_ids.len(),
                block_hashes.len(),
            );

            vec![ZmqRawKvEvent::BlockStored {
                block_hashes,
                parent_block_hash,
                token_ids,
                block_size,
                medium,
                group_idx: 0,
            }]
        }
        KvCacheEventData::Removed(remove_data) => {
            let block_hashes: Vec<u64> = remove_data.block_hashes.iter().map(|h| h.0).collect();
            vec![ZmqRawKvEvent::BlockRemoved {
                block_hashes,
                medium,
                group_idx: 0,
            }]
        }
        KvCacheEventData::Cleared => vec![],
    }
}

fn configure_common_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    builder
        .set_linger(ZMQ_LINGER_MS)
        .set_reconnect_ivl(ZMQ_RECONNECT_IVL_MS)
        .set_reconnect_ivl_max(ZMQ_RECONNECT_IVL_MAX_MS)
        .set_tcp_keepalive(ZMQ_TCP_KEEPALIVE)
}

fn configure_receive_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    configure_common_builder(builder).set_rcvtimeo(ZMQ_RCVTIMEOUT_MS)
}

fn configure_send_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    configure_common_builder(builder).set_sndtimeo(ZMQ_SNDTIMEOUT_MS)
}

fn configure_bidirectional_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    configure_receive_builder(builder).set_sndtimeo(ZMQ_SNDTIMEOUT_MS)
}

async fn bind_pub_socket(endpoint: &str) -> Result<SharedPubSocket> {
    let ctx = Context::new();
    let socket = configure_send_builder(publish(&ctx)).bind(endpoint)?;
    Ok(Arc::new(Mutex::new(socket)))
}

async fn bind_router_socket(endpoint: &str) -> Result<Router> {
    let ctx = Context::new();
    let socket = configure_bidirectional_builder(router(&ctx)).bind(endpoint)?;
    Ok(socket)
}

fn multipart_message(multipart: Multipart) -> MultipartMessage {
    multipart.into_iter().map(|frame| frame.to_vec()).collect()
}

async fn send_multipart<S>(socket: &Arc<Mutex<S>>, frames: MultipartMessage) -> Result<()>
where
    S: Sink<Multipart, Error = tmq::TmqError> + Unpin,
{
    socket.lock().await.send(Multipart::from(frames)).await?;
    Ok(())
}

async fn send_multipart_direct<S>(socket: &mut S, frames: MultipartMessage) -> Result<()>
where
    S: Sink<Multipart, Error = tmq::TmqError> + Unpin,
{
    socket.send(Multipart::from(frames)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData,
        LocalBlockHash,
    };

    use super::*;

    fn stored_event() -> KvCacheEvent {
        KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(10),
                    tokens_hash: LocalBlockHash(100),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }
    }

    #[test]
    fn host_pinned_zmq_stored_event_carries_medium() {
        let events = convert_to_zmq_events(
            &stored_event(),
            Some(&[vec![1, 2, 3, 4]]),
            4,
            StorageTier::HostPinned,
        );

        let [ZmqRawKvEvent::BlockStored { medium, .. }] = events.as_slice() else {
            panic!("expected one BlockStored event");
        };
        assert_eq!(*medium, Some("CPU_PINNED"));
    }

    #[test]
    fn device_zmq_stored_event_omits_medium() {
        let events = convert_to_zmq_events(
            &stored_event(),
            Some(&[vec![1, 2, 3, 4]]),
            4,
            StorageTier::Device,
        );

        let [ZmqRawKvEvent::BlockStored { medium, .. }] = events.as_slice() else {
            panic!("expected one BlockStored event");
        };
        assert_eq!(*medium, None);
    }
}
