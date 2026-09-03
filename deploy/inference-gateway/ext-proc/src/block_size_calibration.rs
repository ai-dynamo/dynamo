// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-calibration of the KV event block size from live engine events.
//!
//! vLLM aligns the attention block size at engine boot for hybrid Mamba models
//! ("attention page size >= mamba page size"), so the true KV event block size
//! is only knowable from the running engine. The full dynamo vLLM worker reads
//! it via `get_kv_cache_group_metadata`; the standalone EPP has no engine
//! handle, so a misconfigured `DYN_KV_CACHE_BLOCK_SIZE` silently drops every
//! event and degrades routing to load-only (the `Block not published` warning
//! in `zmq_wire::convert`).
//!
//! In auto mode (`DYN_KV_CACHE_BLOCK_SIZE` unset or `0`), the calibrator opens
//! one ZMQ SUB socket, connects it to every discovered worker's KV event
//! endpoint, and adopts the block size of the first main-attention
//! `BlockStored` event. Workers are registered with a provisional block size
//! until then (scheduling works, KV overlap scoring stays off), and
//! re-registered with the observed size by the next topology reconcile.

use std::collections::HashSet;
use std::path::Path;
use std::sync::mpsc;

use dynamo_kv_router::zmq_wire::{KvCacheSpecKind, RawKvEvent, decode_event_batch};
use tokio::sync::watch;

/// Reads the block size persisted by a previous calibration, if any.
/// The selection service pins the block size per model for the process
/// lifetime, so a calibration lands by persisting the value and restarting;
/// this is the read side used at boot.
pub fn read_persisted_block_size(path: &str) -> Option<u32> {
    let content = std::fs::read_to_string(Path::new(path)).ok()?;
    content.trim().parse::<u32>().ok().filter(|v| *v > 0)
}

/// Persists the calibrated block size for the next boot to pick up.
pub fn persist_block_size(path: &str, block_size: u32) -> std::io::Result<()> {
    std::fs::write(Path::new(path), format!("{block_size}\n"))
}

/// Block size workers are registered with while auto-calibration is pending.
/// Matches vLLM's default cache block size; mismatched events are dropped by
/// the indexer until the observed size replaces it, which is the pre-existing
/// load-only behavior.
pub const PROVISIONAL_BLOCK_SIZE: u32 = 16;

const POLL_TIMEOUT_MS: i64 = 1_000;

/// Handle owned by the topology adapter: feeds worker endpoints to the probe
/// thread and exposes the observed block size as a watch channel.
pub struct BlockSizeCalibration {
    endpoints_tx: mpsc::Sender<String>,
    observed_rx: watch::Receiver<u32>,
}

impl BlockSizeCalibration {
    /// Start the probe thread. It runs until the first main-attention
    /// `BlockStored` event is observed, then publishes the size and exits.
    pub fn start() -> Self {
        let (endpoints_tx, endpoints_rx) = mpsc::channel::<String>();
        let (observed_tx, observed_rx) = watch::channel(0u32);
        std::thread::Builder::new()
            .name("kv-block-size-probe".into())
            .spawn(move || probe_loop(&endpoints_rx, &observed_tx))
            .expect("spawning the calibration probe thread");
        Self {
            endpoints_tx,
            observed_rx,
        }
    }

    /// Observed block size; `None` until the first event calibrates it.
    pub fn observed(&self) -> Option<u32> {
        match *self.observed_rx.borrow() {
            0 => None,
            v => Some(v),
        }
    }

    /// Watch channel so the topology reconcile can wake up on calibration.
    pub fn subscribe(&self) -> watch::Receiver<u32> {
        self.observed_rx.clone()
    }

    /// Connect the probe to a worker's KV event endpoint. Duplicates are fine,
    /// the probe tracks what it already connected. No-op once calibrated.
    pub fn add_endpoint(&self, endpoint: &str) {
        let _ = self.endpoints_tx.send(endpoint.to_string());
    }
}

fn probe_loop(endpoints_rx: &mpsc::Receiver<String>, observed_tx: &watch::Sender<u32>) {
    let ctx = zmq::Context::new();
    let socket = match ctx.socket(zmq::SUB) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "Calibration probe failed to create its SUB socket");
            return;
        }
    };
    if let Err(e) = socket.set_subscribe(b"") {
        tracing::error!(error = %e, "Calibration probe failed to subscribe");
        return;
    }

    let mut connected: HashSet<String> = HashSet::new();
    loop {
        // Drain newly discovered endpoints; stop once the owning adapter is
        // gone and no stream can ever deliver an event.
        loop {
            match endpoints_rx.try_recv() {
                Ok(ep) => {
                    if connected.insert(ep.clone())
                        && let Err(e) = socket.connect(&ep)
                    {
                        tracing::warn!(endpoint = %ep, error = %e, "Calibration probe connect failed");
                        connected.remove(&ep);
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if connected.is_empty() {
                        return;
                    }
                    break;
                }
            }
        }

        match socket.poll(zmq::POLLIN, POLL_TIMEOUT_MS) {
            Ok(0) => continue,
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "Calibration probe poll failed");
                continue;
            }
        }
        let Ok(frames) = socket.recv_multipart(0) else {
            continue;
        };
        // Live event frames are [topic, seq, payload].
        if frames.len() != 3 {
            continue;
        }
        if let Some(size) = main_attention_block_size(&frames[2]) {
            tracing::info!(
                block_size = size,
                "Calibrated KV event block size from live engine events"
            );
            let _ = observed_tx.send(size);
            return;
        }
    }
}

/// Block size of the first main-attention `BlockStored` event in the batch.
fn main_attention_block_size(payload: &[u8]) -> Option<u32> {
    let batch = decode_event_batch(payload).ok()?;
    batch.events.iter().find_map(|event| match event {
        RawKvEvent::BlockStored {
            block_size,
            kv_cache_spec_kind,
            ..
        } if is_main_attention(kv_cache_spec_kind.as_ref()) => u32::try_from(*block_size).ok(),
        _ => None,
    })
}

/// Mirrors `MAIN_ATTENTION_KV_CACHE_KINDS` in the dynamo vLLM worker's
/// `cache_info`. Untagged events come from single-group engines where every
/// event carries the one true block size.
fn is_main_attention(kind: Option<&KvCacheSpecKind>) -> bool {
    matches!(
        kind,
        None | Some(
            KvCacheSpecKind::FullAttention
                | KvCacheSpecKind::MlaAttention
                | KvCacheSpecKind::SinkFullAttention
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_event_payload(block_size: usize, kind: Option<&str>) -> Vec<u8> {
        let mut event = serde_json::json!({
            "type": "BlockStored",
            "block_hashes": [1u64],
            "parent_block_hash": null,
            "token_ids": (0..block_size as u32).collect::<Vec<u32>>(),
            "block_size": block_size,
        });
        if let Some(kind) = kind {
            event["kv_cache_spec_kind"] = serde_json::json!(kind);
        }
        let batch = serde_json::json!([0.0, [event], null]);
        rmp_serde::to_vec_named(&batch).expect("serializing test event batch")
    }

    #[test]
    fn adopts_main_attention_block_size() {
        let payload = stored_event_payload(2096, Some("full_attention"));
        assert_eq!(main_attention_block_size(&payload), Some(2096));
    }

    #[test]
    fn adopts_untagged_single_group_events() {
        let payload = stored_event_payload(16, None);
        assert_eq!(main_attention_block_size(&payload), Some(16));
    }

    #[test]
    fn ignores_mamba_group_events() {
        let payload = stored_event_payload(2096, Some("mamba"));
        assert_eq!(main_attention_block_size(&payload), None);
    }

    #[test]
    fn calibration_publishes_through_the_watch() {
        let calibration = BlockSizeCalibration::start();
        assert_eq!(calibration.observed(), None);
    }
}
