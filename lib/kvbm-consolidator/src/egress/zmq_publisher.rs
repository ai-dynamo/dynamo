// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ PUB egress.
//!
//! Polls [`Tracker::drain_events`] at `poll_interval`, converts each
//! [`crate::tracker::ConsolidatedEvent`] via [`crate::hash::router_block_hash`] /
//! [`crate::hash::router_parent_hash`] into [`crate::wire::router_out::Event`], wraps a
//! batch in `(timestamp, events, Some(0))`, msgpack-encodes, and sends a 3-frame
//! multipart: `[b"", seq_be_bytes, payload]`.
//!
//! `seq_be_bytes` is an 8-byte big-endian [`AtomicU64`] incremented once per batch.
//!
//! Wave1-D implements this body.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, UNIX_EPOCH};

use anyhow::Result;
use serde::Serialize as _;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::hash::{router_block_hash, router_parent_hash};
use crate::tracker::{ConsolidatedEvent, Tracker};
use crate::wire::router_out::{Event, EventBatch};
use crate::zmq_util;

/// Reorder a freshly drained event stream for the kv-router wire.
///
/// The drain stream is a FIFO of `Store`, `Remove`, and `ClearAll` operations whose
/// **relative type-order is causally meaningful**: a `[Remove A, Store A]` sequence
/// must not be flipped to `[Store A, Remove A]`, or downstream loses block A even though
/// the tracker still considers it live.
///
/// The rule: split the stream into contiguous same-type runs (`Store*`, `Remove*`,
/// `ClearAll`) and sort only *within* each run. Within a `Store` run, sort ascending by
/// PLH position so parents precede children; within a `Remove` run, sort descending so
/// children release before parents. `ClearAll` is its own run and acts as a barrier.
fn sort_for_emission(events: Vec<ConsolidatedEvent>) -> Vec<ConsolidatedEvent> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Store,
        Remove,
        Clear,
    }
    fn kind(e: &ConsolidatedEvent) -> Kind {
        match e {
            ConsolidatedEvent::Store { .. } => Kind::Store,
            ConsolidatedEvent::Remove { .. } => Kind::Remove,
            ConsolidatedEvent::ClearAll => Kind::Clear,
        }
    }

    let mut out = Vec::with_capacity(events.len());
    let mut run: Vec<ConsolidatedEvent> = Vec::new();
    let mut run_kind: Option<Kind> = None;

    let flush_run =
        |out: &mut Vec<ConsolidatedEvent>, run: &mut Vec<ConsolidatedEvent>, k: Option<Kind>| {
            match k {
                Some(Kind::Store) => run.sort_by_key(|e| match e {
                    ConsolidatedEvent::Store { seq_hash, .. } => seq_hash.position(),
                    _ => unreachable!(),
                }),
                Some(Kind::Remove) => run.sort_by_key(|e| match e {
                    ConsolidatedEvent::Remove { seq_hash, .. } => {
                        std::cmp::Reverse(seq_hash.position())
                    }
                    _ => unreachable!(),
                }),
                _ => {}
            }
            out.append(run);
        };

    for ev in events {
        let k = kind(&ev);
        if run_kind != Some(k) {
            flush_run(&mut out, &mut run, run_kind);
            run_kind = Some(k);
        }
        run.push(ev);
    }
    flush_run(&mut out, &mut run, run_kind);
    out
}

fn consolidated_to_event(ev: ConsolidatedEvent) -> anyhow::Result<Event> {
    match ev {
        ConsolidatedEvent::Store {
            seq_hash,
            token_ids,
            block_size,
            lora_name,
            source: _,
        } => {
            let token_ids_i32: Vec<i32> = token_ids
                .into_iter()
                .map(|t| {
                    i32::try_from(t).unwrap_or_else(|_| {
                        tracing::warn!("Token ID {t} exceeds i32::MAX, clamping to i32::MAX");
                        i32::MAX
                    })
                })
                .collect();
            let block_size_i32 = i32::try_from(block_size).unwrap_or_else(|_| {
                tracing::warn!("Block size {block_size} exceeds i32::MAX, clamping to i32::MAX");
                i32::MAX
            });
            Ok(Event::BlockStored {
                block_hashes: vec![router_block_hash(seq_hash)],
                parent_block_hash: router_parent_hash(seq_hash),
                token_ids: token_ids_i32,
                block_size: block_size_i32,
                lora_name,
                medium: None,
            })
        }
        ConsolidatedEvent::Remove {
            seq_hash,
            source: _,
        } => Ok(Event::BlockRemoved {
            block_hashes: vec![router_block_hash(seq_hash)],
            medium: None,
        }),
        ConsolidatedEvent::ClearAll => Ok(Event::AllBlocksCleared {}),
    }
}

/// Spawn the publisher task, binding a ZMQ PUB socket at `endpoint`.
pub async fn spawn(
    endpoint: String,
    tracker: Arc<RwLock<Tracker>>,
    poll_interval: Duration,
    cancel: CancellationToken,
) -> Result<JoinHandle<()>> {
    let socket = zmq_util::bind_pub_socket(&endpoint).await?;
    let seq_counter = Arc::new(AtomicU64::new(0));

    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(poll_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                _ = interval.tick() => {
                    let drained = { tracker.write().await.drain_events() };
                    if drained.is_empty() {
                        continue;
                    }

                    let drained = sort_for_emission(drained);

                    let events: Vec<Event> = drained
                        .into_iter()
                        .filter_map(|ev| match consolidated_to_event(ev) {
                            Ok(e) => Some(e),
                            Err(e) => {
                                tracing::warn!("Failed to convert consolidated event: {e}");
                                None
                            }
                        })
                        .collect();

                    if events.is_empty() {
                        continue;
                    }

                    let now_f64 = std::time::SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();

                    let batch = EventBatch(now_f64, events, Some(0));

                    let mut buf = Vec::new();
                    if let Err(e) = batch.serialize(&mut rmp_serde::Serializer::new(&mut buf)) {
                        tracing::warn!("Failed to publish batch: {e}");
                        continue;
                    }

                    let seq = seq_counter.fetch_add(1, Ordering::Relaxed);
                    let seq_be_bytes: [u8; 8] = seq.to_be_bytes();

                    let frames = vec![vec![], seq_be_bytes.to_vec(), buf];
                    if let Err(e) = zmq_util::send_multipart(&socket, frames).await {
                        tracing::warn!("Failed to publish batch: {e}");
                    }
                }
            }
        }
    });

    Ok(handle)
}

#[cfg(test)]
mod sort_tests {
    use super::*;
    use crate::source::EventSource;
    use dynamo_tokens::PositionalLineageHash;

    fn s(pos: u64) -> ConsolidatedEvent {
        ConsolidatedEvent::Store {
            seq_hash: PositionalLineageHash::new(pos.wrapping_add(0x100), None, pos),
            token_ids: vec![],
            block_size: 0,
            lora_name: None,
            source: EventSource::Vllm,
        }
    }
    fn r(pos: u64) -> ConsolidatedEvent {
        ConsolidatedEvent::Remove {
            seq_hash: PositionalLineageHash::new(pos.wrapping_add(0x100), None, pos),
            source: EventSource::Vllm,
        }
    }
    fn kind_of(e: &ConsolidatedEvent) -> &'static str {
        match e {
            ConsolidatedEvent::Store { .. } => "S",
            ConsolidatedEvent::Remove { .. } => "R",
            ConsolidatedEvent::ClearAll => "C",
        }
    }

    /// `[Remove A, Store A]` must NOT be reordered — the second Store re-introduces a
    /// block that the Remove just retired.
    #[test]
    fn remove_then_store_preserves_lifecycle_order() {
        let out = sort_for_emission(vec![r(0), s(0)]);
        let kinds: Vec<_> = out.iter().map(kind_of).collect();
        assert_eq!(kinds, vec!["R", "S"]);
    }

    #[test]
    fn store_run_sorts_ascending_by_position() {
        let out = sort_for_emission(vec![s(5), s(1), s(3)]);
        let positions: Vec<u64> = out
            .iter()
            .map(|e| match e {
                ConsolidatedEvent::Store { seq_hash, .. } => seq_hash.position(),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(positions, vec![1, 3, 5]);
    }

    #[test]
    fn remove_run_sorts_descending_by_position() {
        let out = sort_for_emission(vec![r(1), r(5), r(3)]);
        let positions: Vec<u64> = out
            .iter()
            .map(|e| match e {
                ConsolidatedEvent::Remove { seq_hash, .. } => seq_hash.position(),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(positions, vec![5, 3, 1]);
    }

    #[test]
    fn clearall_is_a_barrier() {
        let out = sort_for_emission(vec![s(2), s(1), ConsolidatedEvent::ClearAll, s(4), s(3)]);
        let kinds: Vec<_> = out.iter().map(kind_of).collect();
        assert_eq!(kinds, vec!["S", "S", "C", "S", "S"]);
    }

    #[test]
    fn type_runs_alternate() {
        // S S R R S → sort within each run, never across.
        let out = sort_for_emission(vec![s(2), s(1), r(3), r(4), s(0)]);
        let kinds: Vec<_> = out.iter().map(kind_of).collect();
        assert_eq!(kinds, vec!["S", "S", "R", "R", "S"]);
    }
}
