// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional startup replay phase of the ordinary ZMQ listener.

use super::PublisherInput;
use super::zmq_listener::{decode_zmq_kv_batch, normalize_batch};
use crate::utils::zmq::{SubSocket, multipart_message};
use dynamo_kv_router::{protocols::*, zmq_wire::*};
use futures::{FutureExt, SinkExt, StreamExt};
use std::sync::atomic::AtomicU64;
use std::time::Duration;
use tokio::sync::mpsc;

use anyhow::{Result, bail};
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
enum ReplayFrame {
    Batch(u64, Vec<u8>),
    End,
}

fn decode_replay(frames: Vec<Vec<u8>>) -> Result<ReplayFrame> {
    // DEALER removes identity; SGLang omits the topic included by vLLM.
    let (sequence, payload) = match frames.as_slice() {
        [delimiter, sequence, payload] if delimiter.is_empty() => (sequence, payload),
        [delimiter, _, sequence, payload] if delimiter.is_empty() => (sequence, payload),
        _ => bail!("invalid KV replay envelope"),
    };
    let sequence: [u8; 8] = sequence.as_slice().try_into()?;
    if sequence == (-1_i64).to_be_bytes() && payload.is_empty() {
        return Ok(ReplayFrame::End);
    }
    if payload.is_empty() || sequence == (-1_i64).to_be_bytes() {
        bail!("invalid KV replay end marker");
    }
    Ok(ReplayFrame::Batch(
        u64::from_be_bytes(sequence),
        payload.clone(),
    ))
}

/// One cursor per engine incarnation and rank. Never infer history loss from
/// a timeout: an ordered replay suffix can confirm an observed hole.
#[derive(Debug)]
struct ReplayCursor {
    next: u64,
    pending: BTreeMap<u64, Vec<u8>>,
    bytes: usize,
    limit: usize,
}

impl ReplayCursor {
    fn new(max_bytes: usize) -> Self {
        Self {
            next: 0,
            pending: BTreeMap::new(),
            bytes: 0,
            limit: max_bytes,
        }
    }
    fn next(&self) -> u64 {
        self.next
    }
    fn insert(&mut self, sequence: u64, payload: Vec<u8>) -> Result<()> {
        if sequence < self.next || self.pending.contains_key(&sequence) {
            return Ok(());
        }
        if self.pending.len() >= 4096 || payload.len() > self.limit.saturating_sub(self.bytes) {
            bail!("KV replay/live merge buffer exhausted");
        }
        self.bytes += payload.len();
        self.pending.insert(sequence, payload);
        Ok(())
    }
    fn pop_contiguous(&mut self) -> Result<Option<(u64, Vec<u8>)>> {
        let Some(payload) = self.pending.remove(&self.next) else {
            return Ok(None);
        };
        let sequence = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("KV sequence exhausted"))?;
        self.bytes -= payload.len();
        Ok(Some((sequence, payload)))
    }
    fn gap(&self) -> Option<(u64, u64)> {
        self.pending
            .first_key_value()
            .and_then(|(&got, _)| (got > self.next).then_some((self.next, got)))
    }
}

/// One optional startup phase for a rank's existing ZMQ listener.
pub struct ZmqBootstrapConfig {
    pub endpoint: String,
    pub dp_rank: u32,
    pub timeout: std::time::Duration,
    pub completion: tokio::sync::oneshot::Sender<BootstrapOutcome>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BootstrapOutcome {
    Success,
    MissingHistory {
        dp_rank: u32,
        expected: u64,
        got: u64,
    },
    Uncertain {
        reason: String,
    },
}

#[derive(Debug, thiserror::Error)]
#[error("missing KV sequence {expected}; next available sequence is {got}")]
struct MissingHistory {
    expected: u64,
    got: u64,
}

const LIMIT: usize = 32 * 1024 * 1024;
const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(5);

#[allow(clippy::too_many_arguments)]
pub(super) async fn bootstrap(
    live: &mut SubSocket,
    mut live_monitor: tmq::pair::Pair,
    normalizer: &mut ZmqEventNormalizer,
    tx: &mpsc::UnboundedSender<PublisherInput>,
    next_event_id: &AtomicU64,
    worker_id: WorkerId,
    config: &ZmqBootstrapConfig,
) -> std::result::Result<u64, BootstrapOutcome> {
    let result = tokio::time::timeout(config.timeout, async {
        let ctx = tmq::Context::new();
        let mut replay = tmq::dealer(&ctx)
            .set_linger(0).set_ipv6(true).set_rcvhwm(8).set_maxmsgsize(LIMIT as i64)
            .monitor("inproc://replay-monitor", zmq::SocketEvent::HANDSHAKE_SUCCEEDED as i32 | zmq::SocketEvent::DISCONNECTED as i32)
            .connect(&config.endpoint)?;
        let mut replay_monitor = tmq::pair(&ctx).connect("inproc://replay-monitor")?;
        tokio::time::timeout(INACTIVITY_TIMEOUT, async {
            tokio::try_join!(connected(&mut live_monitor), connected(&mut replay_monitor))
        }).await??;
        // Empty replay still requires a functioning local index application path.
        apply(tx, Vec::new()).await?;
        let mut cursor = ReplayCursor::new(LIMIT);
        let worker = WorkerWithDpRank::new(worker_id, config.dp_rank);
        loop {
            tokio::time::timeout(INACTIVITY_TIMEOUT, replay.send(tmq::Multipart::from(vec![
                Vec::new(), cursor.next().to_be_bytes().to_vec(),
            ]))).await??;
            let mut replay_through = None;
            let mut deadline = tokio::time::Instant::now() + INACTIVITY_TIMEOUT;
            loop {
                let (sequence, payload) = tokio::select! {
                    _ = live_monitor.next() => bail!("KV live connection lost during bootstrap"),
                    _ = replay_monitor.next() => bail!("KV replay connection lost during bootstrap"),
                    _ = tokio::time::sleep_until(deadline) => bail!("KV replay timed out; history availability is unknown"),
                    message = replay.next() => {
                        let frames = multipart_message(message.ok_or_else(|| anyhow::anyhow!("KV replay socket ended"))??);
                        deadline = tokio::time::Instant::now() + INACTIVITY_TIMEOUT;
                        match decode_replay(frames)? {
                            ReplayFrame::End => break,
                            ReplayFrame::Batch(sequence, payload) => {
                                replay_through = Some(sequence);
                                (sequence, payload)
                            }
                        }
                    }
                    message = live.next() => live_payload(message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??)?,
                };
                apply_batch(&mut cursor, sequence, payload, normalizer, tx, next_event_id, worker).await?;
            }
            // Include already-queued live events in the bootstrap gap decision.
            while let Some(message) = live.next().now_or_never() {
                let (sequence, payload) = live_payload(message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??)?;
                apply_batch(&mut cursor, sequence, payload, normalizer, tx, next_event_id, worker).await?;
            }
            if let Some((expected, got)) = cursor.gap() {
                // Replay is ordered, but it is independent of the live socket.
                // Only classify a replay hole after merging queued live events.
                if replay_through.is_some_and(|sequence| sequence > expected) {
                    return Err(anyhow::Error::new(MissingHistory { expected, got }));
                }
            } else {
                return Ok(cursor.next());
            }
        }
    }).await;
    match result {
        Ok(Ok(next)) => Ok(next),
        Ok(Err(error)) => Err(match error.downcast_ref::<MissingHistory>() {
            Some(missing) => BootstrapOutcome::MissingHistory {
                dp_rank: config.dp_rank,
                expected: missing.expected,
                got: missing.got,
            },
            None => BootstrapOutcome::Uncertain {
                reason: error.to_string(),
            },
        }),
        Err(_) => Err(BootstrapOutcome::Uncertain {
            reason: "KV bootstrap deadline exceeded".into(),
        }),
    }
}

async fn connected(monitor: &mut tmq::pair::Pair) -> Result<()> {
    let event = monitor
        .next()
        .await
        .ok_or_else(|| anyhow::anyhow!("KV socket monitor ended"))??;
    let data = event
        .iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty monitor event"))?;
    let kind = u16::from_ne_bytes(
        data.get(..2)
            .ok_or_else(|| anyhow::anyhow!("short monitor event"))?
            .try_into()?,
    );
    anyhow::ensure!(
        kind == zmq::SocketEvent::HANDSHAKE_SUCCEEDED as u16,
        "KV socket disconnected during bootstrap"
    );
    Ok(())
}

fn live_payload(message: tmq::Multipart) -> Result<(u64, Vec<u8>)> {
    let mut frames = multipart_message(message);
    anyhow::ensure!(frames.len() == 3, "invalid live KV envelope");
    let payload = frames.pop().unwrap();
    let sequence = u64::from_be_bytes(
        frames
            .pop()
            .unwrap()
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid KV sequence"))?,
    );
    Ok((sequence, payload))
}

async fn apply(
    tx: &mpsc::UnboundedSender<PublisherInput>,
    events: Vec<PlacementEvent>,
) -> Result<()> {
    let (done, result) = tokio::sync::oneshot::channel();
    tx.send(PublisherInput::Recovery(events, done))
        .map_err(|_| anyhow::anyhow!("KV publisher stopped"))?;
    result
        .await
        .map_err(|_| anyhow::anyhow!("KV application acknowledgement dropped"))?
}

async fn apply_batch(
    cursor: &mut ReplayCursor,
    sequence: u64,
    payload: Vec<u8>,
    normalizer: &mut ZmqEventNormalizer,
    tx: &mpsc::UnboundedSender<PublisherInput>,
    next_event_id: &AtomicU64,
    worker: WorkerWithDpRank,
) -> Result<()> {
    cursor.insert(sequence, payload)?;
    while let Some((sequence, payload)) = cursor.pop_contiguous()? {
        // Use the existing listener decoder and normalizer for replay too.
        let decoded =
            decode_zmq_kv_batch(vec![Vec::new(), sequence.to_be_bytes().to_vec(), payload])?;
        anyhow::ensure!(
            !decoded
                .batch
                .data_parallel_rank
                .is_some_and(|rank| rank < 0 || rank as u32 != worker.dp_rank),
            "KV batch belongs to the wrong DP rank"
        );
        apply(
            tx,
            normalize_batch(decoded.batch, normalizer, worker, next_event_id),
        )
        .await?;
    }
    Ok(())
}

#[cfg(test)]
mod socket_tests {
    use super::*;
    use std::sync::Arc;
    use tmq::{AsZmqSocket, Multipart};
    use tokio::sync::oneshot;
    use tokio_util::sync::CancellationToken;

    struct Fixture {
        live: tmq::publish::Publish,
        replay: tmq::router::Router,
        inputs: mpsc::UnboundedReceiver<PublisherInput>,
        outcome: oneshot::Receiver<BootstrapOutcome>,
        cancel: CancellationToken,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            self.cancel.cancel();
        }
    }
    impl Fixture {
        fn new(timeout: Duration) -> Self {
            let ctx = tmq::Context::new();
            let live = tmq::publish(&ctx)
                .set_linger(0)
                .bind("tcp://127.0.0.1:*")
                .unwrap();
            let replay = tmq::router(&ctx)
                .set_linger(0)
                .bind("tcp://127.0.0.1:*")
                .unwrap();
            let (tx, inputs) = mpsc::unbounded_channel();
            let (completion, outcome) = oneshot::channel();
            let cancel = CancellationToken::new();
            tokio::spawn(super::super::zmq_listener::start_zmq_listener(
                live.get_socket().get_last_endpoint().unwrap().unwrap(),
                String::new(),
                1,
                tx,
                cancel.clone(),
                4,
                Arc::new(AtomicU64::new(0)),
                None,
                None,
                Some(ZmqBootstrapConfig {
                    endpoint: replay.get_socket().get_last_endpoint().unwrap().unwrap(),
                    dp_rank: 0,
                    timeout,
                    completion,
                }),
            ));
            Self {
                live,
                replay,
                inputs,
                outcome,
                cancel,
            }
        }
        async fn input(&mut self) -> PublisherInput {
            tokio::time::timeout(Duration::from_secs(5), self.inputs.recv())
                .await
                .unwrap()
                .unwrap()
        }
        async fn ack(&mut self) -> Vec<PlacementEvent> {
            let PublisherInput::Recovery(events, done) = self.input().await else {
                panic!("expected acknowledged bootstrap input")
            };
            done.send(Ok(())).unwrap();
            events
        }
        async fn request(&mut self, expected: u64) -> Vec<u8> {
            let frames = tokio::time::timeout(Duration::from_secs(5), self.replay.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let frames: Vec<_> = frames.into_iter().collect();
            assert_eq!(&frames[2][..], &expected.to_be_bytes());
            frames[0].to_vec()
        }
        async fn send(&mut self, id: &[u8], seq: i64, payload: Vec<u8>) {
            self.replay
                .send(Multipart::from(vec![
                    id.to_vec(),
                    vec![],
                    seq.to_be_bytes().to_vec(),
                    payload,
                ]))
                .await
                .unwrap();
        }
        async fn result(&mut self) -> BootstrapOutcome {
            tokio::time::timeout(Duration::from_secs(5), &mut self.outcome)
                .await
                .unwrap()
                .unwrap()
        }
    }
    fn payload(rank: i32) -> Vec<u8> {
        rmp_serde::to_vec(&(0.0_f64, vec![vec!["AllBlocksCleared"]], Some(rank))).unwrap()
    }

    #[tokio::test]
    async fn replay_waits_for_application_then_same_listener_consumes_live_without_repair() {
        let mut f = Fixture::new(Duration::from_secs(5));
        assert!(f.ack().await.is_empty());
        let id = f.request(0).await;
        f.send(&id, 0, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        let PublisherInput::Recovery(events, done) = f.input().await else {
            panic!("expected replay")
        };
        assert_eq!(events.len(), 1);
        assert!(matches!(
            f.outcome.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        done.send(Ok(())).unwrap();
        assert_eq!(f.result().await, BootstrapOutcome::Success);
        // Queued replay overlap is ignored; runtime gaps use the ordinary live path.
        for seq in [0_u64, 9] {
            f.live
                .send(Multipart::from(vec![
                    vec![],
                    seq.to_be_bytes().to_vec(),
                    payload(0),
                ]))
                .await
                .unwrap();
        }
        let PublisherInput::Events(events) = f.input().await else {
            panic!("live input must not require an ack")
        };
        assert_eq!(events[0].event.event_id, 1);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), f.replay.next())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn confirmed_missing_history_is_distinct_from_timeout() {
        for (send_batch, send_end) in [(true, true), (true, false), (false, false)] {
            let mut f = Fixture::new(Duration::from_secs(1));
            f.ack().await;
            let id = f.request(0).await;
            if send_batch {
                f.send(&id, 500, payload(0)).await;
            }
            if send_end {
                f.send(&id, -1, vec![]).await;
            }
            let result = f.result().await;
            if send_end {
                assert_eq!(
                    result,
                    BootstrapOutcome::MissingHistory {
                        dp_rank: 0,
                        expected: 0,
                        got: 500
                    }
                );
            } else {
                assert!(matches!(result, BootstrapOutcome::Uncertain { .. }));
            }
        }
    }

    #[tokio::test]
    async fn live_fills_hole_observed_first_on_replay() {
        // Exercise both missing initial history and a later replay hole.
        for missing in [0_u64, 1] {
            let mut f = Fixture::new(Duration::from_secs(5));
            f.ack().await;
            let id = f.request(0).await;
            if missing == 1 {
                f.send(&id, 0, payload(0)).await;
                f.ack().await;
            }
            f.send(&id, (missing + 1) as i64, payload(0)).await;
            // Let replay expose the hole before sending its live counterpart.
            // The old implementation reports MissingHistory here.
            assert!(
                tokio::time::timeout(Duration::from_millis(50), &mut f.outcome)
                    .await
                    .is_err()
            );
            f.live
                .send(Multipart::from(vec![
                    vec![],
                    missing.to_be_bytes().to_vec(),
                    payload(0),
                ]))
                .await
                .unwrap();
            assert_eq!(f.ack().await[0].event.event_id, missing);
            assert_eq!(f.ack().await[0].event.event_id, missing + 1);
            f.send(&id, -1, vec![]).await;
            assert_eq!(f.result().await, BootstrapOutcome::Success);
            assert!(f.inputs.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn local_application_failure_and_wrong_rank_prevent_success() {
        for wrong_rank in [true, false] {
            let mut f = Fixture::new(Duration::from_secs(5));
            f.ack().await;
            let id = f.request(0).await;
            f.send(&id, 0, payload(i32::from(wrong_rank))).await;
            f.send(&id, -1, vec![]).await;
            if !wrong_rank {
                let PublisherInput::Recovery(_, done) = f.input().await else {
                    panic!("expected replay")
                };
                done.send(Err(anyhow::anyhow!("application failed")))
                    .unwrap();
            }
            assert!(matches!(
                f.result().await,
                BootstrapOutcome::Uncertain { .. }
            ));
        }
    }

    #[tokio::test]
    async fn bootstrap_deadline_includes_application() {
        let mut f = Fixture::new(Duration::from_secs(1));
        let PublisherInput::Recovery(_, _held_ack) = f.input().await else {
            panic!("expected replay")
        };
        assert!(matches!(
            f.result().await,
            BootstrapOutcome::Uncertain { .. }
        ));
    }

    #[tokio::test]
    async fn queued_live_gap_is_replayed_before_success() {
        let mut f = Fixture::new(Duration::from_secs(5));
        f.ack().await;
        let id = f.request(0).await;
        f.send(&id, 0, payload(0)).await;
        let PublisherInput::Recovery(_, held_ack) = f.input().await else {
            panic!("expected replay")
        };
        f.live
            .send(Multipart::from(vec![
                vec![],
                2_u64.to_be_bytes().to_vec(),
                payload(0),
            ]))
            .await
            .unwrap();
        // Let transport queue the live frame while local application is blocked.
        tokio::time::sleep(Duration::from_millis(50)).await;
        f.send(&id, -1, vec![]).await;
        held_ack.send(Ok(())).unwrap();
        let id = f.request(1).await;
        f.send(&id, 1, payload(0)).await;
        f.send(&id, 2, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        assert_eq!(f.ack().await[0].event.event_id, 1);
        assert_eq!(f.ack().await[0].event.event_id, 2);
        assert_eq!(f.result().await, BootstrapOutcome::Success);
        assert!(f.inputs.try_recv().is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn end_marker_is_explicit() {
        assert_eq!(
            decode_replay(vec![vec![], (-1_i64).to_be_bytes().to_vec(), vec![]]).unwrap(),
            ReplayFrame::End
        );
        assert!(decode_replay(vec![vec![], 0_u64.to_be_bytes().to_vec(), vec![]]).is_err());
        assert!(decode_replay(vec![]).is_err());
    }
    #[test]
    fn overlap_is_applied_once_in_order() {
        let mut cursor = ReplayCursor::new(8);
        cursor.insert(2, vec![2]).unwrap();
        assert_eq!(cursor.gap(), Some((0, 2)));
        cursor.insert(0, vec![0]).unwrap();
        cursor.insert(1, vec![1]).unwrap();
        cursor.insert(2, vec![2]).unwrap();
        for seq in 0..3 {
            assert_eq!(
                cursor.pop_contiguous().unwrap(),
                Some((seq, vec![seq as u8]))
            );
        }
        cursor.insert(1, vec![1]).unwrap();
        assert_eq!(cursor.pop_contiguous().unwrap(), None);
        assert_eq!(cursor.next(), 3);
        cursor.insert(4, vec![4]).unwrap();
        assert_eq!(cursor.gap(), Some((3, 4)));
    }
    #[test]
    fn expired_history_and_buffer_bounds() {
        let mut cursor = ReplayCursor::new(2);
        cursor.insert(500, vec![1, 1]).unwrap();
        assert_eq!(cursor.gap(), Some((0, 500)));
        cursor.insert(500, vec![1, 1]).unwrap();
        assert!(cursor.insert(501, vec![2]).is_err());
    }
}
