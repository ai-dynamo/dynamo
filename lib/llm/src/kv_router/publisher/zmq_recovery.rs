// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in replay recovery for a rank's ZMQ KV event listener.

use super::PublisherInput;
use super::zmq_listener::{decode_zmq_kv_batch, normalize_batch};
use crate::utils::zmq::multipart_message;
use dynamo_kv_router::{protocols::*, zmq_wire::*};
use dynamo_runtime::transports::zmq::ipv6_option_for;
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
    fn gap(&self) -> Option<(u64, u64)> {
        self.pending
            .first_key_value()
            .and_then(|(&got, _)| (got > self.next).then_some((self.next, got)))
    }
}

/// Replay recovery for one rank; listeners without it keep the ordinary live-only policy.
pub struct ZmqRecoveryConfig {
    pub replay_endpoint: String,
    pub dp_rank: u32,
    pub timeout: Duration,
    pub status: tokio::sync::watch::Sender<KvStreamStatus>,
    pub commands: mpsc::Receiver<KvStreamCommand>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvStreamStatus {
    Recovering,
    ProbeRequired,
    Ready,
    MissingHistory { expected: u64, got: u64 },
    Uncertain { reason: String, is_terminal: bool },
}

#[derive(Debug)]
pub enum KvStreamCommand {
    /// Fence ingestion while the sidecar determines the engine incarnation.
    Suspend,
    ReconnectSameInstance(tokio::sync::oneshot::Sender<()>),
    RetryReplay(tokio::sync::oneshot::Sender<()>),
}

#[derive(Debug, thiserror::Error)]
#[error("missing KV sequence {expected}; next available sequence is {got}")]
struct MissingHistory {
    expected: u64,
    got: u64,
}

const LIMIT: usize = 32 * 1024 * 1024;
const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(5);

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
        "KV socket disconnected during recovery"
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

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct TerminalRecovery(anyhow::Error);

fn terminal(error: anyhow::Error) -> anyhow::Error {
    TerminalRecovery(error).into()
}

enum RecoveryAction {
    Suspend,
    Reconnect,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn recovering_listener(
    endpoint: &str,
    topic: &str,
    normalizer: &mut ZmqEventNormalizer,
    tx: &mpsc::UnboundedSender<PublisherInput>,
    next_event_id: &AtomicU64,
    worker_id: WorkerId,
    mut config: ZmqRecoveryConfig,
    cancel: tokio_util::sync::CancellationToken,
) {
    let mut cursor = ReplayCursor::new(LIMIT);
    let mut needs_reconnect = true;
    loop {
        if !needs_reconnect {
            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return,
                    command = config.commands.recv() => match command {
                        Some(KvStreamCommand::ReconnectSameInstance(accepted)) => {
                            config.status.send_replace(KvStreamStatus::Recovering);
                            let _ = accepted.send(());
                            break;
                        },
                        Some(_) => {},
                        None => {
                            cancel.cancelled().await;
                            return;
                        }
                    },
                }
            }
        }
        config.status.send_replace(KvStreamStatus::Recovering);
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            result = recovery_session(
                endpoint, topic, &mut config, &mut cursor, normalizer, tx, next_event_id, worker_id,
            ) => result,
        };
        match result {
            Ok(RecoveryAction::Reconnect) => needs_reconnect = true,
            Ok(RecoveryAction::Suspend) => {
                config.status.send_replace(KvStreamStatus::Recovering);
                needs_reconnect = false;
            }
            Err(error) => {
                if let Some(missing) = error.downcast_ref::<MissingHistory>() {
                    tracing::warn!(dp_rank = config.dp_rank, %error, "KV replay confirms missing history");
                    config.status.send_replace(KvStreamStatus::MissingHistory {
                        expected: missing.expected,
                        got: missing.got,
                    });
                    break;
                }
                let is_terminal = error.is::<TerminalRecovery>();
                tracing::warn!(dp_rank = config.dp_rank, %error, is_terminal, "KV recovery interrupted");
                config.status.send_replace(KvStreamStatus::Uncertain {
                    reason: error.to_string(),
                    is_terminal,
                });
                if is_terminal {
                    break;
                }
                needs_reconnect = false;
            }
        }
    }
    // Hold the terminal status until the owner stops this source rather than ending the listener.
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            Some(_) = config.commands.recv() => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn recovery_session(
    endpoint: &str,
    topic: &str,
    config: &mut ZmqRecoveryConfig,
    cursor: &mut ReplayCursor,
    normalizer: &mut ZmqEventNormalizer,
    tx: &mpsc::UnboundedSender<PublisherInput>,
    next_event_id: &AtomicU64,
    worker_id: WorkerId,
) -> Result<RecoveryAction> {
    let connect = async {
        let (live, mut live_monitor) =
            crate::utils::zmq::connect_sub_socket_with_monitor(endpoint, topic).await?;
        let ctx = tmq::Context::new();
        // Drain replies eagerly: vLLM's ROUTER drops batches once a peer's queue is full.
        let replay = tmq::dealer(&ctx)
            .set_linger(0)
            .set_ipv6(ipv6_option_for(&config.replay_endpoint)?)
            .set_rcvhwm(0)
            .set_maxmsgsize(LIMIT as i64)
            .monitor(
                "inproc://replay-monitor",
                zmq::SocketEvent::HANDSHAKE_SUCCEEDED as i32
                    | zmq::SocketEvent::DISCONNECTED as i32,
            )
            .connect(&config.replay_endpoint)?;
        let mut replay_monitor = tmq::pair(&ctx).connect("inproc://replay-monitor")?;
        tokio::try_join!(connected(&mut live_monitor), connected(&mut replay_monitor))?;
        Ok::<_, anyhow::Error>((live, live_monitor, replay, replay_monitor))
    };
    let (mut live, mut live_monitor, mut replay, mut replay_monitor) = tokio::select! {
        result = tokio::time::timeout(INACTIVITY_TIMEOUT, connect) => result??,
        command = config.commands.recv() => return command_action(command, &config.status),
    };
    let worker = WorkerWithDpRank::new(worker_id, config.dp_rank);
    let timeout = config.timeout;
    let mut witness = None;
    let mut is_ready = false;
    let mut is_replaying = false;
    let mut requested = 0;
    let mut replay_first = None;
    let mut replay_next = None;
    let mut is_reply_truncated = false;
    let mut next_replay = tokio::time::Instant::now();
    let mut replay_deadline = next_replay + INACTIVITY_TIMEOUT;
    let mut recovery_deadline = next_replay + timeout;
    loop {
        tokio::select! {
            biased;
            command = config.commands.recv() => match command {
                Some(KvStreamCommand::RetryReplay(accepted)) => {
                    is_ready = false;
                    recovery_deadline = tokio::time::Instant::now() + timeout;
                    config.status.send_replace(KvStreamStatus::Recovering);
                    let _ = accepted.send(());
                    next_replay = tokio::time::Instant::now();
                },
                command => return command_action(command, &config.status),
            },
            _ = live_monitor.next() => bail!("KV live connection lost; engine identity must be rechecked"),
            _ = replay_monitor.next() => bail!("KV replay connection lost; engine identity must be rechecked"),
            _ = tokio::time::sleep_until(recovery_deadline), if !is_ready => bail!("KV recovery deadline exceeded"),
            _ = tokio::time::sleep_until(replay_deadline), if is_replaying => bail!("KV replay timed out; history availability is unknown"),
            _ = tokio::time::sleep_until(next_replay), if !is_replaying => {
                tokio::time::timeout(INACTIVITY_TIMEOUT, replay.send(tmq::Multipart::from(vec![
                    Vec::new(), cursor.next().to_be_bytes().to_vec(),
                ]))).await??;
                is_replaying = true;
                requested = cursor.next();
                replay_first = None;
                replay_next = None;
                is_reply_truncated = false;
                replay_deadline = tokio::time::Instant::now() + INACTIVITY_TIMEOUT;
            },
            message = replay.next(), if is_replaying => {
                let frames = multipart_message(message.ok_or_else(|| anyhow::anyhow!("KV replay socket ended"))??);
                replay_deadline = tokio::time::Instant::now() + INACTIVITY_TIMEOUT;
                match decode_replay(frames).map_err(terminal)? {
                    ReplayFrame::Batch(sequence, payload) => {
                        // A hole inside one reply is a transport drop; the next round re-requests it.
                        is_reply_truncated |= replay_next.is_some_and(|next| next != sequence);
                        replay_first.get_or_insert(sequence);
                        replay_next = Some(sequence.saturating_add(1));
                        if !is_reply_truncated {
                            if sequence > cursor.next() && is_ready {
                                is_ready = false;
                                recovery_deadline = tokio::time::Instant::now() + timeout;
                                config.status.send_replace(KvStreamStatus::Recovering);
                            }
                            strict_apply(cursor, sequence, payload, normalizer, tx, next_event_id, worker, true).await.map_err(terminal)?;
                        }
                    }
                    ReplayFrame::End => {
                        is_replaying = false;
                        // Bound each drain so a busy producer cannot starve lifecycle commands.
                        for _ in 0..4096 {
                            let Some(message) = live.next().now_or_never() else { break };
                            let (sequence, payload) = live_payload(message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??).map_err(terminal)?;
                            witness = Some(sequence);
                            strict_apply(cursor, sequence, payload, normalizer, tx, next_event_id, worker, true).await.map_err(terminal)?;
                        }
                        // vLLM replays a contiguous buffer, so only a reply that starts
                        // after the requested sequence confirms lost history.
                        if let Some(got) = replay_first.filter(|first| *first > requested && cursor.next() == requested) {
                            return Err(MissingHistory { expected: requested, got }.into());
                        }
                        if cursor.gap().is_some() || is_reply_truncated {
                            if is_ready {
                                is_ready = false;
                                recovery_deadline = tokio::time::Instant::now() + timeout;
                                config.status.send_replace(KvStreamStatus::Recovering);
                            }
                            next_replay = tokio::time::Instant::now();
                        } else if witness.is_some_and(|sequence| sequence < cursor.next()) && cursor.next() > 0 {
                            if !is_ready {
                                tokio::time::timeout(INACTIVITY_TIMEOUT, apply(tx, Vec::new())).await
                                    .map_err(|error| terminal(error.into()))?.map_err(terminal)?;
                                is_ready = true;
                                config.status.send_replace(KvStreamStatus::Ready);
                            }
                            next_replay = tokio::time::Instant::now() + Duration::from_secs(1);
                        } else {
                            config.status.send_if_modified(|status| {
                                if *status == KvStreamStatus::ProbeRequired { false }
                                else { *status = KvStreamStatus::ProbeRequired; true }
                            });
                            next_replay = tokio::time::Instant::now() + Duration::from_millis(250);
                        }
                    }
                }
            },
            message = live.next() => {
                let (sequence, payload) = live_payload(message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??).map_err(terminal)?;
                witness = Some(sequence);
                if sequence > cursor.next() && is_ready {
                    is_ready = false;
                    recovery_deadline = tokio::time::Instant::now() + timeout;
                    config.status.send_replace(KvStreamStatus::Recovering);
                    next_replay = tokio::time::Instant::now();
                }
                strict_apply(cursor, sequence, payload, normalizer, tx, next_event_id, worker, !is_ready).await.map_err(terminal)?;
            },
        }
    }
}

fn command_action(
    command: Option<KvStreamCommand>,
    status: &tokio::sync::watch::Sender<KvStreamStatus>,
) -> Result<RecoveryAction> {
    match command {
        Some(KvStreamCommand::Suspend) => Ok(RecoveryAction::Suspend),
        Some(KvStreamCommand::ReconnectSameInstance(accepted)) => {
            status.send_replace(KvStreamStatus::Recovering);
            let _ = accepted.send(());
            Ok(RecoveryAction::Reconnect)
        }
        Some(KvStreamCommand::RetryReplay(accepted)) => {
            status.send_replace(KvStreamStatus::Recovering);
            let _ = accepted.send(());
            Ok(RecoveryAction::Reconnect)
        }
        None => Err(terminal(anyhow::anyhow!("KV recovery controller stopped"))),
    }
}

#[allow(clippy::too_many_arguments)]
async fn strict_apply(
    cursor: &mut ReplayCursor,
    sequence: u64,
    payload: Vec<u8>,
    normalizer: &mut ZmqEventNormalizer,
    tx: &mpsc::UnboundedSender<PublisherInput>,
    next_event_id: &AtomicU64,
    worker: WorkerWithDpRank,
    needs_ack: bool,
) -> Result<()> {
    if sequence > cursor.next() {
        let batch = decode_event_batch(&payload)?;
        anyhow::ensure!(
            !batch
                .data_parallel_rank
                .is_some_and(|rank| rank < 0 || rank as u32 != worker.dp_rank),
            "KV batch belongs to the wrong DP rank"
        );
    }
    cursor.insert(sequence, payload)?;
    while let Some(payload) = cursor.pending.remove(&cursor.next) {
        cursor.bytes -= payload.len();
        let decoded = decode_zmq_kv_batch(vec![
            Vec::new(),
            cursor.next.to_be_bytes().to_vec(),
            payload,
        ])?;
        anyhow::ensure!(
            !decoded
                .batch
                .data_parallel_rank
                .is_some_and(|rank| rank < 0 || rank as u32 != worker.dp_rank),
            "KV batch belongs to the wrong DP rank"
        );
        let events = normalize_batch(decoded.batch, normalizer, worker, next_event_id);
        if needs_ack {
            tokio::time::timeout(INACTIVITY_TIMEOUT, apply(tx, events)).await??;
        } else if !events.is_empty() {
            tx.send(PublisherInput::Events(events))
                .map_err(|_| anyhow::anyhow!("KV publisher stopped"))?;
        }
        cursor.next = cursor
            .next
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("KV sequence exhausted"))?;
    }
    Ok(())
}

#[cfg(test)]
mod socket_tests {
    use super::*;
    use std::sync::Arc;
    use tmq::{AsZmqSocket, Multipart};
    use tokio::sync::{oneshot, watch};
    use tokio_util::sync::CancellationToken;

    struct Fixture {
        live: tmq::publish::Publish,
        replay: tmq::router::Router,
        inputs: mpsc::UnboundedReceiver<PublisherInput>,
        updates: watch::Receiver<KvStreamStatus>,
        commands: mpsc::Sender<KvStreamCommand>,
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
            let (status, updates) = watch::channel(KvStreamStatus::Recovering);
            let (commands, receiver) = mpsc::channel(4);
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
                Some(ZmqRecoveryConfig {
                    replay_endpoint: replay.get_socket().get_last_endpoint().unwrap().unwrap(),
                    dp_rank: 0,
                    timeout,
                    status,
                    commands: receiver,
                }),
            ));
            Self {
                live,
                replay,
                inputs,
                updates,
                commands,
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
                panic!("expected acknowledged recovery input")
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
        async fn live(&mut self, sequence: u64) {
            self.live
                .send(Multipart::from(vec![
                    vec![],
                    sequence.to_be_bytes().to_vec(),
                    payload(0),
                ]))
                .await
                .unwrap();
        }
        async fn status(&mut self, expected: KvStreamStatus) {
            tokio::time::timeout(
                Duration::from_secs(5),
                self.updates.wait_for(|value| *value == expected),
            )
            .await
            .unwrap()
            .unwrap();
        }
        async fn ready(&mut self) {
            let id = self.request(0).await;
            self.live_ack(0).await;
            self.send(&id, 0, payload(0)).await;
            self.send(&id, -1, vec![]).await;
            self.ack().await;
            self.status(KvStreamStatus::Ready).await;
        }
        async fn live_ack(&mut self, sequence: u64) -> Vec<PlacementEvent> {
            tokio::time::timeout(Duration::from_secs(5), async {
                let mut retry = tokio::time::interval(Duration::from_millis(20));
                loop {
                    tokio::select! {
                        input = self.inputs.recv() => {
                            let Some(PublisherInput::Recovery(events, done)) = input else {
                                panic!("expected acknowledged live history")
                            };
                            done.send(Ok(())).unwrap();
                            break events;
                        }
                        _ = retry.tick() => self.live(sequence).await,
                    }
                }
            })
            .await
            .unwrap()
        }
    }
    fn payload(rank: i32) -> Vec<u8> {
        rmp_serde::to_vec(&(0.0_f64, vec![vec!["AllBlocksCleared"]], Some(rank))).unwrap()
    }

    #[tokio::test]
    async fn empty_replay_and_probe_completion_require_live_history() {
        for is_warm in [false, true] {
            let mut f = Fixture::new(Duration::from_secs(5));
            let id = f.request(0).await;
            if is_warm {
                f.send(&id, 0, payload(0)).await;
                f.ack().await;
            }
            f.send(&id, -1, vec![]).await;
            f.status(KvStreamStatus::ProbeRequired).await;
            let (accepted, acknowledgement) = oneshot::channel();
            f.commands
                .send(KvStreamCommand::RetryReplay(accepted))
                .await
                .unwrap();
            acknowledgement.await.unwrap();
            let first_live = u64::from(is_warm);
            let id = f.request(first_live).await;
            f.send(&id, -1, vec![]).await;
            f.status(KvStreamStatus::ProbeRequired).await;
            // Actual history, rather than the completed inference, opens readiness.
            assert_eq!(f.live_ack(first_live).await[0].event.event_id, first_live);
            let id = f.request(first_live + 1).await;
            f.send(&id, -1, vec![]).await;
            f.ack().await;
            f.status(KvStreamStatus::Ready).await;
        }
    }

    #[tokio::test]
    async fn runtime_repairs_gaps_and_idle_tail_without_reapplying_history() {
        let mut f = Fixture::new(Duration::from_secs(5));
        f.ready().await;
        f.live(2).await;
        f.status(KvStreamStatus::Recovering).await;
        let id = f.request(1).await;
        f.send(&id, 1, payload(0)).await;
        f.send(&id, 2, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        assert_eq!(f.ack().await[0].event.event_id, 1);
        assert_eq!(f.ack().await[0].event.event_id, 2);
        f.ack().await;
        f.status(KvStreamStatus::Ready).await;
        // No subsequent live batch is needed to discover a lost final batch.
        let id = f.request(3).await;
        f.send(&id, 3, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        assert_eq!(f.ack().await[0].event.event_id, 3);
        assert_eq!(*f.updates.borrow(), KvStreamStatus::Ready);
    }

    #[tokio::test]
    async fn same_instance_reconnect_retains_cursor_and_requires_new_live_witness() {
        let mut f = Fixture::new(Duration::from_secs(5));
        f.ready().await;
        f.commands.send(KvStreamCommand::Suspend).await.unwrap();
        f.status(KvStreamStatus::Recovering).await;
        let (accepted, acknowledgement) = oneshot::channel();
        f.commands
            .send(KvStreamCommand::ReconnectSameInstance(accepted))
            .await
            .unwrap();
        acknowledgement.await.unwrap();
        assert_eq!(*f.updates.borrow(), KvStreamStatus::Recovering);
        let id = f.request(1).await;
        f.send(&id, -1, vec![]).await;
        f.status(KvStreamStatus::ProbeRequired).await;
        assert_eq!(f.live_ack(1).await[0].event.event_id, 1);
        let id = f.request(2).await;
        f.send(&id, -1, vec![]).await;
        f.ack().await;
        f.status(KvStreamStatus::Ready).await;
    }

    #[tokio::test]
    async fn application_failure_does_not_advance_cursor() {
        let mut cursor = ReplayCursor::new(LIMIT);
        let mut normalizer = ZmqEventNormalizer::new(4);
        let (tx, mut inputs) = mpsc::unbounded_channel();
        let next_id = AtomicU64::new(0);
        let application = strict_apply(
            &mut cursor,
            0,
            payload(0),
            &mut normalizer,
            &tx,
            &next_id,
            WorkerWithDpRank::new(1, 0),
            true,
        );
        let (result, ()) = tokio::join!(application, async {
            let Some(PublisherInput::Recovery(_, done)) = inputs.recv().await else {
                panic!("expected recovery")
            };
            done.send(Err(anyhow::anyhow!("index application failed")))
                .unwrap();
        });
        assert!(result.is_err());
        assert_eq!(cursor.next(), 0);
    }

    #[tokio::test]
    async fn invalid_batch_and_application_error_are_terminal_uncertainty() {
        for is_wrong_rank in [false, true] {
            let mut f = Fixture::new(Duration::from_secs(5));
            let id = f.request(0).await;
            f.send(
                &id,
                if is_wrong_rank { 500 } else { 0 },
                payload(i32::from(is_wrong_rank)),
            )
            .await;
            f.send(&id, -1, vec![]).await;
            if !is_wrong_rank {
                let PublisherInput::Recovery(_, done) = f.input().await else {
                    panic!("expected recovery application")
                };
                done.send(Err(anyhow::anyhow!("application failed")))
                    .unwrap();
            }
            tokio::time::timeout(
                Duration::from_secs(5),
                f.updates.wait_for(|status| {
                    matches!(
                        status,
                        KvStreamStatus::Uncertain {
                            is_terminal: true,
                            ..
                        }
                    )
                }),
            )
            .await
            .unwrap()
            .unwrap();
        }
    }

    #[tokio::test]
    async fn confirmed_missing_history_is_distinct_from_timeout() {
        for (is_batch_sent, is_end_sent) in [(true, true), (true, false), (false, false)] {
            let mut f = Fixture::new(Duration::from_secs(1));
            let id = f.request(0).await;
            if is_batch_sent {
                f.send(&id, 500, payload(0)).await;
            }
            if is_end_sent {
                f.send(&id, -1, vec![]).await;
            }
            let status = tokio::time::timeout(
                Duration::from_secs(5),
                f.updates
                    .wait_for(|status| !matches!(status, KvStreamStatus::Recovering)),
            )
            .await
            .unwrap()
            .unwrap()
            .clone();
            if is_end_sent {
                assert_eq!(
                    status,
                    KvStreamStatus::MissingHistory {
                        expected: 0,
                        got: 500
                    }
                );
            } else {
                assert!(matches!(
                    status,
                    KvStreamStatus::Uncertain {
                        is_terminal: false,
                        ..
                    }
                ));
            }
        }
    }

    #[tokio::test]
    async fn live_fills_hole_observed_first_on_replay() {
        let mut f = Fixture::new(Duration::from_secs(5));
        let id = f.request(0).await;
        f.send(&id, 1, payload(0)).await;
        // Let replay expose the hole before its live counterpart arrives.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(f.live_ack(0).await[0].event.event_id, 0);
        assert_eq!(f.ack().await[0].event.event_id, 1);
        f.send(&id, -1, vec![]).await;
        f.ack().await;
        f.status(KvStreamStatus::Ready).await;
    }

    #[tokio::test]
    async fn hole_inside_one_replay_reply_is_requested_again() {
        let mut f = Fixture::new(Duration::from_secs(5));
        let id = f.request(0).await;
        f.send(&id, 0, payload(0)).await;
        assert_eq!(f.ack().await[0].event.event_id, 0);
        f.send(&id, 2, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        let id = f.request(1).await;
        assert_eq!(*f.updates.borrow(), KvStreamStatus::Recovering);
        f.send(&id, 1, payload(0)).await;
        f.send(&id, 2, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        assert_eq!(f.ack().await[0].event.event_id, 1);
        assert_eq!(f.ack().await[0].event.event_id, 2);
        f.status(KvStreamStatus::ProbeRequired).await;
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
    fn expired_history_and_buffer_bounds() {
        let mut cursor = ReplayCursor::new(2);
        cursor.insert(500, vec![1, 1]).unwrap();
        assert_eq!(cursor.gap(), Some((0, 500)));
        cursor.insert(500, vec![1, 1]).unwrap();
        assert!(cursor.insert(501, vec![2]).is_err());
    }
}
