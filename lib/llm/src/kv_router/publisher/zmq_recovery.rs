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
use tokio::time::Instant;

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

/// Per-rank sequence cursor; history loss is inferred only from replay, never from a timeout.
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
    fn clear_pending(&mut self) {
        self.pending.clear();
        self.bytes = 0;
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
    /// Deadline for each recovery round; retries within a round never extend it.
    pub timeout: Duration,
    pub status: tokio::sync::watch::Sender<KvStreamStatus>,
    pub commands: mpsc::Receiver<KvStreamCommand>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvStreamStatus {
    Recovering,
    Success,
    MissingHistory { expected: u64, got: u64 },
    Failed { reason: String },
}

#[derive(Debug)]
pub enum KvStreamCommand {
    /// Pause recovery while the sidecar checks the engine identity; the index and cursor are kept.
    Suspend,
    /// The engine is unchanged: start a new recovery round from the cursor.
    Resume(tokio::sync::oneshot::Sender<()>),
}

#[derive(Debug, thiserror::Error)]
#[error("missing KV sequence {expected}; next available sequence is {got}")]
struct MissingHistory {
    expected: u64,
    got: u64,
}

const LIMIT: usize = 32 * 1024 * 1024;
const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(5);
const RETRY_BACKOFF: Duration = Duration::from_millis(500);
/// Delay before the replay that covers the live subscription becoming active.
const CONFIRMATION_DELAY: Duration = Duration::from_secs(1);

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

/// Like ordinary live ingestion, an unreadable message is skipped; a later gap repairs it.
fn live_batch(message: tmq::Multipart) -> Option<(u64, Vec<u8>)> {
    let decoded = <[Vec<u8>; 3]>::try_from(multipart_message(message))
        .ok()
        .and_then(|[_, sequence, payload]| {
            Some((u64::from_be_bytes(sequence.try_into().ok()?), payload))
        });
    if decoded.is_none() {
        tracing::warn!("Failed to decode ZMQ KV batch envelope");
    }
    decoded
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

fn set_status(status: &tokio::sync::watch::Sender<KvStreamStatus>, value: KvStreamStatus) {
    status.send_if_modified(|current| {
        let is_changed = *current != value;
        *current = value;
        is_changed
    });
}

enum SessionEnd {
    Suspended,
    Resumed,
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
    // `None` while recovery has succeeded; otherwise the current round's deadline.
    let mut deadline = Some(Instant::now() + config.timeout);
    let outcome = loop {
        set_status(&config.status, KvStreamStatus::Recovering);
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            result = recovery_session(
                endpoint, topic, &mut config, &mut cursor, &mut deadline, normalizer, tx, next_event_id, worker_id,
            ) => result,
        };
        let error = match result {
            Ok(SessionEnd::Resumed) => {
                deadline = Some(Instant::now() + config.timeout);
                continue;
            }
            Ok(SessionEnd::Suspended) => {
                set_status(&config.status, KvStreamStatus::Recovering);
                if !wait_for_resume(&mut config, &cancel).await {
                    return;
                }
                deadline = Some(Instant::now() + config.timeout);
                continue;
            }
            Err(error) => error,
        };
        if let Some(missing) = error.downcast_ref::<MissingHistory>() {
            tracing::warn!(dp_rank = config.dp_rank, %error, "KV replay confirms missing history");
            break KvStreamStatus::MissingHistory {
                expected: missing.expected,
                got: missing.got,
            };
        }
        if error.is::<TerminalRecovery>() {
            tracing::warn!(dp_rank = config.dp_rank, %error, "KV recovery failed");
            break KvStreamStatus::Failed {
                reason: error.to_string(),
            };
        }
        // A transient failure after success starts a new round; within a round it keeps the deadline.
        let round_deadline = *deadline.get_or_insert_with(|| Instant::now() + config.timeout);
        set_status(&config.status, KvStreamStatus::Recovering);
        if Instant::now() >= round_deadline {
            tracing::warn!(dp_rank = config.dp_rank, %error, "KV recovery deadline exceeded");
            break KvStreamStatus::Failed {
                reason: format!("KV recovery deadline exceeded: {error}"),
            };
        }
        tracing::warn!(dp_rank = config.dp_rank, %error, "KV recovery interrupted; retrying");
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            command = config.commands.recv() => match command {
                Some(KvStreamCommand::Suspend) => {
                    if !wait_for_resume(&mut config, &cancel).await {
                        return;
                    }
                    deadline = Some(Instant::now() + config.timeout);
                }
                Some(KvStreamCommand::Resume(accepted)) => {
                    let _ = accepted.send(());
                    deadline = Some(Instant::now() + config.timeout);
                }
                None => break KvStreamStatus::Failed {
                    reason: "KV recovery controller stopped".to_string(),
                },
            },
            _ = tokio::time::sleep_until(round_deadline.min(Instant::now() + RETRY_BACKOFF)) => {}
        }
    };
    set_status(&config.status, outcome);
    // Hold the final status until the owner stops this source rather than ending the listener.
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            Some(_) = config.commands.recv() => {}
        }
    }
}

/// Waits for `Resume` while suspended; returns false once cancelled.
async fn wait_for_resume(
    config: &mut ZmqRecoveryConfig,
    cancel: &tokio_util::sync::CancellationToken,
) -> bool {
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return false,
            command = config.commands.recv() => match command {
                Some(KvStreamCommand::Resume(accepted)) => {
                    let _ = accepted.send(());
                    return true;
                }
                Some(KvStreamCommand::Suspend) => {}
                None => {
                    cancel.cancelled().await;
                    return false;
                }
            },
        }
    }
}

fn command_end(command: Option<KvStreamCommand>) -> Result<SessionEnd> {
    match command {
        Some(KvStreamCommand::Suspend) => Ok(SessionEnd::Suspended),
        Some(KvStreamCommand::Resume(accepted)) => {
            let _ = accepted.send(());
            Ok(SessionEnd::Resumed)
        }
        None => Err(terminal(anyhow::anyhow!("KV recovery controller stopped"))),
    }
}

fn revoke(
    status: &tokio::sync::watch::Sender<KvStreamStatus>,
    deadline: &mut Option<Instant>,
    timeout: Duration,
) {
    *deadline = Some(Instant::now() + timeout);
    set_status(status, KvStreamStatus::Recovering);
}

#[allow(clippy::too_many_arguments)]
async fn recovery_session(
    endpoint: &str,
    topic: &str,
    config: &mut ZmqRecoveryConfig,
    cursor: &mut ReplayCursor,
    deadline: &mut Option<Instant>,
    normalizer: &mut ZmqEventNormalizer,
    tx: &mpsc::UnboundedSender<PublisherInput>,
    next_event_id: &AtomicU64,
    worker_id: WorkerId,
) -> Result<SessionEnd> {
    // Replay from the cursor refills anything buffered by an earlier connection.
    cursor.clear_pending();
    let connect = async {
        let (live, mut live_monitor) =
            crate::utils::zmq::connect_sub_socket_with_monitor(endpoint, topic).await?;
        let ctx = tmq::Context::new();
        // Attach the monitor reader before connecting so the handshake event cannot be missed.
        let mut replay_monitor = tmq::pair(&ctx).connect("inproc://replay-monitor")?;
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
        tokio::try_join!(connected(&mut live_monitor), connected(&mut replay_monitor))?;
        Ok::<_, anyhow::Error>((live, live_monitor, replay, replay_monitor))
    };
    let (mut live, mut live_monitor, mut replay, mut replay_monitor) = tokio::select! {
        biased;
        command = config.commands.recv() => return command_end(command),
        _ = tokio::time::sleep_until(deadline.unwrap_or_else(Instant::now)), if deadline.is_some() => bail!("KV recovery deadline exceeded while connecting"),
        result = tokio::time::timeout(INACTIVITY_TIMEOUT, connect) => result??,
    };
    let worker = WorkerWithDpRank::new(worker_id, config.dp_rank);
    let timeout = config.timeout;
    let mut is_success = false;
    let mut needs_confirmation = true;
    let mut is_replaying = false;
    let mut requested = 0;
    let mut replay_first = None;
    let mut replay_next = None;
    let mut is_reply_truncated = false;
    let mut next_replay = Some(Instant::now());
    let mut replay_deadline = Instant::now() + INACTIVITY_TIMEOUT;
    loop {
        tokio::select! {
            biased;
            command = config.commands.recv() => return command_end(command),
            _ = live_monitor.next() => bail!("KV live connection lost"),
            _ = replay_monitor.next() => bail!("KV replay connection lost"),
            _ = tokio::time::sleep_until(deadline.unwrap_or_else(Instant::now)), if deadline.is_some() => bail!("KV recovery deadline exceeded"),
            _ = tokio::time::sleep_until(replay_deadline), if is_replaying => bail!("KV replay timed out"),
            _ = tokio::time::sleep_until(next_replay.unwrap_or_else(Instant::now)), if !is_replaying && next_replay.is_some() => {
                tokio::time::timeout(INACTIVITY_TIMEOUT, replay.send(tmq::Multipart::from(vec![
                    Vec::new(), cursor.next().to_be_bytes().to_vec(),
                ]))).await??;
                is_replaying = true;
                next_replay = None;
                requested = cursor.next();
                replay_first = None;
                replay_next = None;
                is_reply_truncated = false;
                replay_deadline = Instant::now() + INACTIVITY_TIMEOUT;
            },
            message = replay.next(), if is_replaying => {
                let frames = multipart_message(message.ok_or_else(|| anyhow::anyhow!("KV replay socket ended"))??);
                replay_deadline = Instant::now() + INACTIVITY_TIMEOUT;
                match decode_replay(frames)? {
                    ReplayFrame::Batch(sequence, payload) => {
                        // A hole inside one reply is a transport drop; the next round re-requests it.
                        is_reply_truncated |= replay_next.is_some_and(|next| next != sequence);
                        replay_first.get_or_insert(sequence);
                        replay_next = Some(sequence.saturating_add(1));
                        if !is_reply_truncated {
                            if sequence > cursor.next() && is_success {
                                is_success = false;
                                revoke(&config.status, deadline, timeout);
                            }
                            apply_in_order(cursor, sequence, payload, normalizer, tx, next_event_id, worker, true).await?;
                        }
                    }
                    ReplayFrame::End => {
                        is_replaying = false;
                        // Bound each drain so a busy producer cannot starve lifecycle commands.
                        for _ in 0..4096 {
                            let Some(message) = live.next().now_or_never() else { break };
                            let message = message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??;
                            if let Some((sequence, payload)) = live_batch(message) {
                                apply_in_order(cursor, sequence, payload, normalizer, tx, next_event_id, worker, !is_success).await?;
                            }
                        }
                        // Replay is contiguous, so only a late first batch proves loss.
                        if let Some(got) = replay_first.filter(|first| *first > requested && cursor.next() == requested) {
                            return Err(MissingHistory { expected: requested, got }.into());
                        }
                        if cursor.gap().is_some() || is_reply_truncated {
                            if is_success {
                                is_success = false;
                                revoke(&config.status, deadline, timeout);
                            }
                            next_replay = Some(Instant::now());
                        } else {
                            if !is_success {
                                // Report success only after the index has applied everything.
                                tokio::time::timeout(INACTIVITY_TIMEOUT, apply(tx, Vec::new())).await
                                    .map_err(|error| terminal(error.into()))?.map_err(terminal)?;
                                is_success = true;
                                *deadline = None;
                                set_status(&config.status, KvStreamStatus::Success);
                            }
                            if needs_confirmation {
                                needs_confirmation = false;
                                next_replay = Some(Instant::now() + CONFIRMATION_DELAY);
                            }
                        }
                    }
                }
            },
            message = live.next() => {
                let message = message.ok_or_else(|| anyhow::anyhow!("KV live socket ended"))??;
                let Some((sequence, payload)) = live_batch(message) else { continue };
                if sequence > cursor.next() && is_success {
                    is_success = false;
                    revoke(&config.status, deadline, timeout);
                    next_replay = Some(Instant::now());
                }
                apply_in_order(cursor, sequence, payload, normalizer, tx, next_event_id, worker, !is_success).await?;
            },
        }
    }
}

fn check_rank(rank: Option<i32>, worker: WorkerWithDpRank) -> Result<()> {
    if rank.is_some_and(|rank| rank < 0 || rank as u32 != worker.dp_rank) {
        return Err(terminal(anyhow::anyhow!(
            "KV batch belongs to the wrong DP rank"
        )));
    }
    Ok(())
}

/// Applies batches in sequence order, handling bad events as ordinary live ingestion does.
#[allow(clippy::too_many_arguments)]
async fn apply_in_order(
    cursor: &mut ReplayCursor,
    sequence: u64,
    payload: Vec<u8>,
    normalizer: &mut ZmqEventNormalizer,
    tx: &mpsc::UnboundedSender<PublisherInput>,
    next_event_id: &AtomicU64,
    worker: WorkerWithDpRank,
    needs_ack: bool,
) -> Result<()> {
    if sequence > cursor.next()
        && let Ok(batch) = decode_event_batch(&payload)
    {
        check_rank(batch.data_parallel_rank, worker)?;
    }
    cursor.insert(sequence, payload)?;
    while let Some(payload) = cursor.pending.remove(&cursor.next) {
        cursor.bytes -= payload.len();
        let sequence = cursor.next;
        match decode_zmq_kv_batch(vec![Vec::new(), sequence.to_be_bytes().to_vec(), payload]) {
            Err(error) => tracing::warn!(%error, sequence, "Failed to decode ZMQ KV batch"),
            Ok(decoded) => {
                check_rank(decoded.batch.data_parallel_rank, worker)?;
                let events = normalize_batch(decoded.batch, normalizer, worker, next_event_id);
                if needs_ack {
                    tokio::time::timeout(INACTIVITY_TIMEOUT, apply(tx, events))
                        .await
                        .map_err(|_| {
                            terminal(anyhow::anyhow!("KV index did not confirm batch {sequence}"))
                        })?
                        .map_err(terminal)?;
                } else if !events.is_empty() {
                    tx.send(PublisherInput::Events(events))
                        .map_err(|_| terminal(anyhow::anyhow!("KV publisher stopped")))?;
                }
            }
        }
        cursor.next = sequence
            .checked_add(1)
            .ok_or_else(|| terminal(anyhow::anyhow!("KV sequence exhausted")))?;
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
        listener: tokio::task::JoinHandle<()>,
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
            let listener = tokio::spawn(super::super::zmq_listener::start_zmq_listener(
                live.get_socket().get_last_endpoint().unwrap().unwrap(),
                String::new(),
                1,
                tx,
                cancel.clone(),
                4,
                Arc::new(AtomicU64::new(0)),
                None,
                None,
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
                listener,
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
            // vLLM replay frames: delimiter, topic (empty on the end marker), sequence, payload.
            let topic = if seq < 0 {
                vec![]
            } else {
                b"kv-events".to_vec()
            };
            self.replay
                .send(Multipart::from(vec![
                    id.to_vec(),
                    vec![],
                    topic,
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
        async fn final_status(&mut self) -> KvStreamStatus {
            tokio::time::timeout(
                Duration::from_secs(5),
                self.updates
                    .wait_for(|status| !matches!(status, KvStreamStatus::Recovering)),
            )
            .await
            .unwrap()
            .unwrap()
            .clone()
        }
        /// Replays batch 0 and reaches `Success`; the cursor is then at 1.
        async fn succeed(&mut self) {
            let id = self.request(0).await;
            self.send(&id, 0, payload(0)).await;
            self.ack().await;
            self.send(&id, -1, vec![]).await;
            self.ack().await;
            self.status(KvStreamStatus::Success).await;
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
        /// Closes the replay socket so the listener sees a disconnect, then binds the same port again.
        async fn restart_replay(&mut self) {
            let endpoint = self
                .replay
                .get_socket()
                .get_last_endpoint()
                .unwrap()
                .unwrap();
            let ctx = tmq::Context::new();
            let placeholder = tmq::router(&ctx)
                .set_linger(0)
                .bind("tcp://127.0.0.1:*")
                .unwrap();
            drop(std::mem::replace(&mut self.replay, placeholder));
            self.replay = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if let Ok(socket) = tmq::router(&ctx).set_linger(0).bind(&endpoint) {
                        break socket;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .unwrap();
        }
    }
    fn payload(rank: i32) -> Vec<u8> {
        rmp_serde::to_vec(&(0.0_f64, vec![vec!["AllBlocksCleared"]], Some(rank))).unwrap()
    }

    #[tokio::test]
    async fn complete_replay_succeeds_without_live_events_and_is_confirmed() {
        for is_empty in [true, false] {
            let mut f = Fixture::new(Duration::from_secs(5));
            let id = f.request(0).await;
            if !is_empty {
                f.send(&id, 0, payload(0)).await;
                f.ack().await;
            }
            f.send(&id, -1, vec![]).await;
            f.ack().await;
            f.status(KvStreamStatus::Success).await;
            // One confirmation replay covers the live subscription becoming active.
            let id = f.request(u64::from(!is_empty)).await;
            f.send(&id, -1, vec![]).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert_eq!(*f.updates.borrow(), KvStreamStatus::Success);
        }
    }

    #[tokio::test]
    async fn confirmation_replay_and_live_gaps_repair_without_reapplying_history() {
        let mut f = Fixture::new(Duration::from_secs(5));
        f.succeed().await;
        // A batch lost while the live subscription became active is applied without revoking success.
        let id = f.request(1).await;
        f.send(&id, 1, payload(0)).await;
        assert_eq!(f.ack().await.len(), 1);
        f.send(&id, -1, vec![]).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(*f.updates.borrow(), KvStreamStatus::Success);
        // A live gap revokes success until replay fills it.
        f.live(3).await;
        f.status(KvStreamStatus::Recovering).await;
        let id = f.request(2).await;
        f.send(&id, 2, payload(0)).await;
        f.send(&id, 3, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        assert_eq!(f.ack().await.len(), 1);
        assert_eq!(f.ack().await.len(), 1);
        assert!(f.ack().await.is_empty());
        f.status(KvStreamStatus::Success).await;
    }

    #[tokio::test]
    async fn transient_reconnect_and_resume_keep_the_cursor() {
        let mut f = Fixture::new(Duration::from_secs(5));
        f.succeed().await;
        f.restart_replay().await;
        f.status(KvStreamStatus::Recovering).await;
        // The listener reconnects by itself and continues from its cursor.
        let id = f.request(1).await;
        f.send(&id, -1, vec![]).await;
        f.ack().await;
        f.status(KvStreamStatus::Success).await;
        f.commands.send(KvStreamCommand::Suspend).await.unwrap();
        f.status(KvStreamStatus::Recovering).await;
        let (accepted, acknowledgement) = oneshot::channel();
        f.commands
            .send(KvStreamCommand::Resume(accepted))
            .await
            .unwrap();
        acknowledgement.await.unwrap();
        let id = f.request(1).await;
        f.send(&id, -1, vec![]).await;
        f.ack().await;
        f.status(KvStreamStatus::Success).await;
    }

    #[tokio::test]
    async fn undecodable_batch_is_skipped_while_wrong_rank_and_failed_application_fail() {
        let mut f = Fixture::new(Duration::from_secs(5));
        let id = f.request(0).await;
        f.send(&id, 0, b"not msgpack".to_vec()).await;
        f.send(&id, 1, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        assert_eq!(f.ack().await.len(), 1);
        f.ack().await;
        f.status(KvStreamStatus::Success).await;

        let mut f = Fixture::new(Duration::from_secs(5));
        let id = f.request(0).await;
        f.send(&id, 500, payload(1)).await;
        assert!(matches!(
            f.final_status().await,
            KvStreamStatus::Failed { .. }
        ));

        let mut f = Fixture::new(Duration::from_secs(5));
        let id = f.request(0).await;
        f.send(&id, 0, payload(0)).await;
        let PublisherInput::Recovery(_, done) = f.input().await else {
            panic!("expected recovery application")
        };
        done.send(Err(anyhow::anyhow!("index offline"))).unwrap();
        assert!(matches!(
            f.final_status().await,
            KvStreamStatus::Failed { .. }
        ));
    }

    #[tokio::test]
    async fn confirmed_missing_history_is_distinct_from_exhausted_retries() {
        for (is_batch_sent, is_end_sent) in [(true, true), (true, false), (false, false)] {
            let mut f = Fixture::new(Duration::from_secs(1));
            let id = f.request(0).await;
            if is_batch_sent {
                f.send(&id, 500, payload(0)).await;
            }
            if is_end_sent {
                f.send(&id, -1, vec![]).await;
            }
            let status = f.final_status().await;
            if is_end_sent {
                assert_eq!(
                    status,
                    KvStreamStatus::MissingHistory {
                        expected: 0,
                        got: 500
                    }
                );
            } else {
                assert!(matches!(status, KvStreamStatus::Failed { .. }));
            }
        }
    }

    #[tokio::test]
    async fn teardown_does_not_wait_for_a_stalled_index() {
        let mut f = Fixture::new(Duration::from_secs(5));
        let id = f.request(0).await;
        f.send(&id, 0, payload(0)).await;
        // Hold the acknowledgement as an index blocked in application would.
        let _stalled = f.input().await;
        f.cancel.cancel();
        tokio::time::timeout(Duration::from_secs(1), &mut f.listener)
            .await
            .unwrap()
            .unwrap();
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
        f.status(KvStreamStatus::Success).await;
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
        f.ack().await;
        f.status(KvStreamStatus::Success).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn end_marker_is_explicit() {
        assert_eq!(
            decode_replay(vec![
                vec![],
                vec![],
                (-1_i64).to_be_bytes().to_vec(),
                vec![]
            ])
            .unwrap(),
            ReplayFrame::End
        );
        assert!(decode_replay(vec![vec![], vec![], 0_u64.to_be_bytes().to_vec(), vec![]]).is_err());
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
