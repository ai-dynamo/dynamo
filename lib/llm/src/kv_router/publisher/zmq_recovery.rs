// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in replay recovery for a rank's ZMQ KV event listener.

use super::PublisherInput;
use super::zmq_listener::{is_mismatched_rank, normalize_batch};
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

fn decode_replay(mut frames: Vec<Vec<u8>>) -> Result<ReplayFrame> {
    // DEALER removes identity; SGLang omits the topic included by vLLM.
    if !matches!(frames.len(), 3 | 4) || !frames[0].is_empty() {
        bail!("invalid KV replay envelope");
    }
    let payload = frames.pop().expect("envelope has a payload frame");
    let sequence: [u8; 8] = frames
        .pop()
        .expect("envelope has a sequence frame")
        .as_slice()
        .try_into()?;
    if sequence == (-1_i64).to_be_bytes() && payload.is_empty() {
        return Ok(ReplayFrame::End);
    }
    if payload.is_empty() || sequence == (-1_i64).to_be_bytes() {
        bail!("invalid KV replay end marker");
    }
    Ok(ReplayFrame::Batch(u64::from_be_bytes(sequence), payload))
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
        if sequence < self.next {
            // Live overlap with replay, or every batch of an engine restarted at the same address.
            tracing::debug!(
                sequence,
                next = self.next,
                "Ignoring KV batch below the replay cursor"
            );
            return Ok(());
        }
        if self.pending.contains_key(&sequence) {
            return Ok(());
        }
        if self.pending.len() >= MAX_PENDING_BATCHES
            || payload.len() > self.limit.saturating_sub(self.bytes)
        {
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
///
/// A completed ZMQ handshake does not prove that the engine has activated the live
/// subscription. A batch published in that window appears as a gap at the next live batch,
/// which replay repairs; with no later batch the index stays behind until one arrives.
/// Positive confirmation needs an engine-side subscription acknowledgement or sync marker.
pub struct ZmqRecoveryConfig {
    pub replay_endpoint: String,
    pub dp_rank: u32,
    /// The owner's deadline; it caps the first recovery round.
    pub deadline: Instant,
    pub status: tokio::sync::watch::Sender<KvStreamStatus>,
    pub commands: mpsc::Receiver<KvStreamCommand>,
}

/// `Success` describes whichever engine answers at the endpoints, so the owner verifies the
/// engine's identity before acting on it. Only a `Failed` round that ran out of transport
/// retries accepts `Restart`. A listener that stops reports `Failed` unless its status is
/// already final.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvStreamStatus {
    Recovering,
    Success,
    MissingHistory { expected: u64, got: u64 },
    Failed { reason: String },
}

#[derive(Debug)]
pub enum KvStreamCommand {
    /// Starts a new round from the cursor, capped by `deadline`; a refusal drops `accepted`.
    Restart {
        deadline: Instant,
        accepted: tokio::sync::oneshot::Sender<()>,
    },
}

/// Each failure site chooses whether a retry within the round can help.
#[derive(Debug, thiserror::Error)]
enum RecoveryError {
    #[error("{0}")]
    Retry(anyhow::Error),
    #[error("{0}")]
    Fatal(anyhow::Error),
    #[error("missing KV sequence {expected}; next available sequence is {got}")]
    MissingHistory { expected: u64, got: u64 },
}

fn retry(error: impl Into<anyhow::Error>) -> RecoveryError {
    RecoveryError::Retry(error.into())
}

fn fatal(error: impl Into<anyhow::Error>) -> RecoveryError {
    RecoveryError::Fatal(error.into())
}

/// Caps both the live/replay merge buffer and one replay message.
const MAX_REPLAY_BYTES: usize = 32 * 1024 * 1024;
const MAX_PENDING_BATCHES: usize = 4096;
const MAX_LIVE_DRAIN: usize = 4096;
const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(5);
const RETRY_BACKOFF: Duration = Duration::from_millis(500);
const ROUND_TIMEOUT: Duration = Duration::from_secs(120);
const BARRIER_TIMEOUT: Duration = Duration::from_secs(30);

async fn wait_for_handshake(monitor: &mut tmq::pair::Pair) -> Result<()> {
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

/// Waits until the processor and index have finished every batch sent before it.
async fn barrier(tx: &mpsc::UnboundedSender<PublisherInput>) -> Result<()> {
    let (done, result) = tokio::sync::oneshot::channel();
    tx.send(PublisherInput::Barrier(done))
        .map_err(|_| anyhow::anyhow!("KV publisher stopped"))?;
    result
        .await
        .map_err(|_| anyhow::anyhow!("KV index barrier dropped"))?
}

fn set_status(status: &tokio::sync::watch::Sender<KvStreamStatus>, value: KvStreamStatus) {
    status.send_if_modified(|current| {
        let is_changed = *current != value;
        *current = value;
        is_changed
    });
}

fn round_deadline(cap: Instant) -> Instant {
    (Instant::now() + ROUND_TIMEOUT).min(cap)
}

/// Runs however the listener exits, including cancellation, abort and panic.
struct ReportStop<'a>(&'a tokio::sync::watch::Sender<KvStreamStatus>);

impl Drop for ReportStop<'_> {
    fn drop(&mut self) {
        self.0.send_if_modified(|status| {
            let is_final = matches!(
                status,
                KvStreamStatus::MissingHistory { .. } | KvStreamStatus::Failed { .. }
            );
            if !is_final {
                *status = KvStreamStatus::Failed {
                    reason: "KV event source stopped".to_string(),
                };
            }
            !is_final
        });
    }
}

/// Applies batches in sequence order through the live path's normalizer and event processor.
pub(super) struct RecoveryIngest<'a> {
    cursor: ReplayCursor,
    normalizer: &'a mut ZmqEventNormalizer,
    tx: &'a mpsc::UnboundedSender<PublisherInput>,
    next_event_id: &'a AtomicU64,
    worker: WorkerWithDpRank,
    expected_rank: Option<u32>,
}

impl<'a> RecoveryIngest<'a> {
    pub(super) fn new(
        normalizer: &'a mut ZmqEventNormalizer,
        tx: &'a mpsc::UnboundedSender<PublisherInput>,
        next_event_id: &'a AtomicU64,
        worker: WorkerWithDpRank,
        expected_rank: Option<u32>,
    ) -> Self {
        Self {
            cursor: ReplayCursor::new(MAX_REPLAY_BYTES),
            normalizer,
            tx,
            next_event_id,
            worker,
            expected_rank,
        }
    }

    /// Handles bad events as ordinary live ingestion does.
    fn apply(&mut self, sequence: u64, payload: Vec<u8>) -> Result<(), RecoveryError> {
        self.cursor.insert(sequence, payload).map_err(retry)?;
        while let Some(payload) = self.cursor.pending.remove(&self.cursor.next) {
            self.cursor.bytes -= payload.len();
            let sequence = self.cursor.next;
            match decode_event_batch(&payload) {
                Err(error) => tracing::warn!(%error, sequence, "Failed to decode ZMQ KV batch"),
                Ok(batch) if is_mismatched_rank(batch.data_parallel_rank, self.expected_rank) => {
                    tracing::warn!(
                        sequence,
                        expected_rank = ?self.expected_rank,
                        "Ignoring mismatched KV source rank"
                    );
                }
                Ok(batch) => {
                    let mut events = Vec::new();
                    normalize_batch(
                        batch,
                        self.normalizer,
                        self.worker,
                        self.next_event_id,
                        &mut events,
                    );
                    if !events.is_empty() {
                        self.tx
                            .send(PublisherInput::Events(events))
                            .map_err(|_| fatal(anyhow::anyhow!("KV publisher stopped")))?;
                    }
                }
            }
            self.cursor.next = sequence
                .checked_add(1)
                .ok_or_else(|| fatal(anyhow::anyhow!("KV sequence exhausted")))?;
        }
        Ok(())
    }
}

pub(super) async fn recovering_listener(
    endpoint: &str,
    topic: &str,
    mut ingest: RecoveryIngest<'_>,
    mut config: ZmqRecoveryConfig,
    cancel: tokio_util::sync::CancellationToken,
) {
    let _report_stop = ReportStop(&config.status);
    // `None` while recovery has succeeded; otherwise the current round's deadline.
    let mut deadline = Some(round_deadline(config.deadline));
    let outcome = loop {
        set_status(&config.status, KvStreamStatus::Recovering);
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            result = recovery_session(endpoint, topic, &config, &mut ingest, &mut deadline) => result,
        };
        let Err(error) = result;
        let error = match error {
            RecoveryError::MissingHistory { expected, got } => {
                tracing::warn!(
                    dp_rank = config.dp_rank,
                    expected,
                    got,
                    "KV replay confirms missing history"
                );
                break KvStreamStatus::MissingHistory { expected, got };
            }
            RecoveryError::Fatal(error) => {
                tracing::warn!(dp_rank = config.dp_rank, %error, "KV recovery failed");
                break KvStreamStatus::Failed {
                    reason: error.to_string(),
                };
            }
            RecoveryError::Retry(error) => error,
        };
        // A transient failure after success starts a new round; within a round it keeps the deadline.
        let round_end = *deadline.get_or_insert_with(|| Instant::now() + ROUND_TIMEOUT);
        set_status(&config.status, KvStreamStatus::Recovering);
        if Instant::now() < round_end {
            tracing::warn!(dp_rank = config.dp_rank, %error, "KV recovery interrupted; retrying");
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep_until(round_end.min(Instant::now() + RETRY_BACKOFF)) => {}
            }
            continue;
        }
        tracing::warn!(dp_rank = config.dp_rank, %error, "KV recovery round exhausted; awaiting restart");
        set_status(
            &config.status,
            KvStreamStatus::Failed {
                reason: format!("KV recovery round exhausted: {error}"),
            },
        );
        // The sockets are closed; the cursor and index wait for the owner to restart replay.
        let command = tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            command = config.commands.recv() => command,
        };
        let Some(KvStreamCommand::Restart {
            deadline: cap,
            accepted,
        }) = command
        else {
            cancel.cancelled().await;
            return;
        };
        deadline = Some(round_deadline(cap));
        set_status(&config.status, KvStreamStatus::Recovering);
        let _ = accepted.send(());
    };
    set_status(&config.status, outcome);
    // Hold the final status, refusing restarts, until the owner stops this source.
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            Some(_) = config.commands.recv() => {}
        }
    }
}

fn revoke(status: &tokio::sync::watch::Sender<KvStreamStatus>, deadline: &mut Option<Instant>) {
    *deadline = Some(Instant::now() + ROUND_TIMEOUT);
    set_status(status, KvStreamStatus::Recovering);
}

async fn recovery_session(
    endpoint: &str,
    topic: &str,
    config: &ZmqRecoveryConfig,
    ingest: &mut RecoveryIngest<'_>,
    deadline: &mut Option<Instant>,
) -> Result<std::convert::Infallible, RecoveryError> {
    // Replay from the cursor refills anything buffered by an earlier connection.
    ingest.cursor.clear_pending();
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
            .set_maxmsgsize(MAX_REPLAY_BYTES as i64)
            .monitor(
                "inproc://replay-monitor",
                zmq::SocketEvent::HANDSHAKE_SUCCEEDED as i32
                    | zmq::SocketEvent::DISCONNECTED as i32,
            )
            .connect(&config.replay_endpoint)?;
        tokio::try_join!(
            wait_for_handshake(&mut live_monitor),
            wait_for_handshake(&mut replay_monitor)
        )?;
        Ok::<_, anyhow::Error>((live, live_monitor, replay, replay_monitor))
    };
    let (mut live, mut live_monitor, mut replay, mut replay_monitor) = tokio::select! {
        biased;
        _ = tokio::time::sleep_until(deadline.unwrap_or_else(Instant::now)), if deadline.is_some() => return Err(retry(anyhow::anyhow!("KV recovery deadline exceeded while connecting"))),
        result = tokio::time::timeout(INACTIVITY_TIMEOUT, connect) => result.map_err(retry)?.map_err(retry)?,
    };
    let mut is_success = false;
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
            _ = live_monitor.next() => return Err(retry(anyhow::anyhow!("KV live connection lost"))),
            _ = replay_monitor.next() => return Err(retry(anyhow::anyhow!("KV replay connection lost"))),
            _ = tokio::time::sleep_until(deadline.unwrap_or_else(Instant::now)), if deadline.is_some() => return Err(retry(anyhow::anyhow!("KV recovery deadline exceeded"))),
            _ = tokio::time::sleep_until(replay_deadline), if is_replaying => return Err(retry(anyhow::anyhow!("KV replay timed out"))),
            _ = tokio::time::sleep_until(next_replay.unwrap_or_else(Instant::now)), if !is_replaying && next_replay.is_some() => {
                tokio::time::timeout(INACTIVITY_TIMEOUT, replay.send(tmq::Multipart::from(vec![
                    Vec::new(), ingest.cursor.next().to_be_bytes().to_vec(),
                ]))).await.map_err(retry)?.map_err(retry)?;
                is_replaying = true;
                next_replay = None;
                requested = ingest.cursor.next();
                replay_first = None;
                replay_next = None;
                is_reply_truncated = false;
                replay_deadline = Instant::now() + INACTIVITY_TIMEOUT;
            },
            message = replay.next(), if is_replaying => {
                let message = message.ok_or_else(|| retry(anyhow::anyhow!("KV replay socket ended")))?.map_err(retry)?;
                replay_deadline = Instant::now() + INACTIVITY_TIMEOUT;
                // A reply in an unknown format fails the same way on every attempt.
                match decode_replay(multipart_message(message)).map_err(fatal)? {
                    ReplayFrame::Batch(sequence, payload) => {
                        // A hole inside one reply is a transport drop; the next round re-requests it.
                        is_reply_truncated |= replay_next.is_some_and(|next| next != sequence);
                        replay_first.get_or_insert(sequence);
                        replay_next = Some(sequence.saturating_add(1));
                        if !is_reply_truncated {
                            ingest.apply(sequence, payload)?;
                        }
                    }
                    ReplayFrame::End => {
                        is_replaying = false;
                        // Bound each drain so a busy producer cannot starve cancellation or the round deadline.
                        for _ in 0..MAX_LIVE_DRAIN {
                            let Some(message) = live.next().now_or_never() else { break };
                            let message = message.ok_or_else(|| retry(anyhow::anyhow!("KV live socket ended")))?.map_err(retry)?;
                            if let Some((sequence, payload)) = live_batch(message) {
                                ingest.apply(sequence, payload)?;
                            }
                        }
                        // Replay is contiguous, so only a late first batch proves loss.
                        if let Some(got) = replay_first.filter(|first| *first > requested && ingest.cursor.next() == requested) {
                            return Err(RecoveryError::MissingHistory { expected: requested, got });
                        }
                        if ingest.cursor.gap().is_some() || is_reply_truncated {
                            next_replay = Some(Instant::now());
                        } else {
                            // Report success only after the index has finished every batch.
                            tokio::time::timeout(BARRIER_TIMEOUT, barrier(ingest.tx)).await
                                .map_err(fatal)?.map_err(fatal)?;
                            is_success = true;
                            *deadline = None;
                            set_status(&config.status, KvStreamStatus::Success);
                        }
                    }
                }
            },
            message = live.next() => {
                let message = message.ok_or_else(|| retry(anyhow::anyhow!("KV live socket ended")))?.map_err(retry)?;
                let Some((sequence, payload)) = live_batch(message) else { continue };
                if sequence > ingest.cursor.next() && is_success {
                    is_success = false;
                    revoke(&config.status, deadline);
                    next_replay = Some(Instant::now());
                }
                ingest.apply(sequence, payload)?;
            },
        }
    }
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
                Some(0),
                None,
                Some(ZmqRecoveryConfig {
                    replay_endpoint: replay.get_socket().get_last_endpoint().unwrap().unwrap(),
                    dp_rank: 0,
                    deadline: Instant::now() + timeout,
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
        async fn events(&mut self) -> Vec<PlacementEvent> {
            let PublisherInput::Events(events) = self.input().await else {
                panic!("expected recovered events")
            };
            events
        }
        async fn barrier(&mut self) {
            let PublisherInput::Barrier(done) = self.input().await else {
                panic!("expected the index barrier")
            };
            done.send(Ok(())).unwrap();
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
            self.events().await;
            self.send(&id, -1, vec![]).await;
            self.barrier().await;
            self.status(KvStreamStatus::Success).await;
        }
        async fn live_events(&mut self, sequence: u64) -> Vec<PlacementEvent> {
            tokio::time::timeout(Duration::from_secs(5), async {
                let mut retry = tokio::time::interval(Duration::from_millis(20));
                loop {
                    tokio::select! {
                        input = self.inputs.recv() => {
                            let Some(PublisherInput::Events(events)) = input else {
                                panic!("expected live history")
                            };
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
    async fn complete_replay_succeeds_without_live_events() {
        for is_empty in [true, false] {
            let mut f = Fixture::new(Duration::from_secs(5));
            let id = f.request(0).await;
            if !is_empty {
                f.send(&id, 0, payload(0)).await;
                f.events().await;
            }
            f.send(&id, -1, vec![]).await;
            f.barrier().await;
            f.status(KvStreamStatus::Success).await;
        }
    }

    #[tokio::test]
    async fn live_gap_after_success_is_repaired_without_reapplying_history() {
        let mut f = Fixture::new(Duration::from_secs(5));
        f.succeed().await;
        // A live gap revokes success until replay fills it; republish until the subscription is active.
        tokio::time::timeout(Duration::from_secs(5), async {
            while *f.updates.borrow() != KvStreamStatus::Recovering {
                f.live(2).await;
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        let id = f.request(1).await;
        f.send(&id, 1, payload(0)).await;
        f.send(&id, 2, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        assert_eq!(f.events().await.len(), 1);
        assert_eq!(f.events().await.len(), 1);
        f.barrier().await;
        f.status(KvStreamStatus::Success).await;
    }

    async fn restart(f: &mut Fixture) -> Result<(), oneshot::error::RecvError> {
        let (accepted, acknowledgement) = oneshot::channel();
        f.commands
            .send(KvStreamCommand::Restart {
                deadline: Instant::now() + Duration::from_secs(5),
                accepted,
            })
            .await
            .unwrap();
        acknowledgement.await
    }

    #[tokio::test]
    async fn exhausted_round_restarts_and_reconnects_keep_the_cursor() {
        let mut f = Fixture::new(Duration::from_secs(1));
        let id = f.request(0).await;
        f.send(&id, 0, payload(0)).await;
        f.events().await;
        // Without an end marker the round runs out; the cursor waits for a restart.
        let KvStreamStatus::Failed { reason } = f.final_status().await else {
            panic!("expected an exhausted round")
        };
        assert!(
            reason.starts_with("KV recovery round exhausted"),
            "{reason}"
        );
        restart(&mut f).await.unwrap();
        let id = f.request(1).await;
        f.send(&id, -1, vec![]).await;
        f.barrier().await;
        f.status(KvStreamStatus::Success).await;
        f.restart_replay().await;
        f.status(KvStreamStatus::Recovering).await;
        // The listener reconnects by itself and continues from its cursor.
        let id = f.request(1).await;
        f.send(&id, -1, vec![]).await;
        f.barrier().await;
        f.status(KvStreamStatus::Success).await;
    }

    #[tokio::test]
    async fn undecodable_and_wrong_rank_batches_are_skipped() {
        let unlabeled =
            rmp_serde::to_vec(&(0.0_f64, vec![vec!["AllBlocksCleared"]], None::<i32>)).unwrap();
        let mut f = Fixture::new(Duration::from_secs(5));
        let id = f.request(0).await;
        f.send(&id, 0, b"not msgpack".to_vec()).await;
        f.send(&id, 1, payload(1)).await;
        f.send(&id, 2, unlabeled).await;
        f.send(&id, 3, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        assert_eq!(f.events().await.len(), 1);
        f.barrier().await;
        f.status(KvStreamStatus::Success).await;
    }

    #[tokio::test]
    async fn final_failures_refuse_restart() {
        for is_malformed in [false, true] {
            // A long round shows the failure is final, not an exhausted round.
            let mut f = Fixture::new(Duration::from_secs(30));
            let id = f.request(0).await;
            if is_malformed {
                // A batch frame without data is an invalid end marker.
                f.send(&id, 0, vec![]).await;
            } else {
                f.send(&id, -1, vec![]).await;
                let PublisherInput::Barrier(done) = f.input().await else {
                    panic!("expected the index barrier")
                };
                done.send(Err(anyhow::anyhow!("index offline"))).unwrap();
            }
            assert!(matches!(
                f.final_status().await,
                KvStreamStatus::Failed { .. }
            ));
            assert!(restart(&mut f).await.is_err());
        }
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
        f.send(&id, -1, vec![]).await;
        // Hold the barrier as an index blocked in application would.
        let _stalled = f.input().await;
        f.cancel.cancel();
        tokio::time::timeout(Duration::from_secs(1), &mut f.listener)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn stopped_listener_does_not_leave_success_behind() {
        let mut f = Fixture::new(Duration::from_secs(5));
        f.succeed().await;
        // As the processor's reset fence or the supervisor stops the source.
        f.cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), &mut f.listener)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(*f.updates.borrow(), KvStreamStatus::Failed { .. }));
    }

    #[tokio::test]
    async fn live_fills_hole_observed_first_on_replay() {
        let mut f = Fixture::new(Duration::from_secs(5));
        let id = f.request(0).await;
        f.send(&id, 1, payload(0)).await;
        // Let replay expose the hole before its live counterpart arrives.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(f.live_events(0).await[0].event.event_id, 0);
        assert_eq!(f.events().await[0].event.event_id, 1);
        f.send(&id, -1, vec![]).await;
        f.barrier().await;
        f.status(KvStreamStatus::Success).await;
    }

    #[tokio::test]
    async fn hole_inside_one_replay_reply_is_requested_again() {
        let mut f = Fixture::new(Duration::from_secs(5));
        let id = f.request(0).await;
        f.send(&id, 0, payload(0)).await;
        assert_eq!(f.events().await[0].event.event_id, 0);
        f.send(&id, 2, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        let id = f.request(1).await;
        assert_eq!(*f.updates.borrow(), KvStreamStatus::Recovering);
        f.send(&id, 1, payload(0)).await;
        f.send(&id, 2, payload(0)).await;
        f.send(&id, -1, vec![]).await;
        assert_eq!(f.events().await[0].event.event_id, 1);
        assert_eq!(f.events().await[0].event.event_id, 2);
        f.barrier().await;
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
