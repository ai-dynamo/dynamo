// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side persistent multiplexed TCP response connection pool.

use std::{
    collections::VecDeque,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use dashmap::{DashMap, mapref::entry::Entry};
use futures::{SinkExt, StreamExt};
use parking_lot::{Mutex, RwLock};
use tokio::{
    net::TcpStream,
    sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot},
};
use tokio_util::{
    codec::{FramedRead, FramedWrite},
    sync::CancellationToken,
};
use uuid::Uuid;

use super::{
    ConnectionHandshake, MuxCodec, MuxFrame, MuxFrameKind, RESPONSE_MUX_CONNECT_TIMEOUT_SECS,
    RESPONSE_MUX_IDLE_TTL_SECS, RESPONSE_MUX_POOL_SIZE, RESPONSE_MUX_STREAM_WRITER_QUEUE,
    RESPONSE_MUX_VERSION, RESPONSE_MUX_WRITER_QUEUE, ResponseMuxConfig,
};
use crate::{
    engine::AsyncEngineContext,
    metrics::response_mux,
    pipeline::network::{
        ConnectionInfo, MultiplexedStreamSender, ResponseStreamPrologue, StreamSender,
        codec::{TwoPartCodec, TwoPartMessage},
        egress::tcp_client::TcpWriteBuffer,
        tcp::ResponseMuxConnectionInfo,
    },
};

struct WriterCommand {
    frame: MuxFrame,
    written: Option<oneshot::Sender<Result<(), String>>>,
    _writer_permit: Option<OwnedSemaphorePermit>,
    _queued_byte_permit: Option<OwnedSemaphorePermit>,
    priority_enqueued_at: Option<Instant>,
    enqueued_at: Instant,
}

impl WriterCommand {
    fn new(frame: MuxFrame, written: Option<oneshot::Sender<Result<(), String>>>) -> Self {
        Self {
            frame,
            written,
            _writer_permit: None,
            _queued_byte_permit: None,
            priority_enqueued_at: None,
            enqueued_at: Instant::now(),
        }
    }

    fn priority(frame: MuxFrame, written: Option<oneshot::Sender<Result<(), String>>>) -> Self {
        let mut command = Self::new(frame, written);
        command.priority_enqueued_at = Some(Instant::now());
        command
    }

    fn with_writer_permit(mut self, permit: OwnedSemaphorePermit) -> Self {
        self._writer_permit = Some(permit);
        self
    }

    fn with_queued_byte_permit(mut self, permit: OwnedSemaphorePermit) -> Self {
        self._queued_byte_permit = Some(permit);
        self
    }

    fn fail(mut self, reason: &str) {
        if let Some(written) = self.written.take() {
            let _ = written.send(Err(reason.to_string()));
        }
    }
}

#[inline]
fn per_frame_metrics_enabled() -> bool {
    true
}

#[derive(Clone, Copy)]
struct PoolConfig {
    pool_size: usize,
    writer_queue: usize,
    stream_writer_queue: usize,
    initial_window: usize,
    connection_window: usize,
    batch_interval: Duration,
    batch_max_bytes: usize,
    batch_max_frames: usize,
    packet_metrics: bool,
    idle_ttl: Duration,
    connect_timeout: Duration,
}

impl PoolConfig {
    fn from_runtime(config: ResponseMuxConfig) -> Self {
        Self {
            pool_size: RESPONSE_MUX_POOL_SIZE,
            writer_queue: RESPONSE_MUX_WRITER_QUEUE,
            stream_writer_queue: RESPONSE_MUX_STREAM_WRITER_QUEUE,
            initial_window: config.stream_window_bytes,
            connection_window: config.connection_window_bytes,
            batch_interval: config.batch_interval,
            batch_max_bytes: config.batch_max_bytes,
            batch_max_frames: config.batch_max_frames,
            packet_metrics: config.packet_metrics,
            idle_ttl: Duration::from_secs(RESPONSE_MUX_IDLE_TTL_SECS),
            connect_timeout: Duration::from_secs(RESPONSE_MUX_CONNECT_TIMEOUT_SECS),
        }
    }
}

#[derive(Default)]
struct StreamWriterState {
    pending: VecDeque<WriterCommand>,
    scheduled: bool,
}

struct WorkerStreamState {
    context: Arc<dyn AsyncEngineContext>,
    credits: Arc<Semaphore>,
    max_credits: usize,
    writer_slots: Arc<Semaphore>,
    writer: Mutex<StreamWriterState>,
    closed: AtomicBool,
    close_token: CancellationToken,
}

type ScheduledStream = (Uuid, Arc<WorkerStreamState>);
type BlockedData = (WriterCommand, Option<ScheduledStream>, Instant);

impl WorkerStreamState {
    fn replenish_credits(&self, credits: usize) -> usize {
        if self.closed.load(Ordering::Acquire) || self.credits.is_closed() {
            return 0;
        }
        let available = self.credits.available_permits();
        let replenished = credits.min(self.max_credits.saturating_sub(available));
        if replenished > 0 {
            self.credits.add_permits(replenished);
        }
        replenished
    }
}

struct MuxConnection {
    id: u64,
    cancel: CancellationToken,
    priority_tx: mpsc::Sender<WriterCommand>,
    ready_tx: mpsc::UnboundedSender<Uuid>,
    streams: DashMap<Uuid, Arc<WorkerStreamState>>,
    healthy: AtomicBool,
    active_streams: AtomicUsize,
    queued_frames: AtomicUsize,
    queued_bytes: AtomicUsize,
    max_queued_bytes: usize,
    queued_byte_slots: Arc<Semaphore>,
    connection_credits: Arc<Semaphore>,
    max_connection_credits: usize,
    sent_data_bytes: AtomicU64,
    acknowledged_data_bytes: AtomicU64,
    batch_interval: Duration,
    batch_max_bytes: usize,
    batch_max_frames: usize,
}

impl MuxConnection {
    async fn connect(
        id: u64,
        address: &str,
        frontend_server_id: Uuid,
        version: u8,
        cancel: CancellationToken,
        config: PoolConfig,
    ) -> Result<Arc<Self>> {
        let stream = tokio::time::timeout(config.connect_timeout, TcpStream::connect(address))
            .await
            .map_err(|_| anyhow!("response mux connect timeout to {address}"))??;
        stream.set_nodelay(true)?;
        let packet_baseline = config
            .packet_metrics
            .then(|| crate::pipeline::network::tcp::tcp_data_segments_out(&stream))
            .flatten();

        let (read_half, write_half) = stream.into_split();
        let mux_codec =
            || TwoPartCodec::new(Some(crate::pipeline::network::get_tcp_max_message_size()));
        let mut handshake_reader = FramedRead::new(read_half, mux_codec());
        let mut handshake_writer = FramedWrite::new(write_half, mux_codec());
        let handshake = ConnectionHandshake::ResponseMux {
            version,
            frontend_server_id,
            connection_id: Uuid::new_v4(),
        };
        let header = serde_json::to_vec(&handshake)?;
        handshake_writer
            .send(TwoPartMessage::from_header(header.into()))
            .await
            .context("failed to send response mux handshake")?;
        let ack = tokio::time::timeout(config.connect_timeout, handshake_reader.next())
            .await
            .map_err(|_| anyhow!("response mux handshake ack timeout from {address}"))?
            .ok_or_else(|| anyhow!("frontend closed before response mux handshake ack"))??;
        let ack = MuxFrame::try_from_two_part(ack)?;
        if ack.kind != MuxFrameKind::ConnectionAck || ack.connection_ack_offset()? != 0 {
            anyhow::bail!("frontend returned invalid response mux connection ack");
        }
        let read_half = handshake_reader.into_inner();
        let write_half = handshake_writer.into_inner();

        let (priority_tx, priority_rx) = mpsc::channel(config.writer_queue);
        let (ready_tx, ready_rx) = mpsc::unbounded_channel();
        let cancel = cancel.child_token();
        let connection = Arc::new(Self {
            id,
            cancel: cancel.clone(),
            priority_tx,
            ready_tx,
            streams: DashMap::new(),
            healthy: AtomicBool::new(true),
            active_streams: AtomicUsize::new(0),
            queued_frames: AtomicUsize::new(0),
            queued_bytes: AtomicUsize::new(0),
            max_queued_bytes: config.connection_window,
            queued_byte_slots: Arc::new(Semaphore::new(config.connection_window)),
            connection_credits: Arc::new(Semaphore::new(config.connection_window)),
            max_connection_credits: config.connection_window,
            sent_data_bytes: AtomicU64::new(0),
            acknowledged_data_bytes: AtomicU64::new(0),
            batch_interval: config.batch_interval,
            batch_max_bytes: config.batch_max_bytes,
            batch_max_frames: config.batch_max_frames,
        });
        response_mux::CONNECTIONS_TOTAL
            .with_label_values(&["worker", "created"])
            .inc();
        response_mux::ACTIVE_CONNECTIONS
            .with_label_values(&["worker"])
            .inc();

        tokio::spawn(Self::writer_task(
            Arc::downgrade(&connection),
            write_half,
            priority_rx,
            ready_rx,
            cancel.clone(),
        ));
        tokio::spawn(Self::reader_task(
            Arc::downgrade(&connection),
            FramedRead::new(read_half, MuxCodec::default()),
            cancel,
            packet_baseline,
        ));
        Ok(connection)
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn replenish_connection_credits(&self, credits: usize) -> usize {
        if !self.is_healthy() || self.connection_credits.is_closed() {
            return 0;
        }
        let available = self.connection_credits.available_permits();
        let replenished = credits.min(self.max_connection_credits.saturating_sub(available));
        if replenished > 0 {
            self.connection_credits.add_permits(replenished);
        }
        replenished
    }

    fn acknowledge_connection_credits(&self, acknowledged_bytes: u64) -> Result<()> {
        let previous = self.acknowledged_data_bytes.load(Ordering::Acquire);
        if acknowledged_bytes < previous {
            anyhow::bail!(
                "response mux connection ACK moved backwards from {previous} to {acknowledged_bytes}"
            );
        }
        if acknowledged_bytes == previous {
            return Ok(());
        }
        let sent = self.sent_data_bytes.load(Ordering::Acquire);
        if acknowledged_bytes > sent {
            anyhow::bail!(
                "response mux connection ACK {acknowledged_bytes} exceeds sent offset {sent}"
            );
        }
        self.acknowledged_data_bytes
            .store(acknowledged_bytes, Ordering::Release);
        let delta = acknowledged_bytes.saturating_sub(previous) as usize;
        self.replenish_connection_credits(delta.min(self.max_connection_credits));
        Ok(())
    }

    fn fail(&self, reason: &str) {
        if !self.healthy.swap(false, Ordering::AcqRel) {
            return;
        }
        tracing::warn!(
            connection_id = self.id,
            reason,
            "response mux connection failed"
        );
        self.cancel.cancel();
        self.connection_credits.close();
        self.queued_byte_slots.close();
        response_mux::CONNECTIONS_TOTAL
            .with_label_values(&["worker", "failed"])
            .inc();
        response_mux::ACTIVE_CONNECTIONS
            .with_label_values(&["worker"])
            .dec();
        let stream_ids: Vec<Uuid> = self.streams.iter().map(|entry| *entry.key()).collect();
        response_mux::CONNECTION_LOST_STREAMS_TOTAL.inc_by(stream_ids.len() as u64);
        for stream_id in stream_ids {
            self.remove_stream(stream_id, reason, true);
        }
    }

    fn close_stream_state(&self, state: &WorkerStreamState, reason: &str) {
        if state.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        state.credits.close();
        state.writer_slots.close();
        state.close_token.cancel();
        let (pending, was_scheduled) = {
            let mut writer = state.writer.lock();
            let pending = writer.pending.drain(..).collect::<Vec<_>>();
            let was_scheduled = writer.scheduled;
            writer.scheduled = false;
            (pending, was_scheduled)
        };
        self.queued_frames
            .fetch_sub(pending.len(), Ordering::AcqRel);
        let pending_bytes = pending
            .iter()
            .map(|command| command.frame.encoded_len())
            .sum::<usize>();
        self.queued_bytes.fetch_sub(pending_bytes, Ordering::AcqRel);
        response_mux::QUEUED_BYTES
            .with_label_values(&["worker"])
            .sub(pending_bytes as i64);
        if was_scheduled {
            response_mux::READY_STREAMS
                .with_label_values(&["worker"])
                .dec();
        }
        for command in pending {
            command.fail(reason);
        }
    }

    fn remove_stream(&self, stream_id: Uuid, reason: &str, kill_context: bool) -> bool {
        let Some((_, state)) = self.streams.remove(&stream_id) else {
            return false;
        };
        self.close_stream_state(&state, reason);
        if kill_context {
            state.context.kill();
        }
        self.active_streams.fetch_sub(1, Ordering::AcqRel);
        response_mux::ACTIVE_STREAMS
            .with_label_values(&["worker"])
            .dec();
        true
    }

    async fn send_priority_command(&self, command: WriterCommand) -> Result<()> {
        if !self.is_healthy() {
            command.fail("response mux connection is unhealthy");
            anyhow::bail!("response mux connection is unhealthy");
        }
        self.queued_frames.fetch_add(1, Ordering::AcqRel);
        match self.priority_tx.try_send(command) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(command)) => {
                if let Err(err) = self.priority_tx.send(command).await {
                    self.queued_frames.fetch_sub(1, Ordering::AcqRel);
                    err.0.fail("response mux priority writer stopped");
                    anyhow::bail!("response mux priority writer stopped");
                }
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(command)) => {
                self.queued_frames.fetch_sub(1, Ordering::AcqRel);
                command.fail("response mux priority writer stopped");
                anyhow::bail!("response mux priority writer stopped")
            }
        }
    }

    fn try_send_priority_command(&self, command: WriterCommand) {
        if !self.is_healthy() {
            command.fail("response mux connection is unhealthy");
            return;
        }
        self.queued_frames.fetch_add(1, Ordering::AcqRel);
        if let Err(err) = self.priority_tx.try_send(command) {
            self.queued_frames.fetch_sub(1, Ordering::AcqRel);
            let reason = match err {
                mpsc::error::TrySendError::Full(command) => {
                    command.fail("response mux priority writer queue is full");
                    "response mux priority writer queue is full"
                }
                mpsc::error::TrySendError::Closed(command) => {
                    command.fail("response mux priority writer is closed");
                    "response mux priority writer is closed"
                }
            };
            self.fail(reason);
        }
    }

    fn enqueue_stream_command(
        &self,
        stream_id: Uuid,
        state: &WorkerStreamState,
        command: WriterCommand,
    ) -> Result<()> {
        if !self.is_healthy() || state.closed.load(Ordering::Acquire) {
            command.fail("response mux stream is closed");
            anyhow::bail!("response mux stream is closed");
        }

        let encoded_len = command.frame.encoded_len();
        self.queued_bytes.fetch_add(encoded_len, Ordering::AcqRel);
        response_mux::QUEUED_BYTES
            .with_label_values(&["worker"])
            .add(encoded_len as i64);

        let mut writer = state.writer.lock();
        if state.closed.load(Ordering::Acquire) {
            self.queued_bytes.fetch_sub(encoded_len, Ordering::AcqRel);
            response_mux::QUEUED_BYTES
                .with_label_values(&["worker"])
                .sub(encoded_len as i64);
            command.fail("response mux stream is closed");
            anyhow::bail!("response mux stream is closed");
        }
        writer.pending.push_back(command);
        self.queued_frames.fetch_add(1, Ordering::AcqRel);
        if per_frame_metrics_enabled() {
            response_mux::STREAM_WRITER_QUEUE_OCCUPANCY.observe(writer.pending.len() as f64);
        }
        if !writer.scheduled {
            writer.scheduled = true;
            response_mux::READY_STREAMS
                .with_label_values(&["worker"])
                .inc();
            if self.ready_tx.send(stream_id).is_err() {
                writer.scheduled = false;
                response_mux::READY_STREAMS
                    .with_label_values(&["worker"])
                    .dec();
                self.queued_frames.fetch_sub(1, Ordering::AcqRel);
                if let Some(command) = writer.pending.pop_back() {
                    self.queued_bytes.fetch_sub(encoded_len, Ordering::AcqRel);
                    response_mux::QUEUED_BYTES
                        .with_label_values(&["worker"])
                        .sub(encoded_len as i64);
                    command.fail("response mux fair writer stopped");
                }
                anyhow::bail!("response mux fair writer stopped");
            }
        }
        Ok(())
    }

    fn reschedule_stream(&self, stream_id: Uuid, state: &Arc<WorkerStreamState>) -> Result<()> {
        response_mux::ROUND_ROBIN_TURNS_TOTAL.inc();
        let mut writer = state.writer.lock();
        if !state.closed.load(Ordering::Acquire) && !writer.pending.is_empty() {
            self.ready_tx
                .send(stream_id)
                .map_err(|_| anyhow!("response mux fair writer stopped"))?;
        } else if writer.scheduled {
            writer.scheduled = false;
            response_mux::READY_STREAMS
                .with_label_values(&["worker"])
                .dec();
        }
        Ok(())
    }

    async fn writer_task(
        weak: Weak<Self>,
        mut write_half: tokio::net::tcp::OwnedWriteHalf,
        mut priority_rx: mpsc::Receiver<WriterCommand>,
        mut ready_rx: mpsc::UnboundedReceiver<Uuid>,
        cancel: CancellationToken,
    ) {
        let mut write_buf = TcpWriteBuffer::new();
        let queue_depth = response_mux::WRITER_QUEUE_DEPTH
            .with_label_values(&["worker"])
            .clone();
        let frames_per_write = response_mux::FRAMES_PER_WRITE
            .with_label_values(&["worker"])
            .clone();
        let frame_counters = response_mux::FrameCounters::for_direction("worker_to_frontend");
        let metrics_enabled = per_frame_metrics_enabled();
        let mut reported_queue_depth = 0_i64;
        let mut blocked_data: Option<BlockedData> = None;
        let result: Result<()> = async {
            loop {
                let connection = weak
                    .upgrade()
                    .ok_or_else(|| anyhow!("response mux connection dropped"))?;
                if metrics_enabled {
                    let current_queue_depth =
                        connection.queued_frames.load(Ordering::Acquire) as i64;
                    queue_depth.add(current_queue_depth - reported_queue_depth);
                    reported_queue_depth = current_queue_depth;
                }

                let (command, scheduled_stream, connection_permit) = if let Some((
                    blocked_command,
                    blocked_stream,
                    blocked_since,
                )) = blocked_data.take()
                {
                    if blocked_command.frame.kind != MuxFrameKind::Data {
                        (blocked_command, blocked_stream, None)
                    } else {
                    enum BlockedNext {
                        Priority(WriterCommand),
                        Credit(OwnedSemaphorePermit),
                        StreamClosed,
                    }
                    let blocked_close = blocked_stream
                        .as_ref()
                        .expect("Data commands are always stream-scheduled")
                        .1
                        .close_token
                        .clone();
                    let credits = connection.connection_credits.clone();
                    let next = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return Ok(()),
                        _ = blocked_close.cancelled() => BlockedNext::StreamClosed,
                        Some(command) = priority_rx.recv() => BlockedNext::Priority(command),
                        permit = credits.acquire_many_owned(
                            blocked_command
                                .frame
                                .encoded_len()
                                .min(connection.max_connection_credits) as u32
                        ) => BlockedNext::Credit(
                            permit.map_err(|_| anyhow!(
                                "response mux connection closed while writer awaited credits"
                            ))?
                        ),
                    };
                    match next {
                        BlockedNext::Priority(command) => {
                            connection.queued_frames.fetch_sub(1, Ordering::AcqRel);
                            blocked_data = Some((blocked_command, blocked_stream, blocked_since));
                            (command, None, None)
                        }
                        BlockedNext::Credit(permit) => {
                            if metrics_enabled {
                                response_mux::CONNECTION_FLOW_CONTROL_STALL_SECONDS
                                    .observe(blocked_since.elapsed().as_secs_f64());
                            }
                            (blocked_command, blocked_stream, Some(permit))
                        }
                        BlockedNext::StreamClosed => {
                            blocked_command
                                .fail("response mux stream closed while writer awaited credits");
                            continue;
                        }
                    }
                    }
                } else {
                    let (command, scheduled_stream) = loop {
                        if let Ok(command) = priority_rx.try_recv() {
                            connection.queued_frames.fetch_sub(1, Ordering::AcqRel);
                            break (command, None);
                        }

                        enum Next {
                            Priority(WriterCommand),
                            Stream(Uuid),
                        }
                        let next = tokio::select! {
                            biased;
                            _ = cancel.cancelled() => return Ok(()),
                            command = priority_rx.recv() => command.map(Next::Priority),
                            stream_id = ready_rx.recv() => stream_id.map(Next::Stream),
                        };
                        let Some(next) = next else {
                            return Ok(());
                        };
                        match next {
                            Next::Priority(command) => {
                                connection.queued_frames.fetch_sub(1, Ordering::AcqRel);
                                break (command, None);
                            }
                            Next::Stream(stream_id) => {
                                let Some(state) = connection
                                    .streams
                                    .get(&stream_id)
                                    .map(|entry| entry.value().clone())
                                else {
                                    continue;
                                };
                                let command = state.writer.lock().pending.pop_front();
                                if let Some(command) = command {
                                    connection.queued_frames.fetch_sub(1, Ordering::AcqRel);
                                    let encoded_len = command.frame.encoded_len();
                                    connection
                                        .queued_bytes
                                        .fetch_sub(encoded_len, Ordering::AcqRel);
                                    response_mux::QUEUED_BYTES
                                        .with_label_values(&["worker"])
                                        .sub(encoded_len as i64);
                                    break (command, Some((stream_id, state)));
                                }
                                let mut writer = state.writer.lock();
                                if writer.scheduled {
                                    writer.scheduled = false;
                                    response_mux::READY_STREAMS
                                        .with_label_values(&["worker"])
                                        .dec();
                                }
                            }
                        }
                    };
                    if command.frame.kind == MuxFrameKind::Data {
                        let required = command
                            .frame
                            .encoded_len()
                            .min(connection.max_connection_credits)
                            as u32;
                        match connection
                            .connection_credits
                            .clone()
                            .try_acquire_many_owned(required)
                        {
                            Ok(permit) => (command, scheduled_stream, Some(permit)),
                            Err(tokio::sync::TryAcquireError::NoPermits) => {
                                blocked_data = Some((command, scheduled_stream, Instant::now()));
                                continue;
                            }
                            Err(tokio::sync::TryAcquireError::Closed) => {
                                return Err(anyhow!(
                                    "response mux connection closed while writer acquired credits"
                                ));
                            }
                        }
                    } else {
                        (command, scheduled_stream, None)
                    }
                };

                let batching_started = Instant::now();
                let first_is_data = command.frame.kind == MuxFrameKind::Data;
                let mut batch = vec![(command, connection_permit)];
                let mut batch_bytes = batch[0].0.frame.encoded_len();
                let mut force_flush = !first_is_data;
                if let Some((stream_id, state)) = scheduled_stream {
                    let mut held_for_next_turn = false;
                    for _ in 1..super::RESPONSE_MUX_SCHEDULER_QUANTUM {
                        if force_flush
                            || batch.len() >= connection.batch_max_frames
                            || batch_bytes >= connection.batch_max_bytes
                        {
                            break;
                        }
                        let Some(next) = state.writer.lock().pending.pop_front() else {
                            break;
                        };
                        connection.queued_frames.fetch_sub(1, Ordering::AcqRel);
                        let encoded_len = next.frame.encoded_len();
                        connection
                            .queued_bytes
                            .fetch_sub(encoded_len, Ordering::AcqRel);
                        response_mux::QUEUED_BYTES
                            .with_label_values(&["worker"])
                            .sub(encoded_len as i64);
                        if batch_bytes.saturating_add(encoded_len) > connection.batch_max_bytes {
                            blocked_data = Some((next, Some((stream_id, state.clone())), Instant::now()));
                            held_for_next_turn = true;
                            break;
                        }
                        let permit = if next.frame.kind == MuxFrameKind::Data {
                            let required =
                                encoded_len.min(connection.max_connection_credits) as u32;
                            match connection
                                .connection_credits
                                .clone()
                                .try_acquire_many_owned(required)
                            {
                                Ok(permit) => Some(permit),
                                Err(tokio::sync::TryAcquireError::NoPermits) => {
                                    blocked_data = Some((
                                        next,
                                        Some((stream_id, state.clone())),
                                        Instant::now(),
                                    ));
                                    held_for_next_turn = true;
                                    break;
                                }
                                Err(tokio::sync::TryAcquireError::Closed) => {
                                    return Err(anyhow!(
                                        "response mux connection credit window closed"
                                    ));
                                }
                            }
                        } else {
                            None
                        };
                        force_flush = next.frame.kind != MuxFrameKind::Data;
                        batch_bytes = batch_bytes.saturating_add(encoded_len);
                        batch.push((next, permit));
                    }
                    if !held_for_next_turn {
                        connection.reschedule_stream(stream_id, &state)?;
                    }
                }
                let deadline = batching_started + connection.batch_interval;

                while first_is_data
                    && !force_flush
                    && batch.len() < connection.batch_max_frames
                    && batch_bytes < connection.batch_max_bytes
                {
                    if let Ok(priority) = priority_rx.try_recv() {
                        connection.queued_frames.fetch_sub(1, Ordering::AcqRel);
                        batch_bytes = batch_bytes.saturating_add(priority.frame.encoded_len());
                        batch.push((priority, None));
                        break;
                    }

                    let next_stream = match ready_rx.try_recv() {
                        Ok(stream_id) => Some(stream_id),
                        Err(mpsc::error::TryRecvError::Disconnected) => return Ok(()),
                        Err(mpsc::error::TryRecvError::Empty)
                            if connection.batch_interval.is_zero() =>
                        {
                            None
                        }
                        Err(mpsc::error::TryRecvError::Empty) => {
                            tokio::select! {
                                biased;
                                _ = cancel.cancelled() => return Ok(()),
                                Some(priority) = priority_rx.recv() => {
                                    connection.queued_frames.fetch_sub(1, Ordering::AcqRel);
                                    batch_bytes = batch_bytes.saturating_add(priority.frame.encoded_len());
                                    batch.push((priority, None));
                                    break;
                                }
                                stream_id = ready_rx.recv() => stream_id,
                                _ = tokio::time::sleep_until(deadline.into()) => None,
                            }
                        }
                    };
                    let Some(stream_id) = next_stream else {
                        break;
                    };
                    let Some(state) = connection
                        .streams
                        .get(&stream_id)
                        .map(|entry| entry.value().clone())
                    else {
                        continue;
                    };
                    let Some(next) = state.writer.lock().pending.pop_front() else {
                        connection.reschedule_stream(stream_id, &state)?;
                        continue;
                    };
                    connection.queued_frames.fetch_sub(1, Ordering::AcqRel);
                    let encoded_len = next.frame.encoded_len();
                    connection
                        .queued_bytes
                        .fetch_sub(encoded_len, Ordering::AcqRel);
                    response_mux::QUEUED_BYTES
                        .with_label_values(&["worker"])
                        .sub(encoded_len as i64);
                    if !batch.is_empty()
                        && (batch.len() + 1 > connection.batch_max_frames
                            || batch_bytes.saturating_add(encoded_len)
                                > connection.batch_max_bytes)
                    {
                        blocked_data = Some((next, Some((stream_id, state)), Instant::now()));
                        break;
                    }
                    let permit = if next.frame.kind == MuxFrameKind::Data {
                        let required = encoded_len.min(connection.max_connection_credits) as u32;
                        match connection
                            .connection_credits
                            .clone()
                            .try_acquire_many_owned(required)
                        {
                            Ok(permit) => Some(permit),
                            Err(tokio::sync::TryAcquireError::NoPermits) => {
                                blocked_data =
                                    Some((next, Some((stream_id, state)), Instant::now()));
                                break;
                            }
                            Err(tokio::sync::TryAcquireError::Closed) => {
                                return Err(anyhow!("response mux connection credit window closed"));
                            }
                        }
                    } else {
                        None
                    };
                    let urgent = next.frame.kind != MuxFrameKind::Data;
                    batch_bytes = batch_bytes.saturating_add(encoded_len);
                    batch.push((next, permit));
                    connection.reschedule_stream(stream_id, &state)?;
                    if urgent {
                        break;
                    }
                }

                for (command, _) in &batch {
                    let (header, payload) = command.frame.encode_parts()?;
                    write_buf.write(header);
                    write_buf.write(payload);
                }
                let observed_batch_wait = batching_started.elapsed();
                let data_bytes = batch
                    .iter()
                    .filter(|(command, _)| command.frame.kind == MuxFrameKind::Data)
                    .map(|(command, _)| command.frame.encoded_len() as u64)
                    .sum::<u64>();
                connection
                    .sent_data_bytes
                    .fetch_add(data_bytes, Ordering::AcqRel);
                let mut write_calls = 0_u64;
                let write_result: Result<()> = async {
                    let (_, calls) = write_buf.write_all_counted(&mut write_half).await?;
                    write_calls = calls;
                    Ok(())
                }
                .await;
                for (command, permit) in &mut batch {
                    if metrics_enabled {
                        response_mux::QUEUE_RESIDENCE_SECONDS
                            .with_label_values(&["worker"])
                            .observe(command.enqueued_at.elapsed().as_secs_f64());
                        if let Some(enqueued_at) = command.priority_enqueued_at {
                            response_mux::PRIORITY_QUEUE_RESIDENCE_SECONDS
                                .observe(enqueued_at.elapsed().as_secs_f64());
                        }
                        frame_counters.inc(command.frame.kind.metric_label());
                    }
                    if let Some(written) = command.written.take() {
                        let _ = written.send(
                            write_result
                                .as_ref()
                                .map(|_| ())
                                .map_err(|err| err.to_string()),
                        );
                    }
                    if let Some(permit) = permit.take() {
                        permit.forget();
                    }
                }
                write_result?;
                if metrics_enabled {
                    frames_per_write.observe(batch.len() as f64);
                    response_mux::BATCH_BYTES
                        .with_label_values(&["worker"])
                        .observe(batch_bytes as f64);
                    response_mux::BATCH_WAIT_SECONDS
                        .with_label_values(&["worker"])
                        .observe(observed_batch_wait.as_secs_f64());
                    response_mux::WRITE_CALLS_TOTAL
                        .with_label_values(&["worker"])
                        .inc_by(write_calls);
                }
            }
        }
        .await;
        if metrics_enabled {
            queue_depth.sub(reported_queue_depth);
        }

        if let Some(connection) = weak.upgrade() {
            connection.fail(
                &result
                    .err()
                    .map(|err| err.to_string())
                    .unwrap_or_else(|| "writer stopped".to_string()),
            );
        }
    }

    async fn reader_task(
        weak: Weak<Self>,
        mut reader: FramedRead<tokio::net::tcp::OwnedReadHalf, MuxCodec>,
        cancel: CancellationToken,
        mut reported_data_segments: Option<u64>,
    ) {
        let frame_counters = response_mux::FrameCounters::for_direction("frontend_to_worker");
        let window_updates = response_mux::WINDOW_UPDATES_TOTAL
            .with_label_values(&["frontend_to_worker"])
            .clone();
        let connection_window_updates = response_mux::WINDOW_UPDATES_TOTAL
            .with_label_values(&["connection_frontend_to_worker"])
            .clone();
        let metrics_enabled = per_frame_metrics_enabled();
        let mut packet_tick = tokio::time::interval(Duration::from_millis(100));
        packet_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let result: Result<()> = async {
            loop {
                let message = tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = packet_tick.tick(), if reported_data_segments.is_some() => {
                        if let Some(current) = crate::pipeline::network::tcp::tcp_data_segments_out(reader.get_ref().as_ref()) {
                            let previous = reported_data_segments.replace(current).unwrap_or(current);
                            response_mux::DATA_SEGMENTS_TOTAL
                                .with_label_values(&["mux", "worker"])
                                .inc_by(current.saturating_sub(previous));
                        }
                        continue;
                    }
                    message = reader.next() => match message {
                        Some(message) => message.map_err(|err| anyhow!(err.to_string()))?,
                        None => return Err(anyhow!("frontend closed response mux connection")),
                    },
                };
                let frame = message;
                if metrics_enabled {
                    frame_counters.inc(frame.kind.metric_label());
                }
                let connection = weak
                    .upgrade()
                    .ok_or_else(|| anyhow!("response mux connection dropped"))?;
                if frame.kind == MuxFrameKind::ConnectionAck {
                    let offset = frame.connection_ack_offset()?;
                    if offset > 0 {
                        connection.acknowledge_connection_credits(offset)?;
                        if metrics_enabled {
                            connection_window_updates.inc();
                        }
                    }
                    continue;
                }
                let Some(state) = connection.streams.get(&frame.stream_id) else {
                    if frame.kind != MuxFrameKind::Reset {
                        connection.try_send_priority_command(WriterCommand::priority(
                            MuxFrame::empty(MuxFrameKind::Reset, frame.stream_id),
                            None,
                        ));
                    }
                    continue;
                };
                match frame.kind {
                    MuxFrameKind::WindowUpdate => {
                        let credits = frame.window_credits()? as usize;
                        if credits == 0 || credits > state.max_credits {
                            drop(state);
                            connection.try_send_priority_command(WriterCommand::priority(
                                MuxFrame::empty(MuxFrameKind::Reset, frame.stream_id),
                                None,
                            ));
                            connection.remove_stream(
                                frame.stream_id,
                                "invalid response mux window update",
                                true,
                            );
                            continue;
                        }
                        state.replenish_credits(credits);
                        if metrics_enabled {
                            window_updates.inc();
                        }
                    }
                    MuxFrameKind::Stop => state.context.stop(),
                    MuxFrameKind::Kill | MuxFrameKind::Reset => {
                        drop(state);
                        connection.remove_stream(
                            frame.stream_id,
                            "frontend closed response mux stream",
                            true,
                        );
                    }
                    _ => {
                        drop(state);
                        connection.try_send_priority_command(WriterCommand::priority(
                            MuxFrame::empty(MuxFrameKind::Reset, frame.stream_id),
                            None,
                        ));
                        connection.remove_stream(
                            frame.stream_id,
                            "invalid frontend response mux frame",
                            true,
                        );
                    }
                }
            }
            Ok(())
        }
        .await;

        if let Some(previous) = reported_data_segments
            && let Some(current) =
                crate::pipeline::network::tcp::tcp_data_segments_out(reader.get_ref().as_ref())
        {
            response_mux::DATA_SEGMENTS_TOTAL
                .with_label_values(&["mux", "worker"])
                .inc_by(current.saturating_sub(previous));
        }

        if let Some(connection) = weak.upgrade() {
            connection.fail(
                &result
                    .err()
                    .map(|err| err.to_string())
                    .unwrap_or_else(|| "reader stopped".to_string()),
            );
        }
    }
}

struct HostPool {
    address: String,
    frontend_server_id: Uuid,
    version: u8,
    connections: RwLock<Vec<Arc<MuxConnection>>>,
    connect_lock: tokio::sync::Mutex<()>,
    next_connection_id: AtomicU64,
    warming: AtomicBool,
    maintenance_started: AtomicBool,
    last_used: parking_lot::Mutex<Instant>,
    cancel: CancellationToken,
    config: PoolConfig,
}

impl HostPool {
    fn new(
        address: String,
        frontend_server_id: Uuid,
        version: u8,
        cancel: CancellationToken,
        config: PoolConfig,
    ) -> Arc<Self> {
        Arc::new(Self {
            address,
            frontend_server_id,
            version,
            connections: RwLock::new(Vec::new()),
            connect_lock: tokio::sync::Mutex::new(()),
            next_connection_id: AtomicU64::new(1),
            warming: AtomicBool::new(false),
            maintenance_started: AtomicBool::new(false),
            last_used: parking_lot::Mutex::new(Instant::now()),
            cancel,
            config,
        })
    }

    fn healthy_connections(&self) -> Vec<Arc<MuxConnection>> {
        self.connections
            .read()
            .iter()
            .filter(|connection| connection.is_healthy())
            .cloned()
            .collect()
    }

    async fn ensure_first(&self) -> Result<Arc<MuxConnection>> {
        let _guard = self.connect_lock.lock().await;
        if let Some(connection) = self.healthy_connections().first().cloned() {
            return Ok(connection);
        }
        self.connect_new().await
    }

    async fn connect_additional(&self) -> Result<Arc<MuxConnection>> {
        let _guard = self.connect_lock.lock().await;
        if self.healthy_connections().len() >= self.config.pool_size {
            return self
                .healthy_connections()
                .first()
                .cloned()
                .ok_or_else(|| anyhow!("response mux host pool has no healthy connection"));
        }
        self.connect_new().await
    }

    async fn connect_new(&self) -> Result<Arc<MuxConnection>> {
        let replacing_failed_connection = self
            .connections
            .read()
            .iter()
            .any(|connection| !connection.is_healthy());
        let id = self.next_connection_id.fetch_add(1, Ordering::Relaxed);
        let connection = MuxConnection::connect(
            id,
            &self.address,
            self.frontend_server_id,
            self.version,
            self.cancel.clone(),
            self.config,
        )
        .await?;
        let mut connections = self.connections.write();
        connections.retain(|candidate| candidate.is_healthy());
        connections.push(connection.clone());
        if replacing_failed_connection {
            response_mux::RECONNECTS_TOTAL
                .with_label_values(&["worker"])
                .inc();
        }
        Ok(connection)
    }

    fn start_maintenance(self: &Arc<Self>) {
        if self.maintenance_started.swap(true, Ordering::AcqRel) {
            return;
        }
        let host = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                if host.cancel.is_cancelled() {
                    break;
                }
                if host.healthy_connections().len() < host.config.pool_size {
                    host.warm();
                }
            }
        });
    }

    fn warm(self: &Arc<Self>) {
        if self.warming.swap(true, Ordering::AcqRel) {
            return;
        }
        let host = Arc::clone(self);
        tokio::spawn(async move {
            while host.healthy_connections().len() < host.config.pool_size
                && !host.cancel.is_cancelled()
            {
                if let Err(err) = host.connect_additional().await {
                    tracing::warn!(address = %host.address, %err, "failed to warm response mux pool");
                    break;
                }
            }
            host.warming.store(false, Ordering::Release);
        });
    }

    async fn connection(self: &Arc<Self>) -> Result<Arc<MuxConnection>> {
        *self.last_used.lock() = Instant::now();
        let mut healthy = self.healthy_connections();
        if healthy.is_empty() {
            healthy.push(self.ensure_first().await?);
        }
        self.start_maintenance();
        self.warm();
        let index = healthy
            .iter()
            .enumerate()
            .min_by_key(|(_, connection)| {
                (
                    connection.active_streams.load(Ordering::Acquire),
                    connection.queued_bytes.load(Ordering::Acquire),
                )
            })
            .map(|(index, _)| index)
            .expect("healthy response mux connection list is non-empty");
        Ok(healthy.swap_remove(index))
    }

    fn is_idle(&self) -> bool {
        self.healthy_connections()
            .iter()
            .all(|connection| connection.active_streams.load(Ordering::Acquire) == 0)
            && self.last_used.lock().elapsed() >= self.config.idle_ttl
    }
}

pub struct ResponseMuxClientPool {
    hosts: DashMap<HostKey, Arc<HostPool>>,
    cancel: CancellationToken,
    enabled: bool,
    config: PoolConfig,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct HostKey {
    address: String,
    frontend_server_id: Uuid,
    version: u8,
}

impl ResponseMuxClientPool {
    pub fn new(cancel: CancellationToken, runtime_config: ResponseMuxConfig) -> Arc<Self> {
        response_mux::CONFIGURED_BATCH_INTERVAL_MS
            .with_label_values(&["worker"])
            .set(runtime_config.batch_interval.as_millis() as i64);
        let pool = Arc::new(Self {
            hosts: DashMap::new(),
            cancel,
            enabled: runtime_config.enabled,
            config: PoolConfig::from_runtime(runtime_config),
        });
        Self::start_cleanup(&pool);
        pool
    }

    #[cfg(test)]
    fn new_for_test_with_connection_window(
        cancel: CancellationToken,
        pool_size: usize,
        writer_queue: usize,
        initial_window: usize,
        connection_window: usize,
        idle_ttl: Duration,
        connect_timeout: Duration,
    ) -> Arc<Self> {
        let pool = Arc::new(Self {
            hosts: DashMap::new(),
            cancel,
            enabled: true,
            config: PoolConfig {
                pool_size: pool_size.max(1),
                writer_queue: writer_queue.max(1),
                stream_writer_queue: RESPONSE_MUX_STREAM_WRITER_QUEUE,
                initial_window: initial_window.max(1),
                connection_window: connection_window.max(1),
                batch_interval: Duration::ZERO,
                batch_max_bytes: 65_536,
                batch_max_frames: 64,
                packet_metrics: false,
                idle_ttl,
                connect_timeout,
            },
        });
        Self::start_cleanup(&pool);
        pool
    }

    fn start_cleanup(pool: &Arc<Self>) {
        let weak = Arc::downgrade(pool);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                let Some(pool) = weak.upgrade() else {
                    break;
                };
                if pool.cancel.is_cancelled() {
                    break;
                }
                pool.hosts.retain(|_, host| {
                    let retain = !host.is_idle();
                    if !retain {
                        host.cancel.cancel();
                    }
                    retain
                });
            }
        });
    }

    pub async fn create_response_stream(
        self: &Arc<Self>,
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
    ) -> Result<StreamSender> {
        if !self.enabled {
            anyhow::bail!("response mux connection info received while mux mode is disabled");
        }
        let info = ResponseMuxConnectionInfo::try_from(info)
            .context("tcp-response-mux-connection-info-error")?;
        if info.version != RESPONSE_MUX_VERSION {
            anyhow::bail!(
                "unsupported response mux version {}; expected {}",
                info.version,
                RESPONSE_MUX_VERSION
            );
        }
        if info.context != context.id() {
            return Err(anyhow!(
                "response mux context mismatch: expected {}, got {}",
                context.id(),
                info.context
            ));
        }
        let stream_id = info.stream_id;
        let host_key = HostKey {
            address: info.address.clone(),
            frontend_server_id: info.frontend_server_id,
            version: info.version,
        };
        let host = self
            .hosts
            .entry(host_key)
            .or_insert_with(|| {
                HostPool::new(
                    info.address.clone(),
                    info.frontend_server_id,
                    info.version,
                    self.cancel.child_token(),
                    self.config,
                )
            })
            .clone();
        let connection = host.connection().await?;
        let state = Arc::new(WorkerStreamState {
            context,
            credits: Arc::new(Semaphore::new(self.config.initial_window)),
            max_credits: self.config.initial_window,
            writer_slots: Arc::new(Semaphore::new(self.config.stream_writer_queue)),
            writer: Mutex::new(StreamWriterState::default()),
            closed: AtomicBool::new(false),
            close_token: CancellationToken::new(),
        });
        match connection.streams.entry(stream_id) {
            Entry::Vacant(entry) => {
                entry.insert(state.clone());
            }
            Entry::Occupied(_) => {
                return Err(anyhow!("duplicate response mux stream id {stream_id}"));
            }
        }
        connection.active_streams.fetch_add(1, Ordering::AcqRel);
        response_mux::ACTIVE_STREAMS
            .with_label_values(&["worker"])
            .inc();
        let sender = StreamSender::multiplexed(Arc::new(MuxResponseStreamSender {
            stream_id,
            connection,
            state,
            finished: AtomicBool::new(false),
        }));
        Ok(sender)
    }

    #[cfg(test)]
    fn host_for_address(&self, address: &str) -> Option<Arc<HostPool>> {
        self.hosts
            .iter()
            .find(|entry| entry.value().address == address)
            .map(|entry| entry.value().clone())
    }

    #[cfg(test)]
    pub(crate) fn healthy_connection_count(&self, address: &str) -> usize {
        self.host_for_address(address)
            .map(|host| host.healthy_connections().len())
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn stream_connection_id(&self, address: &str, stream_id: Uuid) -> Option<u64> {
        self.host_for_address(address).and_then(|host| {
            host.connections
                .read()
                .iter()
                .find(|connection| connection.streams.contains_key(&stream_id))
                .map(|connection| connection.id)
        })
    }

    #[cfg(test)]
    pub(crate) fn fail_connection(&self, address: &str, connection_id: u64) {
        if let Some(host) = self.host_for_address(address)
            && let Some(connection) = host
                .connections
                .read()
                .iter()
                .find(|connection| connection.id == connection_id)
        {
            connection.fail("test-injected physical connection failure");
        }
    }
}

struct MuxResponseStreamSender {
    stream_id: Uuid,
    connection: Arc<MuxConnection>,
    state: Arc<WorkerStreamState>,
    finished: AtomicBool,
}

impl MuxResponseStreamSender {
    async fn acquire_writer_permit(
        &self,
        state: &WorkerStreamState,
    ) -> Result<OwnedSemaphorePermit> {
        let slots = state.writer_slots.clone();
        match slots.try_acquire_owned() {
            Ok(permit) => Ok(permit),
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                let wait_start = Instant::now();
                let permit = state
                    .writer_slots
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| anyhow!("response mux stream closed during writer admission"))?;
                if per_frame_metrics_enabled() {
                    response_mux::WRITER_ADMISSION_STALL_SECONDS
                        .observe(wait_start.elapsed().as_secs_f64());
                }
                Ok(permit)
            }
            Err(tokio::sync::TryAcquireError::Closed) => {
                anyhow::bail!("response mux stream closed during writer admission")
            }
        }
    }

    async fn enqueue_ordered(
        &self,
        frame: MuxFrame,
        written: Option<oneshot::Sender<Result<(), String>>>,
    ) -> Result<()> {
        self.enqueue_ordered_on(&self.connection, &self.state, frame, written)
            .await
    }

    async fn enqueue_ordered_on(
        &self,
        connection: &Arc<MuxConnection>,
        state: &Arc<WorkerStreamState>,
        frame: MuxFrame,
        written: Option<oneshot::Sender<Result<(), String>>>,
    ) -> Result<()> {
        if !connection.is_healthy() {
            anyhow::bail!("response mux connection is unhealthy");
        }
        let writer_permit = self.acquire_writer_permit(state).await?;
        let required = frame.encoded_len().min(connection.max_queued_bytes) as u32;
        let queued_byte_permit = match connection
            .queued_byte_slots
            .clone()
            .try_acquire_many_owned(required)
        {
            Ok(permit) => permit,
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                let wait_start = Instant::now();
                let slots = connection.queued_byte_slots.clone();
                let permit = tokio::select! {
                    _ = state.close_token.cancelled() => {
                        anyhow::bail!("response mux stream closed during byte-queue admission")
                    }
                    permit = slots.acquire_many_owned(required) => permit.map_err(|_| {
                        anyhow!("response mux connection closed during byte-queue admission")
                    })?,
                };
                response_mux::QUEUED_BYTE_ADMISSION_STALL_SECONDS
                    .observe(wait_start.elapsed().as_secs_f64());
                permit
            }
            Err(tokio::sync::TryAcquireError::Closed) => {
                anyhow::bail!("response mux connection closed during byte-queue admission")
            }
        };
        connection.enqueue_stream_command(
            self.stream_id,
            state,
            WriterCommand::new(frame, written)
                .with_writer_permit(writer_permit)
                .with_queued_byte_permit(queued_byte_permit),
        )
    }

    async fn enqueue_priority_and_wait(&self, frame: MuxFrame) -> Result<()> {
        let (written_tx, written_rx) = oneshot::channel();
        self.connection
            .send_priority_command(WriterCommand::priority(frame, Some(written_tx)))
            .await?;
        written_rx
            .await
            .map_err(|_| anyhow!("response mux priority writer dropped acknowledgement"))?
            .map_err(anyhow::Error::msg)
    }

    fn remove_stream(&self) {
        self.connection
            .remove_stream(self.stream_id, "response mux stream completed", false);
    }
}

#[async_trait::async_trait]
impl MultiplexedStreamSender for MuxResponseStreamSender {
    async fn send_data(&self, data: bytes::Bytes) -> Result<()> {
        let encoded_len = super::MUX_HEADER_LEN.saturating_add(data.len());
        let required = encoded_len.min(self.state.max_credits) as u32;
        let permit = match self.state.credits.clone().try_acquire_many_owned(required) {
            Ok(permit) => permit,
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                let wait_start = Instant::now();
                let permit = self
                    .state
                    .credits
                    .clone()
                    .acquire_many_owned(required)
                    .await
                    .map_err(|_| anyhow!("response mux stream closed while waiting for credits"))?;
                if per_frame_metrics_enabled() {
                    response_mux::FLOW_CONTROL_STALL_SECONDS
                        .observe(wait_start.elapsed().as_secs_f64());
                }
                permit
            }
            Err(tokio::sync::TryAcquireError::Closed) => {
                anyhow::bail!("response mux stream closed while waiting for credits")
            }
        };
        let result = self
            .enqueue_ordered(
                MuxFrame::new(MuxFrameKind::Data, self.stream_id, data),
                None,
            )
            .await;
        if result.is_ok() {
            permit.forget();
        }
        result
    }

    async fn send_prologue(&self, error: Option<String>) -> Result<(), String> {
        let terminal_error = error.is_some();
        let payload =
            serde_json::to_vec(&ResponseStreamPrologue { error }).map_err(|err| err.to_string())?;
        let frame = MuxFrame::new(MuxFrameKind::Prologue, self.stream_id, payload.into());
        let result = self.enqueue_priority_and_wait(frame).await;
        if result.is_ok() && terminal_error {
            self.finished.store(true, Ordering::Release);
            self.remove_stream();
        }
        result.map_err(|err| err.to_string())
    }

    async fn finish(&self) -> Result<()> {
        if self.finished.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let (written_tx, written_rx) = oneshot::channel();
        let end = MuxFrame::empty(MuxFrameKind::End, self.stream_id);
        if let Err(err) = self.enqueue_ordered(end, Some(written_tx)).await {
            self.remove_stream();
            return Err(err).context("response mux writer stopped before end");
        }
        let result = written_rx
            .await
            .map_err(|_| anyhow!("response mux writer dropped end acknowledgement"))?
            .map_err(anyhow::Error::msg);
        self.remove_stream();
        result
    }
}

impl Drop for MuxResponseStreamSender {
    fn drop(&mut self) {
        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        response_mux::RESETS_TOTAL
            .with_label_values(&["worker", "publisher_drop"])
            .inc();
        self.connection
            .try_send_priority_command(WriterCommand::priority(
                MuxFrame::new(
                    MuxFrameKind::Reset,
                    self.stream_id,
                    bytes::Bytes::from_static(b"response sender dropped before finish"),
                ),
                None,
            ));
        self.remove_stream();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::network::tcp::mux::ResponseMuxConfig;
    use crate::{
        engine::AsyncEngineContextProvider,
        pipeline::{
            Context,
            network::{ResponseService, StreamOptions},
        },
    };
    use futures::{StreamExt as _, stream::FuturesUnordered};

    const TEST_STREAM_WINDOW: usize = 64;
    const TEST_STREAM_WINDOW_UPDATE: u32 = 32;
    const TEST_CONNECTION_WINDOW: usize = 256;

    fn worker_stream_state(initial_credits: usize, writer_slots: usize) -> Arc<WorkerStreamState> {
        let context = Context::new(());
        Arc::new(WorkerStreamState {
            context: context.context(),
            credits: Arc::new(Semaphore::new(initial_credits)),
            max_credits: TEST_STREAM_WINDOW,
            writer_slots: Arc::new(Semaphore::new(writer_slots)),
            writer: Mutex::new(StreamWriterState::default()),
            closed: AtomicBool::new(false),
            close_token: CancellationToken::new(),
        })
    }

    #[test]
    fn credit_replenishment_never_exceeds_the_fixed_maximum() {
        let state = worker_stream_state(TEST_STREAM_WINDOW, RESPONSE_MUX_STREAM_WRITER_QUEUE);
        state
            .credits
            .clone()
            .try_acquire_many_owned((TEST_STREAM_WINDOW - 16) as u32)
            .expect("initial credits should be available")
            .forget();

        assert_eq!(state.credits.available_permits(), 16);
        assert_eq!(
            state.replenish_credits(TEST_STREAM_WINDOW_UPDATE as usize),
            TEST_STREAM_WINDOW_UPDATE as usize
        );
        assert_eq!(
            state.credits.available_permits(),
            16 + TEST_STREAM_WINDOW_UPDATE as usize
        );
        assert_eq!(
            state.replenish_credits(TEST_STREAM_WINDOW),
            TEST_STREAM_WINDOW - 16 - TEST_STREAM_WINDOW_UPDATE as usize
        );
        assert_eq!(state.credits.available_permits(), TEST_STREAM_WINDOW);
        assert_eq!(
            state.replenish_credits(TEST_STREAM_WINDOW_UPDATE as usize),
            0
        );
        assert_eq!(state.credits.available_permits(), TEST_STREAM_WINDOW);
    }

    #[test]
    fn cumulative_connection_ack_replenishes_credits_without_exceeding_the_window() {
        let (priority_tx, _priority_rx) = mpsc::channel(1);
        let (ready_tx, _ready_rx) = mpsc::unbounded_channel();
        let remaining = 16;
        let consumed = TEST_CONNECTION_WINDOW - remaining;
        let connection = MuxConnection {
            id: 1,
            cancel: CancellationToken::new(),
            priority_tx,
            ready_tx,
            streams: DashMap::new(),
            healthy: AtomicBool::new(true),
            active_streams: AtomicUsize::new(0),
            queued_frames: AtomicUsize::new(0),
            queued_bytes: AtomicUsize::new(0),
            max_queued_bytes: TEST_CONNECTION_WINDOW,
            queued_byte_slots: Arc::new(Semaphore::new(TEST_CONNECTION_WINDOW)),
            connection_credits: Arc::new(Semaphore::new(TEST_CONNECTION_WINDOW)),
            max_connection_credits: TEST_CONNECTION_WINDOW,
            sent_data_bytes: AtomicU64::new(consumed as u64),
            acknowledged_data_bytes: AtomicU64::new(0),
            batch_interval: Duration::ZERO,
            batch_max_bytes: 65_536,
            batch_max_frames: 64,
        };
        connection
            .connection_credits
            .clone()
            .try_acquire_many_owned(consumed as u32)
            .unwrap()
            .forget();

        assert_eq!(connection.connection_credits.available_permits(), remaining);
        connection.acknowledge_connection_credits(128).unwrap();
        assert_eq!(
            connection.connection_credits.available_permits(),
            remaining + 128
        );
        connection
            .acknowledge_connection_credits(consumed as u64)
            .unwrap();
        assert_eq!(
            connection.connection_credits.available_permits(),
            TEST_CONNECTION_WINDOW
        );
        connection
            .acknowledge_connection_credits(consumed as u64)
            .unwrap();
        assert_eq!(
            connection.connection_credits.available_permits(),
            TEST_CONNECTION_WINDOW
        );
        assert!(connection.acknowledge_connection_credits(127).is_err());
        assert!(
            connection
                .acknowledge_connection_credits(consumed as u64 + 1)
                .is_err()
        );
    }

    #[tokio::test]
    async fn closing_a_stream_wakes_credit_and_writer_admission_waiters() {
        let state = worker_stream_state(0, 0);
        let (written_tx, written_rx) = oneshot::channel();
        state.writer.lock().pending.push_back(WriterCommand::new(
            MuxFrame::empty(MuxFrameKind::End, Uuid::new_v4()),
            Some(written_tx),
        ));

        let (priority_tx, _priority_rx) = mpsc::channel(1);
        let (ready_tx, _ready_rx) = mpsc::unbounded_channel();
        let connection = MuxConnection {
            id: 1,
            cancel: CancellationToken::new(),
            priority_tx,
            ready_tx,
            streams: DashMap::new(),
            healthy: AtomicBool::new(true),
            active_streams: AtomicUsize::new(0),
            queued_frames: AtomicUsize::new(1),
            queued_bytes: AtomicUsize::new(super::super::MUX_HEADER_LEN),
            max_queued_bytes: TEST_CONNECTION_WINDOW,
            queued_byte_slots: Arc::new(Semaphore::new(TEST_CONNECTION_WINDOW)),
            connection_credits: Arc::new(Semaphore::new(TEST_CONNECTION_WINDOW)),
            max_connection_credits: TEST_CONNECTION_WINDOW,
            sent_data_bytes: AtomicU64::new(0),
            acknowledged_data_bytes: AtomicU64::new(0),
            batch_interval: Duration::ZERO,
            batch_max_bytes: 65_536,
            batch_max_frames: 64,
        };

        let credit_waiter = tokio::spawn({
            let credits = state.credits.clone();
            async move { credits.acquire_owned().await }
        });
        let writer_waiter = tokio::spawn({
            let writer_slots = state.writer_slots.clone();
            async move { writer_slots.acquire_owned().await }
        });
        tokio::task::yield_now().await;

        connection.close_stream_state(&state, "test stream closed");

        assert!(credit_waiter.await.unwrap().is_err());
        assert!(writer_waiter.await.unwrap().is_err());
        assert_eq!(connection.queued_frames.load(Ordering::Acquire), 0);
        assert_eq!(written_rx.await.unwrap().unwrap_err(), "test stream closed");
        assert_eq!(state.replenish_credits(1_024), 0);
        assert_eq!(state.credits.available_permits(), 0);
    }

    #[tokio::test]
    async fn end_bypasses_exhausted_connection_credit_at_batch_byte_limit() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = TcpStream::connect(address).await.unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let (_, write_half) = client.into_split();
        let (server_read, _) = server.into_split();

        let stream_id = Uuid::new_v4();
        let (end_written_tx, end_written_rx) = oneshot::channel();
        let state = worker_stream_state(TEST_STREAM_WINDOW, RESPONSE_MUX_STREAM_WRITER_QUEUE);
        {
            let mut writer = state.writer.lock();
            writer.pending.push_back(WriterCommand::new(
                MuxFrame::new(
                    MuxFrameKind::Data,
                    stream_id,
                    bytes::Bytes::from(vec![b'x'; 40]),
                ),
                None,
            ));
            writer.pending.push_back(WriterCommand::new(
                MuxFrame::empty(MuxFrameKind::End, stream_id),
                Some(end_written_tx),
            ));
            writer.scheduled = true;
        }

        let (priority_tx, priority_rx) = mpsc::channel(1);
        let (ready_tx, ready_rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let connection = Arc::new(MuxConnection {
            id: 1,
            cancel: cancel.clone(),
            priority_tx,
            ready_tx: ready_tx.clone(),
            streams: DashMap::new(),
            healthy: AtomicBool::new(true),
            active_streams: AtomicUsize::new(1),
            queued_frames: AtomicUsize::new(2),
            queued_bytes: AtomicUsize::new(88),
            max_queued_bytes: TEST_CONNECTION_WINDOW,
            queued_byte_slots: Arc::new(Semaphore::new(TEST_CONNECTION_WINDOW)),
            connection_credits: Arc::new(Semaphore::new(64)),
            max_connection_credits: 64,
            sent_data_bytes: AtomicU64::new(0),
            acknowledged_data_bytes: AtomicU64::new(0),
            batch_interval: Duration::ZERO,
            batch_max_bytes: 64,
            batch_max_frames: 64,
        });
        connection.streams.insert(stream_id, state);
        ready_tx.send(stream_id).unwrap();

        let writer_task = tokio::spawn(MuxConnection::writer_task(
            Arc::downgrade(&connection),
            write_half,
            priority_rx,
            ready_rx,
            cancel,
        ));
        let reader_task = tokio::spawn(async move {
            let mut reader = FramedRead::new(server_read, MuxCodec::default());
            let data = reader.next().await.unwrap().unwrap();
            let end = reader.next().await.unwrap().unwrap();
            (data.kind, end.kind)
        });

        tokio::time::timeout(Duration::from_millis(200), end_written_rx)
            .await
            .expect("End waited for exhausted Data credits")
            .unwrap()
            .unwrap();
        assert_eq!(
            reader_task.await.unwrap(),
            (MuxFrameKind::Data, MuxFrameKind::End)
        );
        writer_task.abort();
    }

    fn integration_config() -> ResponseMuxConfig {
        ResponseMuxConfig {
            enabled: true,
            packet_metrics: false,
            batch_interval: Duration::from_millis(5),
            batch_max_bytes: 65_536,
            batch_max_frames: 64,
            stream_window_bytes: 262_144,
            connection_window_bytes: 262_144,
        }
    }

    async fn open_mux_stream(
        server: Arc<crate::pipeline::network::tcp::server::TcpStreamServer>,
        pool: Arc<ResponseMuxClientPool>,
    ) -> (
        Uuid,
        Context<()>,
        StreamSender,
        crate::pipeline::network::StreamReceiver,
    ) {
        let context = Context::new(());
        let pending = server
            .register(
                StreamOptions::builder()
                    .context(context.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .send_buffer_count(8)
                    .build()
                    .unwrap(),
            )
            .await
            .recv_stream
            .unwrap();
        let (info, provider) = pending.into_parts();
        let stream_id = ResponseMuxConnectionInfo::try_from(info.clone())
            .unwrap()
            .stream_id;
        let mut sender = pool
            .create_response_stream(context.context(), info)
            .await
            .unwrap();
        sender.send_prologue(None).await.unwrap();
        let receiver = provider.await.unwrap().unwrap();
        (stream_id, context, sender, receiver)
    }

    async fn mux_address(
        server: Arc<crate::pipeline::network::tcp::server::TcpStreamServer>,
    ) -> String {
        let probe = server
            .register(
                StreamOptions::builder()
                    .context(Context::new(()).context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let info = probe.recv_stream.as_ref().unwrap().connection_info.clone();
        drop(probe);
        ResponseMuxConnectionInfo::try_from(info).unwrap().address
    }

    #[tokio::test]
    async fn disabled_worker_rejects_mux_connection_info() {
        let mut config = integration_config();
        config.enabled = false;
        let cancel = CancellationToken::new();
        let pool = ResponseMuxClientPool::new(cancel.clone(), config);
        let context = Context::new(());
        let info = ResponseMuxConnectionInfo {
            address: "127.0.0.1:1".to_string(),
            frontend_server_id: Uuid::new_v4(),
            stream_id: Uuid::new_v4(),
            context: context.context().id().to_string(),
            version: RESPONSE_MUX_VERSION,
        };

        let error = pool
            .create_response_stream(context.context(), info.into())
            .await
            .err()
            .expect("disabled mux mode must reject mux connection info");
        assert!(error.to_string().contains("mux mode is disabled"));
        cancel.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_credit_stall_does_not_block_another_stream() {
        let config = ResponseMuxConfig {
            enabled: true,
            packet_metrics: false,
            batch_interval: Duration::ZERO,
            batch_max_bytes: 65_536,
            batch_max_frames: 64,
            stream_window_bytes: 64,
            connection_window_bytes: 512,
        };
        let server = crate::pipeline::network::tcp::server::TcpStreamServer::new_mux_for_test(
            crate::pipeline::network::tcp::server::ServerOptions::default(),
            config,
        )
        .await
        .unwrap();
        let cancel = CancellationToken::new();
        let pool = ResponseMuxClientPool::new_for_test_with_connection_window(
            cancel.clone(),
            1,
            64,
            64,
            512,
            Duration::from_secs(60),
            Duration::from_secs(5),
        );
        let (_, _context_a, sender_a, mut receiver_a) =
            open_mux_stream(server.clone(), pool.clone()).await;
        let (_, _context_b, sender_b, mut receiver_b) = open_mux_stream(server, pool.clone()).await;

        let full_window_payload = bytes::Bytes::from(vec![b'a'; 40]);
        sender_a.send(full_window_payload.clone()).await.unwrap();
        let blocked_send = sender_a.send(full_window_payload.clone());
        tokio::pin!(blocked_send);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut blocked_send)
                .await
                .is_err(),
            "second frame should wait for stream-local credits"
        );

        sender_b
            .send(bytes::Bytes::from_static(b"healthy"))
            .await
            .unwrap();
        sender_b.finish().await.unwrap();
        assert_eq!(
            receiver_b.recv().await.unwrap(),
            bytes::Bytes::from_static(b"healthy")
        );
        assert!(receiver_b.recv().await.is_none());

        assert_eq!(receiver_a.recv().await.unwrap(), full_window_payload);
        tokio::time::timeout(Duration::from_secs(1), &mut blocked_send)
            .await
            .expect("stream credit update did not unblock producer")
            .unwrap();
        sender_a.finish().await.unwrap();
        assert_eq!(receiver_a.recv().await.unwrap(), full_window_payload);
        assert!(receiver_a.recv().await.is_none());
        cancel.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn connection_failure_is_scoped_and_pool_reconnects() {
        let config = integration_config();
        let server = crate::pipeline::network::tcp::server::TcpStreamServer::new_mux_for_test(
            crate::pipeline::network::tcp::server::ServerOptions::default(),
            config,
        )
        .await
        .unwrap();
        let address = mux_address(server.clone()).await;
        let cancel = CancellationToken::new();
        let pool = ResponseMuxClientPool::new(cancel.clone(), config);

        let mut streams = vec![open_mux_stream(server.clone(), pool.clone()).await];
        tokio::time::timeout(Duration::from_secs(5), async {
            while pool.healthy_connection_count(&address) != RESPONSE_MUX_POOL_SIZE {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("response mux pool did not warm to four connections");
        for _ in 1..8 {
            streams.push(open_mux_stream(server.clone(), pool.clone()).await);
        }

        let target_connection = pool
            .stream_connection_id(&address, streams[0].0)
            .expect("stream must be assigned to a connection");
        let assignments = streams
            .iter()
            .map(|stream| {
                pool.stream_connection_id(&address, stream.0)
                    .expect("stream must be assigned to a connection")
            })
            .collect::<Vec<_>>();
        assert!(assignments.iter().any(|id| *id != target_connection));

        pool.fail_connection(&address, target_connection);
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let correctly_scoped =
                    streams
                        .iter()
                        .zip(&assignments)
                        .all(|((_, context, _, _), connection_id)| {
                            context.context().is_killed() == (*connection_id == target_connection)
                        });
                if correctly_scoped {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("connection failure did not remain scoped to assigned streams");

        let (_, _, replacement_sender, mut replacement_receiver) =
            open_mux_stream(server, pool.clone()).await;
        replacement_sender
            .send(bytes::Bytes::from_static(b"replacement"))
            .await
            .unwrap();
        replacement_sender.finish().await.unwrap();
        assert_eq!(
            replacement_receiver.recv().await.unwrap(),
            bytes::Bytes::from_static(b"replacement")
        );
        tokio::time::timeout(Duration::from_secs(5), async {
            while pool.healthy_connection_count(&address) != RESPONSE_MUX_POOL_SIZE {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("response mux pool did not reconnect to four connections");
        cancel.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn one_thousand_logical_streams_share_exactly_four_connections() {
        let config = integration_config();
        let server = crate::pipeline::network::tcp::server::TcpStreamServer::new_mux_for_test(
            crate::pipeline::network::tcp::server::ServerOptions::default(),
            config,
        )
        .await
        .unwrap();
        let cancel = CancellationToken::new();
        let pool = ResponseMuxClientPool::new(cancel.clone(), config);

        async fn round_trip(
            server: Arc<crate::pipeline::network::tcp::server::TcpStreamServer>,
            pool: Arc<ResponseMuxClientPool>,
            value: usize,
        ) -> String {
            let context = Context::new(());
            let pending = server
                .register(
                    StreamOptions::builder()
                        .context(context.context())
                        .enable_request_stream(false)
                        .enable_response_stream(true)
                        .send_buffer_count(8)
                        .build()
                        .unwrap(),
                )
                .await
                .recv_stream
                .unwrap();
            let (info, provider) = pending.into_parts();
            let mut sender = pool
                .create_response_stream(context.context(), info)
                .await
                .unwrap();
            sender.send_prologue(None).await.unwrap();
            let mut receiver = provider.await.unwrap().unwrap();
            let expected = format!("response-{value}");
            sender.send(expected.clone().into()).await.unwrap();
            sender.finish().await.unwrap();
            let actual = receiver.recv().await.unwrap();
            assert!(receiver.recv().await.is_none());
            String::from_utf8(actual.to_vec()).unwrap()
        }

        assert_eq!(
            round_trip(server.clone(), pool.clone(), 0).await,
            "response-0"
        );
        let address = mux_address(server.clone()).await;
        tokio::time::timeout(Duration::from_secs(5), async {
            while pool.healthy_connection_count(&address) != RESPONSE_MUX_POOL_SIZE {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("response mux pool did not warm to four connections");

        let mut tasks = FuturesUnordered::new();
        for value in 1..1_000 {
            tasks.push(round_trip(server.clone(), pool.clone(), value));
        }
        let mut completed = 1;
        while let Some(actual) = tasks.next().await {
            assert!(actual.starts_with("response-"));
            completed += 1;
        }
        assert_eq!(completed, 1_000);
        assert_eq!(
            pool.healthy_connection_count(&address),
            RESPONSE_MUX_POOL_SIZE
        );
        cancel.cancel();
    }
}
