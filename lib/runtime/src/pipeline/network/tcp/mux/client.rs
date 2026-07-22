// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side persistent multiplexed TCP response connection pool.

use std::{
    cmp::Reverse,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use dashmap::{DashMap, mapref::entry::Entry};
use futures::{SinkExt, StreamExt};
use parking_lot::RwLock;
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
    MuxCodec, MuxFrame, MuxFrameKind, RESPONSE_MUX_CONNECT_TIMEOUT_SECS,
    RESPONSE_MUX_CONNECTION_QUEUE_BYTES, RESPONSE_MUX_POOL_SIZE, RESPONSE_MUX_STREAM_WRITER_QUEUE,
    RESPONSE_MUX_VERSION, RESPONSE_MUX_WRITER_QUEUE, ResponseMuxConfig,
};
use crate::{
    engine::AsyncEngineContext,
    metrics::response_mux,
    pipeline::network::{
        ConnectionInfo, StreamSender, egress::tcp_client::TcpWriteBuffer,
        tcp::ResponseMuxConnectionInfo,
    },
};

#[derive(Clone, Copy)]
struct PoolConfig {
    pool_size: usize,
    writer_queue: usize,
    stream_writer_queue: usize,
    initial_window: usize,
    queued_bytes: usize,
    batch_interval: Duration,
    batch_max_bytes: usize,
    batch_max_frames: usize,
    packet_metrics: bool,
    connect_timeout: Duration,
}

impl PoolConfig {
    fn from_runtime(config: ResponseMuxConfig) -> Self {
        Self {
            pool_size: RESPONSE_MUX_POOL_SIZE,
            writer_queue: RESPONSE_MUX_WRITER_QUEUE,
            stream_writer_queue: RESPONSE_MUX_STREAM_WRITER_QUEUE,
            initial_window: config.stream_window_bytes,
            queued_bytes: RESPONSE_MUX_CONNECTION_QUEUE_BYTES,
            batch_interval: config.batch_interval,
            batch_max_bytes: config.batch_max_bytes,
            batch_max_frames: config.batch_max_frames,
            packet_metrics: config.packet_metrics,
            connect_timeout: Duration::from_secs(RESPONSE_MUX_CONNECT_TIMEOUT_SECS),
        }
    }
}

struct WorkerStreamState {
    context: Arc<dyn AsyncEngineContext>,
    cancellation_counter: Option<prometheus::IntCounter>,
    cancellation_recorded: AtomicBool,
    credits: Arc<Semaphore>,
    max_credits: usize,
    writer_slots: Arc<Semaphore>,
    closed: AtomicBool,
    close_token: CancellationToken,
}

impl WorkerStreamState {
    fn record_cancellation(&self) {
        if !self.cancellation_recorded.swap(true, Ordering::AcqRel)
            && let Some(counter) = &self.cancellation_counter
        {
            counter.inc();
        }
    }

    fn replenish_credits(&self, credits: usize) {
        if self.closed.load(Ordering::Acquire) || self.credits.is_closed() {
            return;
        }
        let available = self.credits.available_permits();
        let replenished = credits.min(self.max_credits.saturating_sub(available));
        if replenished > 0 {
            self.credits.add_permits(replenished);
        }
    }
}

struct UrgentCommand {
    frame: MuxFrame,
}

struct OrderedCommand {
    frame: MuxFrame,
    state: Arc<WorkerStreamState>,
    written: Option<oneshot::Sender<Result<(), String>>>,
    _writer_permit: OwnedSemaphorePermit,
    _queued_byte_permit: OwnedSemaphorePermit,
}

impl OrderedCommand {
    fn fail(mut self, reason: &str) {
        if let Some(written) = self.written.take() {
            let _ = written.send(Err(reason.to_string()));
        }
    }
}

enum WriteCommand {
    Urgent(UrgentCommand),
    Ordered(OrderedCommand),
}

impl WriteCommand {
    fn frame(&self) -> &MuxFrame {
        match self {
            Self::Urgent(command) => &command.frame,
            Self::Ordered(command) => &command.frame,
        }
    }

    fn fail(self, reason: &str) {
        if let Self::Ordered(command) = self {
            command.fail(reason);
        }
    }

    fn complete(&mut self, result: &Result<(), String>) {
        if let Self::Ordered(command) = self
            && let Some(written) = command.written.take()
        {
            let _ = written.send(result.clone());
        }
    }
}

struct MuxConnection {
    cancel: CancellationToken,
    urgent_tx: mpsc::Sender<UrgentCommand>,
    ordered_tx: mpsc::Sender<OrderedCommand>,
    streams: DashMap<Uuid, Arc<WorkerStreamState>>,
    healthy: AtomicBool,
    active_streams: AtomicUsize,
    max_queued_bytes: usize,
    queued_byte_slots: Arc<Semaphore>,
    batch_interval: Duration,
    batch_max_bytes: usize,
    batch_max_frames: usize,
}

impl MuxConnection {
    async fn connect(
        address: &str,
        frontend_server_id: Uuid,
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
        let mut reader = FramedRead::new(read_half, MuxCodec::default());
        let mut handshake_writer = FramedWrite::new(write_half, MuxCodec::default());
        handshake_writer
            .send(MuxFrame::connection_hello(
                RESPONSE_MUX_VERSION,
                frontend_server_id,
            ))
            .await
            .context("failed to send response mux connection hello")?;
        let ready = tokio::time::timeout(config.connect_timeout, reader.next())
            .await
            .map_err(|_| anyhow!("response mux connection-ready timeout from {address}"))?
            .ok_or_else(|| anyhow!("frontend closed before response mux connection ready"))??;
        if ready != MuxFrame::connection_ready() {
            anyhow::bail!("frontend returned invalid response mux connection ready");
        }
        let write_half = handshake_writer.into_inner();

        let (urgent_tx, urgent_rx) = mpsc::channel(config.writer_queue);
        let (ordered_tx, ordered_rx) = mpsc::channel(config.writer_queue);
        let cancel = cancel.child_token();
        let connection = Arc::new(Self {
            cancel: cancel.clone(),
            urgent_tx,
            ordered_tx,
            streams: DashMap::new(),
            healthy: AtomicBool::new(true),
            active_streams: AtomicUsize::new(0),
            max_queued_bytes: config.queued_bytes,
            queued_byte_slots: Arc::new(Semaphore::new(config.queued_bytes)),
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
            urgent_rx,
            ordered_rx,
            cancel.clone(),
        ));
        tokio::spawn(Self::reader_task(
            Arc::downgrade(&connection),
            reader,
            cancel,
            packet_baseline,
        ));
        Ok(connection)
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn fail(&self, reason: &str) {
        if !self.healthy.swap(false, Ordering::AcqRel) {
            return;
        }
        tracing::warn!(reason, "response mux connection failed");
        self.cancel.cancel();
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
        tracing::trace!(reason, "closing response mux stream state");
        state.credits.close();
        state.writer_slots.close();
        state.close_token.cancel();
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

    async fn send_urgent(&self, frame: MuxFrame) -> Result<()> {
        if !self.is_healthy() {
            anyhow::bail!("response mux connection is unhealthy");
        }
        self.urgent_tx
            .send(UrgentCommand { frame })
            .await
            .map_err(|_| anyhow!("response mux urgent writer stopped"))
    }

    fn try_send_urgent(&self, frame: MuxFrame) {
        if !self.is_healthy() {
            return;
        }
        if self.urgent_tx.try_send(UrgentCommand { frame }).is_err() {
            self.fail("response mux urgent writer queue is unavailable");
        }
    }

    async fn send_ordered(&self, command: OrderedCommand) -> Result<()> {
        if !self.is_healthy() || command.state.closed.load(Ordering::Acquire) {
            command.fail("response mux stream is closed");
            anyhow::bail!("response mux stream is closed");
        }
        self.ordered_tx.send(command).await.map_err(|err| {
            err.0.fail("response mux ordered writer stopped");
            anyhow!("response mux ordered writer stopped")
        })
    }

    fn command_is_writable(command: &OrderedCommand) -> bool {
        !command.state.closed.load(Ordering::Acquire)
    }

    async fn writer_task(
        weak: Weak<Self>,
        mut write_half: tokio::net::tcp::OwnedWriteHalf,
        mut urgent_rx: mpsc::Receiver<UrgentCommand>,
        mut ordered_rx: mpsc::Receiver<OrderedCommand>,
        cancel: CancellationToken,
    ) {
        let mut write_buf = TcpWriteBuffer::new();
        let frames_per_write = response_mux::FRAMES_PER_WRITE
            .with_label_values(&["worker"])
            .clone();
        let mut pending_urgent = None;
        let mut pending_ordered = None;
        let mut urgent_open = true;
        let mut ordered_open = true;

        let result: Result<()> = async {
            loop {
                let connection = weak
                    .upgrade()
                    .ok_or_else(|| anyhow!("response mux connection dropped"))?;
                let first = loop {
                    if let Some(command) = pending_urgent.take() {
                        break WriteCommand::Urgent(command);
                    }
                    if let Ok(command) = urgent_rx.try_recv() {
                        break WriteCommand::Urgent(command);
                    }
                    if let Some(command) = pending_ordered.take() {
                        if Self::command_is_writable(&command) {
                            break WriteCommand::Ordered(command);
                        }
                        command.fail("response mux stream closed before write");
                        continue;
                    }
                    if let Ok(command) = ordered_rx.try_recv() {
                        if Self::command_is_writable(&command) {
                            break WriteCommand::Ordered(command);
                        }
                        command.fail("response mux stream closed before write");
                        continue;
                    }
                    let next = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return Ok(()),
                        command = urgent_rx.recv(), if urgent_open => match command {
                            Some(command) => Some(WriteCommand::Urgent(command)),
                            None => { urgent_open = false; None }
                        },
                        command = ordered_rx.recv(), if ordered_open => match command {
                            Some(command) => Some(WriteCommand::Ordered(command)),
                            None => { ordered_open = false; None }
                        },
                        else => return Ok(()),
                    };
                    let Some(next) = next else {
                        continue;
                    };
                    match next {
                        WriteCommand::Ordered(command) if !Self::command_is_writable(&command) => {
                            command.fail("response mux stream closed before write");
                        }
                        next => break next,
                    }
                };

                let first_is_data = first.frame().kind == MuxFrameKind::Data;
                let mut batch = vec![first];
                let mut batch_bytes = batch[0].frame().encoded_len();
                if first_is_data {
                    let deadline = tokio::time::Instant::now() + connection.batch_interval;
                    loop {
                        if batch.len() >= connection.batch_max_frames
                            || batch_bytes >= connection.batch_max_bytes
                        {
                            break;
                        }
                        if let Ok(command) = urgent_rx.try_recv() {
                            pending_urgent = Some(command);
                            break;
                        }

                        let next = if let Some(command) = pending_ordered.take() {
                            Some(command)
                        } else if let Ok(command) = ordered_rx.try_recv() {
                            Some(command)
                        } else if connection.batch_interval.is_zero() {
                            None
                        } else {
                            tokio::select! {
                                biased;
                                _ = cancel.cancelled() => return Ok(()),
                                command = urgent_rx.recv(), if urgent_open => {
                                    match command {
                                        Some(command) => pending_urgent = Some(command),
                                        None => urgent_open = false,
                                    }
                                    None
                                }
                                _ = tokio::time::sleep_until(deadline) => None,
                                command = ordered_rx.recv(), if ordered_open => match command {
                                    Some(command) => Some(command),
                                    None => { ordered_open = false; None }
                                },
                            }
                        };
                        let Some(command) = next else {
                            break;
                        };
                        if !Self::command_is_writable(&command) {
                            command.fail("response mux stream closed before write");
                            continue;
                        }
                        let encoded_len = command.frame.encoded_len();
                        if batch_bytes.saturating_add(encoded_len) > connection.batch_max_bytes {
                            pending_ordered = Some(command);
                            break;
                        }
                        let is_data = command.frame.kind == MuxFrameKind::Data;
                        batch_bytes = batch_bytes.saturating_add(encoded_len);
                        batch.push(WriteCommand::Ordered(command));
                        if !is_data {
                            break;
                        }
                    }
                }

                for command in &batch {
                    let (header, payload) = command.frame().encode_parts()?;
                    write_buf.write(header);
                    write_buf.write(payload);
                }
                let write_result = write_buf.write_all_counted(&mut write_half).await;
                let write_calls = write_result
                    .as_ref()
                    .map(|(_, calls)| *calls)
                    .unwrap_or_default();
                let completion = write_result
                    .as_ref()
                    .map(|_| ())
                    .map_err(|err| err.to_string());
                for command in &mut batch {
                    command.complete(&completion);
                }
                response_mux::WRITE_CALLS_TOTAL
                    .with_label_values(&["worker"])
                    .inc_by(write_calls);
                frames_per_write.observe(batch.len() as f64);
                write_result?;
            }
        }
        .await;

        if let Some(command) = pending_ordered.take() {
            command.fail("response mux writer stopped");
        }
        while let Ok(command) = ordered_rx.try_recv() {
            command.fail("response mux writer stopped");
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
        let mut packet_tick = tokio::time::interval(Duration::from_millis(100));
        packet_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let result: Result<()> = async {
            loop {
                let frame = tokio::select! {
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
                let connection = weak
                    .upgrade()
                    .ok_or_else(|| anyhow!("response mux connection dropped"))?;
                if matches!(
                    frame.kind,
                    MuxFrameKind::ConnectionHello | MuxFrameKind::ConnectionReady
                ) {
                    anyhow::bail!("unexpected response mux connection frame after handshake");
                }
                let Some(state) = connection.streams.get(&frame.stream_id) else {
                    if frame.kind != MuxFrameKind::Reset {
                        connection.try_send_urgent(MuxFrame::empty(
                            MuxFrameKind::Reset,
                            frame.stream_id,
                        ));
                    }
                    continue;
                };
                match frame.kind {
                    MuxFrameKind::WindowUpdate => {
                        let credits = frame.window_credits()? as usize;
                        if credits == 0 || credits > state.max_credits {
                            drop(state);
                            connection.try_send_urgent(MuxFrame::empty(
                                MuxFrameKind::Reset,
                                frame.stream_id,
                            ));
                            connection.remove_stream(
                                frame.stream_id,
                                "invalid response mux window update",
                                true,
                            );
                            continue;
                        }
                        state.replenish_credits(credits);
                    }
                    MuxFrameKind::Stop => {
                        state.record_cancellation();
                        state.context.stop();
                    }
                    MuxFrameKind::Kill | MuxFrameKind::Reset => {
                        state.record_cancellation();
                        drop(state);
                        connection.remove_stream(
                            frame.stream_id,
                            "frontend closed response mux stream",
                            true,
                        );
                    }
                    _ => {
                        drop(state);
                        connection.try_send_urgent(MuxFrame::empty(
                            MuxFrameKind::Reset,
                            frame.stream_id,
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
    connections: RwLock<Vec<Arc<MuxConnection>>>,
    connect_lock: tokio::sync::Mutex<()>,
    warming: AtomicBool,
    cancel: CancellationToken,
    config: PoolConfig,
}

impl HostPool {
    fn new(
        address: String,
        frontend_server_id: Uuid,
        cancel: CancellationToken,
        config: PoolConfig,
    ) -> Arc<Self> {
        Arc::new(Self {
            address,
            frontend_server_id,
            connections: RwLock::new(Vec::new()),
            connect_lock: tokio::sync::Mutex::new(()),
            warming: AtomicBool::new(false),
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

    async fn connect_additional(&self) -> Result<()> {
        let _guard = self.connect_lock.lock().await;
        if self.healthy_connections().len() < self.config.pool_size {
            self.connect_new().await?;
        }
        Ok(())
    }

    async fn connect_new(&self) -> Result<Arc<MuxConnection>> {
        let replacing_failed = self
            .connections
            .read()
            .iter()
            .any(|connection| !connection.is_healthy());
        let connection = MuxConnection::connect(
            &self.address,
            self.frontend_server_id,
            self.cancel.clone(),
            self.config,
        )
        .await?;
        let mut connections = self.connections.write();
        connections.retain(|candidate| candidate.is_healthy());
        connections.push(connection.clone());
        if replacing_failed {
            response_mux::RECONNECTS_TOTAL
                .with_label_values(&["worker"])
                .inc();
        }
        Ok(connection)
    }

    fn warm(self: &Arc<Self>) {
        if self.warming.swap(true, Ordering::AcqRel) {
            return;
        }
        let host = self.clone();
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
        let mut healthy = self.healthy_connections();
        if healthy.is_empty() {
            healthy.push(self.ensure_first().await?);
        }
        if healthy.len() < self.config.pool_size {
            self.warm();
        }
        let index = healthy
            .iter()
            .enumerate()
            .min_by_key(|(_, connection)| {
                (
                    connection.active_streams.load(Ordering::Acquire),
                    Reverse(connection.queued_byte_slots.available_permits()),
                )
            })
            .map(|(index, _)| index)
            .ok_or_else(|| anyhow!("response mux host pool has no healthy connection"))?;
        Ok(healthy.swap_remove(index))
    }
}

pub struct ResponseMuxClientPool {
    hosts: DashMap<HostKey, Arc<HostPool>>,
    cancel: CancellationToken,
    config: PoolConfig,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct HostKey {
    address: String,
    frontend_server_id: Uuid,
}

impl ResponseMuxClientPool {
    pub fn new(cancel: CancellationToken, runtime_config: ResponseMuxConfig) -> Arc<Self> {
        response_mux::CONFIGURED_BATCH_INTERVAL_MS
            .with_label_values(&["worker"])
            .set(runtime_config.batch_interval.as_millis() as i64);
        let pool = Arc::new(Self {
            hosts: DashMap::new(),
            cancel,
            config: PoolConfig::from_runtime(runtime_config),
        });
        Self::start_maintenance(&pool);
        pool
    }

    #[cfg(test)]
    fn new_for_test(
        cancel: CancellationToken,
        runtime_config: ResponseMuxConfig,
        pool_size: usize,
        queued_bytes: usize,
    ) -> Arc<Self> {
        let mut config = PoolConfig::from_runtime(runtime_config);
        config.pool_size = pool_size.max(1);
        config.queued_bytes = queued_bytes.max(1);
        let pool = Arc::new(Self {
            hosts: DashMap::new(),
            cancel,
            config,
        });
        Self::start_maintenance(&pool);
        pool
    }

    fn start_maintenance(pool: &Arc<Self>) {
        let pool = Arc::downgrade(pool);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                let Some(pool) = pool.upgrade() else {
                    break;
                };
                if pool.cancel.is_cancelled() {
                    break;
                }
                let hosts = pool
                    .hosts
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect::<Vec<_>>();
                for host in hosts {
                    if host.healthy_connections().len() < host.config.pool_size {
                        host.warm();
                    }
                }
            }
        });
    }

    pub async fn create_response_stream(
        self: &Arc<Self>,
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
        cancellation_counter: Option<prometheus::IntCounter>,
    ) -> Result<StreamSender> {
        let info = ResponseMuxConnectionInfo::try_from(info)
            .context("tcp-response-mux-connection-info-error")?;
        if info.context != context.id() {
            anyhow::bail!(
                "response mux context mismatch: expected {}, got {}",
                context.id(),
                info.context
            );
        }
        let stream_id = info.stream_id;
        let host_key = HostKey {
            address: info.address.clone(),
            frontend_server_id: info.frontend_server_id,
        };
        let host = self
            .hosts
            .entry(host_key)
            .or_insert_with(|| {
                HostPool::new(
                    info.address,
                    info.frontend_server_id,
                    self.cancel.child_token(),
                    self.config,
                )
            })
            .clone();
        let connection = host.connection().await?;
        let state = Arc::new(WorkerStreamState {
            context,
            cancellation_counter,
            cancellation_recorded: AtomicBool::new(false),
            credits: Arc::new(Semaphore::new(self.config.initial_window)),
            max_credits: self.config.initial_window,
            writer_slots: Arc::new(Semaphore::new(self.config.stream_writer_queue)),
            closed: AtomicBool::new(false),
            close_token: CancellationToken::new(),
        });
        match connection.streams.entry(stream_id) {
            Entry::Vacant(entry) => {
                entry.insert(state.clone());
            }
            Entry::Occupied(_) => anyhow::bail!("duplicate response mux stream id {stream_id}"),
        }
        connection.active_streams.fetch_add(1, Ordering::AcqRel);
        response_mux::ACTIVE_STREAMS
            .with_label_values(&["worker"])
            .inc();
        Ok(StreamSender::multiplexed(MuxResponseStreamSender {
            stream_id,
            connection,
            state,
            prologue_sent: false,
            finished: false,
        }))
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
    pub(crate) fn stream_connection_id(&self, address: &str, stream_id: Uuid) -> Option<usize> {
        self.host_for_address(address).and_then(|host| {
            host.connections
                .read()
                .iter()
                .find(|connection| connection.streams.contains_key(&stream_id))
                .map(|connection| Arc::as_ptr(connection) as usize)
        })
    }

    #[cfg(test)]
    pub(crate) fn fail_connection(&self, address: &str, connection_id: usize) {
        if let Some(host) = self.host_for_address(address)
            && let Some(connection) = host
                .connections
                .read()
                .iter()
                .find(|connection| Arc::as_ptr(connection) as usize == connection_id)
        {
            connection.fail("test-injected physical connection failure");
        }
    }
}

pub(crate) struct MuxResponseStreamSender {
    stream_id: Uuid,
    connection: Arc<MuxConnection>,
    state: Arc<WorkerStreamState>,
    prologue_sent: bool,
    finished: bool,
}

impl MuxResponseStreamSender {
    async fn acquire_writer_permit(&self) -> Result<OwnedSemaphorePermit> {
        match self.state.writer_slots.clone().try_acquire_owned() {
            Ok(permit) => Ok(permit),
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                response_mux::STALLS_TOTAL
                    .with_label_values(&["writer_admission"])
                    .inc();
                self.state
                    .writer_slots
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| anyhow!("response mux stream closed during writer admission"))
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
        if !self.connection.is_healthy() {
            anyhow::bail!("response mux connection is unhealthy");
        }
        let writer_permit = self.acquire_writer_permit().await?;
        let required = frame.encoded_len().min(self.connection.max_queued_bytes) as u32;
        let queued_byte_permit = match self
            .connection
            .queued_byte_slots
            .clone()
            .try_acquire_many_owned(required)
        {
            Ok(permit) => permit,
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                response_mux::STALLS_TOTAL
                    .with_label_values(&["queued_byte_admission"])
                    .inc();
                let slots = self.connection.queued_byte_slots.clone();
                tokio::select! {
                    _ = self.state.close_token.cancelled() => {
                        anyhow::bail!("response mux stream closed during byte-queue admission")
                    }
                    permit = slots.acquire_many_owned(required) => permit.map_err(|_| {
                        anyhow!("response mux connection closed during byte-queue admission")
                    })?,
                }
            }
            Err(tokio::sync::TryAcquireError::Closed) => {
                anyhow::bail!("response mux connection closed during byte-queue admission")
            }
        };
        self.connection
            .send_ordered(OrderedCommand {
                frame,
                state: self.state.clone(),
                written,
                _writer_permit: writer_permit,
                _queued_byte_permit: queued_byte_permit,
            })
            .await
    }

    pub(crate) async fn send_data(&self, data: bytes::Bytes) -> Result<()> {
        if !self.prologue_sent || self.finished {
            anyhow::bail!("response mux data sent outside an active stream");
        }
        let encoded_len = super::MUX_HEADER_LEN.saturating_add(data.len());
        let required = encoded_len.min(self.state.max_credits) as u32;
        let permit = match self.state.credits.clone().try_acquire_many_owned(required) {
            Ok(permit) => permit,
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                response_mux::STALLS_TOTAL
                    .with_label_values(&["stream_credit"])
                    .inc();
                self.state
                    .credits
                    .clone()
                    .acquire_many_owned(required)
                    .await
                    .map_err(|_| anyhow!("response mux stream closed while waiting for credits"))?
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

    pub(crate) async fn send_prologue(&mut self, error: Option<String>) -> Result<(), String> {
        if self.prologue_sent || self.finished {
            return Err("response mux prologue already sent".to_string());
        }
        let terminal_error = error.is_some();
        let payload = error.map(bytes::Bytes::from).unwrap_or_default();
        self.connection
            .send_urgent(MuxFrame::new(
                MuxFrameKind::Prologue,
                self.stream_id,
                payload,
            ))
            .await
            .map_err(|err| err.to_string())?;
        self.prologue_sent = true;
        if terminal_error {
            self.finished = true;
            self.remove_stream();
        }
        Ok(())
    }

    pub(crate) async fn finish(mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        if !self.prologue_sent {
            anyhow::bail!("response mux stream finished before its prologue");
        }
        self.finished = true;
        let (written_tx, written_rx) = oneshot::channel();
        let result = self
            .enqueue_ordered(
                MuxFrame::empty(MuxFrameKind::End, self.stream_id),
                Some(written_tx),
            )
            .await;
        if let Err(err) = result {
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

    fn remove_stream(&self) {
        self.connection
            .remove_stream(self.stream_id, "response mux stream completed", false);
    }
}

impl Drop for MuxResponseStreamSender {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        response_mux::RESETS_TOTAL
            .with_label_values(&["worker", "publisher_drop"])
            .inc();
        self.connection
            .try_send_urgent(MuxFrame::empty(MuxFrameKind::Reset, self.stream_id));
        self.remove_stream();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::AsyncEngineContextProvider,
        pipeline::{
            Context,
            network::{ResponseService, StreamOptions},
        },
    };
    use futures::{StreamExt as _, stream::FuturesUnordered};

    fn integration_config() -> ResponseMuxConfig {
        ResponseMuxConfig {
            packet_metrics: false,
            batch_interval: Duration::from_millis(1),
            batch_max_bytes: 65_536,
            batch_max_frames: 64,
            stream_window_bytes: 262_144,
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
            .create_response_stream(context.context(), info, None)
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
    async fn legacy_response_connection_info_is_rejected() {
        use crate::pipeline::network::tcp::{StreamType, TcpStreamConnectionInfo};

        let cancel = CancellationToken::new();
        let pool = ResponseMuxClientPool::new(cancel.clone(), integration_config());
        let context = Context::new(());
        let info = TcpStreamConnectionInfo {
            address: "127.0.0.1:1".to_string(),
            context: context.context().id().to_string(),
            subject: Uuid::new_v4().to_string(),
            stream_type: StreamType::Response,
        };
        let error = pool
            .create_response_stream(context.context(), info.into(), None)
            .await
            .err()
            .expect("legacy response connection info must be rejected");
        assert!(
            error
                .to_string()
                .contains("tcp-response-mux-connection-info-error")
        );
        cancel.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn duplicate_prologue_is_rejected_and_data_order_is_preserved() {
        let config = integration_config();
        let server = crate::pipeline::network::tcp::server::TcpStreamServer::new_mux_for_test(
            crate::pipeline::network::tcp::server::ServerOptions::default(),
            config,
        )
        .await
        .unwrap();
        let cancel = CancellationToken::new();
        let pool = ResponseMuxClientPool::new(cancel.clone(), config);
        let (_, _, mut sender, mut receiver) = open_mux_stream(server, pool).await;
        assert!(sender.send_prologue(None).await.is_err());
        for value in 0_u16..128 {
            sender
                .send(bytes::Bytes::copy_from_slice(&value.to_be_bytes()))
                .await
                .unwrap();
        }
        sender.finish().await.unwrap();
        for expected in 0_u16..128 {
            let actual = receiver.recv().await.unwrap();
            assert_eq!(u16::from_be_bytes(actual[..].try_into().unwrap()), expected);
        }
        assert!(receiver.recv().await.is_none());
        cancel.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocked_stream_does_not_block_another_on_the_same_connection() {
        let config = ResponseMuxConfig {
            stream_window_bytes: 64,
            batch_interval: Duration::ZERO,
            ..integration_config()
        };
        let server = crate::pipeline::network::tcp::server::TcpStreamServer::new_mux_for_test(
            crate::pipeline::network::tcp::server::ServerOptions::default(),
            config,
        )
        .await
        .unwrap();
        let cancel = CancellationToken::new();
        let pool = ResponseMuxClientPool::new_for_test(cancel.clone(), config, 1, 512);
        let (_, _, sender_a, mut receiver_a) = open_mux_stream(server.clone(), pool.clone()).await;
        let (_, _, sender_b, mut receiver_b) = open_mux_stream(server, pool.clone()).await;

        let full_window_payload = bytes::Bytes::from(vec![b'a'; 43]);
        sender_a.send(full_window_payload.clone()).await.unwrap();
        {
            let blocked_send = sender_a.send(full_window_payload.clone());
            tokio::pin!(blocked_send);
            assert!(
                tokio::time::timeout(Duration::from_millis(20), &mut blocked_send)
                    .await
                    .is_err()
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
        }
        sender_a.finish().await.unwrap();
        assert_eq!(receiver_a.recv().await.unwrap(), full_window_payload);
        assert!(receiver_a.recv().await.is_none());
        cancel.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn connection_failure_is_scoped_and_pool_repairs_to_four() {
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

        let target = pool.stream_connection_id(&address, streams[0].0).unwrap();
        let assignments = streams
            .iter()
            .map(|stream| pool.stream_connection_id(&address, stream.0).unwrap())
            .collect::<Vec<_>>();
        assert!(assignments.iter().any(|connection| *connection != target));
        pool.fail_connection(&address, target);
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if streams
                    .iter()
                    .zip(&assignments)
                    .all(|((_, context, _, _), connection)| {
                        context.context().is_killed() == (*connection == target)
                    })
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("connection failure escaped its assigned streams");
        tokio::time::timeout(Duration::from_secs(5), async {
            while pool.healthy_connection_count(&address) != RESPONSE_MUX_POOL_SIZE {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("response mux pool did not repair to four connections");
        cancel.cancel();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn one_thousand_streams_share_exactly_four_connections() {
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
        ) -> usize {
            let (_, _, sender, mut receiver) = open_mux_stream(server, pool).await;
            sender.send(value.to_string().into()).await.unwrap();
            sender.finish().await.unwrap();
            let actual = receiver.recv().await.unwrap();
            assert!(receiver.recv().await.is_none());
            std::str::from_utf8(&actual).unwrap().parse().unwrap()
        }

        assert_eq!(round_trip(server.clone(), pool.clone(), 0).await, 0);
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
        while let Some(value) = tasks.next().await {
            assert!(value < 1_000);
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
