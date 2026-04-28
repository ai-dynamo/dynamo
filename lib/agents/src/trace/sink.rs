// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::fs::{OpenOptions, create_dir_all};
use tokio::io::{AsyncWrite, AsyncWriteExt, BufWriter};
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use super::{bus, config::AgentTracePolicy, types::AgentTraceRecord};

static JSONL_WORKER_STARTED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug)]
pub(crate) struct JsonlSinkOptions {
    pub(crate) buffer_bytes: usize,
    pub(crate) flush_interval: Duration,
}

impl JsonlSinkOptions {
    pub(crate) fn from_policy(policy: &AgentTracePolicy) -> Self {
        Self {
            buffer_bytes: policy.jsonl_buffer_bytes.max(1),
            flush_interval: Duration::from_millis(policy.jsonl_flush_interval_ms.max(1)),
        }
    }
}

pub async fn spawn_jsonl_worker_with_shutdown(
    path: String,
    options: JsonlSinkOptions,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    if JSONL_WORKER_STARTED.load(Ordering::Acquire) {
        tracing::debug!(path, "Agent trace sink already started");
        return Ok(());
    }

    if let Some(parent) = std::path::Path::new(&path).parent()
        && !parent.as_os_str().is_empty()
    {
        create_dir_all(parent).await.with_context(|| {
            format!("creating agent trace parent directory {}", parent.display())
        })?;
    }

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .with_context(|| format!("opening agent trace jsonl sink at {path}"))?;

    if JSONL_WORKER_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        tracing::debug!(path, "Agent trace sink already started");
        return Ok(());
    }

    let mut rx = bus::subscribe();
    tokio::spawn(async move {
        let mut writer = BufWriter::with_capacity(options.buffer_bytes, file);
        let mut buffered_bytes = 0usize;
        let mut flush_tick = tokio::time::interval(options.flush_interval);
        flush_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    loop {
                        match rx.try_recv() {
                            Ok(rec) => {
                                match write_record(&mut writer, &rec).await {
                                    Some(n) => buffered_bytes = buffered_bytes.saturating_add(n),
                                    None => return,
                                }
                                if should_flush(buffered_bytes, options.buffer_bytes)
                                    && !flush_writer(&mut writer, &mut buffered_bytes).await
                                {
                                    return;
                                }
                            }
                            Err(TryRecvError::Lagged(n)) => {
                                tracing::warn!(dropped = n, "agent trace bus lagged during shutdown; dropped records");
                            }
                            Err(TryRecvError::Empty | TryRecvError::Closed) => {
                                let _ = flush_writer(&mut writer, &mut buffered_bytes).await;
                                return;
                            }
                        }
                    }
                }
                _ = flush_tick.tick() => {
                    if !flush_writer(&mut writer, &mut buffered_bytes).await {
                        break;
                    }
                }
                msg = rx.recv() => {
                    match msg {
                        Ok(rec) => {
                            match write_record(&mut writer, &rec).await {
                                Some(n) => buffered_bytes = buffered_bytes.saturating_add(n),
                                None => break,
                            }
                            if should_flush(buffered_bytes, options.buffer_bytes)
                                && !flush_writer(&mut writer, &mut buffered_bytes).await
                            {
                                break;
                            }
                        }
                        Err(RecvError::Lagged(n)) => {
                            tracing::warn!(dropped = n, "agent trace bus lagged; dropped records")
                        }
                        Err(RecvError::Closed) => break,
                    }
                }
            }
        }
    });

    tracing::info!(
        path,
        buffer_bytes = options.buffer_bytes,
        flush_interval_ms = options.flush_interval.as_millis(),
        "Agent trace async JSONL sink ready"
    );
    Ok(())
}

fn should_flush(buffered_bytes: usize, buffer_bytes: usize) -> bool {
    buffered_bytes >= buffer_bytes
}

async fn flush_writer<W>(writer: &mut W, buffered_bytes: &mut usize) -> bool
where
    W: AsyncWrite + Unpin,
{
    if *buffered_bytes == 0 {
        return true;
    }
    if let Err(e) = writer.flush().await {
        tracing::warn!("agent_trace: flush failed: {e}");
        return false;
    }
    *buffered_bytes = 0;
    true
}

async fn write_record<W>(writer: &mut W, rec: &AgentTraceRecord) -> Option<usize>
where
    W: AsyncWrite + Unpin,
{
    match serde_json::to_vec(rec) {
        Ok(mut bytes) => {
            bytes.push(b'\n');
            let bytes_len = bytes.len();
            if let Err(e) = writer.write_all(&bytes).await {
                tracing::warn!("agent_trace: write failed: {e}");
                return None;
            }
            Some(bytes_len)
        }
        Err(e) => {
            tracing::warn!("agent_trace: serialize failed: {e}");
            Some(0)
        }
    }
}
