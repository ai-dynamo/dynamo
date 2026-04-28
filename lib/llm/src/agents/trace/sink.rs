// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tokio_util::sync::CancellationToken;

use crate::recorder::{JsonlFormat, Recorder, RecorderOptions};

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

    let recorder_shutdown = CancellationToken::new();
    let recorder: Recorder<AgentTraceRecord> = Recorder::new_with_options(
        recorder_shutdown.clone(),
        &path,
        RecorderOptions {
            buffer_bytes: options.buffer_bytes,
            flush_interval: Some(options.flush_interval),
            format: JsonlFormat::RawEvent,
            append: true,
            ..Default::default()
        },
    )
    .await
    .with_context(|| format!("opening agent trace jsonl sink at {path}"))?;

    if JSONL_WORKER_STARTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        tracing::debug!(path, "Agent trace sink already started");
        recorder.shutdown();
        return Ok(());
    }

    let mut rx = bus::subscribe();
    let tx = recorder.event_sender();
    tokio::spawn(async move {
        let _recorder = recorder;
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    loop {
                        match rx.try_recv() {
                            Ok(rec) => {
                                if tx.send(rec).await.is_err() {
                                    recorder_shutdown.cancel();
                                    return;
                                }
                            }
                            Err(TryRecvError::Lagged(n)) => {
                                tracing::warn!(dropped = n, "agent trace bus lagged during shutdown; dropped records");
                            }
                            Err(TryRecvError::Empty | TryRecvError::Closed) => {
                                recorder_shutdown.cancel();
                                return;
                            }
                        }
                    }
                }
                msg = rx.recv() => {
                    match msg {
                        Ok(rec) => {
                            if tx.send(rec).await.is_err() {
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
        recorder_shutdown.cancel();
    });

    tracing::info!(
        path,
        buffer_bytes = options.buffer_bytes,
        flush_interval_ms = options.flush_interval.as_millis(),
        "Agent trace async JSONL sink ready"
    );
    Ok(())
}
