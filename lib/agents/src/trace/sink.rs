// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::fs::{OpenOptions, create_dir_all};
use tokio::io::AsyncWriteExt;
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tokio_util::sync::CancellationToken;

use super::{bus, types::AgentTraceRecord};

static JSONL_WORKER_STARTED: AtomicBool = AtomicBool::new(false);

pub async fn spawn_jsonl_worker_with_shutdown(
    path: String,
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
        let mut file = file;
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    loop {
                        match rx.try_recv() {
                            Ok(rec) => {
                                if !write_record(&mut file, &rec).await {
                                    return;
                                }
                            }
                            Err(TryRecvError::Lagged(n)) => {
                                tracing::warn!(dropped = n, "agent trace bus lagged during shutdown; dropped records");
                            }
                            Err(TryRecvError::Empty | TryRecvError::Closed) => {
                                if let Err(e) = file.flush().await {
                                    tracing::warn!("agent_trace: shutdown flush failed: {e}");
                                }
                                return;
                            }
                        }
                    }
                }
                msg = rx.recv() => {
                    match msg {
                        Ok(rec) => {
                            if !write_record(&mut file, &rec).await {
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

    tracing::info!(path, "Agent trace sink ready");
    Ok(())
}

async fn write_record(file: &mut tokio::fs::File, rec: &AgentTraceRecord) -> bool {
    match serde_json::to_vec(rec) {
        Ok(mut bytes) => {
            bytes.push(b'\n');
            if let Err(e) = file.write_all(&bytes).await {
                tracing::warn!("agent_trace: write failed: {e}");
                return false;
            }
            if let Err(e) = file.flush().await {
                tracing::warn!("agent_trace: flush failed: {e}");
                return false;
            }
            true
        }
        Err(e) => {
            tracing::warn!("agent_trace: serialize failed: {e}");
            true
        }
    }
}
