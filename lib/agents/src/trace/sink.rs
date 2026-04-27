// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;
use tokio::fs::{OpenOptions, create_dir_all};
use tokio::io::AsyncWriteExt;

use super::bus;

pub async fn spawn_jsonl_worker(path: String) -> anyhow::Result<()> {
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

    let mut rx = bus::subscribe();
    tokio::spawn(async move {
        let mut file = file;
        loop {
            match rx.recv().await {
                Ok(rec) => match serde_json::to_vec(&rec) {
                    Ok(mut bytes) => {
                        bytes.push(b'\n');
                        if let Err(e) = file.write_all(&bytes).await {
                            tracing::warn!("agent_trace: write failed: {e}");
                        }
                    }
                    Err(e) => tracing::warn!("agent_trace: serialize failed: {e}"),
                },
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(dropped = n, "agent trace bus lagged; dropped records")
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    tracing::info!(path, "Agent trace sink ready");
    Ok(())
}
