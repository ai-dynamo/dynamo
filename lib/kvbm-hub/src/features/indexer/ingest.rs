// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ ingest loop: decode published [`KvbmCacheEvents`] batches and apply
//! them to the [`PositionalIndex`].

use std::sync::Arc;

use futures::StreamExt;
use kvbm_logical::events::KvbmCacheEvents;
use tmq::subscribe::Subscribe;
use tokio_util::sync::CancellationToken;

use super::index::PositionalIndex;

/// Drains the bound `SUB` socket until cancelled. Each multipart frame's
/// **last** frame is the msgpack-encoded [`KvbmCacheEvents`] payload (the
/// publisher prepends the subject as a topic frame). Malformed frames are
/// logged and skipped — one bad message never tears down ingest.
pub async fn run_ingest_loop(
    mut sub: Subscribe,
    index: Arc<PositionalIndex>,
    cancel: CancellationToken,
) {
    tracing::info!("indexer ingest loop started");
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            msg = sub.next() => match msg {
                Some(Ok(multipart)) => {
                    let Some(payload) = multipart.iter().last() else {
                        tracing::trace!("indexer: empty multipart, skipping");
                        continue;
                    };
                    match rmp_serde::from_slice::<KvbmCacheEvents>(payload) {
                        Ok(batch) => index.apply(batch),
                        Err(e) => {
                            tracing::warn!(error = %e, bytes = payload.len(), "indexer: undecodable batch, dropping");
                        }
                    }
                }
                Some(Err(e)) => {
                    tracing::warn!(error = %e, "indexer: ZMQ recv error");
                }
                None => {
                    tracing::info!("indexer: SUB stream ended");
                    break;
                }
            },
        }
    }
    tracing::info!("indexer ingest loop stopped");
}
