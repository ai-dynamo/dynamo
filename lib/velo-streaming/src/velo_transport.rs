// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production [`FrameTransport`] implementation backed by velo-messenger's
//! active message (AM) fire-and-forget system.
//!
//! [`VeloFrameTransport`] uses a single shared `_stream_data` AM handler
//! registered at construction time. Each incoming AM carries the target
//! `anchor_id` as a string value in the AM headers map (key
//! [`ANCHOR_ID_HEADER`]). The payload contains only the raw frame bytes --
//! no binary prefix.
//!
//! # Routing
//!
//! ```text
//! AM headers: { "anchor_id": "<u64>", "session_id": "<u64>" }
//! AM payload: [ frame_bytes: ... ]
//! ```
//!
//! # Construction
//!
//! ```ignore
//! let transport = VeloFrameTransport::new(messenger, worker_id)?;
//! let manager = AnchorManagerBuilder::default()
//!     .worker_id(worker_id)
//!     .transport(Arc::new(transport))
//!     .build()?;
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use futures::future::BoxFuture;
use velo_common::WorkerId;
use velo_messenger::{Context, Handler, Messenger};

use crate::transport::FrameTransport;

/// AM header key used to route frames to the correct anchor's dispatch channel.
const ANCHOR_ID_HEADER: &str = "anchor_id";

/// AM header key for session-level routing within a single anchor.
const SESSION_ID_HEADER: &str = "session_id";

/// Production [`FrameTransport`] backed by velo-messenger active messages.
///
/// Holds its own internal `DashMap<(u64, u64), flume::Sender<Vec<u8>>>` dispatch
/// map keyed by `(anchor_id, session_id)` for transport-level routing. The
/// `_stream_data` AM handler extracts both `anchor_id` and `session_id` from
/// the AM headers and writes the payload directly into the matching dispatch
/// channel. This composite key ensures that stale frames from a prior session
/// are silently dropped after a detach+reattach cycle.
pub struct VeloFrameTransport {
    messenger: Arc<Messenger>,
    dispatch: Arc<DashMap<(u64, u64), flume::Sender<Vec<u8>>>>,
    worker_id: WorkerId,
    /// Number of times `_stream_data` hit the slow (blocking) send path
    /// because the dispatch channel was full. Indicates consumer backpressure.
    backpressure_count: Arc<AtomicU64>,
}

impl VeloFrameTransport {
    /// Create a new `VeloFrameTransport` and register the `_stream_data` handler.
    ///
    /// # Arguments
    ///
    /// * `messenger` - Injected `Arc<Messenger>`, must already be constructed.
    /// * `worker_id` - This worker's identity, used for endpoint URI construction.
    ///
    /// # Errors
    ///
    /// Returns an error if handler registration fails (e.g., duplicate handler name).
    pub fn new(messenger: Arc<Messenger>, worker_id: WorkerId) -> Result<Self> {
        let dispatch: Arc<DashMap<(u64, u64), flume::Sender<Vec<u8>>>> = Arc::new(DashMap::new());
        let backpressure_count: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));

        // Register the shared _stream_data handler.
        // The handler captures dispatch and routes incoming AM payloads to the
        // correct per-anchor transport channel based on the anchor_id AM header.
        let handler_dispatch = dispatch.clone();
        let handler_backpressure = backpressure_count.clone();
        let handler = Handler::am_handler_async("_stream_data", move |ctx: Context| {
            let handler_dispatch = handler_dispatch.clone();
            let backpressure = handler_backpressure.clone();
            async move {
                let headers = match ctx.headers.as_ref() {
                    Some(h) => h,
                    None => {
                        tracing::warn!(
                            "_stream_data: missing headers, dropping frame"
                        );
                        return Ok(());
                    }
                };
                let anchor_id = match headers
                    .get(ANCHOR_ID_HEADER)
                    .and_then(|v| v.parse::<u64>().ok())
                {
                    Some(id) => id,
                    None => {
                        tracing::warn!(
                            "_stream_data: missing or invalid {} header, dropping frame",
                            ANCHOR_ID_HEADER
                        );
                        return Ok(());
                    }
                };
                let session_id = match headers
                    .get(SESSION_ID_HEADER)
                    .and_then(|v| v.parse::<u64>().ok())
                {
                    Some(id) => id,
                    None => {
                        tracing::warn!(
                            anchor_id,
                            "_stream_data: missing or invalid {} header, dropping frame",
                            SESSION_ID_HEADER
                        );
                        return Ok(());
                    }
                };
                let frame_bytes = ctx.payload.to_vec();
                // Clone sender out of DashMap ref before the potential await point.
                let tx = handler_dispatch
                    .get(&(anchor_id, session_id))
                    .map(|entry| entry.value().clone());
                if let Some(tx) = tx {
                    // Fast path: try_send avoids async overhead when channel has space.
                    // Slow path: recover the frame from TrySendError::Full and block
                    // via send_async to guarantee delivery (no frame loss).
                    match tx.try_send(frame_bytes) {
                        Ok(()) => {}
                        Err(flume::TrySendError::Full(frame_bytes)) => {
                            backpressure.fetch_add(1, Ordering::Relaxed);
                            tracing::debug!(
                                anchor_id,
                                "_stream_data: dispatch channel full, blocking"
                            );
                            if tx.send_async(frame_bytes).await.is_err() {
                                tracing::warn!(
                                    anchor_id,
                                    "_stream_data: receiver closed, dropping frame"
                                );
                            }
                        }
                        Err(flume::TrySendError::Disconnected(_)) => {
                            tracing::warn!(
                                anchor_id,
                                "_stream_data: receiver closed, dropping frame"
                            );
                        }
                    }
                }
                Ok(())
            }
        })
        .build();

        messenger.register_streaming_handler(handler)?;

        Ok(Self {
            messenger,
            dispatch,
            worker_id,
            backpressure_count,
        })
    }

    /// Returns the number of times the `_stream_data` handler had to block
    /// because a dispatch channel was full (backpressure from a slow consumer).
    pub fn backpressure_count(&self) -> u64 {
        self.backpressure_count.load(Ordering::Relaxed)
    }

    /// Remove all dispatch entries for the given anchor (any session).
    ///
    /// Called when the reader_pump exits or the anchor is cleaned up.
    /// After unbind, subsequent AM frames targeting this anchor_id are silently dropped.
    pub fn unbind(&self, anchor_id: u64) {
        self.dispatch.retain(|&(aid, _), _| aid != anchor_id);
    }
}

impl FrameTransport for VeloFrameTransport {
    fn bind(&self, anchor_id: u64, session_id: u64) -> BoxFuture<'_, Result<(String, flume::Receiver<Vec<u8>>)>> {
        let worker_id = self.worker_id;
        let dispatch = self.dispatch.clone();
        Box::pin(async move {
            let (tx, rx) = flume::bounded::<Vec<u8>>(256);
            // Remove any stale entry for a prior session on this anchor_id
            // before inserting the new one.
            dispatch.retain(|&(aid, _), _| aid != anchor_id);
            dispatch.insert((anchor_id, session_id), tx);
            let endpoint = format!("velo://{}/stream/{}", worker_id.as_u64(), anchor_id);
            Ok((endpoint, rx))
        })
    }

    fn connect(
        &self,
        endpoint: &str,
        _anchor_id: u64,
        session_id: u64,
    ) -> BoxFuture<'_, Result<flume::Sender<Vec<u8>>>> {
        let endpoint = endpoint.to_string();
        let messenger = self.messenger.clone();
        Box::pin(async move {
            let (target_worker_id, target_anchor_id) = parse_velo_uri(&endpoint)?;
            let (tx, rx) = flume::bounded::<Vec<u8>>(256);

            // Spawn pump task: reads from rx, sends AM per frame with
            // anchor_id + session_id routed via AM headers (no payload prefix).
            tokio::spawn(async move {
                while let Ok(frame_bytes) = rx.recv_async().await {
                    let mut headers = HashMap::with_capacity(2);
                    headers.insert(
                        ANCHOR_ID_HEADER.to_string(),
                        target_anchor_id.to_string(),
                    );
                    headers.insert(
                        SESSION_ID_HEADER.to_string(),
                        session_id.to_string(),
                    );

                    if let Err(e) = messenger
                        .am_send_streaming("_stream_data")
                        .expect("am_send_streaming builder")
                        .headers(headers)
                        .raw_payload(bytes::Bytes::from(frame_bytes))
                        .worker(WorkerId::from_u64(target_worker_id))
                        .send()
                        .await
                    {
                        tracing::error!("_stream_data am_send failed: {}", e);
                        break;
                    }
                }
            });

            Ok(tx)
        })
    }
}

/// Parse a `velo://` URI into `(worker_id, anchor_id)`.
///
/// Expected format: `velo://{worker_id}/stream/{anchor_id}`
///
/// # Errors
///
/// Returns `Err` on malformed URIs (missing prefix, wrong segment count,
/// non-numeric IDs, wrong path segment).
pub fn parse_velo_uri(uri: &str) -> Result<(u64, u64)> {
    let stripped = uri
        .strip_prefix("velo://")
        .ok_or_else(|| anyhow::anyhow!("invalid velo URI: missing velo:// prefix: {}", uri))?;
    let parts: Vec<&str> = stripped.split('/').collect();
    if parts.len() != 3 || parts[1] != "stream" {
        anyhow::bail!(
            "invalid velo URI format: expected velo://{{worker_id}}/stream/{{anchor_id}}, got: {}",
            uri
        );
    }
    let worker_id: u64 = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid worker_id in URI: {}", parts[0]))?;
    let anchor_id: u64 = parts[2]
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid anchor_id in URI: {}", parts[2]))?;
    Ok((worker_id, anchor_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_velo_uri_valid() {
        let (wid, aid) = parse_velo_uri("velo://123/stream/456").unwrap();
        assert_eq!(wid, 123);
        assert_eq!(aid, 456);
    }

    #[test]
    fn test_parse_velo_uri_missing_prefix() {
        assert!(parse_velo_uri("http://123/stream/456").is_err());
    }

    #[test]
    fn test_parse_velo_uri_non_numeric_worker() {
        assert!(parse_velo_uri("velo://abc/stream/456").is_err());
    }

    #[test]
    fn test_parse_velo_uri_non_numeric_anchor() {
        assert!(parse_velo_uri("velo://123/stream/xyz").is_err());
    }

    #[test]
    fn test_parse_velo_uri_wrong_path_segment() {
        assert!(parse_velo_uri("velo://123/wrong/456").is_err());
    }

    #[test]
    fn test_parse_velo_uri_too_few_segments() {
        assert!(parse_velo_uri("velo://123/stream").is_err());
    }

    #[test]
    fn test_parse_velo_uri_too_many_segments() {
        assert!(parse_velo_uri("velo://123/stream/456/extra").is_err());
    }
}
