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
//! AM headers: { "anchor_id": "<u64>" }
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
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use futures::future::BoxFuture;
use velo_common::WorkerId;
use velo_messenger::{Context, Handler, Messenger};

use crate::transport::FrameTransport;

/// AM header key used to route frames to the correct anchor's dispatch channel.
const ANCHOR_ID_HEADER: &str = "anchor_id";

/// Production [`FrameTransport`] backed by velo-messenger active messages.
///
/// Holds its own internal `DashMap<u64, flume::Sender<Vec<u8>>>` dispatch map
/// for transport-level routing. The `_stream_data` AM handler extracts the
/// target `anchor_id` from the AM headers ([`ANCHOR_ID_HEADER`]) and writes
/// the payload directly into the matching dispatch channel. The existing
/// `reader_pump` (spawned by `_anchor_attach`) reads from the transport
/// receiver and bridges to `frame_tx` with heartbeat monitoring intact.
pub struct VeloFrameTransport {
    messenger: Arc<Messenger>,
    dispatch: Arc<DashMap<u64, flume::Sender<Vec<u8>>>>,
    worker_id: WorkerId,
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
        let dispatch: Arc<DashMap<u64, flume::Sender<Vec<u8>>>> = Arc::new(DashMap::new());

        // Register the shared _stream_data handler.
        // The handler captures dispatch and routes incoming AM payloads to the
        // correct per-anchor transport channel based on the anchor_id AM header.
        let handler_dispatch = dispatch.clone();
        let handler = Handler::am_handler("_stream_data", move |ctx: Context| {
            let anchor_id = match ctx
                .headers
                .as_ref()
                .and_then(|h| h.get(ANCHOR_ID_HEADER))
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
            let frame_bytes = ctx.payload.to_vec();
            if let Some(tx) = handler_dispatch.get(&anchor_id) {
                // try_send: non-blocking, drops frame if channel full
                // (back-pressure handled by TCP flow control at lower level)
                let _ = tx.try_send(frame_bytes);
            }
            Ok(())
        })
        .build();

        messenger.register_streaming_handler(handler)?;

        Ok(Self {
            messenger,
            dispatch,
            worker_id,
        })
    }

    /// Remove an anchor's dispatch entry.
    ///
    /// Called when the reader_pump exits or the anchor is cleaned up.
    /// After unbind, subsequent AM frames targeting this anchor_id are silently dropped.
    pub fn unbind(&self, anchor_id: u64) {
        self.dispatch.remove(&anchor_id);
    }
}

impl FrameTransport for VeloFrameTransport {
    fn bind(&self, anchor_id: u64) -> BoxFuture<'_, Result<(String, flume::Receiver<Vec<u8>>)>> {
        let worker_id = self.worker_id;
        let dispatch = self.dispatch.clone();
        Box::pin(async move {
            let (tx, rx) = flume::bounded::<Vec<u8>>(256);
            dispatch.insert(anchor_id, tx);
            let endpoint = format!("velo://{}/stream/{}", worker_id.as_u64(), anchor_id);
            Ok((endpoint, rx))
        })
    }

    fn connect(
        &self,
        endpoint: &str,
        _anchor_id: u64,
        _session_id: u64,
    ) -> BoxFuture<'_, Result<flume::Sender<Vec<u8>>>> {
        let endpoint = endpoint.to_string();
        let messenger = self.messenger.clone();
        Box::pin(async move {
            let (target_worker_id, target_anchor_id) = parse_velo_uri(&endpoint)?;
            let (tx, rx) = flume::bounded::<Vec<u8>>(256);

            // Spawn pump task: reads from rx, sends AM per frame with
            // anchor_id routed via AM headers (no payload prefix).
            tokio::spawn(async move {
                while let Ok(frame_bytes) = rx.recv_async().await {
                    let mut headers = HashMap::with_capacity(1);
                    headers.insert(
                        ANCHOR_ID_HEADER.to_string(),
                        target_anchor_id.to_string(),
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
