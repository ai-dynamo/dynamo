// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire types for bidirectional streaming over the velo request plane.
//!
//! Bidi sits as a parallel path to the existing unary/server-streaming flow.
//! Both directions of a bidi exchange ride two velo SPSC streams (one anchor
//! per side). The kick-off velo unary RPC exchanges
//! [`StreamAnchorHandle`]s; once both sides have attached, [`BidiFrame`]s
//! flow symmetrically.
//!
//! # Frame
//!
//! A single [`BidiFrame<T>`] enum is used for both directions, parameterized
//! by the per-direction payload type. The discriminant is one msgpack byte.
//! The non-terminal [`BidiFrame::Done`] variant signals "this side will not
//! send any more `Data`" while leaving the velo stream open for any
//! in-flight control responses (Control variant reserved; not in v1). When
//! both sides have observed each other's `Done`, both `finalize()` their
//! velo `StreamSender`.
//!
//! `Done` and the underlying velo `Finalized` sentinel are framework-internal
//! — user-visible streams simply end after the last `Data(T)`.

pub mod session;

use serde::{Deserialize, Serialize};
use velo::streaming::StreamAnchorHandle;

/// Symmetric frame for both directions of a bidi exchange.
///
/// `T` is the per-direction payload type (request items going one way,
/// response items going the other way). The same enum is used on both ends
/// of every velo stream in the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BidiFrame<T> {
    /// User payload item.
    Data(T),
    /// Half-close: this side commits to no more `Data`. Stream stays open
    /// for future Control responses (Control variant reserved; not in v1).
    Done,
}

/// Velo unary RPC payload from client to server at session start.
///
/// Carries the client's [`StreamAnchorHandle`] (the consumer-side anchor for
/// server→client items) and a user-typed init payload `I`.
#[derive(Debug, Serialize, Deserialize)]
pub struct BidiInitRequest<I> {
    pub client_handle: StreamAnchorHandle,
    pub request_id: String,
    pub init: I,
    /// Wall-clock send timestamp (nanos since UNIX epoch) for transport latency
    /// breakdown. Mirrors the unary path's `RequestControlMessage::frontend_send_ts_ns`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontend_send_ts_ns: Option<u64>,
}

/// Velo unary ACK from server to client: the server's anchor handle, or an
/// error string the client should surface back to its caller.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BidiInitResponse {
    Ok { server_handle: StreamAnchorHandle },
    Err { reason: String },
}

/// Name of the velo handler the bidi-init RPC targets. Distinct from the
/// unary `dynamo-request-plane` handler so dispatch tables stay clean.
pub const BIDI_INIT_HANDLER: &str = "dynamo-bidi-init";

/// Registry key under which [`BidiIngress`] stuffs the deserialized
/// `init: I` payload on the server-side `ManyIn<T>` context. User handlers
/// retrieve it via `ctx.clone_unique::<I>(BIDI_INIT_KEY)`.
///
/// [`BidiIngress`]: crate::pipeline::network::ingress::bidi_handler::BidiIngress
pub const BIDI_INIT_KEY: &str = "dynamo-bidi-init-payload";

/// Default unattached-anchor TTL for both sides of a bidi session. The
/// timeout fires only between create-and-attach (handshake window). Once a
/// sender attaches, velo cancels the timer and heartbeats own liveness.
///
/// 30s is plenty for handshake RTT under any reasonable load while still
/// reaping anchors abandoned by client crashes during the handshake.
pub const BIDI_UNATTACHED_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bidi_frame_roundtrip_data() {
        let frame: BidiFrame<u32> = BidiFrame::Data(42);
        let bytes = rmp_serde::to_vec(&frame).expect("encode");
        let decoded: BidiFrame<u32> = rmp_serde::from_slice(&bytes).expect("decode");
        match decoded {
            BidiFrame::Data(v) => assert_eq!(v, 42),
            other => panic!("expected Data(42), got {other:?}"),
        }
    }

    #[test]
    fn bidi_frame_roundtrip_done() {
        let frame: BidiFrame<u32> = BidiFrame::Done;
        let bytes = rmp_serde::to_vec(&frame).expect("encode");
        let decoded: BidiFrame<u32> = rmp_serde::from_slice(&bytes).expect("decode");
        assert!(matches!(decoded, BidiFrame::Done));
    }

    #[test]
    fn bidi_init_response_roundtrip() {
        let resp = BidiInitResponse::Err {
            reason: "no handler".into(),
        };
        let bytes = rmp_serde::to_vec(&resp).expect("encode");
        let decoded: BidiInitResponse = rmp_serde::from_slice(&bytes).expect("decode");
        match decoded {
            BidiInitResponse::Err { reason } => assert_eq!(reason, "no handler"),
            other => panic!("expected Err, got {other:?}"),
        }
    }
}
