// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`StreamAnchorHandle`]: compact u128 wire handle encoding WorkerId + local anchor ID.
//!
//! Wire layout (MSB to LSB): `[worker_id: upper 64 bits][local_id: lower 64 bits]`
//!
//! Serializes via rmp-serde as `{hi: u64, lo: u64}` to avoid the u128-as-bytes
//! MessagePack encoding. See WIRE-03.

use serde::{Deserialize, Serialize};
use velo_common::WorkerId;

const WORKER_SHIFT: u32 = 64;

/// Compact wire handle encoding a [`WorkerId`] (upper 64 bits) and a local anchor ID
/// (lower 64 bits) into a single `u128`.
///
/// Serializes via rmp-serde as a two-field struct `{hi: u64, lo: u64}` — not as raw
/// binary bytes — to guarantee correct round-tripping across msgpack boundaries.
///
/// `local_id` of `0` is reserved; [`AnchorManager`] assigns IDs starting at `1`.
///
/// [`AnchorManager`]: crate — defined in Phase 7
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StreamAnchorHandle(u128);

/// Private wire representation for rmp-serde serialization.
///
/// rmp-serde encodes a raw `u128` as a MessagePack binary blob (`bin8`), which
/// cannot be decoded back to a struct. By delegating to this two-field struct we
/// encode as a fixmap with named fields that round-trip correctly.
#[derive(Serialize, Deserialize)]
struct StreamAnchorHandleWire {
    hi: u64,
    lo: u64,
}

impl StreamAnchorHandle {
    /// Encode a [`WorkerId`] and local anchor ID into a [`StreamAnchorHandle`].
    ///
    /// `local_id` of `0` is reserved. Callers should start local IDs at `1`.
    pub fn pack(worker_id: WorkerId, local_id: u64) -> Self {
        let raw = ((worker_id.as_u64() as u128) << WORKER_SHIFT) | (local_id as u128);
        Self(raw)
    }

    /// Decode the [`WorkerId`] and local anchor ID from this handle.
    pub fn unpack(&self) -> (WorkerId, u64) {
        let worker_id = WorkerId::from_u64((self.0 >> WORKER_SHIFT) as u64);
        let local_id = self.0 as u64;
        (worker_id, local_id)
    }

    /// Return the raw `u128` value (useful for logging and display).
    pub fn as_u128(&self) -> u128 {
        self.0
    }
}

impl Serialize for StreamAnchorHandle {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        StreamAnchorHandleWire {
            hi: (self.0 >> 64) as u64,
            lo: self.0 as u64,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StreamAnchorHandle {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = StreamAnchorHandleWire::deserialize(deserializer)?;
        Ok(Self(((wire.hi as u128) << 64) | (wire.lo as u128)))
    }
}

impl std::fmt::Display for StreamAnchorHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (worker_id, local_id) = self.unpack();
        write!(
            f,
            "StreamAnchorHandle(worker={}, local={})",
            worker_id.as_u64(),
            local_id
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use velo_common::WorkerId;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let worker_id = WorkerId::from_u64(0xDEAD_BEEF_1234_5678);
        let local_id: u64 = 42;
        let handle = StreamAnchorHandle::pack(worker_id, local_id);
        let (recovered_worker, recovered_local) = handle.unpack();
        assert_eq!(recovered_worker, worker_id);
        assert_eq!(recovered_local, local_id);
    }

    #[test]
    fn test_rmp_serde_roundtrip() {
        let worker_id = WorkerId::from_u64(0xDEAD_BEEF_1234_5678);
        let local_id: u64 = 99;
        let handle = StreamAnchorHandle::pack(worker_id, local_id);

        let encoded = rmp_serde::to_vec(&handle).expect("serialize");
        let decoded: StreamAnchorHandle = rmp_serde::from_slice(&encoded).expect("deserialize");

        assert_eq!(handle, decoded);
        let (w, l) = decoded.unpack();
        assert_eq!(w, worker_id);
        assert_eq!(l, local_id);
    }

    #[test]
    fn test_zero_local_id_encodes() {
        // local_id = 0 is reserved but must not panic
        let worker_id = WorkerId::from_u64(1);
        let handle = StreamAnchorHandle::pack(worker_id, 0);
        let (w, l) = handle.unpack();
        assert_eq!(w, worker_id);
        assert_eq!(l, 0);
    }

    #[test]
    fn test_max_values_roundtrip() {
        let worker_id = WorkerId::from_u64(u64::MAX);
        let local_id = u64::MAX;
        let handle = StreamAnchorHandle::pack(worker_id, local_id);
        let encoded = rmp_serde::to_vec(&handle).unwrap();
        let decoded: StreamAnchorHandle = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(handle, decoded);
        let (w, l) = decoded.unpack();
        assert_eq!(w.as_u64(), u64::MAX);
        assert_eq!(l, u64::MAX);
    }
}
