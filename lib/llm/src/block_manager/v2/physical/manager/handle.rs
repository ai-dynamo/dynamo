// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Layout handle type encoding worker ID and layout ID.

use bincode::{Decode, Encode};

/// Unique handle for a layout combining worker_id and layout_id.
///
/// The handle encodes:
/// - Bits 0-63: worker_id (u64)
/// - Bits 64-79: layout_id (u16)
/// - Bits 80-127: Reserved (48 bits, currently unused)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
pub struct LayoutHandle(u128);

impl LayoutHandle {
    /// Create a new layout handle from worker_id and layout_id.
    ///
    /// # Arguments
    /// * `worker_id` - Unique identifier for the worker (0-63 bits)
    /// * `layout_id` - Layout identifier within the worker (64-79 bits)
    pub fn new(worker_id: u64, layout_id: u16) -> Self {
        let handle = (worker_id as u128) | ((layout_id as u128) << 64);
        Self(handle)
    }

    /// Extract the worker_id from this handle.
    pub fn worker_id(&self) -> u64 {
        (self.0 & 0xFFFF_FFFF_FFFF_FFFF) as u64
    }

    /// Extract the layout_id from this handle.
    pub fn layout_id(&self) -> u16 {
        ((self.0 >> 64) & 0xFFFF) as u16
    }

    /// Get the raw u128 value.
    pub fn as_u128(&self) -> u128 {
        self.0
    }

    /// Create a handle from a raw u128 value.
    pub fn from_u128(value: u128) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for LayoutHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LayoutHandle(worker={}, layout={})",
            self.worker_id(),
            self.layout_id()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_encoding() {
        let worker_id = 0x1234_5678_9ABC_DEF0u64;
        let layout_id = 0x4242u16;

        let handle = LayoutHandle::new(worker_id, layout_id);

        assert_eq!(handle.worker_id(), worker_id);
        assert_eq!(handle.layout_id(), layout_id);
    }

    #[test]
    fn test_handle_roundtrip() {
        let handle = LayoutHandle::new(42, 100);
        let raw = handle.as_u128();
        let restored = LayoutHandle::from_u128(raw);

        assert_eq!(handle, restored);
        assert_eq!(restored.worker_id(), 42);
        assert_eq!(restored.layout_id(), 100);
    }

    #[test]
    fn test_handle_max_values() {
        let max_worker = u64::MAX;
        let max_layout = u16::MAX;

        let handle = LayoutHandle::new(max_worker, max_layout);

        assert_eq!(handle.worker_id(), max_worker);
        assert_eq!(handle.layout_id(), max_layout);
    }

    #[test]
    fn test_handle_bincode_roundtrip() {
        let handle = LayoutHandle::new(999, 42);

        let encoded = bincode::encode_to_vec(handle, bincode::config::standard()).unwrap();
        let (decoded, _): (LayoutHandle, _) =
            bincode::decode_from_slice(&encoded, bincode::config::standard()).unwrap();

        assert_eq!(handle, decoded);
    }

    #[test]
    fn test_handle_display() {
        let handle = LayoutHandle::new(123, 456);
        let display = format!("{}", handle);
        assert!(display.contains("123"));
        assert!(display.contains("456"));
    }
}
