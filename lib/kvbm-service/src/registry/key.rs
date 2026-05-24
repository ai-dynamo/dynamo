// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`RegistrationKey`] — opaque, byte-stable hash of a
//! [`RegistrationInstance`]'s active arm.
//!
//! The registry uses keys for tenancy comparisons; clients never see the
//! raw bytes. Equal instances (same variant, byte-identical fields) produce
//! equal keys; different variants of [`RegistrationInstance`] hash into
//! disjoint key spaces because each arm prefixes its bytes with the variant
//! tag during hashing.
//!
//! [`RegistrationInstance`]: crate::instance::RegistrationInstance

use serde::{Deserialize, Serialize};

/// 32-byte blake3 digest of a [`RegistrationInstance`]'s active arm.
/// Treat as opaque — equality is the only semantically meaningful op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegistrationKey([u8; 32]);

impl RegistrationKey {
    /// Build from the 32 raw bytes of a digest. Used by
    /// [`RegistrationInstance::key`](crate::instance::RegistrationInstance::key);
    /// callers outside the instance module should compute keys by going
    /// through [`RegistrationInstance::key`] rather than constructing one
    /// here directly.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Hex-encoded short form for logs (first 16 hex chars = 64 bits).
    pub fn short_hex(&self) -> String {
        let mut s = String::with_capacity(16);
        for byte in &self.0[..8] {
            use std::fmt::Write;
            let _ = write!(&mut s, "{byte:02x}");
        }
        s
    }
}

impl std::fmt::Display for RegistrationKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_64_hex_chars() {
        let key = RegistrationKey::from_bytes([0xab; 32]);
        let s = key.to_string();
        assert_eq!(s.len(), 64);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn short_hex_is_16_chars() {
        let key = RegistrationKey::from_bytes([0x12; 32]);
        assert_eq!(key.short_hex().len(), 16);
        assert!(key.short_hex().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn equal_bytes_compare_equal() {
        let a = RegistrationKey::from_bytes([1; 32]);
        let b = RegistrationKey::from_bytes([1; 32]);
        assert_eq!(a, b);
    }

    #[test]
    fn different_bytes_compare_unequal() {
        let a = RegistrationKey::from_bytes([1; 32]);
        let mut b_bytes = [1; 32];
        b_bytes[0] = 2;
        let b = RegistrationKey::from_bytes(b_bytes);
        assert_ne!(a, b);
    }
}
