// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry value types and traits.

use std::fmt::Debug;

pub trait RegistryValue: Clone + Debug + Send + Sync + 'static {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
}

impl RegistryValue for u64 {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bytes.try_into().ok().map(u64::from_le_bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StorageBackend {
    Object = 0,
    Disk = 1,
}

impl TryFrom<u8> for StorageBackend {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Object),
            1 => Ok(Self::Disk),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StorageLocation {
    pub backend: StorageBackend,
    pub path: String,
    pub size_bytes: u64,
}

impl RegistryValue for StorageLocation {
    fn to_bytes(&self) -> Vec<u8> {
        let path_bytes = self.path.as_bytes();
        let mut buf = Vec::with_capacity(1 + 2 + path_bytes.len() + 8);
        buf.push(self.backend as u8);
        buf.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(path_bytes);
        buf.extend_from_slice(&self.size_bytes.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 11 {
            return None;
        }
        let backend = StorageBackend::try_from(bytes[0]).ok()?;
        let path_len = u16::from_le_bytes(bytes[1..3].try_into().ok()?) as usize;
        if bytes.len() < 3 + path_len + 8 {
            return None;
        }
        let path = String::from_utf8(bytes[3..3 + path_len].to_vec()).ok()?;
        let size_bytes = u64::from_le_bytes(bytes[3 + path_len..3 + path_len + 8].try_into().ok()?);
        Some(Self {
            backend,
            path,
            size_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64_roundtrip() {
        let val: u64 = 0x123456789ABCDEF0;
        let bytes = val.to_bytes();
        assert_eq!(u64::from_bytes(&bytes), Some(val));
    }

    #[test]
    fn test_storage_location_roundtrip() {
        let loc = StorageLocation {
            backend: StorageBackend::Object,
            path: "bucket/key/object.bin".to_string(),
            size_bytes: 1024 * 1024,
        };
        let bytes = loc.to_bytes();
        let decoded = StorageLocation::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.backend, loc.backend);
        assert_eq!(decoded.path, loc.path);
        assert_eq!(decoded.size_bytes, loc.size_bytes);
    }
}
