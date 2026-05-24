// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`KvbmInstance`] — the one [`RegistrationInstance`] variant the service
//! currently supports.
//!
//! All KVBM-specific shape (model name, layout, TP size, block size, engine
//! mode) lives here so that [`RegistrationInstance`] stays a thin oneof.
//! New instance types should land as sibling modules.
//!
//! [`RegistrationInstance`]: super::RegistrationInstance

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::error::{ServiceError, ServiceResult};
use crate::mode::ServiceMode;
use crate::proto;

/// A KVBM tenant: leader + TP-group workers attaching to a service-hosted
/// host-memory pool. Field equality is structural; the hash that drives
/// [`crate::registry::RegistrationKey`] is derived from a canonical
/// encoding of these fields plus the variant tag.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvbmInstance {
    pub model_name: String,
    pub layout: LayoutShape,
    pub tp_size: u32,
    pub block_size: u32,
    pub mode: ServiceMode,
}

impl KvbmInstance {
    /// How many slots this instance reserves out of the registry's capacity.
    /// For a KVBM tenant this is `tp_size` (one slot per worker GPU).
    pub fn slot_count(&self) -> u32 {
        self.tp_size
    }

    /// Stable identifier for logs / metrics labels.
    pub fn kind_str(&self) -> &'static str {
        "kvbm"
    }

    /// Validate against per-service bounds:
    /// - `model_name` non-empty
    /// - `tp_size` is a positive power of two, ≤ `capacity_slots`
    /// - `block_size` > 0
    /// - `layout` is not the reserved MLA variant
    pub fn validate(&self, capacity_slots: u32) -> ServiceResult<()> {
        if self.model_name.trim().is_empty() {
            return Err(ServiceError::InvalidArgument(
                "model_name must be non-empty".into(),
            ));
        }
        if self.tp_size == 0 || !self.tp_size.is_power_of_two() {
            return Err(ServiceError::InvalidArgument(format!(
                "tp_size must be a positive power of two (got {})",
                self.tp_size
            )));
        }
        if self.tp_size > capacity_slots {
            return Err(ServiceError::InvalidArgument(format!(
                "tp_size {} exceeds capacity {}",
                self.tp_size, capacity_slots
            )));
        }
        if self.block_size == 0 {
            return Err(ServiceError::InvalidArgument(
                "block_size must be > 0".into(),
            ));
        }
        if self.layout.is_mla() {
            return Err(ServiceError::Unimplemented(
                "MLA layout is reserved and not yet supported by kvbm-engine".into(),
            ));
        }
        Ok(())
    }

    /// Decode the wire-form [`proto::KvbmInstance`].
    pub fn from_proto(value: proto::KvbmInstance) -> ServiceResult<Self> {
        let mode = proto::ServiceMode::try_from(value.mode).map_err(|_| {
            ServiceError::InvalidArgument(format!("unknown ServiceMode value: {}", value.mode))
        })?;
        Ok(Self {
            model_name: value.model_name,
            layout: LayoutShape::from_proto(value.layout_mode)?,
            tp_size: value.tp_size,
            block_size: value.block_size,
            mode: ServiceMode::from_proto(mode),
        })
    }

    /// Feed a canonical byte encoding of this instance into `hasher`.
    /// Used by [`super::RegistrationInstance::key`] — the encoding must be
    /// deterministic across calls and across processes.
    pub(crate) fn hash_into(&self, hasher: &mut Hasher) {
        hash_len_prefixed(hasher, self.model_name.as_bytes());
        self.layout.hash_into(hasher);
        hasher.update(&self.tp_size.to_le_bytes());
        hasher.update(&self.block_size.to_le_bytes());
        hash_len_prefixed(hasher, self.mode.to_string().as_bytes());
    }
}

/// Layout-mode payload for [`KvbmInstance`]. The bytes are opaque to the
/// service — typically a bincode-encoded `kvbm_physical::LayoutDescriptor`
/// — and participate in key equality bytewise.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayoutShape {
    UniversalTp1Canonical { bytes: Vec<u8> },
    OperationalSymmetric { bytes: Vec<u8> },
    Mla,
}

impl LayoutShape {
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::UniversalTp1Canonical { .. } => "universal_tp1_canonical",
            Self::OperationalSymmetric { .. } => "operational_symmetric",
            Self::Mla => "mla",
        }
    }

    pub fn is_mla(&self) -> bool {
        matches!(self, Self::Mla)
    }

    pub fn from_proto(value: Option<proto::LayoutMode>) -> ServiceResult<Self> {
        let kind = value
            .and_then(|m| m.kind)
            .ok_or_else(|| ServiceError::InvalidArgument("layout_mode is required".into()))?;
        Ok(match kind {
            proto::layout_mode::Kind::UniversalTp1Canonical(bytes) => {
                Self::UniversalTp1Canonical { bytes }
            }
            proto::layout_mode::Kind::OperationalSymmetric(bytes) => {
                Self::OperationalSymmetric { bytes }
            }
            proto::layout_mode::Kind::Mla(_) => Self::Mla,
        })
    }

    pub(crate) fn hash_into(&self, hasher: &mut Hasher) {
        match self {
            Self::UniversalTp1Canonical { bytes } => {
                hasher.update(b"layout:u\0");
                hash_len_prefixed(hasher, bytes);
            }
            Self::OperationalSymmetric { bytes } => {
                hasher.update(b"layout:o\0");
                hash_len_prefixed(hasher, bytes);
            }
            Self::Mla => {
                hasher.update(b"layout:m\0");
            }
        }
    }
}

fn hash_len_prefixed(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(tp: u32, block: u32, model: &str) -> KvbmInstance {
        KvbmInstance {
            model_name: model.into(),
            layout: LayoutShape::UniversalTp1Canonical {
                bytes: vec![1, 2, 3],
            },
            tp_size: tp,
            block_size: block,
            mode: ServiceMode::Kvbm,
        }
    }

    #[test]
    fn slot_count_is_tp_size() {
        assert_eq!(make(4, 64, "llm").slot_count(), 4);
    }

    #[test]
    fn rejects_tp_size_not_power_of_two() {
        assert!(matches!(
            make(3, 64, "llm").validate(8),
            Err(ServiceError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_tp_size_zero() {
        assert!(matches!(
            make(0, 64, "llm").validate(8),
            Err(ServiceError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_tp_size_above_capacity() {
        assert!(matches!(
            make(16, 64, "llm").validate(8),
            Err(ServiceError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_empty_model_name() {
        assert!(matches!(
            make(2, 64, "").validate(8),
            Err(ServiceError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_zero_block_size() {
        assert!(matches!(
            make(2, 0, "llm").validate(8),
            Err(ServiceError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_mla_layout() {
        let inst = KvbmInstance {
            layout: LayoutShape::Mla,
            ..make(2, 64, "llm")
        };
        assert!(matches!(
            inst.validate(8),
            Err(ServiceError::Unimplemented(_))
        ));
    }

    #[test]
    fn accepts_valid_instance() {
        assert!(make(4, 64, "llm").validate(8).is_ok());
    }
}
