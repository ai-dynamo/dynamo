// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`RegistrationInstance`] — the tagged-union of registration types the
//! service accepts.
//!
//! New instance types add a sibling module (see [`kvbm`]) and a variant
//! here, plus the corresponding proto oneof arm. The shared API surface
//! is:
//!
//! - [`RegistrationInstance::key`] — canonical [`RegistrationKey`] hash
//!   derived from the active arm's bytes (variant tag included).
//! - [`RegistrationInstance::slot_count`] — capacity bookkeeping value.
//! - [`RegistrationInstance::validate`] — pre-registration sanity check.
//! - [`RegistrationInstance::from_proto`] — wire-form decoder.
//!
//! [`RegistrationKey`]: crate::registry::RegistrationKey

pub mod kvbm;

use blake3::Hasher;
use serde::{Deserialize, Serialize};

pub use self::kvbm::{KvbmInstance, LayoutShape};

use crate::error::{ServiceError, ServiceResult};
use crate::proto;
use crate::registry::RegistrationKey;

/// Tagged union of registration instance types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RegistrationInstance {
    Kvbm(KvbmInstance),
}

impl RegistrationInstance {
    /// Stable opaque hash of the active arm, used for tenancy comparisons
    /// in [`crate::registry::Registry`]. Different variants with otherwise
    /// identical fields hash to different keys.
    pub fn key(&self) -> RegistrationKey {
        let mut hasher = Hasher::new();
        match self {
            Self::Kvbm(inst) => {
                hasher.update(b"kvbm-service.v1:instance:kvbm\0");
                inst.hash_into(&mut hasher);
            }
        }
        RegistrationKey::from_bytes(*hasher.finalize().as_bytes())
    }

    /// Slot count this instance reserves out of the registry's capacity.
    pub fn slot_count(&self) -> u32 {
        match self {
            Self::Kvbm(inst) => inst.slot_count(),
        }
    }

    /// Stable identifier of the active arm for logs / metrics labels.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Kvbm(inst) => inst.kind_str(),
        }
    }

    /// Per-variant validation against the service's slot capacity.
    pub fn validate(&self, capacity_slots: u32) -> ServiceResult<()> {
        match self {
            Self::Kvbm(inst) => inst.validate(capacity_slots),
        }
    }

    /// Decode the wire form. Returns `InvalidArgument` when the oneof is
    /// unset or carries an unknown variant.
    pub fn from_proto(value: proto::RegistrationInstance) -> ServiceResult<Self> {
        let kind = value
            .kind
            .ok_or_else(|| ServiceError::InvalidArgument("instance.kind is required".into()))?;
        match kind {
            proto::registration_instance::Kind::Kvbm(inner) => {
                Ok(Self::Kvbm(KvbmInstance::from_proto(inner)?))
            }
        }
    }

    /// Borrow as a [`KvbmInstance`] when the active arm is KVBM. Returns
    /// `None` for other arms. Helpful when a container only knows how to
    /// handle one instance type.
    pub fn as_kvbm(&self) -> Option<&KvbmInstance> {
        match self {
            Self::Kvbm(inst) => Some(inst),
        }
    }
}

impl From<KvbmInstance> for RegistrationInstance {
    fn from(inst: KvbmInstance) -> Self {
        Self::Kvbm(inst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode::ServiceMode;

    fn kvbm(model: &str, layout_byte: u8) -> RegistrationInstance {
        RegistrationInstance::Kvbm(KvbmInstance {
            model_name: model.into(),
            layout: LayoutShape::UniversalTp1Canonical {
                bytes: vec![layout_byte],
            },
            tp_size: 2,
            block_size: 64,
            mode: ServiceMode::Kvbm,
        })
    }

    #[test]
    fn equal_instances_have_equal_keys() {
        let a = kvbm("llm", 0xab);
        let b = kvbm("llm", 0xab);
        assert_eq!(a.key(), b.key());
    }

    #[test]
    fn different_layout_bytes_change_key() {
        assert_ne!(kvbm("llm", 0xaa).key(), kvbm("llm", 0xbb).key());
    }

    #[test]
    fn different_model_names_change_key() {
        assert_ne!(kvbm("llm-a", 0).key(), kvbm("llm-b", 0).key());
    }

    #[test]
    fn slot_count_delegates_to_variant() {
        assert_eq!(kvbm("llm", 0).slot_count(), 2);
    }

    #[test]
    fn kind_str_matches_variant() {
        assert_eq!(kvbm("llm", 0).kind_str(), "kvbm");
    }
}
