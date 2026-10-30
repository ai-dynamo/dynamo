// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Residency vocabulary: storage tiers, logical owners, and the tolerant wire form of the residency domain.

use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::identity::CacheOwnerId;
use crate::worker::WorkerWithDpRank;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum StorageTier {
    #[default]
    Device,
    HostPinned,
    Disk,
    External,
}

/// Logical lifetime that owns a cache residency independently of its physical tier.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyDomain {
    #[default]
    Worker,
    CacheOwner,
}

/// Exact logical owner recorded inside a physical-tier index.
///
/// Worker ownership is tied to one serving attachment. Cache-owner ownership
/// is stable across those attachments and is projected to a ready worker only
/// by the lib/llm reconciliation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyOwner {
    Worker(WorkerWithDpRank),
    CacheOwner(CacheOwnerId),
}

/// Compact mutation-path identity for an exact residency owner.
///
/// Lower-tier edges carry this fixed-size value instead of embedding the full
/// [`CacheOwnerId`] in every ownership entry. The full owner remains in the
/// reverse index once per logical owner so dumps can round-trip it exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidencyOwnerKey([u8; 16]);

impl ResidencyOwnerKey {
    /// Wrap a 16-byte digest. The router computes the digest; this crate only carries it.
    pub const fn from_digest(digest: [u8; 16]) -> Self {
        Self(digest)
    }

    pub const fn digest(self) -> [u8; 16] {
        self.0
    }
}

impl ResidencyOwner {
    pub const fn worker(worker: WorkerWithDpRank) -> Self {
        Self::Worker(worker)
    }

    pub const fn cache_owner(cache_owner: CacheOwnerId) -> Self {
        Self::CacheOwner(cache_owner)
    }

    pub const fn domain(self) -> ResidencyDomain {
        match self {
            Self::Worker(_) => ResidencyDomain::Worker,
            Self::CacheOwner(_) => ResidencyDomain::CacheOwner,
        }
    }

    pub const fn as_worker(self) -> Option<WorkerWithDpRank> {
        match self {
            Self::Worker(worker) => Some(worker),
            Self::CacheOwner(_) => None,
        }
    }
}

/// Scope of a cache-residency reset.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ResetScope {
    #[default]
    All,
    Domain(ResidencyDomain),
}

/// Tolerant wire representation for the additive `residency_domain` field.
///
/// `Missing` is the legacy compatibility signal. Unsupported values stay at
/// the wire boundary and never enter [`ResidencyDomain`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum WireResidencyDomain {
    #[default]
    Missing,
    Known(ResidencyDomain),
    Unknown(Box<str>),
    Invalid,
}

impl WireResidencyDomain {
    pub fn explicit(domain: ResidencyDomain) -> Self {
        Self::Known(domain)
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }

    pub fn parse(&self) -> Result<Option<ResidencyDomain>, UnsupportedResidencyDomain> {
        match self {
            Self::Missing => Ok(None),
            Self::Known(domain) => Ok(Some(*domain)),
            Self::Unknown(_) | Self::Invalid => Err(UnsupportedResidencyDomain),
        }
    }
}

impl Serialize for WireResidencyDomain {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Missing | Self::Invalid => serializer.serialize_none(),
            Self::Known(ResidencyDomain::Worker) => serializer.serialize_str("worker"),
            Self::Known(ResidencyDomain::CacheOwner) => serializer.serialize_str("cache_owner"),
            Self::Unknown(value) => serializer.serialize_str(value),
        }
    }
}

struct WireResidencyDomainVisitor;

impl<'de> Visitor<'de> for WireResidencyDomainVisitor {
    type Value = WireResidencyDomain;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a residency-domain string")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(match value {
            "worker" => WireResidencyDomain::Known(ResidencyDomain::Worker),
            "cache_owner" => WireResidencyDomain::Known(ResidencyDomain::CacheOwner),
            value => WireResidencyDomain::Unknown(value.into()),
        })
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(match value.as_str() {
            "worker" => WireResidencyDomain::Known(ResidencyDomain::Worker),
            "cache_owner" => WireResidencyDomain::Known(ResidencyDomain::CacheOwner),
            _ => WireResidencyDomain::Unknown(value.into_boxed_str()),
        })
    }

    fn visit_char<E>(self, _value: char) -> Result<Self::Value, E> {
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_bytes<E>(self, _value: &[u8]) -> Result<Self::Value, E> {
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_byte_buf<E>(self, _value: Vec<u8>) -> Result<Self::Value, E> {
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        // Explicit null is malformed. Only an omitted field selects legacy
        // semantics; treating null as omission would turn a malformed clear
        // into an all-domain reset.
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        // MessagePack nil follows the same contract as JSON null above.
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element::<IgnoredAny>()?.is_some() {}
        Ok(WireResidencyDomain::Invalid)
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(WireResidencyDomain::Invalid)
    }
}

impl<'de> Deserialize<'de> for WireResidencyDomain {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(WireResidencyDomainVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unsupported residency domain")]
pub struct UnsupportedResidencyDomain;

impl StorageTier {
    pub fn from_kv_medium(medium: &str) -> Option<Self> {
        match medium {
            "GPU" | "DEVICE" => Some(Self::Device),
            "CPU" | "CPU_PINNED" | "CPU_TIER1" => Some(Self::HostPinned),
            "CPU_TIER2" | "DISK" | "NVME" | "STORAGE" => Some(Self::Disk),
            "EXTERNAL" | "NETWORK" | "REMOTE" | "SHARED" => Some(Self::External),
            _ => None,
        }
    }

    /// Canonical wire-format medium string. `None` for the default GPU tier so
    /// existing consumers that omit the field continue to round-trip.
    pub fn to_kv_medium(self) -> Option<&'static str> {
        match self {
            Self::Device => None,
            Self::HostPinned => Some("CPU_PINNED"),
            Self::Disk => Some("DISK"),
            Self::External => Some("EXTERNAL"),
        }
    }

    pub fn is_gpu(self) -> bool {
        matches!(self, Self::Device)
    }
}
