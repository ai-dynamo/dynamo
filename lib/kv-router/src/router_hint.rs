// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-generated hints that are attached to selected backend requests.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[cfg(feature = "runtime-protocols")]
use dynamo_runtime::protocols::EndpointId;

#[cfg(feature = "runtime-protocols")]
const MAX_JSON_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

use crate::protocols::{
    ExternalSequenceBlockHash, ResidencyOwnerKey, ResidencyRoutingSnapshot, WorkerWithDpRank,
};

/// Key for router-generated backend hints inside KV transfer params.
pub const ROUTER_HINT_EXTRA_ARGS_KEY: &str = "router_hint";

/// Worker runtime_data key. Boolean true means the worker can consume router_hint extra args.
pub const ROUTER_HINT_RUNTIME_CAPABILITY_KEY: &str = "router_hint";

/// Worker runtime_data key for matching router-hint sources to targets by backend role.
pub const ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY: &str = "router_hint_worker_type";

/// Worker runtime_data key for per-global-DP-rank advertised KVCR control endpoints.
pub const ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "router_hint_source_control_endpoints";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouterHint {
    pub source_control_endpoint: String,
    /// Root-aligned source-side KV block hashes. `block_hashes[i]`
    /// corresponds to request block `i`; the target decides which suffix to fetch.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

#[cfg(feature = "runtime-protocols")]
pub(crate) fn validate_opaque_key(name: &str, value: &str, max_len: usize) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    if value.len() > max_len {
        return Err(format!("{name} exceeds the {max_len}-byte limit"));
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/' | b':')
    }) {
        return Err(format!("{name} contains unsupported characters"));
    }
    Ok(())
}

/// Backend-owned transport coordinates copied from a validated draft registration.
#[cfg(feature = "runtime-protocols")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DraftTransport {
    pub protocol: String,
    pub address: String,
    pub orphan_cleanup_timeout_ms: u32,
}

/// Exact draft identity selected for one speculative request.
#[cfg(feature = "runtime-protocols")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeculativeDecodingRouterHintV1 {
    pub schema_version: u16,
    pub draft_endpoint: EndpointId,
    pub draft: WorkerWithDpRank,
    pub draft_incarnation_id: u64,
    pub transport: DraftTransport,
}

#[cfg(feature = "runtime-protocols")]
impl SpeculativeDecodingRouterHintV1 {
    pub const SCHEMA_VERSION: u16 = 1;

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Err(format!(
                "unsupported speculative router-hint schema version {}; expected {}",
                self.schema_version,
                Self::SCHEMA_VERSION
            ));
        }
        validate_endpoint_id(&self.draft_endpoint)?;
        if !(1..=MAX_JSON_SAFE_INTEGER).contains(&self.draft_incarnation_id) {
            return Err("draft_incarnation_id must be a positive JSON-safe integer".into());
        }
        validate_opaque_key("draft protocol", &self.transport.protocol, 128)?;
        validate_transport_address(&self.transport.address)?;
        if !(1..=300_000).contains(&self.transport.orphan_cleanup_timeout_ms) {
            return Err("draft orphan_cleanup_timeout_ms must be between 1 and 300000".into());
        }
        Ok(())
    }
}

/// Shared router-to-worker envelope. The legacy KV fields remain flattened at the top level.
#[cfg(feature = "runtime-protocols")]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RouterHintEnvelope {
    pub kv_transfer: Option<RouterHint>,
    pub speculative_decoding: Option<SpeculativeDecodingRouterHintV1>,
}

#[cfg(feature = "runtime-protocols")]
impl RouterHintEnvelope {
    pub fn kv(hint: RouterHint) -> Self {
        Self {
            kv_transfer: Some(hint),
            speculative_decoding: None,
        }
    }

    pub fn speculative(hint: SpeculativeDecodingRouterHintV1) -> Self {
        Self {
            kv_transfer: None,
            speculative_decoding: Some(hint),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.kv_transfer.is_none() && self.speculative_decoding.is_none() {
            return Err("router-hint envelope must contain at least one section".into());
        }
        if let Some(kv_transfer) = &self.kv_transfer {
            if kv_transfer.source_control_endpoint.is_empty() {
                return Err("source_control_endpoint must not be empty".into());
            }
            if kv_transfer.block_hashes.is_empty() {
                return Err("block_hashes must not be empty".into());
            }
        }
        if let Some(speculative) = &self.speculative_decoding {
            speculative.validate()?;
        }
        Ok(())
    }
}

#[cfg(feature = "runtime-protocols")]
impl Serialize for RouterHintEnvelope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        self.validate().map_err(serde::ser::Error::custom)?;
        let mut map = serializer.serialize_map(None)?;
        if let Some(kv_transfer) = &self.kv_transfer {
            map.serialize_entry(
                "source_control_endpoint",
                &kv_transfer.source_control_endpoint,
            )?;
            map.serialize_entry("block_hashes", &kv_transfer.block_hashes)?;
        }
        if let Some(speculative) = &self.speculative_decoding {
            map.serialize_entry("speculative_decoding", speculative)?;
        }
        map.end()
    }
}

#[cfg(feature = "runtime-protocols")]
impl<'de> Deserialize<'de> for RouterHintEnvelope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireEnvelope {
            #[serde(default)]
            source_control_endpoint: Option<String>,
            #[serde(default)]
            block_hashes: Option<Vec<ExternalSequenceBlockHash>>,
            #[serde(default)]
            speculative_decoding: Option<SpeculativeDecodingRouterHintV1>,
        }

        let wire = WireEnvelope::deserialize(deserializer)?;
        let kv_transfer = match (wire.source_control_endpoint, wire.block_hashes) {
            (Some(source_control_endpoint), Some(block_hashes)) => Some(RouterHint {
                source_control_endpoint,
                block_hashes,
            }),
            (None, None) => None,
            _ => {
                return Err(serde::de::Error::custom(
                    "router hint must contain both legacy KV fields or neither",
                ));
            }
        };
        let envelope = Self {
            kv_transfer,
            speculative_decoding: wire.speculative_decoding,
        };
        envelope.validate().map_err(serde::de::Error::custom)?;
        Ok(envelope)
    }
}

#[cfg(feature = "runtime-protocols")]
pub fn validate_endpoint_id(endpoint: &EndpointId) -> Result<(), String> {
    for (name, value) in [
        ("namespace", endpoint.namespace.as_str()),
        ("component", endpoint.component.as_str()),
        ("endpoint", endpoint.name.as_str()),
    ] {
        validate_opaque_key(name, value, 255)?;
    }
    Ok(())
}

#[cfg(feature = "runtime-protocols")]
pub fn validate_transport_address(value: &str) -> Result<(), String> {
    const MAX_ADDRESS_LEN: usize = 2048;
    if value.is_empty() {
        return Err("draft transport address must not be empty".into());
    }
    if value.len() > MAX_ADDRESS_LEN {
        return Err(format!(
            "draft transport address exceeds the {MAX_ADDRESS_LEN}-byte limit"
        ));
    }
    if value.chars().any(char::is_control) {
        return Err("draft transport address contains control characters".into());
    }

    let (scheme, authority_and_path) = value
        .split_once("://")
        .ok_or_else(|| "draft transport address must include a URI scheme".to_string())?;
    if scheme.is_empty()
        || !scheme
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
    {
        return Err("draft transport address has an invalid URI scheme".into());
    }
    let authority = authority_and_path.split('/').next().unwrap_or_default();
    if authority.is_empty() {
        return Err("draft transport address must include a non-empty authority".into());
    }
    if authority.contains('@') {
        return Err("draft transport address must not contain credentials".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RouterHintCandidateSource {
    Worker(WorkerWithDpRank),
    CacheOwner(ResidencyOwnerKey),
}

impl From<WorkerWithDpRank> for RouterHintCandidateSource {
    fn from(worker: WorkerWithDpRank) -> Self {
        Self::Worker(worker)
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterHintRootCandidates {
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    pub owner_prefix_blocks: Vec<(RouterHintCandidateSource, usize)>,
    pub routing_snapshot: Option<Arc<ResidencyRoutingSnapshot>>,
}

impl RouterHintRootCandidates {
    pub fn best_source<F>(
        &self,
        prefix_blocks_to_beat: usize,
        mut is_eligible_source: F,
    ) -> Option<(RouterHintCandidateSource, Vec<ExternalSequenceBlockHash>)>
    where
        F: FnMut(RouterHintCandidateSource) -> bool,
    {
        let (source, prefix_blocks) = self
            .owner_prefix_blocks
            .iter()
            .copied()
            .filter(|(worker, blocks)| {
                *blocks > prefix_blocks_to_beat && is_eligible_source(*worker)
            })
            .max_by(|(left_worker, left_blocks), (right_worker, right_blocks)| {
                left_blocks
                    .cmp(right_blocks)
                    .then_with(|| right_worker.cmp(left_worker))
            })?;

        Some((source, self.block_hashes.get(..prefix_blocks)?.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_source_selects_longest_eligible_prefix() {
        let worker_a = WorkerWithDpRank::new(7, 0);
        let worker_b = WorkerWithDpRank::new(8, 0);
        let excluded = WorkerWithDpRank::new(9, 0);
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
            ],
            owner_prefix_blocks: vec![
                (worker_b.into(), 2),
                (excluded.into(), 3),
                (worker_a.into(), 3),
            ],
            routing_snapshot: None,
        };

        let selected = candidates.best_source(0, |source| source != excluded.into());

        assert_eq!(
            selected,
            Some((
                worker_a.into(),
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                ],
            ))
        );
    }

    #[test]
    fn best_source_fails_closed_on_invalid_prefix_length() {
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![ExternalSequenceBlockHash(101)],
            owner_prefix_blocks: vec![(WorkerWithDpRank::new(7, 0).into(), 2)],
            routing_snapshot: None,
        };

        assert!(candidates.best_source(0, |_| true).is_none());
    }

    #[test]
    fn best_source_requires_prefix_longer_than_threshold() {
        let worker_a = WorkerWithDpRank::new(7, 0);
        let worker_b = WorkerWithDpRank::new(8, 0);
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
                ExternalSequenceBlockHash(104),
            ],
            owner_prefix_blocks: vec![(worker_a.into(), 3), (worker_b.into(), 4)],
            routing_snapshot: None,
        };

        assert!(
            candidates
                .best_source(3, |source| source == worker_a.into())
                .is_none()
        );
        assert_eq!(
            candidates.best_source(3, |_| true),
            Some((
                worker_b.into(),
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                    ExternalSequenceBlockHash(104),
                ],
            ))
        );
    }

    #[cfg(feature = "runtime-protocols")]
    fn speculative_hint() -> SpeculativeDecodingRouterHintV1 {
        SpeculativeDecodingRouterHintV1 {
            schema_version: 1,
            draft_endpoint: EndpointId::from("specdec/draft/generate"),
            draft: WorkerWithDpRank::new(42, 1),
            draft_incarnation_id: 7,
            transport: DraftTransport {
                protocol: "mock-specdec-zmq-v1".into(),
                address: "tcp://draft-2:50051".into(),
                orphan_cleanup_timeout_ms: 1_000,
            },
        }
    }

    #[cfg(feature = "runtime-protocols")]
    #[test]
    fn kv_only_envelope_preserves_legacy_wire_shape() {
        let hint = RouterHint {
            source_control_endpoint: "tcp://source:23280".into(),
            block_hashes: vec![ExternalSequenceBlockHash(11), ExternalSequenceBlockHash(22)],
        };
        assert_eq!(
            serde_json::to_string(&RouterHintEnvelope::kv(hint.clone())).unwrap(),
            serde_json::to_string(&hint).unwrap()
        );
    }

    #[cfg(feature = "runtime-protocols")]
    #[test]
    fn envelope_supports_speculative_only_and_combined_forms() {
        let speculative = speculative_hint();
        let spec_only = RouterHintEnvelope::speculative(speculative.clone());
        let spec_json = serde_json::to_value(&spec_only).unwrap();
        assert!(spec_json.get("source_control_endpoint").is_none());
        assert_eq!(spec_json["speculative_decoding"]["schema_version"], 1);
        assert_eq!(
            serde_json::from_value::<RouterHintEnvelope>(spec_json).unwrap(),
            spec_only
        );

        let combined = RouterHintEnvelope {
            kv_transfer: Some(RouterHint {
                source_control_endpoint: "tcp://source:23280".into(),
                block_hashes: vec![ExternalSequenceBlockHash(11)],
            }),
            speculative_decoding: Some(speculative),
        };
        let combined_json = serde_json::to_value(&combined).unwrap();
        assert_eq!(
            combined_json["source_control_endpoint"],
            "tcp://source:23280"
        );
        assert!(combined_json.get("speculative_decoding").is_some());
        assert_eq!(
            serde_json::from_value::<RouterHintEnvelope>(combined_json).unwrap(),
            combined
        );
    }

    #[cfg(feature = "runtime-protocols")]
    #[test]
    fn envelope_rejects_half_legacy_or_invalid_speculative_sections() {
        assert!(
            serde_json::from_value::<RouterHintEnvelope>(serde_json::json!({
                "source_control_endpoint": "tcp://source:23280"
            }))
            .is_err()
        );

        let mut invalid = speculative_hint();
        invalid.schema_version = 2;
        assert!(serde_json::to_value(RouterHintEnvelope::speculative(invalid)).is_err());

        let mut invalid = speculative_hint();
        invalid.transport.orphan_cleanup_timeout_ms = 0;
        assert!(serde_json::to_value(RouterHintEnvelope::speculative(invalid)).is_err());

        let mut invalid = speculative_hint();
        invalid.draft_incarnation_id = MAX_JSON_SAFE_INTEGER + 1;
        assert!(serde_json::to_value(RouterHintEnvelope::speculative(invalid)).is_err());
    }
}
