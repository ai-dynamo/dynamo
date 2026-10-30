// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared identities for routing partitions, logical KV indexers, and cluster-local producer pools.
//!
//! Hashed indexer identity material is resolved on control paths. Mutation and query paths carry
//! only those fixed-size values or physical lane indices.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{Error as _, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub const MAX_EXPLICIT_IDENTITY_ENTRIES: usize = 32;
pub const MAX_EXPLICIT_IDENTITY_KEY_BYTES: usize = 128;
pub const MAX_EXPLICIT_IDENTITY_VALUE_BYTES: usize = 1024;

/// Routing group used when a request does not specify one.
pub use dynamo_router_protocol::identity::{
    CacheOwnerId, CacheSemanticsId, ClusterId, DEFAULT_ROUTING_GROUP, IdentityParseError,
    IdentitySource, IndexerDomainId, PoolId, RoutingPartitionId, RoutingPartitionRef,
    RoutingScopeId, StableDpSlotId,
};

const CACHE_SEMANTICS_DEFAULT_V1: &[u8] = b"dynamo/indexer-cache-semantics/default/v1";
const CACHE_SEMANTICS_EXPLICIT_V1: &[u8] = b"dynamo/indexer-cache-semantics/explicit/v1";
const ROUTING_SCOPE_DEFAULT_V1: &[u8] = b"dynamo/indexer-routing-scope/default/v1";
const ROUTING_SCOPE_EXPLICIT_V1: &[u8] = b"dynamo/indexer-routing-scope/explicit/v1";
const STABLE_DP_SLOT_EXPLICIT_V1: &[u8] = b"dynamo/stable-dp-slot/explicit/v1";

#[cfg(any(
    feature = "standalone-indexer",
    feature = "standalone-selection",
    feature = "standalone-slot-tracker"
))]
pub(crate) fn default_routing_group() -> String {
    DEFAULT_ROUTING_GROUP.to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct IndexerIdentitySpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    semantics: Option<ExplicitIdentityMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    routing_scope: Option<ExplicitIdentityMap>,
}

impl IndexerIdentitySpec {
    pub fn new(
        semantics: Option<ExplicitIdentityMap>,
        routing_scope: Option<ExplicitIdentityMap>,
    ) -> Self {
        Self {
            semantics,
            routing_scope,
        }
    }

    pub fn semantics(&self) -> Option<&ExplicitIdentityMap> {
        self.semantics.as_ref()
    }

    pub fn routing_scope(&self) -> Option<&ExplicitIdentityMap> {
        self.routing_scope.as_ref()
    }

    pub const fn is_empty(&self) -> bool {
        self.semantics.is_none() && self.routing_scope.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplicitIdentityMap {
    entries: BTreeMap<String, String>,
}

impl ExplicitIdentityMap {
    pub fn new(entries: BTreeMap<String, String>) -> Result<Self, IdentitySpecError> {
        validate_entries(&entries)?;
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &BTreeMap<String, String> {
        &self.entries
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdentitySpecError {
    #[error("explicit identity map must contain at least one entry")]
    Empty,
    #[error("explicit identity map contains more than {MAX_EXPLICIT_IDENTITY_ENTRIES} entries")]
    TooManyEntries,
    #[error("explicit identity key must not be empty")]
    EmptyKey,
    #[error("explicit identity value for `{key}` must not be empty")]
    EmptyValue { key: String },
    #[error("explicit identity key exceeds {MAX_EXPLICIT_IDENTITY_KEY_BYTES} UTF-8 bytes: `{key}`")]
    KeyTooLong { key: String },
    #[error(
        "explicit identity value for `{key}` exceeds {MAX_EXPLICIT_IDENTITY_VALUE_BYTES} UTF-8 bytes"
    )]
    ValueTooLong { key: String },
}

impl Serialize for ExplicitIdentityMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for (key, value) in &self.entries {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for ExplicitIdentityMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ExplicitIdentityMapVisitor;

        impl<'de> Visitor<'de> for ExplicitIdentityMapVisitor {
            type Value = ExplicitIdentityMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a non-empty map of unique identity keys to string values")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries = BTreeMap::new();
                while let Some((key, value)) = access.next_entry::<String, String>()? {
                    if entries.insert(key.clone(), value).is_some() {
                        return Err(A::Error::custom(format!(
                            "duplicate explicit identity key `{key}`"
                        )));
                    }
                }
                ExplicitIdentityMap::new(entries).map_err(A::Error::custom)
            }
        }

        deserializer.deserialize_map(ExplicitIdentityMapVisitor)
    }
}

/// Canonical bytes hashed by a runtime or service layer that owns BLAKE3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalIdentityMaterial {
    source: IdentitySource,
    bytes: Vec<u8>,
}

impl CanonicalIdentityMaterial {
    pub fn cache_semantics(
        defaults: &[&str],
        explicit: Option<&ExplicitIdentityMap>,
        kv_block_size: u32,
        event_hash_format: u16,
    ) -> Self {
        let (source, tag) = match explicit {
            Some(_) => (IdentitySource::Explicit, CACHE_SEMANTICS_EXPLICIT_V1),
            None => (IdentitySource::DefaultDerived, CACHE_SEMANTICS_DEFAULT_V1),
        };
        let mut bytes = Vec::new();
        append_framed(&mut bytes, tag);
        append_selected_material(&mut bytes, defaults, explicit);
        bytes.extend_from_slice(&kv_block_size.to_le_bytes());
        bytes.extend_from_slice(&event_hash_format.to_le_bytes());
        Self { source, bytes }
    }

    pub fn routing_scope(defaults: &[&str], explicit: Option<&ExplicitIdentityMap>) -> Self {
        let (source, tag) = match explicit {
            Some(_) => (IdentitySource::Explicit, ROUTING_SCOPE_EXPLICIT_V1),
            None => (IdentitySource::DefaultDerived, ROUTING_SCOPE_DEFAULT_V1),
        };
        let mut bytes = Vec::new();
        append_framed(&mut bytes, tag);
        append_selected_material(&mut bytes, defaults, explicit);
        Self { source, bytes }
    }

    /// Canonical material for a stable DP slot.
    ///
    /// Unlike indexer-domain identities, a durable slot has no derived
    /// fallback. Callers must supply explicit, deployment-stable material.
    pub fn stable_dp_slot(explicit: &ExplicitIdentityMap) -> Self {
        let mut bytes = Vec::new();
        append_framed(&mut bytes, STABLE_DP_SLOT_EXPLICIT_V1);
        append_selected_material(&mut bytes, &[], Some(explicit));
        Self {
            source: IdentitySource::Explicit,
            bytes,
        }
    }

    pub const fn source(&self) -> IdentitySource {
        self.source
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

fn append_selected_material(
    bytes: &mut Vec<u8>,
    defaults: &[&str],
    explicit: Option<&ExplicitIdentityMap>,
) {
    match explicit {
        Some(explicit) => {
            append_count(bytes, explicit.entries.len());
            for (key, value) in &explicit.entries {
                append_framed(bytes, key.as_bytes());
                append_framed(bytes, value.as_bytes());
            }
        }
        None => {
            append_count(bytes, defaults.len());
            for value in defaults {
                append_framed(bytes, value.as_bytes());
            }
        }
    }
}

fn append_count(bytes: &mut Vec<u8>, count: usize) {
    let count = u32::try_from(count).expect("identity input count is validated to fit u32");
    bytes.extend_from_slice(&count.to_le_bytes());
}

fn append_framed(bytes: &mut Vec<u8>, value: &[u8]) {
    let len = u32::try_from(value.len()).expect("identity input length is validated to fit u32");
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(value);
}

fn validate_entries(entries: &BTreeMap<String, String>) -> Result<(), IdentitySpecError> {
    if entries.is_empty() {
        return Err(IdentitySpecError::Empty);
    }
    if entries.len() > MAX_EXPLICIT_IDENTITY_ENTRIES {
        return Err(IdentitySpecError::TooManyEntries);
    }
    for (key, value) in entries {
        if key.is_empty() {
            return Err(IdentitySpecError::EmptyKey);
        }
        if key.len() > MAX_EXPLICIT_IDENTITY_KEY_BYTES {
            return Err(IdentitySpecError::KeyTooLong { key: key.clone() });
        }
        if value.is_empty() {
            return Err(IdentitySpecError::EmptyValue { key: key.clone() });
        }
        if value.len() > MAX_EXPLICIT_IDENTITY_VALUE_BYTES {
            return Err(IdentitySpecError::ValueTooLong { key: key.clone() });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn routing_partition_identity_matches_registry_key_semantics() {
        let partition = RoutingPartitionId::new("model-a", "group-a");
        let equivalent = RoutingPartitionId::new("model-a", "group-a");
        let other_group = RoutingPartitionId::new("model-a", "group-b");
        let other_model = RoutingPartitionId::new("model-b", "group-a");
        let mut registry_keys = HashSet::new();

        assert!(registry_keys.insert(partition.clone()));
        assert!(!registry_keys.insert(equivalent));
        assert!(registry_keys.insert(other_group));
        assert!(registry_keys.insert(other_model));
        assert_eq!(registry_keys.len(), 3);

        let borrowed = partition.as_ref();
        assert_eq!(borrowed, RoutingPartitionRef::new("model-a", "group-a"));
        assert_eq!(borrowed.into_owned(), partition);
    }

    #[test]
    fn explicit_identity_map_rejects_duplicate_json_keys() {
        let error = serde_json::from_str::<ExplicitIdentityMap>(r#"{"weights":"a","weights":"b"}"#)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate explicit identity key")
        );
    }

    #[test]
    fn explicit_identity_map_rejects_empty_and_oversized_inputs() {
        assert_eq!(
            ExplicitIdentityMap::new(BTreeMap::new()),
            Err(IdentitySpecError::Empty)
        );
        assert_eq!(
            ExplicitIdentityMap::new(BTreeMap::from([(String::new(), "value".to_string(),)])),
            Err(IdentitySpecError::EmptyKey)
        );
        assert!(matches!(
            ExplicitIdentityMap::new(BTreeMap::from([(
                "key".to_string(),
                "x".repeat(MAX_EXPLICIT_IDENTITY_VALUE_BYTES + 1),
            )])),
            Err(IdentitySpecError::ValueTooLong { .. })
        ));
    }

    #[test]
    fn explicit_material_replaces_defaults_and_is_order_independent() {
        let first = ExplicitIdentityMap::new(BTreeMap::from([
            ("weights".to_string(), "revision-a".to_string()),
            ("mapping".to_string(), "tp2".to_string()),
        ]))
        .unwrap();
        let second: ExplicitIdentityMap =
            serde_json::from_str(r#"{"mapping":"tp2","weights":"revision-a"}"#).unwrap();
        let a = CanonicalIdentityMaterial::cache_semantics(&["default-a"], Some(&first), 512, 1);
        let b = CanonicalIdentityMaterial::cache_semantics(&["default-b"], Some(&second), 512, 1);
        assert_eq!(a, b);
        assert_eq!(a.source(), IdentitySource::Explicit);
    }

    #[test]
    fn default_and_explicit_material_are_distinct() {
        let explicit = ExplicitIdentityMap::new(BTreeMap::from([(
            "model".to_string(),
            "same-text".to_string(),
        )]))
        .unwrap();
        let default = CanonicalIdentityMaterial::cache_semantics(&["same-text"], None, 512, 1);
        let explicit =
            CanonicalIdentityMaterial::cache_semantics(&["ignored"], Some(&explicit), 512, 1);
        assert_ne!(default.bytes(), explicit.bytes());
        assert_eq!(default.source(), IdentitySource::DefaultDerived);
        assert_eq!(explicit.source(), IdentitySource::Explicit);
    }

    #[test]
    fn length_framing_distinguishes_adjacent_inputs() {
        let first = CanonicalIdentityMaterial::routing_scope(&["ab", "c"], None);
        let second = CanonicalIdentityMaterial::routing_scope(&["a", "bc"], None);
        assert_ne!(first.bytes(), second.bytes());
    }

    #[test]
    fn identifiers_render_fixed_width_hex() {
        let semantics = CacheSemanticsId::new([0xab; 16], IdentitySource::Explicit);
        let routing = RoutingScopeId::new([0x01; 16], IdentitySource::DefaultDerived);
        assert_eq!(semantics.to_string(), "abababababababababababababababab");
        assert_eq!(routing.to_string(), "01010101010101010101010101010101");
        assert_eq!(ClusterId::new(1).to_string(), "0000000000000001");
    }

    #[test]
    fn pool_id_json_uses_cluster_id() {
        let pool = PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([0xab; 16], IdentitySource::Explicit),
                RoutingScopeId::new([0xcd; 16], IdentitySource::Explicit),
            ),
            ClusterId::new(7),
        );
        let json = serde_json::to_value(pool).unwrap();
        assert_eq!(json["cluster_id"], 7);
        assert!(json.get("dc_id").is_none());
        assert_eq!(
            serde_json::from_value::<PoolId>(json.clone()).unwrap(),
            pool
        );
        let mut old_json = json;
        let old_cluster_id = old_json
            .as_object_mut()
            .unwrap()
            .remove("cluster_id")
            .unwrap();
        old_json["dc_id"] = old_cluster_id;
        assert!(serde_json::from_value::<PoolId>(old_json).is_err());
    }

    #[test]
    fn cache_owner_canonical_string_round_trips_at_configuration_boundary() {
        let owner = CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([0xab; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([0xcd; 16], IdentitySource::Explicit),
                ),
                ClusterId::new(7),
            ),
            StableDpSlotId::new([0xef; 16], IdentitySource::Explicit),
        );
        assert_eq!(owner.to_string().parse::<CacheOwnerId>().unwrap(), owner);
        assert!("ab/7/ef".parse::<CacheOwnerId>().is_err());
    }
}
