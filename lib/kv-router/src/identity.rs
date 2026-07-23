// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compact identities for logical KV indexers and DC-local producer pools.
//!
//! Identity material is resolved on control paths. Mutation and query paths carry only these
//! fixed-size values or physical lane indices.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{Error as _, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub const MAX_EXPLICIT_IDENTITY_ENTRIES: usize = 32;
pub const MAX_EXPLICIT_IDENTITY_KEY_BYTES: usize = 128;
pub const MAX_EXPLICIT_IDENTITY_VALUE_BYTES: usize = 1024;

const CACHE_SEMANTICS_DEFAULT_V1: &[u8] = b"dynamo/indexer-cache-semantics/default/v1";
const CACHE_SEMANTICS_EXPLICIT_V1: &[u8] = b"dynamo/indexer-cache-semantics/explicit/v1";
const ROUTING_SCOPE_DEFAULT_V1: &[u8] = b"dynamo/indexer-routing-scope/default/v1";
const ROUTING_SCOPE_EXPLICIT_V1: &[u8] = b"dynamo/indexer-routing-scope/explicit/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentitySource {
    DefaultDerived,
    Explicit,
}

macro_rules! digest_identity {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name {
            digest: [u8; 16],
            source: IdentitySource,
        }

        impl $name {
            pub const fn new(digest: [u8; 16], source: IdentitySource) -> Self {
                Self { digest, source }
            }

            pub const fn digest(self) -> [u8; 16] {
                self.digest
            }

            pub const fn source(self) -> IdentitySource {
                self.source
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write_digest(formatter, &self.digest)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    .field("digest", &format_args!("{}", self))
                    .field("source", &self.source)
                    .finish()
            }
        }
    };
}

digest_identity!(CacheSemanticsId);
digest_identity!(RoutingScopeId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IndexerDomainId {
    cache_semantics: CacheSemanticsId,
    routing_scope: RoutingScopeId,
}

impl IndexerDomainId {
    pub const fn new(cache_semantics: CacheSemanticsId, routing_scope: RoutingScopeId) -> Self {
        Self {
            cache_semantics,
            routing_scope,
        }
    }

    pub const fn cache_semantics(self) -> CacheSemanticsId {
        self.cache_semantics
    }

    pub const fn routing_scope(self) -> RoutingScopeId {
        self.routing_scope
    }

    pub const fn relies_on_defaults(self) -> bool {
        matches!(
            self.cache_semantics.source(),
            IdentitySource::DefaultDerived
        ) || matches!(self.routing_scope.source(), IdentitySource::DefaultDerived)
    }
}

impl fmt::Display for IndexerDomainId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.cache_semantics, self.routing_scope)
    }
}

/// Control-plane-stable identity for one logical DC inside a routing federation.
///
/// NOTE: This value survives process restarts, scaling, endpoint replacement, and producer
/// generations. It is meaningful only as the DC dimension of [`PoolId`], not as a globally
/// unique identifier. Do not derive it from endpoint identity or fold it into routing scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DcId(u64);

impl DcId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for DcId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:016x}", self.0)
    }
}

/// One DC-local producer/publication stream within an indexer domain.
///
/// NOTE: A global indexer has one domain and distinct pool lanes whose `dc_id` values differ.
/// Runtime endpoint resolution happens only after a query selects a pool lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PoolId {
    indexer_domain: IndexerDomainId,
    dc_id: DcId,
}

impl PoolId {
    pub const fn new(indexer_domain: IndexerDomainId, dc_id: DcId) -> Self {
        Self {
            indexer_domain,
            dc_id,
        }
    }

    pub const fn indexer_domain(self) -> IndexerDomainId {
        self.indexer_domain
    }

    pub const fn dc_id(self) -> DcId {
        self.dc_id
    }
}

impl fmt::Display for PoolId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.indexer_domain, self.dc_id)
    }
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

fn write_digest(formatter: &mut fmt::Formatter<'_>, digest: &[u8; 16]) -> fmt::Result {
    for byte in digest {
        write!(formatter, "{byte:02x}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(DcId::new(1).to_string(), "0000000000000001");
    }
}
