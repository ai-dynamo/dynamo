// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed delivery controls. Backend adapters own artifact contents and encoding.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

/// Decode JSON without discarding repeated keys at any object depth.
struct UniqueJsonValue(Value);

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct UniqueValueVisitor;

        impl<'de> Visitor<'de> for UniqueValueVisitor {
            type Value = UniqueJsonValue;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a JSON value without duplicate object fields")
            }

            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(Value::Bool(value)))
            }

            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(Value::from(value)))
            }

            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(Value::from(value)))
            }

            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|number| UniqueJsonValue(Value::Number(number)))
                    .ok_or_else(|| E::custom("invalid metadata_upload number"))
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(Value::String(value.to_owned())))
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(Value::String(value)))
            }

            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(Value::Null))
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(Value::Null))
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(UniqueJsonValue(value)) = sequence.next_element()? {
                    values.push(value);
                }
                Ok(UniqueJsonValue(Value::Array(values)))
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut values = serde_json::Map::new();
                while let Some((key, UniqueJsonValue(value))) = map.next_entry()? {
                    if values.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate metadata_upload field"));
                    }
                }
                Ok(UniqueJsonValue(Value::Object(values)))
            }
        }

        deserializer.deserialize_any(UniqueValueVisitor)
    }
}

/// The legacy prefix contract and the explicitly selected artifact contract.
#[derive(Clone, PartialEq)]
pub enum MetadataUpload {
    Legacy(LegacyMetadataUpload),
    GenerationArtifact(GenerationArtifactUpload),
}

impl MetadataUpload {
    pub fn is_artifact(&self) -> bool {
        matches!(self, Self::GenerationArtifact(_))
    }
}

impl Serialize for MetadataUpload {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Legacy(upload) => upload.serialize(serializer),
            Self::GenerationArtifact(upload) => upload.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for MetadataUpload {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let UniqueJsonValue(value) = UniqueJsonValue::deserialize(deserializer)?;
        let object = value
            .as_object()
            .ok_or_else(|| serde::de::Error::custom("metadata_upload must be an object"))?;
        match object.get("kind") {
            None => {
                if object
                    .keys()
                    .any(|name| name != "url" && name != "fallback_url")
                {
                    return Err(serde::de::Error::custom(
                        "unknown legacy metadata_upload field",
                    ));
                }
                if !object.contains_key("url") {
                    return Err(serde::de::Error::custom(
                        "missing legacy metadata_upload url",
                    ));
                }
                let mut upload: LegacyMetadataUpload =
                    serde_json::from_value(value).map_err(|_| {
                        serde::de::Error::custom("invalid legacy metadata_upload shape")
                    })?;
                upload.url =
                    normalized_legacy_url(&upload.url).map_err(serde::de::Error::custom)?;
                upload.fallback_url = upload
                    .fallback_url
                    .as_deref()
                    .map(normalized_legacy_url)
                    .transpose()
                    .map_err(serde::de::Error::custom)?;
                Ok(Self::Legacy(upload))
            }
            Some(Value::String(kind)) if kind == "generation_artifact" => {
                let destination_present = object.contains_key("destination");
                let upload: GenerationArtifactUpload =
                    serde_json::from_value(value).map_err(|_| {
                        serde::de::Error::custom(
                            "invalid generation_artifact metadata_upload shape",
                        )
                    })?;
                upload
                    .validate(destination_present)
                    .map_err(serde::de::Error::custom)?;
                Ok(Self::GenerationArtifact(upload))
            }
            Some(_) => Err(serde::de::Error::custom("unsupported metadata_upload kind")),
        }
    }
}

impl fmt::Debug for MetadataUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Legacy(_) => f.write_str("MetadataUpload::Legacy([REDACTED])"),
            Self::GenerationArtifact(upload) => f
                .debug_tuple("MetadataUpload::GenerationArtifact")
                .field(upload)
                .finish(),
        }
    }
}

/// Existing metadata upload prefixes; no discriminator is emitted.
#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LegacyMetadataUpload {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_url: Option<String>,
}

fn normalized_legacy_url(value: &str) -> Result<String, &'static str> {
    let value = value.trim();
    if value.is_empty() {
        return Err("metadata_upload URL must not be empty");
    }
    Ok(value.to_string())
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    GenerationArtifact,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactMode {
    InlineBase64,
    ObjectStore,
    RelayStore,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GenerationArtifactUpload {
    pub kind: ArtifactKind,
    /// Opaque names admitted and interpreted by the selected backend adapter.
    pub contents: Vec<String>,
    pub mode: ArtifactMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub destination: Option<ArtifactDestination>,
}

impl GenerationArtifactUpload {
    fn validate(&self, destination_present: bool) -> Result<(), &'static str> {
        let mut contents = HashSet::new();
        if self.contents.len() > 64
            || self.contents.iter().any(|name| {
                name.is_empty() || name.len() > 128 || name.trim() != name || !contents.insert(name)
            })
        {
            return Err("artifact contents must contain at most 64 unique nonempty names");
        }
        match (&self.mode, &self.destination) {
            (ArtifactMode::InlineBase64, None) if !destination_present => Ok(()),
            (ArtifactMode::ObjectStore, Some(ArtifactDestination::PresignedHttpPut(target))) => {
                target.validate()
            }
            (ArtifactMode::ObjectStore, Some(ArtifactDestination::ManagedStore(target))) => {
                target.validate()
            }
            (ArtifactMode::RelayStore, Some(ArtifactDestination::MooncakeStore(target))) => {
                target.validate()
            }
            _ => Err("artifact destination is incompatible with the delivery mode"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArtifactDestination {
    PresignedHttpPut(PresignedHttpPutDestination),
    ManagedStore(ManagedStoreDestination),
    MooncakeStore(MooncakeStoreDestination),
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PresignedHttpPutDestination {
    /// An exact PUT capability; never normalize it or append an object key.
    pub url: String,
    pub required_headers: BTreeMap<String, String>,
    pub expires_at: String,
    pub max_bytes: u64,
    pub object_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_ref: Option<String>,
}

impl fmt::Debug for PresignedHttpPutDestination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PresignedHttpPutDestination([REDACTED])")
    }
}

impl PresignedHttpPutDestination {
    fn validate(&self) -> Result<(), &'static str> {
        let url = url::Url::parse(&self.url).map_err(|_| "invalid presigned PUT URL")?;
        if self.url.trim() != self.url
            || !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
        {
            return Err("presigned PUT URL must be HTTP(S) without userinfo or fragment");
        }
        if chrono::DateTime::parse_from_rfc3339(&self.expires_at).is_err() {
            return Err("presigned PUT expires_at must be an RFC3339 timestamp");
        }
        if self.max_bytes == 0
            || !valid_name(&self.object_id)
            || self
                .verification_ref
                .as_deref()
                .is_some_and(|value| !valid_name(value))
        {
            return Err("presigned PUT requires positive max_bytes and nonempty identity fields");
        }
        let mut names = HashSet::new();
        if self.required_headers.len() > 64
            || self.required_headers.iter().any(|(name, value)| {
                axum::http::header::HeaderName::from_bytes(name.as_bytes()).is_err()
                    || axum::http::header::HeaderValue::from_str(value).is_err()
                    || value.len() > 8192
                    || !names.insert(name.to_ascii_lowercase())
            })
        {
            return Err(
                "presigned PUT headers must be valid, bounded and case-insensitively unique",
            );
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ManagedStoreDestination {
    pub profile: String,
    pub object_key: String,
}

impl ManagedStoreDestination {
    fn validate(&self) -> Result<(), &'static str> {
        if !valid_name(&self.profile)
            || self.object_key.is_empty()
            || self.object_key.len() > 1024
            || self.object_key.starts_with('/')
            || self.object_key.contains('\\')
            || self.object_key.contains('\0')
            || self.object_key.contains("://")
            || self
                .object_key
                .split('/')
                .any(|part| matches!(part, "" | "." | ".."))
        {
            return Err(
                "managed store requires a profile and a relative object key without traversal",
            );
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MooncakeStoreDestination {
    pub profile: String,
    pub consumer_group: String,
    pub retention_seconds: u64,
}

impl MooncakeStoreDestination {
    fn validate(&self) -> Result<(), &'static str> {
        if !valid_name(&self.profile)
            || !valid_name(&self.consumer_group)
            || self.retention_seconds == 0
        {
            return Err("Mooncake relay requires a profile, consumer group and positive retention");
        }
        Ok(())
    }
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests;
