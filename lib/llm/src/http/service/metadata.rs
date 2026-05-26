// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP header → context metadata extraction.
//!
//! Any request header whose name starts with `DYNAMO_METADATA_HEADER_PREFIX_DEFAULT`
//! (or the value of the `DYN_METADATA_HEADER` env var) is stripped of its prefix
//! and inserted into the [`dynamo_runtime::pipeline::Context`] metadata map.
//!
//! Example: `x-dynamo-meta-tenant: acme` → `metadata["tenant"] = "acme"`.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use axum::http::HeaderMap;
use tonic::metadata::{KeyAndValueRef, MetadataMap};

/// Default header prefix for context metadata injected from HTTP request headers.
/// Overridable at startup via the [`DYNAMO_METADATA_HEADER_ENV`] environment variable.
pub const DYNAMO_METADATA_HEADER_PREFIX_DEFAULT: &str = "x-dynamo-meta-";

/// Environment variable that overrides [`DYNAMO_METADATA_HEADER_PREFIX_DEFAULT`].
pub const DYNAMO_METADATA_HEADER_ENV: &str = "DYN_METADATA_HEADER";

const DYNAMO_METADATA_MAX_ENTRIES_DEFAULT: usize = 64;
const DYNAMO_METADATA_MAX_TOTAL_BYTES_DEFAULT: usize = 64 * 1024;

static METADATA_HEADER_PREFIX: OnceLock<String> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MetadataHeaderError {
    #[error("metadata headers exceed the limit of {limit} entries")]
    TooManyEntries { limit: usize },
    #[error("metadata headers exceed the limit of {limit_bytes} bytes")]
    TooLarge { limit_bytes: usize },
}

pub(crate) fn metadata_header_prefix() -> &'static str {
    METADATA_HEADER_PREFIX.get_or_init(|| {
        std::env::var(DYNAMO_METADATA_HEADER_ENV)
            .ok()
            .map(|prefix| prefix.trim().to_ascii_lowercase())
            .filter(|prefix| !prefix.is_empty())
            .unwrap_or_else(|| DYNAMO_METADATA_HEADER_PREFIX_DEFAULT.to_string())
    })
}

fn is_sensitive_metadata(raw_key: &str, raw_value: &str) -> bool {
    let value: &str = raw_value.trim_start();
    raw_key.eq_ignore_ascii_case("authorization")
        || value
            .get(.."bearer ".len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("bearer "))
}

fn insert_metadata_entry(
    out: &mut BTreeMap<String, String>,
    total_bytes: &mut usize,
    raw_key: &str,
    raw_value: &str,
) -> Result<(), MetadataHeaderError> {
    if out.contains_key(raw_key) {
        return Ok(());
    }

    if is_sensitive_metadata(raw_key, raw_value) {
        return Ok(());
    }

    if out.len() >= DYNAMO_METADATA_MAX_ENTRIES_DEFAULT {
        return Err(MetadataHeaderError::TooManyEntries {
            limit: DYNAMO_METADATA_MAX_ENTRIES_DEFAULT,
        });
    }

    let value = raw_value.trim();
    let entry_bytes = raw_key.len() + value.len();
    if *total_bytes + entry_bytes > DYNAMO_METADATA_MAX_TOTAL_BYTES_DEFAULT {
        return Err(MetadataHeaderError::TooLarge {
            limit_bytes: DYNAMO_METADATA_MAX_TOTAL_BYTES_DEFAULT,
        });
    }

    *total_bytes += entry_bytes;
    out.insert(raw_key.to_string(), value.to_string());
    Ok(())
}

fn extract_metadata_from_pairs<'a>(
    pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
    prefix: &str,
) -> Result<BTreeMap<String, String>, MetadataHeaderError> {
    let mut out = BTreeMap::new();
    let mut total_bytes = 0;

    for (name, value) in pairs {
        let Some(raw_key) = name.strip_prefix(prefix) else {
            continue;
        };
        insert_metadata_entry(&mut out, &mut total_bytes, raw_key, value)?;
    }

    Ok(out)
}

/// Extract all `<prefix><key>: <value>` headers as a metadata map.
///
/// Headers that are not valid UTF-8 are silently skipped.
/// If a header is repeated, the first value wins.
/// Requests exceeding 64 entries or 64 KiB of key/value payload are rejected.
pub fn extract_metadata_from_http(
    headers: &HeaderMap,
) -> Result<BTreeMap<String, String>, MetadataHeaderError> {
    let prefix = metadata_header_prefix();
    extract_metadata_from_pairs(
        headers
            .iter()
            .filter_map(|(name, value)| value.to_str().ok().map(|value| (name.as_str(), value))),
        prefix,
    )
}

/// Extract all `<prefix><key>: <value>` gRPC metadata entries as a metadata map.
///
/// Binary metadata entries and non-UTF-8 values are ignored.
/// If a key is repeated, the first value wins.
pub fn extract_metadata_from_grpc(
    metadata: &MetadataMap,
) -> Result<BTreeMap<String, String>, MetadataHeaderError> {
    let prefix = metadata_header_prefix();
    let mut out = BTreeMap::new();
    let mut total_bytes = 0;

    for entry in metadata.iter() {
        let KeyAndValueRef::Ascii(name, value) = entry else {
            continue;
        };

        let Ok(value) = value.to_str() else {
            continue;
        };

        let Some(raw_key) = name.as_str().strip_prefix(prefix) else {
            continue;
        };

        insert_metadata_entry(&mut out, &mut total_bytes, raw_key, value)?;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderName;
    use tonic::metadata::{MetadataKey, MetadataValue};

    fn header_name(name: String) -> HeaderName {
        name.parse::<HeaderName>().unwrap()
    }

    #[test]
    fn test_extract_metadata_strips_prefix() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header_name(format!("{}tenant", DYNAMO_METADATA_HEADER_PREFIX_DEFAULT)),
            " acme ".parse().unwrap(),
        );
        headers.insert(
            header_name(format!("{}user-id", DYNAMO_METADATA_HEADER_PREFIX_DEFAULT)),
            "u42".parse().unwrap(),
        );
        headers.insert(
            header_name(format!(
                "{}authorization",
                DYNAMO_METADATA_HEADER_PREFIX_DEFAULT
            )),
            "Bearer secret".parse().unwrap(),
        );
        headers.insert(
            header_name(format!("{}token", DYNAMO_METADATA_HEADER_PREFIX_DEFAULT)),
            "Bearer secret".parse().unwrap(),
        );
        headers.insert("x-request-id", "irrelevant".parse().unwrap());

        let meta = extract_metadata_from_http(&headers).unwrap();
        assert_eq!(meta.get("tenant").map(String::as_str), Some("acme"));
        assert_eq!(meta.get("user-id").map(String::as_str), Some("u42"));
        assert!(!meta.contains_key("x-request-id"));
        assert!(!meta.contains_key("authorization"));
        assert!(!meta.contains_key("token"));
    }

    #[test]
    fn test_extract_metadata_applies_entry_and_total_size_limits() {
        let mut headers = HeaderMap::new();
        let near_budget = "a".repeat(DYNAMO_METADATA_MAX_TOTAL_BYTES_DEFAULT - 1);
        headers.insert(
            header_name(format!("{}a", DYNAMO_METADATA_HEADER_PREFIX_DEFAULT)),
            "ok".parse().unwrap(),
        );
        headers.insert(
            header_name(format!("{}b", DYNAMO_METADATA_HEADER_PREFIX_DEFAULT)),
            near_budget.parse().unwrap(),
        );

        let err = extract_metadata_from_http(&headers).unwrap_err();

        assert_eq!(
            err,
            MetadataHeaderError::TooLarge {
                limit_bytes: DYNAMO_METADATA_MAX_TOTAL_BYTES_DEFAULT
            }
        );
    }

    #[test]
    fn test_extract_metadata_from_grpc_skips_binary_and_applies_same_policy() {
        let mut metadata = MetadataMap::new();
        metadata.insert(
            MetadataKey::from_bytes(b"x-dynamo-meta-tenant").unwrap(),
            MetadataValue::try_from(" acme ").unwrap(),
        );
        metadata.append(
            MetadataKey::from_bytes(b"x-dynamo-meta-tenant").unwrap(),
            MetadataValue::try_from("other").unwrap(),
        );
        metadata.insert_bin(
            MetadataKey::from_bytes(b"x-dynamo-meta-secret-bin").unwrap(),
            MetadataValue::from_bytes(b"opaque"),
        );

        let meta = extract_metadata_from_grpc(&metadata).unwrap();
        assert_eq!(meta.get("tenant").map(String::as_str), Some("acme"));
        assert_eq!(meta.len(), 1);
    }
}
