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
use std::fmt;
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataHeaderError {
    TooManyEntries { limit: usize },
    TooLarge { limit_bytes: usize },
}

impl fmt::Display for MetadataHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyEntries { limit } => {
                write!(f, "metadata headers exceed the limit of {limit} entries")
            }
            Self::TooLarge { limit_bytes } => {
                write!(
                    f,
                    "metadata headers exceed the limit of {limit_bytes} bytes"
                )
            }
        }
    }
}

pub(crate) fn metadata_header_prefix() -> &'static str {
    METADATA_HEADER_PREFIX.get_or_init(|| {
        std::env::var(DYNAMO_METADATA_HEADER_ENV)
            .unwrap_or_else(|_| DYNAMO_METADATA_HEADER_PREFIX_DEFAULT.to_string())
            .to_ascii_lowercase()
    })
}

fn insert_metadata_entry(
    out: &mut BTreeMap<String, String>,
    total_bytes: &mut usize,
    raw_key: &str,
    raw_value: &str,
    max_entries: usize,
    max_total_bytes: usize,
) -> Result<(), MetadataHeaderError> {
    if out.contains_key(raw_key) {
        return Ok(());
    }

    if out.len() >= max_entries {
        return Err(MetadataHeaderError::TooManyEntries { limit: max_entries });
    }

    let value = raw_value.trim();
    let entry_bytes = raw_key.len() + value.len();
    if *total_bytes + entry_bytes > max_total_bytes {
        return Err(MetadataHeaderError::TooLarge {
            limit_bytes: max_total_bytes,
        });
    }

    *total_bytes += entry_bytes;
    out.insert(raw_key.to_string(), value.to_string());
    Ok(())
}

fn extract_metadata_from_pairs<'a>(
    pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
    prefix: &str,
    max_entries: usize,
    max_total_bytes: usize,
) -> Result<BTreeMap<String, String>, MetadataHeaderError> {
    let mut out = BTreeMap::new();
    let mut total_bytes = 0;

    for (name, value) in pairs {
        let Some(raw_key) = name.strip_prefix(prefix) else {
            continue;
        };
        insert_metadata_entry(
            &mut out,
            &mut total_bytes,
            raw_key,
            value,
            max_entries,
            max_total_bytes,
        )?;
    }

    Ok(out)
}

/// Extract all `<prefix><key>: <value>` headers as a metadata map.
///
/// Headers that are not valid UTF-8 are silently skipped.
/// If a header is repeated, the first value wins.
/// Requests exceeding 64 entries or 64 KiB of key/value payload are rejected.
pub fn extract_metadata_from_headers(
    headers: &HeaderMap,
) -> Result<BTreeMap<String, String>, MetadataHeaderError> {
    let prefix = metadata_header_prefix();
    if !headers.keys().any(|name| name.as_str().starts_with(prefix)) {
        return Ok(BTreeMap::new());
    }

    extract_metadata_from_pairs(
        headers
            .iter()
            .filter_map(|(name, value)| value.to_str().ok().map(|value| (name.as_str(), value))),
        prefix,
        DYNAMO_METADATA_MAX_ENTRIES_DEFAULT,
        DYNAMO_METADATA_MAX_TOTAL_BYTES_DEFAULT,
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

        insert_metadata_entry(
            &mut out,
            &mut total_bytes,
            raw_key,
            value,
            DYNAMO_METADATA_MAX_ENTRIES_DEFAULT,
            DYNAMO_METADATA_MAX_TOTAL_BYTES_DEFAULT,
        )?;
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
        headers.insert("x-request-id", "irrelevant".parse().unwrap());

        let meta = extract_metadata_from_headers(&headers).unwrap();
        assert_eq!(meta.get("tenant").map(String::as_str), Some("acme"));
        assert_eq!(meta.get("user-id").map(String::as_str), Some("u42"));
        assert!(!meta.contains_key("x-request-id"));
    }

    #[test]
    fn test_extract_metadata_empty_suffix_is_preserved() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header_name(DYNAMO_METADATA_HEADER_PREFIX_DEFAULT.to_string()),
            "value".parse().unwrap(),
        );

        let meta = extract_metadata_from_headers(&headers).unwrap();
        assert_eq!(meta.get("").map(String::as_str), Some("value"));
    }

    #[test]
    fn test_extract_metadata_duplicate_header_first_wins() {
        let mut headers = HeaderMap::new();
        let key = header_name(format!("{}tenant", DYNAMO_METADATA_HEADER_PREFIX_DEFAULT));
        headers.insert(key.clone(), "first".parse().unwrap());
        headers.append(key, "second".parse().unwrap());

        let meta = extract_metadata_from_headers(&headers).unwrap();
        assert_eq!(meta.get("tenant").map(String::as_str), Some("first"));
    }

    #[test]
    fn test_extract_metadata_non_utf8_is_ignored() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header_name(format!("{}tenant", DYNAMO_METADATA_HEADER_PREFIX_DEFAULT)),
            axum::http::HeaderValue::from_bytes(b"\xFF").unwrap(),
        );

        let meta = extract_metadata_from_headers(&headers).unwrap();
        assert!(meta.is_empty());
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

        let err = extract_metadata_from_headers(&headers).unwrap_err();

        assert_eq!(
            err,
            MetadataHeaderError::TooLarge {
                limit_bytes: DYNAMO_METADATA_MAX_TOTAL_BYTES_DEFAULT
            }
        );
    }

    #[test]
    fn test_extract_metadata_rejects_too_many_entries() {
        let mut headers = HeaderMap::new();
        for idx in 0..DYNAMO_METADATA_MAX_ENTRIES_DEFAULT {
            headers.insert(
                header_name(format!("{}k{idx}", DYNAMO_METADATA_HEADER_PREFIX_DEFAULT)),
                "v".parse().unwrap(),
            );
        }
        headers.insert(
            header_name(format!("{}overflow", DYNAMO_METADATA_HEADER_PREFIX_DEFAULT)),
            "v".parse().unwrap(),
        );

        let err = extract_metadata_from_headers(&headers).unwrap_err();
        assert_eq!(
            err,
            MetadataHeaderError::TooManyEntries {
                limit: DYNAMO_METADATA_MAX_ENTRIES_DEFAULT
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
