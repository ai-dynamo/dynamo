// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared `--kvbm` / `--kvbm-config` override + validation helpers.
//!
//! These operate on a `serde_json::Value` shaped like a connector
//! `kv_connector_extra_config` (the `default` / `leader` / `worker` figment
//! profiles, or flat keys), and round-trip the result through [`KvbmConfig`] so
//! a bad override is rejected up front. Both `kvbmctl`'s render path and the
//! `kvbm_hub` server reuse them, so the two CLIs accept identical syntax and
//! enforce identical validation. Pure (no velo / CUDA), like the rest of this
//! crate.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value};

use crate::KvbmConfig;

/// Apply `--kvbm-config` (deep merge) then the individual `--kvbm` dotted
/// overrides (highest precedence) onto `extra`.
///
/// - `kvbm_config` is an optional full JSON object deep-merged first (lowest
///   precedence among the two).
/// - each entry in `overrides` is `key.path=value`; the value is parsed as JSON
///   (`2` → int, `true` → bool, `{…}` → object) with a string fallback for bare
///   words, then set at the dotted path (creating intermediate objects).
pub fn apply_overrides(
    extra: &mut Value,
    kvbm_config: Option<&str>,
    overrides: &[String],
) -> Result<()> {
    if let Some(blob) = kvbm_config {
        let parsed: Value = serde_json::from_str(blob).context("parsing --kvbm-config JSON")?;
        if !parsed.is_object() {
            bail!("--kvbm-config must be a JSON object");
        }
        deep_merge(extra, parsed);
    }
    for raw in overrides {
        let (path, value) = raw
            .split_once('=')
            .ok_or_else(|| anyhow!("--kvbm override {raw:?} must be of the form key.path=value"))?;
        if path.is_empty() {
            bail!("--kvbm override {raw:?} has an empty key path");
        }
        set_dotted(extra, path, parse_override_value(value));
    }
    Ok(())
}

/// Parse an override RHS as JSON (so `=2` → int, `=true` → bool, `={...}` →
/// object), falling back to a string for bare words like `universal`.
pub fn parse_override_value(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.to_string()))
}

/// Recursively merge `src` into `dst`. Object values merge key-by-key; any
/// other value (or a type mismatch) overwrites.
pub fn deep_merge(dst: &mut Value, src: Value) {
    match (dst, src) {
        (Value::Object(d), Value::Object(s)) => {
            for (k, v) in s {
                deep_merge(d.entry(k).or_insert(Value::Null), v);
            }
        }
        (d, s) => *d = s,
    }
}

/// Set a dotted path (`a.b.c`) in `root`, creating intermediate objects.
pub fn set_dotted(root: &mut Value, path: &str, value: Value) {
    let mut cur = root;
    let mut parts = path.split('.').peekable();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            if let Value::Object(map) = cur {
                map.insert(part.to_string(), value);
            }
            return;
        }
        if !cur.is_object() {
            *cur = Value::Object(Map::new());
        }
        let map = cur.as_object_mut().expect("ensured object above");
        cur = map
            .entry(part.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
    }
}

/// Validate a `kv_connector_extra_config` blob by parsing it with the same
/// loaders the connector uses for both roles. Errors carry the figment message
/// so a bad `--kvbm` override surfaces precisely. This is the shared validation
/// gate; it never builds a runtime, so it stays CUDA-free.
pub fn validate_extra_config(extra: &Value) -> Result<()> {
    let s = extra.to_string();
    KvbmConfig::from_figment_with_json_for_leader(&s)
        .map_err(|e| anyhow!("config rejected by the leader parser: {e}"))?;
    KvbmConfig::from_figment_with_json_for_worker(&s)
        .map_err(|e| anyhow!("config rejected by the worker parser: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dotted_override_is_typed_and_nested() {
        let mut v = json!({});
        apply_overrides(&mut v, None, &["leader.tokio.worker_threads=4".to_string()]).unwrap();
        // Parsed as an integer, not the string "4".
        assert_eq!(v["leader"]["tokio"]["worker_threads"], json!(4));
    }

    #[test]
    fn bare_word_falls_back_to_string() {
        let mut v = json!({});
        apply_overrides(
            &mut v,
            None,
            &["default.block_layout=universal".to_string()],
        )
        .unwrap();
        assert_eq!(v["default"]["block_layout"], json!("universal"));
    }

    #[test]
    fn config_blob_merges_then_dotted_override_wins() {
        let mut v = json!({});
        apply_overrides(
            &mut v,
            Some(r#"{"leader":{"tokio":{"worker_threads":8}}}"#),
            &["leader.tokio.worker_threads=3".to_string()],
        )
        .unwrap();
        assert_eq!(v["leader"]["tokio"]["worker_threads"], json!(3));
    }

    #[test]
    fn override_without_equals_is_rejected() {
        let mut v = json!({});
        assert!(apply_overrides(&mut v, None, &["leader.tokio".to_string()]).is_err());
    }

    #[test]
    fn non_object_config_blob_is_rejected() {
        let mut v = json!({});
        assert!(apply_overrides(&mut v, Some("[1,2,3]"), &[]).is_err());
    }

    #[test]
    fn validate_accepts_good_and_rejects_bad_typing() {
        let mut good = json!({});
        apply_overrides(
            &mut good,
            None,
            &["leader.tokio.worker_threads=2".to_string()],
        )
        .unwrap();
        assert!(validate_extra_config(&good).is_ok());

        let mut bad = json!({});
        apply_overrides(
            &mut bad,
            None,
            &["leader.tokio.worker_threads=notanint".to_string()],
        )
        .unwrap();
        assert!(validate_extra_config(&bad).is_err());
    }
}
