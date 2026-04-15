// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const DEFAULT_TOKENIZER: &str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B";
pub const DEFAULT_BLOCK_SIZE: usize = 64;
pub const DEFAULT_OUTPUT_NAME: &str = "claude_mooncake_trace.jsonl";
pub const SIDE_CAR_TOKEN: &str = ".sidecar";

pub fn parse_utc_timestamp_ms(value: &str) -> Result<i64> {
    if value.is_empty() {
        bail!("missing timestamp");
    }
    let parsed = DateTime::parse_from_rfc3339(value)
        .with_context(|| format!("invalid RFC3339 timestamp: {value}"))?;
    Ok(parsed.with_timezone(&Utc).timestamp_millis())
}

pub fn anonymized_session_id(session_id: &str) -> String {
    let digest = Sha256::digest(session_id.as_bytes());
    let mut hex = String::with_capacity(12);
    for byte in digest.iter().take(6) {
        hex.push_str(&format!("{byte:02x}"));
    }
    format!("session_{hex}")
}

pub fn sidecar_path_for(output_path: &Path) -> PathBuf {
    match (output_path.file_stem(), output_path.extension()) {
        (Some(stem), Some(ext)) => output_path.with_file_name(format!(
            "{}{}{ext_sep}{}",
            stem.to_string_lossy(),
            SIDE_CAR_TOKEN,
            ext.to_string_lossy(),
            ext_sep = "."
        )),
        _ => output_path.with_file_name(format!(
            "{}{}.jsonl",
            output_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            SIDE_CAR_TOKEN
        )),
    }
}

pub fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for path in paths {
        let resolved = path.canonicalize().unwrap_or(path);
        if seen.insert(resolved.clone()) {
            deduped.push(resolved);
        }
    }
    deduped
}

pub fn expand_user_path(raw: &str) -> PathBuf {
    if raw == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from(raw));
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return home_dir()
            .map(|home| home.join(rest))
            .unwrap_or_else(|| PathBuf::from(raw));
    }
    PathBuf::from(raw)
}

pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub fn canonical_json_string(value: &Value) -> Result<String> {
    let mut rendered = String::new();
    write_canonical_json(&mut rendered, value)?;
    Ok(rendered)
}

fn write_canonical_json(buffer: &mut String, value: &Value) -> Result<()> {
    match value {
        Value::Null => buffer.push_str("null"),
        Value::Bool(flag) => {
            if *flag {
                buffer.push_str("true");
            } else {
                buffer.push_str("false");
            }
        }
        Value::Number(number) => buffer.push_str(&number.to_string()),
        Value::String(text) => buffer.push_str(&serde_json::to_string(text)?),
        Value::Array(items) => {
            buffer.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    buffer.push(',');
                }
                write_canonical_json(buffer, item)?;
            }
            buffer.push(']');
        }
        Value::Object(map) => {
            buffer.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    buffer.push(',');
                }
                buffer.push_str(&serde_json::to_string(key)?);
                buffer.push(':');
                write_canonical_json(buffer, &map[key])?;
            }
            buffer.push('}');
        }
    }
    Ok(())
}

pub fn content_blocks(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::String(text)) => {
            vec![Value::Object(
                [
                    ("type".to_string(), Value::String("text".to_string())),
                    ("text".to_string(), Value::String(text.clone())),
                ]
                .into_iter()
                .collect(),
            )]
        }
        Some(Value::Array(items)) => items
            .iter()
            .filter(|item| item.is_object())
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

pub fn flatten_block_content_text(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                match item {
                    Value::String(text) => parts.push(text.clone()),
                    Value::Object(map) => {
                        if let Some(text) = map.get("text").and_then(Value::as_str) {
                            parts.push(text.to_string());
                        } else {
                            parts.push(canonical_json_string(item)?);
                        }
                    }
                    _ => parts.push(canonical_json_string(item)?),
                }
            }
            Ok(parts
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n"))
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                return Ok(text.to_string());
            }
            canonical_json_string(value)
        }
        _ => canonical_json_string(value),
    }
}

pub fn object_field<'a>(value: &'a Value, field: &str) -> Option<&'a Map<String, Value>> {
    value.get(field)?.as_object()
}
