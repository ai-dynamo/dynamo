// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The shadow tap config file, named by `DYN_SHADOW_TAP_CONFIG`.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use super::filter::FilterSet;

pub const CONFIG_SCHEMA_VERSION: u32 = 1;
/// Queue bound of a tap that does not set `capacity`. A tap is never
/// unbounded: an unbounded queue lets one stalled shadow grow frontend memory
/// without limit.
pub const DEFAULT_CAPACITY: usize = 1024;
pub const DEFAULT_NAMESPACE: &str = "dynamo";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    schema_version: u32,
    #[serde(default)]
    namespace: Option<String>,
    taps: Vec<TapFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TapFile {
    name: String,
    #[serde(default)]
    topic: Option<String>,
    capture: Capture,
    #[serde(default)]
    emit: Option<Emit>,
    #[serde(default)]
    filters: Vec<String>,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    response: Option<ResponseOptions>,
    #[serde(default)]
    capacity: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capture {
    /// Publish the request as soon as it passes the tap.
    Request,
    /// Publish the request and the response the primary produced.
    RequestResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Emit {
    /// One record, published when the response stream ends.
    Joined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseOptions {
    /// Record output token ids.
    ///
    /// The tap holds the recorded tokens of every in-flight request until its
    /// stream ends. `capacity` does not bound this. Each in-flight request
    /// holds 4 bytes per recorded token, for each tap: up to `max_tokens` for
    /// each of its `n` choices, or its whole response when `max_tokens` is
    /// unset. `chunk_timing` adds 8 bytes for each chunk that carried tokens,
    /// and `max_tokens` does not cap that.
    #[serde(default = "default_true")]
    pub tokens: bool,
    /// Record at most this many token ids for each choice. The record marks
    /// the choice `truncated`, and `output_tokens` still counts every token.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Record the arrival offset of every response chunk that carried tokens,
    /// not only the first and the last. `end_offset_ns` gives the end of the
    /// stream, which covers a final chunk that carries only a finish reason.
    #[serde(default)]
    pub chunk_timing: bool,
}

impl Default for ResponseOptions {
    fn default() -> Self {
        Self {
            tokens: true,
            max_tokens: None,
            chunk_timing: false,
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone)]
pub struct TapSpec {
    pub name: Arc<str>,
    pub topic: String,
    pub capture: Capture,
    pub filters: FilterSet,
    pub response: ResponseOptions,
    pub capacity: usize,
}

#[derive(Debug, Clone)]
pub struct ShadowConfig {
    pub namespace: String,
    pub taps: Vec<TapSpec>,
}

impl ShadowConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading shadow tap config {}", path.display()))?;
        Self::from_yaml(&text)
            .with_context(|| format!("invalid shadow tap config {}", path.display()))
    }

    pub fn from_yaml(text: &str) -> Result<Self> {
        let file: ConfigFile = serde_yaml::from_str(text)?;
        ensure!(
            file.schema_version == CONFIG_SCHEMA_VERSION,
            "schema_version {} is not supported; expected {CONFIG_SCHEMA_VERSION}",
            file.schema_version
        );

        let mut names = HashSet::new();
        let mut topics = HashSet::new();
        let mut taps = Vec::with_capacity(file.taps.len());
        for tap in file.taps {
            let spec = TapSpec::validate(tap)?;
            ensure!(
                names.insert(spec.name.clone()),
                "duplicate tap name `{}`",
                spec.name
            );
            ensure!(
                topics.insert(spec.topic.clone()),
                "tap `{}`: topic `{}` is already used by another tap",
                spec.name,
                spec.topic
            );
            taps.push(spec);
        }

        Ok(Self {
            namespace: file
                .namespace
                .unwrap_or_else(|| DEFAULT_NAMESPACE.to_string()),
            taps,
        })
    }
}

impl TapSpec {
    fn validate(tap: TapFile) -> Result<Self> {
        let name = tap.name;
        ensure!(is_token(&name), "tap name `{name}` must be [A-Za-z0-9_-]+");
        let topic = tap
            .topic
            .unwrap_or_else(|| format!("shadow-requests-{name}"));
        ensure!(
            is_token(&topic),
            "tap `{name}`: topic `{topic}` must be [A-Za-z0-9_-]+"
        );

        let capacity = tap.capacity.unwrap_or(DEFAULT_CAPACITY);
        ensure!(capacity > 0, "tap `{name}`: capacity must be at least 1");
        if let Some(ResponseOptions {
            max_tokens: Some(0),
            ..
        }) = tap.response
        {
            bail!(
                "tap `{name}`: response.max_tokens must be at least 1; use `tokens: false` to record none"
            );
        }

        if tap.capture == Capture::Request {
            if tap.emit.is_some() {
                bail!("tap `{name}`: `emit` applies only to `capture: request_response`");
            }
            if tap.response.is_some() {
                bail!("tap `{name}`: `response` applies only to `capture: request_response`");
            }
        }

        let filters = FilterSet::resolve(&tap.filters, &tap.models)
            .with_context(|| format!("tap `{name}`"))?;

        Ok(Self {
            name: name.into(),
            topic,
            capture: tap.capture,
            filters,
            response: tap.response.unwrap_or_default(),
            capacity,
        })
    }
}

fn is_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
    use super::super::filter::Projection;
    use super::*;

    const EXAMPLE: &str = r#"
schema_version: 1
taps:
  - name: kv-history-tracker
    topic: kv-history-tracker
    capture: request
    filters: [tokens-only]
    capacity: 4096
  - name: shadow-replay
    topic: shadow-replay
    capture: request_response
    emit: joined
    filters: [strip-multimodal, strip-embeds]
    response:
      tokens: true
      chunk_timing: false
"#;

    #[test]
    fn example_config_parses() {
        let config = ShadowConfig::from_yaml(EXAMPLE).unwrap();
        assert_eq!(config.namespace, DEFAULT_NAMESPACE);
        assert_eq!(config.taps.len(), 2);

        let tracker = &config.taps[0];
        assert_eq!(tracker.capture, Capture::Request);
        assert_eq!(tracker.capacity, 4096);
        assert!(tracker.filters.projection.contains(Projection::TOKENS_ONLY));

        let replay = &config.taps[1];
        assert_eq!(replay.capture, Capture::RequestResponse);
        assert_eq!(
            replay.capacity, DEFAULT_CAPACITY,
            "omitted capacity is bounded"
        );
        assert!(replay.response.tokens);
    }

    fn one_tap(body: &str) -> Result<ShadowConfig> {
        ShadowConfig::from_yaml(&format!(
            "schema_version: 1\ntaps:\n  - name: t\n    capture: request\n{body}"
        ))
    }

    #[test]
    fn topic_defaults_from_name() {
        assert_eq!(one_tap("").unwrap().taps[0].topic, "shadow-requests-t");
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(one_tap("    capacity: 0\n").is_err());
        assert!(one_tap("    filters: [nope]\n").is_err());
        assert!(one_tap("    emit: joined\n").is_err());
        assert!(one_tap("    response: {tokens: true}\n").is_err());
        assert!(
            ShadowConfig::from_yaml(
                "schema_version: 1\ntaps:\n  - {name: t, capture: request_response, response: {max_tokens: 0}}\n"
            )
            .is_err()
        );
        assert!(one_tap("    unknown_key: 1\n").is_err());
        assert!(one_tap("    topic: 'a.b'\n").is_err());
        assert!(ShadowConfig::from_yaml("schema_version: 2\ntaps: []\n").is_err());
        let duplicate = "schema_version: 1\ntaps:\n  - {name: a, capture: request}\n  - {name: a, capture: request, topic: other}\n";
        assert!(ShadowConfig::from_yaml(duplicate).is_err());
        let shared_topic = "schema_version: 1\ntaps:\n  - {name: a, capture: request, topic: t}\n  - {name: b, capture: request, topic: t}\n";
        assert!(ShadowConfig::from_yaml(shared_topic).is_err());
    }
}
