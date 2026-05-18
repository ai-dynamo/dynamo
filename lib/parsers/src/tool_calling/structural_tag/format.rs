// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed representations of xgrammar structural tag formats.
//!
//! This module models the subset of the
//! [xgrammar structural tag API](https://xgrammar.mlc.ai/docs/structural_tag/structural_tag_api.html)
//! used by Dynamo for tool-call guided decoding. The types serialize to the JSON
//! shape expected by xgrammar backends.

use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

/// Top-level structural tag wrapper.
///
/// Serializes to `{"type": "structural_tag", "format": ...}`.
#[derive(Debug, Clone)]
pub struct StructuralTag {
    pub format: Format,
}

impl Serialize for StructuralTag {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("type", "structural_tag")?;
        map.serialize_entry("format", &self.format)?;
        map.end()
    }
}

/// A format node inside a structural tag.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Format {
    /// `{"type": "tag", "begin": ..., "content": ..., "end": ...}`
    Tag(TagFormat),
    /// `{"type": "triggered_tags", "triggers": [...], "tags": [...], ...}`
    TriggeredTags(TriggeredTagsFormat),
    /// `{"type": "tags_with_separator", "tags": [...], "separator": ..., ...}`
    TagsWithSeparator(TagsWithSeparatorFormat),
    /// `{"type": "json_schema", "json_schema": ..., "style": ...}`
    JsonSchema(JsonSchemaFormat),
    /// `{"type": "any_tokens", "exclude_tokens": [...]}`
    AnyTokens(AnyTokensFormat),
    /// `{"type": "any_text", "excludes": [...]}`
    AnyText(AnyTextFormat),
}

/// `tag` format: `begin + content + end`.
///
/// Always serializes with `"type": "tag"` so it is valid both as a
/// standalone format node and inside `TriggeredTagsFormat.tags`.
#[derive(Debug, Clone)]
pub struct TagFormat {
    pub begin: String,
    pub content: Box<Format>,
    pub end: String,
}

impl Serialize for TagFormat {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("type", "tag")?;
        map.serialize_entry("begin", &self.begin)?;
        map.serialize_entry("content", &self.content)?;
        map.serialize_entry("end", &self.end)?;
        map.end()
    }
}

/// `triggered_tags`: free text until a trigger, then one of the configured tags.
#[derive(Debug, Clone, Serialize)]
pub struct TriggeredTagsFormat {
    pub triggers: Vec<String>,
    pub tags: Vec<TagFormat>,
    pub at_least_one: bool,
    pub stop_after_first: bool,
}

/// `tags_with_separator`: a tag sequence with fixed separators and no free text.
#[derive(Debug, Clone, Serialize)]
pub struct TagsWithSeparatorFormat {
    pub tags: Vec<TagFormat>,
    pub separator: String,
    pub at_least_one: bool,
    pub stop_after_first: bool,
}

/// Content style for `json_schema` format nodes.
///
/// Controls how xgrammar renders tool arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JsonSchemaStyle {
    /// Standard JSON output.
    #[default]
    Json,
    /// Qwen-style XML: `<parameter=name>value</parameter>`.
    QwenXml,
    /// MiniMax-style XML: `<parameter name="name">value</parameter>`.
    MinimaxXml,
    /// DeepSeek DSML XML parameter format.
    DeepseekXml,
}

/// `json_schema` content format.
#[derive(Debug, Clone, Serialize)]
pub struct JsonSchemaFormat {
    /// Full schema object, or `true` for unconstrained JSON.
    pub json_schema: Value,

    /// Output style for generated arguments.
    pub style: JsonSchemaStyle,
}

/// `any_tokens` format: match zero or more tokens, excluding specific ones.
#[derive(Debug, Clone, Serialize)]
pub struct AnyTokensFormat {
    pub exclude_tokens: Vec<String>,
}

/// `any_text` format: match zero or more tokens, excluding arbitrary text.
#[derive(Debug, Clone, Serialize)]
pub struct AnyTextFormat {
    pub excludes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn structural_tag_serializes_correctly() {
        let tag = StructuralTag {
            format: Format::Tag(TagFormat {
                begin: "<think>".to_string(),
                content: Box::new(Format::AnyTokens(AnyTokensFormat {
                    exclude_tokens: vec!["<eos>".to_string()],
                })),
                end: "</think>".to_string(),
            }),
        };

        let value = serde_json::to_value(&tag).unwrap();
        assert_eq!(value["type"], "structural_tag");
        assert_eq!(value["format"]["type"], "tag");
        assert_eq!(value["format"]["begin"], "<think>");
        assert_eq!(value["format"]["content"]["type"], "any_tokens");
    }

    #[test]
    fn any_text_serializes_correctly() {
        let format = Format::AnyText(AnyTextFormat {
            excludes: vec!["<tool_call>\n<function=".to_string()],
        });

        let value = serde_json::to_value(&format).unwrap();
        assert_eq!(value["type"], "any_text");
        assert_eq!(value["excludes"][0], "<tool_call>\n<function=");
    }

    #[test]
    fn triggered_tags_serializes_at_least_one_when_false() {
        let format = TriggeredTagsFormat {
            triggers: vec!["<fn=".to_string()],
            tags: vec![],
            at_least_one: false,
            stop_after_first: false,
        };

        let value = serde_json::to_value(&format).unwrap();
        assert_eq!(value["at_least_one"], false);
    }

    #[test]
    fn triggered_tags_includes_at_least_one_when_true() {
        let format = TriggeredTagsFormat {
            triggers: vec!["<fn=".to_string()],
            tags: vec![],
            at_least_one: true,
            stop_after_first: false,
        };

        let value = serde_json::to_value(&format).unwrap();
        assert_eq!(value["at_least_one"], true);
    }

    #[test]
    fn json_schema_serializes_default_style() {
        let format = JsonSchemaFormat {
            json_schema: json!(true),
            style: JsonSchemaStyle::Json,
        };

        let value = serde_json::to_value(&format).unwrap();
        assert_eq!(value["style"], "json");
    }

    #[test]
    fn json_schema_serializes_non_default_style() {
        let format = JsonSchemaFormat {
            json_schema: json!(true),
            style: JsonSchemaStyle::QwenXml,
        };

        let value = serde_json::to_value(&format).unwrap();
        assert_eq!(value["style"], "qwen_xml");
    }
}
