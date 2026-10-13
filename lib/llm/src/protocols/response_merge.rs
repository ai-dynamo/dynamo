// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Joining the pieces of one engine response.
//!
//! One client response can reach the frontend in pieces, split at two seams:
//!
//! - [`Seam::Steps`]: the engine streams the response. Some fields in each
//!   chunk hold only what is new since the last chunk.
//! - [`Seam::PrefillDecode`]: in disaggregated serving, only the prefill worker
//!   runs the prompt, so prompt-side outputs exist only in its response.
//!
//! A [`ResponseMerger`] owns one engine's rules for both seams. [`RuleSet`] is
//! the declarative implementation and the intended plugin format: a list of
//! [`FieldRule`]s that config can supply. [`ResponseAssembler`] drives a
//! merger for one request.

use std::{fmt, sync::Arc};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Request annotation asking the prefill router to send prefill's final
/// `engine_data` ahead of the decode stream, as an annotation event of the
/// same name.
pub(crate) const PREFILL_OUTPUT_ANNOTATION: &str = "prefill_output";

/// Where a response is split into pieces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Seam {
    /// `earlier` is the response joined so far; `later` is the next chunk.
    Steps,
    /// `earlier` is prefill's final response; `later` is a decode response or
    /// decode chunk.
    PrefillDecode,
}

/// One engine's rules for joining response pieces.
pub(crate) trait ResponseMerger: Send + Sync + fmt::Debug {
    /// Whether the request asks for output that only prefill produces.
    fn wants_prefill_output(&self, request: &Value) -> bool;

    /// Join `earlier` into `later` at `seam`. Afterwards `later` holds the
    /// joined value. Fields without a rule keep `later`'s value.
    fn merge(&self, seam: Seam, earlier: Value, later: &mut Value) -> Result<(), MergeError>;

    /// Check a fully joined response before it is returned.
    fn finish(&self, _response: &mut Value) -> Result<(), MergeError> {
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum MergeError {
    #[error("expected {expected} at {path}")]
    Shape {
        path: String,
        expected: &'static str,
    },
    #[error("invalid base64 at {path}: {source}")]
    Base64 {
        path: String,
        source: base64::DecodeError,
    },
    #[error("{0}")]
    Invalid(String),
}

/// How the two values of one field combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JoinOp {
    /// Earlier's items, then later's items. A missing side counts as empty.
    Append,
    /// Earlier's text, then later's text. A missing side counts as empty.
    AppendText,
    /// Earlier's value whenever earlier has one.
    TakeEarlier,
    /// Decode both base64 blobs. Keep earlier's bytes, then later's bytes past
    /// that length. Applies only when both sides have the field.
    OverlayBase64Prefix,
}

/// What to do with later's value when earlier has none to join.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OnMissing {
    /// Keep later's value. Right when a partial value is visibly partial.
    #[default]
    KeepLater,
    /// Remove the field. Right when later's value looks complete but is wrong.
    Remove,
}

/// One declarative join rule. `path` is a JSON pointer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FieldRule {
    pub seam: Seam,
    pub path: String,
    pub op: JoinOp,
    #[serde(default)]
    pub on_missing: OnMissing,
}

impl FieldRule {
    pub(crate) fn new(seam: Seam, path: &str, op: JoinOp, on_missing: OnMissing) -> Self {
        Self {
            seam,
            path: path.to_string(),
            op,
            on_missing,
        }
    }

    fn apply(&self, earlier: &mut Value, later: &mut Value) -> Result<(), MergeError> {
        let (parent, key) = split_pointer(&self.path)?;
        let head = earlier
            .pointer_mut(parent)
            .and_then(Value::as_object_mut)
            .and_then(|object| object.remove(&key))
            .filter(|value| !value.is_null());
        let Some(container) = later.pointer_mut(parent).and_then(Value::as_object_mut) else {
            return Ok(());
        };
        let Some(head) = head else {
            if self.on_missing == OnMissing::Remove {
                container.remove(&key);
            }
            return Ok(());
        };
        let shape = |expected| MergeError::Shape {
            path: self.path.clone(),
            expected,
        };
        let tail = container.get_mut(&key).filter(|value| !value.is_null());
        match self.op {
            JoinOp::Append => match (head, tail) {
                (Value::Array(mut head), Some(Value::Array(tail))) => {
                    head.append(tail);
                    *tail = head;
                }
                (head @ Value::Array(_), None) => {
                    container.insert(key, head);
                }
                _ => return Err(shape("arrays")),
            },
            JoinOp::AppendText => match (head, tail) {
                (Value::String(mut head), Some(Value::String(tail))) => {
                    head.push_str(tail);
                    *tail = head;
                }
                (head @ Value::String(_), None) => {
                    container.insert(key, head);
                }
                _ => return Err(shape("strings")),
            },
            JoinOp::TakeEarlier => {
                container.insert(key, head);
            }
            JoinOp::OverlayBase64Prefix => match (head, tail) {
                (_, None) => {}
                (Value::String(head), Some(Value::String(tail))) => {
                    let base64 = |source| MergeError::Base64 {
                        path: self.path.clone(),
                        source,
                    };
                    let mut bytes = STANDARD.decode(&head).map_err(base64)?;
                    let rest = STANDARD.decode(tail.as_bytes()).map_err(base64)?;
                    bytes.extend_from_slice(rest.get(bytes.len()..).unwrap_or_default());
                    *tail = STANDARD.encode(bytes);
                }
                _ => return Err(shape("base64 strings")),
            },
        }
        Ok(())
    }

    fn discard(&self, later: &mut Value) {
        if self.on_missing == OnMissing::Remove
            && let Ok((parent, key)) = split_pointer(&self.path)
            && let Some(container) = later.pointer_mut(parent).and_then(Value::as_object_mut)
        {
            container.remove(&key);
        }
    }
}

/// Declarative [`ResponseMerger`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RuleSet {
    pub rules: Vec<FieldRule>,
    /// Top-level request fields that turn on prefill-only output. A field
    /// counts when it is `true`, or a list containing `true`.
    #[serde(default)]
    pub prefill_request_flags: Vec<String>,
}

impl ResponseMerger for RuleSet {
    fn wants_prefill_output(&self, request: &Value) -> bool {
        self.prefill_request_flags
            .iter()
            .any(|flag| match request.get(flag) {
                Some(Value::Bool(value)) => *value,
                Some(Value::Array(values)) => values.contains(&Value::Bool(true)),
                _ => false,
            })
    }

    fn merge(&self, seam: Seam, mut earlier: Value, later: &mut Value) -> Result<(), MergeError> {
        if !later.is_object() {
            return Err(MergeError::Shape {
                path: String::new(),
                expected: "a JSON object",
            });
        }
        for rule in self.rules.iter().filter(|rule| rule.seam == seam) {
            if let Err(error) = rule.apply(&mut earlier, later) {
                tracing::warn!(path = %rule.path, ?seam, %error, "cannot join response field");
                rule.discard(later);
            }
        }
        Ok(())
    }
}

/// Builds one client response from engine pieces using a [`ResponseMerger`].
pub(crate) struct ResponseAssembler {
    merger: Arc<dyn ResponseMerger>,
    prefill: Option<Value>,
    joined: Option<Value>,
}

impl ResponseAssembler {
    pub(crate) fn new(merger: Arc<dyn ResponseMerger>) -> Self {
        Self {
            merger,
            prefill: None,
            joined: None,
        }
    }

    /// Record prefill's final response.
    pub(crate) fn set_prefill(&mut self, prefill: Value) {
        self.prefill = Some(prefill);
    }

    /// Streaming: return `chunk` with prefill's outputs joined in.
    pub(crate) fn stream_chunk(&self, mut chunk: Value) -> Result<Value, MergeError> {
        if let Some(prefill) = &self.prefill {
            self.merger
                .merge(Seam::PrefillDecode, prefill.clone(), &mut chunk)?;
        }
        Ok(chunk)
    }

    /// Unary: join `chunk` onto the response so far.
    pub(crate) fn push(&mut self, mut chunk: Value) -> Result<(), MergeError> {
        if let Some(earlier) = self.joined.take() {
            self.merger.merge(Seam::Steps, earlier, &mut chunk)?;
        }
        self.joined = Some(chunk);
        Ok(())
    }

    /// Unary: the joined response with prefill's outputs, or `None` when no
    /// chunk arrived.
    pub(crate) fn finish(self) -> Result<Option<Value>, MergeError> {
        let Some(mut response) = self.joined else {
            return Ok(None);
        };
        if let Some(prefill) = self.prefill {
            self.merger
                .merge(Seam::PrefillDecode, prefill, &mut response)?;
        }
        self.merger.finish(&mut response)?;
        Ok(Some(response))
    }
}

fn split_pointer(path: &str) -> Result<(&str, String), MergeError> {
    let (parent, key) = path
        .rsplit_once('/')
        .ok_or_else(|| MergeError::Invalid(format!("`{path}` is not a JSON pointer")))?;
    Ok((parent, key.replace("~1", "/").replace("~0", "~")))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn rules(rules: Vec<FieldRule>) -> Arc<dyn ResponseMerger> {
        Arc::new(RuleSet {
            rules,
            prefill_request_flags: vec!["return_x".into()],
        })
    }

    fn b64(bytes: &[u8]) -> String {
        STANDARD.encode(bytes)
    }

    #[test]
    fn steps_append_deltas_and_keep_the_last_snapshot() {
        let merger = rules(vec![
            FieldRule::new(
                Seam::Steps,
                "/text",
                JoinOp::AppendText,
                OnMissing::KeepLater,
            ),
            FieldRule::new(Seam::Steps, "/ids", JoinOp::Append, OnMissing::KeepLater),
        ]);
        let mut assembler = ResponseAssembler::new(merger);
        for chunk in [
            json!({"text": "a", "ids": [1], "meta": {"count": 1}}),
            json!({"text": "b", "ids": [2, 3], "meta": {"count": 3}}),
            json!({"text": "", "ids": [], "meta": {"count": 3, "done": true}}),
        ] {
            assembler.push(chunk).unwrap();
        }
        assert_eq!(
            assembler.finish().unwrap().unwrap(),
            json!({"text": "ab", "ids": [1, 2, 3], "meta": {"count": 3, "done": true}})
        );
    }

    #[test]
    fn steps_keep_joined_items_when_a_chunk_omits_the_field() {
        let merger = rules(vec![FieldRule::new(
            Seam::Steps,
            "/meta/custom",
            JoinOp::Append,
            OnMissing::KeepLater,
        )]);
        let mut assembler = ResponseAssembler::new(merger);
        assembler.push(json!({"meta": {"custom": [1]}})).unwrap();
        assembler.push(json!({"meta": {}})).unwrap();
        assembler.push(json!({"meta": {"custom": [2]}})).unwrap();
        assert_eq!(
            assembler.finish().unwrap().unwrap(),
            json!({"meta": {"custom": [1, 2]}})
        );
    }

    #[test]
    fn prefill_outputs_join_every_streamed_chunk_and_the_unary_response() {
        let merger = rules(vec![
            FieldRule::new(Seam::Steps, "/ids", JoinOp::Append, OnMissing::KeepLater),
            FieldRule::new(
                Seam::PrefillDecode,
                "/meta/prompt",
                JoinOp::TakeEarlier,
                OnMissing::KeepLater,
            ),
        ]);
        let prefill = json!({"ids": [9], "meta": {"prompt": [1, 2]}});
        let mut streamed = ResponseAssembler::new(merger.clone());
        streamed.set_prefill(prefill.clone());
        assert_eq!(
            streamed
                .stream_chunk(json!({"ids": [7], "meta": {"prompt": []}}))
                .unwrap(),
            json!({"ids": [7], "meta": {"prompt": [1, 2]}})
        );

        let mut unary = ResponseAssembler::new(merger);
        unary.set_prefill(prefill);
        unary
            .push(json!({"ids": [7], "meta": {"prompt": []}}))
            .unwrap();
        unary
            .push(json!({"ids": [8], "meta": {"prompt": []}}))
            .unwrap();
        assert_eq!(
            unary.finish().unwrap().unwrap(),
            json!({"ids": [7, 8], "meta": {"prompt": [1, 2]}})
        );
    }

    #[test]
    fn overlay_keeps_earlier_prefix_and_missing_values_follow_policy() {
        let rule = FieldRule::new(
            Seam::PrefillDecode,
            "/meta/blob",
            JoinOp::OverlayBase64Prefix,
            OnMissing::Remove,
        );
        let merger = rules(vec![rule]);
        let mut later = json!({"meta": {"blob": b64(&[0, 0, 0, 4, 5])}});
        merger
            .merge(
                Seam::PrefillDecode,
                json!({"meta": {"blob": b64(&[1, 2, 3])}}),
                &mut later,
            )
            .unwrap();
        assert_eq!(later, json!({"meta": {"blob": b64(&[1, 2, 3, 4, 5])}}));

        for earlier in [json!({"meta": {}}), json!({"meta": {"blob": "%%"}})] {
            let mut later = json!({"meta": {"blob": b64(&[0, 4]), "other": 1}});
            merger
                .merge(Seam::PrefillDecode, earlier, &mut later)
                .unwrap();
            assert_eq!(later, json!({"meta": {"other": 1}}));
        }

        let mut later = json!({"meta": {}});
        merger
            .merge(
                Seam::PrefillDecode,
                json!({"meta": {"blob": b64(&[1])}}),
                &mut later,
            )
            .unwrap();
        assert_eq!(later, json!({"meta": {}}));
    }

    #[test]
    fn shape_errors_keep_later_unless_removal_is_requested() {
        let merger = rules(vec![FieldRule::new(
            Seam::Steps,
            "/ids",
            JoinOp::Append,
            OnMissing::KeepLater,
        )]);
        let mut later = json!({"ids": [2]});
        merger
            .merge(Seam::Steps, json!({"ids": "oops"}), &mut later)
            .unwrap();
        assert_eq!(later, json!({"ids": [2]}));
        assert!(
            merger
                .merge(Seam::Steps, json!({}), &mut json!("not an object"))
                .is_err()
        );
    }

    #[test]
    fn request_flags_select_prefill_output() {
        let merger = rules(Vec::new());
        assert!(merger.wants_prefill_output(&json!({"return_x": true})));
        assert!(merger.wants_prefill_output(&json!({"return_x": [false, true]})));
        assert!(!merger.wants_prefill_output(&json!({"return_x": false})));
        assert!(!merger.wants_prefill_output(&json!({"other": true})));
    }

    #[test]
    fn rule_sets_have_a_stable_declarative_format() {
        let parsed: RuleSet = serde_json::from_value(json!({
            "prefill_request_flags": ["return_routed_experts"],
            "rules": [
                {"seam": "steps", "path": "/meta_info/custom", "op": "append"},
                {
                    "seam": "prefill_decode",
                    "path": "/meta_info/routed_experts",
                    "op": "overlay_base64_prefix",
                    "on_missing": "remove"
                }
            ]
        }))
        .unwrap();
        assert_eq!(
            parsed.rules,
            [
                FieldRule::new(
                    Seam::Steps,
                    "/meta_info/custom",
                    JoinOp::Append,
                    OnMissing::KeepLater
                ),
                FieldRule::new(
                    Seam::PrefillDecode,
                    "/meta_info/routed_experts",
                    JoinOp::OverlayBase64Prefix,
                    OnMissing::Remove
                ),
            ]
        );
    }
}
