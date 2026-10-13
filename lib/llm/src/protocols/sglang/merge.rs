// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang `/generate` join rules.
//!
//! Dynamo runs SGLang with incremental streaming output, so each chunk holds
//! only the new part of `text`, `output_ids`, and the output-side lists in
//! `_INCREMENTAL_STREAMING_META_INFO_KEYS`. Everything else in a chunk is the
//! current full value. Joining the chunks therefore follows SGLang's own
//! `_coalesce_streaming_chunks`: append those fields, take the rest from the
//! last chunk.
//!
//! In P/D, decode never runs the prompt. Prompt-side outputs come from
//! prefill: input logprobs are taken from prefill, and the prompt rows of
//! `routed_experts` are overlaid from prefill's blob (sglang-miles PR #22916).
//! When prefill has no blob, decode's prompt rows are stale, so the field is
//! removed instead of returned wrong.

use std::sync::{Arc, LazyLock};

use serde_json::Value;

use crate::protocols::response_merge::{
    FieldRule, JoinOp, MergeError, OnMissing, ResponseMerger, RuleSet, Seam,
};

/// Output-side lists that incremental streaming sends as deltas.
const INCREMENTAL_META_INFO_KEYS: [&str; 5] = [
    "output_token_logprobs",
    "output_top_logprobs",
    "output_token_ids_logprobs",
    "output_token_sampling_mask",
    "output_token_sampling_logprobs",
];

/// Prompt-side outputs that decode leaves empty and prefill computes.
const PREFILL_META_INFO_KEYS: [&str; 10] = [
    "input_top_logprobs",
    "input_token_ids_logprobs",
    "input_top_logprobs_val_flat",
    "input_top_logprobs_idx_flat",
    "input_top_logprobs_val_flat_b64",
    "input_top_logprobs_idx_flat_b64",
    "input_top_logprobs_val_flat_b64_dtype",
    "input_top_logprobs_idx_flat_b64_dtype",
    "input_top_logprobs_shape",
    "input_top_logprobs_null_prefix",
];

/// SGLang join rules plus the checks that need SGLang semantics.
#[derive(Debug)]
pub(crate) struct SglangMerger {
    rules: RuleSet,
}

static DEFAULT: LazyLock<Arc<SglangMerger>> =
    LazyLock::new(|| Arc::new(SglangMerger::new(Vec::new())));

impl SglangMerger {
    /// Built-in rules plus `extra` rules, for example `customized_info` keys,
    /// which SGLang appends like the output logprobs.
    pub(crate) fn new(extra: Vec<FieldRule>) -> Self {
        let rule = |seam, path: &str, op, on_missing| FieldRule::new(seam, path, op, on_missing);
        let mut rules = vec![
            rule(
                Seam::Steps,
                "/text",
                JoinOp::AppendText,
                OnMissing::KeepLater,
            ),
            rule(
                Seam::Steps,
                "/output_ids",
                JoinOp::Append,
                OnMissing::KeepLater,
            ),
        ];
        rules.extend(INCREMENTAL_META_INFO_KEYS.map(|key| {
            rule(
                Seam::Steps,
                &format!("/meta_info/{key}"),
                JoinOp::Append,
                OnMissing::KeepLater,
            )
        }));
        rules.push(rule(
            Seam::PrefillDecode,
            "/meta_info/input_token_logprobs",
            JoinOp::Append,
            OnMissing::KeepLater,
        ));
        rules.extend(PREFILL_META_INFO_KEYS.map(|key| {
            rule(
                Seam::PrefillDecode,
                &format!("/meta_info/{key}"),
                JoinOp::TakeEarlier,
                OnMissing::KeepLater,
            )
        }));
        rules.extend(["routed_experts", "indexer_topk"].map(|key| {
            rule(
                Seam::PrefillDecode,
                &format!("/meta_info/{key}"),
                JoinOp::OverlayBase64Prefix,
                OnMissing::Remove,
            )
        }));
        rules.extend(extra);
        Self {
            rules: RuleSet {
                rules,
                prefill_request_flags: [
                    "return_logprob",
                    "return_routed_experts",
                    "return_indexer_topk",
                ]
                .map(String::from)
                .to_vec(),
            },
        }
    }

    pub(crate) fn shared() -> Arc<dyn ResponseMerger> {
        DEFAULT.clone()
    }
}

/// SGLang's abort echo (`_handle_abort_req`) is a full snapshot: whole text and
/// whole output lists, but `output_ids` holds only the last token again. It is
/// the only chunk whose `meta_info` has no `prompt_tokens`.
fn is_abort_echo(chunk: &Value) -> bool {
    chunk
        .pointer("/meta_info/finish_reason/type")
        .and_then(Value::as_str)
        == Some("abort")
        && chunk.pointer("/meta_info/prompt_tokens").is_none()
}

impl ResponseMerger for SglangMerger {
    fn wants_prefill_output(&self, request: &Value) -> bool {
        self.rules.wants_prefill_output(request)
    }

    fn merge(&self, seam: Seam, mut earlier: Value, later: &mut Value) -> Result<(), MergeError> {
        if seam == Seam::Steps && is_abort_echo(later) {
            if let (Some(ids), Some(object)) = (
                earlier.get_mut("output_ids").map(Value::take),
                later.as_object_mut(),
            ) {
                object.insert("output_ids".into(), ids);
            }
            return Ok(());
        }
        self.rules.merge(seam, earlier, later)
    }

    fn finish(&self, response: &mut Value) -> Result<(), MergeError> {
        let joined = response
            .get("output_ids")
            .and_then(Value::as_array)
            .map(Vec::len);
        let reported = response
            .pointer("/meta_info/completion_tokens")
            .and_then(Value::as_u64);
        match (joined, reported) {
            (Some(joined), Some(reported)) if joined as u64 != reported => {
                Err(MergeError::Invalid(format!(
                    "joined {joined} output tokens but SGLang reported {reported}"
                )))
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use serde_json::json;

    use super::*;
    use crate::protocols::response_merge::ResponseAssembler;

    fn b64(bytes: &[u8]) -> String {
        STANDARD.encode(bytes)
    }

    fn unary(prefill: Option<Value>, chunks: Vec<Value>) -> Result<Value, MergeError> {
        let mut assembler = ResponseAssembler::new(SglangMerger::shared());
        if let Some(prefill) = prefill {
            assembler.set_prefill(prefill);
        }
        for chunk in chunks {
            assembler.push(chunk)?;
        }
        Ok(assembler.finish()?.unwrap())
    }

    /// Incremental chunks as SGLang's tokenizer manager builds them.
    fn chunks() -> Vec<Value> {
        vec![
            json!({"text": "Hel", "output_ids": [10], "meta_info": {
                "id": "r", "finish_reason": null, "prompt_tokens": 3, "completion_tokens": 1,
                "cached_tokens": 0,
                "input_token_logprobs": [[null, 1, null], [-0.5, 2, null], [-0.4, 3, null]],
                "output_token_logprobs": [[-0.1, 10, null]],
                "output_token_logprobs_length": 1,
            }}),
            json!({"text": "lo", "output_ids": [11, 12], "meta_info": {
                "id": "r", "finish_reason": null, "prompt_tokens": 3, "completion_tokens": 3,
                "cached_tokens": 0,
                "input_token_logprobs": [[null, 1, null], [-0.5, 2, null], [-0.4, 3, null]],
                "output_token_logprobs": [[-0.2, 11, null], [-0.3, 12, null]],
                "output_token_logprobs_length": 3,
            }}),
            json!({"text": "", "output_ids": [], "meta_info": {
                "id": "r", "finish_reason": {"type": "length", "length": 3}, "prompt_tokens": 3,
                "completion_tokens": 3, "cached_tokens": 0, "e2e_latency": 0.5,
                "input_token_logprobs": [[null, 1, null], [-0.5, 2, null], [-0.4, 3, null]],
                "output_token_logprobs": [],
                "output_token_logprobs_length": 3,
                "routed_experts": b64(&[1, 2, 3, 4, 5]),
            }}),
        ]
    }

    #[test]
    fn unary_matches_sglang_non_streaming_output() {
        assert_eq!(
            unary(None, chunks()).unwrap(),
            json!({"text": "Hello", "output_ids": [10, 11, 12], "meta_info": {
                "id": "r", "finish_reason": {"type": "length", "length": 3}, "prompt_tokens": 3,
                "completion_tokens": 3, "cached_tokens": 0, "e2e_latency": 0.5,
                "input_token_logprobs": [[null, 1, null], [-0.5, 2, null], [-0.4, 3, null]],
                "output_token_logprobs": [[-0.1, 10, null], [-0.2, 11, null], [-0.3, 12, null]],
                "output_token_logprobs_length": 3,
                "routed_experts": b64(&[1, 2, 3, 4, 5]),
            }})
        );
    }

    /// Decode in P/D: no input logprobs, stale prompt rows in routed_experts.
    fn decode_chunks() -> Vec<Value> {
        chunks()
            .into_iter()
            .map(|mut chunk| {
                chunk["meta_info"]["input_token_logprobs"] = json!([]);
                chunk["meta_info"]["input_top_logprobs"] = json!([]);
                if chunk["meta_info"].get("routed_experts").is_some() {
                    chunk["meta_info"]["routed_experts"] = json!(b64(&[0, 0, 0, 4, 5]));
                }
                chunk
            })
            .collect()
    }

    fn prefill() -> Value {
        json!({"text": "Hel", "output_ids": [10], "meta_info": {
            "id": "p", "finish_reason": {"type": "length", "length": 1}, "prompt_tokens": 3,
            "completion_tokens": 1, "cached_tokens": 2,
            "input_token_logprobs": [[null, 1, null], [-0.5, 2, null], [-0.4, 3, null]],
            "input_top_logprobs": [null, [[-0.5, 2, null]], [[-0.4, 3, null]]],
            "output_token_logprobs": [[-0.1, 10, null]],
            "routed_experts": b64(&[1, 2, 3]),
        }})
    }

    #[test]
    fn unary_pd_takes_prompt_outputs_from_prefill() {
        let merged = unary(Some(prefill()), decode_chunks()).unwrap();
        let mut expected = unary(None, chunks()).unwrap();
        expected["meta_info"]["input_top_logprobs"] =
            json!([null, [[-0.5, 2, null]], [[-0.4, 3, null]]]);
        assert_eq!(merged, expected);
    }

    #[test]
    fn streamed_pd_chunks_carry_prompt_outputs_and_final_overlay() {
        let mut assembler = ResponseAssembler::new(SglangMerger::shared());
        assembler.set_prefill(prefill());
        let streamed: Vec<_> = decode_chunks()
            .into_iter()
            .map(|chunk| assembler.stream_chunk(chunk).unwrap())
            .collect();
        for (chunk, original) in streamed.iter().zip(chunks()) {
            assert_eq!(chunk["output_ids"], original["output_ids"]);
            assert_eq!(
                chunk["meta_info"]["input_token_logprobs"],
                original["meta_info"]["input_token_logprobs"]
            );
        }
        assert_eq!(
            streamed[2]["meta_info"]["routed_experts"],
            b64(&[1, 2, 3, 4, 5])
        );
        assert!(streamed[1]["meta_info"].get("routed_experts").is_none());
    }

    #[test]
    fn pd_without_prefill_routed_experts_drops_stale_rows() {
        let mut prefill = prefill();
        prefill["meta_info"]
            .as_object_mut()
            .unwrap()
            .remove("routed_experts");
        let merged = unary(Some(prefill), decode_chunks()).unwrap();
        assert!(merged["meta_info"].get("routed_experts").is_none());
        assert_eq!(merged["output_ids"], json!([10, 11, 12]));
    }

    #[test]
    fn abort_echo_keeps_joined_tokens_and_the_snapshot() {
        let mut chunks = chunks();
        chunks.truncate(2);
        chunks.push(json!({"text": "Hello", "output_ids": [12], "meta_info": {
            "id": "r", "finish_reason": {"type": "abort", "message": "Abort in waiting queue"},
            "e2e_latency": 0.2, "completion_tokens": 3,
            "output_token_logprobs": [[-0.1, 10, null], [-0.2, 11, null], [-0.3, 12, null]],
        }}));
        let merged = unary(None, chunks).unwrap();
        assert_eq!(merged["text"], "Hello");
        assert_eq!(merged["output_ids"], json!([10, 11, 12]));
        assert_eq!(
            merged["meta_info"]["output_token_logprobs"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn token_count_mismatch_is_reported() {
        let mut chunks = chunks();
        chunks[2]["meta_info"]["completion_tokens"] = json!(4);
        assert!(unary(None, chunks).is_err());
    }

    #[test]
    fn extra_rules_append_custom_info() {
        let merger: Arc<dyn ResponseMerger> = Arc::new(SglangMerger::new(vec![FieldRule::new(
            Seam::Steps,
            "/meta_info/my_info",
            JoinOp::Append,
            OnMissing::KeepLater,
        )]));
        let mut assembler = ResponseAssembler::new(merger);
        assembler
            .push(json!({"output_ids": [1], "meta_info": {"my_info": ["a"]}}))
            .unwrap();
        assembler
            .push(json!({"output_ids": [2], "meta_info": {"my_info": ["b"]}}))
            .unwrap();
        assert_eq!(
            assembler.finish().unwrap().unwrap()["meta_info"]["my_info"],
            json!(["a", "b"])
        );
    }

    #[test]
    fn prefill_output_is_requested_by_prompt_side_flags() {
        let merger = SglangMerger::shared();
        assert!(merger.wants_prefill_output(&json!({"return_logprob": true})));
        assert!(merger.wants_prefill_output(&json!({"return_routed_experts": true})));
        assert!(!merger.wants_prefill_output(&json!({"routed_experts_start_len": 2})));
    }
}
