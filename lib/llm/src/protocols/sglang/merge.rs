// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang `/generate` join rules.
//!
//! With incremental streaming, each chunk holds only the new part of `text`,
//! `output_ids` and the output-side lists; the rest is the current full value.
//! In P/D, decode never runs the prompt: input logprobs come from prefill, and
//! prefill's `routed_experts` rows replace decode's stale prompt rows
//! (sgl-project/sglang#44002).
//!
//! The rules live in `merge_rules.json` and mirror SGLang code. When bumping
//! SGLang, re-check these upstream sources and update the file:
//!
//! - `python/sglang/srt/managers/tokenizer_manager.py`:
//!   `_INCREMENTAL_STREAMING_META_INFO_KEYS` and `_coalesce_streaming_chunks`
//!   (`append`), `_handle_batch_output` and `add_logprob_to_meta_info` (every
//!   `meta_info` key), `_build_flat_input_top_logprobs_fields*` (flat input
//!   top-logprob keys).
//! - `python/sglang/srt/managers/io_struct.py`: `GenerateReqInput` (the
//!   `return_*` request flags).
//! - `python/sglang/srt/managers/scheduler_components/output_streamer.py`:
//!   decode servers send no input logprobs (`take_prefill`).
//! - `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`
//!   (`_maybe_collect_routed_experts`) and `disaggregation/prefill.py`: which
//!   server reports which `routed_experts` rows (`overlay_base64_prefix`).
//!
//! `components/src/dynamo/sglang/tests/test_sglang_generate_merge_contract.py`
//! checks the file against the installed SGLang and fails on unclassified
//! fields.

use std::sync::{Arc, LazyLock};

use serde::Deserialize;

use crate::protocols::response_merge::{FieldRule, JoinOp, ResponseMerger, RuleSet};

/// Shared with the SGLang contract test, which also reads `take_last` and
/// `other_request_flags`: fields that keep the last value and need no rule.
const RULES_JSON: &str = include_str!("merge_rules.json");

#[derive(Deserialize)]
struct RulesFile {
    append: Vec<String>,
    take_prefill: Vec<String>,
    overlay_base64_prefix: Vec<String>,
    prefill_request_flags: PrefillRequestFlags,
}

/// Groups of request fields; a request needs prefill's output when every
/// field of one group is set.
#[derive(Deserialize)]
struct PrefillRequestFlags {
    /// Every chunk carries the output: prompt logprobs, which SGLang computes
    /// only from a non-negative `logprob_start_len`; decode returns output
    /// logprobs.
    before_first: Vec<Vec<String>>,
    /// Only the final chunk carries routed experts and indexer top-k.
    before_last: Vec<Vec<String>>,
}

static RULES: LazyLock<Arc<RuleSet>> = LazyLock::new(|| {
    let file: RulesFile = serde_json::from_str(RULES_JSON).expect("valid merge_rules.json");
    let rules = |paths: Vec<String>, op, drop_if_missing| {
        paths.into_iter().map(move |path| FieldRule {
            path,
            op,
            drop_if_missing,
        })
    };
    let first = str_groups(&file.prefill_request_flags.before_first);
    let last = str_groups(&file.prefill_request_flags.before_last);
    let before_first: Vec<&[&str]> = first.iter().map(Vec::as_slice).collect();
    let before_last: Vec<&[&str]> = last.iter().map(Vec::as_slice).collect();
    Arc::new(RuleSet::new(
        rules(file.append, JoinOp::Append, false)
            .chain(rules(file.take_prefill, JoinOp::TakePrefill, false))
            .chain(rules(
                file.overlay_base64_prefix,
                JoinOp::OverlayBase64Prefix,
                true,
            )),
        &before_first,
        &before_last,
    ))
});

fn str_groups(groups: &[Vec<String>]) -> Vec<Vec<&str>> {
    groups
        .iter()
        .map(|group| group.iter().map(String::as_str).collect())
        .collect()
}

pub fn merger() -> Arc<dyn ResponseMerger> {
    RULES.clone()
}

#[cfg(test)]
mod tests {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use proptest::prelude::*;
    use serde_json::{Value, json};

    use super::*;
    use crate::protocols::response_merge::{PrefillNeed, ResponseAssembler};

    /// One SGLang response and how SGLang streams it, aggregated and in P/D.
    #[derive(Debug)]
    struct Case {
        /// What SGLang returns for `stream=false`.
        unary: Value,
        /// Aggregated incremental chunks.
        aggregated: Vec<Value>,
        /// Decode's chunks: no input logprobs, stale routed-experts prompt rows.
        decode: Vec<Value>,
        /// Prefill's final response.
        prefill: Value,
    }

    prop_compose! {
        fn case()(
            prompt in prop::collection::vec(any::<u32>(), 1..5),
            output in prop::collection::vec((any::<u32>(), "\\PC{0,3}", -100i32..0), 1..8),
            row_bytes in 1usize..5,
        )(
            experts in prop::collection::vec(any::<u8>(), (prompt.len() + output.len() - 1) * row_bytes),
            stale in prop::collection::vec(any::<u8>(), prompt.len() * row_bytes),
            cuts in prop::collection::vec(any::<bool>(), output.len() - 1),
            prompt in Just(prompt),
            output in Just(output),
        ) -> Case {
            let input_logprobs: Vec<Value> = prompt
                .iter()
                .enumerate()
                .map(|(i, id)| json!([if i == 0 { Value::Null } else { json!(-(i as i64)) }, id, null]))
                .collect();
            // Prefill reports the prompt rows; decode's copy of them is stale.
            let prompt_bytes = stale.len();
            let mut decode_experts = stale.clone();
            decode_experts.extend_from_slice(&experts[prompt_bytes..]);

            // Chunk boundaries: after token i when cuts[i], and at the end.
            let mut ends: Vec<usize> = (1..output.len()).filter(|&i| cuts[i - 1]).collect();
            ends.push(output.len());
            let mut aggregated = Vec::new();
            let mut start = 0;
            for &end in &ends {
                let tokens = &output[start..end];
                let last = end == output.len();
                let mut meta = json!({
                    "id": "rid",
                    "prompt_tokens": prompt.len(),
                    "completion_tokens": end,
                    "finish_reason": if last { json!({"type": "length"}) } else { Value::Null },
                    "input_token_logprobs": input_logprobs,
                    "output_token_logprobs": tokens.iter().map(|(id, _, lp)| json!([lp, id, null])).collect::<Vec<_>>(),
                });
                if last {
                    meta["routed_experts"] = json!(STANDARD.encode(&experts));
                }
                aggregated.push(json!({
                    "text": tokens.iter().map(|(_, text, _)| text.as_str()).collect::<String>(),
                    "output_ids": tokens.iter().map(|(id, ..)| *id).collect::<Vec<_>>(),
                    "meta_info": meta,
                }));
                start = end;
            }
            let decode = aggregated
                .iter()
                .map(|chunk| {
                    let mut chunk = chunk.clone();
                    chunk["meta_info"]["input_token_logprobs"] = json!([]);
                    if chunk["meta_info"].get("routed_experts").is_some() {
                        chunk["meta_info"]["routed_experts"] = json!(STANDARD.encode(&decode_experts));
                    }
                    chunk
                })
                .collect();
            let mut unary = aggregated.last().unwrap().clone();
            unary["text"] = json!(output.iter().map(|(_, text, _)| text.as_str()).collect::<String>());
            unary["output_ids"] = json!(output.iter().map(|(id, ..)| *id).collect::<Vec<_>>());
            unary["meta_info"]["output_token_logprobs"] =
                json!(output.iter().map(|(id, _, lp)| json!([lp, id, null])).collect::<Vec<_>>());
            let prefill = json!({
                "text": "", "output_ids": [output[0].0],
                "meta_info": {
                    "id": "prefill-rid",
                    "completion_tokens": 1,
                    "finish_reason": {"type": "length"},
                    "input_token_logprobs": input_logprobs,
                    "routed_experts": STANDARD.encode(&experts[..prompt_bytes]),
                },
            });
            Case { unary, aggregated, decode, prefill }
        }
    }

    fn unary(prefill: Option<&Value>, chunks: &[Value]) -> Value {
        let mut assembler = ResponseAssembler::new(merger());
        if let Some(prefill) = prefill {
            assembler.set_prefill(prefill.clone());
        }
        chunks
            .iter()
            .cloned()
            .for_each(|chunk| assembler.push(chunk));
        assembler.finish().unwrap()
    }

    proptest! {
        /// Aggregated chunks join into SGLang's unary response.
        #[test]
        fn aggregated_chunks_join_into_the_unary_response(case in case()) {
            prop_assert_eq!(unary(None, &case.aggregated), case.unary);
        }

        /// P/D returns what aggregated serving returns, unary and streamed:
        /// prompt logprobs and routed-experts prompt rows come from prefill.
        #[test]
        fn disaggregated_responses_match_aggregated(case in case()) {
            prop_assert_eq!(unary(Some(&case.prefill), &case.decode), case.unary);
            let mut streamed = ResponseAssembler::new(merger());
            streamed.set_prefill(case.prefill.clone());
            for (decode, aggregated) in case.decode.into_iter().zip(&case.aggregated) {
                prop_assert_eq!(&streamed.render_chunk(decode, Value::clone), aggregated);
            }
        }

        /// Without prefill's rows, decode's stale routed experts are dropped
        /// rather than returned; everything else still matches.
        #[test]
        fn missing_prefill_rows_drop_routed_experts(case in case()) {
            let mut prefill = case.prefill;
            prefill["meta_info"].as_object_mut().unwrap().remove("routed_experts");
            let mut expected = case.unary;
            expected["meta_info"].as_object_mut().unwrap().remove("routed_experts");
            prop_assert_eq!(unary(Some(&prefill), &case.decode), expected);
        }
    }

    #[test]
    fn rules_file_classifies_each_field_once() {
        let file: Value = serde_json::from_str(RULES_JSON).unwrap();
        let mut seen = std::collections::HashSet::new();
        for category in [
            "append",
            "take_prefill",
            "overlay_base64_prefix",
            "take_last",
        ] {
            for path in file[category].as_array().unwrap() {
                let path = path.as_str().unwrap();
                assert!(path.starts_with('/'), "{path} is not a JSON pointer");
                assert!(seen.insert(path), "{path} is classified twice");
            }
        }
        let flags = &file["prefill_request_flags"];
        for group in ["before_first", "before_last"] {
            assert!(!flags[group].as_array().unwrap().is_empty());
        }
    }

    #[test]
    fn prompt_side_flags_request_prefill_output() {
        let need = |request: Value| merger().prefill_need(&request);
        let experts = json!({"return_routed_experts": true});
        assert_eq!(need(experts), PrefillNeed::BeforeLast);
        let prompt = json!({"return_logprob": true, "logprob_start_len": 0});
        assert_eq!(need(prompt), PrefillNeed::BeforeFirst);
        for output_only in [
            json!({"return_logprob": true}),
            json!({"return_logprob": true, "logprob_start_len": -1}),
            json!({"routed_experts_start_len": 2}),
        ] {
            assert_eq!(need(output_only), PrefillNeed::None);
        }
    }
}
