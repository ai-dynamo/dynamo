// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang `/generate` join rules.
//!
//! With incremental streaming, each chunk holds only the new part of `text`,
//! `output_ids` and the output-side lists below; the rest is the current full
//! value (SGLang's `_coalesce_streaming_chunks`). In P/D, decode never runs the
//! prompt: input logprobs come from prefill, and prefill's `routed_experts`
//! rows replace decode's stale prompt rows (sgl-project/sglang#44002).

use std::sync::{Arc, LazyLock};

use crate::protocols::response_merge::{FieldRule, JoinOp, ResponseMerger, RuleSet};

const APPENDED: [&str; 7] = [
    "/text",
    "/output_ids",
    "/meta_info/output_token_logprobs",
    "/meta_info/output_top_logprobs",
    "/meta_info/output_token_ids_logprobs",
    "/meta_info/output_token_sampling_mask",
    "/meta_info/output_token_sampling_logprobs",
];

const FROM_PREFILL: [&str; 11] = [
    "/meta_info/input_token_logprobs",
    "/meta_info/input_top_logprobs",
    "/meta_info/input_token_ids_logprobs",
    "/meta_info/input_top_logprobs_val_flat",
    "/meta_info/input_top_logprobs_idx_flat",
    "/meta_info/input_top_logprobs_val_flat_b64",
    "/meta_info/input_top_logprobs_idx_flat_b64",
    "/meta_info/input_top_logprobs_val_flat_b64_dtype",
    "/meta_info/input_top_logprobs_idx_flat_b64_dtype",
    "/meta_info/input_top_logprobs_shape",
    "/meta_info/input_top_logprobs_null_prefix",
];

static RULES: LazyLock<Arc<RuleSet>> = LazyLock::new(|| {
    let rule = |op, drop_if_missing| {
        move |path: &str| FieldRule {
            path: path.into(),
            op,
            drop_if_missing,
        }
    };
    let overlay = ["/meta_info/routed_experts", "/meta_info/indexer_topk"];
    Arc::new(RuleSet::new(
        APPENDED
            .map(rule(JoinOp::Append, false))
            .into_iter()
            .chain(FROM_PREFILL.map(rule(JoinOp::TakePrefill, false)))
            .chain(overlay.map(rule(JoinOp::OverlayBase64Prefix, true))),
        // Every chunk carries prompt logprobs, which SGLang computes only from
        // a non-negative `logprob_start_len`; decode returns output logprobs.
        &[&["return_logprob", "logprob_start_len"]],
        // Only the final chunk carries routed experts and indexer top-k.
        &[&["return_routed_experts"], &["return_indexer_topk"]],
    ))
});

pub fn merger() -> Arc<dyn ResponseMerger> {
    RULES.clone()
}

#[cfg(test)]
mod tests {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use serde_json::{Value, json};

    use super::*;
    use crate::protocols::response_merge::{PrefillNeed, ResponseAssembler};

    fn chunk(
        text: &str,
        ids: &[u32],
        prompt: Value,
        finish: Value,
        experts: Option<&[u8]>,
    ) -> Value {
        let mut meta = json!({
            "finish_reason": finish, "input_token_logprobs": prompt,
            "output_token_logprobs": ids.iter().map(|id| json!([-0.1, id, null])).collect::<Vec<_>>(),
        });
        if let Some(experts) = experts {
            meta["routed_experts"] = json!(STANDARD.encode(experts));
        }
        json!({"text": text, "output_ids": ids, "meta_info": meta})
    }

    fn unary(prefill: Option<Value>, chunks: Vec<Value>) -> Value {
        let mut assembler = ResponseAssembler::new(merger());
        if let Some(prefill) = prefill {
            assembler.set_prefill(prefill);
        }
        chunks.into_iter().for_each(|chunk| assembler.push(chunk));
        assembler.finish().unwrap()
    }

    #[test]
    fn aggregated_and_disaggregated_responses_match_sglang_unary() {
        let prompt = json!([[null, 1, null], [-0.5, 2, null]]);
        let stop = json!({"type": "stop"});
        let aggregated = vec![
            chunk("Hel", &[10], prompt.clone(), Value::Null, None),
            chunk(
                "lo",
                &[11, 12],
                prompt.clone(),
                stop.clone(),
                Some(&[1, 2, 3, 4, 5]),
            ),
        ];
        let expected = json!({"text": "Hello", "output_ids": [10, 11, 12], "meta_info": {
            "finish_reason": stop, "input_token_logprobs": prompt,
            "output_token_logprobs": [[-0.1, 10, null], [-0.1, 11, null], [-0.1, 12, null]],
            "routed_experts": STANDARD.encode([1, 2, 3, 4, 5]),
        }});
        assert_eq!(unary(None, aggregated), expected);

        let prefill = json!({"meta_info": {
            "input_token_logprobs": prompt, "routed_experts": STANDARD.encode([1, 2, 3]),
        }});
        let decode = || {
            vec![
                chunk("Hel", &[10], json!([]), Value::Null, None),
                chunk(
                    "lo",
                    &[11, 12],
                    json!([]),
                    stop.clone(),
                    Some(&[0, 0, 0, 4, 5]),
                ),
            ]
        };
        assert_eq!(unary(Some(prefill.clone()), decode()), expected);

        let mut streamed = ResponseAssembler::new(merger());
        streamed.set_prefill(prefill);
        let rendered: Vec<Value> = decode()
            .into_iter()
            .map(|chunk| streamed.render_chunk(chunk, Value::clone))
            .collect();
        assert!(
            rendered
                .iter()
                .all(|chunk| chunk["meta_info"]["input_token_logprobs"] == prompt)
        );
        assert_eq!(
            rendered[1]["meta_info"]["routed_experts"],
            expected["meta_info"]["routed_experts"]
        );

        let without_prefill_rows = unary(Some(json!({"meta_info": {}})), decode());
        assert!(
            without_prefill_rows["meta_info"]
                .get("routed_experts")
                .is_none()
        );
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
