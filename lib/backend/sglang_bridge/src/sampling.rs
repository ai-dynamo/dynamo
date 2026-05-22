// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request shaping: Dynamo `PreprocessedRequest` -> SGLang v1 `SamplingParams`.

use dynamo_backend_common::{FinishReason, PreprocessedRequest};

use crate::proto::v1::SamplingParams;

pub fn build_sampling_params(req: &PreprocessedRequest) -> SamplingParams {
    let (json_schema, regex) = build_structured_constraint(req);
    SamplingParams {
        temperature: req.sampling_options.temperature,
        top_p: req.sampling_options.top_p,
        top_k: req.sampling_options.top_k,
        min_p: req.sampling_options.min_p,
        frequency_penalty: req.sampling_options.frequency_penalty,
        presence_penalty: req.sampling_options.presence_penalty,
        repetition_penalty: req.sampling_options.repetition_penalty,
        max_new_tokens: req.stop_conditions.max_tokens.map(|v| v as i32),
        min_new_tokens: req.stop_conditions.min_tokens.map(|v| v as i32),
        stop: req.stop_conditions.stop.clone().unwrap_or_default(),
        stop_token_ids: req
            .stop_conditions
            .stop_token_ids_hidden
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|t| t as i32)
            .collect(),
        ignore_eos: req.stop_conditions.ignore_eos,
        n: req.sampling_options.n.map(|v| v as i32),
        json_schema,
        regex,
    }
}

/// Priority: json > regex > choice (anchored alternation) > grammar
/// (forwarded as regex; v1 has no EBNF field).
fn build_structured_constraint(req: &PreprocessedRequest) -> (Option<String>, Option<String>) {
    let Some(g) = req.sampling_options.guided_decoding.as_ref() else {
        return (None, None);
    };
    if let Some(schema) = &g.json {
        match serde_json::to_string(schema) {
            Ok(s) => return (Some(s), None),
            Err(e) => {
                tracing::warn!(error = %e, "guided_decoding.json serialize failed; dropping constraint");
            }
        }
    }
    if let Some(regex) = &g.regex {
        return (None, Some(regex.clone()));
    }
    if let Some(choices) = &g.choice
        && !choices.is_empty()
    {
        let alt: Vec<String> = choices.iter().map(|c| regex_escape(c)).collect();
        return (None, Some(format!("^({})$", alt.join("|"))));
    }
    if let Some(grammar) = &g.grammar {
        tracing::warn!("guided_decoding.grammar set but v1 proto has no EBNF field; forwarding as regex");
        return (None, Some(grammar.clone()));
    }
    (None, None)
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(
            c,
            '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | '\\'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

pub fn parse_finish_reason(raw: &str) -> FinishReason {
    match raw {
        "stop" | "" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "abort" => FinishReason::Cancelled,
        other => FinishReason::Error(format!("unknown sglang finish_reason: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_finish_reasons() {
        assert!(matches!(parse_finish_reason("stop"), FinishReason::Stop));
        assert!(matches!(parse_finish_reason(""), FinishReason::Stop));
        assert!(matches!(parse_finish_reason("length"), FinishReason::Length));
        assert!(matches!(parse_finish_reason("abort"), FinishReason::Cancelled));
        match parse_finish_reason("???") {
            FinishReason::Error(msg) => assert!(msg.contains("???")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn regex_escape_handles_metas_and_passthrough() {
        assert_eq!(regex_escape("foo"), "foo");
        assert_eq!(regex_escape("a.b"), "a\\.b");
        assert_eq!(regex_escape("(x|y)"), "\\(x\\|y\\)");
        assert_eq!(regex_escape("\\"), "\\\\");
    }
}
