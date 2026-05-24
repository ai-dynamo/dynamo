// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request shaping: Dynamo sampling/stop options -> SGLang v1 `SamplingParams`.

use dynamo_backend_common::{FinishReason, SamplingOptions, StopConditions};

use crate::proto::v1::SamplingParams;

pub fn build_sampling_params(
    sampling: &SamplingOptions,
    stop: &StopConditions,
) -> SamplingParams {
    let (json_schema, regex) = build_structured_constraint(sampling);
    SamplingParams {
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        frequency_penalty: sampling.frequency_penalty,
        presence_penalty: sampling.presence_penalty,
        repetition_penalty: sampling.repetition_penalty,
        max_new_tokens: stop.max_tokens.map(|v| v as i32),
        min_new_tokens: stop.min_tokens.map(|v| v as i32),
        stop: stop.stop.clone().unwrap_or_default(),
        stop_token_ids: stop
            .stop_token_ids_hidden
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|t| t as i32)
            .collect(),
        ignore_eos: stop.ignore_eos,
        n: sampling.n.map(|v| v as i32),
        json_schema,
        regex,
    }
}

/// Priority: json > regex > choice (anchored alternation) > grammar
/// (forwarded as regex; v1 has no EBNF field).
fn build_structured_constraint(sampling: &SamplingOptions) -> (Option<String>, Option<String>) {
    let Some(g) = sampling.guided_decoding.as_ref() else {
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
    // No EBNF field on v1; forward grammar as regex (best-effort).
    if let Some(grammar) = &g.grammar {
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

/// SGLang emits exactly three `finish_reason.type` values from `schedule_batch`
/// (stop / length / abort). An absent/empty value on a `finished=true` chunk
/// indicates the gRPC error path (empty meta_info), so we surface it as an
/// error rather than silently calling it "stop".
pub fn parse_finish_reason(raw: &str) -> FinishReason {
    match raw {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "abort" => FinishReason::Cancelled,
        "" => FinishReason::Error("missing finish_reason on terminal chunk".to_string()),
        other => FinishReason::Error(format!("unknown sglang finish_reason: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_finish_reasons() {
        assert!(matches!(parse_finish_reason("stop"), FinishReason::Stop));
        assert!(matches!(parse_finish_reason("length"), FinishReason::Length));
        assert!(matches!(parse_finish_reason("abort"), FinishReason::Cancelled));
        assert!(matches!(parse_finish_reason(""), FinishReason::Error(_)));
        match parse_finish_reason("???") {
            FinishReason::Error(msg) => assert!(msg.contains("???")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    /// Snapshot test for the sampling-param mapping. If a SGLang proto field
    /// is renamed/moved during an upstream bump, this fires immediately
    /// rather than silently dropping the value at runtime.
    #[test]
    fn build_sampling_params_maps_all_fields() {
        let sampling = SamplingOptions {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.2),
            repetition_penalty: Some(1.1),
            n: Some(2),
            ..Default::default()
        };
        let stop = StopConditions {
            max_tokens: Some(128),
            min_tokens: Some(4),
            stop: Some(vec!["</s>".into()]),
            stop_token_ids_hidden: Some(vec![2, 11]),
            ignore_eos: Some(true),
            ..Default::default()
        };
        let params = build_sampling_params(&sampling, &stop);
        assert_eq!(params.temperature, Some(0.7));
        assert_eq!(params.top_p, Some(0.9));
        assert_eq!(params.top_k, Some(40));
        assert_eq!(params.min_p, Some(0.05));
        assert_eq!(params.frequency_penalty, Some(0.1));
        assert_eq!(params.presence_penalty, Some(0.2));
        assert_eq!(params.repetition_penalty, Some(1.1));
        assert_eq!(params.max_new_tokens, Some(128));
        assert_eq!(params.min_new_tokens, Some(4));
        assert_eq!(params.stop, vec!["</s>".to_string()]);
        assert_eq!(params.stop_token_ids, vec![2, 11]);
        assert_eq!(params.ignore_eos, Some(true));
        assert_eq!(params.n, Some(2));
        assert!(params.json_schema.is_none());
        assert!(params.regex.is_none());
    }

    #[test]
    fn build_sampling_params_empty_when_no_options() {
        let params = build_sampling_params(&SamplingOptions::default(), &StopConditions::default());
        assert!(params.temperature.is_none());
        assert!(params.max_new_tokens.is_none());
        assert!(params.stop.is_empty());
        assert!(params.stop_token_ids.is_empty());
        assert!(params.json_schema.is_none());
        assert!(params.regex.is_none());
    }
}
