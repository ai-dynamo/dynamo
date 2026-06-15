// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request shaping: Dynamo sampling/stop options -> SMG SGLang `SamplingParams`.

use dynamo_backend_common::{FinishReason, OutputOptions, SamplingOptions, StopConditions};

use crate::proto::scheduler::{SamplingParams, sampling_params::Constraint};

pub fn build_sampling_params(
    sampling: &SamplingOptions,
    stop: &StopConditions,
    output: &OutputOptions,
) -> SamplingParams {
    let constraint = build_structured_constraint(sampling);
    SamplingParams {
        temperature: sampling.temperature.unwrap_or(1.0),
        top_p: sampling.top_p.unwrap_or(1.0),
        top_k: sampling.top_k.unwrap_or(-1),
        min_p: sampling.min_p.unwrap_or(0.0),
        frequency_penalty: sampling.frequency_penalty.unwrap_or(0.0),
        presence_penalty: sampling.presence_penalty.unwrap_or(0.0),
        repetition_penalty: sampling.repetition_penalty.unwrap_or(1.0),
        max_new_tokens: stop.max_tokens,
        min_new_tokens: stop.min_tokens.unwrap_or(0),
        stop: stop.stop.clone().unwrap_or_default(),
        stop_token_ids: stop_token_ids(stop),
        ignore_eos: stop.ignore_eos.unwrap_or(false),
        n: sampling.n.map(u32::from).unwrap_or(1),
        skip_special_tokens: output.skip_special_tokens.unwrap_or(true),
        spaces_between_special_tokens: true,
        no_stop_trim: false,
        stream_interval: None,
        logit_bias: Default::default(),
        custom_params: None,
        constraint,
    }
}

fn stop_token_ids(stop: &StopConditions) -> Vec<u32> {
    let mut out = stop.stop_token_ids.clone().unwrap_or_default();
    if let Some(hidden) = stop.stop_token_ids_hidden.as_ref() {
        out.extend(hidden.iter().copied());
    }
    out
}

/// Priority: json > regex > choice (anchored alternation) > grammar.
fn build_structured_constraint(sampling: &SamplingOptions) -> Option<Constraint> {
    let Some(g) = sampling.guided_decoding.as_ref() else {
        return None;
    };
    if let Some(schema) = &g.json {
        match serde_json::to_string(schema) {
            Ok(s) => return Some(Constraint::JsonSchema(s)),
            Err(e) => {
                tracing::warn!(error = %e, "guided_decoding.json serialize failed; dropping constraint");
            }
        }
    }
    if let Some(regex) = &g.regex {
        return Some(Constraint::Regex(regex.clone()));
    }
    if let Some(choices) = &g.choice
        && !choices.is_empty()
    {
        let alt: Vec<String> = choices.iter().map(|c| regex_escape(c)).collect();
        return Some(Constraint::Regex(format!("^({})$", alt.join("|"))));
    }
    if let Some(grammar) = &g.grammar {
        return Some(Constraint::EbnfGrammar(grammar.clone()));
    }
    None
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

/// SMG emits exactly three terminal `finish_reason` values from SGLang
/// (stop / length / abort). An absent value is unexpected on a terminal
/// response, so surface it as an error rather than silently calling it "stop".
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
        assert!(matches!(
            parse_finish_reason("length"),
            FinishReason::Length
        ));
        assert!(matches!(
            parse_finish_reason("abort"),
            FinishReason::Cancelled
        ));
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
        let params = build_sampling_params(&sampling, &stop, &OutputOptions::default());
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 40);
        assert_eq!(params.min_p, 0.05);
        assert_eq!(params.frequency_penalty, 0.1);
        assert_eq!(params.presence_penalty, 0.2);
        assert_eq!(params.repetition_penalty, 1.1);
        assert_eq!(params.max_new_tokens, Some(128));
        assert_eq!(params.min_new_tokens, 4);
        assert_eq!(params.stop, vec!["</s>".to_string()]);
        assert_eq!(params.stop_token_ids, vec![2, 11]);
        assert!(params.ignore_eos);
        assert_eq!(params.n, 2);
        assert!(params.skip_special_tokens);
        assert!(params.spaces_between_special_tokens);
        assert!(params.constraint.is_none());
    }

    #[test]
    fn build_sampling_params_uses_sglang_defaults_when_unset() {
        let params = build_sampling_params(
            &SamplingOptions::default(),
            &StopConditions::default(),
            &OutputOptions::default(),
        );
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, -1);
        assert_eq!(params.min_p, 0.0);
        assert_eq!(params.repetition_penalty, 1.0);
        assert!(params.max_new_tokens.is_none());
        assert_eq!(params.min_new_tokens, 0);
        assert!(params.stop.is_empty());
        assert!(params.stop_token_ids.is_empty());
        assert!(!params.ignore_eos);
        assert_eq!(params.n, 1);
        assert!(params.constraint.is_none());
    }

    #[test]
    fn build_sampling_params_forwards_skip_special_tokens() {
        let params = build_sampling_params(
            &SamplingOptions::default(),
            &StopConditions::default(),
            &OutputOptions {
                skip_special_tokens: Some(false),
                ..Default::default()
            },
        );
        assert!(!params.skip_special_tokens);
    }
}
