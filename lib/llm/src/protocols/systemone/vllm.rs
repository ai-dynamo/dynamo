// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use super::SystemOneError;
use crate::protocols::common::{FinishReason, llm_backend::LLMEngineOutput};
use crate::protocols::openai::generate::GenerateRequest;

/// Build an isolated one-token read, without inherited generation controls.
pub fn causal_scoring_request(
    prompt_ids: Vec<u32>,
    label_ids: &[u32],
    cache_salt: &str,
) -> Result<GenerateRequest, SystemOneError> {
    validate_inputs(label_ids, prompt_ids.len())?;
    if cache_salt.is_empty() {
        return Err(invalid("cache salt cannot be empty"));
    }
    serde_json::from_value(serde_json::json!({
        "token_ids": prompt_ids,
        "cache_salt": cache_salt,
        "sampling_params": {
            "max_tokens": 1,
            "min_tokens": 0,
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "logprob_token_ids": label_ids,
            "logprobs": null,
            "prompt_logprobs": null,
            "detokenize": false,
            "stop": [],
            "stop_token_ids": [],
            "ignore_eos": false,
            "bad_words": [],
            "logit_bias": {},
            "allowed_token_ids": null,
            "structured_outputs": null
        }
    }))
    .map_err(|cause| invalid(format!("invalid vLLM scoring request: {cause}")))
}

#[derive(Debug, PartialEq)]
pub struct VllmScoreResult {
    pub scores: Vec<f64>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cached_tokens: Option<usize>,
}

/// Collect delta outputs before accepting a complete, measured scoring result.
pub struct VllmScoreCollector {
    expected_ids: Vec<u32>,
    prompt_tokens: usize,
    sampled_token: Option<u32>,
    scores: Option<HashMap<u32, f64>>,
    cached_tokens: Option<usize>,
    terminal: bool,
    failed: bool,
}

impl VllmScoreCollector {
    pub fn new(expected_ids: Vec<u32>, prompt_tokens: usize) -> Self {
        Self {
            expected_ids,
            prompt_tokens,
            sampled_token: None,
            scores: None,
            cached_tokens: None,
            terminal: false,
            failed: false,
        }
    }

    pub fn observe(&mut self, frame: &LLMEngineOutput) -> Result<(), SystemOneError> {
        let result = self.observe_frame(frame);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn observe_frame(&mut self, frame: &LLMEngineOutput) -> Result<(), SystemOneError> {
        validate_inputs(&self.expected_ids, self.prompt_tokens)?;
        if self.failed || self.terminal {
            return Err(invalid("scoring frame after failure or terminal result"));
        }
        if frame.index.is_some_and(|index| index != 0) {
            return Err(invalid("scoring output index must be zero"));
        }
        if frame.token_ids.len() > 1
            || (!frame.token_ids.is_empty() && self.sampled_token.is_some())
        {
            return Err(invalid("scoring must generate exactly one token"));
        }
        if let Some(&token) = frame.token_ids.first() {
            self.sampled_token = Some(token);
        }
        if let Some(rows) = &frame.top_logprobs {
            self.observe_scores(rows)?;
        }
        if let Some(reason) = &frame.finish_reason {
            if !matches!(
                reason,
                FinishReason::Length | FinishReason::Stop | FinishReason::EoS
            ) {
                return Err(invalid("backend did not finish scoring successfully"));
            }
            if self.sampled_token.is_none() || frame.completion_usage.is_none() {
                return Err(invalid(
                    "terminal scoring result lacks generated token or usage",
                ));
            }
        }
        self.observe_usage(frame)?;
        self.terminal = frame.finish_reason.is_some();
        Ok(())
    }

    fn observe_scores(
        &mut self,
        rows: &[Vec<crate::protocols::common::llm_backend::TopLogprob>],
    ) -> Result<(), SystemOneError> {
        if self.scores.is_some()
            || rows.len() != 1
            || rows[0].len() < self.expected_ids.len()
            || rows[0].len() > self.expected_ids.len() + 1
        {
            return Err(invalid("scoring requires one complete candidate position"));
        }
        let mut scores = HashMap::with_capacity(rows[0].len());
        for entry in &rows[0] {
            if !entry.logprob.is_finite() || entry.logprob > 0.0 {
                return Err(invalid("candidate logprobs must be finite and nonpositive"));
            }
            if scores.insert(entry.token_id, entry.logprob).is_some() {
                return Err(invalid("candidate token IDs must be unique"));
            }
        }
        if self.expected_ids.iter().any(|id| !scores.contains_key(id)) {
            return Err(invalid("scoring result lacks requested candidate IDs"));
        }
        self.scores = Some(scores);
        Ok(())
    }

    fn observe_usage(&mut self, frame: &LLMEngineOutput) -> Result<(), SystemOneError> {
        let Some(usage) = &frame.completion_usage else {
            return Ok(());
        };
        let output_tokens = usize::from(self.sampled_token.is_some());
        if usage.prompt_tokens as usize != self.prompt_tokens
            || usage.completion_tokens as usize != output_tokens
            || u64::from(usage.total_tokens)
                != u64::from(usage.prompt_tokens) + u64::from(usage.completion_tokens)
        {
            return Err(invalid(
                "scoring usage does not match prompt and generated tokens",
            ));
        }
        let cached = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .map(|tokens| tokens as usize);
        if cached.is_some_and(|tokens| tokens > self.prompt_tokens)
            || self.cached_tokens.zip(cached).is_some_and(|(a, b)| a != b)
        {
            return Err(invalid("invalid or inconsistent cache-read token count"));
        }
        self.cached_tokens = cached.or(self.cached_tokens);
        Ok(())
    }

    pub fn finish(self) -> Result<VllmScoreResult, SystemOneError> {
        validate_inputs(&self.expected_ids, self.prompt_tokens)?;
        if self.failed || !self.terminal {
            return Err(invalid("scoring stream failed or lacks a terminal result"));
        }
        let sampled = self
            .sampled_token
            .ok_or_else(|| invalid("missing generated token"))?;
        let scores = self
            .scores
            .ok_or_else(|| invalid("missing candidate scores"))?;
        if scores
            .keys()
            .any(|id| !self.expected_ids.contains(id) && *id != sampled)
        {
            return Err(invalid(
                "scoring returned an unrequested, unsampled token ID",
            ));
        }
        let ordered = self
            .expected_ids
            .iter()
            .map(|id| {
                scores
                    .get(id)
                    .copied()
                    .ok_or_else(|| invalid("missing requested candidate"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(VllmScoreResult {
            scores: ordered,
            input_tokens: self.prompt_tokens,
            output_tokens: 1,
            cached_tokens: self.cached_tokens,
        })
    }
}

fn validate_inputs(ids: &[u32], prompt_tokens: usize) -> Result<(), SystemOneError> {
    if prompt_tokens == 0 || ids.is_empty() {
        return Err(invalid("prompt and candidate token IDs cannot be empty"));
    }
    if ids.iter().copied().collect::<HashSet<_>>().len() != ids.len() {
        return Err(invalid("requested candidate token IDs must be unique"));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> SystemOneError {
    SystemOneError::CandidateScores(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn output(value: Value) -> LLMEngineOutput {
        serde_json::from_value(value).unwrap()
    }

    fn complete() -> LLMEngineOutput {
        output(json!({
            "token_ids": [99], "index": 0,
            "top_logprobs": [[
                {"rank": 2, "token_id": 20, "token": null, "logprob": -2.0},
                {"rank": 1, "token_id": 99, "token": null, "logprob": -0.5},
                {"rank": 3, "token_id": 10, "token": null, "logprob": -3.0}
            ]],
            "finish_reason": "length",
            "completion_usage": {
                "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 2}
            }
        }))
    }

    #[test]
    fn scoring_request_has_explicit_neutral_one_token_sampling() {
        let request = causal_scoring_request(vec![1, 2, 3], &[20, 10], "tenant").unwrap();
        request.validate().unwrap();
        assert_eq!(request.token_ids, [1, 2, 3]);
        assert_eq!(request.cache_salt.as_deref(), Some("tenant"));
        assert!(!request.stream);
        let params = request.sampling_params.as_value();
        for (key, expected) in [
            ("max_tokens", json!(1)),
            ("n", json!(1)),
            ("temperature", json!(0.0)),
            ("top_p", json!(1.0)),
            ("top_k", json!(-1)),
            ("min_p", json!(0.0)),
            ("frequency_penalty", json!(0.0)),
            ("presence_penalty", json!(0.0)),
            ("repetition_penalty", json!(1.0)),
            ("logprob_token_ids", json!([20, 10])),
            ("detokenize", json!(false)),
            ("stop", json!([])),
            ("stop_token_ids", json!([])),
            ("bad_words", json!([])),
        ] {
            assert_eq!(params[key], expected, "{key}");
        }
        assert!(causal_scoring_request(vec![], &[10], "tenant").is_err());
        assert!(causal_scoring_request(vec![1], &[], "tenant").is_err());
        assert!(causal_scoring_request(vec![1], &[10, 10], "tenant").is_err());
        assert!(causal_scoring_request(vec![1], &[10], "").is_err());
    }

    #[test]
    fn complete_scores_follow_requested_order_and_preserve_measured_usage() {
        let mut collector = VllmScoreCollector::new(vec![10, 20], 3);
        collector.observe(&complete()).unwrap();
        let result = collector.finish().unwrap();
        assert_eq!(result.scores, [-3.0, -2.0]);
        assert_eq!(result.input_tokens, 3);
        assert_eq!(result.output_tokens, 1);
        assert_eq!(result.cached_tokens, Some(2));
    }

    #[test]
    fn separate_score_and_terminal_frames_preserve_unknown_cache_usage() {
        let scores = LLMEngineOutput {
            finish_reason: None,
            completion_usage: None,
            ..complete()
        };
        let terminal = output(json!({
            "token_ids": [], "index": 0, "finish_reason": "length",
            "completion_usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        }));
        let mut collector = VllmScoreCollector::new(vec![20, 10], 3);
        collector.observe(&scores).unwrap();
        collector.observe(&terminal).unwrap();
        let result = collector.finish().unwrap();
        assert_eq!(result.scores, [-2.0, -3.0]);
        assert_eq!(result.cached_tokens, None);
    }

    #[test]
    fn score_only_frame_can_precede_the_sampled_token_and_terminal_usage() {
        let scores = LLMEngineOutput {
            token_ids: vec![],
            finish_reason: None,
            completion_usage: None,
            ..complete()
        };
        let terminal = LLMEngineOutput {
            top_logprobs: None,
            ..complete()
        };
        let mut collector = VllmScoreCollector::new(vec![20, 10], 3);
        collector.observe(&scores).unwrap();
        collector.observe(&terminal).unwrap();
        assert_eq!(collector.finish().unwrap().output_tokens, 1);
    }

    #[test]
    fn rejects_incomplete_duplicate_nonfinite_and_unrequested_score_entries() {
        let original = complete();
        let mut cases = Vec::new();
        let mut missing = original.clone();
        missing.top_logprobs.as_mut().unwrap()[0].pop();
        cases.push(missing);
        let mut duplicate = original.clone();
        let row = &mut duplicate.top_logprobs.as_mut().unwrap()[0];
        row.push(row[0].clone());
        cases.push(duplicate);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.1] {
            let mut frame = original.clone();
            frame.top_logprobs.as_mut().unwrap()[0][0].logprob = invalid;
            cases.push(frame);
        }
        let mut unrelated = original.clone();
        unrelated.top_logprobs.as_mut().unwrap()[0][1].token_id = 100;
        cases.push(unrelated);
        let mut extra_position = original.clone();
        extra_position.top_logprobs.as_mut().unwrap().push(vec![]);
        cases.push(extra_position);
        for frame in cases {
            let mut collector = VllmScoreCollector::new(vec![20, 10], 3);
            let accepted = collector.observe(&frame).is_ok() && collector.finish().is_ok();
            assert!(!accepted, "accepted malformed scoring frame: {frame:?}");
        }
        assert!(VllmScoreCollector::new(vec![], 3).finish().is_err());
        assert!(VllmScoreCollector::new(vec![10, 10], 3).finish().is_err());
    }

    #[test]
    fn rejects_wrong_index_bad_terminals_extra_tokens_and_post_terminal_frames() {
        for reason in [
            FinishReason::Cancelled,
            FinishReason::Error("failed".into()),
            FinishReason::ContentFilter,
        ] {
            let frame = LLMEngineOutput {
                finish_reason: Some(reason),
                ..complete()
            };
            assert!(
                VllmScoreCollector::new(vec![20, 10], 3)
                    .observe(&frame)
                    .is_err()
            );
        }
        let wrong_index = LLMEngineOutput {
            index: Some(1),
            ..complete()
        };
        assert!(
            VllmScoreCollector::new(vec![20, 10], 3)
                .observe(&wrong_index)
                .is_err()
        );
        let extra_tokens = LLMEngineOutput {
            token_ids: vec![99, 100],
            ..complete()
        };
        assert!(
            VllmScoreCollector::new(vec![20, 10], 3)
                .observe(&extra_tokens)
                .is_err()
        );
        let mut collector = VllmScoreCollector::new(vec![20, 10], 3);
        collector.observe(&complete()).unwrap();
        assert!(collector.observe(&complete()).is_err());
        assert!(collector.finish().is_err());
        let no_terminal = LLMEngineOutput {
            finish_reason: None,
            ..complete()
        };
        let mut collector = VllmScoreCollector::new(vec![20, 10], 3);
        collector.observe(&no_terminal).unwrap();
        assert!(collector.finish().is_err());
    }

    #[test]
    fn rejects_missing_inconsistent_and_out_of_range_usage() {
        let mut missing = complete();
        missing.completion_usage = None;
        assert!(
            VllmScoreCollector::new(vec![20, 10], 3)
                .observe(&missing)
                .is_err()
        );
        for (field, value) in [
            ("prompt_tokens", 2),
            ("completion_tokens", 0),
            ("total_tokens", 9),
        ] {
            let mut value_frame = serde_json::to_value(complete()).unwrap();
            value_frame["completion_usage"][field] = json!(value);
            let frame = output(value_frame);
            assert!(
                VllmScoreCollector::new(vec![20, 10], 3)
                    .observe(&frame)
                    .is_err(),
                "{field}"
            );
        }
        let mut frame = serde_json::to_value(complete()).unwrap();
        frame["completion_usage"]["prompt_tokens_details"]["cached_tokens"] = json!(4);
        assert!(
            VllmScoreCollector::new(vec![20, 10], 3)
                .observe(&output(frame))
                .is_err()
        );

        let prefill = output(json!({
            "token_ids": [], "index": 0,
            "completion_usage": {
                "prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 1}
            }
        }));
        let mut collector = VllmScoreCollector::new(vec![20, 10], 3);
        collector.observe(&prefill).unwrap();
        assert!(collector.observe(&complete()).is_err());
    }
}
