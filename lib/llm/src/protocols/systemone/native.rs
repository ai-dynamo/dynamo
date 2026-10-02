// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use serde_json::Value;

use super::SystemOneError;

/// Build the exact native SGLang score envelope used by `/v1/score`.
pub fn build_native_score_request(
    prompt_token_ids: &[u32],
    label_token_ids: &[u32],
    cache_salt: &str,
) -> Result<Value, SystemOneError> {
    if prompt_token_ids.is_empty() {
        return Err(candidate_error("prompt token IDs cannot be empty"));
    }
    if label_token_ids.is_empty() {
        return Err(candidate_error("label token IDs cannot be empty"));
    }
    if label_token_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len()
        != label_token_ids.len()
    {
        return Err(candidate_error("label token IDs must be unique"));
    }
    if cache_salt.is_empty() {
        return Err(candidate_error("cache salt cannot be empty"));
    }

    Ok(serde_json::json!({
        "input_ids": prompt_token_ids,
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "n": 1
        },
        "return_logprob": true,
        "return_text_in_logprobs": false,
        "token_ids_logprob": label_token_ids,
        "cache_salt": cache_salt,
        "stream": true
    }))
}

/// Parse exact requested-token logprobs from one terminal native SGLang frame.
pub fn parse_candidate_scores(
    response: &Value,
    expected_token_ids: &[u32],
) -> Result<Vec<f64>, SystemOneError> {
    if expected_token_ids.is_empty() {
        return Err(candidate_error("expected token IDs cannot be empty"));
    }
    let unique: HashSet<_> = expected_token_ids.iter().copied().collect();
    if unique.len() != expected_token_ids.len() {
        return Err(candidate_error("expected token IDs must be unique"));
    }

    let meta = response
        .get("meta_info")
        .and_then(Value::as_object)
        .ok_or_else(|| candidate_error("missing meta_info object"))?;
    if response
        .get("output_ids")
        .is_some_and(|ids| ids.as_array().is_none_or(|ids| !ids.is_empty()))
        || meta
            .get("completion_tokens")
            .is_some_and(|count| count.as_u64() != Some(0))
    {
        return Err(candidate_error(
            "prefill-only scoring must not generate tokens",
        ));
    }
    let finish = meta
        .get("finish_reason")
        .and_then(Value::as_object)
        .ok_or_else(|| candidate_error("response is not terminal"))?;
    match finish.get("type").and_then(Value::as_str) {
        Some("length" | "stop") => {}
        Some("abort") => return Err(candidate_error("backend aborted candidate scoring")),
        Some(other) => {
            return Err(candidate_error(format!(
                "unsupported backend finish reason {other:?}"
            )));
        }
        None => return Err(candidate_error("backend finish reason has no type")),
    }

    let rows = meta
        .get("output_token_ids_logprobs")
        .and_then(Value::as_array)
        .ok_or_else(|| candidate_error("missing output_token_ids_logprobs"))?;
    if rows.len() != 1 {
        return Err(candidate_error(format!(
            "expected one scored position, received {}",
            rows.len()
        )));
    }
    let entries = rows[0]
        .as_array()
        .ok_or_else(|| candidate_error("candidate row must be an array"))?;
    if entries.len() != expected_token_ids.len() {
        return Err(candidate_error(format!(
            "expected {} candidate entries, received {}",
            expected_token_ids.len(),
            entries.len()
        )));
    }

    entries
        .iter()
        .zip(expected_token_ids.iter().copied())
        .enumerate()
        .map(|(index, (entry, expected_id))| {
            let tuple = entry
                .as_array()
                .filter(|tuple| tuple.len() == 3)
                .ok_or_else(|| candidate_error(format!("candidate {index} must be a 3-tuple")))?;
            let logprob = tuple[0].as_f64().ok_or_else(|| {
                candidate_error(format!("candidate {index} logprob must be finite"))
            })?;
            if !logprob.is_finite() {
                return Err(candidate_error(format!(
                    "candidate {index} logprob must be finite"
                )));
            }
            let token_id = tuple[1]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    candidate_error(format!("candidate {index} token ID must be a u32"))
                })?;
            if token_id != expected_id {
                return Err(candidate_error(format!(
                    "candidate {index} returned token ID {token_id}, expected {expected_id}"
                )));
            }
            if !tuple[2].is_null() && !tuple[2].is_string() {
                return Err(candidate_error(format!(
                    "candidate {index} token text must be null or a string"
                )));
            }
            Ok(logprob)
        })
        .collect()
}

fn candidate_error(message: impl Into<String>) -> SystemOneError {
    SystemOneError::CandidateScores(message.into())
}
