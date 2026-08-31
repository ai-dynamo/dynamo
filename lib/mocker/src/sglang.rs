// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic SGLang response metadata shared by Mocker transports.

use serde_json::{Map, Value, json};

pub const MAX_TOP_LOGPROBS: usize = 20;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LogprobOptions {
    return_logprob: bool,
    top_logprobs_num: usize,
    /// Prompt offset the caller wants input logprobs from. `None` is SGLang's
    /// `-1`, meaning it wants none.
    prompt_start: Option<usize>,
}

impl LogprobOptions {
    pub fn new(
        return_logprob: bool,
        top_logprobs_num: i64,
        logprob_start_len: i64,
    ) -> Result<Self, String> {
        if !(0..=MAX_TOP_LOGPROBS as i64).contains(&top_logprobs_num) {
            return Err(format!(
                "top_logprobs_num must be between 0 and {MAX_TOP_LOGPROBS}"
            ));
        }
        if logprob_start_len < -1 {
            return Err("logprob_start_len must be -1 or greater".to_string());
        }
        Ok(Self {
            return_logprob,
            top_logprobs_num: top_logprobs_num as usize,
            prompt_start: usize::try_from(logprob_start_len).ok(),
        })
    }
}

#[derive(Debug)]
pub struct ResponseMetadata {
    request_id: String,
    prompt_tokens: usize,
    /// The requested slice of the prompt, kept only when the caller wants prompt
    /// logprobs on the terminal response.
    prompt_logprob_tokens: Option<Vec<u32>>,
    logprob_options: LogprobOptions,
}

impl ResponseMetadata {
    pub fn new(
        request_id: impl Into<String>,
        prompt_tokens: &[u32],
        logprob_options: LogprobOptions,
    ) -> Self {
        let prompt_logprob_tokens = logprob_options
            .return_logprob
            .then_some(logprob_options.prompt_start)
            .flatten()
            .map(|start| prompt_tokens[start.min(prompt_tokens.len())..].to_vec());
        Self {
            request_id: request_id.into(),
            prompt_tokens: prompt_tokens.len(),
            prompt_logprob_tokens,
            logprob_options,
        }
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn response(
        &self,
        output_ids: &[u32],
        completion_tokens: usize,
        finish_reason: Option<Value>,
    ) -> Value {
        json!({
            "output_ids": output_ids,
            "meta_info": self.meta_info(output_ids, completion_tokens, finish_reason),
        })
    }

    pub fn meta_info(
        &self,
        output_ids: &[u32],
        completion_tokens: usize,
        finish_reason: Option<Value>,
    ) -> Map<String, Value> {
        let terminal = finish_reason.is_some();
        let mut meta_info = Map::from_iter([
            ("id".to_string(), Value::String(self.request_id.clone())),
            (
                "finish_reason".to_string(),
                finish_reason.unwrap_or(Value::Null),
            ),
            ("prompt_tokens".to_string(), Value::from(self.prompt_tokens)),
            (
                "completion_tokens".to_string(),
                Value::from(completion_tokens),
            ),
        ]);

        if self.logprob_options.return_logprob {
            meta_info.insert(
                "output_token_logprobs".to_string(),
                Value::Array(output_ids.iter().copied().map(logprob_entry).collect()),
            );
            if self.logprob_options.top_logprobs_num > 0 {
                meta_info.insert(
                    "output_top_logprobs".to_string(),
                    Value::Array(
                        output_ids
                            .iter()
                            .copied()
                            .map(|token| {
                                top_logprob_entries(token, self.logprob_options.top_logprobs_num)
                            })
                            .collect(),
                    ),
                );
            }
            if terminal {
                self.insert_prompt_logprobs(&mut meta_info);
            }
        }

        meta_info
    }

    fn insert_prompt_logprobs(&self, meta_info: &mut Map<String, Value>) {
        let Some(prompt_tokens) = self.prompt_logprob_tokens.as_deref() else {
            return;
        };
        let mut input_token_logprobs = Vec::with_capacity(prompt_tokens.len());
        let mut input_top_logprobs = Vec::with_capacity(prompt_tokens.len());
        if let Some((first, remaining)) = prompt_tokens.split_first() {
            // Native SGLang retains the first token ID but uses a null
            // logprob because no preceding token predicts it.
            input_token_logprobs.push(json!([null, first, null]));
            input_token_logprobs.extend(remaining.iter().copied().map(logprob_entry));
            if self.logprob_options.top_logprobs_num > 0 {
                input_top_logprobs.push(Value::Null);
                input_top_logprobs.extend(remaining.iter().copied().map(|token| {
                    top_logprob_entries(token, self.logprob_options.top_logprobs_num)
                }));
            }
        }
        meta_info.insert(
            "input_token_logprobs".to_string(),
            Value::Array(input_token_logprobs),
        );
        if self.logprob_options.top_logprobs_num > 0 {
            meta_info.insert(
                "input_top_logprobs".to_string(),
                Value::Array(input_top_logprobs),
            );
        }
    }
}

fn selected_logprob(token_id: u32) -> f64 {
    -0.1 * f64::from((token_id % 10) + 1)
}

fn logprob_entry(token_id: u32) -> Value {
    json!([
        selected_logprob(token_id),
        token_id,
        format!("<token:{token_id}>")
    ])
}

fn top_logprob_entries(token_id: u32, count: usize) -> Value {
    Value::Array(
        (0..count)
            .map(|offset| {
                let candidate = token_id.saturating_add(offset as u32);
                json!([
                    selected_logprob(candidate) - (offset as f64 * 0.01),
                    candidate,
                    format!("<token:{candidate}>")
                ])
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_logprob_controls() {
        assert!(LogprobOptions::new(false, -1, -1).is_err());
        assert!(LogprobOptions::new(false, 21, -1).is_err());
        assert!(LogprobOptions::new(false, 0, -2).is_err());
    }

    #[test]
    fn emits_incremental_native_metadata_and_logprobs() {
        let options = LogprobOptions::new(true, 2, 1).unwrap();
        let metadata = ResponseMetadata::new("request-1", &[10, 11, 12], options);
        let response = metadata.response(&[42], 3, Some(json!({"type": "length"})));

        assert_eq!(response["output_ids"], json!([42]));
        assert_eq!(response["meta_info"]["id"], "request-1");
        assert_eq!(response["meta_info"]["prompt_tokens"], 3);
        assert_eq!(response["meta_info"]["completion_tokens"], 3);
        assert_eq!(
            response["meta_info"]["finish_reason"],
            json!({"type": "length"})
        );
        let selected = &response["meta_info"]["output_token_logprobs"][0];
        assert!((selected[0].as_f64().unwrap() + 0.3).abs() < f64::EPSILON);
        assert_eq!(selected[1], 42);
        assert_eq!(selected[2], "<token:42>");
        assert_eq!(
            response["meta_info"]["output_top_logprobs"][0]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            response["meta_info"]["input_token_logprobs"][0],
            json!([null, 11, null])
        );
        assert_eq!(response["meta_info"]["input_token_logprobs"][1][1], 12);
        assert_eq!(response["meta_info"]["input_top_logprobs"][0], Value::Null);

        // Prompt logprobs belong to the terminal response only.
        let incremental = metadata.response(&[42], 1, None);
        assert!(
            incremental["meta_info"]
                .get("input_token_logprobs")
                .is_none()
        );
    }

    #[test]
    fn omits_logprob_metadata_by_default() {
        let metadata = ResponseMetadata::new("request-1", &[10], LogprobOptions::default());
        let response = metadata.response(&[42], 1, None);

        assert!(response["meta_info"].get("output_token_logprobs").is_none());
        assert_eq!(response["meta_info"]["finish_reason"], Value::Null);
    }
}
