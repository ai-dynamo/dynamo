// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Map, Value, json};

#[derive(Debug)]
pub(crate) struct EngineResponse {
    pub(crate) new_token_ids: Vec<Vec<u32>>,
    pub(crate) is_final: bool,
    pub(crate) finish_reasons: Option<Vec<Option<String>>>,
    pub(crate) stop_reasons: Option<Vec<Option<String>>>,
}

#[derive(Debug, Default)]
struct ChoiceState {
    token_ids: Vec<u32>,
    emitted: usize,
    finish_reason: Option<String>,
    stop_reason: Option<String>,
}

#[derive(Debug)]
pub(crate) struct OwnedResponseState {
    prompt_tokens: usize,
    choices: Vec<ChoiceState>,
}

impl OwnedResponseState {
    pub(crate) fn new(prompt_tokens: usize, num_choices: usize) -> Self {
        Self {
            prompt_tokens,
            choices: (0..num_choices).map(|_| ChoiceState::default()).collect(),
        }
    }

    pub(crate) fn apply(&mut self, response: EngineResponse) -> Vec<Value> {
        for (index, new_tokens) in response.new_token_ids.into_iter().enumerate() {
            let Some(choice) = self.choices.get_mut(index) else {
                break;
            };
            choice.token_ids.extend(new_tokens);
        }

        Self::update_reasons(
            &mut self.choices,
            response.finish_reasons,
            |choice, reason| {
                choice.finish_reason = reason;
            },
        );
        Self::update_reasons(
            &mut self.choices,
            response.stop_reasons,
            |choice, reason| {
                choice.stop_reason = reason;
            },
        );

        let completion_tokens = self
            .choices
            .iter()
            .map(|choice| choice.token_ids.len())
            .sum::<usize>();

        self.choices
            .iter_mut()
            .enumerate()
            .map(|(index, choice)| {
                let mut frame = Map::new();
                frame.insert(
                    "token_ids".to_string(),
                    json!(choice.token_ids[choice.emitted..]),
                );
                frame.insert("index".to_string(), json!(index));
                choice.emitted = choice.token_ids.len();

                let terminal = choice.finish_reason.is_some() || response.is_final;
                if terminal {
                    frame.insert(
                        "finish_reason".to_string(),
                        json!(choice.finish_reason.as_deref().unwrap_or("unknown")),
                    );
                }
                if let Some(stop_reason) = choice.stop_reason.as_deref() {
                    frame.insert("stop_reason".to_string(), json!(stop_reason));
                }
                if terminal {
                    frame.insert(
                        "completion_usage".to_string(),
                        json!({
                            "prompt_tokens": self.prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": self.prompt_tokens + completion_tokens,
                            "prompt_tokens_details": null
                        }),
                    );
                }
                Value::Object(frame)
            })
            .collect()
    }

    fn update_reasons(
        choices: &mut [ChoiceState],
        reasons: Option<Vec<Option<String>>>,
        update: impl Fn(&mut ChoiceState, Option<String>),
    ) {
        let Some(reasons) = reasons else {
            return;
        };
        for (choice, reason) in choices.iter_mut().zip(reasons) {
            if reason.is_some() {
                update(choice, reason);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EngineResponse, OwnedResponseState};
    use serde_json::json;

    fn assert_send_sync_static<T: Send + Sync + 'static>() {}

    #[test]
    fn owned_response_state_is_send_sync_static() {
        assert_send_sync_static::<OwnedResponseState>();
    }

    #[test]
    fn builds_delta_frames_with_python_semantic_parity() {
        let mut state = OwnedResponseState::new(3, 2);

        let first = state.apply(EngineResponse {
            new_token_ids: vec![vec![10], vec![20]],
            is_final: false,
            finish_reasons: None,
            stop_reasons: None,
        });
        assert_eq!(
            first,
            vec![
                json!({"token_ids": [10], "index": 0}),
                json!({"token_ids": [20], "index": 1}),
            ]
        );

        let final_frames = state.apply(EngineResponse {
            new_token_ids: vec![vec![11, 12], vec![21]],
            is_final: true,
            finish_reasons: Some(vec![Some("stop".to_string()), Some("length".to_string())]),
            stop_reasons: Some(vec![None, Some("eos".to_string())]),
        });
        assert_eq!(
            final_frames,
            vec![
                json!({
                    "token_ids": [11, 12],
                    "index": 0,
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 5,
                        "total_tokens": 8,
                        "prompt_tokens_details": null
                    }
                }),
                json!({
                    "token_ids": [21],
                    "index": 1,
                    "finish_reason": "length",
                    "stop_reason": "eos",
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 5,
                        "total_tokens": 8,
                        "prompt_tokens_details": null
                    }
                }),
            ]
        );
    }

    #[test]
    fn final_response_without_finish_reason_uses_unknown() {
        let mut state = OwnedResponseState::new(1, 1);

        let frames = state.apply(EngineResponse {
            new_token_ids: vec![vec![42]],
            is_final: true,
            finish_reasons: None,
            stop_reasons: None,
        });

        assert_eq!(frames[0]["finish_reason"], "unknown");
        assert_eq!(frames[0]["completion_usage"]["completion_tokens"], 1);
    }

    #[test]
    fn extra_engine_choices_are_ignored_like_the_python_path() {
        let mut state = OwnedResponseState::new(0, 1);

        let frames = state.apply(EngineResponse {
            new_token_ids: vec![vec![1], vec![999]],
            is_final: false,
            finish_reasons: None,
            stop_reasons: None,
        });

        assert_eq!(frames, vec![json!({"token_ids": [1], "index": 0})]);
    }
}
