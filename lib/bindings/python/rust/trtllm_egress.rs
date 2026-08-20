// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
