// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
#[serial]
async fn required_nemotron_delays_guided_json_until_force_reasoning_ends() {
    temp_env::async_with_vars(ENV, async {
        let expected = ExpectedResponse {
            reasoning: Some(load_nemotron_expected_reasoning().unwrap()),
            content: ContentExpectation::Empty,
            calls: vec![ExpectedCall {
                name: "get_weather",
                arguments: json!({"location": "San Francisco"}),
            }],
            finish_reason: "tool_calls",
        };
        let requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, true),
            load_nemotron_required_trace().unwrap(),
            chat_body(
                "required",
                vec![weather_tool()],
                false,
                "enable_thinking",
                true,
            ),
            expected,
        )
        .await;
        assert_json_guidance(&requests, &[weather_tool()]);
        assert_reasoning_metadata(&requests, false, &json!({"enable_thinking": true}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_force_nonempty_content_only_falls_back_for_reasoning_only_response() {
    temp_env::async_with_vars(ENV, async {
        for force_nonempty_content in [false, true] {
            let requests = run_streaming_and_unary(
                model_card("nemotron_v3", "nemotron_nano", false, true),
                scripted_text(&["terminal thought", "</thi", "nk>"]),
                chat_body_with_template_kwargs(
                    "auto",
                    vec![weather_tool()],
                    false,
                    json!({"force_nonempty_content": force_nonempty_content}),
                ),
                ExpectedResponse {
                    reasoning: (!force_nonempty_content).then(|| "terminal thought".to_string()),
                    content: if force_nonempty_content {
                        ContentExpectation::Exact("terminal thought")
                    } else {
                        ContentExpectation::Empty
                    },
                    calls: Vec::new(),
                    finish_reason: "stop",
                },
            )
            .await;
            assert_no_guidance(&requests);
            assert_reasoning_metadata(
                &requests,
                false,
                &json!({"force_nonempty_content": force_nonempty_content}),
            );
        }
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_force_nonempty_content_keeps_reasoning_with_direct_answer() {
    temp_env::async_with_vars(ENV, async {
        let requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, true),
            scripted_text(&["I need to calculate.", "</thi", "nk>", "The answer is 42."]),
            chat_body_with_template_kwargs(
                "auto",
                vec![weather_tool()],
                false,
                json!({"force_nonempty_content": true}),
            ),
            ExpectedResponse {
                reasoning: Some("I need to calculate.".to_string()),
                content: ContentExpectation::Exact("The answer is 42."),
                calls: Vec::new(),
                finish_reason: "stop",
            },
        )
        .await;
        assert_no_guidance(&requests);
        assert_reasoning_metadata(&requests, false, &json!({"force_nonempty_content": true}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_force_nonempty_content_keeps_reasoning_with_tool_call() {
    temp_env::async_with_vars(ENV, async {
        let requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, true),
            scripted_text(&[
                "I need weather data.",
                "</thi",
                "nk>",
                r#"[{"name":"get_weather","parameters":{"location":"Reykjavik"}}]"#,
            ]),
            chat_body_with_template_kwargs(
                "required",
                vec![weather_tool()],
                false,
                json!({"force_nonempty_content": true}),
            ),
            ExpectedResponse {
                reasoning: Some("I need weather data.".to_string()),
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Reykjavik"}),
                }],
                finish_reason: "tool_calls",
            },
        )
        .await;
        assert_json_guidance(&requests, &[weather_tool()]);
        assert_reasoning_metadata(&requests, false, &json!({"force_nonempty_content": true}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_guided_json_handles_reasoning_off_and_multiple_tools() {
    temp_env::async_with_vars(ENV, async {
        let script = scripted_text(&[
            r#"[{"name":"get_weather","parameters":{"location":"Tokyo"}},"#,
            r#"{"name":"get_time","parameters":{"timezone":"UTC"}}]"#,
        ]);
        let expected = ExpectedResponse {
            reasoning: None,
            content: ContentExpectation::Empty,
            calls: vec![
                ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Tokyo"}),
                },
                ExpectedCall {
                    name: "get_time",
                    arguments: json!({"timezone": "UTC"}),
                },
            ],
            finish_reason: "tool_calls",
        };
        let requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, true),
            script,
            chat_body(
                "required",
                vec![weather_tool(), time_tool()],
                true,
                "enable_thinking",
                false,
            ),
            expected,
        )
        .await;
        assert_json_guidance(&requests, &[weather_tool(), time_tool()]);
        assert_reasoning_metadata(&requests, true, &json!({"enable_thinking": false}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_guided_json_supports_a_second_force_reasoning_family() {
    temp_env::async_with_vars(ENV, async {
        let expected = ExpectedResponse {
            reasoning: Some("I need the weather tool before answering.".to_string()),
            content: ContentExpectation::Empty,
            calls: vec![ExpectedCall {
                name: "get_weather",
                arguments: json!({"location": "Paris"}),
            }],
            finish_reason: "tool_calls",
        };
        let requests = run_streaming_and_unary(
            model_card("deepseek_r1", "qwen3_coder", false, true),
            scripted_text(&[
                "I need the weather tool before answering.",
                "</think>",
                r#"[{"name":"get_weather","parameters":{"location":"Paris"}}]"#,
            ]),
            chat_body("required", vec![weather_tool()], false, "thinking", true),
            expected,
        )
        .await;
        assert_json_guidance(&requests, &[weather_tool()]);
        assert_reasoning_metadata(&requests, false, &json!({"thinking": true}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_guided_json_supports_non_force_reasoning() {
    temp_env::async_with_vars(ENV, async {
        let requests = run_streaming_and_unary(
            model_card("qwen3", "hermes", false, true),
            scripted_text(&[
                "<think>I need the weather tool.",
                "</think>",
                r#"[{"name":"get_weather","parameters":{"location":"Prague"}}]"#,
            ]),
            chat_body(
                "required",
                vec![weather_tool()],
                false,
                "enable_thinking",
                true,
            ),
            ExpectedResponse {
                reasoning: Some("I need the weather tool.".to_string()),
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Prague"}),
                }],
                finish_reason: "tool_calls",
            },
        )
        .await;
        assert_json_guidance(&requests, &[weather_tool()]);
        assert_reasoning_metadata(&requests, false, &json!({"enable_thinking": true}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_guided_json_supports_minimax_append_reasoning() {
    temp_env::async_with_vars(ENV, async {
        let requests = run_streaming_and_unary(
            model_card("minimax_append_think", "minimax_m2", false, true),
            scripted_text(&[
                "I need the weather tool.",
                "</think>",
                r#"[{"name":"get_weather","parameters":{"location":"Osaka"}}]"#,
            ]),
            chat_body("required", vec![weather_tool()], false, "thinking", true),
            ExpectedResponse {
                reasoning: Some("I need the weather tool.".to_string()),
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Osaka"}),
                }],
                finish_reason: "tool_calls",
            },
        )
        .await;
        assert_json_guidance(&requests, &[weather_tool()]);
        assert_reasoning_metadata(&requests, false, &json!({"thinking": true}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_prompt_injected_reasoning_covers_json_and_structural_guidance() {
    temp_env::async_with_vars(ENV, async {
        let json_requests = run_streaming_and_unary(
            model_card_with_prompt_injected_reasoning("qwen3", "hermes", false),
            scripted_text(&[
                "Reasoning from the prefilled opener.",
                "</think>",
                r#"[{"name":"get_weather","parameters":{"location":"Dublin"}}]"#,
            ]),
            chat_body(
                "required",
                vec![weather_tool()],
                false,
                "enable_thinking",
                true,
            ),
            ExpectedResponse {
                reasoning: Some("Reasoning from the prefilled opener.".to_string()),
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Dublin"}),
                }],
                finish_reason: "tool_calls",
            },
        )
        .await;
        assert_json_guidance(&json_requests, &[weather_tool()]);
        assert_reasoning_metadata(
            &json_requests,
            false,
            &json!({"enable_thinking": true}),
        );

        let structural_requests = run_streaming_and_unary(
            model_card_with_prompt_injected_reasoning("qwen3", "qwen3_coder", true),
            scripted_text(&[
                "Reasoning before a structurally guided call.",
                "</think>",
                "<tool_call>\n<function=get_weather>\n<parameter=location>\nVienna\n</parameter>\n</function>\n</tool_call>",
            ]),
            chat_body(
                "required",
                vec![strict_tool(weather_tool())],
                false,
                "enable_thinking",
                true,
            ),
            ExpectedResponse {
                reasoning: Some("Reasoning before a structurally guided call.".to_string()),
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Vienna"}),
                }],
                finish_reason: "tool_calls",
            },
        )
        .await;
        assert_structural_guidance(
            &structural_requests,
            &[strict_tool(weather_tool())],
            true,
            false,
            true,
        );
        assert_reasoning_metadata(
            &structural_requests,
            true,
            &json!({"enable_thinking": true}),
        );
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_glm_json_object_response_format_keeps_json_in_content() {
    temp_env::async_with_vars(ENV, async {
        let response = r#"{"city":"Paris","temperature_c":18}"#;
        // JSON guidance may emit a bare object from token zero. Even though
        // the GLM template prefilled `<think>`, that object is assistant
        // content and must not be consumed as reasoning.
        let requests = run_streaming_and_unary(
            model_card_with_prompt_injected_reasoning_capability("glm45", "glm47", false, false),
            scripted_text(&[r#"{"city":"Par"#, r#"is","temperature_c":18}"#]),
            chat_body_with_response_format(
                "auto",
                vec![weather_tool()],
                false,
                json!({}),
                json!({"type": "json_object"}),
            ),
            ExpectedResponse {
                reasoning: None,
                content: ContentExpectation::Exact(response),
                calls: Vec::new(),
                finish_reason: "stop",
            },
        )
        .await;
        assert_response_format_json_guidance(&requests, &json!({"type": "object"}));
        assert!(requests.iter().all(|request| request.extra_args.is_none()));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_force_reasoning_json_object_response_format_keeps_json_in_content() {
    temp_env::async_with_vars(ENV, async {
        let response = r#"{"city":"Lima","temperature_c":24}"#;
        let requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, false),
            scripted_text(&[r#"{"city":"Li"#, r#"ma","temperature_c":24}"#]),
            chat_body_with_response_format(
                "auto",
                vec![weather_tool()],
                false,
                json!({"enable_thinking": true}),
                json!({"type": "json_object"}),
            ),
            ExpectedResponse {
                reasoning: None,
                content: ContentExpectation::Exact(response),
                calls: Vec::new(),
                finish_reason: "stop",
            },
        )
        .await;
        assert_response_format_json_guidance(&requests, &json!({"type": "object"}));
        assert!(requests.iter().all(|request| request.extra_args.is_none()));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn none_glm_json_schema_response_format_keeps_json_in_content() {
    temp_env::async_with_vars(ENV, async {
        let response = r#"{"city":"Tokyo","temperature_c":22}"#;
        let schema = json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temperature_c": {"type": "integer"}
            },
            "required": ["city", "temperature_c"],
            "additionalProperties": false
        });
        let requests = run_streaming_and_unary(
            model_card_with_prompt_injected_reasoning_capability("glm45", "glm47", false, false),
            scripted_text(&[r#"{"city":"Tok"#, r#"yo","temperature_c":22}"#]),
            chat_body_with_response_format(
                "none",
                vec![weather_tool()],
                false,
                json!({}),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "weather_response",
                        "strict": true,
                        "schema": schema.clone()
                    }
                }),
            ),
            ExpectedResponse {
                reasoning: None,
                content: ContentExpectation::Exact(response),
                calls: Vec::new(),
                finish_reason: "stop",
            },
        )
        .await;
        assert_response_format_json_guidance(&requests, &schema);
        assert!(requests.iter().all(|request| request.extra_args.is_none()));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_glm_response_format_preserves_reasoning_with_capable_backend() {
    temp_env::async_with_vars(ENV, async {
        let response = r#"{"city":"Oslo","temperature_c":12}"#;
        let requests = run_streaming_and_unary(
            model_card_with_prompt_injected_reasoning("glm45", "glm47", false),
            scripted_text(&[
                "I should answer with structured JSON.",
                "</thi",
                "nk>",
                response,
            ]),
            chat_body_with_response_format(
                "auto",
                vec![weather_tool()],
                false,
                json!({}),
                json!({"type": "json_object"}),
            ),
            ExpectedResponse {
                reasoning: Some("I should answer with structured JSON.".to_string()),
                content: ContentExpectation::Exact(response),
                calls: Vec::new(),
                finish_reason: "stop",
            },
        )
        .await;
        assert_response_format_json_guidance(&requests, &json!({"type": "object"}));
        assert_reasoning_metadata(&requests, false, &json!({}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_response_format_defers_to_tool_guidance() {
    temp_env::async_with_vars(ENV, async {
        let requests = run_streaming_and_unary(
            model_card_with_prompt_injected_reasoning("glm45", "glm47", false),
            scripted_text(&[
                "I need the weather tool.",
                "</think>",
                r#"[{"name":"get_weather","parameters":{"location":"Berlin"}}]"#,
            ]),
            chat_body_with_response_format(
                "required",
                vec![weather_tool()],
                false,
                json!({}),
                json!({"type": "json_object"}),
            ),
            ExpectedResponse {
                reasoning: Some("I need the weather tool.".to_string()),
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Berlin"}),
                }],
                finish_reason: "tool_calls",
            },
        )
        .await;
        assert_json_guidance(&requests, &[weather_tool()]);
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_text_response_format_does_not_enable_guidance() {
    temp_env::async_with_vars(ENV, async {
        let requests = run_streaming_and_unary(
            model_card_with_prompt_injected_reasoning("glm45", "glm47", false),
            scripted_text(&[
                "I can answer directly.",
                "</think>",
                "The temperature is 18 C.",
            ]),
            chat_body_with_response_format(
                "auto",
                vec![weather_tool()],
                false,
                json!({}),
                json!({"type": "text"}),
            ),
            ExpectedResponse {
                reasoning: Some("I can answer directly.".to_string()),
                content: ContentExpectation::Exact("The temperature is 18 C."),
                calls: Vec::new(),
                finish_reason: "stop",
            },
        )
        .await;
        assert_no_guidance(&requests);
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_guided_json_fails_closed_without_backend_reasoning_capability() {
    temp_env::async_with_vars(ENV, async {
        let expected = ExpectedResponse {
            reasoning: None,
            content: ContentExpectation::Empty,
            calls: vec![ExpectedCall {
                name: "get_weather",
                arguments: json!({"location": "Rome"}),
            }],
            finish_reason: "tool_calls",
        };
        let requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, false),
            scripted_text(&[r#"[{"name":"get_weather","parameters":{"location":"Rome"}}]"#]),
            chat_body(
                "required",
                vec![weather_tool()],
                false,
                "enable_thinking",
                true,
            ),
            expected,
        )
        .await;
        assert_json_guidance(&requests, &[weather_tool()]);
        for request in requests {
            assert!(
                request.extra_args.is_none(),
                "an unverified backend must not receive native reasoner controls"
            );
        }
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_tool_calls_replay_real_vllm_and_sglang_qwen_streams() {
    temp_env::async_with_vars(ENV, async {
        let traces = [
            (
                "vllm/qwen3-0.6B/chat_completion_stream_8f33c28b-tool.json",
                json!({"location": "San Francisco, CA", "unit": "celsius"}),
            ),
            (
                "sglang/qwen3-0.6B/chat_completion_stream_c42ba578-tool.json",
                json!({"location": "Tokyo", "unit": "celsius"}),
            ),
        ];
        for (trace, arguments) in traces {
            let expected = ExpectedResponse {
                reasoning: Some(load_recorded_expected_text(trace, "reasoning_content").unwrap()),
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments,
                }],
                finish_reason: "tool_calls",
            };
            let requests = run_streaming_and_unary(
                model_card("qwen3", "hermes", false, true),
                load_recorded_parser_trace(trace).unwrap(),
                chat_body("auto", vec![weather_tool()], false, "enable_thinking", true),
                expected,
            )
            .await;
            assert_no_guidance(&requests);
            assert_reasoning_metadata(&requests, false, &json!({"enable_thinking": true}));
        }
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_harmony_replays_real_channel_token_stream() {
    temp_env::async_with_vars(ENV, async {
        let expected = ExpectedResponse {
            reasoning: Some(
                load_recorded_expected_text(
                    "vllm/gpt-oss-20b/chat_completion_stream_f0c86d72-tool.json",
                    "reasoning_content",
                )
                .unwrap(),
            ),
            content: ContentExpectation::Empty,
            calls: vec![ExpectedCall {
                name: "get_weather",
                arguments: json!({"location": "San Francisco, CA", "unit": "celsius"}),
            }],
            finish_reason: "tool_calls",
        };
        let requests = run_streaming_and_unary(
            model_card("gpt_oss", "harmony", false, true),
            load_recorded_parser_trace(
                "vllm/gpt-oss-20b/chat_completion_stream_f0c86d72-tool.json",
            )
            .unwrap(),
            chat_body("auto", vec![weather_tool()], false, "thinking", true),
            expected,
        )
        .await;
        for request in &requests {
            assert_eq!(
                request.output_options.skip_special_tokens,
                Some(false),
                "Harmony markers must survive worker decoding for the parsers"
            );
        }
        assert_no_guidance(&requests);
        assert_reasoning_metadata(&requests, false, &json!({"thinking": true}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_direct_answer_separates_reasoning_and_honors_reasoning_off() {
    temp_env::async_with_vars(ENV, async {
        let qwen_expected = ExpectedResponse {
            reasoning: Some(
                load_recorded_expected_text(
                    "vllm/qwen3-0.6B/chat_completion_stream_5627a4c6-no-tool.json",
                    "reasoning_content",
                )
                .unwrap(),
            ),
            content: ContentExpectation::Exact(
                "\n\nI'm here to help! If you have any questions about NYC, weather, or anything else, feel free to ask! 😊",
            ),
            calls: Vec::new(),
            finish_reason: "stop",
        };
        let qwen_requests = run_streaming_and_unary(
            model_card("qwen3", "hermes", false, true),
            load_recorded_parser_trace(
                "vllm/qwen3-0.6B/chat_completion_stream_5627a4c6-no-tool.json",
            )
            .unwrap(),
            chat_body("auto", vec![weather_tool()], false, "enable_thinking", true),
            qwen_expected,
        )
        .await;
        assert_no_guidance(&qwen_requests);
        assert_reasoning_metadata(
            &qwen_requests,
            false,
            &json!({"enable_thinking": true}),
        );

        let direct_expected = ExpectedResponse {
            reasoning: None,
            content: ContentExpectation::Exact("37 multiplied by 19 is 703."),
            calls: Vec::new(),
            finish_reason: "stop",
        };
        let direct_requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, true),
            scripted_text(&["37 multiplied by 19 is 703."]),
            chat_body(
                "auto",
                vec![weather_tool()],
                false,
                "enable_thinking",
                false,
            ),
            direct_expected,
        )
        .await;
        assert_no_guidance(&direct_requests);
        assert_reasoning_metadata(
            &direct_requests,
            true,
            &json!({"enable_thinking": false}),
        );
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_reasoning_off_still_parses_tool_call() {
    temp_env::async_with_vars(ENV, async {
        let requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, true),
            scripted_text(&[
                "<tool_call>\n<function=get_weather>\n<parameter=location>\nSeoul\n</parameter>\n</function>\n</tool_call>",
            ]),
            chat_body(
                "auto",
                vec![weather_tool()],
                false,
                "enable_thinking",
                false,
            ),
            ExpectedResponse {
                reasoning: None,
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Seoul"}),
                }],
                finish_reason: "tool_calls",
            },
        )
        .await;
        assert_no_guidance(&requests);
        assert_reasoning_metadata(
            &requests,
            true,
            &json!({"enable_thinking": false}),
        );
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_preserves_narration_before_tool_call() {
    temp_env::async_with_vars(ENV, async {
        let requests = run_streaming_and_unary(
            model_card("qwen3", "hermes", false, true),
            scripted_text(&[
                "<think>I need current weather.",
                "</think>",
                "\n\nI'll check it now.\n",
                r#"<tool_call>{"name":"get_weather","arguments":{"location":"Lisbon"}}</tool_call>"#,
            ]),
            chat_body("auto", vec![weather_tool()], false, "enable_thinking", true),
            ExpectedResponse {
                reasoning: Some("I need current weather.".to_string()),
                content: ContentExpectation::Exact("\n\nI'll check it now.\n"),
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Lisbon"}),
                }],
                finish_reason: "tool_calls",
            },
        )
        .await;
        assert_no_guidance(&requests);
        assert_reasoning_metadata(
            &requests,
            false,
            &json!({"enable_thinking": true}),
        );
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_structural_tags_allow_direct_answer() {
    temp_env::async_with_vars(ENV, async {
        let requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", true, true),
            scripted_text(&[
                "I can answer directly without a tool.",
                "</think>",
                "The answer is 42.",
            ]),
            chat_body(
                "auto",
                vec![strict_tool(weather_tool())],
                false,
                "enable_thinking",
                true,
            ),
            ExpectedResponse {
                reasoning: Some("I can answer directly without a tool.".to_string()),
                content: ContentExpectation::Exact("The answer is 42."),
                calls: Vec::new(),
                finish_reason: "stop",
            },
        )
        .await;
        assert_structural_guidance(
            &requests,
            &[strict_tool(weather_tool())],
            false,
            false,
            false,
        );
        assert_reasoning_metadata(&requests, false, &json!({"enable_thinking": true}));
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_structural_tags_parse_reasoning_and_parallel_calls() {
    temp_env::async_with_vars(ENV, async {
        let expected = ExpectedResponse {
            reasoning: Some(
                load_recorded_expected_text(
                    "vllm/deepseek-v4/chat_completion_stream_multi_tool.json",
                    "reasoning_content",
                )
                .unwrap(),
            ),
            content: ContentExpectation::Empty,
            calls: vec![
                ExpectedCall {
                    name: "get_current_weather",
                    arguments: json!({"location": "Beijing", "format": "celsius"}),
                },
                ExpectedCall {
                    name: "get_current_weather",
                    arguments: json!({"location": "Shanghai", "format": "celsius"}),
                },
            ],
            finish_reason: "tool_calls",
        };
        let requests = run_streaming_and_unary(
            model_card("deepseek_v4", "deepseek_v4", true, true),
            load_recorded_parser_trace("vllm/deepseek-v4/chat_completion_stream_multi_tool.json")
                .unwrap(),
            chat_body(
                "required",
                vec![strict_tool(deepseek_weather_tool())],
                true,
                "thinking",
                true,
            ),
            expected,
        )
        .await;
        assert_structural_guidance(
            &requests,
            &[strict_tool(deepseek_weather_tool())],
            true,
            true,
            false,
        );
        assert_reasoning_metadata(
            &requests,
            false,
            &json!({"thinking": true, "enable_thinking": true}),
        );
    })
    .await;
}

#[tokio::test]
#[serial]
async fn auto_structural_tags_require_reasoning_aware_backend_capability() {
    temp_env::async_with_vars(ENV, async {
        for capability in [false, true] {
            let expected = ExpectedResponse {
                reasoning: Some("I should check the weather.".to_string()),
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Berlin"}),
                }],
                finish_reason: "tool_calls",
            };
            let requests = run_streaming_and_unary(
                model_card("nemotron_v3", "nemotron_nano", true, capability),
                scripted_text(&[
                    "I should check the weather.",
                    "</think>",
                    "<tool_call>\n<function=get_weather>\n<parameter=location>\nBerlin\n</parameter>\n</function>\n</tool_call>",
                ]),
                chat_body(
                    "auto",
                    vec![strict_tool(weather_tool())],
                    false,
                    "enable_thinking",
                    true,
                ),
                expected,
            )
            .await;
            if capability {
                assert_structural_guidance(
                    &requests,
                    &[strict_tool(weather_tool())],
                    false,
                    false,
                    false,
                );
                assert_reasoning_metadata(
                    &requests,
                    false,
                    &json!({"enable_thinking": true}),
                );
            } else {
                assert_no_guidance(&requests);
                for request in requests {
                    assert!(request.extra_args.is_none());
                }
            }
        }
    })
    .await;
}

#[tokio::test]
#[serial]
async fn required_guided_json_preserves_length_and_hides_incomplete_json() {
    temp_env::async_with_vars(ENV, async {
        let body = chat_body(
            "required",
            vec![weather_tool()],
            false,
            "enable_thinking",
            false,
        );
        let complete = ExpectedResponse {
            reasoning: None,
            content: ContentExpectation::Empty,
            calls: vec![ExpectedCall {
                name: "get_weather",
                arguments: json!({"location": "Madrid"}),
            }],
            finish_reason: "length",
        };
        let complete_requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, true),
            scripted_text_with_finish(
                &[r#"[{"name":"get_weather","parameters":{"location":"Madrid"}}]"#],
                BackendFinishReason::Length,
            ),
            body.clone(),
            complete,
        )
        .await;
        assert_json_guidance(&complete_requests, &[weather_tool()]);
        assert_reasoning_metadata(&complete_requests, true, &json!({"enable_thinking": false}));

        let incomplete = ExpectedResponse {
            reasoning: None,
            content: ContentExpectation::Empty,
            calls: Vec::new(),
            finish_reason: "length",
        };
        let incomplete_requests = run_streaming_and_unary(
            model_card("nemotron_v3", "nemotron_nano", false, true),
            scripted_text_with_finish(
                &[r#"[{"name":"get_weather","parameters":{"location":"Mad"#],
                BackendFinishReason::Length,
            ),
            body,
            incomplete,
        )
        .await;
        assert_json_guidance(&incomplete_requests, &[weather_tool()]);
        assert_reasoning_metadata(
            &incomplete_requests,
            true,
            &json!({"enable_thinking": false}),
        );
    })
    .await;
}

#[tokio::test]
#[serial]
async fn parsed_tool_response_emits_request_end_trace_metadata() {
    temp_env::async_with_vars(ENV, async {
        ensure_request_trace().await;
        let mut traces = request_trace::subscribe();
        let svc = RawChatHarness::start(
            model_card("nemotron_v3", "nemotron_nano", false, true),
            [scripted_text(&[
                r#"[{"name":"get_weather","parameters":{"location":"Oslo"}}]"#,
            ])],
        )
        .await;
        let body = chat_body(
            "required",
            vec![weather_tool()],
            false,
            "enable_thinking",
            false,
        );
        let observed = post_chat(&svc, &body, true).await;
        assert_response(
            &observed,
            &ExpectedResponse {
                reasoning: None,
                content: ContentExpectation::Empty,
                calls: vec![ExpectedCall {
                    name: "get_weather",
                    arguments: json!({"location": "Oslo"}),
                }],
                finish_reason: "tool_calls",
            },
        );

        let record = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let record = traces.recv().await.expect("request trace bus closed");
                if record.event_type == RequestTraceEventType::RequestEnd {
                    break record;
                }
            }
        })
        .await
        .expect("request-end trace was not emitted");
        assert_eq!(
            record
                .agent_context
                .as_ref()
                .map(|context| context.session_id.as_str()),
            Some("cpu-tool-e2e-session")
        );
        let metrics = record.request.expect("request-end trace has no metrics");
        assert_eq!(metrics.model.as_deref(), Some(MODEL));
        let replay = metrics
            .replay
            .expect("request-end trace has no replay data");
        assert_eq!(replay.trace_block_size, 16);
        assert!(replay.input_length > 0);
        let finish = metrics
            .finish_reason_metadata
            .expect("request-end trace has no finish metadata");
        assert_eq!(finish.finish_reason, Some(OpenAIFinishReason::ToolCalls));
        assert_eq!(finish.backend_finish_reason.as_deref(), Some("stop"));
        assert_eq!(finish.tool_calls.len(), 1);
        assert_eq!(finish.tool_calls[0].name.as_deref(), Some("get_weather"));

        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}
