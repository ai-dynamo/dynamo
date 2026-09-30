// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::systemone::{
    SystemOneAnswer, SystemOneRequest, SystemOneResponse, SystemOneUsage, answer_from_logprobs,
    build_native_score_request, parse_candidate_scores, render_question_prompt,
};
use indexmap::IndexMap;
use serde_json::json;

#[test]
fn parses_type_safe_request_and_ignores_unknown_top_level_fields() {
    let request: SystemOneRequest = serde_json::from_value(json!({
        "model": "jev-latest",
        "state": {"ticket": "payment failed"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Choose a team",
                "criteria": {"billing": null, "technical": "Integration failures"}
            },
            "severity": {"type": "score", "criteria": ["low", "high"]},
            "urgent": {"type": "noul", "instructions": "This needs an answer today"}
        },
        "future_top_level_field": {"ignored": true}
    }))
    .unwrap();

    request.validate().unwrap();
    assert_eq!(request.questions.len(), 3);
}

#[test]
fn rejects_unknown_nested_fields_and_invalid_criteria() {
    assert!(
        serde_json::from_value::<SystemOneRequest>(json!({
            "model": "test-model",
            "state": "x",
            "questions": {"q": {"type": "choice", "criteria": {"a": null}, "extra": true}}
        }))
        .is_err()
    );

    let duplicate_options: SystemOneRequest = serde_json::from_value(json!({
        "model": "test-model",
        "state": "x",
        "questions": {"q": {
            "type": "choice",
            "criteria": {" Billing ": null, "billing": null}
        }}
    }))
    .unwrap();
    assert!(duplicate_options.validate().is_err());

    let bare_noul: SystemOneRequest = serde_json::from_value(json!({
        "model": "test-model",
        "state": null,
        "questions": {"q": {"type": "noul"}}
    }))
    .unwrap();
    assert!(bare_noul.validate().is_err());
}

#[test]
fn rejects_decisions_only_controls_when_non_null() {
    for field in [
        "temperature",
        "prompt_format_version",
        "return_prompt_token_ids",
    ] {
        let mut body = json!({
            "model": "test-model",
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "yes?"}}
        });
        body.as_object_mut()
            .unwrap()
            .insert(field.to_string(), json!(1));
        let request: SystemOneRequest = serde_json::from_value(body).unwrap();
        assert!(request.validate().unwrap_err().to_string().contains(field));
    }

    let request: SystemOneRequest = serde_json::from_value(json!({
        "model": "test-model",
        "state": "x",
        "questions": {"q": {"type": "noul", "instructions": "yes?"}},
        "temperature": null,
        "prompt_format_version": null,
        "return_prompt_token_ids": null
    }))
    .unwrap();
    request.validate().unwrap();
}

#[test]
fn builds_type_safe_answers_from_full_vocabulary_logprobs() {
    let request: SystemOneRequest = serde_json::from_value(json!({
        "model": "test-model",
        "state": "x",
        "questions": {
            "route": {"type": "choice", "criteria": {"billing": null, "technical": null}},
            "severity": {"type": "score", "criteria": ["low", "medium", "high"]},
            "urgent": {"type": "noul", "instructions": "urgent?"}
        }
    }))
    .unwrap();

    let choice = answer_from_logprobs(&request.questions["route"], &[-2.0, -1.0]).unwrap();
    let SystemOneAnswer::Choice(choice) = choice else {
        panic!("expected choice")
    };
    assert_eq!(choice.choice, "technical");
    assert!(
        (choice.probabilities["billing"] + choice.probabilities["technical"] - 1.0).abs() < 1e-12
    );
    assert!(choice.confidence > 0.0 && choice.confidence < 1.0);

    let score = answer_from_logprobs(&request.questions["severity"], &[-3.0, -2.0, -1.0]).unwrap();
    let SystemOneAnswer::Score(score) = score else {
        panic!("expected score")
    };
    assert!(score.score > 1.0 && score.score < 2.0);
    assert_eq!(score.legend["2"], json!("high"));

    let noul = answer_from_logprobs(&request.questions["urgent"], &[-0.2, -3.0]).unwrap();
    let SystemOneAnswer::Noul(noul) = noul else {
        panic!("expected noul")
    };
    assert!(noul.noul > 0.8);
    assert!(noul.x_label_mass > 0.0 && noul.x_label_mass <= 1.0);
}

#[test]
fn preserves_caller_order_for_ties_and_serializes_the_typesafe_shape() {
    let request: SystemOneRequest = serde_json::from_value(json!({
        "model": "test-model",
        "state": "x",
        "questions": {
            "route": {"type": "choice", "criteria": {"first": null, "second": null}}
        }
    }))
    .unwrap();

    let answer = answer_from_logprobs(&request.questions["route"], &[-1.0, -1.0]).unwrap();
    let SystemOneAnswer::Choice(choice) = &answer else {
        panic!("expected choice")
    };
    assert_eq!(choice.choice, "first");

    let response = SystemOneResponse {
        model: "Qwen/Qwen3.8-27B".to_string(),
        answers: IndexMap::from([("route".to_string(), answer)]),
        usage: SystemOneUsage {
            input_tokens: 12,
            output_tokens: 0,
        },
    };
    assert_eq!(
        serde_json::to_value(response).unwrap(),
        json!({
            "model": "Qwen/Qwen3.8-27B",
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "first",
                    "probabilities": {"first": 0.5, "second": 0.5},
                    "confidence": 0.0,
                    "x_label_mass": 2.0 * (-1.0_f64).exp()
                }
            },
            "usage": {"input_tokens": 12, "output_tokens": 0}
        })
    );
}

#[test]
fn rejects_missing_or_non_finite_logprobs() {
    let request: SystemOneRequest = serde_json::from_value(json!({
        "model": "test-model",
        "state": "x",
        "questions": {"q": {"type": "noul", "instructions": "yes?"}}
    }))
    .unwrap();

    assert!(answer_from_logprobs(&request.questions["q"], &[-1.0]).is_err());
    assert!(answer_from_logprobs(&request.questions["q"], &[f64::NAN, -1.0]).is_err());
}

#[test]
fn parses_exact_native_candidate_scores_in_requested_order() {
    let response = json!({
        "output_ids": [42],
        "meta_info": {
            "finish_reason": {"type": "length"},
            "output_token_ids_logprobs": [[
                [-0.2, 17, null],
                [-1.3, 4, "token"]
            ]]
        }
    });

    assert_eq!(
        parse_candidate_scores(&response, &[17, 4]).unwrap(),
        vec![-0.2, -1.3]
    );
}

#[test]
fn rejects_partial_reordered_or_aborted_native_candidate_scores() {
    for response in [
        json!({"meta_info": {"finish_reason": null, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {"type": "abort"}, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {"type": "length"}, "output_token_ids_logprobs": [[[-0.2, 4, null], [-1.3, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {"type": "length"}, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {"type": "error"}, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
        json!({"meta_info": {"finish_reason": {}, "output_token_ids_logprobs": [[[-0.2, 17, null]]]}}),
    ] {
        assert!(parse_candidate_scores(&response, &[17, 4]).is_err());
    }
}

#[test]
fn renders_sglang_prompt_format_version_one_in_request_order() {
    let request: SystemOneRequest = serde_json::from_value(json!({
        "model": "test-model",
        "state": {"ticket": "payment failed"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Choose a team",
                "criteria": {"billing": null, "technical": "Integration failures"}
            },
            "severity": {"type": "score", "criteria": ["low", "high"]},
            "urgent": {
                "type": "noul",
                "criteria": {"true": "today", "false": "later"}
            }
        }
    }))
    .unwrap();

    let choice = render_question_prompt(&request.state, &request.questions["route"]).unwrap();
    assert_eq!(choice.labels, ["A", "B"]);
    assert_eq!(
        choice.content,
        "{\"ticket\":\"payment failed\"}\n\nQuestion: Choose a team\nA: billing\nB: technical - Integration failures\nAnswer with the letter of one option only."
    );

    let score = render_question_prompt(&request.state, &request.questions["severity"]).unwrap();
    assert_eq!(score.labels, ["0", "1"]);
    assert!(
        score
            .content
            .ends_with("0: low\n1: high\nAnswer with the number of one level only.")
    );

    let noul = render_question_prompt(&request.state, &request.questions["urgent"]).unwrap();
    assert_eq!(noul.labels, ["yes", "no"]);
    assert!(
        noul.content.ends_with(
            "Is the following true?\nyes: today\nno: later\nAnswer with yes or no only."
        )
    );
}

#[test]
fn builds_zero_decode_native_sglang_score_request() {
    let request = build_native_score_request(&[1, 2, 3], &[17, 4], "request-salt").unwrap();
    assert_eq!(request["input_ids"], json!([1, 2, 3]));
    assert_eq!(
        request["sampling_params"],
        json!({
            "max_new_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "n": 1
        })
    );
    assert_eq!(request["token_ids_logprob"], json!([17, 4]));
    assert_eq!(request["return_logprob"], true);
    assert_eq!(request["return_text_in_logprobs"], false);
    assert_eq!(request["cache_salt"], "request-salt");
    assert_eq!(request["stream"], true);

    assert!(build_native_score_request(&[], &[17], "salt").is_err());
    assert!(build_native_score_request(&[1], &[17, 17], "salt").is_err());
    assert!(build_native_score_request(&[1], &[17], "").is_err());
}

#[test]
fn renders_all_two_letter_choice_labels_above_twenty_six_options() {
    let criteria = (0..27)
        .map(|index| (format!("option-{index}"), json!(null)))
        .collect::<serde_json::Map<_, _>>();
    let request: SystemOneRequest = serde_json::from_value(json!({
        "model": "test-model",
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": criteria}}
    }))
    .unwrap();

    let rendered = render_question_prompt(&request.state, &request.questions["q"]).unwrap();
    assert_eq!(rendered.labels.first().map(String::as_str), Some("AA"));
    assert_eq!(rendered.labels.get(25).map(String::as_str), Some("AZ"));
    assert_eq!(rendered.labels.last().map(String::as_str), Some("BA"));
}

#[test]
fn renders_two_letter_choice_labels_at_the_supported_limit() {
    let criteria = (0..255)
        .map(|index| (format!("option-{index}"), json!(null)))
        .collect::<serde_json::Map<_, _>>();
    let request: SystemOneRequest = serde_json::from_value(json!({
        "model": "test-model",
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": criteria}}
    }))
    .unwrap();

    request.validate().unwrap();
    let rendered = render_question_prompt(&request.state, &request.questions["q"]).unwrap();
    assert_eq!(rendered.labels.len(), 255);
    assert_eq!(rendered.labels.last().map(String::as_str), Some("JU"));
}
