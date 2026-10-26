// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building an OpenEngine `GenerateRequest` from a Dynamo request.

use std::collections::HashMap;

use super::*;
use crate::client::{SupportedInputs, SupportedMedia};
use crate::test_fixtures::*;
use dynamo_backend_common::MultimodalItemRef;
use serde_json::json;

#[test]
fn request_maps_sampling_stop_and_output_fields() {
    let request = request();
    let proto =
        build_generate_request(request, "req-1", "served-model", None, AGG).expect("build request");
    assert_eq!(proto.request_id, "req-1");
    assert_eq!(proto.model, "served-model");
    match proto.input.as_ref().expect("input") {
        pb::generate_request::Input::TokenIds(tokens) => assert_eq!(tokens.ids, [11, 22, 33]),
        other => panic!("expected token IDs input, got {other:?}"),
    }

    assert_eq!(
        proto.extra.as_ref().unwrap().fields["detokenize"].kind,
        Some(prost_types::value::Kind::BoolValue(false)),
        "Dynamo consumes token IDs and must disable unused engine text"
    );

    let stopping = proto.stopping.as_ref().unwrap();
    assert_eq!(stopping.max_tokens, Some(16));
    assert_eq!(stopping.min_tokens, Some(1));
    assert_eq!(stopping.ignore_eos, Some(true));
    // Stop-string retention is never enabled from `include_stop_str_in_output`.
    assert_eq!(stopping.include_stop_in_output, None);
    let conditions: Vec<_> = stopping
        .conditions
        .iter()
        .map(|condition| condition.condition.clone().unwrap())
        .collect();
    assert!(
        conditions.contains(&pb::stop_condition::Condition::StopText("done".to_string())),
        "missing stop text: {conditions:?}"
    );
    assert!(
        conditions.contains(&pb::stop_condition::Condition::StopTokenId(2)),
        "missing stop token id: {conditions:?}"
    );

    let sampling = proto.sampling.as_ref().unwrap();
    assert_eq!(sampling.top_k, Some(4));
    assert_eq!(sampling.top_p, Some(f64::from(0.9_f32)));
    assert_eq!(sampling.min_p, Some(f64::from(0.1_f32)));
    assert_eq!(sampling.temperature, Some(f64::from(0.2_f32)));
    assert_eq!(sampling.seed, Some(123));
    assert_eq!(sampling.repetition_penalty, Some(f64::from(1.1_f32)));
    assert_eq!(sampling.num_sequences, Some(1));

    let response = proto.response.as_ref().unwrap();
    assert_eq!(response.return_output_logprobs, Some(true));
    match response
        .output_candidates
        .as_ref()
        .unwrap()
        .selection
        .as_ref()
        .unwrap()
    {
        pb::candidate_token_selection::Selection::TopN(n) => assert_eq!(*n, 1),
        other => panic!("expected top_n selection, got {other:?}"),
    }

    match proto.guided.as_ref().unwrap().guide.as_ref().unwrap() {
        pb::guided_decoding::Guide::JsonSchema(guide) => assert!(guide.contains("object")),
        other => panic!("expected JSON schema guide, got {other:?}"),
    }
}

#[test]
fn text_input_model_receives_the_formatted_audio_prompt() {
    let mut req = request();
    req.token_ids = Arc::new(Vec::new());
    req.multi_modal_data = Some(HashMap::from([(
        "audio_url".to_string(),
        vec![MultimodalData::RawUrl(
            "https://example.com/audio.wav".to_string(),
        )],
    )]));
    let task_prompt = "<|startoftranscript|><|fr|><|translate|><|notimestamps|>";
    req.extra_args = Some(json!({"formatted_prompt": task_prompt}));
    let mut model_limits = all_media_limits(4096).expect("limits");
    model_limits.supported_inputs = Some(SupportedInputs {
        text: true,
        token_ids: false,
    });

    let proto = build_generate_request(req, "req", "model", Some(model_limits), AGG)
        .expect("text-input audio request");
    assert_eq!(
        proto.input,
        Some(pb::generate_request::Input::Prompt(task_prompt.to_string()))
    );
}

#[test]
fn text_input_model_accepts_an_empty_formatted_prompt() {
    let mut req = request();
    req.token_ids = Arc::new(Vec::new());
    req.extra_args = Some(json!({"formatted_prompt": ""}));
    let mut model_limits = limits(4096).expect("limits");
    model_limits.supported_inputs = Some(SupportedInputs {
        text: true,
        token_ids: false,
    });

    let proto = build_generate_request(req, "req", "model", Some(model_limits), AGG)
        .expect("an empty prompt lets the engine select its model default");
    assert_eq!(
        proto.input,
        Some(pb::generate_request::Input::Prompt(String::new()))
    );
}

#[test]
fn text_input_model_requires_a_formatted_prompt() {
    let req = request();
    let mut model_limits = limits(4096).expect("limits");
    model_limits.supported_inputs = Some(SupportedInputs {
        text: true,
        token_ids: false,
    });

    let error = build_generate_request(req, "req", "model", Some(model_limits), AGG)
        .expect_err("a token-only preprocessed request cannot serve a text-input model");
    assert!(error.to_string().contains("extra_args.formatted_prompt"));
}

#[test]
fn image_urls_data_uris_and_uuids_are_mapped_in_order() {
    let mut req = request();
    // Routing-only multimodal metadata has already served its purpose by the
    // time the request reaches a worker and must not block engine delivery.
    req.mm_routing_info = Some(Default::default());
    add_images(
        &mut req,
        [
            "https://example.com/first.png",
            "data:image/png;base64,aW1hZ2U=",
        ],
    );
    req.multi_modal_uuids = Some(HashMap::from([(
        "image_url".to_string(),
        vec![Some("first-image".to_string()), None],
    )]));
    let proto = build_generate_request(req, "req", "model", image_limits(4096), AGG)
        .expect("image request");
    assert_eq!(proto.media.len(), 2);
    assert_eq!(proto.media[0].modality, pb::Modality::Image as i32);
    assert_eq!(
        proto.media[0].source,
        Some(pb::media_item::Source::Url(
            "https://example.com/first.png".to_string()
        ))
    );
    assert_eq!(proto.media[0].uuid, "first-image");
    assert_eq!(
        proto.media[1].source,
        Some(pb::media_item::Source::DataUri(
            "data:image/png;base64,aW1hZ2U=".to_string()
        ))
    );
    assert!(proto.media[1].uuid.is_empty());
}

#[test]
fn inline_media_conversion_preserves_the_payload_allocation() {
    for (key, mime) in [("image_url", "image/png"), ("video_url", "video/mp4")] {
        for parsed in [false, true] {
            let uri = format!("data:{mime};base64,{}", "AAAA".repeat(16384));
            let item = if parsed {
                MultimodalData::Url(uri.parse().unwrap())
            } else {
                MultimodalData::RawUrl(uri)
            };
            let original = match &item {
                MultimodalData::Url(url) => url.as_str().as_ptr(),
                MultimodalData::RawUrl(url) => url.as_ptr(),
                _ => unreachable!(),
            };
            let mut req = request();
            req.multi_modal_data = Some(HashMap::from([(key.to_string(), vec![item])]));

            let proto = build_generate_request(req, "req", "model", image_video_limits(4096), AGG)
                .expect("inline media request");

            let Some(pb::media_item::Source::DataUri(value)) = &proto.media[0].source else {
                panic!("expected inline media");
            };
            assert_eq!(value.as_ptr(), original, "{key}, parsed={parsed}");
        }
    }
}

#[test]
fn mixed_media_follow_prompt_order() {
    let mut req = request();
    req.multi_modal_data = Some(HashMap::from([
        (
            "image_url".to_string(),
            vec![
                MultimodalData::RawUrl("https://example.com/first.png".to_string()),
                MultimodalData::RawUrl("data:image/png;base64,aW1hZ2U=".to_string()),
            ],
        ),
        (
            "video_url".to_string(),
            vec![
                MultimodalData::RawUrl("https://example.com/first.mp4".to_string()),
                MultimodalData::RawUrl("data:video/mp4;base64,dmlkZW8=".to_string()),
            ],
        ),
        (
            "audio_url".to_string(),
            vec![MultimodalData::RawUrl(
                "data:audio/wav;base64,YXVkaW8=".to_string(),
            )],
        ),
    ]));
    req.multi_modal_uuids = Some(HashMap::from([
        (
            "image_url".to_string(),
            vec![Some("image-0".to_string()), None],
        ),
        (
            "video_url".to_string(),
            vec![Some("video-0".to_string()), Some("video-1".to_string())],
        ),
        ("audio_url".to_string(), vec![Some("audio-0".to_string())]),
    ]));
    req.multi_modal_order = Some(vec![
        MultimodalItemRef {
            modality: "video_url".to_string(),
            index: 0,
        },
        MultimodalItemRef {
            modality: "image_url".to_string(),
            index: 0,
        },
        MultimodalItemRef {
            modality: "audio_url".to_string(),
            index: 0,
        },
        MultimodalItemRef {
            modality: "video_url".to_string(),
            index: 1,
        },
        MultimodalItemRef {
            modality: "image_url".to_string(),
            index: 1,
        },
    ]);

    let proto = build_generate_request(req, "req", "model", all_media_limits(4096), AGG)
        .expect("mixed media request");

    assert_eq!(
        proto
            .media
            .iter()
            .map(|item| item.modality)
            .collect::<Vec<_>>(),
        [
            pb::Modality::Video as i32,
            pb::Modality::Image as i32,
            pb::Modality::Audio as i32,
            pb::Modality::Video as i32,
            pb::Modality::Image as i32,
        ]
    );
    assert_eq!(
        proto
            .media
            .iter()
            .map(|item| item.uuid.as_str())
            .collect::<Vec<_>>(),
        ["video-0", "image-0", "audio-0", "video-1", ""]
    );
}

#[test]
fn mixed_media_without_a_complete_unique_order_is_rejected() {
    let mixed_media = || {
        HashMap::from([
            (
                "image_url".to_string(),
                vec![MultimodalData::RawUrl(
                    "https://example.com/image.png".to_string(),
                )],
            ),
            (
                "video_url".to_string(),
                vec![MultimodalData::RawUrl(
                    "https://example.com/video.mp4".to_string(),
                )],
            ),
        ])
    };

    let mut req = request();
    req.multi_modal_data = Some(mixed_media());
    let error = build_generate_request(req.clone(), "req", "model", image_video_limits(4096), AGG)
        .expect_err("mixed legacy request has lost its order");
    assert!(error.to_string().contains("requires multi_modal_order"));

    req.multi_modal_order = Some(vec![
        MultimodalItemRef {
            modality: "image_url".to_string(),
            index: 0,
        },
        MultimodalItemRef {
            modality: "image_url".to_string(),
            index: 0,
        },
    ]);
    let error = build_generate_request(req, "req", "model", image_video_limits(4096), AGG)
        .expect_err("duplicate order entry must fail");
    assert!(error.to_string().contains("repeats `image_url` item 0"));
}

#[test]
fn mm_routing_hashes_are_not_used_as_engine_cache_uuids() {
    let mut req = request();
    add_images(
        &mut req,
        [
            "https://example.com/first.png",
            "https://example.com/second.png",
        ],
    );
    req.extra_args = Some(json!({
        "mm_hashes": ["0123456789abcdef", "fedcba9876543210"]
    }));

    let proto = build_generate_request(req, "req", "model", image_limits(4096), AGG)
        .expect("image request");
    assert!(proto.media.iter().all(|item| item.uuid.is_empty()));
}

#[test]
fn media_requires_the_loaded_models_advertised_capability() {
    for (key, modality) in [
        ("image_url", "image"),
        ("video_url", "video"),
        ("audio_url", "audio"),
    ] {
        let mut req = request();
        req.multi_modal_data = Some(HashMap::from([(
            key.to_string(),
            vec![MultimodalData::RawUrl(
                "https://example.com/media".to_string(),
            )],
        )]));
        for model_limits in [None, limits(4096)] {
            let error = build_generate_request(req.clone(), "req", "model", model_limits, AGG)
                .expect_err("a text-only or unknown model must reject media")
                .to_string();
            assert!(error.contains(&format!("does not advertise {modality} support")));
            assert_eq!(error.contains("OpenCV (cv2)"), modality == "video");
        }
    }
}

#[test]
fn images_reject_unknown_or_overridden_engine_windows() {
    for engine_window in [None, Some(32768)] {
        for max_tokens in [None, Some(1)] {
            let mut req = request();
            add_images(&mut req, ["https://example.com/image.png"]);
            req.stop_conditions.max_tokens = max_tokens;
            let limits = Some(ModelLimits {
                context_length: Some(4096),
                engine_context_length: engine_window,
                supported_media: Some(SupportedMedia {
                    image: true,
                    video: true,
                    ..Default::default()
                }),
                ..Default::default()
            });
            let error = build_generate_request(req, "req", "model", limits, AGG)
                .expect_err("media expansion cannot honor a different window");
            assert!(error.to_string().contains("--context-length to match"));
        }
    }
}

#[test]
fn image_video_models_reject_audio() {
    let mut req = request();
    req.multi_modal_data = Some(HashMap::from([(
        "audio_url".to_string(),
        vec![MultimodalData::RawUrl(
            "https://example.com/audio.wav".to_string(),
        )],
    )]));
    let error = build_generate_request(req, "req", "model", image_video_limits(4096), AGG)
        .expect_err("a model without audio support must reject audio");
    assert!(
        error
            .to_string()
            .contains("does not advertise audio support"),
        "unexpected error: {error}"
    );
}

#[test]
fn unsupported_media_modalities_and_sources_are_rejected() {
    let mut req = request();
    req.multi_modal_data = Some(HashMap::from([(
        "future_media".to_string(),
        vec![MultimodalData::RawUrl(
            "https://example.com/media".to_string(),
        )],
    )]));
    let error = build_generate_request(req, "req", "model", all_media_limits(4096), AGG)
        .expect_err("unsupported modalities must be rejected");
    assert!(
        error.to_string().contains("future_media"),
        "unexpected error: {error}"
    );

    let mut req = request();
    add_images(&mut req, ["file:///tmp/image.png"]);
    let error = build_generate_request(req, "req", "model", image_limits(4096), AGG)
        .expect_err("the engine cannot fetch a local Dynamo path");
    assert!(error.to_string().contains("http, https, or data URI"));

    let mut req = request();
    req.multi_modal_data = Some(HashMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::UuidOnly("cached-image".to_string())],
    )]));
    let error = build_generate_request(req, "req", "model", image_limits(4096), AGG)
        .expect_err("OpenEngine cannot resolve a source-less UUID");
    assert!(
        error
            .to_string()
            .contains("requires a source for image input")
    );
}

#[test]
fn image_metadata_that_cannot_be_honored_is_rejected() {
    let mut req = request();
    add_images(&mut req, ["https://example.com/image.png"]);
    req.multi_modal_uuids = Some(HashMap::from([(
        "image_url".to_string(),
        vec![Some("one".to_string()), Some("two".to_string())],
    )]));
    let error = build_generate_request(req.clone(), "req", "model", image_limits(4096), AGG)
        .expect_err("UUIDs must align positionally");
    assert!(error.to_string().contains("UUID count"));

    req.multi_modal_uuids = None;
    req.mm_processor_kwargs = Some(serde_json::json!({"do_resize": false}));
    let error = build_generate_request(req.clone(), "req", "model", image_limits(4096), AGG)
        .expect_err("processor overrides cannot be dropped");
    assert!(error.to_string().contains("mm_processor_kwargs"));

    req.mm_processor_kwargs = Some(serde_json::json!({}));
    req.media_io_kwargs = Some(serde_json::json!({"image": {"mode": "RGB"}}));
    let error = build_generate_request(req, "req", "model", image_limits(4096), AGG)
        .expect_err("media I/O overrides cannot be dropped");
    assert!(error.to_string().contains("media_io_kwargs"));
}

#[test]
fn visible_stop_tokens_stop_in_engine_and_stay_in_output() {
    let mut req = request();
    req.stop_conditions.stop_token_ids = Some(vec![7]);
    req.stop_conditions.stop_token_ids_hidden = Some(vec![2, 200002]);
    req.stop_conditions.stop_token_ids_visible = Some(vec![200012, 7]);
    let proto = build_generate_request(req, "req", "model", None, AGG).expect("build");
    let stopping = proto.stopping.unwrap();
    assert_eq!(stopping.include_stop_in_output, Some(true));
    let stop_ids: Vec<_> = stopping
        .conditions
        .iter()
        .filter_map(|condition| match condition.condition {
            Some(pb::stop_condition::Condition::StopTokenId(id)) => Some(id),
            _ => None,
        })
        .collect();
    assert_eq!(stop_ids, [2, 7, 200002, 200012]);
}

#[test]
fn empty_visible_stop_tokens_leave_stop_trimming_on() {
    let mut req = request();
    req.stop_conditions.stop_token_ids_visible = Some(Vec::new());
    let proto = build_generate_request(req, "req", "model", None, AGG).expect("build");
    assert_eq!(proto.stopping.unwrap().include_stop_in_output, None);
}

#[test]
fn omitted_max_tokens_without_context_length_is_rejected() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    let error = build_generate_request(req, "req", "model", None, AGG)
        .expect_err("must require max_tokens");
    assert!(error.to_string().contains("max_tokens"));
}

#[test]
fn omitted_max_tokens_defaults_to_remaining_context() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // request() carries three prompt tokens ([11, 22, 33]); the default fills the
    // remaining context: max(1, context_length - prompt_len).
    let proto =
        build_generate_request(req, "req", "model", limits(100), AGG).expect("build with fallback");
    assert_eq!(proto.stopping.unwrap().max_tokens, Some(97));
}

#[test]
fn omitted_max_tokens_default_is_floored_at_one() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // Prompt already fills (or exceeds) the context: default must not underflow to 0.
    let proto =
        build_generate_request(req, "req", "model", limits(2), AGG).expect("build with fallback");
    assert_eq!(proto.stopping.unwrap().max_tokens, Some(1));
}

#[test]
fn top_k_all_tokens_is_left_unset() {
    let mut req = request();
    req.sampling_options.top_k = Some(-1);
    let proto = build_generate_request(req, "req", "model", None, AGG).expect("build");
    assert_eq!(proto.sampling.unwrap().top_k, None);
}

#[test]
fn unsupported_request_controls_are_rejected() {
    // Controls the OpenEngine contract can neither forward nor faithfully honor:
    // reject rather than fail open.
    assert_rejected(
        |r| r.sampling_options.include_stop_str_in_output = Some(true),
        "include_stop_str_in_output",
    );
    assert_rejected(
        |r| r.sampling_options.logit_bias = Some(std::collections::HashMap::from([(18, -10.0)])),
        "logit_bias",
    );
    assert_rejected(
        |r| r.stop_conditions.max_thinking_tokens = Some(32),
        "max_thinking_tokens",
    );
    assert_rejected(
        |r| {
            r.routing = Some(dynamo_backend_common::engine::RoutingHints {
                cache_namespace: Some("tenant-a".to_string()),
                ..Default::default()
            })
        },
        "cache namespace",
    );
    assert_rejected(
        |r| {
            r.routing = Some(dynamo_backend_common::engine::RoutingHints {
                priority: Some(5),
                ..Default::default()
            })
        },
        "priority",
    );
    // A negative top_k other than -1/0 (the "all tokens" sentinels) is invalid,
    // not a silent widening to "all tokens".
    assert_rejected(|r| r.sampling_options.top_k = Some(-5), "top_k must be");
    assert_rejected(
        |r| r.sampling_options.seed = Some(-1),
        "seed must be non-negative",
    );
}

#[test]
fn logprobs_zero_keeps_selected_without_alternatives() {
    let mut req = request();
    req.output_options.logprobs = Some(0);
    // The wire request must still ask TRT-LLM for one candidate so the selected
    // token's logprob is computed.
    let proto = build_generate_request(req.clone(), "req", "model", None, AGG).expect("build");
    match proto
        .response
        .unwrap()
        .output_candidates
        .unwrap()
        .selection
        .unwrap()
    {
        pb::candidate_token_selection::Selection::TopN(n) => assert_eq!(n, 1),
        other => panic!("expected top_n selection, got {other:?}"),
    }

    let mut state = ResponseState::new(&req, AGG);
    let delta = state
        .convert(token_response(vec![logprob_token(7, -0.1)]))
        .expect("convert")
        .expect("delta");
    assert_eq!(delta.log_probs.as_deref(), Some(&[-0.1_f64][..]));
    // logprobs=0 surfaces the selected-token logprob but no top alternatives.
    assert!(delta.top_logprobs.is_none());
}

/// The engine's own output cap can be smaller than the context window leaves
/// room for, and a request derived past it is rejected by the engine.
#[test]
fn the_derived_max_tokens_respects_the_engines_output_cap() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    let proto = build_generate_request(
        req,
        "req",
        "model",
        Some(ModelLimits {
            context_length: Some(4096),
            max_output_tokens: Some(256),
            supported_inputs: None,
            supported_media: Some(SupportedMedia::default()),
            ..Default::default()
        }),
        AGG,
    )
    .expect("build with an output cap");
    assert_eq!(
        proto.stopping.expect("stopping").max_tokens,
        Some(256),
        "the derived default must not exceed what the engine will generate"
    );
}

/// A `max_tokens` derived from the context window is clamped by the engine's
/// output cap and floored at 1, so it can land under an explicit `min_tokens`.
/// TensorRT-LLM does not cross-validate the pair, so a request carrying
/// `min_tokens > max_tokens` is one the engine resolves however it likes --
/// reject it here, naming both values.
#[test]
fn a_derived_budget_below_min_tokens_is_rejected() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    req.stop_conditions.min_tokens = Some(64);
    // A 40-token window against this request's prompt leaves fewer than 64.
    let prompt_len = req.token_ids.len() as u32;
    let window = prompt_len + 8;

    let error = build_generate_request(req, "req", "model", limits(window), AGG)
        .expect_err("a minimum above the remaining budget cannot be served");
    let message = error.to_string();
    assert!(
        message.contains("64") && message.contains('8'),
        "the error must name both the minimum and what is left: {message}"
    );
}

/// The prefill worker must not judge the minimum: it applies to the decode
/// worker's budget, and the two roles are paired without requiring equal
/// context lengths. A window too small here says nothing about the window the
/// decode worker will use.
#[test]
fn a_prefill_request_does_not_judge_a_minimum_against_its_own_window() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    req.stop_conditions.min_tokens = Some(64);
    let window = req.token_ids.len() as u32 + 8;

    build_generate_request(req, "req", "model", limits(window), PREFILL)
        .expect("only the decode worker's window governs the minimum");
}

/// The same request is served when the window leaves room for the minimum --
/// the guard must reject the contradiction, not every request that has one.
#[test]
fn a_minimum_within_the_derived_budget_is_served() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    req.stop_conditions.min_tokens = Some(8);
    let window = req.token_ids.len() as u32 + 64;

    let proto = build_generate_request(req, "req", "model", limits(window), AGG)
        .expect("a minimum that fits is not a conflict");
    let stopping = proto.stopping.expect("stopping options");
    assert_eq!(stopping.min_tokens, Some(8));
    assert_eq!(stopping.max_tokens, Some(64));
}

// Anonymous requests must not allocate native per-conversation cache state
// unless the rank-hint contract requires an ID. Only this worker's leg counts.
#[test]
fn conversation_identity_follows_session_or_transmitted_rank_hint() {
    use dynamo_backend_common::engine::RoutingHints;

    for (mode, allow_dp_rank, dp_rank, prefill_dp_rank, expected) in [
        (AGG, false, None, None, None),
        (AGG, false, Some(0), None, None),
        (AGG, true, None, None, None),
        (AGG, true, Some(0), None, Some("req")),
        (
            DisaggregationMode::Prefill,
            true,
            None,
            Some(1),
            Some("req"),
        ),
        (
            DisaggregationMode::Prefill,
            true,
            Some(1),
            None,
            Some("req"),
        ),
        (DisaggregationMode::Decode, true, None, Some(1), None),
        (DisaggregationMode::Decode, true, Some(1), None, Some("req")),
    ] {
        for session_id in [None, Some(" "), Some(" session ")] {
            let mut req = request();
            req.routing = Some(RoutingHints {
                dp_rank,
                prefill_dp_rank,
                ..Default::default()
            });
            req.agent_context = session_id.map(|session_id| {
                serde_json::from_value(json!({"session_id": session_id})).unwrap()
            });
            let proto =
                build_generate_request_with_routing(req, "req", "model", None, mode, allow_dp_rank)
                    .expect("build request");
            let expected = if session_id == Some(" session ") {
                Some("session")
            } else {
                expected
            };
            assert_eq!(
                proto
                    .extra
                    .as_ref()
                    .unwrap()
                    .fields
                    .get("conversation_id")
                    .and_then(|value| value.kind.as_ref()),
                expected
                    .map(|value| prost_types::value::Kind::StringValue(value.to_string()))
                    .as_ref(),
                "mode={mode:?}, targeting={allow_dp_rank}, rank={dp_rank:?}, prefill_rank={prefill_dp_rank:?}",
            );
        }
    }
}

#[test]
fn image_requests_preserve_rank_hint_conversation_identity() {
    use dynamo_backend_common::engine::RoutingHints;

    for mode in [AGG, DisaggregationMode::Prefill, DisaggregationMode::Decode] {
        let mut req = request();
        add_images(&mut req, ["https://example.com/image.png"]);
        req.routing = Some(RoutingHints {
            dp_rank: Some(1),
            ..Default::default()
        });
        let proto = build_generate_request_with_routing(
            req,
            "image-req",
            "model",
            image_limits(4096),
            mode,
            true,
        )
        .expect("image request with rank targeting");
        assert_eq!(proto.media.len(), 1);
        assert_eq!(
            proto.extra.unwrap().fields["conversation_id"].kind,
            Some(prost_types::value::Kind::StringValue("image-req".into())),
        );
    }
}

/// Applies `mutate` to a baseline request and asserts `build_generate_request`
/// rejects it with a message mentioning `expect`.
fn assert_rejected(mutate: impl FnOnce(&mut PreprocessedRequest), expect: &str) {
    let mut req = request();
    mutate(&mut req);
    let error = build_generate_request(req, "req", "model", None, AGG)
        .expect_err("request must be rejected");
    assert!(
        error.to_string().contains(expect),
        "error {error:?} should mention {expect:?}"
    );
}

/// The prefill worker marks its request `context_only` and caps generation at
/// the single token the context phase produces.
#[test]
fn prefill_request_is_marked_context_only() {
    let mut req = request();
    req.agent_context = Some(
        serde_json::from_value(json!({"session_id": " agentx-session "}))
            .expect("valid agent context"),
    );
    req.stop_conditions.max_tokens = Some(128);
    req.stop_conditions.min_tokens = Some(8);
    req.output_options.logprobs = Some(1);
    let proto = build_generate_request(req, "req", "model", None, DisaggregationMode::Prefill)
        .expect("build prefill request");

    assert!(
        is_context_only(&proto),
        "prefill must set extra.request_type"
    );
    assert_eq!(
        proto
            .extra
            .as_ref()
            .and_then(|extra| extra.fields.get("conversation_id"))
            .and_then(|value| value.kind.as_ref()),
        Some(&prost_types::value::Kind::StringValue(
            "agentx-session".to_string()
        )),
        "prefill must preserve both its request type and AgentX session"
    );
    assert_eq!(
        proto.extra.as_ref().unwrap().fields["detokenize"].kind,
        Some(prost_types::value::Kind::BoolValue(false)),
        "token-only output must coexist with the prefill session extensions"
    );
    let stopping = proto.stopping.expect("stopping options");
    assert_eq!(stopping.max_tokens, Some(1));
    // The minimum masks EOS rather than extending generation, so it must
    // survive: without it the context phase can sample EOS on its single token
    // and terminate with `Stop` instead of the `PrefillReady` handoff.
    assert_eq!(stopping.min_tokens, Some(8));
    // The prefill worker surfaces no tokens, but it must still compute
    // logprobs: the first generated token comes from the context phase and its
    // logprob only reaches the decode worker through the handoff.
    assert_eq!(
        proto
            .response
            .expect("response options")
            .return_output_logprobs,
        Some(true)
    );
    assert!(proto.kv.is_none(), "prefill carries no session to replay");
}

#[test]
fn mixed_media_is_preserved_across_prefill_and_decode_requests() {
    let mut req = request();
    req.multi_modal_data = Some(HashMap::from([
        (
            "image_url".to_string(),
            vec![MultimodalData::RawUrl(
                "https://example.com/image.png".to_string(),
            )],
        ),
        (
            "video_url".to_string(),
            vec![MultimodalData::RawUrl(
                "data:video/mp4;base64,dmlkZW8=".to_string(),
            )],
        ),
    ]));
    req.multi_modal_uuids = Some(HashMap::from([
        ("image_url".to_string(), vec![Some("image-id".to_string())]),
        ("video_url".to_string(), vec![Some("video-id".to_string())]),
    ]));
    req.multi_modal_order = Some(vec![
        MultimodalItemRef {
            modality: "video_url".to_string(),
            index: 0,
        },
        MultimodalItemRef {
            modality: "image_url".to_string(),
            index: 0,
        },
    ]);

    let prefill = build_generate_request(
        req.clone(),
        "req",
        "model",
        image_video_limits(4096),
        DisaggregationMode::Prefill,
    )
    .expect("build media prefill");
    assert!(is_context_only(&prefill));
    assert_eq!(prefill.stopping.as_ref().unwrap().max_tokens, Some(1));
    assert_eq!(prefill.media.len(), 2);
    assert_eq!(prefill.media[0].modality, pb::Modality::Video as i32);
    assert_eq!(prefill.media[0].uuid, "video-id");
    assert_eq!(prefill.media[1].modality, pb::Modality::Image as i32);
    assert_eq!(prefill.media[1].uuid, "image-id");

    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: crate::disagg::session_to_json(fake_session())
            .expect("encode handoff"),
        prompt_tokens_details: None,
    });
    let decode = build_generate_request(
        req,
        "req",
        "model",
        image_video_limits(4096),
        DisaggregationMode::Decode,
    )
    .expect("build media decode");
    assert!(!is_context_only(&decode));
    assert_eq!(decode.media, prefill.media);
    assert_eq!(
        decode.kv.and_then(|kv| kv.session),
        Some(fake_session()),
        "decode must carry both ordered media items and the KV handoff"
    );
}

/// Conditional disaggregation dispatches straight to a decode worker with no
/// handoff, expecting it to run the context phase itself. Rejecting that would
/// fail every bypassed request.
#[test]
fn decode_without_a_prefill_result_runs_the_whole_request() {
    let proto = build_generate_request(
        request(),
        "req",
        "model",
        limits(4096),
        DisaggregationMode::Decode,
    )
    .expect("a decode request without a handoff is the bypass path");
    assert!(
        proto
            .kv
            .as_ref()
            .and_then(|kv| kv.session.as_ref())
            .is_none(),
        "no session should be replayed when none was handed off"
    );
    assert!(
        !is_context_only(&proto),
        "a bypassed request must not be marked context_only"
    );
}

/// A handoff on a non-decode worker means the frontend routed the request to
/// the wrong role - fail loudly rather than silently prefill it again.
#[test]
fn prefill_result_on_a_non_decode_worker_is_rejected() {
    let mut req = request();
    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: json!({"session_id": "1"}),
        prompt_tokens_details: None,
    });
    for mode in [DisaggregationMode::Aggregated, DisaggregationMode::Prefill] {
        let error = build_generate_request(req.clone(), "req", "model", None, mode)
            .expect_err("handoff must be rejected");
        assert!(
            error
                .to_string()
                .contains("must be routed to a decode worker"),
            "unexpected error for {mode}: {error}"
        );
    }
}
