// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building an OpenEngine `GenerateRequest` from a Dynamo request.

use super::*;

#[test]
fn request_maps_sampling_stop_and_output_fields() {
    let request = request();
    let proto = build_generate_request(&request, "req-1", "served-model", None, AGG)
        .expect("build request");
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
fn omitted_max_tokens_without_context_length_is_rejected() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    let error = build_generate_request(&req, "req", "model", None, AGG)
        .expect_err("must require max_tokens");
    assert!(error.to_string().contains("max_tokens"));
}

#[test]
fn omitted_max_tokens_defaults_to_remaining_context() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // request() carries three prompt tokens ([11, 22, 33]); the default fills the
    // remaining context: max(1, context_length - prompt_len).
    let proto = build_generate_request(&req, "req", "model", limits(100), AGG)
        .expect("build with fallback");
    assert_eq!(proto.stopping.unwrap().max_tokens, Some(97));
}

#[test]
fn omitted_max_tokens_default_is_floored_at_one() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // Prompt already fills (or exceeds) the context: default must not underflow to 0.
    let proto =
        build_generate_request(&req, "req", "model", limits(2), AGG).expect("build with fallback");
    assert_eq!(proto.stopping.unwrap().max_tokens, Some(1));
}

#[test]
fn top_k_all_tokens_is_left_unset() {
    let mut req = request();
    req.sampling_options.top_k = Some(-1);
    let proto = build_generate_request(&req, "req", "model", None, AGG).expect("build");
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
    let proto = build_generate_request(&req, "req", "model", None, AGG).expect("build");
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
        &req,
        "req",
        "model",
        Some(ModelLimits {
            context_length: Some(4096),
            max_output_tokens: Some(256),
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

    let error = build_generate_request(&req, "req", "model", limits(window), AGG)
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

    build_generate_request(&req, "req", "model", limits(window), PREFILL)
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

    let proto = build_generate_request(&req, "req", "model", limits(window), AGG)
        .expect("a minimum that fits is not a conflict");
    let stopping = proto.stopping.expect("stopping options");
    assert_eq!(stopping.min_tokens, Some(8));
    assert_eq!(stopping.max_tokens, Some(64));
}

const IMAGE: &str = "data:image/jpeg;base64,aW1hZ2UtYQ==";

fn build_image_request(req: &PreprocessedRequest) -> Result<pb::GenerateRequest, DynamoError> {
    build_generate_request(req, "req", "model", multimodal_limits(4096), AGG)
}

#[test]
fn images_are_forwarded_as_media_in_request_order() {
    let mut req = with_images(request(), &[IMAGE, "https://example.com/cat.jpg"]);
    // Routing data that the frontend attaches to image requests.
    req.mm_routing_info = Some(
        serde_json::from_value(json!({
            "routing_token_ids": vec![7; 40],
            "block_mm_infos": [],
            "expanded_prompt_len": 40,
        }))
        .expect("routing info"),
    );
    let proto = build_image_request(&req).expect("image request must build");
    let media: Vec<_> = proto
        .media
        .into_iter()
        .map(|item| (item.modality, item.source, item.uuid))
        .collect();
    assert_eq!(
        media,
        [
            (
                pb::Modality::Image as i32,
                Some(pb::media_item::Source::DataUri(IMAGE.to_string())),
                String::new(),
            ),
            (
                pb::Modality::Image as i32,
                Some(pb::media_item::Source::Url(
                    "https://example.com/cat.jpg".to_string()
                )),
                String::new(),
            ),
        ]
    );
    // The engine expands the image placeholders itself.
    match proto.input.expect("input") {
        pb::generate_request::Input::TokenIds(tokens) => assert_eq!(tokens.ids, [11, 22, 33]),
        other => panic!("expected token IDs input, got {other:?}"),
    }
}

#[test]
fn images_need_an_engine_that_advertises_multimodal_support() {
    let req = with_images(request(), &[IMAGE]);
    for limits in [None, limits(4096)] {
        let error =
            build_generate_request(&req, "req", "model", limits, AGG).expect_err("must reject");
        assert_eq!(
            error.public_message(),
            Some("the TensorRT-LLM engine does not advertise multimodal support")
        );
    }
    let text = build_generate_request(&request(), "req", "model", None, AGG)
        .expect("text request must build");
    assert!(text.media.is_empty());
}

#[test]
fn images_are_refused_in_disaggregated_modes() {
    let req = with_images(request(), &[IMAGE]);
    for mode in [PREFILL, DisaggregationMode::Decode] {
        let error = build_generate_request(&req, "req", "model", multimodal_limits(4096), mode)
            .expect_err("must reject");
        assert_eq!(
            error.public_message(),
            Some("the TensorRT-LLM sidecar supports multimodal requests only in aggregated mode")
        );
    }
}

#[test]
fn omitted_max_tokens_for_an_image_is_an_upper_bound() {
    let mut req = with_images(request(), &[IMAGE]);
    req.stop_conditions.max_tokens = None;
    // Sized from the three unexpanded prompt tokens; TensorRT-LLM clamps it to
    // what its expanded prompt leaves of the context.
    let proto = build_generate_request(&req, "req", "model", multimodal_limits(100), AGG)
        .expect("build with fallback");
    assert_eq!(proto.stopping.expect("stopping").max_tokens, Some(97));
}

#[test]
fn unsupported_image_inputs_carry_a_public_message() {
    let refused = |req: PreprocessedRequest| build_image_request(&req).expect_err("must reject");
    let with_image = |mutate: fn(&mut PreprocessedRequest)| {
        let mut req = with_images(request(), &[IMAGE]);
        mutate(&mut req);
        refused(req)
    };
    for (error, public) in [
        (
            refused(with_images(request(), &["ftp://example.com/cat.jpg"])),
            "TensorRT-LLM image input must use an http://, https://, or data: URI",
        ),
        (
            with_image(|r| {
                r.multi_modal_data = Some(HashMap::from([(
                    "image_url".to_string(),
                    vec![MultimodalData::UuidOnly("cached".to_string())],
                )]))
            }),
            "the TensorRT-LLM sidecar cannot resolve UUID-only media",
        ),
        (
            with_image(|r| r.mm_processor_kwargs = Some(json!({"do_resize": false}))),
            "mm_processor_kwargs is not supported by the TensorRT-LLM sidecar",
        ),
        (
            with_image(|r| r.media_io_kwargs = Some(json!({"image": {"max_width": 64}}))),
            "media_io_kwargs.image is not supported by the TensorRT-LLM sidecar",
        ),
    ] {
        assert_eq!(error.public_message(), Some(public));
    }
}

#[test]
fn other_media_and_encoder_results_are_rejected() {
    let mut video = request();
    video.multi_modal_data = Some(HashMap::from([(
        "video_url".to_string(),
        vec![MultimodalData::RawUrl(
            "data:video/mp4;base64,AA==".to_string(),
        )],
    )]));
    let error = build_image_request(&video).expect_err("must reject");
    assert!(
        error
            .to_string()
            .contains("`video_url` input is not supported")
    );
    assert_rejected(|r| r.encoder_result = Some(json!({})), "encoder results");
}

#[test]
fn image_processing_options_are_rejected() {
    let with_options = |mm_processor_kwargs, media_io_kwargs| {
        let mut req = with_images(request(), &[IMAGE]);
        req.mm_processor_kwargs = mm_processor_kwargs;
        req.media_io_kwargs = media_io_kwargs;
        build_image_request(&req)
    };
    // The engine cannot apply processor or image decoder options, so the
    // sidecar must not drop them without notice.
    let error = with_options(None, Some(json!("max_width=64"))).expect_err("must reject");
    assert!(error.to_string().contains("media_io_kwargs.image"));
    // Null, `{}`, and options for other media set no image option.
    with_options(
        Some(json!({})),
        Some(json!({"image": {}, "video": {"num_frames": 8}})),
    )
    .expect("no image options must build");
    with_options(Some(serde_json::Value::Null), None).expect("null options must build");

    // Without images there is nothing for the options to change.
    let mut text = request();
    text.mm_processor_kwargs = Some(json!({"do_resize": false}));
    build_generate_request(&text, "req", "model", None, AGG).expect("text request must build");
}

#[test]
fn multimodal_cache_uuids_are_rejected() {
    let with_uuids = |mut req: PreprocessedRequest, uuids: [Option<&str>; 2]| {
        req.multi_modal_uuids = Some(HashMap::from([(
            "image_url".to_string(),
            uuids.map(|uuid| uuid.map(str::to_string)).to_vec(),
        )]));
        build_image_request(&req)
    };
    let images = || with_images(request(), &[IMAGE, IMAGE]);
    for req in [images(), request()] {
        let error = with_uuids(req, [None, Some("cat")]).expect_err("must reject");
        assert_eq!(
            error.public_message(),
            Some("multimodal cache UUIDs are not supported by the TensorRT-LLM sidecar")
        );
    }
    with_uuids(images(), [None, Some("")]).expect("empty UUID entries must build");
}
