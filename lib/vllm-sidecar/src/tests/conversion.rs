// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{
    BackendError, ErrorType, MultimodalData, MultimodalDataMap, PreprocessedRequest, RoutingHints,
};

use super::request;
use crate::proto as pb;
use crate::request::build_generate_request;
use crate::wire::{
    disagg_json_to_kv_session, json_to_prost_struct, kv_session_to_disagg_json,
    prost_struct_to_json, validate_generate_response,
};

fn request_with_media(map: MultimodalDataMap) -> PreprocessedRequest {
    let mut req = request(Some(16));
    req.multi_modal_data = Some(map);
    req
}

fn url_media(value: &str) -> MultimodalData {
    serde_json::from_value(serde_json::json!({ "Url": value })).expect("parse MultimodalData::Url")
}

fn decoded_media() -> MultimodalData {
    serde_json::from_value(serde_json::json!({
        "Decoded": {
            "nixl_metadata": "",
            "nixl_descriptor": { "addr": 0, "size": 0, "mem_type": "Dram", "device_id": 0 },
            "shape": [1, 1, 1],
            "dtype": "UINT8",
            "metadata": null,
        }
    }))
    .expect("synthesize MultimodalData::Decoded")
}

#[test]
fn build_media_passes_through_http_url_as_url_source() {
    let map = MultimodalDataMap::from([(
        "image_url".to_string(),
        vec![url_media("http://example.com/cat.png")],
    )]);
    let req = build_generate_request(&request_with_media(map), "req-1", false).unwrap();

    assert_eq!(req.media.len(), 1);
    let item = &req.media[0];
    assert_eq!(item.modality, pb::Modality::Image as i32);
    assert_eq!(
        item.source,
        Some(pb::media_item::Source::Url(
            "http://example.com/cat.png".to_string()
        ))
    );
}

#[test]
fn build_media_passes_through_data_uri_as_data_uri_source() {
    let data_uri = "data:image/png;base64,AAAA";
    let map = MultimodalDataMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::RawUrl(data_uri.to_string())],
    )]);
    let req = build_generate_request(&request_with_media(map), "req-2", false).unwrap();

    assert_eq!(req.media.len(), 1);
    assert_eq!(
        req.media[0].source,
        Some(pb::media_item::Source::DataUri(data_uri.to_string()))
    );
}

#[test]
fn build_media_maps_modality_from_map_key() {
    let map = MultimodalDataMap::from([
        (
            "image_url".to_string(),
            vec![MultimodalData::RawUrl("http://h/a.png".to_string())],
        ),
        (
            "video_url".to_string(),
            vec![MultimodalData::RawUrl("http://h/b.mp4".to_string())],
        ),
        (
            "audio_url".to_string(),
            vec![MultimodalData::RawUrl("http://h/c.wav".to_string())],
        ),
    ]);
    let req = build_generate_request(&request_with_media(map), "req-3", false).unwrap();

    let modalities: Vec<i32> = req.media.iter().map(|media| media.modality).collect();
    assert_eq!(
        modalities,
        vec![
            pb::Modality::Image as i32,
            pb::Modality::Video as i32,
            pb::Modality::Audio as i32,
        ]
    );
}

#[test]
fn build_media_is_empty_for_text_only_request() {
    let req = build_generate_request(&request(Some(8)), "req-4", false).unwrap();
    assert!(req.media.is_empty());
}

#[test]
fn build_media_rejects_decoded_rdma_descriptor() {
    let map = MultimodalDataMap::from([("image_url".to_string(), vec![decoded_media()])]);
    let err = build_generate_request(&request_with_media(map), "req-5", false).unwrap_err();

    assert_eq!(
        err.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    assert!(err.message().contains("media_decoder"));
}

#[test]
fn disagg_json_round_trips_through_kv_session() {
    let attrs = serde_json::json!({
        "remote_engine_id": "engine-7",
        "remote_port": 20097,
        "tp_size": 1,
        "do_remote_prefill": true,
        "remote_block_ids": [4, 5, 6],
    });
    let original = pb::KvSessionRef {
        session_id: "sess-1".to_string(),
        transfer_backend: "NixlConnector".to_string(),
        dp_rank: 3,
        attributes_struct: json_to_prost_struct(&attrs),
    };

    let json = kv_session_to_disagg_json(original.clone()).unwrap();
    let restored = disagg_json_to_kv_session(&json).unwrap();

    assert_eq!(restored.session_id, original.session_id);
    assert_eq!(restored.transfer_backend, original.transfer_backend);
    assert_eq!(restored.dp_rank, original.dp_rank);
    assert_eq!(
        prost_struct_to_json(restored.attributes_struct.as_ref().unwrap()),
        attrs
    );
}

#[test]
fn disagg_json_rejects_incomplete_or_malformed_handoffs() {
    for value in [
        serde_json::json!([]),
        serde_json::json!({}),
        serde_json::json!({
            "session_id": "session",
            "transfer_backend": "NixlConnector",
            "dp_rank": u64::from(u32::MAX) + 1,
            "attributes_struct": {"remote_port": 8000},
        }),
        serde_json::json!({
            "session_id": "session",
            "transfer_backend": "NixlConnector",
            "dp_rank": 0,
            "attributes_struct": {},
        }),
        serde_json::json!({
            "session_id": "session",
            "transfer_backend": "NixlConnector",
            "dp_rank": 0,
            "attributes_struct": [],
        }),
    ] {
        let error = disagg_json_to_kv_session(&value).unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }
}

fn generate_response(
    request_id: &str,
    event: Option<pb::generate_response::Event>,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: request_id.to_string(),
        event,
        usage: None,
    }
}

fn valid_kv_session() -> pb::KvSessionRef {
    pb::KvSessionRef {
        session_id: "session".to_string(),
        transfer_backend: "NixlConnector".to_string(),
        dp_rank: 0,
        attributes_struct: json_to_prost_struct(&serde_json::json!({
            "remote_port": 8000
        })),
    }
}

#[test]
fn generate_response_validation_fails_closed() {
    let finished = || {
        pb::generate_response::Event::Finished(pb::GenerationFinished {
            reason: pb::FinishReason::Stop as i32,
            message: String::new(),
        })
    };
    let prefill_ready = |session| {
        pb::generate_response::Event::PrefillReady(pb::PrefillReady {
            kv_session: session,
        })
    };

    let invalid = [
        validate_generate_response(generate_response("wrong", Some(finished())), "req", false),
        validate_generate_response(generate_response("req", None), "req", false),
        validate_generate_response(generate_response("req", Some(finished())), "req", true),
        validate_generate_response(
            generate_response("req", Some(prefill_ready(Some(valid_kv_session())))),
            "req",
            false,
        ),
        validate_generate_response(
            generate_response("req", Some(prefill_ready(None))),
            "req",
            true,
        ),
        validate_generate_response(
            generate_response(
                "req",
                Some(prefill_ready(Some(pb::KvSessionRef {
                    attributes_struct: None,
                    ..valid_kv_session()
                }))),
            ),
            "req",
            true,
        ),
        validate_generate_response(
            generate_response(
                "req",
                Some(pb::generate_response::Event::Finished(
                    pb::GenerationFinished {
                        reason: pb::FinishReason::Unspecified as i32,
                        message: String::new(),
                    },
                )),
            ),
            "req",
            false,
        ),
        validate_generate_response(
            generate_response(
                "req",
                Some(pb::generate_response::Event::Finished(
                    pb::GenerationFinished {
                        reason: 999,
                        message: String::new(),
                    },
                )),
            ),
            "req",
            false,
        ),
    ];

    for result in invalid {
        let error = result.err().expect("malformed response must fail");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::Unknown)
        );
    }
}

#[test]
fn build_request_forwards_hidden_stops_and_priority() {
    let mut request = request(Some(8));
    request.stop_conditions.stop_token_ids_hidden = Some(vec![17, 19]);
    request.routing = Some(RoutingHints {
        priority: Some(-3),
        ..Default::default()
    });

    let wire = build_generate_request(&request, "req-priority", false).unwrap();
    assert_eq!(wire.priority, Some(-3));
    let stop_ids: Vec<u32> = wire
        .stop
        .into_iter()
        .filter_map(|stop| match stop.condition {
            Some(pb::stop_condition::Condition::StopTokenId(id)) => Some(id),
            _ => None,
        })
        .collect();
    assert_eq!(stop_ids, vec![17, 19]);
}

#[test]
fn prefill_clamps_min_and_max_tokens_to_one() {
    let mut request = request(Some(8));
    request.stop_conditions.min_tokens = Some(7);

    let wire = build_generate_request(&request, "req-prefill", true).unwrap();
    let sampling = wire.sampling.unwrap();
    assert_eq!(sampling.max_tokens, Some(1));
    assert_eq!(sampling.min_tokens, Some(1));
}

#[test]
fn prefill_uses_prefill_rank_while_decode_uses_decode_rank() {
    let mut request = request(Some(8));
    request.routing = Some(RoutingHints {
        dp_rank: Some(3),
        prefill_dp_rank: Some(7),
        ..Default::default()
    });

    let prefill = build_generate_request(&request, "req-prefill-rank", true).unwrap();
    let decode = build_generate_request(&request, "req-decode-rank", false).unwrap();

    assert_eq!(prefill.data_parallel_rank, Some(7));
    assert_eq!(decode.data_parallel_rank, Some(3));
}

#[test]
fn build_request_rejects_visible_stop_ids() {
    let mut request = request(Some(8));
    request.stop_conditions.stop_token_ids_visible = Some(vec![17]);

    let error = build_generate_request(&request, "req-visible-stop", false).unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
}

#[test]
fn build_request_rejects_prompt_embeddings() {
    let mut request = request(Some(8));
    request.prompt_embeds = Some("opaque-embedding-handle".to_string());

    let error = build_generate_request(&request, "req-embeds", false).unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
}
