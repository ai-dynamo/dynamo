// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ClientCodec, Encoded, SglangFormat};
use axum::http::{HeaderMap, StatusCode};
use dynamo_llm::generate_format::*;
use serde_json::{Value, json};
use std::sync::Arc;

struct Wrapped;
impl ClientCodec for Wrapped {
    fn decode_request(&self, body: &[u8]) -> Result<Option<Value>, ClientError> {
        let value: Value =
            serde_json::from_slice(body).map_err(|error| ClientError::new(error.to_string()))?;
        Ok(Some(value["request"].clone()))
    }
    fn encode_response(
        &self,
        response: &Value,
    ) -> Result<Encoded, dynamo_runtime::error::DynamoError> {
        Ok(Encoded::Replace(json!({"response": response})))
    }
    fn error_body(&self, message: &str) -> Value {
        json!({"failure": message})
    }
}

#[test]
fn catalog_format_owns_client_shape_and_reuses_native_exchange() {
    let mut registry = Registry::default();
    registry.register(Arc::new(SglangFormat::native())).unwrap();
    registry
        .register(Arc::new(SglangFormat::with_codec("wrapped", Wrapped)))
        .unwrap();
    assert!(registry.register(Arc::new(SglangFormat::native())).is_err());
    assert!(registry.get("missing").is_err());
    let format = registry.get("wrapped").unwrap();
    assert_eq!(format.capability(), crate::CAPABILITY);
    assert_eq!(
        format.error_body(StatusCode::BAD_REQUEST, "bad"),
        json!({"failure": "bad"})
    );
    let opened = format
        .open(ClientRequest {
            body: br#"{"request":{"input_ids":[1],"sampling_params":{"max_new_tokens":2}}}"#,
            headers: &HeaderMap::new(),
            default_request_id: "public",
        })
        .unwrap();
    let mut bound = (opened.bind)(&crate::testkit::worker()).unwrap();
    assert_eq!(bound.request.token_ids.as_slice(), &[1]);
    for (ids, finish) in [
        (json!([2]), Value::Null),
        (json!([3]), json!({"type":"length"})),
    ] {
        let output = crate::inspect(
            json!({"output_ids": ids, "meta_info": {
                "id": "private", "finish_reason": finish, "custom_metadata": {"score": 7},
            }}),
            false,
        )
        .unwrap()
        .output;
        bound.exchange.chunk(output, false, &mut |_| {}).unwrap();
    }
    let response = bound.exchange.finish().unwrap().unwrap();
    assert_eq!(response["response"]["output_ids"], json!([2, 3]));
    assert_eq!(response["response"]["meta_info"]["id"], "public");
    assert_eq!(
        response["response"]["meta_info"]["custom_metadata"]["score"],
        7
    );
}

#[test]
fn codec_extends_native_field_rules_and_joins_structured_sequences() {
    use crate::response_merge::{FieldClass, PrefillFlags, RuleSet};
    struct Structured;
    impl ClientCodec for Structured {
        fn rules(&self) -> Arc<RuleSet> {
            Arc::new(crate::rules().extend(
                [("/meta_info/custom_scores".into(), FieldClass::Delta)],
                PrefillFlags::default(),
            ))
        }
        fn overlay_sequence(&self, path: &str, prefix: &Value, target: &mut Value) -> bool {
            if path != "/meta_info/routed_experts" {
                return crate::Native.overlay_sequence(path, prefix, target);
            }
            let (Some(prefix), Some(target)) =
                (prefix["data"].as_array(), target["data"].as_array_mut())
            else {
                return false;
            };
            target.splice(..prefix.len().min(target.len()), prefix.iter().cloned());
            true
        }
    }
    let format = SglangFormat::with_codec("structured", Structured);
    let opened = format
        .open(ClientRequest {
            body: br#"{"input_ids":[1],"return_routed_experts":true}"#,
            headers: &HeaderMap::from_iter([(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            )]),
            default_request_id: "public",
        })
        .unwrap();
    let mut bound = (opened.bind)(&crate::testkit::worker()).unwrap();
    assert!(
        bound
            .request
            .annotations
            .iter()
            .any(|a| a == dynamo_llm::protocols::response_merge::PREFILL_OUTPUT_ANNOTATION)
    );
    for (response, prefill) in [
        (
            json!({"meta_info":{"routed_experts":{"data":[1],"dtype":"u8"}}}),
            true,
        ),
        (
            json!({"output_ids":[2],"meta_info":{"custom_scores":[0.25]}}),
            false,
        ),
        (
            json!({"output_ids":[3],"meta_info":{"custom_scores":[0.5],"routed_experts":{"data":[0,2,3],"dtype":"u8"}}}),
            false,
        ),
    ] {
        bound
            .exchange
            .chunk(
                crate::inspect(response, prefill).unwrap().output,
                prefill,
                &mut |_| {},
            )
            .unwrap();
    }
    let result = bound.exchange.finish().unwrap().unwrap();
    assert_eq!(result["output_ids"], json!([2, 3]));
    assert_eq!(result["meta_info"]["custom_scores"], json!([0.25, 0.5]));
    assert_eq!(
        result["meta_info"]["routed_experts"],
        json!({"data":[1,2,3],"dtype":"u8"})
    );
}

fn bind_native(body: &Value, staged: bool) -> Result<Bound, ClientError> {
    bind_media(body, [staged, false, false])
}

fn bind_media(body: &Value, [images, videos, audios]: [bool; 3]) -> Result<Bound, ClientError> {
    let opened = SglangFormat::native().open(ClientRequest {
        body: &serde_json::to_vec(body).unwrap(),
        headers: &HeaderMap::from_iter([(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        )]),
        default_request_id: "public",
    })?;
    opened.validation?;
    (opened.bind)(&Worker {
        sglang_staged_images: images,
        sglang_staged_video: videos,
        sglang_staged_audio: audios,
        ..crate::testkit::worker()
    })
}

#[test]
fn staged_generate_projects_ordered_images_and_retains_token_ids_and_hashes() {
    let first = "data:image/png;base64,AQ==";
    let second = "data:image/jpeg;base64,Ag==";
    let ids = [11, 151655, 12, 151655, 13];
    let req = bind_native(
        &json!({
            "input_ids": ids, "image_data": [first, second], "mm_hashes": ["A", "B"],
            "sampling_params": {"max_new_tokens": 5}, "rid": "original"
        }),
        true,
    )
    .unwrap()
    .request;
    assert_eq!(req.token_ids.as_slice(), ids);
    assert_eq!(
        serde_json::to_value(req.multi_modal_data.as_ref().unwrap()).unwrap(),
        json!({"image_url": [{"Url": first}, {"Url": second}]})
    );
    let payload = &req.extra_args.as_ref().unwrap()[crate::PAYLOAD_KEY];
    assert!(payload.get("image_data").is_none());
    assert_eq!(payload["mm_hashes"], json!(["A", "B"]));
    assert_eq!(payload["sampling_params"]["max_new_tokens"], 5);
    // One image as rendered by moonshotai/Kimi-VL-A3B-Instruct.
    let kimi_media_ids = [163602, 4017, 163603, 163605, 163604];
    let one = bind_native(
        &json!({"input_ids":kimi_media_ids, "image_data":first}),
        true,
    )
    .unwrap();
    assert_eq!(one.request.token_ids.as_slice(), kimi_media_ids);
    assert_eq!(one.request.multi_modal_data.unwrap()["image_url"].len(), 1);
}

#[test]
fn staged_generate_bypasses_text_and_legacy_generate_preserves_opaque_media() {
    for media in [Value::Null, json!([])] {
        let req = bind_native(&json!({"input_ids": [1], "image_data": media}), true).unwrap();
        assert!(req.request.multi_modal_data.is_none());
    }
    let legacy = json!({
        "input_ids":[1], "image_data":[{"url":"/native/file.png"}],
        "audio_data":"native-audio", "images_config":{"detail":"high"}
    });
    let req = bind_native(&legacy, false).unwrap().request;
    assert!(req.multi_modal_data.is_none());
    let payload = &req.extra_args.as_ref().unwrap()[crate::PAYLOAD_KEY];
    for key in ["image_data", "audio_data", "images_config"] {
        assert_eq!(payload[key], legacy[key]);
    }
}

#[test]
fn staged_generate_rejects_unsupported_media_before_encoding() {
    let image = "data:image/png;base64,AQ==";
    for fields in [
        json!({"image_data": [[image]]}),
        json!({"image_data": 7}),
        json!({"image_data": "file:///native/file.png"}),
        json!({"image_data": image, "audio_data": ["sound"]}),
        json!({"video_data": "video"}),
        json!({"image_data": image, "images_config":{"width":1024}}),
        json!({"image_data": image, "mm_processor_kwargs":{"size":42}}),
    ] {
        let mut body = fields;
        body["input_ids"] = json!([1]);
        assert!(bind_native(&body, true).is_err(), "{body}");
    }
}

#[test]
fn staged_video_preserves_clip_order_and_interleaved_prompt() {
    let clip = "data:video/mp4;base64,AA==";
    let clip2 = "data:video/mp4;base64,Ag==";
    let image = "data:image/png;base64,AQ==";
    let body = json!({
        "input_ids": [7, 151656, 8, 151655, 9, 151656],
        "video_data": [clip, clip2], "image_data": image,
        "mm_hashes": ["a1","a2","a3"],
    });
    assert!(bind_media(&body, [true, false, false]).is_err());
    let req = bind_media(&body, [true, true, false]).unwrap().request;
    assert_eq!(req.token_ids.as_slice(), [7, 151656, 8, 151655, 9, 151656]);
    assert_eq!(
        serde_json::to_value(req.multi_modal_data.unwrap()).unwrap(),
        json!({
            "video_url":[{"Url":clip},{"Url":clip2}], "image_url":[{"Url":image}]
        })
    );
    let native = &req.extra_args.unwrap()[crate::PAYLOAD_KEY];
    assert_eq!(native["mm_hashes"], body["mm_hashes"]);
    assert!(native.get("video_data").is_none() && native.get("image_data").is_none());
    assert!(
        bind_media(
            &json!({"input_ids":[151656], "video_data":clip}),
            [true, true, false]
        )
        .is_ok()
    );
    assert!(
        bind_media(
            &json!({"input_ids":[151656], "video_data":clip, "audio_data":"x"}),
            [true, true, false]
        )
        .is_err()
    );
}

#[test]
fn staged_audio_uses_native_token_order_and_requires_advertised_modalities() {
    let first = "data:audio/wav;base64,AQ==";
    let second = "https://example.test/sound.wav";
    let image = "https://example.test/image.jpg";
    let ids = [41, 1, 50, 2, 41];
    let mut body = json!({"input_ids":ids, "audio_data":[first, second]});
    let req = bind_media(&body, [false, false, true]).unwrap().request;
    assert_eq!(req.token_ids.as_slice(), ids);
    assert!(
        req.extra_args.unwrap()[crate::PAYLOAD_KEY]
            .get("audio_data")
            .is_none()
    );
    assert!(bind_media(&body, [true, true, false]).is_err());
    body["image_data"] = json!(image);
    assert!(bind_media(&body, [false, false, true]).is_err());
    let req = bind_media(&body, [true, true, true]).unwrap().request;
    assert_eq!(req.token_ids.as_slice(), ids);
    assert_eq!(
        serde_json::to_value(req.multi_modal_data.unwrap()).unwrap(),
        json!({"image_url":[{"Url":image}], "audio_url":[{"Url":first},{"Url":second}]})
    );
    for value in [json!([[first]]), json!("file:///tmp/sound.wav")] {
        body["audio_data"] = value;
        assert!(bind_media(&body, [true, true, true]).is_err());
    }
}
