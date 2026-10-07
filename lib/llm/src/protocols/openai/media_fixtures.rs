// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The JSON fixtures in `media_fixtures/`: one request and one response per
//! media module. The tests here parse each fixture with the Rust type and
//! check that it serializes back unchanged. The Python tests in
//! `components/src/dynamo/common/tests/test_media_fixtures.py` do the same
//! with the generated models. One message that both sides accept and
//! reproduce shows what the schema check alone does not: that the generated
//! model reads what the frontend writes, and the frontend reads what a worker
//! writes.

use std::path::Path;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::audios::{NvAudioSpeechResponse, NvCreateAudioSpeechRequest};
use super::images::{NvCreateImageRequest, NvImagesResponse};
use super::videos::{NvCreateVideoRequest, NvVideosResponse};

fn round_trips<T: DeserializeOwned + Serialize>(file: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/protocols/openai/media_fixtures")
        .join(file);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let expected: Value = serde_json::from_str(&text).unwrap();
    let parsed: T = serde_json::from_str(&text).unwrap_or_else(|err| panic!("{file}: {err}"));
    let actual = without_nulls(serde_json::to_value(parsed).unwrap());
    assert_eq!(
        actual, expected,
        "{file} changed on the way through the Rust type"
    );
}

/// A fixture omits an absent field. A response type writes it as `null`.
fn without_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| (key, without_nulls(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(without_nulls).collect()),
        other => other,
    }
}

#[test]
fn audio_request_fixture_round_trips() {
    round_trips::<NvCreateAudioSpeechRequest>("audio_request.json");
}

#[test]
fn audio_response_fixture_round_trips() {
    round_trips::<NvAudioSpeechResponse>("audio_response.json");
}

#[test]
fn image_request_fixture_round_trips() {
    round_trips::<NvCreateImageRequest>("image_request.json");
}

#[test]
fn image_response_fixture_round_trips() {
    round_trips::<NvImagesResponse>("image_response.json");
}

#[test]
fn video_request_fixture_round_trips() {
    round_trips::<NvCreateVideoRequest>("video_request.json");
}

#[test]
fn video_response_fixture_round_trips() {
    round_trips::<NvVideosResponse>("video_response.json");
}
