// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The JSON fixtures in `media_fixtures/`, shared with the Python tests in
//! `components/src/dynamo/common/tests/test_media_fixtures.py`, which hold the
//! generated models to the same files.
//!
//! - `<module>_request.json` and `<module>_response.json`: one full message
//!   each. Both sides parse it and serialize it back unchanged, so the
//!   generated model reads what the frontend writes, and the frontend reads
//!   what a worker writes.
//! - `<module>_cases.json`: bodies that both sides must accept, and bodies
//!   that both sides must reject: a wrong type, a value outside a bound, a
//!   missing required field, an unknown value of a response enum. A request
//!   is accepted when it parses and passes the frontend's validation, as at
//!   the HTTP boundary. A case marked `frontend_only` is rejected here and
//!   accepted by the generated model: an unknown value of a request enum,
//!   which a worker must accept from a newer frontend, or a rule the schema
//!   cannot state. This is the test of the claim that a worker rejects what
//!   the frontend rejects, for everything the generated model mirrors.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::audios::{NvAudioSpeechResponse, NvCreateAudioSpeechRequest};
use super::images::{NvCreateImageRequest, NvImagesResponse};
use super::videos::{NvCreateVideoRequest, NvVideosResponse};
use crate::engines::ValidateRequest;

fn fixture(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/protocols/openai/media_fixtures")
        .join(file)
}

fn round_trips<T: DeserializeOwned + Serialize>(file: &str) {
    let path = fixture(file);
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

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    valid: bool,
    /// The frontend rejects the body and the generated model accepts it.
    #[serde(default)]
    frontend_only: bool,
    body: Value,
}

#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

/// The frontend accepts a request that parses and passes validation.
fn request_accepts<T: DeserializeOwned + ValidateRequest>(body: Value) -> bool {
    serde_json::from_value::<T>(body)
        .map(|request| ValidateRequest::validate(&request).is_ok())
        .unwrap_or(false)
}

/// The frontend accepts a worker's response that parses.
fn response_accepts<T: DeserializeOwned>(body: Value) -> bool {
    serde_json::from_value::<T>(body).is_ok()
}

/// Every case in `file` is accepted or rejected as it says. The file must
/// hold request and response cases, and no two cases of a kind share a name.
fn cases_match_the_rust_types<Req, Resp>(file: &str)
where
    Req: DeserializeOwned + ValidateRequest,
    Resp: DeserializeOwned,
{
    let path = fixture(file);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let cases: Cases = serde_json::from_str(&text).unwrap_or_else(|err| panic!("{file}: {err}"));
    let mut seen = HashSet::new();
    let (mut requests, mut responses) = (0, 0);
    let mut failures = Vec::new();
    for case in cases.cases {
        assert!(
            seen.insert((case.kind.clone(), case.name.clone())),
            "{file}: two {} cases named {}",
            case.kind,
            case.name
        );
        assert!(
            !(case.frontend_only && case.valid),
            "{file}: {} {}: the frontend rejects a frontend_only case, so valid must be false",
            case.kind,
            case.name
        );
        let accepted = match case.kind.as_str() {
            "request" => {
                requests += 1;
                request_accepts::<Req>(case.body)
            }
            "response" => {
                responses += 1;
                response_accepts::<Resp>(case.body)
            }
            other => panic!("{file}: {}: unknown case type {other:?}", case.name),
        };
        if accepted != case.valid {
            failures.push(format!(
                "{} {}: expected valid={}, got {accepted}",
                case.kind, case.name, case.valid
            ));
        }
    }
    assert!(
        requests > 0 && responses > 0,
        "{file} must hold request and response cases"
    );
    assert!(failures.is_empty(), "{file}:\n{}", failures.join("\n"));
}

#[test]
fn audio_cases_match_the_rust_types() {
    cases_match_the_rust_types::<NvCreateAudioSpeechRequest, NvAudioSpeechResponse>(
        "audio_cases.json",
    );
}

#[test]
fn image_cases_match_the_rust_types() {
    cases_match_the_rust_types::<NvCreateImageRequest, NvImagesResponse>("image_cases.json");
}

#[test]
fn video_cases_match_the_rust_types() {
    cases_match_the_rust_types::<NvCreateVideoRequest, NvVideosResponse>("video_cases.json");
}
