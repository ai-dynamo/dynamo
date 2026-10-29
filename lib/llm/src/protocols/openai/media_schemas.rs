// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The JSON schemas of the media protocol types, committed in
//! `media_schemas/`. They are the source of the generated Python models in
//! `components/src/dynamo/common/protocols/`.
//!
//! 1. A change to a media type changes its schema. The test in this module
//!    fails until the committed schema matches.
//! 2. Rewrite the schemas with
//!    `DYNAMO_UPDATE_MEDIA_SCHEMAS=1 cargo test -p dynamo-llm --lib media_schemas`.
//! 3. Commit the schemas. The `generate-media-protocols` pre-commit hook
//!    regenerates the Python models from them.
//! 4. A new media module gets an [`OpenApi`] struct here, an entry in
//!    [`media_schemas`], a bounds test like [`image_schema_bounds_hold_on_the_rust_types`],
//!    and an entry in `MODULES` in `scripts/generate_media_protocols.py`.
//!
//! A numeric bound is written twice: `#[schema(minimum, maximum)]` feeds the
//! schema and the generated model, `#[validate(range)]` feeds the frontend
//! check. The bounds tests tie the two together: every bound the schema
//! states is checked against the Rust type, so the generated model never
//! rejects a value the frontend forwards.
//!
//! What the generated model mirrors is what the schema states, less the
//! values of a request enum the worker does not read: the type of each
//! field, its bounds, its default, the values of a response enum, and the
//! values of the delivery mode, which every worker reads. Any other request
//! enum generates as a plain string, so a worker accepts what a newer
//! frontend forwards for a field it does not read;
//! `scripts/generate_media_protocols.py` says why. A field whose wire has
//! two shapes and whose worker type is one, the image references
//! ([`super::images::ImageReferences`]), generates as a hand-written type
//! that the script maps onto the field (`WORKER_TYPES` there).
//! The shared cases in `media_fixtures/` ([`super::media_fixtures`]) hold
//! both sides to that for accepted and rejected input. A Rust rule the
//! schema does not state, such as an integer width, is checked by the
//! frontend alone.

use std::path::Path;

use serde::de::DeserializeOwned;
use serde_json::Value;
use utoipa::openapi::schema::{AdditionalProperties, ObjectBuilder, Schema, SchemaType, Type};
use utoipa::openapi::{Info, OpenApi as Document, RefOr};
use utoipa::{OpenApi, ToSchema};

use super::audios::{NvAudioSpeechResponse, NvCreateAudioSpeechRequest};
use super::images::{NvCreateImageRequest, NvImagesResponse};
use super::videos::{NvCreateVideoRequest, NvVideosResponse};
use crate::engines::ValidateRequest;

/// Set to rewrite the committed schemas instead of checking them.
const UPDATE_VAR: &str = "DYNAMO_UPDATE_MEDIA_SCHEMAS";

#[derive(OpenApi)]
#[openapi(components(schemas(NvCreateAudioSpeechRequest, NvAudioSpeechResponse)))]
struct AudioSchemas;

#[derive(OpenApi)]
#[openapi(components(schemas(NvCreateImageRequest, NvImagesResponse)))]
struct ImageSchemas;

#[derive(OpenApi)]
#[openapi(components(schemas(NvCreateVideoRequest, NvVideosResponse)))]
struct VideoSchemas;

/// One committed schema file per Python module: the file name and the
/// committed text.
fn media_schemas() -> [(&'static str, String); 3] {
    [
        (
            "audio.json",
            schema_json(AudioSchemas::openapi(), &NvCreateAudioSpeechRequest::name()),
        ),
        (
            "image.json",
            schema_json(ImageSchemas::openapi(), &NvCreateImageRequest::name()),
        ),
        (
            "video.json",
            schema_json(VideoSchemas::openapi(), &NvCreateVideoRequest::name()),
        ),
    ]
}

/// The committed text of `document`, whose request schema is `request`.
fn schema_json(mut document: Document, request: &str) -> String {
    // The derived info block carries the crate version. A fixed block keeps
    // the schema stable across releases.
    document.info = Info::new("Dynamo media protocol", "1");
    add_extra_args(&mut document, request);
    let json = document
        .to_pretty_json()
        .expect("an OpenAPI document serializes to JSON");
    format!("{json}\n")
}

/// Add the worker-boundary `extra_args` field to the request schema.
///
/// A client never sets the field, so it is `#[serde(skip_deserializing)]`,
/// and utoipa drops such a field. The workers read it.
fn add_extra_args(document: &mut Document, request: &str) {
    let Some(RefOr::T(Schema::Object(schema))) = document
        .components
        .as_mut()
        .and_then(|components| components.schemas.get_mut(request))
    else {
        panic!("the document has no object schema named {request}");
    };
    let extra_args = ObjectBuilder::new()
        .schema_type(SchemaType::from_iter([Type::Object, Type::Null]))
        .additional_properties(Some(AdditionalProperties::FreeForm(true)))
        .description(Some(
            "Worker-boundary passthrough. The frontend nests unknown top-level \
             request fields (an OpenAI client's extra_body) under the \
             \"media_passthrough\" key.",
        ));
    schema
        .properties
        .insert("extra_args".to_string(), extra_args.into());
}

#[test]
fn media_schemas_match_the_committed_files() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/protocols/openai/media_schemas");
    let update = std::env::var_os(UPDATE_VAR).is_some();
    let mut stale = Vec::new();
    for (file, schema) in media_schemas() {
        let path = dir.join(file);
        if update {
            std::fs::write(&path, schema).unwrap();
        } else if std::fs::read_to_string(&path).ok() != Some(schema) {
            stale.push(file);
        }
    }
    assert!(
        stale.is_empty(),
        "{stale:?} in {} are out of date. Run `{UPDATE_VAR}=1 cargo test -p dynamo-llm --lib \
         media_schemas` and commit the result.",
        dir.display()
    );
}

// --- Schema bounds versus the Rust types ---

/// A numeric bound a schema states for one property: the JSON pointer of
/// the property in an instance, whether the property is an integer, and
/// the inclusive bounds.
struct SchemaBound {
    pointer: String,
    integer: bool,
    minimum: Option<f64>,
    maximum: Option<f64>,
}

/// Every bound the schema named `root` states, following `$ref`, the
/// `oneOf` that utoipa writes for an `Option`, and array items.
fn schema_bounds(document: &Document, root: &str) -> Vec<SchemaBound> {
    let document = serde_json::to_value(document).unwrap();
    let schemas = &document["components"]["schemas"];
    let schema = schemas
        .get(root)
        .unwrap_or_else(|| panic!("the document has no schema named {root}"));
    let mut bounds = Vec::new();
    collect_bounds(schemas, schema, String::new(), &mut bounds);
    bounds
}

fn collect_bounds(schemas: &Value, schema: &Value, pointer: String, bounds: &mut Vec<SchemaBound>) {
    if let Some(reference) = schema["$ref"].as_str() {
        let name = reference.rsplit('/').next().unwrap();
        let target = schemas
            .get(name)
            .unwrap_or_else(|| panic!("{pointer}: {reference} is not in the document"));
        collect_bounds(schemas, target, pointer, bounds);
        return;
    }
    if let Some(variants) = schema["oneOf"].as_array() {
        for variant in variants.iter().filter(|variant| variant["type"] != "null") {
            collect_bounds(schemas, variant, pointer.clone(), bounds);
        }
        return;
    }
    if let Some(items) = schema.get("items") {
        collect_bounds(schemas, items, format!("{pointer}/0"), bounds);
        return;
    }
    if let Some(properties) = schema["properties"].as_object() {
        for (name, property) in properties {
            collect_bounds(schemas, property, format!("{pointer}/{name}"), bounds);
        }
        return;
    }
    for exclusive in ["exclusiveMinimum", "exclusiveMaximum"] {
        assert!(
            schema.get(exclusive).is_none(),
            "{pointer}: {exclusive} is not checked here"
        );
    }
    let minimum = schema["minimum"].as_f64();
    let maximum = schema["maximum"].as_f64();
    if minimum.is_some() || maximum.is_some() {
        let integer = match &schema["type"] {
            Value::String(kind) => kind == "integer",
            Value::Array(kinds) => kinds.iter().any(|kind| kind == "integer"),
            _ => false,
        };
        bounds.push(SchemaBound {
            pointer,
            integer,
            minimum,
            maximum,
        });
    }
}

/// The fixture `file` in `media_fixtures/`, as JSON.
fn fixture(file: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/protocols/openai/media_fixtures")
        .join(file);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// Checks every bound that the schema `root` states against `T`, starting
/// from the fixture `file`, which must set every bounded field: `T` accepts
/// the bound and rejects the value just outside it, in the parser or in
/// `accepts`.
fn assert_bounds_hold<T: DeserializeOwned>(
    document: Document,
    root: &str,
    file: &str,
    accepts: impl Fn(&T) -> bool,
) {
    let fixture = fixture(file);
    let accepted = |pointer: &str, value: f64| {
        let mut instance = fixture.clone();
        let field = instance
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("{file} does not set {pointer}, which {root} bounds"));
        // A whole value is written as an integer: serde rejects `11.0` for
        // an integer field, which would pass the check for the wrong reason.
        *field = if value.fract() == 0.0 {
            Value::from(value as i64)
        } else {
            Value::from(value)
        };
        serde_json::from_value::<T>(instance).is_ok_and(|parsed| accepts(&parsed))
    };
    for bound in schema_bounds(&document, root) {
        let SchemaBound {
            pointer,
            integer,
            minimum,
            maximum,
        } = bound;
        let outside = |bound: f64, above: bool| match (integer, above) {
            (true, true) => bound + 1.0,
            (true, false) => bound - 1.0,
            (false, true) => bound.next_up(),
            (false, false) => bound.next_down(),
        };
        if let Some(minimum) = minimum {
            assert!(
                accepted(&pointer, minimum),
                "{root}{pointer}: the minimum {minimum} is rejected"
            );
            assert!(
                !accepted(&pointer, outside(minimum, false)),
                "{root}{pointer}: a value below the minimum {minimum} is accepted"
            );
        }
        if let Some(maximum) = maximum {
            assert!(
                accepted(&pointer, maximum),
                "{root}{pointer}: the maximum {maximum} is rejected"
            );
            assert!(
                !accepted(&pointer, outside(maximum, true)),
                "{root}{pointer}: a value above the maximum {maximum} is accepted"
            );
        }
    }
}

#[test]
fn schema_bounds_follow_the_nested_request_types() {
    let pointers: Vec<String> =
        schema_bounds(&VideoSchemas::openapi(), &NvCreateVideoRequest::name())
            .into_iter()
            .map(|bound| bound.pointer)
            .collect();
    assert!(pointers.contains(&"/nvext/fps".to_string()), "{pointers:?}");
}

#[test]
fn audio_schema_bounds_hold_on_the_rust_types() {
    assert_bounds_hold(
        AudioSchemas::openapi(),
        &NvCreateAudioSpeechRequest::name(),
        "audio_request.json",
        |request: &NvCreateAudioSpeechRequest| ValidateRequest::validate(request).is_ok(),
    );
    assert_bounds_hold(
        AudioSchemas::openapi(),
        &NvAudioSpeechResponse::name(),
        "audio_response.json",
        |_: &NvAudioSpeechResponse| true,
    );
}

#[test]
fn image_schema_bounds_hold_on_the_rust_types() {
    assert_bounds_hold(
        ImageSchemas::openapi(),
        &NvCreateImageRequest::name(),
        "image_request.json",
        |request: &NvCreateImageRequest| ValidateRequest::validate(request).is_ok(),
    );
    assert_bounds_hold(
        ImageSchemas::openapi(),
        &NvImagesResponse::name(),
        "image_response.json",
        |_: &NvImagesResponse| true,
    );
}

#[test]
fn video_schema_bounds_hold_on_the_rust_types() {
    assert_bounds_hold(
        VideoSchemas::openapi(),
        &NvCreateVideoRequest::name(),
        "video_request.json",
        |request: &NvCreateVideoRequest| ValidateRequest::validate(request).is_ok(),
    );
    assert_bounds_hold(
        VideoSchemas::openapi(),
        &NvVideosResponse::name(),
        "video_response.json",
        |_: &NvVideosResponse| true,
    );
}
