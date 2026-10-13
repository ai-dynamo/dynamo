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
//!    [`media_schemas`], and an entry in `MODULES` in
//!    `scripts/generate_media_protocols.py`.

use std::path::Path;

use utoipa::openapi::schema::{AdditionalProperties, ObjectBuilder, Schema, SchemaType, Type};
use utoipa::openapi::{Info, OpenApi as Document, RefOr};
use utoipa::{OpenApi, ToSchema};

use super::audios::{NvAudioSpeechResponse, NvCreateAudioSpeechRequest};
use super::images::{NvCreateImageRequest, NvImagesResponse};
use super::videos::{NvCreateVideoRequest, NvVideosResponse};

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
