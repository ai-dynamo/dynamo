// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{io::Cursor, sync::LazyLock};

use axum::{
    body::Body,
    extract::{FromRequest, Multipart, multipart::MultipartError},
    http::{Request, StatusCode, header::CONTENT_TYPE},
};
use base64::Engine as _;
use dynamo_runtime::error::ErrorClass;
use serde_json::{Map, Value};

use super::{
    error::HttpError,
    openai::{ErrorMessage, ErrorResponse, parse_json_request, read_json_request_body},
};
use crate::protocols::openai::images::NvCreateImageRequest;

#[derive(Clone, Debug)]
pub(super) struct Limits {
    pub images: usize,
    pub file_bytes: usize,
    pub total_bytes: usize,
    pub text_bytes: usize,
    pub parts: usize,
    pub pixels: u64,
    pub total_pixels: u64,
    pub request_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            images: 4,
            file_bytes: 10 * 1024 * 1024,
            total_bytes: 20 * 1024 * 1024,
            text_bytes: 64 * 1024,
            parts: 32,
            pixels: 16_000_000,
            total_pixels: 32_000_000,
            // Leave room for the request-plane envelope below the default
            // 32 MiB TCP frame cap, including 20 MiB of base64-encoded uploads.
            request_bytes: 28 * 1024 * 1024,
        }
    }
}

impl Limits {
    fn from_env() -> Result<Self, String> {
        fn read(name: &str, default: usize) -> Result<usize, String> {
            match std::env::var(name) {
                Err(std::env::VarError::NotPresent) => Ok(default),
                value => value
                    .ok()
                    .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))
                    .and_then(|v| v.parse().ok())
                    .filter(|v| *v > 0)
                    .ok_or_else(|| format!("{name} must be a positive integer")),
            }
        }
        let defaults = Self::default();
        Ok(Self {
            images: read("DYN_IMAGE_EDIT_MAX_IMAGES", defaults.images)?,
            file_bytes: read("DYN_IMAGE_EDIT_MAX_FILE_BYTES", defaults.file_bytes)?,
            total_bytes: read("DYN_IMAGE_EDIT_MAX_TOTAL_BYTES", defaults.total_bytes)?,
            text_bytes: read("DYN_IMAGE_EDIT_MAX_TEXT_BYTES", defaults.text_bytes)?,
            parts: read("DYN_IMAGE_EDIT_MAX_PARTS", defaults.parts)?,
            pixels: read("DYN_IMAGE_EDIT_MAX_PIXELS", defaults.pixels as usize)? as u64,
            total_pixels: read(
                "DYN_IMAGE_EDIT_MAX_TOTAL_PIXELS",
                defaults.total_pixels as usize,
            )? as u64,
            request_bytes: read("DYN_IMAGE_EDIT_MAX_REQUEST_BYTES", defaults.request_bytes)?,
        })
    }
}

static LIMITS: LazyLock<Result<Limits, String>> = LazyLock::new(Limits::from_env);

pub(super) fn validate_config() -> anyhow::Result<()> {
    LIMITS
        .as_ref()
        .map(|_| ())
        .map_err(|message| anyhow::anyhow!(message.clone()))
}

fn configured_limits() -> Result<&'static Limits, ErrorResponse> {
    LIMITS.as_ref().map_err(|message| {
        ErrorMessage::internal_server_error_with_details(
            "Invalid image edit configuration",
            message.clone(),
        )
    })
}

fn error(status: StatusCode, message: impl Into<String>) -> ErrorResponse {
    let class = match status {
        StatusCode::PAYLOAD_TOO_LARGE => ErrorClass::PayloadTooLarge,
        StatusCode::UNSUPPORTED_MEDIA_TYPE => ErrorClass::UnsupportedMedia,
        _ => ErrorClass::InvalidRequest,
    };
    ErrorMessage::from_http_error(
        class,
        HttpError {
            code: status.as_u16(),
            message: message.into(),
        },
    )
}

fn malformed(message: &'static str) -> ErrorResponse {
    error(StatusCode::BAD_REQUEST, message)
}

fn oversized(limit: &'static str) -> ErrorResponse {
    error(
        StatusCode::PAYLOAD_TOO_LARGE,
        format!("Image edit exceeds the {limit} limit"),
    )
}

fn multipart_error(failure: MultipartError) -> ErrorResponse {
    if failure.status() == StatusCode::PAYLOAD_TOO_LARGE {
        oversized("HTTP body")
    } else {
        malformed("Malformed image edit multipart body")
    }
}

pub(super) async fn parse(request: Request<Body>) -> Result<NvCreateImageRequest, ErrorResponse> {
    parse_with_limits(request, configured_limits()?).await
}

pub(super) fn prepare(request: &mut NvCreateImageRequest) -> Result<(), ErrorResponse> {
    prepare_with_limits(request, configured_limits()?)
}

async fn parse_with_limits(
    request: Request<Body>,
    limits: &Limits,
) -> Result<NvCreateImageRequest, ErrorResponse> {
    let media_type = request
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<mime::Mime>().ok())
        .ok_or_else(|| {
            error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Expected application/json or multipart/form-data",
            )
        })?;
    let mut parsed =
        if media_type.type_() == mime::MULTIPART && media_type.subtype() == mime::FORM_DATA {
            let multipart = Multipart::from_request(request, &())
                .await
                .map_err(|_| malformed("Missing or invalid multipart boundary"))?;
            parse_multipart(multipart, limits).await?
        } else {
            let (parts, body) = request.into_parts();
            let body = read_json_request_body(&parts.headers, body).await?;
            parse_json_request::<NvCreateImageRequest>("image edits", &body)?
        };
    if parsed.input_reference.is_none() {
        return Err(malformed(
            "At least one image is required for /v1/images/edits",
        ));
    }
    prepare_with_limits(&mut parsed, limits)?;
    Ok(parsed)
}

fn prepare_with_limits(
    parsed: &mut NvCreateImageRequest,
    limits: &Limits,
) -> Result<(), ErrorResponse> {
    if parsed
        .input_reference
        .as_ref()
        .is_some_and(|references| references.len() > limits.images)
    {
        return Err(malformed(
            "Image edit exceeds the configured input-image count",
        ));
    }
    if parsed.input_reference.is_some()
        && ["mask", "mask_image", "image", "image[]", "url", "url[]"]
            .iter()
            .any(|key| parsed.passthrough.contains_key(*key))
    {
        return Err(malformed("Masks and mixed image sources are not supported"));
    }
    parsed.nest_passthrough();
    if parsed.input_reference.is_some() {
        check_serialized_size(parsed, limits.request_bytes)?;
    }
    Ok(())
}

async fn parse_multipart(
    mut multipart: Multipart,
    limits: &Limits,
) -> Result<NvCreateImageRequest, ErrorResponse> {
    let mut fields = Map::new();
    let mut references = Vec::new();
    let mut image_alias = None;
    let mut encoded_bytes = 0;
    let (mut parts, mut total_bytes, mut text_bytes, mut total_pixels) = (0, 0, 0, 0u64);
    while let Some(mut field) = multipart.next_field().await.map_err(multipart_error)? {
        parts += 1;
        if parts > limits.parts {
            return Err(oversized("multipart part count"));
        }
        let name = field
            .name()
            .ok_or_else(|| malformed("Multipart fields must have a name"))?
            .to_owned();
        let is_image = matches!(name.as_str(), "image" | "image[]");
        if is_image {
            if image_alias.as_ref().is_some_and(|alias| alias != &name) {
                return Err(malformed("Do not mix image and image[] fields"));
            }
            image_alias = Some(name.clone());
            if references.len() >= limits.images {
                return Err(malformed(
                    "Image edit exceeds the configured input-image count",
                ));
            }
        } else if field.file_name().is_some()
            || !matches!(
                name.as_str(),
                "prompt"
                    | "model"
                    | "n"
                    | "quality"
                    | "background"
                    | "output_format"
                    | "response_format"
                    | "size"
                    | "style"
                    | "user"
                    | "moderation"
                    | "nvext"
            )
        {
            return Err(malformed(
                "Unsupported multipart field, mask, or mixed image source",
            ));
        } else if fields.contains_key(&name) {
            return Err(malformed("Duplicate multipart scalar field"));
        }

        let mut bytes = Vec::new();
        while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
            if is_image {
                if chunk.len() > limits.file_bytes.saturating_sub(bytes.len()) {
                    return Err(error(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        format!(
                            "input_reference[{}] exceeds the per-file limit of {} bytes",
                            references.len(),
                            limits.file_bytes
                        ),
                    ));
                }
                if chunk.len() > limits.total_bytes.saturating_sub(total_bytes) {
                    return Err(oversized("aggregate image bytes"));
                }
                total_bytes += chunk.len();
            } else {
                if chunk.len() > limits.text_bytes.saturating_sub(text_bytes) {
                    return Err(oversized("form text bytes"));
                }
                text_bytes += chunk.len();
            }
            bytes.extend_from_slice(&chunk);
        }
        if is_image {
            let (reference, pixels) = encode_image(
                &bytes,
                limits,
                total_pixels,
                encoded_bytes,
                references.len(),
            )?;
            total_pixels += pixels;
            encoded_bytes += reference.len();
            references.push(Value::String(reference));
        } else {
            let text =
                String::from_utf8(bytes).map_err(|_| malformed("Multipart text must be UTF-8"))?;
            let value = match name.as_str() {
                "n" => Value::from(
                    text.parse::<u32>()
                        .map_err(|_| malformed("n must be an integer"))?,
                ),
                "nvext" => {
                    let value: Value = serde_json::from_str(&text)
                        .map_err(|_| malformed("nvext must be a JSON object"))?;
                    if !value.is_object() {
                        return Err(malformed("nvext must be a JSON object"));
                    }
                    value
                }
                _ => Value::String(text),
            };
            fields.insert(name, value);
        }
    }
    if references.is_empty() {
        return Err(malformed("At least one image file is required"));
    }
    fields.insert("input_reference".into(), Value::Array(references));
    serde_json::from_value(Value::Object(fields))
        .map_err(|_| malformed("Invalid image edit form fields"))
}

fn encode_image(
    bytes: &[u8],
    limits: &Limits,
    used_pixels: u64,
    used_encoded_bytes: usize,
    index: usize,
) -> Result<(String, u64), ErrorResponse> {
    let indexed_error =
        |status, detail: &str| error(status, format!("input_reference[{index}]: {detail}"));
    if bytes.is_empty() {
        return Err(indexed_error(
            StatusCode::BAD_REQUEST,
            "Image files must not be empty",
        ));
    }
    let format = image::guess_format(bytes).map_err(|_| {
        indexed_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Expected a PNG, JPEG, or WebP image",
        )
    })?;
    let media = match format {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        image::ImageFormat::WebP => "image/webp",
        _ => {
            return Err(indexed_error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Expected a PNG, JPEG, or WebP image",
            ));
        }
    };
    let (width, height) = image::ImageReader::with_format(Cursor::new(bytes), format)
        .into_dimensions()
        .map_err(|_| indexed_error(StatusCode::BAD_REQUEST, "Invalid image content"))?;
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 {
        return Err(indexed_error(
            StatusCode::BAD_REQUEST,
            "Invalid image dimensions",
        ));
    }
    if pixels > limits.pixels || pixels > limits.total_pixels.saturating_sub(used_pixels) {
        return Err(indexed_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Decoded input pixel limit exceeded",
        ));
    }
    let length = bytes
        .len()
        .checked_add(2)
        .and_then(|n| (n / 3).checked_mul(4))
        .and_then(|n| n.checked_add(media.len() + "data:;base64,".len()))
        .filter(|length| *length <= limits.request_bytes.saturating_sub(used_encoded_bytes))
        .ok_or_else(|| oversized("serialized worker request"))?;
    let mut encoded = String::with_capacity(length);
    encoded.push_str("data:");
    encoded.push_str(media);
    encoded.push_str(";base64,");
    base64::engine::general_purpose::STANDARD.encode_string(bytes, &mut encoded);
    Ok((encoded, pixels))
}

pub(super) fn check_serialized_size(
    request: &NvCreateImageRequest,
    limit: usize,
) -> Result<(), ErrorResponse> {
    struct Budget(usize);
    impl std::io::Write for Budget {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_sub(bytes.len())
                .ok_or_else(|| std::io::Error::other("request budget exceeded"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(Budget(limit), request)
        .map_err(|_| oversized("serialized worker request"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, extract::DefaultBodyLimit, response::IntoResponse, routing::post};
    use bytes::Bytes;
    use tower::ServiceExt;

    struct Part<'a> {
        name: &'a str,
        filename: Option<&'a str>,
        data: Vec<u8>,
    }

    fn text<'a>(name: &'a str, data: &str) -> Part<'a> {
        Part {
            name,
            filename: None,
            data: data.as_bytes().to_vec(),
        }
    }

    fn image_part(name: &str, red: u8) -> Part<'_> {
        let mut bytes = Cursor::new(Vec::new());
        image::RgbImage::from_pixel(2, 2, image::Rgb([red, 0, 0]))
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        Part {
            name,
            filename: Some("input.dat"),
            data: bytes.into_inner(),
        }
    }

    fn multipart_body(parts: &[Part<'_>]) -> Vec<u8> {
        let mut body = Vec::new();
        for part in parts {
            body.extend_from_slice(
                format!(
                    "--edit-boundary\r\nContent-Disposition: form-data; name=\"{}\"",
                    part.name
                )
                .as_bytes(),
            );
            if let Some(filename) = part.filename {
                body.extend_from_slice(format!("; filename=\"{filename}\"").as_bytes());
            }
            body.extend_from_slice(b"\r\n\r\n");
            body.extend_from_slice(&part.data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--edit-boundary--\r\n");
        body
    }

    async fn send(
        body: Vec<u8>,
        content_type: &str,
        limits: Limits,
        body_limit: usize,
    ) -> (StatusCode, Value) {
        let app = Router::new()
            .route(
                "/",
                post(move |request: Request<Body>| {
                    let limits = limits.clone();
                    async move {
                        parse_with_limits(request, &limits)
                            .await
                            .map(Json)
                            .into_response()
                    }
                }),
            )
            .layer(DefaultBodyLimit::max(body_limit));
        // No Content-Length: body limits must cover streaming/chunked uploads.
        let chunks = body
            .chunks(17)
            .map(Bytes::copy_from_slice)
            .map(Ok::<_, std::io::Error>)
            .collect::<Vec<_>>();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/")
                    .header(CONTENT_TYPE, content_type)
                    .body(Body::from_stream(futures::stream::iter(chunks)))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    async fn upload(parts: Vec<Part<'_>>, limits: Limits) -> (StatusCode, Value) {
        send(
            multipart_body(&parts),
            "multipart/form-data; boundary=edit-boundary; charset=utf-8",
            limits,
            1024 * 1024,
        )
        .await
    }

    #[tokio::test]
    async fn uploads_preserve_part_order_and_typed_fields() {
        for alias in ["image", "image[]"] {
            let (status, body) = upload(
                vec![
                    image_part(alias, 10),
                    text("n", "2"),
                    text("prompt", "edit"),
                    image_part(alias, 20),
                    text("nvext", r#"{"seed":7}"#),
                ],
                Limits::default(),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(body["n"], 2);
            assert_eq!(body["nvext"]["seed"], 7);
            let refs = body["input_reference"].as_array().unwrap();
            for (reference, red) in refs.iter().zip([10, 20]) {
                let (_, base64) = reference.as_str().unwrap().split_once(',').unwrap();
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(base64)
                    .unwrap();
                assert_eq!(
                    image::load_from_memory(&bytes)
                        .unwrap()
                        .to_rgb8()
                        .get_pixel(0, 0)
                        .0,
                    [red, 0, 0]
                );
            }
        }
        let (status, body) = upload(
            vec![text("prompt", "edit"), image_part("image", 0)],
            Limits::default(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["input_reference"].is_string());
    }

    #[tokio::test]
    async fn ambiguous_or_unsupported_fields_fail_before_dispatch() {
        for additional in [
            text("prompt", "duplicate"),
            text("input_reference", "secret"),
            text("url", "secret"),
            text("url[]", "secret"),
            text("mask", "secret"),
            image_part("mask_image", 0),
            image_part("unknown", 0),
            image_part("prompt", 0),
            image_part("image[]", 0),
            text("nvext", "not-json"),
            text("nvext", "[]"),
            text("n", "1.5"),
        ] {
            let (status, body) = upload(
                vec![text("prompt", "edit"), image_part("image", 0), additional],
                Limits::default(),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert!(!body.to_string().contains("secret"));
        }
        for parts in [
            vec![text("prompt", "edit")],
            vec![image_part("image", 0)],
            vec![
                text("prompt", "edit"),
                Part {
                    name: "image",
                    filename: Some("input.png"),
                    data: vec![],
                },
            ],
        ] {
            assert_eq!(
                upload(parts, Limits::default()).await.0,
                StatusCode::BAD_REQUEST
            );
        }
        assert_eq!(
            upload(
                vec![text("prompt", "edit"), text("image", "not an image")],
                Limits::default()
            )
            .await
            .0,
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
    }

    #[tokio::test]
    async fn upload_limits_have_exact_boundaries() {
        let file_bytes = image_part("image", 0).data.len();
        let parts = || {
            vec![
                text("prompt", "edit"),
                image_part("image", 0),
                image_part("image", 0),
            ]
        };
        let limits = Limits {
            images: 2,
            file_bytes,
            total_bytes: 2 * file_bytes,
            text_bytes: 4,
            parts: 3,
            pixels: 4,
            total_pixels: 8,
            ..Limits::default()
        };
        assert_eq!(upload(parts(), limits.clone()).await.0, StatusCode::OK);
        for reduced in [
            Limits {
                file_bytes: file_bytes - 1,
                ..limits.clone()
            },
            Limits {
                total_bytes: 2 * file_bytes - 1,
                ..limits.clone()
            },
            Limits {
                text_bytes: 3,
                ..limits.clone()
            },
            Limits {
                parts: 2,
                ..limits.clone()
            },
            Limits {
                pixels: 3,
                ..limits.clone()
            },
            Limits {
                total_pixels: 7,
                ..limits.clone()
            },
            Limits {
                request_bytes: 20,
                ..limits.clone()
            },
        ] {
            assert_eq!(
                upload(parts(), reduced).await.0,
                StatusCode::PAYLOAD_TOO_LARGE
            );
        }
        assert_eq!(
            upload(
                parts(),
                Limits {
                    images: 1,
                    ..limits
                }
            )
            .await
            .0,
            StatusCode::BAD_REQUEST
        );
        let body = multipart_body(&parts());
        assert_eq!(
            send(
                body.clone(),
                "multipart/form-data; boundary=edit-boundary",
                Limits::default(),
                body.len()
            )
            .await
            .0,
            StatusCode::OK
        );
        assert_eq!(
            send(
                body.clone(),
                "multipart/form-data; boundary=edit-boundary",
                Limits::default(),
                body.len() - 1
            )
            .await
            .0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    #[tokio::test]
    async fn json_and_form_validation_keep_the_same_reference_contract() {
        for references in [
            serde_json::json!("https://example.com/a"),
            serde_json::json!(["https://example.com/a", "https://example.com/b"]),
        ] {
            let body = serde_json::to_vec(
                &serde_json::json!({"prompt":"edit", "input_reference": references}),
            )
            .unwrap();
            let (status, parsed) = send(
                body,
                "application/json; charset=utf-8",
                Limits::default(),
                1024,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(parsed["input_reference"], references);
        }
        for references in [
            serde_json::json!([]),
            serde_json::json!(["a", " "]),
            serde_json::json!(["a", 1]),
            serde_json::json!(null),
        ] {
            let body = serde_json::to_vec(
                &serde_json::json!({"prompt":"edit", "input_reference": references}),
            )
            .unwrap();
            assert_eq!(
                send(body, "application/json", Limits::default(), 1024)
                    .await
                    .0,
                StatusCode::BAD_REQUEST
            );
        }
        assert_eq!(
            send(vec![], "multipart/form-data", Limits::default(), 1024)
                .await
                .0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            send(vec![], "text/plain", Limits::default(), 1024).await.0,
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
    }

    #[test]
    fn serialized_budget_counts_encoding_and_all_fields() {
        let mut request: NvCreateImageRequest = serde_json::from_value(serde_json::json!({"prompt":"edit", "input_reference":["first","second"], "custom":"value"})).unwrap();
        request.nest_passthrough();
        let length = serde_json::to_vec(&request).unwrap().len();
        assert!(check_serialized_size(&request, length).is_ok());
        assert_eq!(
            check_serialized_size(&request, length - 1).unwrap_err().0,
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }
}
