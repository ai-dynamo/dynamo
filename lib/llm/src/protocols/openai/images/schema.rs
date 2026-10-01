// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema mirrors for the image protocol types.
//!
//! [`super::NvCreateImageRequest`] has hand-rolled serde. The upstream OpenAI
//! types carry no schema derive. The generated Python model for images comes
//! from the mirrors in this module. The tests hold each mirror equal to the
//! upstream key set and enum values.
//!
//! The upstream `u8` fields are signed here on purpose. The frontend rejects
//! a negative value through the upstream type. The worker model stays a
//! tolerant reader, so its own range checks keep their error messages.
//!
//! To change an image field:
//! 1. Change the upstream type or [`super::NvCreateImageRequest`].
//! 2. Change the same field in the mirror below.
//! 3. Run the tests in this module. They fail when the two key sets differ.
//! 4. Regenerate the Python model: `cargo run -p dynamo-llm --bin generate-media-protocols`.
//!
//! The mirrors are schema-only. No code constructs or serializes them.

use utoipa::ToSchema;

use super::NvExt;

/// Request for image generation (/v1/images/generations endpoint).
///
/// Mirrors the upstream `CreateImageRequest` plus the Dynamo fields
/// `input_reference` and `nvext`.
#[derive(ToSchema)]
#[schema(as = NvCreateImageRequest)]
pub struct NvCreateImageRequestSchema {
    /// The text prompt for image generation.
    pub prompt: String,

    /// The model to use for image generation.
    pub model: Option<String>,

    /// Number of images to generate (1 to 10).
    pub n: Option<i64>,

    /// Image quality.
    pub quality: Option<ImageQuality>,

    /// How the generated images are returned.
    pub response_format: Option<ImageResponseFormat>,

    /// Output format of the generated images.
    pub output_format: Option<ImageOutputFormat>,

    /// Compression level of the output image, in percent.
    pub output_compression: Option<i64>,

    /// Whether to stream partial images.
    pub stream: Option<bool>,

    /// Number of partial images to stream.
    pub partial_images: Option<i64>,

    /// Image size in WxH format (e.g. 1024x1024).
    pub size: Option<String>,

    /// Content moderation level.
    pub moderation: Option<ImageModeration>,

    /// Background of the generated images.
    pub background: Option<ImageBackground>,

    /// Image style.
    pub style: Option<ImageStyle>,

    /// Optional user identifier.
    pub user: Option<String>,

    /// Optional image reference that guides generation (for I2I/TI2I).
    pub input_reference: Option<String>,

    /// NVIDIA extensions.
    pub nvext: Option<NvExt>,
}

/// Individual image data in a response.
///
/// Mirrors the untagged upstream `Image` enum as one object with optional
/// fields.
#[derive(ToSchema)]
pub struct ImageData {
    /// URL of the generated image (if response_format is url).
    pub url: Option<String>,

    /// Base64-encoded image (if response_format is b64_json).
    pub b64_json: Option<String>,

    /// Revised prompt, when the model rewrites the original prompt.
    pub revised_prompt: Option<String>,
}

/// Response structure for image generation.
///
/// Mirrors the upstream `ImagesResponse`.
#[derive(ToSchema)]
#[schema(as = NvImagesResponse)]
pub struct NvImagesResponseSchema {
    /// Unix timestamp of creation.
    pub created: i64,

    /// List of generated images.
    #[schema(default = json!([]))]
    pub data: Vec<ImageData>,

    /// Background of the generation: transparent or opaque.
    pub background: Option<ImageResponseBackground>,

    /// Output format of the generated images: png, webp, or jpeg.
    pub output_format: Option<ImageOutputFormat>,

    /// Size of the generated images in WxH format.
    pub size: Option<String>,

    /// Quality of the generated images: low, medium, or high.
    pub quality: Option<ImageQuality>,

    /// Token usage of the generation, when the model reports it.
    #[schema(value_type = Option<Object>)]
    pub usage: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Image quality. Values match the upstream enum.
#[derive(ToSchema)]
#[schema(rename_all = "lowercase")]
pub enum ImageQuality {
    Standard,
    HD,
    High,
    Medium,
    Low,
    Auto,
}

/// How the generated images are returned. Values match the upstream enum.
#[derive(ToSchema)]
#[schema(rename_all = "lowercase")]
pub enum ImageResponseFormat {
    Url,
    #[schema(rename = "b64_json")]
    B64Json,
}

/// Output format of the generated images. Values match the upstream enum.
#[derive(ToSchema)]
#[schema(rename_all = "lowercase")]
pub enum ImageOutputFormat {
    Png,
    Jpeg,
    Webp,
}

/// Content moderation level. Values match the upstream enum.
#[derive(ToSchema)]
#[schema(rename_all = "lowercase")]
pub enum ImageModeration {
    Auto,
    Low,
}

/// Background of the generated images. Values match the upstream enum.
#[derive(ToSchema)]
#[schema(rename_all = "lowercase")]
pub enum ImageBackground {
    Auto,
    Transparent,
    Opaque,
}

/// Image style. Values match the upstream enum.
#[derive(ToSchema)]
#[schema(rename_all = "lowercase")]
pub enum ImageStyle {
    Vivid,
    Natural,
}

/// Background reported in a response. Values match the upstream enum.
#[derive(ToSchema)]
#[schema(rename_all = "lowercase")]
pub enum ImageResponseBackground {
    Transparent,
    Opaque,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use std::any::type_name;

    use dynamo_protocols::types as upstream;
    use utoipa::PartialSchema;

    use super::*;
    use crate::protocols::openai::images::{NvCreateImageRequest, NvImagesResponse};
    use crate::protocols::openai::schema_tests::object_schema;

    fn sorted_property_names<T: PartialSchema>() -> Vec<String> {
        let mut names: Vec<String> = object_schema::<T>().properties.keys().cloned().collect();
        names.sort();
        names
    }

    fn sorted_wire_keys<T: serde::Serialize>(value: &T) -> Vec<String> {
        let serde_json::Value::Object(map) = serde_json::to_value(value).unwrap() else {
            panic!("the value does not serialize to an object");
        };
        let mut keys: Vec<String> = map.keys().cloned().collect();
        keys.sort();
        keys
    }

    fn enum_values<T: PartialSchema>() -> Vec<serde_json::Value> {
        object_schema::<T>()
            .enum_values
            .unwrap_or_else(|| panic!("{} has no enum values", type_name::<T>()))
    }

    fn upstream_values<T: serde::Serialize>(variants: Vec<T>) -> Vec<serde_json::Value> {
        variants
            .into_iter()
            .map(|variant| serde_json::to_value(variant).unwrap())
            .collect()
    }

    #[test]
    fn image_request_schema_mirrors_upstream_fields() {
        // Every upstream field is set, so every wire key appears.
        let inner = upstream::CreateImageRequest {
            prompt: "a cat".into(),
            model: Some(upstream::ImageModel::DallE3),
            n: Some(1),
            quality: Some(upstream::ImageQuality::High),
            response_format: Some(upstream::ImageResponseFormat::Url),
            output_format: Some(upstream::ImageOutputFormat::Png),
            output_compression: Some(50),
            stream: Some(false),
            partial_images: Some(1),
            size: Some(upstream::ImageSize::S1024x1024),
            moderation: Some(upstream::ImageModeration::Auto),
            background: Some(upstream::ImageBackground::Opaque),
            style: Some(upstream::ImageStyle::Vivid),
            user: Some("user".into()),
        };
        let request = NvCreateImageRequest {
            inner,
            input_reference: Some("ref".into()),
            nvext: Some(NvExt::default()),
            extra_args: None,
            passthrough: serde_json::Map::new(),
        };

        assert_eq!(
            sorted_property_names::<NvCreateImageRequestSchema>(),
            sorted_wire_keys(&request)
        );
    }

    #[test]
    fn image_data_schema_mirrors_both_upstream_variants() {
        let url = upstream::Image::Url {
            url: "http://x/a.png".into(),
            revised_prompt: Some("a cat".into()),
        };
        let b64_json = upstream::Image::B64Json {
            b64_json: Arc::new("abc==".into()),
            revised_prompt: Some("a cat".into()),
        };
        let mut wire = sorted_wire_keys(&url);
        wire.extend(sorted_wire_keys(&b64_json));
        wire.sort();
        wire.dedup();

        assert_eq!(sorted_property_names::<ImageData>(), wire);
    }

    #[test]
    fn images_response_schema_mirrors_upstream_fields() {
        let inner = upstream::ImagesResponse {
            created: 0,
            data: vec![],
            background: Some(upstream::ImageResponseBackground::Opaque),
            output_format: Some(upstream::ImageOutputFormat::Png),
            size: Some(upstream::ImageSize::S1024x1024),
            quality: Some(upstream::ImageQuality::High),
            usage: Some(upstream::ImageGenUsage {
                input_tokens: 1,
                total_tokens: 2,
                output_tokens: 1,
                output_token_details: None,
                input_tokens_details: upstream::ImageGenInputUsageDetails {
                    text_tokens: 1,
                    image_tokens: 0,
                },
            }),
        };
        let response = NvImagesResponse { inner };

        assert_eq!(
            sorted_property_names::<NvImagesResponseSchema>(),
            sorted_wire_keys(&response)
        );
    }

    #[test]
    fn image_enum_values_match_upstream() {
        assert_eq!(
            enum_values::<ImageQuality>(),
            upstream_values(vec![
                upstream::ImageQuality::Standard,
                upstream::ImageQuality::HD,
                upstream::ImageQuality::High,
                upstream::ImageQuality::Medium,
                upstream::ImageQuality::Low,
                upstream::ImageQuality::Auto,
            ])
        );
        assert_eq!(
            enum_values::<ImageResponseFormat>(),
            upstream_values(vec![
                upstream::ImageResponseFormat::Url,
                upstream::ImageResponseFormat::B64Json,
            ])
        );
        assert_eq!(
            enum_values::<ImageOutputFormat>(),
            upstream_values(vec![
                upstream::ImageOutputFormat::Png,
                upstream::ImageOutputFormat::Jpeg,
                upstream::ImageOutputFormat::Webp,
            ])
        );
        assert_eq!(
            enum_values::<ImageModeration>(),
            upstream_values(vec![
                upstream::ImageModeration::Auto,
                upstream::ImageModeration::Low,
            ])
        );
        assert_eq!(
            enum_values::<ImageBackground>(),
            upstream_values(vec![
                upstream::ImageBackground::Auto,
                upstream::ImageBackground::Transparent,
                upstream::ImageBackground::Opaque,
            ])
        );
        assert_eq!(
            enum_values::<ImageStyle>(),
            upstream_values(vec![
                upstream::ImageStyle::Vivid,
                upstream::ImageStyle::Natural,
            ])
        );
        assert_eq!(
            enum_values::<ImageResponseBackground>(),
            upstream_values(vec![
                upstream::ImageResponseBackground::Transparent,
                upstream::ImageResponseBackground::Opaque,
            ])
        );
    }
}
