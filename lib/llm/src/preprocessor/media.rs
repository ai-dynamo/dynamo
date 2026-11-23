// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
mod decoders;
mod jpeg_turbo;
mod loader;
mod rdma;

/// Client-safe image download failures, also used by the Python error bridge.
pub const IMAGE_FETCH_FORBIDDEN_MESSAGE: &str = "Image download returned HTTP 403. Refresh the URL or access permissions, or resend the image data before retrying.";
/// Client-safe recovery advice for a missing image.
pub const IMAGE_FETCH_NOT_FOUND_MESSAGE: &str = "Image download returned HTTP 404. Verify the image URL or resend the image data before retrying.";

use anyhow::{Context, Result};
use dynamo_protocols::types::ChatCompletionRequestMessageContentPartImage;

pub use common::EncodedMediaData;
pub use decoders::{Decoder, ImageDecoder, MediaDecoder};
pub use loader::{MediaFetcher, MediaLoader, max_data_url_bytes};

pub use rdma::{DecodedMediaData, RdmaMediaDataDescriptor, get_nixl_agent, get_nixl_metadata};

#[doc(hidden)]
pub fn libjpeg_turbo_available() -> bool {
    jpeg_turbo::available()
}

pub(super) fn require_image_url(
    part: &ChatCompletionRequestMessageContentPartImage,
) -> Result<&url::Url> {
    Ok(&part
        .image_url
        .as_ref()
        .context(
            "Cannot decode an image content part without a URL; UUID-only parts must be resolved by the backend cache",
        )?
        .url)
}
