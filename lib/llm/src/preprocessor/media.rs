// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
mod decoders;
mod loader;
mod rdma;

use anyhow::{Context, Result};
use dynamo_protocols::types::ChatCompletionRequestMessageContentPartImage;

pub use common::EncodedMediaData;
pub use decoders::{Decoder, ImageDecoder, MediaDecoder};
pub use loader::{MediaFetcher, MediaLoader};

pub use rdma::{DecodedMediaData, RdmaMediaDataDescriptor, get_nixl_agent, get_nixl_metadata};

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
