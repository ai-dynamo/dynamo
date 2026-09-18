// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
mod decoded;
mod decoders;
mod jpeg_turbo;
mod loader;
#[cfg(feature = "media-nixl")]
mod rdma;
mod rdma_descriptor;

pub use common::EncodedMediaData;
pub use decoded::DecodedMediaData;
pub use decoders::{Decoder, ImageDecoder, MediaDecoder};
pub use loader::MediaFetcher;
#[cfg(feature = "media-nixl")]
pub use rdma::{MediaLoader, get_nixl_agent, get_nixl_metadata};
pub use rdma_descriptor::RdmaMediaDataDescriptor;

/// Marker used internally when frontend media decoding is not compiled in.
#[cfg(not(feature = "media-nixl"))]
pub(crate) struct MediaLoader;

#[doc(hidden)]
pub fn libjpeg_turbo_available() -> bool {
    jpeg_turbo::available()
}
