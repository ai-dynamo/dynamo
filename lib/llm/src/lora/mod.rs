// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA downloading and caching infrastructure
//!
//! This module provides a minimal, extensible interface for downloading LoRA adapters
//! from various sources (local filesystem, S3, etc.) with automatic caching.

mod cache;
mod downloader;
mod source;

pub use cache::LoRACache;
pub use downloader::LoRADownloader;
pub use source::{LoRASource, LocalLoRASource, S3LoRASource};
