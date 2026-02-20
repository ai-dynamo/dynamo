// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA downloading and caching infrastructure
//!
//! This module provides a minimal, extensible interface for downloading LoRA adapters
//! from various sources (local filesystem, S3, etc.) with automatic caching.
//! It also provides routing and lora allocation algorithms for distributing LoRA adapters
//! across workers in a cluster.

mod cache;
mod downloader;
pub mod load_estimator;
pub mod routing;
mod source;

pub use cache::LoRACache;
pub use downloader::LoRADownloader;
pub use load_estimator::{LoadEstimator, LoadEstimatorConfig, LoadSample};
pub use routing::{
    AllocationAlgorithmType, LoraAllocator, LoraReplicaConfig, LoraRoutingTable, RendezvousHasher,
    create_lora_allocator,
};
pub use source::{LoRASource, LocalLoRASource, S3LoRASource};
