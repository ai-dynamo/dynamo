// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral media preprocessing.
//!
//! Fetching, decoding, transport, and inference-engine adaptation are kept
//! outside this crate. Model modules translate Hugging Face configuration and
//! outputs into the common [`processed::ProcessedMedia`] contract.

pub mod models;
pub mod processed;
pub mod registry;
pub mod types;
pub mod vision;

pub use models::qwen3_vl::{Qwen3VlVideoConfig, Qwen3VlVideoPreprocessor, VideoTiming};
