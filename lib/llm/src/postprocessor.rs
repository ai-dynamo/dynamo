// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Postprocessor module provides encoding utilities for output media data.
//!
//! - `media`: Encoders for images and other media formats (PNG, base64).

pub mod media;

pub use media::encoders::{encode_base64, ImageEncoder};
