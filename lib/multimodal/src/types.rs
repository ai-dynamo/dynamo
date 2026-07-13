// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Borrowed contiguous RGB frame data.
#[derive(Debug, Clone, Copy)]
pub struct RgbFrameRef<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
}

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Image,
    Video,
    Audio,
}

/// Declares how a tensor's first dimension maps to media items.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FieldLayout {
    Batched,
    Flat { sizes_key: String },
    Shared,
}

impl FieldLayout {
    pub fn flat(sizes_key: impl Into<String>) -> Self {
        Self::Flat {
            sizes_key: sizes_key.into(),
        }
    }
}
