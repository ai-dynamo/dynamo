// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;

use crate::protocols::BlockExtraInfo;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum KvCacheEventTrailingField {
    GroupIdx(u32),
    KvCacheSpecKind(KvCacheSpecKind),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum BlockStoredTrailingField {
    Common(KvCacheEventTrailingField),
    BlockMmInfos(Vec<Option<BlockExtraInfo>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheSpecKind {
    FullAttention,
    MlaAttention,
    SlidingWindow,
    SlidingWindowMla,
    Mamba,
    ChunkedLocalAttention,
    SinkFullAttention,
    EncoderOnlyAttention,
    CrossAttention,
    Unknown,
}

impl KvCacheSpecKind {
    pub(crate) fn from_wire(value: &str) -> Self {
        match value {
            "full_attention" => Self::FullAttention,
            "mla_attention" => Self::MlaAttention,
            "sliding_window" => Self::SlidingWindow,
            "sliding_window_mla" => Self::SlidingWindowMla,
            "mamba" => Self::Mamba,
            "chunked_local_attention" => Self::ChunkedLocalAttention,
            "sink_full_attention" => Self::SinkFullAttention,
            "encoder_only_attention" => Self::EncoderOnlyAttention,
            "cross_attention" => Self::CrossAttention,
            unknown => {
                tracing::warn!(
                    kv_cache_spec_kind = unknown,
                    "Unknown KV cache spec kind; treating KV event as non-main"
                );
                Self::Unknown
            }
        }
    }

    pub(crate) fn as_wire(self) -> &'static str {
        match self {
            Self::FullAttention => "full_attention",
            Self::MlaAttention => "mla_attention",
            Self::SlidingWindow => "sliding_window",
            Self::SlidingWindowMla => "sliding_window_mla",
            Self::Mamba => "mamba",
            Self::ChunkedLocalAttention => "chunked_local_attention",
            Self::SinkFullAttention => "sink_full_attention",
            Self::EncoderOnlyAttention => "encoder_only_attention",
            Self::CrossAttention => "cross_attention",
            Self::Unknown => "unknown",
        }
    }

    pub(crate) fn is_main_attention(self) -> bool {
        matches!(
            self,
            Self::FullAttention | Self::MlaAttention | Self::SinkFullAttention
        )
    }
}

impl Serialize for KvCacheSpecKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire())
    }
}

impl<'de> Deserialize<'de> for KvCacheSpecKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from_wire(&value))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct KvCacheEventMetadata {
    pub(crate) group_idx: Option<u32>,
    pub(crate) kv_cache_spec_kind: Option<KvCacheSpecKind>,
    pub(crate) kv_cache_spec_sliding_window: Option<u32>,
}

impl KvCacheEventMetadata {
    pub(super) fn record_trailing(&mut self, trailing: KvCacheEventTrailingField) {
        match trailing {
            KvCacheEventTrailingField::GroupIdx(value) => {
                if self.group_idx.is_none() {
                    self.group_idx = Some(value);
                } else if self.kv_cache_spec_sliding_window.is_none() {
                    self.kv_cache_spec_sliding_window = Some(value);
                }
            }
            KvCacheEventTrailingField::KvCacheSpecKind(kind) => {
                self.kv_cache_spec_kind = Some(kind);
            }
        }
    }
}
