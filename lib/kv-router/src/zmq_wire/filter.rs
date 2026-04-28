// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;
use serde::Deserializer;

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
pub(super) enum KvCacheSpecKind {
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
    fn from_wire(value: &str) -> Self {
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

    fn is_main_attention(self) -> bool {
        matches!(
            self,
            Self::FullAttention | Self::MlaAttention | Self::SinkFullAttention
        )
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
pub(super) struct KvCacheEventFilter {
    pub(super) group_idx: Option<u32>,
    pub(super) kv_cache_spec_kind: Option<KvCacheSpecKind>,
}

impl KvCacheEventFilter {
    pub(super) fn record_trailing(&mut self, trailing: KvCacheEventTrailingField) {
        match trailing {
            KvCacheEventTrailingField::GroupIdx(group_idx) if self.kv_cache_spec_kind.is_none() => {
                self.group_idx = Some(group_idx);
            }
            KvCacheEventTrailingField::GroupIdx(_) => {}
            KvCacheEventTrailingField::KvCacheSpecKind(kind) => {
                self.kv_cache_spec_kind = Some(kind);
            }
        }
    }

    pub(super) fn should_ignore(self) -> bool {
        if let Some(kind) = self.kv_cache_spec_kind {
            return !kind.is_main_attention();
        }

        matches!(self.group_idx, Some(group_idx) if group_idx != 0)
    }
}
