// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::LazyLock;

use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;

use crate::protocols::BlockExtraInfo;

/// Env var that extends the default admission allowlist. CSV of kind names
/// (currently only `sliding_window` is recognised). Stopgap for
/// effectively-full-window models like Phi-3-vision whose window equals
/// `max_model_len`; see [`KvCacheSpecKind::is_admitted`] for the rationale.
const ENV_ACCEPT_EVENT_KINDS: &str = "DYN_KV_ROUTER_ACCEPT_EVENT_KINDS";

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

    /// Whether this kind's blocks are admitted into the flat prefix-match
    /// index. Defaults to full/MLA/sink only; sliding-window admission is
    /// opt-in via [`ENV_ACCEPT_EVENT_KINDS`] and intended for
    /// effectively-full-window models (e.g. Phi-3-vision, where
    /// `sliding_window == max_model_len`).
    ///
    /// The flat indexer is not group-aware: admitting SW blocks for a
    /// hybrid model with a real window would alias independent cache
    /// groups and silently degrade routing. Native hybrid/group-aware
    /// indexing is tracked separately — see
    /// <https://github.com/llm-d/llm-d-kv-cache/issues/336> and
    /// <https://github.com/llm-d/llm-d-kv-cache/pull/533> for the
    /// reference shape (per-block group bitmask + windowed-prefix scorer).
    pub(crate) fn is_admitted(self) -> bool {
        matches!(
            self,
            Self::FullAttention | Self::MlaAttention | Self::SinkFullAttention
        ) || EXTRA_ADMITTED.contains(&self)
    }
}

/// Extra kinds admitted at startup via [`ENV_ACCEPT_EVENT_KINDS`]. Empty
/// unless an operator opts in. `sliding_window_mla` is intentionally not
/// recognised: no concrete model exercises it and the MLA+SWA matching
/// math is unvalidated.
static EXTRA_ADMITTED: LazyLock<Vec<KvCacheSpecKind>> = LazyLock::new(|| {
    let Ok(csv) = std::env::var(ENV_ACCEPT_EVENT_KINDS) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for tok in csv.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match tok.to_ascii_lowercase().as_str() {
            "sliding_window" => {
                if !out.contains(&KvCacheSpecKind::SlidingWindow) {
                    out.push(KvCacheSpecKind::SlidingWindow);
                }
            }
            other => tracing::warn!(
                env = ENV_ACCEPT_EVENT_KINDS,
                kind = other,
                "ignoring unrecognized KV event kind"
            ),
        }
    }
    if !out.is_empty() {
        tracing::warn!(
            env = ENV_ACCEPT_EVENT_KINDS,
            admitted = ?out,
            "extending KV event admission beyond the default safe set — \
             stopgap for effectively-full-window models only. \
             Hybrid-aware indexing is a follow-up; see \
             llm-d/llm-d-kv-cache#336 and llm-d/llm-d-kv-cache#533."
        );
    }
    out
});

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
