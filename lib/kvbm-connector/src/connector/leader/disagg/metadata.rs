// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector-facing re-exports of engine-owned peer metadata cache helpers.

pub use kvbm_engine::disagg::{
    CoalescingPeerMetadataCache, EnginePeerMetadataCache, NoopPeerMetadataCache, PeerMetadataCache,
};
