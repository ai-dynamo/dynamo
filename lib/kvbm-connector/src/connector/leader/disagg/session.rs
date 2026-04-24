// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector-facing re-exports of engine-owned conditional-disaggregation
//! session primitives.

pub use kvbm_engine::disagg::{
    DisaggSession, PrefillSession, PrefillSessionFactory, SessionBlocks, SessionEvent,
    SessionEventStream, VELO_STREAM_ENDPOINT_KIND, VeloPrefillSessionFactory, hash_to_wire,
    hashes_to_wire,
};
