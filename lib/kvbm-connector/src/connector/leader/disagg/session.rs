// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector-facing re-exports of engine-owned conditional-disaggregation
//! session primitives.

pub use kvbm_engine::disagg::{
    CONDITIONAL_DISAGG_STREAM_SCHEMA, DisaggSession, PrefillSession, PrefillSessionFactory,
    SessionBlocks, SessionEvent, SessionEventStream, VeloPrefillSessionFactory,
};
