// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the offload engine.
//!
//! This module tests the offload engine's ability to:
//! - Transfer blocks between tiers (G2→G3) with policy-based filtering
//! - Handle cancellation correctly with resource cleanup
//!
//! # Test Coverage
//!
//! - **g2_g3_flow**: Tests G2→G3 offloading with presence and LFU frequency filters
//! - **cancellation**: Tests cancellation, sweeper cleanup, and resource release

mod cancellation;
mod g2_g3_flow;
