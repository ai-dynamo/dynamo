// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testing utilities for KVBM v2.
//!
//! This module provides reusable test infrastructure for:
//! - Token block generation
//! - BlockManager setup and configuration
//! - Nova instance creation and pairing
//! - Distributed leader test scenarios
//! - Physical layout and transfer testing

pub mod distributed;
pub mod managers;
pub mod nova;
pub mod physical;
pub mod token_blocks;
