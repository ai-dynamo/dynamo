// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed coordination primitives for KVBM.
//!
//! This module provides the infrastructure for coordinating distributed
//! block manager operations across multiple workers led by a single leader.

// pub mod cohort;

pub mod leader;
pub mod worker;

pub mod object;
pub mod offload;
pub mod parallelism;
