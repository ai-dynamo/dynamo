// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Implementations of [`crate::pool_discovery`], one module per
//! backend: registrations kept in memory or in DynamoDB, and the
//! EndpointSlices that MCS imports. [`clock`] is the wall clock that the
//! DynamoDB backend decides leases on.

pub mod clock;
pub mod dynamodb;
pub mod kubernetes;
pub mod memory;
mod stream;
