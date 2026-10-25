// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Types shared by the Global Router components.
//!
//! Taken from `dynamo_kv_router::global_view` upstream, which this branch does
//! not carry.

mod pool;

pub use pool::{PoolId, PoolIdDeriver, PoolKey, PoolKeyError, PoolLocation, V1PoolIdDeriver};
