// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Types shared by the Global Router components.

mod pool;
pub use pool::{Pool, PoolId, PoolIdDeriver, PoolKey, PoolKeyError, PoolLocation, V1PoolIdDeriver};
