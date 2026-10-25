// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery Service: which pools are live, and which Global Router
//! replicas a PoolRelay must connect to.
//!
//! - [`pool_discovery`]: the discovery traits and records (from upstream
//!   `dynamo_kv_router::pool_discovery`, ai-dynamo/dynamo#15933).
//! - [`backends`]: the memory and Kubernetes (MCS) stores behind those traits.
//! - [`service`]: the HTTP service over a store (ai-dynamo/dynamo#15949).

pub mod backends;
pub mod pool_discovery;
pub mod service;
