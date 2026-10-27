// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery Service: which Global Router pools exist, and which Global
//! Router replicas a PoolRelay connects to.
//!
//! The contract comes first: [`pool_discovery`], the store traits that the
//! service reads and writes through.
//!
//! [`backends`] holds the stores behind that contract: in memory, and from
//! the EndpointSlices that SIG Multicluster (MCS) imports.

pub mod backends;
pub mod pool_discovery;
