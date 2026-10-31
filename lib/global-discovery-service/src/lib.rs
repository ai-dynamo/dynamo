// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool discovery for the Global Router, served over HTTP.
//!
//! The contract comes first: [`pool_discovery`], the traits that the
//! service reads and writes through.
//!
//! [`backends`] implements those traits: registrations kept in memory, and
//! the EndpointSlices that SIG Multicluster (MCS) imports.

pub mod backends;
pub mod pool_discovery;
