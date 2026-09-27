// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PR #13187 stats service adapter for Global View.

pub mod proto {
    #![allow(clippy::all)]
    tonic::include_proto!("dynamo.kvdc.relay.v1");
}

mod projection;

pub use projection::{StatsCatalog, combined_capacity, project_load, project_usage};
