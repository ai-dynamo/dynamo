// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact-producer Relay CKF subscription for the first aggregated Global View POC.
//! Catalog ownership and reconnect policy remain with the caller.

mod scorer;
mod stream;

pub use scorer::RelayCkfOverlapStore;
pub use stream::run_exact_aggregated_producer;
