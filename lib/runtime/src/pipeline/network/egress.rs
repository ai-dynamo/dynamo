// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod addressed_router;
pub mod http_router;
pub mod push_router;

pub use crate::pipeline::network::adaptive_client::AdaptiveRequestPlaneClient;

use super::*;
