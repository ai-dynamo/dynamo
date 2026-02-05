// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP middleware for Dynamo
//!
//! This module contains middleware components for request processing,
//! including session extraction from trusted upstream headers.

pub mod session;

pub use session::{RequestSession, extract_session_middleware};
