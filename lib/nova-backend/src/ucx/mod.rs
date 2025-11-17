// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! UCX Transport Module
//!
//! This module provides a high-performance UCX transport implementation with:
//! - Zero-copy RDMA-like messaging via Active Messages
//! - Lazy endpoint creation with deduplication
//! - 3 fixed lanes for Message/Response/Event routing
//! - LocalSet-based runtime for Rc<Worker> and Rc<Endpoint>
//! - Lock-free connection management via DashMap

#[cfg(feature = "ucx")]
mod runtime;
#[cfg(feature = "ucx")]
mod transport;

#[cfg(feature = "ucx")]
pub use transport::{UcxTransport, UcxTransportBuilder};
