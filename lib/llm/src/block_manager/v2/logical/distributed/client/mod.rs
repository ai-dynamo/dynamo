// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry client implementations.
//!
//! Clients connect to a registry hub to query and register entries.

mod local;
mod zmq;

pub use local::LocalRegistry;
pub use zmq::ZmqRegistryClient;

